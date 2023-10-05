import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn.model_selection import TimeSeriesSplit
from spinesTS.ml_model import (
    GBRTPreprocessing,
    MultiOutputRegressor
)

from spinesTS.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from spinesUtils import generate_function_kwargs
from lightgbm import LGBMRegressor

from PipelineTS.base import GBDTModelMixin, IntervalEstimationMixin


class WideGBRTModel(GBDTModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            n_estimators=100,
            quantile=0.9,
            random_state=None,
            differential_n=0,
            moving_avg_n=0,
            extend_daily_target_features=True,
            use_standard_scaler=True,
            **lightgbm_model_configs
    ):
        super().__init__()

        self.all_configs['model_configs'] = generate_function_kwargs(
            LGBMRegressor,
            n_estimators=n_estimators,
            random_state=random_state,
            **lightgbm_model_configs
        )

        self.last_dt = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'lower_limit': 0,
                'higher_limit': 0,
                'differential_n': differential_n,
                'moving_avg_n': moving_avg_n,
                'extend_daily_target_features': extend_daily_target_features,
                'use_standard_scaler': use_standard_scaler
            }
        )

        self.processor = GBRTPreprocessing(
            in_features=self.all_configs['lags'],
            out_features=self.all_configs['lags'],
            target_col=self.all_configs['target_col'],
            date_col=self.all_configs['time_col'],
            differential_n=self.all_configs['differential_n'],
            moving_avg_n=self.all_configs['moving_avg_n'],
            extend_daily_target_features=self.all_configs['extend_daily_target_features'],
            train_size=None
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        if self.all_configs['use_standard_scaler']:
            model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('estimator', MultiOutputRegressor(
                            LGBMRegressor(**self.all_configs['model_configs'])
                        ))
                    ])
        else:
            model = Pipeline([
                        ('scaler', MinMaxScaler()),
                        ('estimator', MultiOutputRegressor(
                            LGBMRegressor(**self.all_configs['model_configs'])
                        ))
                    ])

        return model

    def _data_preprocess(self, data, mode='train'):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        self.last_dt = data[self.all_configs['time_col']].max()
        if mode == 'train':
            self.processor.fit(data)

        return self.processor.transform(data, mode=mode)  # X, y

    def calculate_confidence_interval_gbrt(self, data, cv=5, fit_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        tscv = TimeSeriesSplit(n_splits=cv)

        residuals = []

        data_x, data_y = self._data_preprocess(data, 'train')
        data_x = pd.DataFrame(data_x)
        data_y = pd.DataFrame(data_y)

        for (train_index, test_index) in tscv.split(data_x, data_y):
            train_x, train_y = data_x.iloc[train_index, :], data_y.iloc[train_index, :]

            test_x, test_y = data_x.iloc[test_index, :], data_y.iloc[test_index, :]

            model = self._define_model()

            model.fit(train_x, train_y, eval_set=None, **fit_kwargs)
            res = model.predict(test_x)

            residuals.append(np.mean(np.abs(test_y - res)))

        sample_mean = np.mean(residuals)
        sample_std = np.std(residuals)

        n = len(residuals)
        # 使用正态分布的累积分布函数的逆函数（ppf）计算出临界值（crit_value），用于计算置信区间
        z_score = sp.norm.ppf(self.all_configs['quantile'])
        lower_limit = sample_mean - (z_score * (sample_std / np.sqrt(n)))
        higher_limit = sample_mean + (z_score * (sample_std / np.sqrt(n)))

        return abs(lower_limit), abs(higher_limit)

    def fit(self, data, cv=5, fit_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, 'train')
        x = pd.DataFrame(x)

        self.x = pd.DataFrame(
            self._data_preprocess(data, 'predict')
        ).iloc[-1:, :]

        self.model.fit(x, y, eval_set=None, **fit_kwargs)

        self.all_configs['lower_limit'], self.all_configs['higher_limit'] = \
            self.calculate_confidence_interval_gbrt(data, cv=cv, fit_kwargs=fit_kwargs)

        return self

    def predict(self, n):
        x = self.x.values
        res = self.model.extend_predict(x, n).squeeze()  # numpy.ndarray
        assert len(res) == n
        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            self.last_dt + pd.to_timedelta(range(res.index.shape[0]+1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return res
