from copy import deepcopy

import numpy as np
import pandas as pd
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
            linear_tree=False,
            **lightgbm_model_configs
    ):
        super().__init__()

        self.all_configs['model_configs'] = generate_function_kwargs(
            LGBMRegressor,
            n_estimators=n_estimators,
            random_state=random_state,
            linear_tree=linear_tree,
            **lightgbm_model_configs
        )

        self.last_dt = None
        self.last_lags_dataframe = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0,
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
            res = model.predict(test_x).flatten()
            test_y = test_y.values.flatten()
            error_rate = np.abs((test_y - res) / test_y)
            error_rate = np.where((error_rate == np.inf) | (error_rate == np.nan), 0., error_rate)

            residuals.extend(error_rate.tolist())

        quantile = np.percentile(residuals, self.all_configs['quantile'])

        return quantile

    def fit(self, data, cv=5, fit_kwargs=None):
        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        self.last_lags_dataframe = data.iloc[-(2 * self.all_configs['lags'] + 1):, :]

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, 'train')
        x = pd.DataFrame(x)

        self.x = pd.DataFrame(
            self._data_preprocess(data, 'predict')
        ).iloc[-1:, :]

        self.model.fit(x, y, eval_set=None, **fit_kwargs)

        self.all_configs['quantile_error'] = \
            self.calculate_confidence_interval_gbrt(data, cv=cv, fit_kwargs=fit_kwargs)

        return self

    def _extend_predict(self, x, n):
        """Extrapolation prediction.

        Parameters
        ----------
        x: to_predict data, must be 2 dims data
        n: predict steps, must be int

        Returns
        -------
        np.ndarray, which has 2 dims

        """

        assert isinstance(n, int)
        assert x.ndim == 2

        current_res = self.model.predict(x)

        if n is None:
            return current_res.squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res[-1][:n].tolist()
        else:
            res = current_res.squeeze().tolist()
            last_data = self.last_lags_dataframe
            last_data[self.all_configs['time_col']] = pd.to_datetime(last_data[self.all_configs['time_col']])

            last_dt = deepcopy(self.last_dt)
            for i in range(n - self.all_configs['lags']):
                tmp_data = pd.DataFrame(columns=[self.all_configs['time_col'], self.all_configs['target_col']])
                tmp_data[self.all_configs['time_col']] = (last_dt +
                                                          pd.to_timedelta(range(self.all_configs['lags'] + 1),
                                                                          unit='D'))[1:]

                tmp_data[self.all_configs['target_col']] = res[-self.all_configs['lags']:]
                last_data = pd.concat((last_data.iloc[1:, :], tmp_data.iloc[:1, :]), axis=0)

                last_dt = last_data[self.all_configs['time_col']].max()

                to_predict_x = pd.DataFrame(
                    self._data_preprocess(last_data, 'predict')
                ).iloc[-1:, :]

                current_res = self.model.predict(to_predict_x).squeeze()
                res.append(current_res[0])

            return res

    def predict(self, n):
        x = self.x.values
        res = self._extend_predict(x, n)  # list
        assert len(res) == n
        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            self.last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return res
