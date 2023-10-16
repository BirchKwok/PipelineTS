from spinesTS.ml_model import MultiOutputRegressor as MOR, MultiStepRegressor as MSR

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lightgbm import LGBMRegressor

from spinesTS.pipeline import Pipeline
from spinesTS.preprocessing import split_series
from spinesUtils import generate_function_kwargs

from PipelineTS.base import GBDTModelMixin, IntervalEstimationMixin


class MultiOutputRegressorModel(GBDTModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            estimator=LGBMRegressor,
            use_standard_scaler=None,
            **model_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self._base_estimator = estimator
        self.all_configs['model_configs'] = generate_function_kwargs(
            self._base_estimator,
            **model_configs
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
                'use_standard_scaler': use_standard_scaler
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        if self.all_configs['use_standard_scaler'] is not None:
            if self.all_configs['use_standard_scaler']:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', MOR(
                        self._base_estimator(**self.all_configs['model_configs'])
                    ))
                ])
            else:
                model = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('estimator', MOR(
                        self._base_estimator(**self.all_configs['model_configs'])
                    ))
                ])
        else:
            model = Pipeline([
                ('estimator', MOR(
                    self._base_estimator(**self.all_configs['model_configs'])
                ))
            ])

        return model

    def _data_preprocess(self, data, update_last_dt=False):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        if update_last_dt:
            self.last_dt = data[self.all_configs['time_col']].max()

        # x_train, y_train
        return split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                            window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

    def fit(self, data, cv=5, fit_kwargs=None):
        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        self.last_lags_dataframe = data.iloc[-(2 * self.all_configs['lags'] + 1):, :]

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, update_last_dt=True)
        x = pd.DataFrame(x)

        self.x = x.iloc[-1:, :]

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_mor(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n, predict_kwargs):
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

        current_res = self.model.predict(x, **predict_kwargs)

        if n is None:
            return current_res.squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res.squeeze().tolist()[:n]
        else:
            res = current_res.squeeze().tolist()
            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
                current_res = self.model.predict(x, **predict_kwargs)

                res.append(current_res.squeeze().tolist()[-1])

            return res

    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        x = self.x.values.reshape(1, -1)
        res = self._extend_predict(x, n, predict_kwargs=predict_kwargs)  # list

        assert len(res) == n

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            self.last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


class MultiStepRegressorModel(GBDTModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            estimator=LGBMRegressor,
            use_standard_scaler=None,
            **model_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self._base_estimator = estimator
        self.all_configs['model_configs'] = generate_function_kwargs(
            self._base_estimator,
            **model_configs
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
                'use_standard_scaler': use_standard_scaler
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        if self.all_configs['use_standard_scaler'] is not None:
            if self.all_configs['use_standard_scaler']:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', MSR(
                        self._base_estimator(**self.all_configs['model_configs'])
                    ))
                ])
            else:
                model = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('estimator', MSR(
                        self._base_estimator(**self.all_configs['model_configs'])
                    ))
                ])
        else:
            model = Pipeline([
                ('estimator', MSR(
                    self._base_estimator(**self.all_configs['model_configs'])
                ))
            ])

        return model

    def _data_preprocess(self, data, update_last_dt=False):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        if update_last_dt:
            self.last_dt = data[self.all_configs['time_col']].max()

        # x_train, y_train
        return split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                            window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

    def fit(self, data, cv=5, fit_kwargs=None):
        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        self.last_lags_dataframe = data.iloc[-(2 * self.all_configs['lags'] + 1):, :]

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, update_last_dt=True)
        x = pd.DataFrame(x)

        self.x = x.iloc[-1:, :]

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_mor(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n, predict_kwargs):
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

        current_res = self.model.predict(x, **predict_kwargs).reshape(1, -1)

        if n is None:
            return current_res.squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res.squeeze().tolist()[:n]
        else:
            res = current_res.squeeze().tolist()
            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
                current_res = self.model.predict(x, **predict_kwargs).reshape(1, -1)

                res.append(current_res.squeeze().tolist()[-1])

            return res

    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        x = self.x.values.reshape(1, -1)
        res = self._extend_predict(x, n, predict_kwargs=predict_kwargs)  # list

        assert len(res) == n

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            self.last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)