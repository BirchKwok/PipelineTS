from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from spinesTS.metrics import wmape
from spinesTS.preprocessing import split_series
from spinesUtils import ParameterTypeAssert

from PipelineTS.base import NNModelMixin, IntervalEstimationMixin


class SpinesNNModelMixin(NNModelMixin, IntervalEstimationMixin):

    def _define_model(self):
        raise NotImplementedError

    def _data_preprocess(self, data, update_last_dt=False):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        if update_last_dt:
            self.last_dt = data[self.all_configs['time_col']].max()

        # x_train, y_train
        return split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                            window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

    def calculate_confidence_interval_nn(self, data, cv=5, fit_kwargs=None):
        if fit_kwargs is None:
            kwargs = {}
        else:
            kwargs = deepcopy(fit_kwargs)

        kwargs.update({'verbose': False})

        tscv = TimeSeriesSplit(n_splits=cv, test_size=self.all_configs['lags'])

        residuals = []

        data_x, data_y = self._data_preprocess(data)
        data_x = pd.DataFrame(data_x)
        data_y = pd.DataFrame(data_y)

        for (train_index, test_index) in tscv.split(data_x, data_y):
            train_x, train_y = data_x.iloc[train_index, :].values, data_y.iloc[train_index, :].values

            test_x, test_y = data_x.iloc[test_index, :].values, data_y.iloc[test_index, :].values

            model = self._define_model()

            model.fit(train_x, train_y, eval_set=[(train_x, train_y)], **kwargs)
            res = model.predict(test_x).flatten()

            test_y = test_y.flatten()

            error_rate = wmape(test_y, res)

            residuals.append(error_rate)

        quantile = np.percentile(residuals, self.all_configs['quantile'])

        return quantile

    @ParameterTypeAssert({
        'valid_data': (None, pd.DataFrame)
    })
    def fit(self, data, valid_data=None, cv=5, fit_kwargs=None):
        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        assert valid_data is None or (valid_data.shape[0] >= (2 * self.all_configs['lags'] + 1))

        if fit_kwargs is None:
            fit_kwargs = {}

        for fit_param in [
            'verbose', 'epochs', 'batch_size', 'patience',
            'min_delta', 'lr_scheduler', 'lr_scheduler_patience',
            'lr_factor', 'restore_best_weights'
        ]:
            if fit_param not in fit_kwargs:
                fit_kwargs.update({fit_param: self.all_configs[fit_param]})

        x, y = self._data_preprocess(data, update_last_dt=True)

        self.x = data[self.all_configs['target_col']].iloc[-self.all_configs['lags']:]

        if valid_data is None:
            eval_set = [(x, y)]
        else:
            valid_x, valid_y = self._data_preprocess(valid_data)
            if valid_x.ndim == 1:
                valid_x = valid_x.reshape(1, -1)

            eval_set = [(valid_x, valid_y)]

        self.model.fit(x, y, eval_set=eval_set, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_nn(data, cv=cv, fit_kwargs=fit_kwargs)

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
            return current_res[-1][:n].tolist()
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
