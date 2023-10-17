import numpy as np
import pandas as pd
from spinesTS.preprocessing import split_series
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert

from PipelineTS.base import NNModelMixin, IntervalEstimationMixin


class SpinesNNModelMixin(NNModelMixin, IntervalEstimationMixin):

    def __init__(self, time_col, target_col, device=None):

        super().__init__(time_col, target_col, device=device)
        self.last_x = None

    def _define_model(self):
        raise NotImplementedError

    @ParameterValuesAssert({
        'mode': ('train', 'predict')
    })
    def _data_preprocess(self, data, update_last_data=False, mode='train'):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        if update_last_data:
            self.last_dt = data[self.all_configs['time_col']].max()

        if mode == 'train':
            if update_last_data:
                self.last_x = data[self.all_configs['target_col']].iloc[-(2 * self.all_configs['lags']):]
            # x_train, y_train
            x_train, y_train = split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                                            window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            if x_train.ndim == 1:
                x_train = x_train.reshape(1, -1)

            if y_train.ndim == 1:
                y_train = y_train.reshape(1, -1)

            return x_train, y_train
        else:
            x, y = split_series(pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            if x.ndim == 1:
                x = x.reshape(1, -1)

            if y.ndim == 1:
                y = y.reshape(1, -1)

            return x, y

    @ParameterTypeAssert({
        'valid_data': (None, pd.DataFrame)
    })
    def fit(self, data, valid_data=None, cv=5, fit_kwargs=None):
        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        if fit_kwargs is None:
            fit_kwargs = {}

        for fit_param in [
            'verbose', 'epochs', 'batch_size', 'patience',
            'min_delta', 'lr_scheduler', 'lr_scheduler_patience',
            'lr_factor', 'restore_best_weights'
        ]:
            if fit_param not in fit_kwargs:
                fit_kwargs.update({fit_param: self.all_configs[fit_param]})

        x, y = self._data_preprocess(data, update_last_data=True, mode='train')

        self.x = data[self.all_configs['target_col']].iloc[-self.all_configs['lags']:]

        if valid_data is None:
            eval_set = [(x, y)]
        else:
            valid_x, valid_y = self._data_preprocess(valid_data, update_last_data=False, mode='predict')

            eval_set = [(valid_x, valid_y)]

        self.model.fit(x, y, eval_set=eval_set, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_nn(data, fit_kwargs=fit_kwargs, cv=cv)

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
