import numpy as np
import pandas as pd
from spinesTS.preprocessing import split_series, lag_splits

from spinesUtils import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.asserts import raise_if_not
from spinesUtils.preprocessing import gc_collector, reshape_if

from PipelineTS.base import NNModelMixin, IntervalEstimationMixin


class SpinesNNModelMixin(NNModelMixin, IntervalEstimationMixin):

    def __init__(self, time_col, target_col, device=None):

        super().__init__(time_col, target_col, device=device)
        self.last_x = None
        self.scaler = None

    def _define_model(self):
        raise NotImplementedError

    @ParameterValuesAssert({
        'mode': ('train', 'validation', 'predict')
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

            x_train = reshape_if(x_train, x_train.ndim == 1, (1, -1))
            y_train = reshape_if(y_train, y_train.ndim == 1, (1, -1))

            return x_train, y_train

        elif mode == 'validation':
            x, y = split_series(pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            x = reshape_if(x, x.ndim == 1, (1, -1))
            y = reshape_if(y, y.ndim == 1, (1, -1))

            return x, y

        else:
            x = lag_splits(data[self.all_configs['target_col']], window_size=self.all_configs['lags'])
            x = reshape_if(x, x.ndim == 1, (1, -1))

            return x

    @ParameterTypeAssert({
        'valid_data': (None, pd.DataFrame)
    })
    @gc_collector(3)
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
            valid_x, valid_y = self._data_preprocess(valid_data, update_last_data=False, mode='validation')

            eval_set = [(valid_x, valid_y)]

        self.model.fit(x, y, eval_set=eval_set, **fit_kwargs)

        del x, y

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

        current_res = reshape_if(current_res, current_res.ndim == 1, (1, -1))

        if n is None:
            return current_res.squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res[-1][:n].tolist()
        else:
            res = current_res.squeeze().tolist()
            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
                current_res = self.model.predict(x, **predict_kwargs)
                current_res = reshape_if(current_res, current_res.ndim == 1, (1, -1))

                res.append(current_res.squeeze().tolist()[-1])

            return res

    def predict(self, n, series=None, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        if series is not None:
            raise_if_not(
                ValueError, len(series) >= self.all_configs['lags'],
                'The length of the series must greater than or equal to the lags. '
            )

            x = self._data_preprocess(series.iloc[-self.all_configs['lags']:, :],
                                      update_last_data=False, mode='predict')
            last_dt = series[self.all_configs['time_col']].max()
        else:
            x = reshape_if(self.x.values, self.x.values.ndim == 1, (1, -1))
            last_dt = self.last_dt

        res = self._extend_predict(x, n, predict_kwargs=predict_kwargs)  # list

        assert len(res) == n

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
