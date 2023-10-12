from copy import deepcopy

import pandas as pd
import numpy as np

from spinesTS.nn import StackingRNN
from spinesTS.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from spinesUtils import generate_function_kwargs

from PipelineTS.nn_model.sps_nn_model_base import SpinesNNModelMixin


class StackingRNNModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=30,
            quantile=0.9,
            random_state=None,
            dropout=0.1,
            learning_rate=0.001,
            device='cpu',
            use_standard_scaler=False,
            verbose=False,
            epochs=1000,
            batch_size='auto',
            patience=100,
            min_delta=0,
            lr_scheduler='ReduceLROnPlateau',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
    ):
        super().__init__(time_col=time_col, target_col=target_col, device=device)

        self.all_configs['model_configs'] = generate_function_kwargs(
            StackingRNN,
            in_features=lags,
            out_features=lags,
            loss_fn='mae',
            bias=True,
            dropout=dropout,
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.device,
        )

        self.last_dt = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0,
                'use_standard_scaler': use_standard_scaler,
                'verbose': verbose,
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'min_delta': min_delta,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_patience': lr_scheduler_patience,
                'lr_factor': lr_factor,
                'restore_best_weights': restore_best_weights
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        if self.all_configs['use_standard_scaler'] is not None:
            if self.all_configs['use_standard_scaler']:
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('estimator', StackingRNN(**self.all_configs['model_configs']))
                ])
            else:
                model = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('estimator', StackingRNN(**self.all_configs['model_configs']))
                ])
        else:
            model = StackingRNN(**self.all_configs['model_configs'])

        return model

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
            return current_res[0].squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res[0].squeeze()[:n].tolist()
        else:
            res = deepcopy(current_res)

            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=-1)
                current_res = self.model.predict(x, **predict_kwargs)  # 2D

                res = np.concatenate((res, current_res[:, -1:]), axis=-1)

            return res[0].squeeze().tolist()

    def predict(self, n, predict_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}

        x = self.x.values.reshape(1, -1)
        x = np.concatenate((x, x), axis=0)

        res = self._extend_predict(x, n, predict_kwargs=predict_kwargs)  # list

        assert len(res) == n

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            self.last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)

