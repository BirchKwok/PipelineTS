from spinesTS.nn import StackingRNN
from spinesUtils import generate_function_kwargs

from PipelineTS.base.sps_nn_model_base import SpinesNNModelMixin


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
            accelerator='auto',
            verbose=False,
            epochs=1000,
            batch_size='auto',
            patience=100,
            min_delta=0,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
    ):
        super().__init__(time_col=time_col, target_col=target_col, device=accelerator)

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
        return StackingRNN(**self.all_configs['model_configs'])
