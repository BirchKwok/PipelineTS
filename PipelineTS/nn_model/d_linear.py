from PipelineTS.spinesTS.nn import DLinear
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.spines_base import SpinesNNModelMixin


class DLinearModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            kernel_size=None,
            use_revin=True,
            dropout=0.1,
            quantile=0.9,
            random_state=None,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=1000,
            batch_size='auto',
            patience=20,
            min_delta=0,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            loss_type='min',
            weight_decay=1e-4
    ):
        """
        DLinearModel: A wrapper for the DLinear model from spinesTS with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        kernel_size : int or None, optional, default: None
            The kernel size for moving average decomposition. None for auto.
        use_revin : bool, optional, default: True
            Whether to use Reversible Instance Normalization.
        dropout : float, optional, default: 0.1
            The dropout rate.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        learning_rate : float, optional, default: 0.001
            The learning rate for training.
        accelerator : str, optional, default: 'auto'
            The accelerator to use during training.
        verbose : bool, optional, default: False
            Whether to display verbose output during training.
        epochs : int, optional, default: 1000
            The number of epochs for training.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training.
        patience : int, optional, default: 20
            The patience for early stopping.
        min_delta : int, optional, default: 0
            Minimum change to qualify as an improvement.
        lr_scheduler : str, optional, default: 'CosineAnnealingLR'
            The learning rate scheduler.
        lr_scheduler_patience : int, optional, default: 10
            Patience for the learning rate scheduler.
        lr_factor : float, optional, default: 0.7
            Factor for learning rate reduction.
        restore_best_weights : bool, optional, default: True
            Whether to restore best weights after training.
        loss_type : str, optional, default: 'min'
            The loss type for early stopping.
        weight_decay : float, optional, default: 1e-4
            Weight decay for AdamW optimizer.

        Attributes
        ----------
        model : spinesTS.nn.DLinear
            The DLinear model from spinesTS.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            DLinear,
            in_features=lags,
            out_features=lags,
            kernel_size=kernel_size,
            use_revin=use_revin,
            dropout=dropout,
            loss_fn='huber',
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            weight_decay=weight_decay
        )

        self.last_dt = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': (0, 0),
                'verbose': verbose,
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
                'min_delta': min_delta,
                'lr_scheduler': lr_scheduler,
                'lr_scheduler_patience': lr_scheduler_patience,
                'lr_factor': lr_factor,
                'restore_best_weights': restore_best_weights,
                'loss_type': loss_type
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the DLinear model from spinesTS.

        Returns
        -------
        spinesTS.nn.DLinear
            The DLinear model.
        """
        return DLinear(**self.all_configs['model_configs'])
