from PipelineTS.models.nn.backbones import NLinear

from PipelineTS.models.nn._wrapper import NNBackboneForecastingMixin


class NLinearModel(NNBackboneForecastingMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            use_revin=True,
            dropout=0.1,
            quantile=0.9,
            random_state=None,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=1500,
            batch_size='auto',
            patience=80,
            min_delta=0,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            loss_type='min',
            weight_decay=1e-4,
            use_gtb=False,
            gtb_d_model=64,
            routing_mode='static',
            use_ema=False,
            ema_decay=0.999,
            use_swa=False,
            swa_start_frac=0.75,
            warmup_epochs=0,
            use_residual_gate=False,
    ):
        """
        NLinearModel: A wrapper for the NLinear model from PipelineTS backbones with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
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
        epochs : int, optional, default: 1500
            The number of epochs for training.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training.
        patience : int, optional, default: 80
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
        use_gtb : bool, optional, default: False
            Whether to use Gated Temporal Blocks (GTB).
        gtb_d_model : int, optional, default: 64
            The dimension of the model in GTB.
        routing_mode : str, optional, default: 'static'
            The routing mode for GTB.
        use_ema : bool, optional, default: False
            Whether to use Exponential Moving Average (EMA) of model weights.
        ema_decay : float, optional, default: 0.999
            The decay rate for EMA.
        use_swa : bool, optional, default: False
            Whether to use Stochastic Weight Averaging (SWA).
        swa_start_frac : float, optional, default: 0.75
            The fraction of total epochs to start SWA.
        warmup_epochs : int, optional, default: 0
            The number of warmup epochs for learning rate scheduling.
        use_residual_gate : bool, optional, default: False
            Whether to use a residual gate in the model.

        Attributes
        ----------
        model : PipelineTS.models.nn.backbones.NLinear
            The NLinear model from PipelineTS backbones.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self._init_backbone_model(
            backbone_cls=NLinear,
            lags=lags,
            quantile=quantile,
            time_col=time_col,
            target_col=target_col,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            restore_best_weights=restore_best_weights,
            loss_type=loss_type,
            use_ema=use_ema,
            ema_decay=ema_decay,
            use_swa=use_swa,
            swa_start_frac=swa_start_frac,
            warmup_epochs=warmup_epochs,
            use_residual_gate=use_residual_gate,
            model_kwargs=dict(
            use_revin=use_revin,
            dropout=dropout,
            loss_fn='huber',
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            weight_decay=weight_decay,
            use_gtb=use_gtb,
            gtb_d_model=gtb_d_model,
            routing_mode=routing_mode,
            ),
        )
