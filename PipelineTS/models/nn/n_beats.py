from PipelineTS.nn_model.backbones import NBeats

from PipelineTS.nn_model._wrapper import NNBackboneForecastingMixin


class NBeatsModel(NNBackboneForecastingMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            generic_architecture=True,
            num_stacks=2,
            num_blocks=3,
            num_layers=4,
            layer_widths=256,
            expansion_coeff_dim=32,
            trend_degree=3,
            dropout=0.1,
            use_revin=True,
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
        NBeatsModel: A wrapper for the N-BEATS model from PipelineTS backbones.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features.
        generic_architecture : bool, optional, default: True
            Whether to use generic architecture. False uses interpretable (trend+seasonality).
        num_stacks : int, optional, default: 2
            Number of stacks in the architecture.
        num_blocks : int, optional, default: 3
            Number of blocks per stack.
        num_layers : int, optional, default: 4
            Number of FC layers per block.
        layer_widths : int, optional, default: 256
            Width of FC layers.
        expansion_coeff_dim : int, optional, default: 32
            Expansion coefficient dimension (generic architecture only).
        trend_degree : int, optional, default: 3
            Polynomial degree for trend basis (interpretable architecture only).
        dropout : float, optional, default: 0.1
            Dropout rate.
        use_revin : bool, optional, default: True
            Whether to use RevIN normalization.
        quantile : float, optional, default: 0.9
            Quantile for interval prediction. None for point prediction.
        random_state : int or None, optional, default: None
            Random seed.
        learning_rate : float, optional, default: 0.001
            Learning rate.
        accelerator : str, optional, default: 'auto'
            Accelerator for training.
        verbose : bool, optional, default: False
            Whether to display verbose output.
        epochs : int, optional, default: 1500
            Number of training epochs.
        batch_size : int or 'auto', optional, default: 'auto'
            Batch size.
        patience : int, optional, default: 80
            Early stopping patience.
        min_delta : int, optional, default: 0
            Minimum improvement delta.
        lr_scheduler : str, optional, default: 'CosineAnnealingLR'
            Learning rate scheduler.
        lr_scheduler_patience : int, optional, default: 10
            LR scheduler patience.
        lr_factor : float, optional, default: 0.7
            LR reduction factor.
        restore_best_weights : bool, optional, default: True
            Whether to restore best weights.
        loss_type : str, optional, default: 'min'
            Loss type for early stopping.
        weight_decay : float, optional, default: 1e-4
            Weight decay for AdamW.

        Attributes
        ----------
        model : PipelineTS.nn_model.backbones.NBeats
            The N-BEATS model from PipelineTS backbones.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self._init_backbone_model(
            backbone_cls=NBeats,
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
            generic_architecture=generic_architecture,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            expansion_coeff_dim=expansion_coeff_dim,
            trend_degree=trend_degree,
            dropout=dropout,
            use_revin=use_revin,
            loss_fn='huber',
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            weight_decay=weight_decay,
            use_gtb=use_gtb,
            gtb_d_model=gtb_d_model,
            routing_mode=routing_mode
            ),
        )
