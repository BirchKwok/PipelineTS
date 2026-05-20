from PipelineTS.models.nn.backbones import TFT

from PipelineTS.models.nn._wrapper import NNBackboneForecastingMixin


class TFTModel(NNBackboneForecastingMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            hidden_size=32,
            lstm_layers=1,
            n_heads=4,
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
        TFTModel: A wrapper for the TFT model from PipelineTS backbones with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features for training and prediction.
        hidden_size : int, optional, default: 32
            The size of the hidden layer in the TFT model.
        lstm_layers : int, optional, default: 1
            The number of LSTM layers in the TFT model.
        n_heads : int, optional, default: 4
            The number of attention heads.
        dropout : float, optional, default: 0.1
            The dropout rate.
        use_revin : bool, optional, default: True
            Whether to use RevIN normalization.
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

        Attributes
        ----------
        model : PipelineTS.models.nn.backbones.TFT
            The TFT model from PipelineTS backbones.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self._init_backbone_model(
            backbone_cls=TFT,
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
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            n_heads=n_heads,
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
