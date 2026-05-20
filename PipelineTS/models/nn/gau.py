from PipelineTS.models.nn.backbones import GAUNet
from PipelineTS.models.nn._wrapper import NNBackboneForecastingMixin


class GAUModel(NNBackboneForecastingMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=30,
            quantile=0.9,
            random_state=None,
            level=3,
            learning_rate=0.001,
            accelerator='auto',
            verbose=False,
            epochs=1500,
            batch_size='auto',
            patience=80,
            min_delta=0,
            lr_scheduler='OneCycleLR',
            lr_scheduler_patience=10,
            lr_factor=0.5,
            restore_best_weights=True,
            loss_type='min',
            dropout=0.2,
            weight_decay=1e-4,
            query_key_dim=512,
            expansion_factor=4.0,
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
        GAUModel: A wrapper for the GAUNet neural network model.

        The model uses a learned feature projection head inside GAUBase to automatically
        extract rich per-timestep features from raw input values, enabling proper temporal
        attention across all lag time steps. This replaces external feature engineering.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 30
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        level : int, optional, default: 3
            The number of stacked GAU layers.
        learning_rate : float, optional, default: 0.001
            The learning rate for training the GAUNet model.
        accelerator : str, optional, default: 'auto'
            The PyTorch Lightning accelerator to use during training.
        verbose : bool, optional, default: False
            Whether to print detailed information during training.
        epochs : int, optional, default: 1500
            The number of epochs for training the GAUNet model.
        batch_size : int or 'auto', optional, default: 'auto'
            The batch size used during training. If set to 'auto', it is determined automatically.
        patience : int, optional, default: 80
            The patience parameter for early stopping during training.
        min_delta : float, optional, default: 0
            The minimum change in the monitored quantity to qualify as an improvement during early stopping.
        lr_scheduler : str, optional, default: 'OneCycleLR'
            The learning rate scheduler used during training.
        lr_scheduler_patience : int, optional, default: 10
            The patience parameter for the learning rate scheduler.
        lr_factor : float, optional, default: 0.5
            The factor by which the learning rate is reduced during training.
        restore_best_weights : bool, optional, default: True
            Whether to restore the model weights from the epoch with the best value of the monitored quantity.
        loss_type : str, optional, default: 'min'
            The type of loss function to use during training.
        dropout : float, optional, default: 0.2
            The dropout rate for the GAUNet model.
        weight_decay : float, optional, default: 1e-4
            The weight decay for the optimizer.
        query_key_dim : int, optional, default: 512
            The query key dimension for the GAU layer.
        expansion_factor : float, optional, default: 4.0
            The expansion factor for the GAU layer.

        Attributes
        ----------
        model : PipelineTS.models.nn.backbones.GAUNet
            The GAUNet neural network model.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self._init_backbone_model(
            backbone_cls=GAUNet,
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
            level=level,
            learning_rate=learning_rate,
            random_seed=random_state,
            device=self.accelerator,
            loss_fn='huber',
            dropout=dropout,
            weight_decay=weight_decay,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            use_gtb=use_gtb,
            gtb_d_model=gtb_d_model,
            routing_mode=routing_mode
            ),
        )
