from PipelineTS.spinesTS.nn import TSTransformer
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.spines_base import SpinesNNModelMixin


class TransformerModel(SpinesNNModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            use_revin=True,
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
            weight_decay=1e-4,
            use_gtb=False,
            gtb_d_model=64,
            routing_mode='static'
    ):
        """
        TransformerModel: A wrapper for the Transformer model from spinesTS.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 6
            The number of lagged values to use as input features.
        d_model : int, optional, default: 64
            Dimensionality of the model.
        nhead : int, optional, default: 4
            Number of attention heads.
        num_encoder_layers : int, optional, default: 3
            Number of encoder layers.
        dim_feedforward : int, optional, default: 256
            Feedforward dimension.
        dropout : float, optional, default: 0.1
            Dropout rate.
        use_revin : bool, optional, default: True
            Whether to use RevIN normalization.
        quantile : float, optional, default: 0.9
            Quantile for interval prediction.
        random_state : int or None, optional, default: None
            Random seed.
        learning_rate : float, optional, default: 0.001
            Learning rate.
        accelerator : str, optional, default: 'auto'
            Accelerator for training.
        verbose : bool, optional, default: False
            Whether to display verbose output.
        epochs : int, optional, default: 1000
            Number of training epochs.
        batch_size : int or 'auto', optional, default: 'auto'
            Batch size.
        patience : int, optional, default: 20
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
        model : spinesTS.nn.TSTransformer
            The Transformer model from spinesTS.
        """
        super().__init__(time_col=time_col, target_col=target_col, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            TSTransformer,
            in_features=lags,
            out_features=lags,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
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
        Define the Transformer model from spinesTS.

        Returns
        -------
        spinesTS.nn.TSTransformer
            The Transformer model.
        """
        return TSTransformer(**self.all_configs['model_configs'])
