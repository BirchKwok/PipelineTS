from PipelineTS.spinesTS.nn import ITransformer
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.spines_base import SpinesMultivariateNNModelMixin


class ITransformerModel(SpinesMultivariateNNModelMixin):
    _train_on_all_features = True

    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            feature_cols=None,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            e_layers=2,
            factor=1,
            embed='timeF',
            freq='h',
            dropout=0.1,
            activation='gelu',
            output_attention=False,
            use_norm=True,
            class_strategy='projection',
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
            use_ema=False,
            ema_decay=0.999,
            use_swa=False,
            swa_start_frac=0.75,
            warmup_epochs=0,
            use_residual_gate=False
    ):
        """
        ITransformerModel: A wrapper for the ITransformer model from spinesTS.

        The ITransformer (Inverted Transformer) treats each variate as a token
        and applies attention across variates instead of time steps.
        Paper: https://arxiv.org/abs/2310.06625

        Supports three prediction modes:
        - Univariate: target_col='y', feature_cols=None
        - Multi-input, single-output: target_col='y', feature_cols=['a', 'b', 'y']
        - Multi-input, multi-output: target_col=['a', 'b'], feature_cols=['a', 'b', 'c']

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str or list of str
            The column(s) containing the target variable(s) to predict.
        lags : int, optional, default: 6
            The number of lagged values to use as input features.
        feature_cols : list of str or None, optional, default: None
            Input feature columns. If None, uses target_col only (univariate mode).
        d_model : int, optional, default: 512
            Dimensionality of the model.
        n_heads : int, optional, default: 8
            Number of attention heads.
        d_ff : int, optional, default: 2048
            Feedforward dimension.
        e_layers : int, optional, default: 2
            Number of encoder layers.
        factor : int, optional, default: 1
            Attention factor.
        embed : str, optional, default: 'timeF'
            Time features encoding type. Options: 'timeF', 'fixed', 'learned'.
        freq : str, optional, default: 'h'
            Frequency of time series.
        dropout : float, optional, default: 0.1
            Dropout rate.
        activation : str, optional, default: 'gelu'
            Activation function. Options: 'relu', 'gelu'.
        output_attention : bool, optional, default: False
            Whether to output attention weights.
        use_norm : bool, optional, default: True
            Whether to use Non-stationary Transformer normalization.
        class_strategy : str, optional, default: 'projection'
            Classification strategy. Options: 'projection', 'average', 'cls_token'.
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
        model : spinesTS.nn.ITransformer
            The ITransformer model from spinesTS.
        """
        super().__init__(time_col=time_col, target_col=target_col,
                         feature_cols=feature_cols, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            ITransformer,
            in_features=lags,
            out_features=lags,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            factor=factor,
            embed=embed,
            freq=freq,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
            use_norm=use_norm,
            class_strategy=class_strategy,
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
                'target_col': self._primary_target,
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
                'loss_type': loss_type,
                'use_ema': use_ema,
                'ema_decay': ema_decay,
                'use_swa': use_swa,
                'swa_start_frac': swa_start_frac,
                'warmup_epochs': warmup_epochs,
                'use_residual_gate': use_residual_gate
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the ITransformer model from spinesTS.

        Returns
        -------
        spinesTS.nn.ITransformer
            The ITransformer model.
        """
        return ITransformer(**self.all_configs['model_configs'])
