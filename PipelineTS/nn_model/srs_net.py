from PipelineTS.nn_model.backbones import SRSNet
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.model_mixins import MultivariateNNForecastingMixin


class SRSNetModel(MultivariateNNForecastingMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=6,
            feature_cols=None,
            d_model=64,
            patch_sizes=None,
            n_heads=4,
            top_k_ratio=0.5,
            dropout=0.1,
            stride_ratio=0.5,
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
            lr_factor=0.5,
            restore_best_weights=True,
            loss_type='min',
            weight_decay=1e-4,
            use_ema=False,
            ema_decay=0.999,
            use_swa=False,
            swa_start_frac=0.75,
            warmup_epochs=0,
            use_residual_gate=False,
    ):
        """
        SRSNetModel: A wrapper for the SRSNet model from PipelineTS backbones.

        SRSNet (Selective Representation Space Network) uses multi-scale adaptive
        patching and selective representation via attention-based scoring and selection,
        combined with a simple MLP head for time series forecasting.

        Architecture:
            Input -> RevIN -> SRSBlock (Multi-Scale Patching + Selective Representation) -> MLP Head -> Output

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
        d_model : int, optional, default: 64
            Embedding dimension for patch representations.
        patch_sizes : list of int or None, optional, default: None
            Patch sizes for multi-scale patching. Auto-determined if None.
        n_heads : int, optional, default: 4
            Number of attention heads in SRS cross-attention.
        top_k_ratio : float, optional, default: 0.5
            Fraction of patches to retain after selection (0.0 to 1.0).
        dropout : float, optional, default: 0.1
            Dropout rate.
        stride_ratio : float, optional, default: 0.5
            Stride as fraction of patch size (controls overlap).
        use_revin : bool, optional, default: True
            Whether to use Reversible Instance Normalization.
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
        lr_factor : float, optional, default: 0.5
            LR reduction factor.
        restore_best_weights : bool, optional, default: True
            Whether to restore best weights.
        loss_type : str, optional, default: 'min'
            Loss type for early stopping.
        weight_decay : float, optional, default: 1e-4
            Weight decay for AdamW.

        Attributes
        ----------
        model : PipelineTS.nn_model.backbones.SRSNet
            The SRSNet model from PipelineTS backbones.
        """
        super().__init__(time_col=time_col, target_col=target_col,
                         feature_cols=feature_cols, accelerator=accelerator)

        self.all_configs['model_configs'] = generate_function_kwargs(
            SRSNet,
            in_features=lags,
            out_features=lags,
            n_vars=self._n_vars,
            n_targets=self._n_targets,
            d_model=d_model,
            patch_sizes=patch_sizes,
            n_heads=n_heads,
            top_k_ratio=top_k_ratio,
            dropout=dropout,
            stride_ratio=stride_ratio,
            use_revin=use_revin,
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
                'use_residual_gate': use_residual_gate,
            }
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the SRSNet model from PipelineTS backbones.

        Returns
        -------
        PipelineTS.nn_model.backbones.SRSNet
            The SRSNet model.
        """
        return SRSNet(**self.all_configs['model_configs'])
