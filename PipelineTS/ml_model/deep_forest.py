from PipelineTS.ml_model.gbdt import _DirectGBDTMixin
from PipelineTS.ml_model._torch_tree import _TorchTreeWrapper


class DeepForestModel(_DirectGBDTMixin):
    """GPU-accelerated Deep Forest (gcForest) for time series forecasting.

    Multi-layer cascade of differentiable tree ensembles (Zhou & Feng 2017).
    Each layer's tree outputs are concatenated with the original features
    and fed to the next layer — trained end-to-end via backpropagation.
    Supports GPU acceleration via ``accelerator='cuda'``.

    Inherits the full time-series infrastructure from ``_DirectGBDTMixin``:
    lag feature engineering (26 statistical features), autoregressive
    multi-step prediction, multi-series (id_col) support, covariate
    support, and conformal prediction intervals.

    Parameters
    ----------
    time_col : str
        Time column name.
    target_col : str
        Target column name.
    lags : int, optional, default: 1
        Number of lagged time steps.
    quantile : float or None, optional, default: 0.9
        Quantile for prediction intervals.
    accelerator : str or None, optional
        ``'cuda'``, ``'mps'``, ``'cpu'``, or ``None`` (auto-detect).
    n_trees : int, optional, default: 32
        Trees per cascade layer.
    tree_depth : int, optional, default: 4
        Depth of each oblivious decision tree.
    n_layers : int, optional, default: 3
        Number of cascade layers.
    learning_rate : float, optional, default: 0.08
    n_epochs : int, optional, default: 200
    batch_size : int, optional, default: 0
        0 = full batch.
    early_stop_patience : int, optional, default: 12
    dropout : float, optional, default: 0.1
    weight_decay : float, optional, default: 1e-4
    random_state : int or None, optional
    verbose : bool, optional, default: False

    Examples
    --------
    >>> from PipelineTS.ml_model import DeepForestModel
    >>> model = DeepForestModel(time_col='date', target_col='value', lags=16)
    >>> model.fit(train_data)
    >>> preds = model.predict(n=8)

    >>> # GPU acceleration
    >>> model = DeepForestModel(
    ...     time_col='date', target_col='value', lags=16,
    ...     accelerator='cuda',
    ... )

    >>> from PipelineTS.pipeline import ModelPipeline
    >>> pipe = ModelPipeline(
    ...     time_col='date', target_col='value', lags=16,
    ...     include_models=['deep_forest'],
    ...     deep_forest__n_trees=48,
    ...     deep_forest__n_layers=4,
    ... )
    >>> pipe.fit(data)
    """

    def __init__(
        self,
        time_col,
        target_col,
        lags=1,
        quantile=0.9,
        accelerator=None,
        n_trees=32,
        tree_depth=4,
        n_layers=3,
        learning_rate=0.08,
        n_epochs=200,
        batch_size=0,
        early_stop_patience=12,
        dropout=0.1,
        weight_decay=1e-4,
        random_state=None,
        verbose=False,
        auto_complexity=False,
        diversity_weight=0.01,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            n_trees=n_trees,
            tree_depth=tree_depth,
            ensemble_mode='cascade',
            n_layers=n_layers,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            dropout=dropout,
            weight_decay=weight_decay,
            accelerator=accelerator,
            random_state=random_state,
            verbose=verbose,
            auto_complexity=auto_complexity,
            diversity_weight=diversity_weight,
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return _TorchTreeWrapper(**self.all_configs['model_configs'])

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_trees'] = min(8, cv_configs.get('n_trees', 24))
        cv_configs['n_epochs'] = min(40, cv_configs.get('n_epochs', 120))
        cv_configs['n_layers'] = min(2, cv_configs.get('n_layers', 2))
        return _TorchTreeWrapper(**cv_configs)
