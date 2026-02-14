"""GPU-accelerated differentiable tree ensembles for PipelineTS.

Provides three complementary PyTorch-based tree ensemble architectures
inspired by Google's Yggdrasil Decision Forests:

- ``TorchBoostingForestModel``  — staged gradient boosting (MART/DART)
  with GrowNet corrective step.  Replaces LightGBM / XGBoost / CatBoost.
- ``TorchBaggingForestModel``   — bagging ensemble with tree-level dropout
  for decorrelation.  Replaces RandomForest.
- ``TorchDeepForestModel``      — cascade (gcForest) multi-layer ensemble.

All models inherit ``_DirectGBDTMixin`` for full time series support:
lag feature engineering, autoregressive multi-step prediction, multi-series
(id_col) support, covariate support, and conformal prediction intervals.

Key YDF-inspired improvements:
- Exponential recency sample weighting for structural-break adaptation
- Validation-based early stopping with look-ahead
- Temperature-annealed sparse feature selection
- Staged residual boosting with GrowNet corrective step

GPU acceleration is automatic: CUDA > MPS > CPU fallback.
"""

from PipelineTS.ml_model.gbdt import _DirectGBDTMixin
from PipelineTS.ml_model._torch_tree import _TorchTreeWrapper


class TorchBoostingForestModel(_DirectGBDTMixin):
    """GPU-accelerated gradient boosting forest (MART/DART).

    Unified differentiable boosting ensemble that subsumes LightGBM,
    XGBoost and CatBoost via staged residual learning with oblivious
    decision trees trained end-to-end on GPU.

    Parameters
    ----------
    time_col : str
    target_col : str
    lags : int
    quantile : float or None
    accelerator : str or None
        'cuda', 'mps', 'cpu', 'auto', or None (auto-detect).
    n_trees : int
        Number of trees per boosting stage.
    tree_depth : int
    learning_rate : float
    n_epochs : int
    batch_size : int
        0 = full batch.
    early_stop_patience : int
    dropout : float
    weight_decay : float
    boosting_stages : int
        Number of sequential residual boosting stages.
    boosting_shrinkage : float
        Shrinkage applied to each stage's contribution.
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        time_col,
        target_col,
        lags=1,
        quantile=0.9,
        accelerator=None,
        n_trees=64,
        tree_depth=5,
        learning_rate=0.08,
        n_epochs=200,
        batch_size=0,
        early_stop_patience=15,
        dropout=0.0,
        weight_decay=1e-4,
        boosting_stages=3,
        boosting_shrinkage=0.5,
        random_state=None,
        verbose=False,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            n_trees=n_trees,
            tree_depth=tree_depth,
            ensemble_mode='additive',
            n_layers=1,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            dropout=dropout,
            weight_decay=weight_decay,
            boosting_stages=boosting_stages,
            boosting_shrinkage=boosting_shrinkage,
            accelerator=accelerator,
            random_state=random_state,
            verbose=verbose,
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
        cv_configs['n_trees'] = min(16, cv_configs.get('n_trees', 64))
        cv_configs['n_epochs'] = min(50, cv_configs.get('n_epochs', 200))
        return _TorchTreeWrapper(**cv_configs)


class TorchBaggingForestModel(_DirectGBDTMixin):
    """GPU-accelerated bagging forest (RandomForest-style).

    Differentiable bagging ensemble where each tree votes independently
    and tree-level dropout during training decorrelates the ensemble,
    analogous to random subspace selection in classical Random Forests.

    Parameters
    ----------
    time_col : str
    target_col : str
    lags : int
    quantile : float or None
    accelerator : str or None
    n_trees : int
    tree_depth : int
    learning_rate : float
    n_epochs : int
    batch_size : int
    early_stop_patience : int
    dropout : float
        Tree-level dropout for decorrelation.
    weight_decay : float
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        time_col,
        target_col,
        lags=1,
        quantile=0.9,
        accelerator=None,
        n_trees=128,
        tree_depth=5,
        learning_rate=0.08,
        n_epochs=300,
        batch_size=0,
        early_stop_patience=15,
        dropout=0.15,
        weight_decay=1e-4,
        boosting_stages=3,
        boosting_shrinkage=0.5,
        random_state=None,
        verbose=False,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            n_trees=n_trees,
            tree_depth=tree_depth,
            ensemble_mode='additive',
            n_layers=1,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            dropout=dropout,
            weight_decay=weight_decay,
            boosting_stages=boosting_stages,
            boosting_shrinkage=boosting_shrinkage,
            accelerator=accelerator,
            random_state=random_state,
            verbose=verbose,
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
        cv_configs['n_trees'] = min(16, cv_configs.get('n_trees', 128))
        cv_configs['n_epochs'] = min(50, cv_configs.get('n_epochs', 300))
        return _TorchTreeWrapper(**cv_configs)


# Backward compatibility alias — TorchDeepForestModel has been merged into
# DeepForestModel in deep_forest.py.  Keep this import so existing code
# referencing TorchDeepForestModel from this module still works.
from PipelineTS.ml_model.deep_forest import DeepForestModel as TorchDeepForestModel
