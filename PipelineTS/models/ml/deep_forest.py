from PipelineTS.ml_model.gbdt import _DirectGBDTMixin
from PipelineTS.ml_model.native_tree_models import gcForestModel


class DeepForestModel(_DirectGBDTMixin):
    """gcForest (Deep Forest) cascade model for time series forecasting.

    Multi-layer cascade of RandomForest + ExtraTrees estimators,
    following the gcForest architecture by Zhou & Feng (2017).
    Each layer's predictions are concatenated with the original features
    to form the input for the next layer.

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
    n_layers : int, optional, default: 3
        Number of cascade layers.
    n_estimators_per_layer : int, optional, default: 100
        Trees per estimator per layer.
    max_depth : int or None, optional
        Maximum depth of each tree.
    min_samples_leaf : int, optional, default: 1
    random_state : int or None, optional
    verbose : bool, optional, default: False

    Examples
    --------
    >>> from PipelineTS.ml_model import DeepForestModel
    >>> model = DeepForestModel(time_col='date', target_col='value', lags=16)
    >>> model.fit(train_data)
    >>> preds = model.predict(n=8)

    >>> from PipelineTS.pipeline import ModelPipeline
    >>> pipe = ModelPipeline(
    ...     time_col='date', target_col='value', lags=16,
    ...     include_models=['deep_forest'],
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
        n_layers=3,
        n_estimators_per_layer=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=None,
        verbose=False,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            n_layers=n_layers,
            n_estimators_per_layer=n_estimators_per_layer,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
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
        return gcForestModel(**self.all_configs['model_configs'])

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_layers'] = min(2, cv_configs.get('n_layers', 3))
        cv_configs['n_estimators_per_layer'] = min(
            50, cv_configs.get('n_estimators_per_layer', 100)
        )
        return gcForestModel(**cv_configs)
