"""Tree ensemble models for PipelineTS.

Provides tree-based ensemble architectures for time series forecasting:

- ``BoostingForestModel``  — XGBoost-based gradient boosting ensemble.
- ``BaggingForestModel``   — RandomForest-based bagging ensemble.
- ``DeepForestModel``      — gcForest cascade ensemble (see deep_forest.py).

All models inherit ``_DirectGBDTMixin`` for full time series support:
lag feature engineering, autoregressive multi-step prediction, multi-series
(id_col) support, covariate support, and conformal prediction intervals.
"""

from PipelineTS.ml_model.gbdt import _DirectGBDTMixin
from PipelineTS.ml_model.native_tree_models import _XGBoostWrapper
from sklearn.ensemble import RandomForestRegressor


class BoostingForestModel(_DirectGBDTMixin):
    """XGBoost-based gradient boosting forest for time series forecasting.

    Uses XGBoost gradient boosting with configurable number of trees
    and early stopping for time series prediction.

    Parameters
    ----------
    time_col : str
    target_col : str
    lags : int
    quantile : float or None
    n_estimators : int
        Number of boosting trees.
    max_depth : int
        Maximum depth of each tree.
    learning_rate : float
    early_stopping_rounds : int
    subsample : float
        Subsample ratio of training instances.
    colsample_bytree : float
        Subsample ratio of features.
    reg_lambda : float
        L2 regularization.
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        time_col,
        target_col,
        lags=1,
        quantile=0.9,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=None,
        verbose=False,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbosity=0 if not verbose else 1,
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
        return _XGBoostWrapper(**self.all_configs['model_configs'])

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 500))
        return _XGBoostWrapper(**cv_configs)


class BaggingForestModel(_DirectGBDTMixin):
    """RandomForest-based bagging ensemble for time series forecasting.

    Uses scikit-learn's RandomForestRegressor with configurable number
    of trees and feature/row subsampling.

    Parameters
    ----------
    time_col : str
    target_col : str
    lags : int
    quantile : float or None
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree.
    max_samples : float
        Fraction of samples used for each tree.
    max_features : float
        Fraction of features used for each tree.
    random_state : int or None
    n_jobs : int
        Number of parallel jobs.
    """

    def __init__(
        self,
        time_col,
        target_col,
        lags=1,
        quantile=0.9,
        n_estimators=100,
        max_depth=None,
        max_samples=0.8,
        max_features=0.8,
        random_state=None,
        n_jobs=-1,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
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
        return RandomForestRegressor(**self.all_configs['model_configs'])

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(50, cv_configs.get('n_estimators', 128))
        return RandomForestRegressor(**cv_configs)


# Backward compatibility aliases
from PipelineTS.ml_model.deep_forest import DeepForestModel

TorchBoostingForestModel = BoostingForestModel
TorchBaggingForestModel = BaggingForestModel
TorchDeepForestModel = DeepForestModel
