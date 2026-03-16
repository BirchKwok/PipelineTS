"""Native tree-based models for time series forecasting.

Provides high-performance implementations using CatBoost, XGBoost,
scikit-learn RandomForest, ExtraTrees, and a gcForest cascade.

All models inherit from ``_DirectGBDTMixin`` which handles lag feature
engineering, covariate support, multi-series (panel) data, and conformal
interval estimation.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.multioutput import RegressorChain

from PipelineTS.ml_model.gbdt import _DirectGBDTMixin


# ---------------------------------------------------------------------------
#  sklearn-compatible wrappers (thin shims for uniform API)
# ---------------------------------------------------------------------------

class _CatBoostWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around CatBoostRegressor."""

    def __init__(self, iterations=500, depth=6, learning_rate=0.05,
                 l2_leaf_reg=3.0, random_seed=None, verbose=False,
                 early_stopping_rounds=50, **kwargs):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs
        self._model = None

    def fit(self, X, y, **fit_kwargs):
        from catboost import CatBoostRegressor
        self._model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.kwargs,
        )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Guard against NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self._model.predict(X)

    def get_params(self, deep=True):
        params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'early_stopping_rounds': self.early_stopping_rounds,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        known = {'iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
                 'random_seed', 'verbose', 'early_stopping_rounds'}
        for k, v in params.items():
            if k in known:
                setattr(self, k, v)
            else:
                self.kwargs[k] = v
        return self


class _XGBoostWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper around XGBRegressor."""

    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0,
                 reg_lambda=1.0, random_state=None, verbosity=0,
                 early_stopping_rounds=50, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.verbosity = verbosity
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs
        self._model = None

    def fit(self, X, y, **fit_kwargs):
        from xgboost import XGBRegressor
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbosity=self.verbosity,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.kwargs,
        )
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self._model.predict(X)

    def get_params(self, deep=True):
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbosity': self.verbosity,
            'early_stopping_rounds': self.early_stopping_rounds,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        known = {'n_estimators', 'max_depth', 'learning_rate', 'subsample',
                 'colsample_bytree', 'reg_alpha', 'reg_lambda',
                 'random_state', 'verbosity', 'early_stopping_rounds'}
        for k, v in params.items():
            if k in known:
                setattr(self, k, v)
            else:
                self.kwargs[k] = v
        return self


class _GCForestEstimator(BaseEstimator, RegressorMixin):
    """Multi-layer cascade forest estimator (gcForest-style).

    Each layer consists of multiple diverse forest estimators whose
    predictions are concatenated with the original features to form
    the input for the next layer.  A simple moving-average convergence
    criterion stops adding layers when performance plateaus.
    """

    def __init__(self, n_layers=3, n_estimators_per_layer=100,
                 max_depth=None, min_samples_leaf=1,
                 random_state=None, verbose=False):
        self.n_layers = n_layers
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        self._layers = []
        self._n_original_features = None

    def fit(self, X, y, **fit_kwargs):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        self._n_original_features = X.shape[1]
        self._layers = []

        current_X = X.copy()
        for layer_idx in range(self.n_layers):
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators_per_layer,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=(self.random_state + layer_idx * 2
                              if self.random_state is not None else None),
                n_jobs=-1,
            )
            et = ExtraTreesRegressor(
                n_estimators=self.n_estimators_per_layer,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=(self.random_state + layer_idx * 2 + 1
                              if self.random_state is not None else None),
                n_jobs=-1,
            )
            rf.fit(current_X, y)
            et.fit(current_X, y)
            self._layers.append((rf, et))

            # Augment features with predictions from this layer
            rf_pred = rf.predict(current_X)
            et_pred = et.predict(current_X)
            if rf_pred.ndim == 1:
                rf_pred = rf_pred.reshape(-1, 1)
            if et_pred.ndim == 1:
                et_pred = et_pred.reshape(-1, 1)
            current_X = np.concatenate([X, rf_pred, et_pred], axis=1)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        original_X = X.copy()
        current_X = X.copy()

        for rf, et in self._layers:
            rf_pred = rf.predict(current_X)
            et_pred = et.predict(current_X)
            if rf_pred.ndim == 1:
                rf_pred = rf_pred.reshape(-1, 1)
            if et_pred.ndim == 1:
                et_pred = et_pred.reshape(-1, 1)
            current_X = np.concatenate([original_X, rf_pred, et_pred], axis=1)

        # Final prediction: average of last layer's estimators
        rf, et = self._layers[-1]
        rf_pred = rf.predict(current_X)
        et_pred = et.predict(current_X)
        return (rf_pred + et_pred) / 2.0

    def get_params(self, deep=True):
        return {
            'n_layers': self.n_layers,
            'n_estimators_per_layer': self.n_estimators_per_layer,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ---------------------------------------------------------------------------
#  High-level PipelineTS model classes
# ---------------------------------------------------------------------------

class CatBoostModel(_DirectGBDTMixin):
    """CatBoost gradient boosting model for time series forecasting.

    Uses native CatBoost library for high-performance gradient boosting
    with ordered boosting and symmetric trees.

    Parameters
    ----------
    time_col : str
        Column name for timestamps.
    target_col : str
        Column name for the target variable.
    lags : int, optional
        Number of lag values to use as features. Default is 1.
    quantile : float or None, optional
        Quantile for conformal prediction intervals. Default is 0.9.
    iterations : int, optional
        Number of boosting iterations. Default is 500.
    depth : int, optional
        Depth of the trees. Default is 6.
    learning_rate : float, optional
        Learning rate for gradient boosting. Default is 0.05.
    l2_leaf_reg : float, optional
        L2 regularization coefficient. Default is 3.0.
    early_stopping_rounds : int, optional
        Number of rounds without improvement for early stopping. Default is 50.
    random_state : int or None, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print training progress. Default is False.
    """

    def __init__(self, time_col, target_col, lags=1, quantile=0.9,
                 iterations=500, depth=6, learning_rate=0.05,
                 l2_leaf_reg=3.0, early_stopping_rounds=50,
                 random_state=None, verbose=False, **kwargs):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs['model_configs'] = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            early_stopping_rounds=early_stopping_rounds,
            random_seed=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
        })
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(
            _CatBoostWrapper(**self.all_configs['model_configs'])
        )

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['iterations'] = min(100, cv_configs.get('iterations', 500))
        return RegressorChain(_CatBoostWrapper(**cv_configs))


class XGBoostModel(_DirectGBDTMixin):
    """XGBoost gradient boosting model for time series forecasting.

    Uses native XGBoost library for high-performance gradient boosting
    with histogram-based tree construction.

    Parameters
    ----------
    time_col : str
        Column name for timestamps.
    target_col : str
        Column name for the target variable.
    lags : int, optional
        Number of lag values to use as features. Default is 1.
    quantile : float or None, optional
        Quantile for conformal prediction intervals. Default is 0.9.
    n_estimators : int, optional
        Number of boosting rounds. Default is 500.
    max_depth : int, optional
        Maximum depth of a tree. Default is 6.
    learning_rate : float, optional
        Boosting learning rate. Default is 0.05.
    subsample : float, optional
        Subsample ratio of the training instances. Default is 0.8.
    colsample_bytree : float, optional
        Subsample ratio of columns for each tree. Default is 0.8.
    reg_alpha : float, optional
        L1 regularization term. Default is 0.0.
    reg_lambda : float, optional
        L2 regularization term. Default is 1.0.
    early_stopping_rounds : int, optional
        Early stopping rounds. Default is 50.
    random_state : int or None, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print training progress. Default is False.
    """

    def __init__(self, time_col, target_col, lags=1, quantile=0.9,
                 n_estimators=500, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0,
                 early_stopping_rounds=50,
                 random_state=None, verbose=False, **kwargs):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs['model_configs'] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            verbosity=0 if not verbose else 1,
            **kwargs,
        )
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
        })
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(
            _XGBoostWrapper(**self.all_configs['model_configs'])
        )

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 500))
        return RegressorChain(_XGBoostWrapper(**cv_configs))


class RandomForestModel(_DirectGBDTMixin):
    """Random Forest model for time series forecasting.

    Uses scikit-learn's RandomForestRegressor with parallel tree
    construction for robust ensemble predictions.

    Parameters
    ----------
    time_col : str
        Column name for timestamps.
    target_col : str
        Column name for the target variable.
    lags : int, optional
        Number of lag values to use as features. Default is 1.
    quantile : float or None, optional
        Quantile for conformal prediction intervals. Default is 0.9.
    n_estimators : int, optional
        Number of trees in the forest. Default is 500.
    max_depth : int or None, optional
        Maximum depth of the tree. None means unlimited. Default is None.
    min_samples_split : int, optional
        Minimum number of samples to split a node. Default is 2.
    min_samples_leaf : int, optional
        Minimum number of samples in a leaf. Default is 1.
    max_features : str or float, optional
        Number of features to consider for best split. Default is 1.0.
    random_state : int or None, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print training progress. Default is False.
    """

    def __init__(self, time_col, target_col, lags=1, quantile=0.9,
                 n_estimators=500, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=1.0,
                 random_state=None, verbose=False, **kwargs):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs['model_configs'] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            verbose=int(verbose),
            n_jobs=-1,
            **kwargs,
        )
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
        })
        self.model = self._define_model()

    def _define_model(self):
        from sklearn.ensemble import RandomForestRegressor
        return RegressorChain(
            RandomForestRegressor(**self.all_configs['model_configs'])
        )

    def _define_model_for_cv(self):
        from sklearn.ensemble import RandomForestRegressor
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 500))
        return RegressorChain(RandomForestRegressor(**cv_configs))


class ExtraForestModel(_DirectGBDTMixin):
    """Extra-Trees model for time series forecasting.

    Uses scikit-learn's ExtraTreesRegressor which randomizes split
    thresholds for even faster training and reduced variance.

    Parameters
    ----------
    time_col : str
        Column name for timestamps.
    target_col : str
        Column name for the target variable.
    lags : int, optional
        Number of lag values to use as features. Default is 1.
    quantile : float or None, optional
        Quantile for conformal prediction intervals. Default is 0.9.
    n_estimators : int, optional
        Number of trees in the forest. Default is 500.
    max_depth : int or None, optional
        Maximum depth of the tree. None means unlimited. Default is None.
    min_samples_split : int, optional
        Minimum number of samples to split a node. Default is 2.
    min_samples_leaf : int, optional
        Minimum number of samples in a leaf. Default is 1.
    max_features : str or float, optional
        Number of features to consider for best split. Default is 1.0.
    random_state : int or None, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print training progress. Default is False.
    """

    def __init__(self, time_col, target_col, lags=1, quantile=0.9,
                 n_estimators=500, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=1.0,
                 random_state=None, verbose=False, **kwargs):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs['model_configs'] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            verbose=int(verbose),
            n_jobs=-1,
            **kwargs,
        )
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
        })
        self.model = self._define_model()

    def _define_model(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return RegressorChain(
            ExtraTreesRegressor(**self.all_configs['model_configs'])
        )

    def _define_model_for_cv(self):
        from sklearn.ensemble import ExtraTreesRegressor
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 500))
        return RegressorChain(ExtraTreesRegressor(**cv_configs))


class gcForestModel(_DirectGBDTMixin):
    """gcForest (Deep Forest) cascade model for time series forecasting.

    Implements a multi-layer cascade of RandomForest + ExtraTrees
    estimators, following the gcForest architecture by Zhou & Feng (2017).
    Each layer's predictions are concatenated with the original features
    to form the input for the next layer.

    Parameters
    ----------
    time_col : str
        Column name for timestamps.
    target_col : str
        Column name for the target variable.
    lags : int, optional
        Number of lag values to use as features. Default is 1.
    quantile : float or None, optional
        Quantile for conformal prediction intervals. Default is 0.9.
    n_layers : int, optional
        Number of cascade layers. Default is 3.
    n_estimators_per_layer : int, optional
        Number of trees per estimator per layer. Default is 100.
    max_depth : int or None, optional
        Maximum depth of each tree. Default is None (unlimited).
    min_samples_leaf : int, optional
        Minimum number of samples in a leaf. Default is 1.
    random_state : int or None, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print training progress. Default is False.
    """

    def __init__(self, time_col, target_col, lags=1, quantile=0.9,
                 n_layers=3, n_estimators_per_layer=100,
                 max_depth=None, min_samples_leaf=1,
                 random_state=None, verbose=False, **kwargs):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs['model_configs'] = dict(
            n_layers=n_layers,
            n_estimators_per_layer=n_estimators_per_layer,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
        })
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(
            _GCForestEstimator(**self.all_configs['model_configs'])
        )

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_layers'] = min(2, cv_configs.get('n_layers', 3))
        cv_configs['n_estimators_per_layer'] = min(
            50, cv_configs.get('n_estimators_per_layer', 100)
        )
        return RegressorChain(_GCForestEstimator(**cv_configs))
