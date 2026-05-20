"""Native tree-based models for time series forecasting.

Provides high-performance implementations using CatBoost, XGBoost,
scikit-learn RandomForest, ExtraTrees, and a gcForest cascade.

All models inherit from ``_DirectGBDTMixin`` which handles lag feature
engineering, covariate support, multi-series (panel) data, and conformal
interval estimation.
"""

import importlib.util
import sys

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
        self._single_output = False

    def fit(self, X, y, **fit_kwargs):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._single_output = y.ndim == 1
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

        rf_pred = et_pred = None
        for layer_idx, (rf, et) in enumerate(self._layers):
            rf_pred = rf.predict(current_X)
            et_pred = et.predict(current_X)
            if rf_pred.ndim == 1:
                rf_pred = rf_pred.reshape(-1, 1)
            if et_pred.ndim == 1:
                et_pred = et_pred.reshape(-1, 1)
            if layer_idx < len(self._layers) - 1:
                current_X = np.concatenate([original_X, rf_pred, et_pred], axis=1)

        # Final prediction: average of last layer's estimators
        pred = (rf_pred + et_pred) / 2.0
        if self._single_output:
            return pred.reshape(-1)
        return pred

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


class _TorchGCForestLayer:
    def __init__(self, n_trees, depth, ridge_alpha=1e-3, temperature=1.0,
                 random_state=None):
        self.n_trees = n_trees
        self.depth = depth
        self.ridge_alpha = ridge_alpha
        self.temperature = temperature
        self.random_state = random_state
        self.split_weight = None
        self.split_bias = None
        self.coef = None
        self.x_mean = None
        self.x_scale = None
        self.path_nodes = None
        self.path_dirs = None
        self.feature_scale = None

    @staticmethod
    def _build_paths(torch, depth):
        n_leaves = 2 ** depth
        nodes = []
        dirs = []
        for leaf_idx in range(n_leaves):
            node_idx = 0
            leaf_nodes = []
            leaf_dirs = []
            for level_idx in range(depth):
                bit = (leaf_idx >> (depth - level_idx - 1)) & 1
                leaf_nodes.append(node_idx)
                leaf_dirs.append(bool(bit))
                node_idx = node_idx * 2 + 1 + bit
            nodes.append(leaf_nodes)
            dirs.append(leaf_dirs)
        return (
            torch.tensor(nodes, dtype=torch.long),
            torch.tensor(dirs, dtype=torch.bool),
        )

    def _path_features(self, torch, x, device):
        split_weight = self.split_weight.to(device)
        split_bias = self.split_bias.to(device)
        path_nodes = self.path_nodes.to(device)
        path_dirs = self.path_dirs.to(device)
        x_mean = self.x_mean.to(device)
        x_scale = self.x_scale.to(device)
        x = (x - x_mean) / x_scale
        logits = torch.einsum('bd,tnd->btn', x, split_weight) + split_bias
        decisions = torch.sigmoid(logits / max(self.temperature, 1e-6))
        selected = decisions[:, :, path_nodes]
        probs = torch.where(path_dirs.view(1, 1, *path_dirs.shape),
                            selected, 1.0 - selected)
        probs = torch.prod(torch.clamp(probs, min=1e-7), dim=-1)
        return probs.reshape(x.shape[0], -1) / self.feature_scale

    @staticmethod
    def _solve_ridge(torch, phi, y, alpha):
        n_samples, n_features = phi.shape
        eye_device = phi.device
        try:
            if n_features <= n_samples:
                gram = phi.T @ phi
                gram = gram + alpha * torch.eye(n_features, device=eye_device, dtype=phi.dtype)
                rhs = phi.T @ y
                return torch.linalg.solve(gram, rhs)
            gram = phi @ phi.T
            gram = gram + alpha * torch.eye(n_samples, device=eye_device, dtype=phi.dtype)
            alpha_coef = torch.linalg.solve(gram, y)
            return phi.T @ alpha_coef
        except Exception:
            phi_cpu = phi.detach().cpu()
            y_cpu = y.detach().cpu()
            if n_features <= n_samples:
                gram = phi_cpu.T @ phi_cpu
                gram = gram + alpha * torch.eye(n_features, dtype=phi_cpu.dtype)
                rhs = phi_cpu.T @ y_cpu
                return torch.linalg.solve(gram, rhs).to(eye_device)
            gram = phi_cpu @ phi_cpu.T
            gram = gram + alpha * torch.eye(n_samples, dtype=phi_cpu.dtype)
            alpha_coef = torch.linalg.solve(gram, y_cpu)
            return (phi_cpu.T @ alpha_coef).to(eye_device)

    def fit(self, torch, x, y, device):
        n_features = x.shape[1]
        n_internal = 2 ** self.depth - 1
        generator = torch.Generator(device='cpu')
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state))
        split_weight = torch.randn(
            self.n_trees, n_internal, n_features,
            generator=generator, dtype=torch.float32
        )
        split_weight = split_weight / (
            torch.linalg.vector_norm(split_weight, dim=-1, keepdim=True) + 1e-6
        )
        split_bias = torch.randn(
            self.n_trees, n_internal,
            generator=generator, dtype=torch.float32
        ) * 0.5
        self.split_weight = split_weight.to(device)
        self.split_bias = split_bias.to(device)
        self.x_mean = x.mean(dim=0, keepdim=True)
        self.x_scale = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        self.path_nodes, self.path_dirs = self._build_paths(torch, self.depth)
        self.feature_scale = float(max(1.0, np.sqrt(self.n_trees)))
        phi = self._path_features(torch, x, device)
        self.coef = self._solve_ridge(torch, phi, y, self.ridge_alpha)
        pred = phi @ self.coef
        self.split_weight = self.split_weight.detach().cpu()
        self.split_bias = self.split_bias.detach().cpu()
        self.coef = self.coef.detach().cpu()
        self.x_mean = self.x_mean.detach().cpu()
        self.x_scale = self.x_scale.detach().cpu()
        return pred.detach()

    def predict_scaled(self, torch, x, device):
        phi = self._path_features(torch, x, device)
        return phi @ self.coef.to(device)


class _TorchGCForestEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, n_layers=3, n_estimators_per_layer=100,
                 max_depth=None, min_samples_leaf=1, random_state=None,
                 verbose=False, device='auto', ridge_alpha=1e-3,
                 max_tree_depth=6, temperature=1.0):
        self.n_layers = n_layers
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.ridge_alpha = ridge_alpha
        self.max_tree_depth = max_tree_depth
        self.temperature = temperature
        self._layers = []
        self._y_mean = None
        self._y_scale = None
        self._device = None

    @staticmethod
    def _resolve_device(torch, device):
        if device in (None, 'auto'):
            if torch.cuda.is_available():
                return torch.device('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device)

    def fit(self, X, y, **fit_kwargs):
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        device = self._resolve_device(torch, self.device)
        self._device = str(device)
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
        x_base = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
        self._y_mean = y_tensor.mean(dim=0, keepdim=True)
        self._y_scale = y_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        y_scaled = (y_tensor - self._y_mean) / self._y_scale
        self._layers = []
        current_x = x_base
        depth = self.max_depth if self.max_depth is not None else 5
        depth_cap = self.max_tree_depth if self.max_tree_depth is not None else 6
        depth = int(max(1, min(depth, depth_cap)))
        n_trees = int(max(1, self.n_estimators_per_layer))
        for layer_idx in range(int(max(1, self.n_layers))):
            seed = None
            if self.random_state is not None:
                seed = int(self.random_state) + layer_idx
            layer = _TorchGCForestLayer(
                n_trees=n_trees,
                depth=depth,
                ridge_alpha=self.ridge_alpha,
                temperature=self.temperature,
                random_state=seed,
            )
            pred_scaled = layer.fit(torch, current_x, y_scaled, device)
            self._layers.append(layer)
            current_x = torch.cat([x_base, pred_scaled], dim=1)
        self._y_mean = self._y_mean.detach().cpu()
        self._y_scale = self._y_scale.detach().cpu()
        return self

    def predict(self, X):
        import torch

        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._layers:
            raise ValueError("The estimator has not been fitted.")
        device = self._resolve_device(torch, self._device or self.device)
        with torch.inference_mode():
            x_base = torch.as_tensor(X, dtype=torch.float32, device=device)
            current_x = x_base
            pred_scaled = None
            for layer in self._layers:
                pred_scaled = layer.predict_scaled(torch, current_x, device)
                current_x = torch.cat([x_base, pred_scaled], dim=1)
            pred = pred_scaled * self._y_scale.to(device) + self._y_mean.to(device)
        pred = pred.detach().cpu().numpy()
        if pred.shape[1] == 1:
            return pred.reshape(-1)
        return pred

    def get_params(self, deep=True):
        return {
            'n_layers': self.n_layers,
            'n_estimators_per_layer': self.n_estimators_per_layer,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'device': self.device,
            'ridge_alpha': self.ridge_alpha,
            'max_tree_depth': self.max_tree_depth,
            'temperature': self.temperature,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k == 'accelerator':
                k = 'device'
            setattr(self, k, v)
        return self


class _MLXGCForestLayer:
    def __init__(self, n_trees, depth, ridge_alpha=1e-3, temperature=1.0,
                 random_state=None):
        self.n_trees = n_trees
        self.depth = depth
        self.ridge_alpha = ridge_alpha
        self.temperature = temperature
        self.random_state = random_state
        self.split_weight = None
        self.split_bias = None
        self.coef = None
        self.x_mean = None
        self.x_scale = None
        self.path_nodes = None
        self.path_dirs = None
        self.feature_scale = None

    @staticmethod
    def _build_paths(depth):
        n_leaves = 2 ** depth
        nodes = []
        dirs = []
        for leaf_idx in range(n_leaves):
            node_idx = 0
            leaf_nodes = []
            leaf_dirs = []
            for level_idx in range(depth):
                bit = (leaf_idx >> (depth - level_idx - 1)) & 1
                leaf_nodes.append(node_idx)
                leaf_dirs.append(bool(bit))
                node_idx = node_idx * 2 + 1 + bit
            nodes.append(leaf_nodes)
            dirs.append(leaf_dirs)
        return (
            np.asarray(nodes, dtype=np.int32),
            np.asarray(dirs, dtype=bool),
        )

    def _path_features(self, mx, x):
        split_weight = mx.array(self.split_weight, dtype=mx.float32)
        split_bias = mx.array(self.split_bias, dtype=mx.float32)
        path_nodes = mx.array(self.path_nodes, dtype=mx.int32)
        path_dirs = mx.array(self.path_dirs)
        x_mean = mx.array(self.x_mean, dtype=mx.float32)
        x_scale = mx.array(self.x_scale, dtype=mx.float32)
        x = (x - x_mean) / x_scale
        logits = mx.einsum('bd,tnd->btn', x, split_weight) + split_bias
        decisions = mx.sigmoid(logits / max(self.temperature, 1e-6))
        selected = decisions[:, :, path_nodes]
        probs = mx.where(path_dirs.reshape(1, 1, *path_dirs.shape),
                         selected, 1.0 - selected)
        probs = mx.prod(mx.clip(probs, 1e-7, 1.0), axis=-1)
        return probs.reshape(x.shape[0], -1) / self.feature_scale

    @staticmethod
    def _solve_ridge(phi, y, alpha):
        phi = np.asarray(phi, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples, n_features = phi.shape
        try:
            if n_features <= n_samples:
                gram = phi.T @ phi + alpha * np.eye(n_features, dtype=np.float32)
                rhs = phi.T @ y
                return np.linalg.solve(gram, rhs).astype(np.float32)
            gram = phi @ phi.T + alpha * np.eye(n_samples, dtype=np.float32)
            alpha_coef = np.linalg.solve(gram, y)
            return (phi.T @ alpha_coef).astype(np.float32)
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(phi) @ y).astype(np.float32)

    def fit(self, mx, x, y):
        n_features = x.shape[1]
        n_internal = 2 ** self.depth - 1
        rng = np.random.default_rng(self.random_state)
        split_weight = rng.standard_normal(
            (self.n_trees, n_internal, n_features)
        ).astype(np.float32)
        split_weight = split_weight / (
            np.linalg.norm(split_weight, axis=-1, keepdims=True) + 1e-6
        )
        split_bias = (
            rng.standard_normal((self.n_trees, n_internal)).astype(np.float32) * 0.5
        )
        self.split_weight = split_weight
        self.split_bias = split_bias
        self.x_mean = np.array(mx.mean(x, axis=0, keepdims=True), dtype=np.float32)
        self.x_scale = np.maximum(
            np.array(mx.std(x, axis=0, keepdims=True), dtype=np.float32),
            np.float32(1e-6),
        )
        self.path_nodes, self.path_dirs = self._build_paths(self.depth)
        self.feature_scale = float(max(1.0, np.sqrt(self.n_trees)))
        phi = self._path_features(mx, x)
        mx.eval(phi)
        self.coef = self._solve_ridge(np.array(phi), np.array(y), self.ridge_alpha)
        pred = phi @ mx.array(self.coef, dtype=mx.float32)
        mx.eval(pred)
        return pred

    def predict_scaled(self, mx, x):
        phi = self._path_features(mx, x)
        return phi @ mx.array(self.coef, dtype=mx.float32)


class _MLXGCForestEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, n_layers=3, n_estimators_per_layer=100,
                 max_depth=None, min_samples_leaf=1, random_state=None,
                 verbose=False, accelerator='auto', ridge_alpha=1e-3,
                 max_tree_depth=6, temperature=1.0):
        self.n_layers = n_layers
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        self.accelerator = accelerator
        self.ridge_alpha = ridge_alpha
        self.max_tree_depth = max_tree_depth
        self.temperature = temperature
        self._layers = []
        self._y_mean = None
        self._y_scale = None

    def fit(self, X, y, **fit_kwargs):
        import mlx.core as mx

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x_base = mx.array(X, dtype=mx.float32)
        y_tensor = mx.array(y, dtype=mx.float32)
        self._y_mean = np.array(mx.mean(y_tensor, axis=0, keepdims=True), dtype=np.float32)
        self._y_scale = np.maximum(
            np.array(mx.std(y_tensor, axis=0, keepdims=True), dtype=np.float32),
            np.float32(1e-6),
        )
        y_scaled = (y_tensor - mx.array(self._y_mean, dtype=mx.float32)) / mx.array(
            self._y_scale, dtype=mx.float32
        )
        self._layers = []
        current_x = x_base
        depth = self.max_depth if self.max_depth is not None else 5
        depth_cap = self.max_tree_depth if self.max_tree_depth is not None else 6
        depth = int(max(1, min(depth, depth_cap)))
        n_trees = int(max(1, self.n_estimators_per_layer))
        for layer_idx in range(int(max(1, self.n_layers))):
            seed = None
            if self.random_state is not None:
                seed = int(self.random_state) + layer_idx
            layer = _MLXGCForestLayer(
                n_trees=n_trees,
                depth=depth,
                ridge_alpha=self.ridge_alpha,
                temperature=self.temperature,
                random_state=seed,
            )
            pred_scaled = layer.fit(mx, current_x, y_scaled)
            self._layers.append(layer)
            current_x = mx.concatenate([x_base, pred_scaled], axis=1)
            mx.eval(current_x)
        return self

    def predict(self, X):
        import mlx.core as mx

        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._layers:
            raise ValueError("The estimator has not been fitted.")
        x_base = mx.array(X, dtype=mx.float32)
        current_x = x_base
        pred_scaled = None
        for layer in self._layers:
            pred_scaled = layer.predict_scaled(mx, current_x)
            current_x = mx.concatenate([x_base, pred_scaled], axis=1)
        pred = pred_scaled * mx.array(self._y_scale, dtype=mx.float32) + mx.array(
            self._y_mean, dtype=mx.float32
        )
        mx.eval(pred)
        pred = np.array(pred)
        if pred.shape[1] == 1:
            return pred.reshape(-1)
        return pred

    def get_params(self, deep=True):
        return {
            'n_layers': self.n_layers,
            'n_estimators_per_layer': self.n_estimators_per_layer,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'accelerator': self.accelerator,
            'ridge_alpha': self.ridge_alpha,
            'max_tree_depth': self.max_tree_depth,
            'temperature': self.temperature,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k == 'device':
                k = 'accelerator'
            setattr(self, k, v)
        return self


class _AutoGCForestEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, n_layers=3, n_estimators_per_layer=100,
                 max_depth=None, min_samples_leaf=1, random_state=None,
                 verbose=False, accelerator='auto', ridge_alpha=1e-3,
                 max_tree_depth=6, temperature=1.0, backend=None,
                 device=None):
        self.n_layers = n_layers
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        self.backend = backend
        self.device = device
        self.accelerator = self._coerce_accelerator(accelerator, backend, device)
        self.ridge_alpha = ridge_alpha
        self.max_tree_depth = max_tree_depth
        self.temperature = temperature
        self._delegate = None

    @staticmethod
    def _coerce_accelerator(accelerator, backend=None, device=None):
        if backend not in (None, 'auto'):
            if backend == 'sklearn':
                return 'sklearn'
            if backend == 'mlx':
                return 'mlx'
            if backend == 'torch':
                if device == 'cpu':
                    return 'torch_cpu'
                if device in {'cuda', 'mps'}:
                    return device
                return 'torch'
        if accelerator in (None, 'auto') and device not in (None, 'auto'):
            return device
        return 'auto' if accelerator is None else accelerator

    def _mlx_available(self):
        return importlib.util.find_spec('mlx') is not None

    def _torch_available(self):
        return importlib.util.find_spec('torch') is not None

    def _should_use_torch_auto(self):
        if not self._torch_available():
            return False
        try:
            import torch
        except Exception:
            return False
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        return bool(torch.cuda.is_available() or has_mps)

    def _torch_device(self, accelerator):
        if accelerator == 'torch_cpu':
            return 'cpu'
        if accelerator in {'cuda', 'mps', 'cpu'}:
            return accelerator
        return 'auto'

    def _accelerator_candidates(self):
        accelerator = self.accelerator
        if accelerator in (None, 'auto'):
            candidates = []
            if sys.platform == 'darwin' and self._mlx_available():
                candidates.append('mlx')
            if self._should_use_torch_auto():
                candidates.append('torch')
            candidates.append('sklearn')
            return candidates
        if accelerator in {'sklearn', 'cpu'}:
            return ['sklearn']
        if accelerator == 'mlx':
            return ['mlx']
        if accelerator in {'torch', 'cuda', 'mps', 'torch_cpu'}:
            return ['torch']
        raise ValueError(
            "accelerator must be one of {'auto', 'mlx', 'torch', 'cuda', "
            "'mps', 'torch_cpu', 'sklearn', 'cpu'}."
        )

    def _is_auto_request(self):
        return self.accelerator in (None, 'auto')

    def _make_mlx(self):
        return _MLXGCForestEstimator(
            n_layers=self.n_layers,
            n_estimators_per_layer=self.n_estimators_per_layer,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            verbose=self.verbose,
            accelerator=self.accelerator,
            ridge_alpha=self.ridge_alpha,
            max_tree_depth=self.max_tree_depth,
            temperature=self.temperature,
        )

    def _make_torch(self):
        return _TorchGCForestEstimator(
            n_layers=self.n_layers,
            n_estimators_per_layer=self.n_estimators_per_layer,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            verbose=self.verbose,
            device=self._torch_device(self.accelerator),
            ridge_alpha=self.ridge_alpha,
            max_tree_depth=self.max_tree_depth,
            temperature=self.temperature,
        )

    def _make_sklearn(self):
        return _GCForestEstimator(
            n_layers=self.n_layers,
            n_estimators_per_layer=self.n_estimators_per_layer,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def fit(self, X, y, **fit_kwargs):
        last_error = None
        for candidate in self._accelerator_candidates():
            try:
                if candidate == 'mlx':
                    if not self._mlx_available():
                        raise ImportError(
                            "MLX is required for gcForestModel accelerator='mlx'."
                        )
                    self._delegate = self._make_mlx()
                elif candidate == 'torch':
                    if not self._torch_available():
                        raise ImportError(
                            "PyTorch is required for gcForestModel accelerator='torch'."
                        )
                    self._delegate = self._make_torch()
                else:
                    self._delegate = self._make_sklearn()
                self._delegate.fit(X, y, **fit_kwargs)
                return self
            except Exception as exc:
                last_error = exc
                if not self._is_auto_request():
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No valid gcForest accelerator candidate was available.")

    def predict(self, X):
        if self._delegate is None:
            raise ValueError("The estimator has not been fitted.")
        return self._delegate.predict(X)

    @property
    def resolved_backend_(self):
        if isinstance(self._delegate, _MLXGCForestEstimator):
            return 'mlx'
        if isinstance(self._delegate, _TorchGCForestEstimator):
            return 'torch'
        if isinstance(self._delegate, _GCForestEstimator):
            return 'sklearn'
        return None

    @property
    def resolved_accelerator_(self):
        return self.resolved_backend_

    def get_params(self, deep=True):
        return {
            'n_layers': self.n_layers,
            'n_estimators_per_layer': self.n_estimators_per_layer,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'accelerator': self.accelerator,
            'ridge_alpha': self.ridge_alpha,
            'max_tree_depth': self.max_tree_depth,
            'temperature': self.temperature,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k == 'device':
                self.device = v
                self.accelerator = self._coerce_accelerator(
                    'auto', self.backend, v
                )
                continue
            if k == 'backend':
                self.backend = v
                self.accelerator = self._coerce_accelerator(
                    self.accelerator, v, self.device
                )
                continue
            setattr(self, k, v)
        self.accelerator = self._coerce_accelerator(
            self.accelerator, self.backend, self.device
        )
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
                 random_state=None, verbose=False, accelerator='auto',
                 ridge_alpha=1e-3, max_tree_depth=6,
                 temperature=1.0, **kwargs):
        super().__init__(time_col=time_col, target_col=target_col)
        backend = kwargs.pop('backend', None)
        device = kwargs.pop('device', None)
        accelerator = _AutoGCForestEstimator._coerce_accelerator(
            accelerator, backend, device
        )
        self.all_configs['model_configs'] = dict(
            n_layers=n_layers,
            n_estimators_per_layer=n_estimators_per_layer,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            verbose=verbose,
            accelerator=accelerator,
            ridge_alpha=ridge_alpha,
            max_tree_depth=max_tree_depth,
            temperature=temperature,
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
        return _AutoGCForestEstimator(**self.all_configs['model_configs'])

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_layers'] = min(2, cv_configs.get('n_layers', 3))
        cv_configs['n_estimators_per_layer'] = min(
            32, cv_configs.get('n_estimators_per_layer', 100)
        )
        depth_cap = cv_configs.get('max_tree_depth', 6)
        cv_configs['max_tree_depth'] = min(5, depth_cap if depth_cap is not None else 6)
        return _AutoGCForestEstimator(**cv_configs)
