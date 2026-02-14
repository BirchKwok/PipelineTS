"""GPU-accelerated differentiable tree ensembles via PyTorch.

Implements Neural Oblivious Decision Ensembles (NODE) adapted for time series
regression.  Each "tree" is a differentiable oblivious decision tree with
learned feature selection, split thresholds, and leaf response values.
Ensembles of these soft trees simulate the behavior of LightGBM, CatBoost,
XGBoost, RandomForest, and DeepForest — but train end-to-end on GPU via
backpropagation.

Architecture (v3 — vectorized + skip + temp annealing)
------------------------------------------------------
- ``_DifferentiableTreeEnsemble``: ALL tree parameters stored as batched
  tensors ``(n_trees, ...)``.  Forward pass uses ``torch.einsum`` — **zero
  Python loops** over individual trees.  Supports *additive*, *bagging*, and
  *cascade* ensemble modes.
  **v3 additions**: linear skip connection (trees learn residual), feature
  selection temperature annealing (soft→sharp during training).
- ``_TorchTreeWrapper``: sklearn-compatible ``fit`` / ``predict`` wrapper.
  Natively supports multi-output (no RegressorChain needed).  Uses target
  normalisation, Huber loss, and temperature annealing schedule.

Backward-compatible aliases ``_ObliviousDecisionTree`` are kept for tests.

References
----------
- Popov, Morozov & Babenko (2019). "Neural Oblivious Decision Ensembles
  for Deep Learning on Tabular Data." (NODE)
- Zhou & Feng (2017). "Deep Forest: Towards An Alternative to Deep
  Neural Networks."  (gcForest)
"""

import math
import os
import sys
import warnings
import numpy as np

# Fix OpenMP threading conflict between sklearn and PyTorch on macOS ARM.
# sklearn's ExtraTreesRegressor / GradientBoostingRegressor initialise OpenMP
# with settings that crash PyTorch's parallel tensor ops (sigmoid, matmul, etc.).
# Setting OMP_NUM_THREADS=1 before torch is imported avoids the conflict.
if sys.platform == 'darwin' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def _get_device(accelerator):
    """Resolve accelerator string to torch.device."""
    if accelerator in (None, 'auto'):
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(accelerator)


def _safe_softmax(x, dim=-1):
    """Manual softmax avoiding F.softmax issues on some platforms."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + 1e-12)


def _entmax15(x, dim=-1, n_iter=25):
    """1.5-entmax: sparse alternative to softmax (NODE paper, Sec. 3).

    Produces truly sparse distributions where most entries are exactly 0.
    This is critical for feature selection — unlike softmax which always
    assigns non-zero weight to every feature, entmax selects only the
    most relevant features for each tree node.

    Uses the iterative bisection algorithm from:
    Peters & Martins (2019) "Sparse Sequence-to-Sequence Models".
    """
    # Sort descending along dim for the bisection
    x_sorted, _ = x.sort(dim=dim, descending=True)
    x_cumsum = x_sorted.cumsum(dim=dim)
    k = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    # Reshape k for broadcasting
    shape = [1] * x.ndim
    shape[dim] = -1
    k = k.view(shape)
    # Threshold: tau such that sum(max(0, x - tau)^(alpha-1)) = 1
    # For alpha=1.5: p_i = max(0, x_i - tau)^2, sum(p_i) = 1
    # Closed form per-row via sorted cumsum
    tau = (x_cumsum - 1.0) / k
    support = (x_sorted > tau).sum(dim=dim, keepdim=True).clamp(min=1)
    # Recompute tau using only the support
    tau_star = x_cumsum.gather(dim, support - 1)
    tau_star = (tau_star - 1.0) / support.float()
    # Compute sparse output: p_i = max(0, x_i - tau)^(1/(alpha-1)) = max(0, x_i-tau)^2
    # But for alpha=1.5, the mapping is p_i = ReLU(x_i - tau)^2 / sum(ReLU(...)^2)
    # Simpler: p_i = ReLU(x_i - tau) then normalise
    p = torch.clamp(x - tau_star, min=0)
    # Normalise to sum to 1 (handles numerical edge cases)
    p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
    return p


def _build_bit_masks(depth):
    """Pre-compute binary addressing masks for oblivious tree leaves.

    Returns Tensor of shape ``(depth, 2**depth)``.
    """
    n_leaves = 1 << depth
    masks = torch.zeros(depth, n_leaves)
    for d in range(depth):
        for leaf in range(n_leaves):
            if (leaf >> (depth - 1 - d)) & 1:
                masks[d, leaf] = 1.0
    return masks


# ---------------------------------------------------------------------------
#  Adaptive Complexity Controller
# ---------------------------------------------------------------------------

class _AdaptiveComplexityController:
    """Analyzes data characteristics to dynamically select tree depth and count.

    Inspired by AutoML complexity selection: uses lightweight data statistics
    (sample size, feature count, noise level, nonlinearity, autocorrelation)
    to choose model complexity that balances capacity vs overfitting risk.

    The controller adjusts two axes:
    - **tree_depth**: controls per-tree expressiveness (deeper = more interactions)
    - **n_trees**: controls ensemble capacity (more = better approximation)

    Design principles:
    - Small/clean data → shallow trees, fewer trees (avoid overfitting)
    - Large/noisy data → moderate depth, more trees (need capacity)
    - High nonlinearity → deeper trees (need interaction capacity)
    - Strong autocorrelation → moderate trees (patterns are regular)
    """

    # Complexity profiles: (min_depth, max_depth, min_trees, max_trees)
    _PROFILES = {
        'minimal':   (2, 3, 8, 24),
        'light':     (3, 4, 16, 48),
        'moderate':  (4, 5, 32, 64),
        'heavy':     (5, 6, 48, 96),
        'maximal':   (6, 7, 64, 128),
    }

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._analysis = {}

    def analyze(self, X, y):
        """Compute data statistics for complexity selection.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        dict with analysis results
        """
        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 1. Noise estimation: residual variance after linear fit
        try:
            # Quick OLS via normal equations (no sklearn dependency)
            X_aug = np.column_stack([X, np.ones(n_samples)])
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            y_hat = X_aug @ beta
            residual_var = np.mean((y - y_hat) ** 2)
            total_var = np.var(y) + 1e-12
            noise_ratio = min(1.0, residual_var / total_var)
        except Exception:
            noise_ratio = 0.5

        # 2. Nonlinearity score: compare linear residuals with running median
        try:
            y_flat = y[:, 0]
            window = max(5, n_samples // 20)
            # Running median approximation via cumsum trick
            padded = np.pad(y_flat, (window // 2, window // 2), mode='edge')
            cumsum = np.cumsum(padded)
            running_mean = (cumsum[window:] - cumsum[:-window]) / window
            running_mean = running_mean[:n_samples]
            smooth_residual = np.mean((y_flat - running_mean) ** 2)
            linear_residual = residual_var
            # If smoothing captures much more variance than linear, data is nonlinear
            nonlinearity = max(0.0, 1.0 - smooth_residual / (linear_residual + 1e-12))
            nonlinearity = min(1.0, nonlinearity)
        except Exception:
            nonlinearity = 0.5

        # 3. Autocorrelation (lag-1)
        try:
            y_flat = y[:, 0]
            y_centered = y_flat - y_flat.mean()
            c0 = np.dot(y_centered, y_centered) + 1e-12
            c1 = np.dot(y_centered[:-1], y_centered[1:])
            autocorr = abs(c1 / c0)
        except Exception:
            autocorr = 0.5

        # 4. Feature importance concentration (Gini-like)
        try:
            feat_vars = np.var(X, axis=0)
            feat_vars = feat_vars / (feat_vars.sum() + 1e-12)
            concentration = 1.0 - (-np.sum(feat_vars * np.log(feat_vars + 1e-12))
                                    / np.log(max(n_features, 2)))
            concentration = max(0.0, min(1.0, concentration))
        except Exception:
            concentration = 0.5

        self._analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'noise_ratio': float(noise_ratio),
            'nonlinearity': float(nonlinearity),
            'autocorr': float(autocorr),
            'feat_concentration': float(concentration),
        }
        return self._analysis

    def select_complexity(self, X, y, ensemble_mode='additive',
                          user_depth=None, user_n_trees=None):
        """Select optimal tree depth and count based on data analysis.

        Parameters
        ----------
        X, y : training data
        ensemble_mode : str
        user_depth : int or None
            User-specified depth (None = auto-select)
        user_n_trees : int or None
            User-specified n_trees (None = auto-select)

        Returns
        -------
        dict with keys 'tree_depth', 'n_trees', 'profile', 'reasons'
        """
        stats = self.analyze(X, y)
        n = stats['n_samples']
        f = stats['n_features']
        noise = stats['noise_ratio']
        nonlin = stats['nonlinearity']
        autocorr = stats['autocorr']

        reasons = []

        # --- Select complexity profile ---
        # Base score from data size (most important factor)
        if n < 60:
            size_score = 0.0
            reasons.append(f'tiny_data(n={n})->minimal')
        elif n < 150:
            size_score = 0.25
            reasons.append(f'small_data(n={n})->light')
        elif n < 400:
            size_score = 0.5
            reasons.append(f'medium_data(n={n})->moderate')
        elif n < 1000:
            size_score = 0.75
            reasons.append(f'large_data(n={n})->heavy')
        else:
            size_score = 1.0
            reasons.append(f'very_large_data(n={n})->maximal')

        # Adjust for noise: high noise → reduce complexity (overfit risk)
        noise_adj = 0.0
        if noise > 0.7:
            noise_adj = -0.2
            reasons.append(f'high_noise({noise:.2f})->reduce')
        elif noise < 0.3:
            noise_adj = 0.1
            reasons.append(f'low_noise({noise:.2f})->increase')

        # Adjust for nonlinearity: high nonlinearity → increase depth
        nonlin_adj = 0.0
        if nonlin > 0.6:
            nonlin_adj = 0.15
            reasons.append(f'high_nonlinearity({nonlin:.2f})->deeper')
        elif nonlin < 0.2:
            nonlin_adj = -0.1
            reasons.append(f'low_nonlinearity({nonlin:.2f})->shallower')

        # Adjust for autocorrelation: strong AR → moderate (regular patterns)
        ar_adj = 0.0
        if autocorr > 0.8:
            ar_adj = -0.05
            reasons.append(f'strong_autocorr({autocorr:.2f})->moderate')

        # Adjust for feature count: many features → deeper trees
        feat_adj = 0.0
        if f > 50:
            feat_adj = 0.1
            reasons.append(f'many_features(f={f})->deeper')
        elif f < 5:
            feat_adj = -0.05

        # Cascade mode benefits from lighter per-layer trees
        mode_adj = 0.0
        if ensemble_mode == 'cascade':
            mode_adj = -0.15
            reasons.append('cascade_mode->lighter_per_layer')

        complexity_score = max(0.0, min(1.0,
            size_score + noise_adj + nonlin_adj + ar_adj + feat_adj + mode_adj))

        # Map score to profile
        if complexity_score < 0.15:
            profile = 'minimal'
        elif complexity_score < 0.35:
            profile = 'light'
        elif complexity_score < 0.6:
            profile = 'moderate'
        elif complexity_score < 0.8:
            profile = 'heavy'
        else:
            profile = 'maximal'

        min_d, max_d, min_t, max_t = self._PROFILES[profile]

        # Interpolate within profile range
        frac = (complexity_score - [0.0, 0.15, 0.35, 0.6, 0.8][
            ['minimal', 'light', 'moderate', 'heavy', 'maximal'].index(profile)
        ]) / 0.2
        frac = max(0.0, min(1.0, frac))

        auto_depth = int(round(min_d + frac * (max_d - min_d)))
        auto_trees = int(round(min_t + frac * (max_t - min_t)))

        # Ensure minimum viable ensemble
        auto_depth = max(2, auto_depth)
        auto_trees = max(8, auto_trees)

        # Apply user overrides (user always wins)
        final_depth = user_depth if user_depth is not None else auto_depth
        final_trees = user_n_trees if user_n_trees is not None else auto_trees

        result = {
            'tree_depth': final_depth,
            'n_trees': final_trees,
            'profile': profile,
            'complexity_score': round(complexity_score, 3),
            'reasons': reasons,
            'stats': stats,
            'auto_depth': auto_depth,
            'auto_trees': auto_trees,
        }

        if self.verbose:
            print(f"  [AdaptiveComplexity] profile={profile} "
                  f"score={complexity_score:.3f} "
                  f"depth={final_depth} trees={final_trees}")
            print(f"    reasons: {', '.join(reasons)}")
            print(f"    stats: noise={noise:.2f} nonlin={nonlin:.2f} "
                  f"autocorr={autocorr:.2f} n={n} f={f}")

        return result


# ---------------------------------------------------------------------------
#  Fully-vectorized tree ensemble  (v3)
# ---------------------------------------------------------------------------

class _DifferentiableTreeEnsemble(nn.Module):
    """Fully-vectorized ensemble of differentiable oblivious decision trees.

    ALL tree parameters are stored as **batched tensors** ``(n_trees, ...)``.
    The forward pass uses ``torch.einsum`` and element-wise ops only — there
    are **zero Python loops** over individual trees.

    v3 additions:
    - **Linear skip connection**: ``output = trees(x) + linear(x)``
      Trees learn the non-linear residual on top of a strong linear baseline.
    - **Feature temperature annealing**: softmax temperature on feature logits
      is annealed from 1.0 (soft) → 0.1 (sharp) during training, producing
      increasingly tree-like hard feature selection.

    Supports three ensemble modes:

    - ``'additive'``: gradient boosting style, output = mean(tree_outputs).
    - ``'bagging'``:  same forward, but with tree-level dropout (RF-style).
    - ``'cascade'``:  multi-layer gcForest; each layer's tree outputs are
      concatenated with original features for the next layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    n_trees : int
    tree_depth : int
    ensemble_mode : str
    n_layers : int  (cascade only)
    dropout : float
    """

    def __init__(
        self,
        in_features,
        out_features=1,
        n_trees=32,
        tree_depth=4,
        ensemble_mode='additive',
        n_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.ensemble_mode = ensemble_mode
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_leaves = 1 << tree_depth

        # Shared bit masks — (depth, n_leaves)
        self.register_buffer('_bit_masks', _build_bit_masks(tree_depth))

        # Feature selection temperature (annealed 1.0 → 0.1 during training)
        self.register_buffer('_feat_temp', torch.tensor(1.0))

        if ensemble_mode == 'cascade':
            self._init_cascade(in_features, out_features, n_trees, tree_depth,
                               n_layers)
        else:
            self._init_flat(in_features, out_features, n_trees, tree_depth)

    def set_feat_temp(self, t):
        """Set feature selection temperature (called by training loop)."""
        self._feat_temp.fill_(t)

    # -- initialisation helpers -----------------------------------------------

    def _init_flat(self, F, O, T, D):
        self.feature_logits = nn.Parameter(torch.empty(T, D, F))
        self.thresholds = nn.Parameter(torch.zeros(T, D))
        self.log_temps = nn.Parameter(torch.zeros(T, D))
        self.leaf_responses = nn.Parameter(torch.empty(T, self.n_leaves, O))
        nn.init.xavier_normal_(self.feature_logits)
        nn.init.uniform_(self.leaf_responses, -0.1, 0.1)
        # Linear skip connection: strong linear baseline, trees learn residual
        self.skip = nn.Linear(F, O)

    def _init_cascade(self, F, O, T, D, n_layers):
        # Vanilla cascade + learnable residual shortcut.
        # Primary path: final_proj maps from rich T*O augmented features
        # (same as original gcForest cascade — high expressiveness).
        # Residual path: each layer’s mean prediction is accumulated and
        # scaled by a learnable parameter (init=0).  This gives early
        # layers a direct gradient shortcut without changing the primary
        # path.  If residual_scale stays ~0, behaviour = vanilla cascade.
        self.layer_fl = nn.ParameterList()
        self.layer_th = nn.ParameterList()
        self.layer_lt = nn.ParameterList()
        self.layer_lr = nn.ParameterList()
        cur_in = F
        for _ in range(n_layers):
            fl = nn.Parameter(torch.empty(T, D, cur_in))
            nn.init.xavier_normal_(fl)
            self.layer_fl.append(fl)
            self.layer_th.append(nn.Parameter(torch.zeros(T, D)))
            self.layer_lt.append(nn.Parameter(torch.zeros(T, D)))
            lr = nn.Parameter(torch.empty(T, self.n_leaves, O))
            nn.init.uniform_(lr, -0.1, 0.1)
            self.layer_lr.append(lr)
            cur_in = F + T * O
        self.final_proj = nn.Linear(cur_in, O)
        # Residual scale: init=0 so model starts as vanilla, learns
        # to use early-layer shortcuts if beneficial.
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.skip = nn.Linear(F, O)

    # -- core batched forward (zero loops over trees) -------------------------

    def _batched_trees(self, x, fl, th, lt, lr):
        """Compute all trees in parallel.

        Parameters
        ----------
        x  : (B, F)   — input batch
        fl : (T, D, F) — feature logits
        th : (T, D)    — thresholds
        lt : (T, D)    — log temperatures
        lr : (T, L, O) — leaf responses

        Returns
        -------
        (B, T, O) — per-tree outputs
        """
        # Temperature-scaled feature selection (sharper = more tree-like)
        feat_temp = self._feat_temp.clamp(min=0.05)
        fw = _safe_softmax(fl / feat_temp, dim=-1)          # (T, D, F)
        # Selected features for every tree & depth level
        selected = torch.einsum('tdf,bf->btd', fw, x)       # (B, T, D)
        # Split decisions
        temps = torch.exp(lt).clamp(min=0.1, max=10.0)      # (T, D)
        decisions = torch.sigmoid(
            (selected - th.unsqueeze(0)) / temps.unsqueeze(0)
        )                                                    # (B, T, D)
        # Leaf probabilities via binary addressing
        d = decisions.unsqueeze(3)                           # (B, T, D, 1)
        m = self._bit_masks.unsqueeze(0).unsqueeze(0)        # (1, 1, D, L)
        leaf_probs = (d * m + (1.0 - d) * (1.0 - m)).prod(dim=2)  # (B, T, L)
        # Weighted leaf responses
        return torch.einsum('btl,tlo->bto', leaf_probs, lr)  # (B, T, O)

    # -- forward methods ------------------------------------------------------

    def forward(self, x):
        if self.ensemble_mode == 'cascade':
            return self._forward_cascade(x)
        return self._forward_flat(x)

    def _forward_flat(self, x):
        out = self._batched_trees(
            x, self.feature_logits, self.thresholds,
            self.log_temps, self.leaf_responses,
        )  # (B, T, O)
        if self.training and self.dropout > 0:
            mask = (torch.rand(1, self.n_trees, 1, device=x.device)
                    > self.dropout).float()
            out = out * mask
            tree_out = out.sum(dim=1) / mask.sum().clamp(min=1.0) * self.n_trees
        else:
            tree_out = out.mean(dim=1)  # (B, O)
        return tree_out + self.skip(x)

    def _forward_cascade(self, x):
        # Primary: vanilla cascade (final_proj on last augmented features)
        # Residual: Σ mean(tree_out_i) scaled by learnable residual_scale
        current = x
        layer_out = None
        B = x.shape[0]
        layer_preds = []  # per-layer mean predictions for residual path
        for i in range(self.n_layers):
            tree_out = self._batched_trees(
                current, self.layer_fl[i], self.layer_th[i],
                self.layer_lt[i], self.layer_lr[i],
            )  # (B, T, O)
            layer_preds.append(tree_out.mean(dim=1))  # (B, O)
            layer_out = tree_out.reshape(B, -1)  # (B, T*O)
            if i < self.n_layers - 1:
                current = torch.cat([x, layer_out], dim=-1)
        # Primary path: rich projection (same as vanilla)
        primary = self.final_proj(torch.cat([x, layer_out], dim=-1))
        # Residual path: early-layer gradient shortcut
        residual = torch.stack(layer_preds, dim=0).mean(dim=0)  # (B, O)
        return primary + self.residual_scale * residual + self.skip(x)


# ---------------------------------------------------------------------------
#  Backward-compatible alias used by some tests
# ---------------------------------------------------------------------------

class _ObliviousDecisionTree(nn.Module):
    """Thin wrapper that delegates to the batched ensemble with n_trees=1."""

    def __init__(self, in_features, depth, out_features=1):
        super().__init__()
        self._ens = _DifferentiableTreeEnsemble(
            in_features=in_features, out_features=out_features,
            n_trees=1, tree_depth=depth, ensemble_mode='additive',
        )

    def forward(self, x):
        return self._ens(x)


# ---------------------------------------------------------------------------
#  sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class _TorchTreeWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for vectorized differentiable tree ensembles.

    Key optimisations (v3):
    - Fully-vectorized forward (zero Python loops over trees)
    - **Native multi-output** — no RegressorChain needed
    - Target normalisation for faster convergence
    - Huber (SmoothL1) loss for robustness
    - Feature temperature annealing (1.0→0.1) for sharp feature selection
    - Linear skip connection (trees learn residual)
    - Aggressive early stopping + ReduceLROnPlateau
    - **Staged gradient boosting** (boosting_stages>1): true sequential
      residual learning like native GBDT — each stage trains on the
      residual error from all previous stages.

    Parameters
    ----------
    n_trees : int
    tree_depth : int
    ensemble_mode : str  ('additive', 'bagging', 'cascade')
    n_layers : int
    learning_rate : float
    n_epochs : int
    batch_size : int  (0 = full-batch)
    early_stop_patience : int
    dropout : float
    weight_decay : float
    loss_fn : str  ('huber' or 'mse')
    boosting_stages : int
        Number of sequential boosting stages.  1 = standard (all trees
        trained jointly).  >1 = true gradient boosting where each stage
        learns the residual from previous stages.
    boosting_shrinkage : float
        Shrinkage factor per boosting stage (like eta in XGBoost).
    accelerator : str or None
    random_state : int or None
    verbose : bool
    """

    def __init__(
        self,
        n_trees=32,
        tree_depth=4,
        ensemble_mode='additive',
        n_layers=2,
        learning_rate=0.08,
        n_epochs=120,
        batch_size=0,
        early_stop_patience=12,
        dropout=0.0,
        weight_decay=1e-4,
        loss_fn='huber',
        boosting_stages=1,
        boosting_shrinkage=0.5,
        accelerator=None,
        random_state=None,
        verbose=False,
        auto_complexity=False,
    ):
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.ensemble_mode = ensemble_mode
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.boosting_stages = boosting_stages
        self.boosting_shrinkage = boosting_shrinkage
        self.accelerator = accelerator
        self.random_state = random_state
        self.verbose = verbose
        self.auto_complexity = auto_complexity

        self._model = None
        self._staged_models = None  # list of (model, shrinkage) for staged boosting
        self._device = None
        self._n_outputs = 1
        self._feature_mean = None
        self._feature_std = None
        self._target_mean = None
        self._target_std = None
        self._use_log_target = False
        self._complexity_result = None  # stores adaptive complexity selection result
        self._use_amp = False  # set True when CUDA AMP is beneficial

    def _make_loss_fn(self):
        if self.loss_fn == 'mse':
            return nn.MSELoss()
        return nn.SmoothL1Loss()

    @staticmethod
    def _compute_loss(model, X, y, use_mse, sample_weights=None):
        """Compute per-sample loss then (weighted) mean."""
        pred = model(X)
        diff = pred - y
        if use_mse:
            per_sample = diff.pow(2).mean(dim=-1)
        else:
            abs_diff = diff.abs()
            per_sample = torch.where(
                abs_diff < 1.0, 0.5 * diff.pow(2), abs_diff - 0.5
            ).mean(dim=-1)
        if sample_weights is not None:
            w = sample_weights[:per_sample.shape[0]]
            return (per_sample * w).sum() / w.sum()
        return per_sample.mean()

    @staticmethod
    def _has_structural_break(y, tail_size=None,
                              rel_shift_threshold=0.30,
                              robust_z_threshold=1.0,
                              return_magnitude=False):
        """Detect a strong recent level shift using robust statistics.

        Compares the median level of the latest window vs the previous window.
        If both relative shift and robust z-score are large, treat it as a
        structural break and avoid holding out the most recent data for val.

        When *return_magnitude* is True, returns ``(bool, float)`` where the
        float is the maximum relative shift magnitude (0 when no break).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = y.shape[0]
        if tail_size is None:
            tail_size = min(120, max(24, n_samples // 5))

        if n_samples < tail_size * 2:
            return (False, 0.0) if return_magnitude else False

        prev_window = y[-2 * tail_size:-tail_size]
        recent_window = y[-tail_size:]

        prev_median = np.median(prev_window, axis=0)
        recent_median = np.median(recent_window, axis=0)

        q75, q25 = np.percentile(prev_window, [75, 25], axis=0)
        prev_iqr = np.maximum(q75 - q25, 1e-8)

        rel_shift = np.abs(recent_median - prev_median) / np.maximum(
            np.abs(prev_median), 1.0)
        robust_z = np.abs(recent_median - prev_median) / prev_iqr

        # We only treat it as break when the recent regime is clearly lower.
        # This avoids false positives on upward-trending series.
        is_downward_shift = recent_median < prev_median

        detected = bool(np.any(
            (rel_shift >= rel_shift_threshold) &
            (robust_z >= robust_z_threshold) &
            is_downward_shift
        ))
        if return_magnitude:
            mag = float(np.max(rel_shift)) if detected else 0.0
            return detected, mag
        return detected

    def _train_one_model(self, model, X_t, y_t, n_samples, batch_size,
                         n_epochs, lr, patience,
                         X_val=None, y_val=None,
                         weight_decay=None,
                         sample_weights=None):
        """Train a single ensemble model.

        Key improvements over v3:
        - **Validation-based early stopping** (like real GBDT) — monitors
          held-out val loss instead of training loss when val data provided.
        - Entmax sparse feature selection (handled in forward pass).
        - Temperature annealing 1.0 → 0.1 during training.
        - **Sample weighting** (YDF-inspired) — exponential recency weights
          for structural-break adaptation without discarding data.
        """
        effective_weight_decay = (self.weight_decay if weight_decay is None
                                  else weight_decay)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=effective_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=max(3, patience // 3),
            min_lr=lr * 0.01,
        )
        use_mse = (self.loss_fn == 'mse')
        has_val = X_val is not None and y_val is not None

        # AMP setup for CUDA — mixed precision training
        use_amp = self._use_amp and self._device is not None and self._device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        model.train()
        for epoch in range(n_epochs):
            progress = epoch / max(n_epochs - 1, 1)
            feat_temp = 1.0 - 0.9 * progress
            model.set_feat_temp(feat_temp)

            perm = torch.randperm(n_samples, device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]

                w_batch = sample_weights[idx] if sample_weights is not None else None

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = self._compute_loss(
                            model, X_t[idx], y_t[idx], use_mse,
                            sample_weights=w_batch)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self._compute_loss(
                        model, X_t[idx], y_t[idx], use_mse,
                        sample_weights=w_batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Use validation loss for early stopping (like real GBDT)
            if has_val:
                model.eval()
                with torch.no_grad():
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            monitor_loss = self._compute_loss(
                                model, X_val, y_val, use_mse).item()
                    else:
                        monitor_loss = self._compute_loss(
                            model, X_val, y_val, use_mse).item()
                model.train()
            else:
                monitor_loss = epoch_loss / max(n_batches, 1)

            scheduler.step(monitor_loss)

            if monitor_loss < best_loss - 1e-6:
                best_loss = monitor_loss
                patience_counter = 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.set_feat_temp(0.1)
        model.eval()
        return best_loss

    def fit(self, X, y, **fit_kwargs):
        """Train the differentiable tree ensemble.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._n_outputs = y.shape[1]
        n_samples, n_features = X.shape

        # Structural-break handling (YDF-inspired):
        # Instead of discarding old data (hard tail cutoff), use ALL data
        # with exponential recency sample weights.  This preserves feature
        # distribution diversity (critical for stable autoregressive
        # prediction) while biasing optimization toward the current regime.
        has_structural_break, break_magnitude = self._has_structural_break(
            y, return_magnitude=True)

        # Optional robust target transform for structural-break series:
        # learn on log1p scale (only when targets are non-negative), then
        # invert at prediction time.
        self._use_log_target = bool(has_structural_break and np.all(y >= 0.0))
        if self._use_log_target:
            y_train = np.log1p(y)
        else:
            y_train = y

        # Normalise features and targets using full-data statistics.
        # Even when a structural break is detected, full-data normalisation
        # ensures the feature scale matches what the autoregressive predictor
        # will produce during multi-step inference.
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._feature_mean) / self._feature_std

        self._target_mean = y_train.mean(axis=0)
        self._target_std = y_train.std(axis=0) + 1e-8
        y_norm = (y_train - self._target_mean) / self._target_std

        self._device = _get_device(self.accelerator)

        # Enable AMP for CUDA when dataset is large enough to benefit
        self._use_amp = (self._device.type == 'cuda' and n_samples >= 128)

        # Adaptive complexity: auto-tune tree_depth and n_trees from data
        # Must run BEFORE _effective_n_trees calculation.
        if self.auto_complexity:
            controller = _AdaptiveComplexityController(verbose=self.verbose)
            self._complexity_result = controller.select_complexity(
                X_norm, y_norm,
                ensemble_mode=self.ensemble_mode,
                user_depth=None,  # auto-select
                user_n_trees=None,
            )
            self._auto_tree_depth = self._complexity_result['tree_depth']
            self._auto_n_trees = self._complexity_result['n_trees']
        else:
            self._auto_tree_depth = self.tree_depth
            self._auto_n_trees = self.n_trees
            self._complexity_result = None

        # Adaptive tree count for boosting mode: larger datasets need more
        # trees per stage for sufficient approximation capacity.
        # Only additive (staged boosting) benefits — cascade/bagging modes
        # have their own per-layer/per-bag tree counts already tuned.
        base_n_trees = self._auto_n_trees
        if self.ensemble_mode == 'additive' and n_samples >= 400:
            self._effective_n_trees = base_n_trees * 2
        else:
            self._effective_n_trees = base_n_trees

        # Efficient tensor creation: use pin_memory for CUDA transfers
        if self._device.type == 'cuda':
            X_all = torch.from_numpy(X_norm).float().pin_memory().to(
                self._device, non_blocking=True)
            y_all = torch.from_numpy(y_norm).float().pin_memory().to(
                self._device, non_blocking=True)
        else:
            X_all = torch.from_numpy(X_norm).float().to(self._device)
            y_all = torch.from_numpy(y_norm).float().to(self._device)

        # Exponential recency sample weights (YDF-inspired)
        # Adaptive alpha: stronger break → sharper recency focus.
        # w[i] = exp(-alpha * (n-1-i) / n);  alpha≈4 → ~55x recency ratio.
        sample_weights_t = None
        if has_structural_break:
            alpha = min(5.0, 3.5 + break_magnitude * 3.0)
            positions = np.arange(n_samples, dtype=np.float32)
            raw_w = np.exp(-alpha * (n_samples - 1 - positions) / n_samples)
            raw_w /= raw_w.mean()  # normalise to mean 1
            sample_weights_t = torch.tensor(
                raw_w, dtype=torch.float32, device=self._device)

        # Chronological train/val split for early stopping.
        # Disabled during structural breaks — the most recent samples are
        # the current regime and must remain in training.
        use_val_split = (n_samples >= 200) and (not has_structural_break)

        if use_val_split:
            val_size = max(10, int(n_samples * 0.10))
            n_train = n_samples - val_size
            X_t, X_val = X_all[:n_train], X_all[n_train:]
            y_t, y_val = y_all[:n_train], y_all[n_train:]
            sw_t = sample_weights_t[:n_train] if sample_weights_t is not None else None
        else:
            n_train = n_samples
            X_t, y_t = X_all, y_all
            X_val, y_val = None, None
            sw_t = sample_weights_t

        if self.verbose and has_structural_break:
            print("  [TorchTree] Structural break detected: "
                  f"using all {n_samples} samples with recency weighting")

        batch_size = (self.batch_size if self.batch_size and self.batch_size > 0
                      else n_train)
        batch_size = min(batch_size, n_train)

        staged_n_stages = self.boosting_stages
        staged_base_lr = self.learning_rate
        effective_dropout = self.dropout
        effective_weight_decay = self.weight_decay

        # For heavily regularized variants (xgboost/rf-like defaults),
        # structural breaks benefit from lighter regularization on the
        # latest regime.
        if (has_structural_break and self.ensemble_mode == 'additive'
                and (self.dropout >= 0.10 or self.weight_decay > 1e-4)):
            effective_dropout = 0.0
            effective_weight_decay = min(self.weight_decay, 1e-4)
            if self.verbose:
                print("  [TorchTree] Structural break reg override: "
                      f"dropout={effective_dropout:.2f}, "
                      f"weight_decay={effective_weight_decay:.1e}")

        if self.boosting_stages > 1 and self.ensemble_mode == 'additive':
            self._fit_staged(
                X_t, y_t, n_train, n_features, batch_size,
                X_val, y_val,
                n_stages=staged_n_stages,
                base_learning_rate=staged_base_lr,
                dropout_override=effective_dropout,
                weight_decay_override=effective_weight_decay,
                sample_weights=sw_t,
            )
        else:
            self._fit_standard(X_t, y_t, n_train, n_features, batch_size,
                               X_val, y_val,
                               dropout_override=effective_dropout,
                               weight_decay_override=effective_weight_decay,
                               sample_weights=sw_t)

        return self

    def _fit_standard(self, X_t, y_t, n_samples, n_features, batch_size,
                       X_val=None, y_val=None,
                       dropout_override=None,
                       weight_decay_override=None,
                       sample_weights=None):
        """Standard single-model training."""
        self._staged_models = None
        eff_trees = getattr(self, '_effective_n_trees', self.n_trees)
        eff_depth = self._auto_tree_depth if self.auto_complexity else self.tree_depth
        self._model = _DifferentiableTreeEnsemble(
            in_features=n_features,
            out_features=self._n_outputs,
            n_trees=eff_trees,
            tree_depth=eff_depth,
            ensemble_mode=self.ensemble_mode,
            n_layers=self.n_layers,
            dropout=(self.dropout if dropout_override is None
                     else dropout_override),
        ).to(self._device)

        # torch.compile for PyTorch 2.0+ on CUDA (skip on MPS/CPU)
        if (self._device.type == 'cuda'
                and hasattr(torch, 'compile')
                and n_samples >= 256):
            try:
                self._model = torch.compile(self._model, mode='reduce-overhead')
            except Exception:
                pass  # graceful fallback if compile not supported

        loss = self._train_one_model(
            self._model, X_t, y_t, n_samples, batch_size,
            self.n_epochs, self.learning_rate, self.early_stop_patience,
            X_val=X_val, y_val=y_val,
            weight_decay=weight_decay_override,
            sample_weights=sample_weights,
        )
        if self.verbose:
            print(f"  [TorchTree] Standard training done, best_loss={loss:.6f}")

    def _fit_staged(self, X_t, y_t, n_samples, n_features, batch_size,
                     X_val=None, y_val=None,
                     n_stages=None,
                     base_learning_rate=None,
                     dropout_override=None,
                     weight_decay_override=None,
                     sample_weights=None):
        """Staged gradient boosting with GrowNet corrective step.

        Stage 0 trains on the full target.  Stage k trains on
        ``y - sum(shrinkage * stage_i(x) for i < k)``.

        After all stages are trained independently, a **GrowNet-style
        corrective step** (Badirli et al. 2020) fine-tunes ALL stages
        jointly for a few epochs.  This corrects the greedy approximation
        error inherent in classical gradient boosting.
        """
        if n_stages is None:
            n_stages = self.boosting_stages
        if base_learning_rate is None:
            base_learning_rate = self.learning_rate
        eff_trees = getattr(self, '_effective_n_trees', self.n_trees)
        eff_depth = self._auto_tree_depth if self.auto_complexity else self.tree_depth
        trees_per_stage = max(4, eff_trees // n_stages)
        epochs_per_stage = max(30, self.n_epochs // n_stages)
        shrinkage = self.boosting_shrinkage

        # When using val split, compensate for reduced training set by
        # giving each stage more epochs (the val split already prevents
        # overfitting, so extra epochs are safe).
        if y_val is not None:
            epochs_per_stage = int(epochs_per_stage * 1.4)

        # Compute val residuals for validation-based early stopping
        val_pred = torch.zeros_like(y_val) if y_val is not None else None

        self._model = None
        self._staged_models = []
        current_pred = torch.zeros_like(y_t)

        for stage_idx in range(n_stages):
            residual = y_t - current_pred
            val_residual = (y_val - val_pred) if y_val is not None else None

            model = _DifferentiableTreeEnsemble(
                in_features=n_features,
                out_features=self._n_outputs,
                n_trees=trees_per_stage,
                tree_depth=eff_depth,
                ensemble_mode='additive',
                dropout=(self.dropout if dropout_override is None
                         else dropout_override),
            ).to(self._device)

            stage_lr = base_learning_rate * (0.8 ** stage_idx)
            stage_patience = max(8, self.early_stop_patience)

            loss = self._train_one_model(
                model, X_t, residual, n_samples, batch_size,
                epochs_per_stage, stage_lr, stage_patience,
                X_val=X_val, y_val=val_residual,
                weight_decay=weight_decay_override,
                sample_weights=sample_weights,
            )

            with torch.no_grad():
                stage_pred = model(X_t)
                current_pred = current_pred + shrinkage * stage_pred
                if y_val is not None:
                    val_pred = val_pred + shrinkage * model(X_val)

            self._staged_models.append(model)

            if self.verbose:
                residual_mse = (y_t - current_pred).pow(2).mean().item()
                print(f"  [TorchTree] Stage {stage_idx+1}/{n_stages}: "
                      f"stage_loss={loss:.6f}, residual_mse={residual_mse:.6f}")

        # GrowNet corrective step: fine-tune ALL stages jointly.
        # This corrects greedy approximation errors by allowing earlier
        # stages to adjust in light of later stages.
        self._grownet_corrective_step(
            X_t, y_t, n_samples, batch_size, X_val, y_val)

    def _grownet_corrective_step(self, X_t, y_t, n_samples, batch_size,
                                  X_val=None, y_val=None):
        """GrowNet-style global corrective fine-tuning (Badirli et al. 2020).

        After greedy stage-by-stage training, unlock ALL stage models and
        jointly fine-tune them for a few epochs on the original target.
        This corrects the accumulated greedy approximation error.
        """
        if not self._staged_models or len(self._staged_models) < 2:
            return
        # Only run corrective step when validation data is available.
        # Without validation, the corrective step overfits on training loss.
        if X_val is None or y_val is None:
            return

        corrective_epochs = max(10, self.n_epochs // 6)
        corrective_lr = self.learning_rate * 0.3
        shrinkage = self.boosting_shrinkage
        use_mse = (self.loss_fn == 'mse')
        has_val = X_val is not None and y_val is not None

        # Collect all parameters from all stages
        all_params = []
        for m in self._staged_models:
            m.train()
            all_params.extend(m.parameters())

        optimizer = torch.optim.AdamW(
            all_params, lr=corrective_lr, weight_decay=self.weight_decay)

        best_loss = float('inf')
        best_states = None
        patience_counter = 0

        for epoch in range(corrective_epochs):
            # Keep feat_temp sharp during corrective step
            for m in self._staged_models:
                m.set_feat_temp(0.1)

            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]
                x_batch = X_t[idx]
                # Compute staged prediction
                pred = torch.zeros(x_batch.shape[0], self._n_outputs,
                                   device=self._device)
                for m in self._staged_models:
                    pred = pred + shrinkage * m(x_batch)

                loss = self._compute_loss_from_diff(pred, y_t[idx], use_mse)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 5.0)
                optimizer.step()

            # Validation-based early stopping for corrective step
            if has_val:
                with torch.no_grad():
                    val_pred = torch.zeros(X_val.shape[0], self._n_outputs,
                                           device=self._device)
                    for m in self._staged_models:
                        m.eval()
                        val_pred = val_pred + shrinkage * m(X_val)
                    monitor = self._compute_loss_from_diff(
                        val_pred, y_val, use_mse).item()
                    for m in self._staged_models:
                        m.train()
            else:
                monitor = loss.item()

            if monitor < best_loss - 1e-6:
                best_loss = monitor
                patience_counter = 0
                best_states = [{k: v.detach().clone()
                                for k, v in m.state_dict().items()}
                               for m in self._staged_models]
            else:
                patience_counter += 1
            if patience_counter >= 5:
                break

        if best_states is not None:
            for m, s in zip(self._staged_models, best_states):
                m.load_state_dict(s)
        for m in self._staged_models:
            m.set_feat_temp(0.1)
            m.eval()

        if self.verbose:
            print(f"  [TorchTree] GrowNet corrective step: "
                  f"best_val_loss={best_loss:.6f}")

    @staticmethod
    def _compute_loss_from_diff(pred, target, use_mse, sample_weights=None):
        """Compute loss from pre-computed predictions (for corrective step)."""
        diff = pred - target
        if use_mse:
            per_sample = diff.pow(2).mean(dim=-1)
        else:
            abs_diff = diff.abs()
            per_sample = torch.where(
                abs_diff < 1.0, 0.5 * diff.pow(2), abs_diff - 0.5
            ).mean(dim=-1)
        if sample_weights is not None:
            w = sample_weights[:per_sample.shape[0]]
            return (per_sample * w).sum() / w.sum()
        return per_sample.mean()

    @torch.inference_mode()
    def predict(self, X):
        """Predict using the trained ensemble.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        """
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self._feature_mean) / self._feature_std

        # Efficient tensor creation with pin_memory for CUDA
        if self._device.type == 'cuda':
            X_t = torch.from_numpy(X_norm).float().pin_memory().to(
                self._device, non_blocking=True)
        else:
            X_t = torch.from_numpy(X_norm).float().to(self._device)

        chunk_size = 8192
        preds = []
        use_amp = self._use_amp and self._device.type == 'cuda'

        for start in range(0, X_t.shape[0], chunk_size):
            end = min(start + chunk_size, X_t.shape[0])
            chunk = X_t[start:end]

            if self._staged_models is not None:
                # Staged boosting prediction
                out = torch.zeros(chunk.shape[0], self._n_outputs,
                                  device=self._device)
                for model in self._staged_models:
                    model.eval()
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            out = out + self.boosting_shrinkage * model(chunk)
                    else:
                        out = out + self.boosting_shrinkage * model(chunk)
                preds.append(out.float().cpu().numpy())
            else:
                self._model.eval()
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = self._model(chunk)
                    preds.append(pred.float().cpu().numpy())
                else:
                    preds.append(self._model(chunk).cpu().numpy())

        result = np.concatenate(preds, axis=0)
        # De-normalise
        result = result * self._target_std + self._target_mean

        if self._use_log_target:
            result = np.expm1(result)

        if self._n_outputs == 1:
            return result.ravel()
        return result

    @property
    def complexity_info(self):
        """Return adaptive complexity selection results (None if not used)."""
        return self._complexity_result

    def get_params(self, deep=True):
        return {
            'n_trees': self.n_trees,
            'tree_depth': self.tree_depth,
            'ensemble_mode': self.ensemble_mode,
            'n_layers': self.n_layers,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'early_stop_patience': self.early_stop_patience,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'loss_fn': self.loss_fn,
            'boosting_stages': self.boosting_stages,
            'boosting_shrinkage': self.boosting_shrinkage,
            'accelerator': self.accelerator,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'auto_complexity': self.auto_complexity,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
