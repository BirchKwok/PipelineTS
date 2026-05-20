"""SRSNet: Selective Representation Space Network for Time Series Forecasting.

A novel architecture that front-loads the complexity of handling irregular
time series into a selective patch representation module (SRS), allowing
a simple MLP head for prediction.

Architecture:
    Input → RevIN → SRSBlock (Multi-Scale Patching + Selective Representation) → MLP Head → Output

Supports:
    - Univariate: (B, L) → (B, pred_len)
    - Multivariate-to-univariate: (B, L, C) → (B, pred_len)
    - Multivariate-to-multivariate: (B, L, C) → (B, pred_len, n_targets)
"""

from typing import Any, Union

import torch
from torch import nn

from PipelineTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.models.nn.layers._srs import RevIN, SRSBlock


class SRSBackbone(nn.Module):
    """SRS + MLP backbone for time series forecasting.

    The backbone combines:
    1. (Optional) RevIN for distribution shift handling
    2. Multi-scale adaptive patching via SRSBlock
    3. Selective representation via SRS attention-based scoring and selection
    4. Weighted patch aggregation
    5. Simple MLP head for final prediction

    Args:
        in_features: Input sequence length (lookback window).
        out_features: Prediction horizon length.
        n_vars: Number of input variables.
        n_targets: Number of target variables to predict (1 for many-to-one).
        d_model: Embedding dimension for patch representations.
        patch_sizes: List of patch sizes. Auto-determined if None.
        n_heads: Number of attention heads in SRS.
        top_k_ratio: Fraction of patches to retain after selection.
        dropout: Dropout rate.
        stride_ratio: Stride as fraction of patch size.
        use_revin: Whether to apply RevIN normalization.
        target_idx: Index of target variable for many-to-one denormalization.
    """

    def __init__(self, in_features, out_features, n_vars=1, n_targets=1,
                 d_model=64, patch_sizes=None, n_heads=4,
                 top_k_ratio=0.5, dropout=0.1, stride_ratio=0.5,
                 use_revin=True, target_idx=-1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_vars = n_vars
        self.n_targets = n_targets
        self.target_idx = target_idx
        self.use_revin = use_revin

        # RevIN for distribution shift
        if use_revin:
            self.revin = RevIN()
        else:
            self.revin = None

        # SRS Block: multi-scale patching + selective representation
        self.srs_block = SRSBlock(
            n_vars=n_vars, d_model=d_model, patch_sizes=patch_sizes,
            seq_len=in_features, n_heads=n_heads, top_k_ratio=top_k_ratio,
            dropout=dropout, stride_ratio=stride_ratio
        )

        # MLP Head — intentionally simple; complexity is in SRS
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_features * n_targets)
        )

    def forward(self, x):
        # Handle 2D univariate input
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, L) → (B, L, 1)

        B = x.shape[0]

        # RevIN normalization
        if self.revin is not None:
            x = self.revin.normalize(x)

        # SRS processing: multi-scale patching + selective representation
        selected, weights = self.srs_block(x)  # (B, K, d_model), (B, K)

        # Weighted aggregation of selected patches
        aggregated = (selected * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)

        # MLP prediction
        out = self.mlp_head(aggregated)  # (B, out_features * n_targets)

        # Reshape and denormalize
        if self.n_targets == 1:
            out = out.reshape(B, self.out_features)  # (B, pred_len)
            if self.revin is not None:
                out = out.unsqueeze(-1)  # (B, pred_len, 1)
                out = self.revin.denormalize(out, target_idx=self.target_idx)
                out = out.squeeze(-1)  # (B, pred_len)
        else:
            out = out.reshape(B, self.out_features, self.n_targets)  # (B, pred_len, n_targets)
            if self.revin is not None:
                out = self.revin.denormalize(out)

        return out


class SRSNet(TorchModelMixin, ForecastingMixin):
    """SRS-based Time Series Forecasting Model.

    Uses Selective Representation Space (SRS) to handle irregular time series
    with improved prediction stability. Front-loads irregularity handling
    into multi-scale selective patch representation, then uses a simple
    MLP head for prediction.

    The SRS module can also be used as a plug-and-play enhancement for
    other patch-based models (see SRSBlock in layers).

    Parameters
    ----------
    in_features : int
        Input sequence length (lookback window).
    out_features : int
        Prediction horizon length.
    n_vars : int
        Number of input variables. Default 1 (univariate).
    n_targets : int
        Number of target variables to predict. Default 1 (many-to-one).
        Set to n_vars for many-to-many prediction.
    d_model : int
        Embedding dimension for patch representations.
    patch_sizes : list of int, optional
        Patch sizes for multi-scale patching. Auto-determined if None.
    n_heads : int
        Number of attention heads in SRS cross-attention.
    top_k_ratio : float
        Fraction of patches to select (0.0 to 1.0).
    dropout : float
        Dropout rate.
    stride_ratio : float
        Stride as fraction of patch size (controls overlap).
    learning_rate : float
        Learning rate for AdamW optimizer.
    random_seed : int
        Random seed for reproducibility.
    device : str
        Device for computation ('auto', 'cpu', 'cuda', 'mps').
    loss_fn : str
        Loss function name ('mae', 'mse', 'huber', 'wmape', 'rmse').
    weight_decay : float
        Weight decay for AdamW optimizer.
    use_revin : bool
        Whether to use Reversible Instance Normalization.
    target_idx : int
        Index of target variable for many-to-one prediction.
        Default -1 (last variable).

    Examples
    --------
    Univariate:
        >>> model = SRSNet(in_features=36, out_features=12)
        >>> model.fit(X_train, y_train, epochs=100)  # X: (N, 36), y: (N, 12)

    Multivariate-to-univariate:
        >>> model = SRSNet(in_features=36, out_features=12, n_vars=7, n_targets=1)
        >>> model.fit(X_train, y_train, epochs=100)  # X: (N, 36, 7), y: (N, 12)

    Multivariate-to-multivariate:
        >>> model = SRSNet(in_features=36, out_features=12, n_vars=7, n_targets=7)
        >>> model.fit(X_train, y_train, epochs=100)  # X: (N, 36, 7), y: (N, 12, 7)
    """

    def __init__(self, in_features, out_features, n_vars=1, n_targets=1,
                 d_model=64, patch_sizes=None, n_heads=4, top_k_ratio=0.5,
                 dropout=0.1, stride_ratio=0.5, learning_rate=0.001,
                 random_seed=42, device='auto', loss_fn='mae',
                 weight_decay=1e-4, use_revin=True, target_idx=-1):
        self.in_features = in_features
        self.out_features = out_features
        self.n_vars = n_vars
        self.n_targets = n_targets
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.n_heads = n_heads
        self.top_k_ratio = top_k_ratio
        self.dropout = dropout
        self.stride_ratio = stride_ratio
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_revin = use_revin
        self.target_idx = target_idx
        self.loss_fn_name = loss_fn

        # this sentence needs to be the last one
        super(SRSNet, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        model = SRSBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            n_vars=self.n_vars,
            n_targets=self.n_targets,
            d_model=self.d_model,
            patch_sizes=self.patch_sizes,
            n_heads=self.n_heads,
            top_k_ratio=self.top_k_ratio,
            dropout=self.dropout,
            stride_ratio=self.stride_ratio,
            use_revin=self.use_revin,
            target_idx=self.target_idx
        )
        loss_fn = self.loss_fn
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return model, loss_fn, optimizer

    def fit(self,
            X_train: Any,
            y_train: Any,
            epochs: int = 1000,
            batch_size: Union[str, int] = 'auto',
            eval_set: Any = None,
            monitor: str = 'val_loss',
            min_delta: int = 0,
            patience: int = 100,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.5,
            restore_best_weights: bool = True,
            loss_type='min',
            verbose: bool = True,
            **kwargs: Any) -> Any:
        return super().fit(
            X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
            metrics_name=self.loss_fn_name,
            monitor=monitor, lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            min_delta=min_delta, patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose, **kwargs
        )
