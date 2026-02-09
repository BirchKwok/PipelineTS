"""DLinear: Decomposition-Linear for time series forecasting.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.

Key idea: Decompose input into trend (moving average) and remainder (seasonal),
apply separate linear layers to each, then sum the results.

Enhancements over darts implementation:
- RevIN for distribution shift handling
- Adaptive kernel size based on input length
- Huber loss by default
- AdamW with weight decay
"""

from typing import Any, Union

import torch
import torch.nn as nn

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper


class MovingAvgBlock(nn.Module):
    """Moving average block for trend extraction."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size += 1
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1,
                                padding=self.kernel_size // 2, count_include_pad=False)

    def forward(self, x):
        # x: (B, L)
        # AvgPool1d expects (B, C, L)
        out = self.avg(x.unsqueeze(1)).squeeze(1)
        # Trim to match input length if needed
        if out.shape[1] > x.shape[1]:
            out = out[:, :x.shape[1]]
        return out


class DLinearBackbone(nn.Module):
    """DLinear: Decomposition-Linear architecture.

    Decomposes input into trend (via moving average) and seasonal (remainder),
    applies separate linear projections to each component, then sums.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        kernel_size: Kernel size for moving average decomposition.
            If None, auto-determined from in_features.
        use_revin: Whether to use RevIN normalization.
        dropout: Dropout rate.
    """

    def __init__(self, in_features, out_features, kernel_size=None,
                 use_revin=True, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_revin = use_revin

        # Auto kernel size: odd number, roughly 1/4 of input length
        if kernel_size is None:
            kernel_size = max(3, in_features // 4)
            if kernel_size % 2 == 0:
                kernel_size += 1

        self.decomposition = MovingAvgBlock(kernel_size)

        # Separate linear layers for trend and seasonal
        self.linear_trend = nn.Linear(in_features, out_features)
        self.linear_seasonal = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

        self.eps = 1e-5

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.linear_trend.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.linear_seasonal.weight, nonlinearity='linear')
        if self.linear_trend.bias is not None:
            nn.init.zeros_(self.linear_trend.bias)
        if self.linear_seasonal.bias is not None:
            nn.init.zeros_(self.linear_seasonal.bias)

    def forward(self, x):
        # x: (B, L) for univariate
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN normalization
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - mean) / std

        # Decomposition
        trend = self.decomposition(x)     # (B, L)
        seasonal = x - trend              # (B, L)

        # Separate linear projections
        trend_out = self.linear_trend(trend)        # (B, out_features)
        seasonal_out = self.linear_seasonal(seasonal)  # (B, out_features)

        out = self.dropout(trend_out + seasonal_out)

        # RevIN denormalization
        if self.use_revin:
            out = out * std + mean

        return out


class DLinear(TorchModelMixin, ForecastingMixin):
    """DLinear time series forecasting model for spinesTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables (1 for univariate).
        kernel_size: Kernel size for moving average. None for auto.
        use_revin: Whether to use RevIN normalization.
        dropout: Dropout rate.
        loss_fn: Loss function name.
        learning_rate: Learning rate.
        random_seed: Random seed.
        device: Device name.
        weight_decay: Weight decay for AdamW.
        channel_mixing: Whether to use channel mixing for multivariate.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_vars: int = 1,
                 kernel_size: int = None,
                 use_revin: bool = True,
                 dropout: float = 0.1,
                 loss_fn='huber',
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 weight_decay: float = 1e-4,
                 channel_mixing: bool = True
                 ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.n_vars = n_vars
        self.kernel_size = kernel_size
        self.use_revin = use_revin
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        super(DLinear, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = DLinearBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            kernel_size=self.kernel_size,
            use_revin=self.use_revin,
            dropout=self.dropout
        )
        if self.n_vars > 1:
            model = MultivariateWrapper(
                backbone, self.n_vars, self.out_features,
                channel_mixing=self.channel_mixing
            )
        else:
            model = backbone
        loss_fn = self.loss_fn
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
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
            patience: int = 10,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.7,
            restore_best_weights: bool = True,
            verbose: bool = True,
            loss_type='min',
            **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)
