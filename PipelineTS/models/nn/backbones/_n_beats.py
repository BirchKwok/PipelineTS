"""N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting.

Reference: Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting", ICLR 2020.

Key ideas:
- Doubly residual stacking: both backcast and forecast residuals
- Interpretable architecture: trend (polynomial) + seasonality (Fourier) basis
- Generic architecture: learnable basis functions

Enhancements:
- RevIN normalization
- GELU activation instead of ReLU
- Huber loss by default
- Better initialization
- AdamW with weight decay
"""

import math
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn

from PipelineTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.models.nn.layers import MultivariateWrapper, GlobalTemporalBlock


class NBEATSBlock(nn.Module):
    """Base N-BEATS block with FC stack.

    Args:
        in_features: Input (backcast) length.
        out_features: Output (forecast) length.
        layer_widths: Width of FC layers.
        num_layers: Number of FC layers.
        expansion_coeff_dim: Dimension of expansion coefficients.
        dropout: Dropout rate.
    """

    def __init__(self, in_features, out_features, layer_widths=256,
                 num_layers=4, expansion_coeff_dim=5, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.expansion_coeff_dim = expansion_coeff_dim

        # FC stack
        layers = []
        current_dim = in_features
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, layer_widths))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = layer_widths
        self.fc_stack = nn.Sequential(*layers)

        # Coefficient projections (to be overridden in subclasses)
        self.backcast_coeff = nn.Linear(layer_widths, expansion_coeff_dim)
        self.forecast_coeff = nn.Linear(layer_widths, expansion_coeff_dim)

    def forward(self, x):
        # x: (B, in_features)
        h = self.fc_stack(x)
        backcast_coeff = self.backcast_coeff(h)
        forecast_coeff = self.forecast_coeff(h)
        return backcast_coeff, forecast_coeff


class GenericBlock(NBEATSBlock):
    """Generic N-BEATS block with learnable basis functions."""

    def __init__(self, in_features, out_features, layer_widths=256,
                 num_layers=4, expansion_coeff_dim=32, dropout=0.1):
        super().__init__(in_features, out_features, layer_widths,
                         num_layers, expansion_coeff_dim, dropout)

        # Learnable basis for backcast and forecast
        self.backcast_basis = nn.Linear(expansion_coeff_dim, in_features)
        self.forecast_basis = nn.Linear(expansion_coeff_dim, out_features)

    def forward(self, x):
        bc, fc = super().forward(x)
        backcast = self.backcast_basis(bc)
        forecast = self.forecast_basis(fc)
        return backcast, forecast


class TrendBlock(NBEATSBlock):
    """Trend block using polynomial basis functions."""

    def __init__(self, in_features, out_features, layer_widths=256,
                 num_layers=4, degree=3, dropout=0.1):
        super().__init__(in_features, out_features, layer_widths,
                         num_layers, degree + 1, dropout)
        self.degree = degree

        # Pre-compute polynomial basis (not learnable)
        backcast_time = torch.arange(in_features, dtype=torch.float32) / in_features
        forecast_time = torch.arange(out_features, dtype=torch.float32) / out_features

        # (degree+1, L) polynomial basis: t^0, t^1, ..., t^degree
        backcast_basis = torch.stack([backcast_time ** i for i in range(degree + 1)])
        forecast_basis = torch.stack([forecast_time ** i for i in range(degree + 1)])

        self.register_buffer('backcast_basis', backcast_basis)   # (degree+1, in_features)
        self.register_buffer('forecast_basis', forecast_basis)   # (degree+1, out_features)

    def forward(self, x):
        bc, fc = super().forward(x)
        # bc: (B, degree+1), backcast_basis: (degree+1, in_features)
        backcast = torch.matmul(bc, self.backcast_basis)   # (B, in_features)
        forecast = torch.matmul(fc, self.forecast_basis)   # (B, out_features)
        return backcast, forecast


class SeasonalityBlock(NBEATSBlock):
    """Seasonality block using Fourier basis functions."""

    def __init__(self, in_features, out_features, layer_widths=256,
                 num_layers=4, num_harmonics=None, dropout=0.1):
        if num_harmonics is None:
            num_harmonics = max(1, out_features // 2)
        expansion_coeff_dim = 2 * num_harmonics

        super().__init__(in_features, out_features, layer_widths,
                         num_layers, expansion_coeff_dim, dropout)

        self.num_harmonics = num_harmonics

        # Pre-compute Fourier basis
        backcast_time = torch.arange(in_features, dtype=torch.float32) / in_features
        forecast_time = torch.arange(out_features, dtype=torch.float32) / out_features

        backcast_cos = torch.stack([torch.cos(2 * math.pi * k * backcast_time)
                                    for k in range(1, num_harmonics + 1)])
        backcast_sin = torch.stack([torch.sin(2 * math.pi * k * backcast_time)
                                    for k in range(1, num_harmonics + 1)])
        backcast_basis = torch.cat([backcast_cos, backcast_sin], dim=0)  # (2*H, in_features)

        forecast_cos = torch.stack([torch.cos(2 * math.pi * k * forecast_time)
                                    for k in range(1, num_harmonics + 1)])
        forecast_sin = torch.stack([torch.sin(2 * math.pi * k * forecast_time)
                                    for k in range(1, num_harmonics + 1)])
        forecast_basis = torch.cat([forecast_cos, forecast_sin], dim=0)  # (2*H, out_features)

        self.register_buffer('backcast_basis', backcast_basis)
        self.register_buffer('forecast_basis', forecast_basis)

    def forward(self, x):
        bc, fc = super().forward(x)
        backcast = torch.matmul(bc, self.backcast_basis)
        forecast = torch.matmul(fc, self.forecast_basis)
        return backcast, forecast


class NBEATSBackbone(nn.Module):
    """N-BEATS backbone with doubly residual stacking.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        generic_architecture: If True, use generic blocks. If False, use
            interpretable (trend+seasonality) blocks.
        num_stacks: Number of stacks.
        num_blocks: Number of blocks per stack.
        num_layers: Number of FC layers per block.
        layer_widths: Width of FC layers.
        expansion_coeff_dim: Expansion coefficient dimension (generic only).
        trend_degree: Polynomial degree for trend (interpretable only).
        dropout: Dropout rate.
        use_revin: Whether to use RevIN.
    """

    def __init__(self, in_features, out_features, generic_architecture=True,
                 num_stacks=2, num_blocks=3, num_layers=4, layer_widths=256,
                 expansion_coeff_dim=32, trend_degree=3, dropout=0.1,
                 use_revin=True, use_gtb=False, gtb_d_model=64, routing_mode='static'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_revin = use_revin
        self.eps = 1e-5

        blocks = []
        if generic_architecture:
            for _ in range(num_stacks):
                for _ in range(num_blocks):
                    blocks.append(GenericBlock(
                        in_features, out_features, layer_widths,
                        num_layers, expansion_coeff_dim, dropout
                    ))
        else:
            # Interpretable: trend stacks then seasonality stacks
            num_trend_stacks = max(1, num_stacks // 2)
            num_seasonal_stacks = num_stacks - num_trend_stacks
            for _ in range(num_trend_stacks):
                for _ in range(num_blocks):
                    blocks.append(TrendBlock(
                        in_features, out_features, layer_widths,
                        num_layers, trend_degree, dropout
                    ))
            for _ in range(num_seasonal_stacks):
                for _ in range(num_blocks):
                    blocks.append(SeasonalityBlock(
                        in_features, out_features, layer_widths,
                        num_layers, dropout=dropout
                    ))

        self.blocks = nn.ModuleList(blocks)

        # Global Temporal Block (pluggable enhancement)
        self.use_gtb = use_gtb
        if use_gtb:
            self.gtb = GlobalTemporalBlock(in_features, d_model=gtb_d_model, dropout=dropout, routing_mode=routing_mode)

    def forward(self, x):
        # x: (B, L)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - mean) / std

        if self.use_gtb:
            x = self.gtb(x)

        residual = x
        forecast = torch.zeros(x.shape[0], self.out_features, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # RevIN denormalize
        if self.use_revin:
            forecast = forecast * std + mean

        return forecast


class NBeats(TorchModelMixin, ForecastingMixin):
    """N-BEATS time series forecasting backbone for PipelineTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables.
        generic_architecture: Use generic (True) or interpretable (False) architecture.
        num_stacks: Number of stacks.
        num_blocks: Number of blocks per stack.
        num_layers: Number of FC layers per block.
        layer_widths: Width of FC layers.
        expansion_coeff_dim: Expansion coefficient dimension.
        trend_degree: Polynomial degree for trend basis.
        dropout: Dropout rate.
        use_revin: Whether to use RevIN.
        loss_fn: Loss function name.
        learning_rate: Learning rate.
        random_seed: Random seed.
        device: Device name.
        weight_decay: Weight decay.
        channel_mixing: Channel mixing for multivariate.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_vars: int = 1,
                 generic_architecture: bool = True,
                 num_stacks: int = 2,
                 num_blocks: int = 3,
                 num_layers: int = 4,
                 layer_widths: int = 256,
                 expansion_coeff_dim: int = 32,
                 trend_degree: int = 3,
                 dropout: float = 0.1,
                 use_revin: bool = True,
                 loss_fn='huber',
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 weight_decay: float = 1e-4,
                 channel_mixing: bool = True,
                 use_gtb: bool = False,
                 gtb_d_model: int = 64,
                 routing_mode: str = 'static'
                 ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.n_vars = n_vars
        self.generic_architecture = generic_architecture
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.expansion_coeff_dim = expansion_coeff_dim
        self.trend_degree = trend_degree
        self.dropout = dropout
        self.use_revin = use_revin
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        self.use_gtb = use_gtb
        self.gtb_d_model = gtb_d_model
        self.routing_mode = routing_mode

        super(NBeats, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = NBEATSBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            generic_architecture=self.generic_architecture,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            expansion_coeff_dim=self.expansion_coeff_dim,
            trend_degree=self.trend_degree,
            dropout=self.dropout,
            use_revin=self.use_revin,
            use_gtb=self.use_gtb, gtb_d_model=self.gtb_d_model,
            routing_mode=self.routing_mode
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
