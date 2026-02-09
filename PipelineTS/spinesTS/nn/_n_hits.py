"""N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.

Reference: Challu et al., "N-HiTS: Neural Hierarchical Interpolation for
Time Series Forecasting", AAAI 2023.

Key ideas:
- Multi-rate signal sampling via MaxPool downsampling
- Hierarchical interpolation for multi-scale temporal patterns
- Doubly residual stacking like N-BEATS

Enhancements over darts:
- RevIN normalization
- GELU activation
- Huber loss by default
- AdamW with weight decay
- Better default hyperparameters
"""

import math
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper


class NHiTSBlock(nn.Module):
    """N-HiTS block with multi-rate signal sampling.

    Each block:
    1. Downsamples input via MaxPool
    2. Processes through FC stack
    3. Produces backcast and forecast coefficients
    4. Interpolates forecast to target resolution

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        layer_widths: Width of FC layers.
        num_layers: Number of FC layers.
        pool_kernel_size: Kernel size for MaxPool downsampling.
        n_freq_downsample: Factor to downsample the output frequency.
        dropout: Dropout rate.
    """

    def __init__(self, in_features, out_features, layer_widths=512,
                 num_layers=2, pool_kernel_size=1, n_freq_downsample=1,
                 dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.n_freq_downsample = max(1, n_freq_downsample)

        # MaxPool downsampling
        if pool_kernel_size > 1:
            self.pooling = nn.MaxPool1d(kernel_size=pool_kernel_size,
                                        stride=pool_kernel_size,
                                        ceil_mode=True)
        else:
            self.pooling = None

        # Compute input dim after pooling
        pooled_len = math.ceil(in_features / max(pool_kernel_size, 1))

        # FC stack
        layers = []
        current_dim = pooled_len
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, layer_widths))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = layer_widths
        self.fc_stack = nn.Sequential(*layers)

        # Backcast coefficients (at downsampled resolution)
        backcast_out = math.ceil(in_features / max(self.n_freq_downsample, 1))
        self.backcast_fc = nn.Linear(layer_widths, backcast_out)

        # Forecast coefficients (at downsampled resolution)
        forecast_out = math.ceil(out_features / max(self.n_freq_downsample, 1))
        self.forecast_fc = nn.Linear(layer_widths, forecast_out)

        self.backcast_target_len = in_features
        self.forecast_target_len = out_features

    def forward(self, x):
        # x: (B, L)
        # Downsample via MaxPool
        if self.pooling is not None:
            h = self.pooling(x.unsqueeze(1)).squeeze(1)  # (B, L_pooled)
        else:
            h = x

        # FC stack
        h = self.fc_stack(h)

        # Backcast and forecast at downsampled resolution
        backcast_coeff = self.backcast_fc(h)  # (B, L_down)
        forecast_coeff = self.forecast_fc(h)  # (B, H_down)

        # Interpolate to full resolution
        if backcast_coeff.shape[1] != self.backcast_target_len:
            backcast = F.interpolate(
                backcast_coeff.unsqueeze(1),
                size=self.backcast_target_len,
                mode='linear', align_corners=False
            ).squeeze(1)
        else:
            backcast = backcast_coeff

        if forecast_coeff.shape[1] != self.forecast_target_len:
            forecast = F.interpolate(
                forecast_coeff.unsqueeze(1),
                size=self.forecast_target_len,
                mode='linear', align_corners=False
            ).squeeze(1)
        else:
            forecast = forecast_coeff

        return backcast, forecast


class NHiTSBackbone(nn.Module):
    """N-HiTS backbone with hierarchical interpolation.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        num_stacks: Number of stacks.
        num_blocks: Number of blocks per stack.
        num_layers: Number of FC layers per block.
        layer_widths: Width of FC layers.
        pooling_kernel_sizes: List of pool kernel sizes per stack.
        n_freq_downsample: List of downsample factors per stack.
        dropout: Dropout rate.
        use_revin: Whether to use RevIN.
    """

    def __init__(self, in_features, out_features, num_stacks=3,
                 num_blocks=1, num_layers=2, layer_widths=512,
                 pooling_kernel_sizes=None, n_freq_downsample=None,
                 dropout=0.1, use_revin=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_revin = use_revin
        self.eps = 1e-5

        # Auto-determine pooling kernel sizes (increasing across stacks)
        if pooling_kernel_sizes is None:
            pooling_kernel_sizes = []
            for i in range(num_stacks):
                k = min(2 ** i, max(1, in_features // 4))
                pooling_kernel_sizes.append(k)

        # Auto-determine frequency downsampling (decreasing across stacks)
        if n_freq_downsample is None:
            n_freq_downsample = []
            for i in range(num_stacks):
                d = max(1, out_features // (2 ** i))
                n_freq_downsample.append(d)
            n_freq_downsample = list(reversed(n_freq_downsample))

        # Pad if needed
        while len(pooling_kernel_sizes) < num_stacks:
            pooling_kernel_sizes.append(pooling_kernel_sizes[-1])
        while len(n_freq_downsample) < num_stacks:
            n_freq_downsample.append(1)

        blocks = []
        for s in range(num_stacks):
            for _ in range(num_blocks):
                blocks.append(NHiTSBlock(
                    in_features=in_features,
                    out_features=out_features,
                    layer_widths=layer_widths,
                    num_layers=num_layers,
                    pool_kernel_size=pooling_kernel_sizes[s],
                    n_freq_downsample=n_freq_downsample[s],
                    dropout=dropout
                ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: (B, L)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - mean) / std

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


class NHiTS(TorchModelMixin, ForecastingMixin):
    """N-HiTS time series forecasting model for spinesTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables.
        num_stacks: Number of stacks.
        num_blocks: Number of blocks per stack.
        num_layers: Number of FC layers per block.
        layer_widths: Width of FC layers.
        pooling_kernel_sizes: List of pool kernel sizes per stack. None=auto.
        n_freq_downsample: List of downsample factors per stack. None=auto.
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
                 num_stacks: int = 3,
                 num_blocks: int = 1,
                 num_layers: int = 2,
                 layer_widths: int = 512,
                 pooling_kernel_sizes=None,
                 n_freq_downsample=None,
                 dropout: float = 0.1,
                 use_revin: bool = True,
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
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.pooling_kernel_sizes = pooling_kernel_sizes
        self.n_freq_downsample = n_freq_downsample
        self.dropout = dropout
        self.use_revin = use_revin
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        super(NHiTS, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = NHiTSBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            num_layers=self.num_layers,
            layer_widths=self.layer_widths,
            pooling_kernel_sizes=self.pooling_kernel_sizes,
            n_freq_downsample=self.n_freq_downsample,
            dropout=self.dropout,
            use_revin=self.use_revin
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
