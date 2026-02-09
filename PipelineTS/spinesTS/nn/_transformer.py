"""Transformer for Time Series Forecasting.

A lightweight encoder-only Transformer tailored for time series.

Enhancements over darts:
- Pre-LayerNorm for stable training
- Learnable positional encoding
- RevIN normalization
- GELU activation
- Huber loss by default
- AdamW with weight decay
"""

import math
from typing import Any, Union

import torch
import torch.nn as nn

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attn(h, h, h, attn_mask=mask)[0]
        x = x + h
        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class TransformerBackbone(nn.Module):
    """Encoder-only Transformer for time series forecasting.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        dim_feedforward: Feedforward dimension.
        dropout: Dropout rate.
        use_revin: Whether to use RevIN.
    """

    def __init__(self, in_features, out_features, d_model=64, nhead=4,
                 num_encoder_layers=3, dim_feedforward=256, dropout=0.1,
                 use_revin=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.d_model = d_model
        self.use_revin = use_revin
        self.eps = 1e-5

        # Ensure nhead divides d_model
        for h in [nhead, 4, 2, 1]:
            if d_model % h == 0:
                nhead = h
                break

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, in_features, d_model) * 0.02
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: flatten and project
        self.output_head = nn.Sequential(
            nn.Linear(in_features * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_features)
        )

    def forward(self, x):
        # x: (B, L) for univariate
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - mean) / std

        B, L = x.shape

        # Reshape to (B, L, 1) and project to d_model
        h = self.input_proj(x.unsqueeze(-1))  # (B, L, d_model)
        h = h + self.pos_encoding[:, :L, :]

        # Encoder
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.final_norm(h)

        # Flatten and project to output
        h = h.reshape(B, -1)  # (B, L * d_model)
        out = self.output_head(h)  # (B, out_features)

        # RevIN denormalize
        if self.use_revin:
            out = out * std + mean

        return out


class TSTransformer(TorchModelMixin, ForecastingMixin):
    """Transformer time series forecasting model for spinesTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        dim_feedforward: Feedforward dimension.
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
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
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
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.use_revin = use_revin
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        super(TSTransformer, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = TransformerBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
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
