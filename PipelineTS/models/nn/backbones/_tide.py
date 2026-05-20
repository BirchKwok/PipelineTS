"""TiDE: Time-series Dense Encoder for long-term forecasting.

Reference: Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder",
Transactions on Machine Learning Research, 2023.

Key ideas:
- Dense encoder maps the lookback to a compact representation
- Dense decoder maps back to forecast horizon
- Temporal decoder for per-step refinement
- Residual connection from lookback to forecast

Enhancements:
- RevIN normalization
- GELU activation
- Huber loss by default
- AdamW with weight decay
- Better residual connections
"""

from typing import Any, Union

import torch
import torch.nn as nn

from PipelineTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.nn_model.layers import MultivariateWrapper, GlobalTemporalBlock


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm + GELU."""

    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        return self.norm(self.skip(x) + self.dropout(self.act(self.fc(x))))


class TiDEBackbone(nn.Module):
    """TiDE: Time-series Dense Encoder backbone.

    Architecture:
    1. Dense encoder: lookback → hidden representation
    2. Dense decoder: hidden → forecast representation
    3. Temporal decoder: per-step refinement
    4. Residual: direct linear projection from lookback to forecast

    Args:
        in_features: Input sequence length (lookback).
        out_features: Prediction horizon.
        num_encoder_layers: Number of encoder residual blocks.
        num_decoder_layers: Number of decoder residual blocks.
        hidden_size: Hidden dimension.
        decoder_output_dim: Decoder output dimension per step.
        temporal_decoder_hidden: Temporal decoder hidden size.
        dropout: Dropout rate.
        use_revin: Whether to use RevIN.
    """

    def __init__(self, in_features, out_features, num_encoder_layers=2,
                 num_decoder_layers=2, hidden_size=128,
                 decoder_output_dim=16, temporal_decoder_hidden=32,
                 dropout=0.1, use_revin=True, use_gtb=False, gtb_d_model=64, routing_mode='static'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_revin = use_revin
        self.eps = 1e-5
        self.decoder_output_dim = decoder_output_dim

        # Dense Encoder
        encoder_layers = []
        current_dim = in_features
        for i in range(num_encoder_layers):
            out_dim = hidden_size
            encoder_layers.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Dense Decoder
        decoder_layers = []
        current_dim = hidden_size
        for i in range(num_decoder_layers):
            if i == num_decoder_layers - 1:
                out_dim = out_features * decoder_output_dim
            else:
                out_dim = hidden_size
            decoder_layers.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Temporal decoder: per-step refinement
        self.temporal_decoder = nn.Sequential(
            nn.Linear(decoder_output_dim, temporal_decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_decoder_hidden, 1)
        )

        # Global residual: direct linear from lookback to forecast
        self.global_residual = nn.Linear(in_features, out_features)

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

        B = x.shape[0]

        # Dense encoder
        h = self.encoder(x)  # (B, hidden_size)

        # Dense decoder
        h = self.decoder(h)  # (B, out_features * decoder_output_dim)
        h = h.view(B, self.out_features, self.decoder_output_dim)  # (B, H, d)

        # Temporal decoder (per-step)
        out = self.temporal_decoder(h).squeeze(-1)  # (B, H)

        # Global residual connection
        out = out + self.global_residual(x)

        # RevIN denormalize
        if self.use_revin:
            out = out * std + mean

        return out


class TiDE(TorchModelMixin, ForecastingMixin):
    """TiDE time series forecasting backbone for PipelineTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        hidden_size: Hidden dimension.
        decoder_output_dim: Decoder output dimension per step.
        temporal_decoder_hidden: Temporal decoder hidden size.
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
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 hidden_size: int = 128,
                 decoder_output_dim: int = 16,
                 temporal_decoder_hidden: int = 32,
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
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.decoder_output_dim = decoder_output_dim
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.dropout = dropout
        self.use_revin = use_revin
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        self.use_gtb = use_gtb
        self.gtb_d_model = gtb_d_model
        self.routing_mode = routing_mode

        super(TiDE, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = TiDEBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            hidden_size=self.hidden_size,
            decoder_output_dim=self.decoder_output_dim,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
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
