"""TFT: Temporal Fusion Transformer for time series forecasting.

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", International Journal of Forecasting, 2021.

Simplified but effective implementation focusing on:
- Gated Residual Networks (GRN)
- LSTM encoder for temporal processing
- Interpretable multi-head attention
- RevIN normalization

Enhancements over darts:
- Pre-LayerNorm for stable training
- RevIN for distribution shift
- Huber loss by default
- AdamW with weight decay
- Simplified architecture that's faster to train
"""

from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper


class GatedLinearUnit(nn.Module):
    """GLU: Gated Linear Unit."""

    def __init__(self, d_model, d_out=None):
        super().__init__()
        d_out = d_out or d_model
        self.fc = nn.Linear(d_model, d_out)
        self.gate = nn.Linear(d_model, d_out)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """GRN: Gated Residual Network.

    x → LayerNorm → Linear → GELU → Linear → Dropout → GLU → + skip → LayerNorm
    """

    def __init__(self, d_model, d_hidden=None, d_output=None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or d_model
        d_output = d_output or d_model

        self.norm_in = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(d_output, d_output)
        self.norm_out = nn.LayerNorm(d_output)

        # Skip connection with projection if dimensions differ
        self.skip_proj = nn.Linear(d_model, d_output) if d_model != d_output else nn.Identity()

    def forward(self, x):
        skip = self.skip_proj(x)
        h = self.norm_in(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.norm_out(skip + h)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention.

    Unlike standard MHA, this shares values across heads to produce
    interpretable attention weights.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.d_head)  # Shared value
        self.out_proj = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** 0.5

    def forward(self, q, k, v):
        B, L, _ = q.shape

        # Project queries and keys (multi-head), values (single head)
        Q = self.q_proj(q).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, L, d)
        K = self.k_proj(k).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(v)  # (B, L, d_head) - shared across heads

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Average attention across heads, then apply to shared value
        attn_avg = attn.mean(dim=1)  # (B, L, L)
        out = torch.matmul(attn_avg, V)  # (B, L, d_head)
        out = self.out_proj(out)  # (B, L, d_model)

        return out


class TFTBackbone(nn.Module):
    """Temporal Fusion Transformer backbone.

    Architecture:
    1. Input embedding + positional encoding
    2. LSTM encoder for local temporal patterns
    3. GRN for gated processing
    4. Interpretable Multi-Head Attention for long-range dependencies
    5. Output projection

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        hidden_size: Hidden dimension.
        lstm_layers: Number of LSTM layers.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        use_revin: Whether to use RevIN.
    """

    def __init__(self, in_features, out_features, hidden_size=32,
                 lstm_layers=1, n_heads=4, dropout=0.1, use_revin=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.use_revin = use_revin
        self.eps = 1e-5

        # Ensure n_heads divides hidden_size
        for h in [n_heads, 4, 2, 1]:
            if hidden_size % h == 0:
                n_heads = h
                break

        # Input embedding
        self.input_embed = nn.Linear(1, hidden_size)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, in_features, hidden_size) * 0.02
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Post-LSTM GRN
        self.grn_temporal = GatedResidualNetwork(hidden_size, dropout=dropout)

        # Interpretable attention
        self.attn = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.grn_attn = GatedResidualNetwork(hidden_size, dropout=dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(in_features * hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, out_features)
        )

    def forward(self, x):
        # x: (B, L)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - mean) / std

        B, L = x.shape

        # Input embedding
        h = self.input_embed(x.unsqueeze(-1))  # (B, L, hidden_size)
        h = h + self.pos_encoding[:, :L, :]

        # LSTM encoding
        lstm_out, _ = self.lstm(h)  # (B, L, hidden_size)

        # Post-LSTM GRN
        h = self.grn_temporal(lstm_out)

        # Self-attention
        attn_out = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.grn_attn(h)

        # Output
        h = h.reshape(B, -1)
        out = self.output_head(h)

        # RevIN denormalize
        if self.use_revin:
            out = out * std + mean

        return out


class TFT(TorchModelMixin, ForecastingMixin):
    """Temporal Fusion Transformer for spinesTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables.
        hidden_size: Hidden dimension.
        lstm_layers: Number of LSTM layers.
        n_heads: Number of attention heads.
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
                 hidden_size: int = 32,
                 lstm_layers: int = 1,
                 n_heads: int = 4,
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
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_revin = use_revin
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        super(TFT, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = TFTBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            n_heads=self.n_heads,
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
