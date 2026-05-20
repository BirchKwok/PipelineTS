import torch
from torch import nn
import torch.nn.functional as F

from PipelineTS.models.nn.layers import Time2Vec, MultivariateWrapper, RWKVEncoder, GlobalTemporalBlock
from PipelineTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.models.nn.backbones.utils import get_weight_norm



class MovingAvgDecompose(nn.Module):
    """Decompose input into trend (moving average) and seasonal (remainder)."""

    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, count_include_pad=False
        )

    def forward(self, x):
        # x: (B, L)
        trend = self.avg(x.unsqueeze(1)).squeeze(1)
        if trend.shape[1] > x.shape[1]:
            trend = trend[:, :x.shape[1]]
        seasonal = x - trend
        return trend, seasonal


class StableTime2Vec(nn.Module):
    """Time2Vec with structured frequency initialization for stable training.

    Instead of random initialization, uses log-spaced base frequencies
    so that the model starts with a useful multi-scale periodic basis.
    """

    def __init__(self, n_freqs=16):
        super().__init__()
        self.n_freqs = n_freqs

        # Initialize frequencies as log-spaced: covers low to high frequency
        init_freqs = torch.logspace(-2, 1, n_freqs)  # 0.01 to 10
        self.sin_w = nn.Parameter(init_freqs.unsqueeze(0))  # (1, n_freqs)
        self.sin_p = nn.Parameter(torch.zeros(n_freqs))  # phase starts at 0
        self.cos_w = nn.Parameter(init_freqs.unsqueeze(0) * 1.5)  # offset cos freqs
        self.cos_p = nn.Parameter(torch.zeros(n_freqs))

        # Linear component
        self.W = nn.Parameter(torch.zeros(1, 1))  # (1, 1)
        self.P = nn.Parameter(torch.zeros(1))

        # Output dim: sin(n) + cos(n) + linear(1) = 2*n + 1
        self.out_dim = n_freqs * 2 + 1

    def forward(self, x):
        # x: (B, L, 1)
        sin_part = torch.sin(x * self.sin_w + self.sin_p)  # (B, L, n_freqs)
        cos_part = torch.cos(x * self.cos_w + self.cos_p)  # (B, L, n_freqs)
        lin_part = x * self.W + self.P  # (B, L, 1)
        return torch.cat([sin_part, cos_part, lin_part], dim=-1)  # (B, L, 2*n+1)


class T2V(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, num_layers=2, device=None, use_gtb=False, gtb_d_model=64, routing_mode='static'):
        super(T2V, self).__init__()

        self.in_features, self.out_features = in_features, out_features
        self.eps = 1e-5
        d_model = 64

        # --- Trend-seasonal decomposition ---
        kernel_size = max(3, in_features // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.decompose = MovingAvgDecompose(kernel_size)

        # --- Trend path: lightweight DLinear-style ---
        self.trend_proj = nn.Linear(in_features, out_features)

        # --- Seasonal path: StableTime2Vec + raw signal → RWKV ---
        n_freqs = 24
        self.t2v = StableTime2Vec(n_freqs=n_freqs)
        t2v_out = n_freqs * 2 + 1  # 49

        # Combine periodic features + raw value → d_model
        self.embed = nn.Sequential(
            nn.Linear(t2v_out + 1, d_model),  # +1 for raw seasonal value
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # RWKV temporal mixing
        self.rwkv = RWKVEncoder(
            seq_len=in_features, d_model=d_model,
            n_blocks=max(2, num_layers), expand_ratio=2.0,
            dropout=dropout
        )

        # Per-timestep compress → flatten → output head
        self.step_compress = nn.Linear(d_model, d_model // 4)
        flatten_dim = in_features * (d_model // 4)
        self.seasonal_head = nn.Sequential(
            nn.Linear(flatten_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_features)
        )

        # Direct residual shortcut
        self.residual_proj = nn.Linear(in_features, out_features)

        # Global Temporal Block (pluggable enhancement)
        self.use_gtb = use_gtb
        if use_gtb:
            self.gtb = GlobalTemporalBlock(in_features, d_model=gtb_d_model, dropout=dropout, routing_mode=routing_mode)

    def forward(self, x):
        # x: (B, L)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x_norm = (x - mean) / std

        if self.use_gtb:
            x_norm = self.gtb(x_norm)

        B, L = x_norm.shape

        # Decompose into trend + seasonal
        trend, seasonal = self.decompose(x_norm)

        # --- Trend path ---
        trend_out = self.trend_proj(trend)  # (B, out_features)

        # --- Seasonal path ---
        s = seasonal.unsqueeze(-1)  # (B, L, 1)
        t2v_feats = self.t2v(s)     # (B, L, 49)

        # Concatenate raw seasonal value with periodic features
        h = torch.cat([s, t2v_feats], dim=-1)  # (B, L, 50)
        h = self.embed(h)  # (B, L, d_model)

        # RWKV temporal mixing
        h = self.rwkv(h)  # (B, L, d_model)

        # Per-timestep compress → flatten → output
        h = self.step_compress(h)  # (B, L, d_model//4)
        seasonal_out = self.seasonal_head(h.reshape(B, -1))  # (B, out_features)

        # Combine: trend + seasonal + residual
        out = trend_out + seasonal_out + self.residual_proj(x_norm)

        # RevIN denormalize
        out = out * std + mean
        return out


class Time2VecNet(TorchModelMixin, ForecastingMixin):
    def __init__(self, in_features, out_features, n_vars=1, learning_rate=0.001,
                 random_seed=42, device='auto', loss_fn='mae',
                 dropout=0.1, num_layers=2, weight_decay=1e-4,
                 channel_mixing=True, use_gtb=False, gtb_d_model=64, routing_mode='static'):
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.device = device
        self.loss_fn_name = loss_fn
        self.dropout = dropout
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        self.use_gtb = use_gtb
        self.gtb_d_model = gtb_d_model
        self.routing_mode = routing_mode
        # this sentence needs to be the last one
        super(Time2VecNet, self).__init__(random_seed, device=device, loss_fn=loss_fn)

    def call(self):
        backbone = T2V(
            self.in_features, self.out_features,
            dropout=self.dropout, num_layers=self.num_layers,
            device=self.device,
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

    def fit(
            self,
            X_train,
            y_train,
            epochs=1000,
            batch_size='auto',
            eval_set=None,
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            verbose=True,
            loss_type='min',
            **kwargs
    ):
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)
