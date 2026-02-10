from typing import Any, Union

import torch
from torch import nn

from PipelineTS.spinesTS.layers import GAU, PositionalEncoding, MultivariateWrapper, GlobalTemporalBlock
from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin


class GAUBlock(nn.Module):
    def __init__(self, d_model, level=2, query_key_dim=512, expansion_factor=4., skip_connect=True, dropout=0.2, **kwargs):
        super(GAUBlock, self).__init__()
        self.gau = nn.ModuleList([
            nn.Sequential(
                PositionalEncoding(d_model, add_x=True),
                GAU(d_model, query_key_dim=query_key_dim, expansion_factor=expansion_factor, 
                    skip_connect=skip_connect, dropout=dropout, **kwargs)
            )
            for i in range(level)
            ])
        self.level = level

    def forward(self, x):
        # x: (B, seq_len, d_model) — always 3D
        for i in self.gau:
            x = i(x)
        return x


class GAUBase(nn.Module):
    def __init__(self, in_features, out_features, d_model=32, num_heads=4,
                 level=2, dropout=0.1, use_gtb=False, gtb_d_model=64, routing_mode='static'):
        super(GAUBase, self).__init__()
        self.in_features = in_features   # lags (sequence length)
        self.out_features = out_features
        self.eps = 1e-5

        # Learned feature projection: each raw value -> d_model dimensional representation
        self.feature_head = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # GAU blocks: temporal attention across lag time steps (GAU already has attention)
        self.gau = GAUBlock(d_model, level=level, dropout=dropout)

        # Temporal compression via learned weighted pooling
        self.temporal_weight = nn.Linear(d_model, 1)
        self.final_norm = nn.LayerNorm(d_model)

        # Output head: (B, d_model) -> (B, out_features)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
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
        # x: (B, lags) 2D — standard univariate input
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x_norm = (x - mean) / std

        if self.use_gtb:
            x_norm = self.gtb(x_norm)

        B, L = x_norm.shape

        # Learned feature projection: (B, lags, 1) -> (B, lags, d_model)
        h = x_norm.unsqueeze(-1)  # (B, lags, 1)
        h = self.feature_head(h)

        # GAU temporal processing: attention across lag time steps
        h = self.gau(h)  # (B, lags, d_model)

        # Learned weighted pooling: (B, lags, d_model) -> (B, d_model)
        weights = torch.softmax(self.temporal_weight(h).squeeze(-1), dim=1)  # (B, lags)
        h = (h * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        h = self.final_norm(h)

        # Output projection + residual shortcut
        out = self.output_head(h) + self.residual_proj(x_norm)

        # RevIN denormalize
        out = out * std + mean
        return out


class GAUNet(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: Any,
                 out_features: Any,
                 n_vars: int = 1,
                 d_model: int = 32,
                 num_heads: int = 4,
                 level: int = 3,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 loss_fn='huber',
                 dropout: float = 0.2,
                 weight_decay: float = 1e-4,
                 query_key_dim: int = 512,
                 expansion_factor: float = 4.0,
                 channel_mixing: bool = True,
                 use_gtb: bool = False,
                 gtb_d_model: int = 64,
                 routing_mode: str = 'static'
                 ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.d_model = d_model
        self.num_heads = num_heads
        self.level = level
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        self.use_gtb = use_gtb
        self.gtb_d_model = gtb_d_model
        self.routing_mode = routing_mode

        # this sentence needs to be the last one
        super(GAUNet, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = GAUBase(
            self.in_features, 
            self.out_features,
            d_model=self.d_model,
            num_heads=self.num_heads,
            level=self.level,
            dropout=self.dropout,
            use_gtb=self.use_gtb, 
            gtb_d_model=self.gtb_d_model,
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
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True
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
            patience: int = 20,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.5,
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
