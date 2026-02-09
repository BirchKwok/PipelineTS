from typing import Any, Union

import torch
from torch import nn

from PipelineTS.spinesTS.layers import GAU, PositionalEncoding, MultivariateWrapper
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
    def __init__(self, in_features, out_features, level=2, dropout=0.2):
        super(GAUBase, self).__init__()
        self.in_features = in_features   # lags (sequence length)
        self.out_features = out_features
        d_model = in_features

        # Learned feature projection: each raw value -> d_model dimensional representation
        # This replaces external feature engineering with learned features
        self.feature_head = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # GAU blocks: temporal attention across lag time steps
        self.gau = GAUBlock(d_model, level=level, dropout=dropout)
        
        # Multi-head self-attention - dynamically choose valid num_heads
        num_heads = 1
        for h in [8, 4, 2, 1]:
            if d_model % h == 0:
                num_heads = h
                break
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Temporal compression: (B, lags, d_model) -> (B, lags)
        self.temporal_proj = nn.Linear(d_model, 1)

        # Output head: (B, lags) -> (B, out_features)
        self.output_head = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features * 2, out_features)
        )

    def forward(self, x):
        # x: (B, lags) 2D — standard univariate input
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, lags) -> (B, lags, 1)

        # Learned feature projection: (B, lags, 1) -> (B, lags, d_model)
        x = self.feature_head(x)

        # GAU temporal processing: attention across lag time steps
        x = self.gau(x)  # (B, lags, d_model)

        # Multi-head self-attention with residual connection
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout_layer(attn_output)
        x = self.norm(x)

        # Temporal compression: (B, lags, d_model) -> (B, lags, 1) -> (B, lags)
        x = self.temporal_proj(x).squeeze(-1)

        # Output projection: (B, lags) -> (B, out_features)
        return self.output_head(x)


class GAUNet(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: Any,
                 out_features: Any,
                 n_vars: int = 1,
                 level: int = 3,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 loss_fn='huber',
                 dropout: float = 0.2,
                 weight_decay: float = 1e-4,
                 query_key_dim: int = 512,
                 expansion_factor: float = 4.0,
                 channel_mixing: bool = True
                 ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.level = level
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.query_key_dim = query_key_dim
        self.expansion_factor = expansion_factor
        self.channel_mixing = channel_mixing

        # this sentence needs to be the last one
        super(GAUNet, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = GAUBase(
            self.in_features, 
            self.out_features,
            level=self.level,
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
