from typing import Any, Union

import torch
import torch.nn.functional as F
from torch import nn

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper


class GatedResBlock(nn.Module):
    """Gated residual block with SiLU activation.

    x → LayerNorm → [sigmoid(gate) * SiLU(up)] → dropout → + residual
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.up = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        return x + self.drop(torch.sigmoid(self.gate(h)) * F.silu(self.up(h)))


class Seq2SeqBlock(nn.Module):
    """Gated residual MLP backbone for time series forecasting.

    Processes the full sequence as a single flattened vector through
    stacked GatedResBlocks with SiLU gating and residual connections.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        d_model: Hidden dimension.
        n_blocks: Number of gated residual blocks.
        dropout: Dropout rate.
    """

    def __init__(self, in_features, out_features, d_model=48, n_blocks=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model)
        )

        self.blocks = nn.ModuleList([
            GatedResBlock(d_model, dropout)
            for _ in range(n_blocks)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_features)

    def forward(self, x):
        # x: (B, L)
        if x.ndim == 3:
            x = x.squeeze(-1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.output_proj(x)


class StackingRNN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_vars: int = 1,
                 d_model: int = 48,
                 n_blocks: int = 3,
                 dropout=0.1,
                 loss_fn='mae',
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 weight_decay: float = 1e-4,
                 channel_mixing: bool = True
                 ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        # this sentence needs to be the last one
        super(StackingRNN, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = Seq2SeqBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
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
