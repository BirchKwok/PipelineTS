"""NLinear: A simple yet effective baseline for time series forecasting.

Reference: Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.

Key idea: Subtract the last value of the input sequence before feeding into a linear layer,
then add it back to the output. This simple normalization handles distribution shift effectively.

Enhancements over darts implementation:
- RevIN (Reversible Instance Normalization) for better distribution shift handling
- Huber loss by default (more robust than MSE)
- AdamW with weight decay for regularization
- Residual connection option
"""

from typing import Any, Union

import torch
import torch.nn as nn

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers import MultivariateWrapper


class NLinearBackbone(nn.Module):
    """NLinear: Normalize-Linear architecture.

    Subtracts the last value of the input, applies a linear layer,
    and adds the last value back. Optionally uses RevIN for further
    distribution normalization.

    Args:
        in_features: Input sequence length (lookback window).
        out_features: Prediction horizon.
        use_revin: Whether to use Reversible Instance Normalization.
        dropout: Dropout rate applied after the linear layer.
    """

    def __init__(self, in_features, out_features, use_revin=True, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_revin = use_revin

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

        if use_revin:
            self.revin_mean = None
            self.revin_std = None
            self.eps = 1e-5

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: (B, L) for univariate
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN normalization
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - mean) / std

        # NLinear: subtract last value
        last_val = x[:, -1:]  # (B, 1)
        x = x - last_val

        # Linear projection
        out = self.linear(x)  # (B, out_features)
        out = self.dropout(out)

        # Add back last value
        out = out + last_val

        # RevIN denormalization
        if self.use_revin:
            out = out * std + mean

        return out


class NLinear(TorchModelMixin, ForecastingMixin):
    """NLinear time series forecasting model for spinesTS.

    Args:
        in_features: Input sequence length.
        out_features: Prediction horizon.
        n_vars: Number of input variables (1 for univariate).
        use_revin: Whether to use RevIN normalization.
        dropout: Dropout rate.
        loss_fn: Loss function name ('mae', 'mse', 'huber').
        learning_rate: Learning rate for optimizer.
        random_seed: Random seed for reproducibility.
        device: Device to use ('auto', 'cuda', 'cpu', 'mps').
        weight_decay: Weight decay for AdamW optimizer.
        channel_mixing: Whether to use channel mixing for multivariate.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_vars: int = 1,
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
        self.use_revin = use_revin
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing

        super(NLinear, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = NLinearBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
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
