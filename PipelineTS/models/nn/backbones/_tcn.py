# reference: https://github.com/locuslab/TCN/tree/master
from typing import Any, Union

import torch
import torch.nn as nn

from PipelineTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.models.nn.layers import MultivariateWrapper, GlobalTemporalBlock
from PipelineTS.models.nn.backbones.utils import get_weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, device=None):
        super(TemporalBlock, self).__init__()
        weight_norm = get_weight_norm(device)

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1,
                                 self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.norm = nn.LayerNorm(n_outputs)
        self.act_out = nn.GELU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='linear')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = self.act_out(out + res)
        # Apply LayerNorm on the feature dimension (transpose for Conv1d format)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=2, dropout=0.2, num_levels=None, hidden_channels=None, device=None, use_gtb=False, gtb_d_model=64, routing_mode='static'):
        super(TemporalConvNet, self).__init__()
        self.in_features = in_features
        self.eps = 1e-5

        # Auto-determine number of TCN levels based on input length for full receptive field
        if num_levels is None:
            import math
            num_levels = max(2, int(math.ceil(math.log2(max(in_features, 4) / (kernel_size - 1)))) + 1)
            num_levels = min(num_levels, 6)  # Cap at 6 levels

        if hidden_channels is None:
            hidden_channels = min(max(in_features, 32), 128)

        # First layer: 1 input channel (univariate), rest: hidden_channels
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = 1 if i == 0 else hidden_channels
            out_ch = hidden_channels
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout, device=device)
            )

        self.network = nn.Sequential(*layers)
        # Adaptive pooling to collapse temporal dim, then project
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(hidden_channels, out_features)

        # Direct residual shortcut
        self.residual_proj = nn.Linear(in_features, out_features)

        # Global Temporal Block (pluggable enhancement)
        self.use_gtb = use_gtb
        if use_gtb:
            self.gtb = GlobalTemporalBlock(in_features, d_model=gtb_d_model, dropout=dropout, routing_mode=routing_mode)

    def forward(self, x):
        # x: (B, L) for univariate
        if x.ndim == 3:
            x = x.squeeze(-1)

        # RevIN
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.std(dim=1, keepdim=True) + self.eps).detach()
        x_norm = (x - mean) / std

        if self.use_gtb:
            x_norm = self.gtb(x_norm)

        # Conv1d expects (B, C, L): treat as 1 channel with L timesteps
        h = x_norm.unsqueeze(1)  # (B, 1, L)
        h = self.network(h)      # (B, hidden_channels, L)
        h = self.pool(h).squeeze(-1)  # (B, hidden_channels)
        out = self.output_proj(h) + self.residual_proj(x_norm)

        # RevIN denormalize
        out = out * std + mean
        return out


class TCN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: Any,
                 out_features: Any,
                 n_vars: int = 1,
                 kernel_size: int = 3,
                 dropout: float=0.15,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 loss_fn='mae',
                 num_levels: int = None,
                 hidden_channels: int = None,
                 weight_decay: float = 1e-4,
                 channel_mixing: bool = True,
                 use_gtb: bool = False,
                 gtb_d_model: int = 64,
                 routing_mode: str = 'static'
                 ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device
        self.num_levels = num_levels
        self.hidden_channels = hidden_channels
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        self.use_gtb = use_gtb
        self.gtb_d_model = gtb_d_model
        self.routing_mode = routing_mode
        # this sentence needs to be the last one
        super(TCN, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = TemporalConvNet(
            self.in_features, self.out_features, kernel_size=self.kernel_size,
            dropout=self.dropout, num_levels=self.num_levels,
            hidden_channels=self.hidden_channels, device=self.device,
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