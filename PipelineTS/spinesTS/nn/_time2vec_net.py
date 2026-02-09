import torch
from torch import nn

from PipelineTS.spinesTS.layers import Time2Vec, MultivariateWrapper
from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.nn.utils import get_weight_norm

# in case of using MPS
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


class T2V(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, num_layers=2, device=None):
        super(T2V, self).__init__()
        weight_norm = get_weight_norm(device)

        self.in_features, self.out_features = in_features, out_features
        t2v_out_dim = in_features * 2 - 1
        self.t2v = Time2Vec(self.in_features, in_features)

        self.lstm = nn.LSTM(
            t2v_out_dim, in_features,
            batch_first=True, bidirectional=True, bias=False,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.
        )

        lstm_out_dim = in_features * 2  # bidirectional
        self.norm = nn.LayerNorm(lstm_out_dim)

        # Residual projection from input to match lstm output dim
        self.residual_proj = nn.Linear(in_features, lstm_out_dim)

        # Richer output head
        hidden_dim = min(max(in_features * 2, 64), 256)
        self.output_head = nn.Sequential(
            weight_norm(nn.Linear(lstm_out_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hidden_dim, out_features))
        )

    def forward(self, x):
        residual = x
        x = self.t2v(x)
        output, (h, c) = self.lstm(x.unsqueeze(1))

        # Handle squeeze safely for batch_size=1
        if output.ndim == 3:
            output = output.squeeze(1)
        if output.ndim == 1:
            output = output.unsqueeze(0)

        # Residual connection
        output = self.norm(output + self.residual_proj(residual))

        return self.output_head(output)


class Time2VecNet(TorchModelMixin, ForecastingMixin):
    def __init__(self, in_features, out_features, n_vars=1, learning_rate=0.001,
                 random_seed=42, device='auto', loss_fn='mae',
                 dropout=0.1, num_layers=2, weight_decay=1e-4,
                 channel_mixing=True):
        self.in_features, self.out_features = in_features, out_features
        self.n_vars = n_vars
        self.learning_rate = learning_rate
        self.device = device
        self.loss_fn_name = loss_fn
        self.dropout = dropout
        self.num_layers = num_layers
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        # this sentence needs to be the last one
        super(Time2VecNet, self).__init__(random_seed, device=device, loss_fn=loss_fn)

    def call(self):
        backbone = T2V(
            self.in_features, self.out_features,
            dropout=self.dropout, num_layers=self.num_layers,
            device=self.device
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
