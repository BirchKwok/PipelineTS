from typing import Any, Union

import torch
import torch.nn as nn

from PipelineTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.models.nn._modern_ts_specs import MODERN_TS_MODEL_SPECS
from PipelineTS.models.nn.layers import (
    MultivariateWrapper, GlobalTemporalBlock, ModernEncoderBlock,
    ModernMixerBlock, ModernSeriesTokenizer, TemporalProjectionHead,
)


class ModernTSBackbone(nn.Module):
    def __init__(self, in_features, out_features, variant, d_model=64, n_heads=4,
                 e_layers=2, d_ff=128, patch_len=4, stride=None, dropout=0.1,
                 use_revin=True, moving_avg=3, top_k=3, use_gtb=False,
                 gtb_d_model=64, routing_mode='static'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.variant = variant
        self.d_model = int(d_model)
        self.use_revin = use_revin
        self.eps = 1e-5
        self.use_gtb = use_gtb

        n_heads = self._resolve_heads(self.d_model, n_heads)
        d_ff = max(int(d_ff), self.d_model)

        self.tokenizer = ModernSeriesTokenizer(
            in_features=in_features,
            d_model=self.d_model,
            patch_len=patch_len,
            stride=stride,
            moving_avg=moving_avg,
            top_k=top_k,
        )
        self.encoder = nn.ModuleList([
            ModernEncoderBlock(self.d_model, n_heads, d_ff, dropout)
            for _ in range(max(1, int(e_layers)))
        ])
        self.mixers = nn.ModuleList([
            ModernMixerBlock(self.d_model, d_ff, dropout)
            for _ in range(max(1, int(e_layers)))
        ])
        self.gru = nn.GRU(self.d_model, self.d_model, num_layers=1, batch_first=True)
        self.projection = TemporalProjectionHead(in_features, out_features, self.d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(self.d_model)

        if use_gtb:
            self.gtb = GlobalTemporalBlock(in_features, d_model=gtb_d_model, dropout=dropout, routing_mode=routing_mode)

    @staticmethod
    def _resolve_heads(d_model, n_heads):
        for h in [int(n_heads), 8, 4, 2, 1]:
            if h > 0 and d_model % h == 0:
                return h
        return 1

    def _normalize(self, x):
        if not self.use_revin:
            return x, torch.zeros_like(x[:, :1]), torch.ones_like(x[:, :1])
        mean = x.mean(dim=1, keepdim=True).detach()
        var = torch.var(x - mean, dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps).detach()
        return (x - mean) / std, mean, std

    def _encode(self, tokens, mixer=False):
        h = self.dropout(tokens)
        layers = self.mixers if mixer else self.encoder
        for layer in layers:
            h = layer(h)
        return self.final_norm(h)

    def _tokens(self, x_norm, trend, seasonal):
        extra = None
        mixer = False

        if self.variant in {'time_mixer', 'etsformer', 'autoformer', 'lightts'}:
            source = torch.cat([seasonal[:, :self.in_features // 2], trend[:, self.in_features // 2:]], dim=1)
            if source.shape[1] != self.in_features:
                source = x_norm
            tokens = self.tokenizer.sequence(source)
            extra = self.projection.trend_proj(trend)
            mixer = self.variant in {'time_mixer', 'lightts'}
        elif self.variant in {'timesnet', 'fedformer', 'timefilter', 'msgnet'}:
            filtered = self.tokenizer.frequency_filter(seasonal)
            tokens = self.tokenizer.conv(filtered if self.variant in {'timesnet', 'msgnet'} else filtered + trend)
            extra = self.projection.level_proj(trend) if self.variant in {'fedformer', 'msgnet'} else None
        elif self.variant in {'patchtst', 'multi_patch_former', 'wpmixer', 'timexer'}:
            tokens = self.tokenizer.multi_patch(x_norm) if self.variant in {'multi_patch_former', 'wpmixer', 'timexer'} else self.tokenizer.patch(x_norm)
            mixer = self.variant == 'wpmixer'
            if self.variant == 'timexer':
                tokens = tokens + self.tokenizer.sequence(x_norm).mean(dim=1, keepdim=True)
        elif self.variant == 'tsmixer':
            tokens = self.tokenizer.sequence(x_norm)
            mixer = True
        elif self.variant in {'seg_rnn', 'tirex'}:
            tokens = self.tokenizer.patch(x_norm, stride=self.tokenizer.patch_len)
            tokens, _ = self.gru(tokens)
            extra = self.projection.trend_proj(trend) if self.variant == 'tirex' else None
        else:
            source = x_norm
            if self.variant == 'nonstationary_transformer':
                source = x_norm + 0.5 * seasonal
            elif self.variant in {'informer', 'reformer', 'pyraformer'}:
                source = self.tokenizer.frequency_filter(x_norm)
            tokens = self.tokenizer.sequence(source)

        return tokens, mixer, extra

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(-1) if x.shape[-1] == 1 else x.reshape(x.shape[0], -1)
        x_norm, mean, std = self._normalize(x)
        if self.use_gtb:
            x_norm = self.gtb(x_norm)

        trend = self.tokenizer.moving_average(x_norm)
        seasonal = x_norm - trend
        tokens, mixer, extra = self._tokens(x_norm, trend, seasonal)
        h = self._encode(tokens, mixer=mixer)
        return self.projection(h, x_norm, mean, std, use_revin=self.use_revin, extra=extra)


class _ModernTSModel(TorchModelMixin, ForecastingMixin):
    variant = 'modern_ts'

    def __init__(self, in_features: int, out_features: int, n_vars: int = 1,
                 d_model: int = 64, n_heads: int = 4, nhead: int = None,
                 e_layers: int = 2, num_encoder_layers: int = None,
                 d_ff: int = 128, dim_feedforward: int = None,
                 patch_len: int = 4, stride: int = None, dropout: float = 0.1,
                 use_revin: bool = True, moving_avg: int = 3, top_k: int = 3,
                 loss_fn='huber', learning_rate: float = 0.001,
                 random_seed: int = 42, device='auto', weight_decay: float = 1e-4,
                 channel_mixing: bool = True, use_gtb: bool = False,
                 gtb_d_model: int = 64, routing_mode: str = 'static', **kwargs) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_heads = n_heads if nhead is None else nhead
        self.e_layers = e_layers if num_encoder_layers is None else num_encoder_layers
        self.d_ff = d_ff if dim_feedforward is None else dim_feedforward
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.use_revin = use_revin
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay
        self.channel_mixing = channel_mixing
        self.use_gtb = use_gtb
        self.gtb_d_model = gtb_d_model
        self.routing_mode = routing_mode
        super().__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        backbone = ModernTSBackbone(
            in_features=self.in_features,
            out_features=self.out_features,
            variant=self.variant,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout,
            use_revin=self.use_revin,
            moving_avg=self.moving_avg,
            top_k=self.top_k,
            use_gtb=self.use_gtb,
            gtb_d_model=self.gtb_d_model,
            routing_mode=self.routing_mode,
        )
        if self.n_vars > 1:
            model = MultivariateWrapper(backbone, self.n_vars, self.out_features, channel_mixing=self.channel_mixing)
        else:
            model = backbone
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return model, self.loss_fn, optimizer

    def fit(self, X_train: Any, y_train: Any, epochs: int = 1000,
            batch_size: Union[str, int] = 'auto', eval_set: Any = None,
            monitor: str = 'val_loss', min_delta: int = 0, patience: int = 10,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10, lr_factor: float = 0.7,
            restore_best_weights: bool = True, verbose: bool = True,
            loss_type='min', **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name, monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience, lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)


def _build_modern_ts_class(class_name, variant):
    return type(class_name, (_ModernTSModel,), {'variant': variant, '__module__': __name__})


for _spec in MODERN_TS_MODEL_SPECS:
    globals()[_spec.backbone_class] = _build_modern_ts_class(_spec.backbone_class, _spec.variant)


__all__ = ['ModernTSBackbone'] + [spec.backbone_class for spec in MODERN_TS_MODEL_SPECS]
