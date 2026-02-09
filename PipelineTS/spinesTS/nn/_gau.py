from typing import Any, Union

import torch
from torch import nn

from PipelineTS.spinesTS.layers import GAU, PositionalEncoding, MultivariateWrapper
from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin


class GAUBlock(nn.Module):
    def __init__(self, in_features, level=2, query_key_dim=512, expansion_factor=4., skip_connect=True, dropout=0.2, **kwargs):
        super(GAUBlock, self).__init__()
        self.gau = nn.ModuleList([
            nn.Sequential(
                PositionalEncoding(in_features, add_x=True),
                GAU(in_features, query_key_dim=query_key_dim, expansion_factor=expansion_factor, 
                    skip_connect=skip_connect, dropout=dropout, **kwargs)
            )
            for i in range(level)
            ])
        self.level = level

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        for i in self.gau:
            x = i(x)

        return x


class GAUBase(nn.Module):
    def __init__(self, in_shapes, out_features, level=2, dropout=0.2):
        super(GAUBase, self).__init__()
        self.in_shapes_type = type(in_shapes)

        self.in_features, self.out_features = \
            in_shapes[-1] if self.in_shapes_type == tuple else in_shapes, out_features

        self.gau = GAUBlock(self.in_features, level=level, dropout=dropout)
        
        # 添加多头注意力机制 - dynamically choose valid num_heads
        num_heads = 1
        for h in [8, 4, 2, 1]:
            if self.in_features % h == 0:
                num_heads = h
                break
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.in_features, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 添加层归一化
        self.norm1 = nn.LayerNorm(self.in_features)
        self.norm2 = nn.LayerNorm(self.in_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        ln_layer_in_fea = in_shapes[0] * in_shapes[1] if self.in_shapes_type == tuple else self.in_features

        # 使用更复杂的输出层
        self.linear = nn.Sequential(
            nn.Linear(ln_layer_in_fea, ln_layer_in_fea*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ln_layer_in_fea*2, out_features)
        )

    def forward(self, x):
        # GAU处理
        gau_output = self.gau(x)
        
        # 对于注意力机制，确保输入形状正确
        if gau_output.ndim == 3:
            # 应用多头注意力并添加残差连接
            attn_output, _ = self.multihead_attn(gau_output, gau_output, gau_output)
            gau_output = gau_output + self.dropout(attn_output)
            gau_output = self.norm1(gau_output)
            
            # 将3D张量压缩为2D用于线性层
            # 如果batch维度为1，需要特殊处理
            if gau_output.shape[0] == 1:
                gau_output = gau_output.reshape(1, -1)
            else:
                gau_output = gau_output.reshape(gau_output.shape[0], -1)
        
        # 如果输入已经是2D张量
        elif x.ndim == 2:
            gau_output = gau_output.reshape(gau_output.shape[0], -1)

        # 应用线性层并归一化
        output = self.linear(gau_output)
        return output


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
