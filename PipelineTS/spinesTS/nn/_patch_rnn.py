from typing import Any, Union

import torch
from spinesUtils.asserts import raise_if_not
from torch import nn

from PipelineTS.spinesTS.base import TorchModelMixin, ForecastingMixin
from PipelineTS.spinesTS.layers._position_encoder import LearnablePositionalEncoding
from PipelineTS.spinesTS.nn.utils import get_weight_norm


class SegmentationBlock(nn.Module):
    """将输入数据分割成多个块，每个块的大小为window_size"""

    def __init__(self, in_features, kernel_size=4, device=None):
        super(SegmentationBlock, self).__init__()
        weight_norm = get_weight_norm(device)

        self.kernel_size = kernel_size

        self.encoder = nn.Sequential(
            LearnablePositionalEncoding(self.kernel_size),
            weight_norm(nn.Linear(self.kernel_size, self.kernel_size)),
            nn.GELU(),
            nn.BatchNorm1d(in_features - kernel_size + 1)
        )

    def forward(self, x):
        raise_if_not(ValueError, x.ndim == 2, "x must be a 2-dimensional tensor")

        x = x.unfold(dimension=-1, size=self.kernel_size, step=1)

        return self.encoder(x)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, device=None):
        super(FCBlock, self).__init__()
        weight_norm = get_weight_norm(device)

        self.fc_layer = nn.Sequential(
            weight_norm(nn.Linear(in_features, 512)),
            nn.GELU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(512, out_features))
        )

    def forward(self, x):
        return self.fc_layer(x)


# class PatchRNNBlock(nn.Module):
#     def __init__(self, in_features, out_features, kernel_size=4, dropout=0.1, device=None):
#         super(PatchRNNBlock, self).__init__()
#         self.splitter = SegmentationBlock(in_features, kernel_size, device=device)
#
#         self.encoder_rnn = nn.LSTM(kernel_size, kernel_size // 2, num_layers=2,
#                                    bias=False, bidirectional=True, batch_first=True)
#
#         seq_len_adjusted = in_features - kernel_size + 1
#         fc_input_features = seq_len_adjusted * kernel_size
#         self.decoder = FCBlock(fc_input_features, out_features, dropout=dropout, device=device)
#
#         self.layer_norm = nn.LayerNorm(kernel_size)
#
#     def forward(self, x):
#         x = self.splitter(x)
#
#         output, _ = self.encoder_rnn(x)
#
#         # 保持批次和时间步的顺序，将output转换为二维进行LayerNorm
#         batch_size, seq_len, features = output.shape
#         output_2d = output.reshape(batch_size * seq_len, features)
#
#         # 对2D形式的output进行Layer Norm
#         output_norm = self.layer_norm(output_2d)
#
#         # 将output恢复到原来的3D形状以进行后续操作
#         output = output_norm.view(batch_size, seq_len, features)
#
#         output += x  # 恢复残差连接，注意这里假设x已经是正确的形状
#
#         res = self.decoder(output.reshape(output.size(0), -1))
#
#         return res
#

class PatchRNNBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, dropout=0.1, device=None, multi_steps=False):
        super(PatchRNNBlock, self).__init__()
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.device = device

        self.splitter = SegmentationBlock(in_features, kernel_size, device=device)
        self.encoder_rnn = nn.LSTM(kernel_size, kernel_size // 2, num_layers=2,
                                   bias=False, bidirectional=True, batch_first=True)
        seq_len_adjusted = in_features - kernel_size + 1
        fc_input_features = seq_len_adjusted * kernel_size if not multi_steps else kernel_size
        self.decoder = FCBlock(fc_input_features, 1 if multi_steps else out_features,
                               dropout=dropout, device=device) # Assuming each step predicts one value
        self.layer_norm = nn.LayerNorm(kernel_size)

        self.multi_steps = multi_steps

    def forward(self, x):
        x = self.splitter(x)
        output, (h, c) = self.encoder_rnn(x)  # Preserve the last state of RNN for next step generation

        batch_size, seq_len, features = output.shape
        output_2d = output.reshape(batch_size * seq_len, features)
        output_norm = self.layer_norm(output_2d)
        output = output_norm.view(batch_size, seq_len, features)

        output += x  # Assuming x is already in the correct shape for residual connection

        if not self.multi_steps:
            res = self.decoder(output.reshape(output.size(0), -1))
            return res

        # Initialize hidden and cell state for generation
        h_gen, c_gen = h, c
        # Initialize feedback input as the last output step
        feedback_input = output[:, -1, :].unsqueeze(1)

        outputs = []
        for _ in range(self.out_features):
            # Generate next step output
            output_gen, (h_gen, c_gen) = self.encoder_rnn(feedback_input, (h_gen, c_gen))
            output_gen_2d = output_gen.reshape(output_gen.size(0), -1)
            output_gen_norm = self.layer_norm(output_gen_2d)
            res = self.decoder(output_gen_norm)
            outputs.append(res.squeeze(-1))  # Remove the last dimension to ensure 2D output
            feedback_input = output_gen

        # Concatenate along the sequence length dimension to form a 2D output
        outputs = torch.stack(outputs, dim=1)

        return outputs


class PatchRNN(TorchModelMixin, ForecastingMixin):
    """长序列预测模型
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size=4,
                 dropout=0.1,
                 loss_fn='mae',
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto',
                 multi_steps=True
                 ) -> None:
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.loss_fn_name = loss_fn
        self.multi_steps = multi_steps

        # this sentence needs to be the last one
        super(PatchRNN, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self) -> tuple:
        model = PatchRNNBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            device=self.device,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            multi_steps=self.multi_steps
        )
        loss_fn = self.loss_fn
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, loss_fn, optimizer

    def fit(self,
            X_train: Any,
            y_train: Any,
            epochs: int = 1000,
            batch_size: Union[str, int] = 'auto',
            eval_set: Any = None,
            monitor: str = 'val_loss',
            min_delta: int = 0,
            patience: int = 100,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.7,
            restore_best_weights: bool = True,
            loss_type='min',
            verbose: bool = True,
            **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)
