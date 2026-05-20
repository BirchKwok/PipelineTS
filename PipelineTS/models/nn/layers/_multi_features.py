import torch
from spinesUtils.asserts import raise_if_not
from torch import nn


class MultivariateWrapper(nn.Module):
    """Channel-independent wrapper for multivariate time series.

    Processes each variable through a shared backbone independently,
    then optionally applies cross-variable mixing (like PatchTST approach).

    Parameters
    ----------
    backbone : nn.Module
        The univariate model backbone that takes (batch, seq_len) input
        and returns (batch, pred_len) output.
    n_vars : int
        Number of input/output variables.
    out_features : int
        Prediction horizon length.
    channel_mixing : bool
        Whether to apply a learnable cross-variable mixing layer.
    """

    def __init__(self, backbone, n_vars, out_features, channel_mixing=True):
        super(MultivariateWrapper, self).__init__()
        self.backbone = backbone
        self.n_vars = n_vars
        self.out_features = out_features

        if channel_mixing and n_vars > 1:
            self.mixer = nn.Sequential(
                nn.Linear(n_vars, n_vars * 2),
                nn.GELU(),
                nn.Linear(n_vars * 2, n_vars)
            )
        else:
            self.mixer = None

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        if x.ndim == 2:
            # Univariate fallback: (batch, seq_len) -> (batch, pred_len)
            return self.backbone(x)

        B, L, N = x.shape
        # Process each variable independently through shared backbone
        # Reshape: (B, L, N) -> (B*N, L)
        x = x.permute(0, 2, 1).reshape(B * N, L)
        out = self.backbone(x)  # (B*N, pred_len)
        # Reshape back: (B*N, pred_len) -> (B, pred_len, N)
        out = out.reshape(B, N, -1).permute(0, 2, 1)

        # Optional cross-variable mixing with residual
        if self.mixer is not None:
            out = out + self.mixer(out)

        return out


class SeriesRecombinationLayer(nn.Module):
    def __init__(self, in_shapes, out_features=128, rnn_layers=1, dropout=0.):
        raise_if_not(ValueError, isinstance(in_shapes, tuple) and len(in_shapes) == 2,
                     "in_shapes must be a tuple with length 2")
        super(SeriesRecombinationLayer, self).__init__()
        (self.rows, self.cols), self.out_features = in_shapes, out_features

        self.encoder_rnn = nn.GRU(self.rows, out_features, batch_first=True, num_layers=rnn_layers,
                                  dropout=dropout if rnn_layers > 1 else 0.)
        self.decoder_rnn = nn.GRU(out_features, out_features, batch_first=True, num_layers=rnn_layers,
                                  dropout=dropout if rnn_layers > 1 else 0.)

        self.out = nn.Linear(self.cols * out_features, out_features)

    def forward(self, x):
        raise_if_not(ValueError, x.ndim == 3, "x must be a 3-dimensional tensor")

        x = x.permute((0, 2, 1))
        x, h = self.encoder_rnn(x)
        x, h = self.decoder_rnn(x, h)
        x = x.permute((0, 2, 1))

        return self.out(x.reshape((-1, self.cols * self.out_features)))  # (batch_size, out_features)

