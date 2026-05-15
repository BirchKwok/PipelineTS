"""Lightweight RWKV-inspired blocks for time series forecasting.

Efficient TSMixer-style architecture adapted from RWKV concepts:
- ALL operations are nn.Linear (fast single-BLAS matmuls)
- Temporal mixing via linear projection across time dimension
- Channel mixing via SiLU-gated linear unit (GLU)
- No Conv1d, no depthwise ops, no sequential loops
- O(T) time, constant memory

Key insight: in lookback-window forecasting the entire window is
observed, so causal masking is unnecessary. Linear temporal mixing
(a single matmul across time positions) is both faster and more
expressive than causal convolutions or sequential recurrences.

Reference: Inspired by RWKV (Peng et al., 2023), TSMixer (Google, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTimeMixing(nn.Module):
    """Linear temporal mixing with gating.

    Mixes information across time positions via a learnable linear
    projection — equivalent to a fully-connected layer along the
    time axis. This is a single fast matrix multiply.

    The gate allows the model to selectively pass temporal information.

    Args:
        seq_len: Fixed input sequence length.
        d_model: Model dimension (used for gating).
    """

    def __init__(self, seq_len, d_model):
        super().__init__()
        # Temporal mixing: single Linear across time positions
        self.temporal_proj = nn.Linear(seq_len, seq_len)
        # Gate: controls how much temporal mixing to apply
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (B, T, C) → (B, T, C)"""
        # Mix across time: transpose → linear → transpose
        h = self.temporal_proj(x.transpose(1, 2)).transpose(1, 2)
        # Gating
        g = torch.sigmoid(self.gate(x))
        return g * h


class LightChannelMixing(nn.Module):
    """Lightweight SiLU-gated channel mixing (GLU-style).

    Uses SiLU(gate) * up projection — parameter-efficient and fast.

    Args:
        d_model: Model dimension.
        expand_ratio: Hidden dimension expansion ratio.
    """

    def __init__(self, d_model, expand_ratio=2.0):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.W_gate = nn.Linear(d_model, hidden, bias=False)
        self.W_up = nn.Linear(d_model, hidden, bias=False)
        self.W_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        """x: (B, T, C) → (B, T, C)"""
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class RWKVBlock(nn.Module):
    """Temporal Mixer block: GatedTimeMixing + GLU ChannelMixing.

    Each block:
    1. LayerNorm → GatedTimeMixing → residual (temporal)
    2. LayerNorm → LightChannelMixing → residual (channel)

    All operations are nn.Linear — maximally fast on both CPU and GPU.

    Args:
        seq_len: Input sequence length.
        d_model: Model dimension.
        expand_ratio: FFN expansion ratio.
        dropout: Dropout rate.
    """

    def __init__(self, seq_len, d_model, expand_ratio=2.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.time_mixing = GatedTimeMixing(seq_len, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mixing = LightChannelMixing(d_model, expand_ratio)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, C) → (B, T, C)"""
        x = x + self.drop1(self.time_mixing(self.ln1(x)))
        x = x + self.drop2(self.channel_mixing(self.ln2(x)))
        return x


class RWKVEncoder(nn.Module):
    """Lightweight temporal mixer encoder for time series.

    Stacks multiple RWKVBlocks — all operations are nn.Linear.

    Args:
        seq_len: Input sequence length.
        d_model: Model dimension.
        n_blocks: Number of blocks to stack.
        expand_ratio: FFN expansion ratio.
        dropout: Dropout rate.
    """

    def __init__(self, seq_len, d_model, n_blocks=3, expand_ratio=2.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            RWKVBlock(seq_len, d_model, expand_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (B, T, C) → (B, T, C)"""
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)
