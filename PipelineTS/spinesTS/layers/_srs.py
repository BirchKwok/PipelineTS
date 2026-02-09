"""Selective Representation Space (SRS) Module.

A novel plug-and-play module for irregular time series forecasting that:
1. Front-loads irregularity handling via multi-scale adaptive patching
2. Builds a selective representation space to choose the most informative patches
3. Dynamically recombines selected patches via cross-attention

Architecture:
    Input (B, L, C) → RevIN → MultiScalePatchEmbedding → SRS → Output (B, K, d_model)

The SRS module can be used standalone or as a plug-in enhancement for
existing patch-based models (e.g., PatchRNN).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization for non-stationary time series.

    Normalizes input by per-instance statistics and provides denormalization
    for the output. Handles both many-to-one and many-to-many prediction.

    Reference: Kim et al., "Reversible Instance Normalization for
    Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def normalize(self, x):
        """Normalize input. x: (B, L, C)"""
        self._mean = x.mean(dim=1, keepdim=True).detach()  # (B, 1, C)
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()  # (B, 1, C)
        return (x - self._mean) / self._std

    def denormalize(self, x, target_idx=None):
        """Denormalize output.

        Args:
            x: (B, pred_len) for many-to-one, or (B, pred_len, n_targets) for many-to-many
            target_idx: int, index of target variable for many-to-one. None for many-to-many.
        """
        if self._mean is None:
            return x

        C = self._mean.shape[-1]

        if target_idx is not None:
            # Many-to-one: use specific variable's stats
            idx = target_idx if target_idx >= 0 else C + target_idx
            mean = self._mean[:, 0, idx]  # (B,)
            std = self._std[:, 0, idx]  # (B,)
            if x.ndim == 2:
                return x * std.unsqueeze(1) + mean.unsqueeze(1)
            else:
                return x * std.unsqueeze(1).unsqueeze(-1) + mean.unsqueeze(1).unsqueeze(-1)
        else:
            # Many-to-many
            n = x.shape[-1] if x.ndim == 3 else 1
            if x.ndim == 2:
                x = x.unsqueeze(-1)
                out = x * self._std[:, :, :n] + self._mean[:, :, :n]
                return out.squeeze(-1)
            return x * self._std[:, :, :n] + self._mean[:, :, :n]


class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale patch embedding for irregular time series.

    Generates patches at multiple temporal scales to capture patterns at
    different granularities. Smaller scales capture local variations and
    rapid changes; larger scales capture trends and slow dynamics.

    This multi-scale approach inherently handles irregularity by providing
    redundant coverage at multiple resolutions — even if some patches
    at one scale are corrupted by irregular sampling, patches at other
    scales can compensate.

    Args:
        n_vars: Number of input variables (channels).
        d_model: Embedding dimension for each patch.
        patch_sizes: List of patch (kernel) sizes for multi-scale extraction.
        stride_ratio: Stride as a fraction of patch size (controls overlap).
    """

    def __init__(self, n_vars, d_model, patch_sizes, stride_ratio=0.5):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.n_scales = len(patch_sizes)

        self.patch_convs = nn.ModuleList()
        for ps in patch_sizes:
            stride = max(1, int(ps * stride_ratio))
            self.patch_convs.append(nn.Sequential(
                nn.Conv1d(n_vars, d_model, kernel_size=ps, stride=stride, padding=ps // 2),
                nn.GELU(),
                nn.BatchNorm1d(d_model)
            ))

        # Scale-level embedding to distinguish patches from different scales
        self.scale_embed = nn.Parameter(torch.randn(self.n_scales, 1, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (B, L, C) — time series with C variables
        Returns:
            patches: (B, N_total, d_model) — all patches from all scales
            patch_counts: list[int] — number of patches per scale
        """
        x_t = x.transpose(1, 2)  # (B, C, L) for Conv1d

        all_patches = []
        patch_counts = []
        for i, conv in enumerate(self.patch_convs):
            p = conv(x_t)           # (B, d_model, N_i)
            p = p.transpose(1, 2)   # (B, N_i, d_model)
            p = p + self.scale_embed[i]
            all_patches.append(p)
            patch_counts.append(p.shape[1])

        patches = torch.cat(all_patches, dim=1)  # (B, N_total, d_model)
        return patches, patch_counts


class TemporalDecayBias(nn.Module):
    """Learnable temporal position bias for patch importance scoring.

    Generates a smooth, position-dependent bias that allows the model
    to learn temporal preferences (e.g., favoring recent patches for
    short-term forecasting, or balancing recent and historical for
    long-term forecasting).
    """

    def __init__(self, max_patches=1024):
        super().__init__()
        self.decay_rate = nn.Parameter(torch.tensor(0.0))
        self.max_patches = max_patches

    def forward(self, n_patches, device):
        """Returns bias of shape (1, n_patches)."""
        # Learnable exponential decay: recent patches get higher bias
        rate = torch.sigmoid(self.decay_rate)  # [0, 1]
        positions = torch.arange(n_patches, device=device, dtype=torch.float32)
        positions = positions / max(n_patches - 1, 1)  # normalize to [0, 1]
        bias = rate * positions  # linear ramp (recent = higher)
        return bias.unsqueeze(0)  # (1, N)


class SelectiveRepresentationSpace(nn.Module):
    """SRS: Selective Representation Space Module.

    Core innovation for irregular time series forecasting. Adaptively selects
    the most informative patches from a multi-scale patch set and dynamically
    recombines them through cross-attention.

    Process:
    1. **Importance Scoring**: An MLP scores each patch's informativeness,
       augmented by a learnable temporal decay bias.
    2. **Differentiable Top-K**: Selects the top-K most informative patches
       using hard selection (forward) with soft gradient flow (backward)
       via importance-weighted representations.
    3. **Cross-Attention Recombination**: Selected patches attend back to
       the full patch set, enriching their representations with global context.

    This module is designed as a plug-and-play component that can enhance
    any patch-based time series model.

    Args:
        d_model: Patch embedding dimension.
        n_heads: Number of attention heads for cross-attention.
        top_k_ratio: Fraction of patches to select (0.0 to 1.0).
        dropout: Dropout rate.
    """

    def __init__(self, d_model, n_heads=4, top_k_ratio=0.5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.top_k_ratio = top_k_ratio

        # Importance scoring network
        self.score_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # Temporal decay bias
        self.temporal_bias = TemporalDecayBias()

        # Learnable temperature for selection sharpness
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # Cross-attention: selected patches attend to ALL original patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward for post-recombination processing
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, patches):
        """
        Args:
            patches: (B, N, d_model) — patch embeddings from MultiScalePatchEmbedding
        Returns:
            selected: (B, K, d_model) — selected and recombined patch representations
            selection_weights: (B, K) — normalized importance weights of selected patches
        """
        B, N, D = patches.shape
        K = max(1, int(N * self.top_k_ratio))

        # Step 1: Score each patch
        scores = self.score_net(patches).squeeze(-1)  # (B, N)

        # Add temporal position bias
        temporal_bias = self.temporal_bias(N, patches.device)  # (1, N)
        scores = scores + temporal_bias

        # Step 2: Differentiable top-k selection
        temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)

        # Hard top-k selection
        top_scores, top_indices = torch.topk(scores, K, dim=-1)  # (B, K)

        # Soft importance weights for gradient flow
        selection_weights = F.softmax(top_scores / temperature, dim=-1)  # (B, K)

        # Gather selected patches
        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
        selected = torch.gather(patches, 1, idx_expanded)  # (B, K, D)

        # Weight selected patches by importance
        selected = selected * selection_weights.unsqueeze(-1)  # (B, K, D)

        # Step 3: Cross-attention recombination
        # Selected patches attend to ALL original patches for context enrichment
        attended = self.cross_attn(selected, patches, patches)[0]  # (B, K, D)
        selected = self.norm1(selected + attended)

        # Feed-forward refinement
        selected = self.norm2(selected + self.ffn(selected))

        return selected, selection_weights


class SRSBlock(nn.Module):
    """Complete Selective Representation Space block (plug-and-play).

    Combines multi-scale adaptive patching with selective representation
    to create high-quality, regularized patch representations from
    potentially irregular time series data.

    Can be used:
    1. As a standalone preprocessing module for any downstream model
    2. As part of SRSNet (SRS + MLP head)
    3. As a drop-in enhancement for SegmentationBlock in PatchRNN

    Args:
        n_vars: Number of input variables.
        d_model: Embedding dimension for patches.
        patch_sizes: List of patch sizes for multi-scale patching. Auto-determined if None.
        seq_len: Input sequence length (used for auto patch size computation).
        n_heads: Number of attention heads in SRS cross-attention.
        top_k_ratio: Fraction of patches to select (0.0 to 1.0).
        dropout: Dropout rate.
        stride_ratio: Stride as fraction of patch size (controls overlap).
    """

    def __init__(self, n_vars, d_model=64, patch_sizes=None, seq_len=None,
                 n_heads=4, top_k_ratio=0.5, dropout=0.1, stride_ratio=0.5):
        super().__init__()

        # Auto-determine patch sizes if not provided
        if patch_sizes is None:
            if seq_len is not None:
                patch_sizes = self._auto_patch_sizes(seq_len)
            else:
                patch_sizes = [4, 8, 16]

        # Validate n_heads divides d_model
        actual_n_heads = n_heads
        for h in [n_heads, 4, 2, 1]:
            if d_model % h == 0:
                actual_n_heads = h
                break

        self.patch_embed = MultiScalePatchEmbedding(
            n_vars, d_model, patch_sizes, stride_ratio
        )

        # Learnable positional encoding for patches
        max_patches = 1024
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        # SRS module
        self.srs = SelectiveRepresentationSpace(
            d_model, actual_n_heads, top_k_ratio, dropout
        )

    @staticmethod
    def _auto_patch_sizes(seq_len):
        """Auto-determine good patch sizes based on sequence length."""
        base = max(2, seq_len // 8)
        sizes = [base]
        if base * 2 <= seq_len:
            sizes.append(base * 2)
        if base * 4 <= seq_len and len(sizes) < 3:
            sizes.append(min(base * 4, seq_len // 2))
        if len(sizes) < 2:
            sizes = [max(2, seq_len // 4), max(4, seq_len // 2)]
        return sizes

    def forward(self, x):
        """
        Args:
            x: (B, L, C) — time series input (already normalized if using RevIN externally)
        Returns:
            selected: (B, K, d_model) — selected patch representations
            weights: (B, K) — selection importance weights
        """
        patches, _ = self.patch_embed(x)  # (B, N, d_model)
        N = patches.shape[1]

        # Add positional encoding
        patches = patches + self.pos_embed[:, :N, :]

        # Apply SRS
        selected, weights = self.srs(patches)

        return selected, weights
