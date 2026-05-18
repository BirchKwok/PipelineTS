"""Global Temporal Block (GTB) — A pluggable enhancement block for time series NN models.

Combines three key innovations as "experts":

1. Frequency-Guided 2D Convolution (TimesNet + Image Generation):
   - FFT to discover dominant periods in the signal
   - Reshape 1D→2D by period, creating a 2D "image" of the time series
   - Apply 2D conv for global intra/inter-period pattern capture
   Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling", ICLR 2023.

2. Gated Linear Attention (LLM Efficiency):
   - O(N) linear attention with ELU+1 feature maps (vs O(N²) softmax)
   - Sigmoid gating for selective information flow (RWKV/Mamba-inspired)
   Reference: Katharopoulos et al., "Transformers are RNNs", ICML 2020.

3. SwiGLU Feed-Forward (LLM Architecture):
   - SiLU-gated linear unit for efficient channel mixing
   Reference: Shazeer, "GLU Variants Improve Transformer", 2020.

Routing modes:
- 'static':   Manual enable/disable via use_freq_mixing, use_attention, use_swiglu.
- 'adaptive': MoE-style learned Top-K sparse routing (DeepSeek-V2 inspired).
              A lightweight router network dynamically selects which experts to
              activate per sample. Includes load-balancing auxiliary loss to
              prevent routing collapse, and optional shared expert.

Usage:
    # Static mode (default, backward compatible)
    gtb = GlobalTemporalBlock(seq_len=16, d_model=64)

    # Adaptive MoE routing — auto-selects top-2 of 3 experts per sample
    gtb = GlobalTemporalBlock(seq_len=16, d_model=64, routing_mode='adaptive', top_k_experts=2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm — no mean subtraction, just scale normalization.
    From LLM literature (Zhang & Sennrich, 2019).
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _temporal_moving_average(x, kernel_size):
    L = x.shape[1]
    k = min(kernel_size, L if L % 2 == 1 else max(1, L - 1))
    if k <= 1:
        return x
    pad = k // 2
    x_t = x.transpose(1, 2)
    smooth = F.avg_pool1d(F.pad(x_t, (pad, pad), mode='replicate'), kernel_size=k, stride=1)
    return smooth.transpose(1, 2)


class FreqMixingBlock(nn.Module):
    """Frequency-guided 2D convolution for global temporal pattern capture.

    Inspired by TimesNet's 1D→2D transformation and image generation models'
    pixel-level global correlation modeling.

    Process:
    1. FFT to find top-k dominant periods
    2. For each period p, reshape (B, D, L) → (B, D, p, L//p) as 2D "image"
       - Rows = different cycles (inter-period)
       - Columns = positions within one cycle (intra-period)
    3. Apply 2D inception-style conv to capture:
       - Intra-period patterns (within one cycle) via horizontal conv
       - Inter-period patterns (across cycles) via vertical conv
       This is analogous to how image generation models capture
       pixel-level global correlations in a 2D spatial grid.
    4. Average across all period-based views
    """

    def __init__(self, d_model, top_k=3, conv_channels=None):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        conv_channels = conv_channels or max(16, d_model // 2)

        # Inception-style 2D conv: separable for both directions
        self.conv2d = nn.Sequential(
            nn.Conv2d(d_model, conv_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
            nn.Conv2d(conv_channels, d_model, kernel_size=1),
        )

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape

        # FFT to find dominant periods
        x_freq = torch.fft.rfft(x.mean(dim=-1), dim=1)  # (B, L//2+1)
        amp = x_freq.abs().mean(dim=0).clone()  # (L//2+1,) average across batch

        # Ignore DC component
        amp[0] = 0

        # Top-k frequencies → periods
        top_k = min(self.top_k, max(1, len(amp) - 1))
        top_amp, top_freq_idx = torch.topk(amp, top_k)

        periods = []
        period_weights = []
        for raw_amp, idx in zip(top_amp, top_freq_idx):
            freq = idx.item()
            if freq > 0:
                p = max(2, round(L / freq))
                if p not in periods and p <= L:
                    periods.append(p)
                    period_weights.append(raw_amp)

        if len(periods) == 0:
            periods = [max(2, L // 2)]
            period_weights = [x.new_tensor(1.0)]
        period_weights = torch.stack(period_weights)
        period_weights = period_weights / period_weights.sum().clamp_min(1e-6)

        # For each period, reshape to 2D and apply conv
        aggregated = torch.zeros_like(x)

        for p, period_weight in zip(periods, period_weights):
            n_cols = math.ceil(L / p)
            pad_len = p * n_cols - L
            if pad_len > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_len))  # pad time dim
            else:
                x_pad = x

            # (B, L_pad, D) → (B, D, p, n_cols)
            x_2d = x_pad.permute(0, 2, 1).reshape(B, D, p, n_cols)

            # 2D conv
            out_2d = self.conv2d(x_2d)  # (B, D, p, n_cols)

            # Reshape back to 1D and trim padding
            out_1d = out_2d.reshape(B, D, -1)[:, :, :L]  # (B, D, L)
            aggregated = aggregated + period_weight * out_1d.permute(0, 2, 1)  # (B, L, D)

        return aggregated


class GatedLinearAttention(nn.Module):
    """Linear attention with gating — O(N·d²) complexity instead of O(N²·d).

    Inspired by LLM efficiency mechanisms (RWKV, Mamba, RetNet).
    Uses ELU+1 feature map for non-negative attention weights, plus
    sigmoid gating for selective information flow.

    Complexity: O(N·d²) vs O(N²·d) for softmax attention.
    For d << N (typical in time series), this is significantly faster.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_head = d_model // n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_gate = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _feature_map(x):
        """ELU+1 feature map for non-negative attention."""
        return F.elu(x) + 1.0

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head

        qkv = self.W_qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)

        # Apply feature map for non-negative attention
        q = self._feature_map(q)
        k = self._feature_map(k)

        # Linear attention: O = φ(Q) @ (φ(K)^T @ V) / normalizer
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)       # (B, H, d, d)
        qkv_out = torch.einsum('bhnd,bhdm->bhnm', q, kv) # (B, H, L, d)

        # Normalizer for numerical stability
        k_sum = k.sum(dim=2)  # (B, H, d)
        normalizer = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1)  # (B, H, L, 1)
        normalizer = normalizer.clamp(min=1e-6)

        out = qkv_out / normalizer  # (B, H, L, d)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)

        # Sigmoid gating for selective information flow
        gate = torch.sigmoid(self.W_gate(x))  # (B, L, D)
        out = gate * self.W_out(out)

        return self.dropout(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward from LLaMA/PaLM architecture.

    SwiGLU(x) = W_down( SiLU(W_gate · x) ⊙ W_up · x )

    More expressive than standard FFN: the gating allows adaptive
    feature selection per channel per position.
    """

    def __init__(self, d_model, expand_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.W_gate = nn.Linear(d_model, hidden, bias=False)
        self.W_up = nn.Linear(d_model, hidden, bias=False)
        self.W_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W_down(F.silu(self.W_gate(x)) * self.W_up(x)))


class TrendExtractionBlock(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        smooth = _temporal_moving_average(x, self.kernel_size)
        gate = torch.sigmoid(self.gate(x))
        return self.dropout(gate * self.proj(smooth))


class SeasonalResidualBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)
        self.pointwise = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x - _temporal_moving_average(x, self.kernel_size)
        out = self.depthwise(residual.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.pointwise(out))


class NoiseResidualBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x - _temporal_moving_average(x, self.kernel_size)
        scale = residual.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
        gate = torch.sigmoid(self.gate(residual / scale))
        return self.dropout(self.proj(gate * residual))


class ExpertRouter(nn.Module):
    """MoE-style Top-K sparse routing network for GTB experts.

    Inspired by DeepSeek-V2's MoE routing and Switch Transformer:
    - Lightweight gating network produces logits for each expert
    - Top-K selection activates only K experts per sample (sparse)
    - Load-balancing auxiliary loss prevents routing collapse
    - Optional noise injection for exploration during training
    - Optional shared expert (always active, DeepSeek-V2 inspired)

    References:
        Fedus et al., "Switch Transformers", 2021.
        DeepSeek-AI, "DeepSeek-V2: A Strong, Economical MoE LLM", 2024.

    Args:
        d_model: Input feature dimension for the router.
        n_experts: Total number of experts (default 3 for GTB).
        top_k: Number of experts to activate per sample.
        noise_std: Gaussian noise stddev added to logits during training.
        balance_coeff: Coefficient for load-balancing auxiliary loss.
    """

    def __init__(self, d_model, n_experts=3, top_k=2, noise_std=0.1, balance_coeff=0.01,
                 balance_update_rate=0.01, bias_clamp=2.0, min_top_k=None, max_top_k=None,
                 feature_adaptive=False, expert_prior_strength=1.0, pool_min_experts=None,
                 pool_max_experts=None):
        super().__init__()
        self.n_experts = n_experts
        self.dynamic_top_k = isinstance(top_k, str) and top_k in {'auto', 'dynamic'}
        if self.dynamic_top_k:
            self.min_top_k = max(1, min(min_top_k or 2, n_experts))
            default_max = max(self.min_top_k, min(n_experts, math.ceil(n_experts * 0.67)))
            self.max_top_k = max(self.min_top_k, min(max_top_k or default_max, n_experts))
            self.top_k = self.max_top_k
        else:
            self.top_k = min(int(top_k), n_experts)
            self.min_top_k = self.top_k
            self.max_top_k = self.top_k
        self.noise_std = noise_std
        self.balance_coeff = balance_coeff
        self.balance_update_rate = balance_update_rate
        self.bias_clamp = bias_clamp
        self.feature_adaptive = feature_adaptive
        self.expert_prior_strength = expert_prior_strength
        self.profile_dim = 8
        if self.feature_adaptive:
            self.pool_min_experts = max(self.min_top_k, min(pool_min_experts or max(3, self.min_top_k), n_experts))
            self.pool_max_experts = max(self.pool_min_experts, min(pool_max_experts or n_experts, n_experts))
        else:
            self.pool_min_experts = n_experts
            self.pool_max_experts = n_experts

        # Lightweight router: pool sequence → project to expert logits
        hidden = max(8, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_experts),
        )
        if self.feature_adaptive:
            self.profile_gate = nn.Sequential(
                nn.Linear(self.profile_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, n_experts),
            )
            nn.init.zeros_(self.profile_gate[-1].weight)
            nn.init.zeros_(self.profile_gate[-1].bias)

        # Running statistics for monitoring routing distribution
        self.register_buffer('_expert_counts', torch.zeros(n_experts))
        self.register_buffer('_total_samples', torch.tensor(0.0))
        self.register_buffer('_load_bias', torch.zeros(n_experts))
        self.register_buffer('_active_expert_total', torch.tensor(0.0))
        self.register_buffer('_candidate_expert_total', torch.tensor(0.0))
        self.register_buffer('_profile_total', torch.zeros(self.profile_dim))
        self._aux_loss = torch.tensor(0.0)

    def _extract_profile(self, h):
        eps = 1e-6
        B, L, _ = h.shape
        y = h.mean(dim=-1)
        centered = y - y.mean(dim=1, keepdim=True)
        scale = centered.std(dim=1, unbiased=False).clamp_min(eps)
        if L > 1:
            diff = y[:, 1:] - y[:, :-1]
            diff_abs = diff.abs()
            noise = (diff.std(dim=1, unbiased=False) / scale / 3.0).clamp(0.0, 1.0)
            volatility = (diff_abs.std(dim=1, unbiased=False) / scale / 3.0).clamp(0.0, 1.0)
        else:
            noise = y.new_zeros(B)
            volatility = y.new_zeros(B)
        trend = ((y[:, -1] - y[:, 0]).abs() / scale / 3.0).clamp(0.0, 1.0)
        mid = max(1, L // 2)
        first = y[:, :mid].mean(dim=1)
        second = y[:, mid:].mean(dim=1) if mid < L else y[:, -1]
        regime = ((second - first).abs() / scale / 3.0).clamp(0.0, 1.0)
        smooth = _temporal_moving_average(y.unsqueeze(-1), min(7, max(1, L if L % 2 == 1 else L - 1))).squeeze(-1)
        residual = y - smooth
        residual_strength = (residual.std(dim=1, unbiased=False) / scale / 2.0).clamp(0.0, 1.0)
        spike = ((centered.abs().amax(dim=1) / scale - 2.0) / 4.0).clamp(0.0, 1.0)
        if L > 2:
            amp = torch.fft.rfft(centered, dim=1).abs()
            amp[:, 0] = 0.0
            freq_energy = amp.sum(dim=1, keepdim=True).clamp_min(eps)
            prob = amp / freq_energy
            peak = prob.max(dim=1).values
            denom = max(1.0, math.log(float(amp.shape[1])))
            entropy = -(prob * prob.clamp_min(eps).log()).sum(dim=1) / denom
            seasonal = (peak * math.sqrt(float(amp.shape[1]))).clamp(0.0, 1.0)
            spectral_complexity = entropy.clamp(0.0, 1.0)
        else:
            seasonal = y.new_zeros(B)
            spectral_complexity = y.new_zeros(B)
        return torch.stack([
            trend,
            seasonal,
            noise,
            regime,
            spectral_complexity,
            volatility,
            residual_strength,
            spike,
        ], dim=1)

    def _profile_prior(self, profile):
        prior = profile.new_zeros(profile.shape[0], self.n_experts)
        trend, seasonal, noise, regime, complexity, volatility, residual, spike = profile.unbind(dim=1)
        if self.n_experts > 0:
            prior[:, 0] = 1.2 * seasonal + 0.3 * (1.0 - complexity)
        if self.n_experts > 1:
            prior[:, 1] = 0.7 * complexity + 0.6 * regime + 0.3 * trend
        if self.n_experts > 2:
            prior[:, 2] = 0.4 + 0.4 * complexity + 0.2 * noise
        if self.n_experts > 3:
            prior[:, 3] = 1.2 * trend + 0.5 * regime
        if self.n_experts > 4:
            prior[:, 4] = 1.1 * seasonal + 0.6 * residual
        if self.n_experts > 5:
            prior[:, 5] = 1.1 * noise + 0.7 * spike
        if self.n_experts > 6:
            simple = 1.0 - torch.stack([
                trend,
                seasonal,
                noise,
                regime,
                complexity,
                volatility,
                residual,
                spike,
            ], dim=1).max(dim=1).values
            prior[:, 6] = 1.0 * simple.clamp(0.0, 1.0) + 0.2 * (1.0 - complexity)
        if self.n_experts > 7:
            prior[:, 7] = 1.2 * regime + 0.5 * trend + 0.4 * spike
        if self.n_experts > 8:
            prior[:, 8] = 1.2 * volatility + 0.7 * noise + 0.3 * spike
        return prior

    def _profile_complexity(self, profile):
        trend, seasonal, noise, regime, complexity, volatility, residual, spike = profile.unbind(dim=1)
        structural = torch.stack([
            0.5 * trend + 0.5 * seasonal,
            noise,
            regime,
            complexity,
            volatility,
            residual,
            spike,
        ], dim=1)
        return (0.45 * structural.mean(dim=1) + 0.55 * structural.max(dim=1).values).clamp(0.0, 1.0)

    def forward(self, h):
        """Compute routing weights.

        Args:
            h: (B, L, D) input representation

        Returns:
            weights: (B, n_experts) routing weights (sparse, top-k nonzero)
            indices: (B, top_k) indices of selected experts
            aux_loss: scalar load-balancing loss
        """
        B = h.shape[0]

        # Global average pooling over sequence dim → (B, D)
        h_mean = h.mean(dim=1)
        h_std = h.std(dim=1, unbiased=False)
        h_last = h[:, -1, :]
        h_trend = h[:, -1, :] - h[:, 0, :]
        h_pool = torch.cat([h_mean, h_std, h_last, h_trend], dim=-1)

        logits = self.gate(h_pool)
        profile = None
        complexity = None
        candidate_count = torch.full((B,), self.n_experts, device=h.device, dtype=torch.long)

        if self.feature_adaptive:
            profile = self._extract_profile(h)
            complexity = self._profile_complexity(profile)
            logits = logits + self.profile_gate(profile) + self.expert_prior_strength * self._profile_prior(profile)
            pool_float = self.pool_min_experts + complexity * (self.pool_max_experts - self.pool_min_experts)
            candidate_count = torch.ceil(pool_float).to(torch.long).clamp(
                min=self.pool_min_experts,
                max=self.pool_max_experts
            )

        # Add noise during training for exploration (Switch Transformer trick)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        selection_logits = logits + self._load_bias.to(logits.device)
        if self.feature_adaptive:
            candidate_slots = min(self.pool_max_experts, self.n_experts)
            _, candidate_indices = torch.topk(selection_logits, candidate_slots, dim=-1)
            candidate_rank = torch.arange(candidate_slots, device=h.device).unsqueeze(0)
            candidate_rank_mask = candidate_rank < candidate_count.unsqueeze(1)
            candidate_mask = torch.zeros_like(selection_logits)
            candidate_mask.scatter_(1, candidate_indices, candidate_rank_mask.to(selection_logits.dtype))
            selection_logits = selection_logits.masked_fill(candidate_mask <= 0, -1e9)

        if self.dynamic_top_k:
            selection_probs = F.softmax(selection_logits, dim=-1)
            entropy = -(selection_probs * selection_probs.clamp_min(1e-8).log()).sum(dim=-1)
            denom = torch.log(candidate_count.to(selection_logits.dtype).clamp_min(2.0))
            entropy = (entropy / denom).clamp(0.0, 1.0)
            activation_score = entropy if complexity is None else (0.45 * entropy + 0.55 * complexity)
            k_float = self.min_top_k + activation_score * (self.max_top_k - self.min_top_k)
            k_upper = torch.minimum(
                torch.full_like(candidate_count, self.max_top_k),
                candidate_count
            )
            k_per_sample = torch.ceil(k_float).to(torch.long).clamp(min=self.min_top_k)
            k_per_sample = torch.minimum(k_per_sample, k_upper)
        else:
            k_per_sample = torch.full((B,), self.top_k, device=h.device, dtype=torch.long)
            k_per_sample = torch.minimum(k_per_sample, candidate_count)

        # Top-K selection
        _, top_k_indices = torch.topk(selection_logits, self.max_top_k, dim=-1)  # (B, top_k)
        top_k_logits = logits.gather(1, top_k_indices)
        top_k_rank = torch.arange(self.max_top_k, device=h.device).unsqueeze(0)
        top_k_mask = top_k_rank < k_per_sample.unsqueeze(1)

        # Softmax only over selected experts (sparse softmax)
        top_k_logits = top_k_logits.masked_fill(~top_k_mask, -1e9)
        top_k_weights = F.softmax(top_k_logits, dim=-1) * top_k_mask.to(logits.dtype)  # (B, top_k)

        # Build full sparse weight matrix for convenient indexing
        weights = torch.zeros(B, self.n_experts, device=h.device)  # (B, n_experts)
        weights.scatter_(1, top_k_indices, top_k_weights)

        # Load-balancing auxiliary loss (Switch Transformer / DeepSeek style)
        # L_balance = n_experts * Σ_i (f_i * P_i)
        #   f_i = fraction of samples routed to expert i
        #   P_i = mean routing probability for expert i
        if self.training:
            probs = F.softmax(logits, dim=-1)  # (B, n_experts)
            # f_i: fraction of samples where expert i is in top-k
            mask = torch.zeros_like(logits)
            mask.scatter_(1, top_k_indices, top_k_mask.to(logits.dtype))
            f = mask.mean(dim=0)  # (n_experts,)
            # P_i: mean probability for expert i
            P = probs.mean(dim=0)  # (n_experts,)
            aux_loss = self.balance_coeff * self.n_experts * (f * P).sum()

            # Update running stats
            with torch.no_grad():
                self._expert_counts += mask.sum(dim=0)
                self._total_samples += B
                self._active_expert_total += k_per_sample.float().sum()
                self._candidate_expert_total += candidate_count.float().sum()
                if profile is not None:
                    self._profile_total += profile.detach().sum(dim=0)
                target = k_per_sample.float().mean() / float(self.n_experts)
                self._load_bias += self.balance_update_rate * (target - f)
                self._load_bias -= self._load_bias.mean()
                self._load_bias.clamp_(min=-self.bias_clamp, max=self.bias_clamp)
        else:
            aux_loss = torch.tensor(0.0, device=h.device)

        self._aux_loss = aux_loss
        return weights, top_k_indices, aux_loss

    def get_routing_stats(self):
        """Return routing distribution statistics for monitoring."""
        if self._total_samples > 0:
            freq = self._expert_counts / self._total_samples
            avg_profile = self._profile_total / self._total_samples
            return {
                'expert_freq': freq.cpu().tolist(),
                'load_bias': self._load_bias.cpu().tolist(),
                'avg_active_experts': float((self._active_expert_total / self._total_samples).item()),
                'avg_candidate_experts': float((self._candidate_expert_total / self._total_samples).item()),
                'avg_profile': avg_profile.cpu().tolist(),
                'dynamic_top_k': self.dynamic_top_k,
                'feature_adaptive': self.feature_adaptive,
                'total_samples': int(self._total_samples.item())
            }
        return {
            'expert_freq': [0.0] * self.n_experts,
            'load_bias': [0.0] * self.n_experts,
            'avg_active_experts': 0.0,
            'avg_candidate_experts': 0.0,
            'avg_profile': [0.0] * self.profile_dim,
            'dynamic_top_k': self.dynamic_top_k,
            'feature_adaptive': self.feature_adaptive,
            'total_samples': 0
        }

    def reset_stats(self):
        """Reset running routing statistics."""
        self._expert_counts.zero_()
        self._total_samples.zero_()
        self._load_bias.zero_()
        self._active_expert_total.zero_()
        self._candidate_expert_total.zero_()
        self._profile_total.zero_()


class GlobalTemporalBlock(nn.Module):
    """Pluggable Global Temporal Block (GTB) for time series enhancement.

    Three expert components in a pre-norm residual architecture:
    1. FreqMixing: 2D conv on period-reshaped data (TS SOTA + image gen)
    2. GatedLinearAttention: O(N) global attention (LLM efficiency)
    3. SwiGLU: Gated channel mixing (LLM architecture)

    Routing modes:
    - 'static':   Components controlled by use_freq_mixing / use_attention / use_swiglu.
    - 'adaptive': MoE-style learned Top-K routing via ExpertRouter.
                  Router dynamically selects which experts to activate per sample.
                  Includes load-balancing loss and optional shared expert.

    Can operate on:
    - (B, L) input: auto-embeds, processes, projects back to (B, L)
    - (B, L, D) input: processes directly, returns (B, L, D)

    Args:
        seq_len: Input sequence length.
        d_model: Internal embedding dimension.
        n_heads: Number of attention heads for GatedLinearAttention.
        top_k_freqs: Number of dominant frequencies for FreqMixing.
        expand_ratio: SwiGLU expansion ratio.
        dropout: Dropout rate.
        use_freq_mixing: Enable FreqMixing (static mode only).
        use_attention: Enable GatedLinearAttention (static mode only).
        use_swiglu: Enable SwiGLU (static mode only).
        routing_mode: 'static' or 'adaptive'.
        top_k_experts: How many experts to activate per sample (adaptive mode).
        shared_expert: Index of always-active expert (adaptive mode). None=no shared.
        router_noise_std: Noise stddev for router exploration (adaptive mode).
        balance_coeff: Load-balancing loss coefficient (adaptive mode).
    """

    # Expert name mapping for logging
    EXPERT_NAMES = ['FreqMix', 'Attention', 'SwiGLU']

    def __init__(self, seq_len, d_model=64, n_heads=4, top_k_freqs=3,
                 expand_ratio=2.0, dropout=0.1,
                 use_freq_mixing=True, use_attention=True, use_swiglu=True,
                 routing_mode='static', top_k_experts=2, shared_expert=None,
                 router_noise_std=0.1, balance_coeff=0.01):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.routing_mode = routing_mode
        self.shared_expert = shared_expert
        self.use_extended_experts = routing_mode == 'adaptive_plus'
        self.expert_names = ['FreqMix', 'Attention', 'SwiGLU']
        if self.use_extended_experts:
            self.expert_names = self.expert_names + ['Trend', 'SeasonalResidual', 'NoiseResidual', 'Skip']
        self.n_experts = len(self.expert_names)

        # Ensure n_heads divides d_model
        for h in [n_heads, 4, 2, 1]:
            if d_model % h == 0:
                n_heads = h
                break

        # Input/output projections for 2D input mode
        self.input_embed = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)

        # Lazy projection for 3D input mode (created on first forward if needed)
        self._input_proj_3d = None
        self._output_proj_3d = None

        # --- Always create all three expert components ---
        # In adaptive mode, all experts exist but are selectively activated.
        # In static mode, flags control which are created (backward compat).

        if routing_mode in {'adaptive', 'adaptive_plus'}:
            # Adaptive: always create all experts
            self.use_freq_mixing = True
            self.use_attention = True
            self.use_swiglu = True
        else:
            # Static: honor the manual flags
            self.use_freq_mixing = use_freq_mixing
            self.use_attention = use_attention
            self.use_swiglu = use_swiglu

        # Expert 0: Frequency-guided 2D convolution
        if self.use_freq_mixing:
            self.freq_norm = RMSNorm(d_model)
            self.freq_mixing = FreqMixingBlock(d_model, top_k=top_k_freqs)

        # Expert 1: Gated Linear Attention
        if self.use_attention:
            self.attn_norm = RMSNorm(d_model)
            self.attention = GatedLinearAttention(d_model, n_heads, dropout)

        # Expert 2: SwiGLU FFN
        if self.use_swiglu:
            self.ffn_norm = RMSNorm(d_model)
            self.ffn = SwiGLU(d_model, expand_ratio, dropout)

        if self.use_extended_experts:
            self.trend_norm = RMSNorm(d_model)
            self.trend = TrendExtractionBlock(d_model, dropout=dropout)
            self.seasonal_norm = RMSNorm(d_model)
            self.seasonal = SeasonalResidualBlock(d_model, dropout=dropout)
            self.noise_norm = RMSNorm(d_model)
            self.noise = NoiseResidualBlock(d_model, dropout=dropout)

        # --- Router (adaptive mode only) ---
        if routing_mode in {'adaptive', 'adaptive_plus'}:
            router_top_k = top_k_experts
            router_balance_coeff = balance_coeff
            router_max_top_k = min(4, self.n_experts)
            feature_adaptive_router = False
            if routing_mode == 'adaptive_plus' and top_k_experts == 2:
                router_top_k = 'dynamic'
                router_balance_coeff = min(balance_coeff, 0.005)
                router_max_top_k = min(4, self.n_experts)
                feature_adaptive_router = True
            self.router = ExpertRouter(
                d_model=d_model,
                n_experts=self.n_experts,
                top_k=router_top_k,
                noise_std=router_noise_std,
                balance_coeff=router_balance_coeff,
                min_top_k=2,
                max_top_k=router_max_top_k,
                feature_adaptive=feature_adaptive_router,
                expert_prior_strength=0.8,
                pool_min_experts=4,
                pool_max_experts=self.n_experts
            )

        # Learnable residual scale — starts small so block doesn't disrupt
        # pre-existing model behavior when first added (zero-init idea from DiT)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Store latest auxiliary loss for external retrieval
        self._aux_loss = torch.tensor(0.0)

        self._init_weights()

    def _init_weights(self):
        """Initialize output projection to near-zero for stable integration."""
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _run_expert(self, idx, h):
        """Run a single expert by index.

        Args:
            idx: 0=FreqMix, 1=Attention, 2=SwiGLU
            h: (B, L, D) input
        Returns:
            Expert output (B, L, D), same shape as input.
        """
        if idx == 0:
            return self.freq_mixing(self.freq_norm(h))
        elif idx == 1:
            return self.attention(self.attn_norm(h))
        elif idx == 2:
            return self.ffn(self.ffn_norm(h))
        elif idx == 3 and self.use_extended_experts:
            return self.trend(self.trend_norm(h))
        elif idx == 4 and self.use_extended_experts:
            return self.seasonal(self.seasonal_norm(h))
        elif idx == 5 and self.use_extended_experts:
            return self.noise(self.noise_norm(h))
        return torch.zeros_like(h)

    def _forward_static(self, h):
        """Static routing: sequential application of enabled experts."""
        if self.use_freq_mixing:
            h = h + self.freq_mixing(self.freq_norm(h))
        if self.use_attention:
            h = h + self.attention(self.attn_norm(h))
        if self.use_swiglu:
            h = h + self.ffn(self.ffn_norm(h))
        return h

    def _forward_adaptive(self, h):
        """Adaptive MoE routing: Top-K sparse expert selection per sample.

        Each sample gets routed to its top-K experts. Experts not selected
        are entirely skipped (sparse activation = compute saving).
        If a shared expert is specified, it's always active in addition.
        """
        B, L, D = h.shape

        # Router decides which experts to use → (B, n_experts) weights
        weights, indices, aux_loss = self.router(h)
        self._aux_loss = aux_loss

        # Determine which experts are active (union across batch)
        active_experts = set()
        for i in range(self.n_experts):
            if weights[:, i].sum() > 0:
                active_experts.add(i)

        # Always include shared expert if specified
        if self.shared_expert is not None:
            active_experts.add(self.shared_expert)

        # Run only active experts and combine with routing weights
        combined = torch.zeros_like(h)  # (B, L, D)
        for idx in active_experts:
            expert_out = self._run_expert(idx, h)  # (B, L, D)
            # Weight per sample: (B, 1, 1) broadcast over (L, D)
            w = weights[:, idx].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

            # Shared expert gets full weight (always on)
            if idx == self.shared_expert:
                # Shared expert: blend between routing weight and minimum floor
                w = torch.clamp(w, min=0.3)

            combined = combined + w * expert_out

        return h + combined

    def get_aux_loss(self):
        """Return the latest auxiliary load-balancing loss.

        Should be added to the main training loss:
            total_loss = task_loss + model.get_gtb_aux_loss()
        """
        return self._aux_loss

    def get_routing_stats(self):
        """Return routing statistics (adaptive mode only)."""
        if self.routing_mode in {'adaptive', 'adaptive_plus'}:
            stats = self.router.get_routing_stats()
            stats['expert_names'] = self.expert_names
            return stats
        return None

    def forward(self, x):
        """
        Args:
            x: (B, L) or (B, L, D) input tensor
        Returns:
            Same shape as input, enhanced with global temporal features
        """
        input_2d = (x.ndim == 2)

        if input_2d:
            # (B, L) → (B, L, d_model)
            h = self.input_embed(x.unsqueeze(-1))
        else:
            D_in = x.shape[-1]
            if D_in != self.d_model:
                # Lazily create projection layers for mismatched dimensions
                if self._input_proj_3d is None or self._input_proj_3d.in_features != D_in:
                    self._input_proj_3d = nn.Linear(D_in, self.d_model).to(x.device)
                    self._output_proj_3d = nn.Linear(self.d_model, D_in).to(x.device)
                    nn.init.zeros_(self._output_proj_3d.weight)
                    nn.init.zeros_(self._output_proj_3d.bias)
                h = self._input_proj_3d(x)
            else:
                h = x

        residual_h = h

        # Route through experts
        if self.routing_mode in {'adaptive', 'adaptive_plus'}:
            h = self._forward_adaptive(h)
        else:
            h = self._forward_static(h)

        # Scale the enhancement (small initially for training stability)
        scale = torch.sigmoid(self.residual_scale)
        h = residual_h + scale * (h - residual_h)

        if input_2d:
            return x + scale * self.output_proj(h).squeeze(-1)
        else:
            if x.shape[-1] != self.d_model and self._output_proj_3d is not None:
                return x + scale * self._output_proj_3d(h - residual_h)
            return h
