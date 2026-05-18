import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.attn(z, z, z, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class ModernMixerBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        z = self.norm1(x).transpose(1, 2)
        x = x + self.token_mix(z).transpose(1, 2)
        return x + self.channel_mix(self.norm2(x))


class ModernSeriesTokenizer(nn.Module):
    def __init__(self, in_features, d_model, patch_len=4, stride=None, moving_avg=3, top_k=3):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride or max(1, self.patch_len // 2)))
        self.moving_avg = max(1, int(moving_avg))
        self.top_k = max(1, int(top_k))
        self.multi_patch_sizes = [2, 4, 8]
        max_tokens = max(8, in_features * 4)
        self.input_proj = nn.Linear(1, d_model)
        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.multi_patch_projs = nn.ModuleDict({
            str(k): nn.Linear(k, d_model) for k in self.multi_patch_sizes
        })
        self.pos_encoding = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])

    def moving_average(self, x):
        if self.moving_avg <= 1:
            return x
        pad = self.moving_avg // 2
        z = F.pad(x.unsqueeze(1), (pad, pad), mode='replicate')
        z = F.avg_pool1d(z, kernel_size=self.moving_avg, stride=1)
        return z.squeeze(1)[..., :x.shape[1]]

    def frequency_filter(self, x):
        freq = torch.fft.rfft(x, dim=1)
        if freq.shape[1] <= 1:
            return x
        mag = torch.abs(freq)
        k = min(self.top_k, mag.shape[1])
        idx = torch.topk(mag, k=k, dim=1).indices
        mask = torch.zeros_like(mag)
        mask.scatter_(1, idx, 1.0)
        return torch.fft.irfft(freq * mask, n=x.shape[1], dim=1)

    def add_pos(self, tokens):
        if tokens.shape[1] > self.pos_encoding.shape[1]:
            return tokens
        return tokens + self.pos_encoding[:, :tokens.shape[1], :]

    def sequence(self, x):
        return self.add_pos(self.input_proj(x.unsqueeze(-1)))

    def patch(self, x, patch_len=None, stride=None):
        patch_len = int(patch_len or self.patch_len)
        stride = int(stride or self.stride)
        if x.shape[1] < patch_len:
            x = F.pad(x.unsqueeze(1), (patch_len - x.shape[1], 0), mode='replicate').squeeze(1)
        patches = x.unfold(dimension=1, size=patch_len, step=max(1, stride))
        if patch_len == self.patch_len:
            tokens = self.patch_proj(patches)
        else:
            tokens = self.multi_patch_projs[str(patch_len)](patches)
        return self.add_pos(tokens)

    def multi_patch(self, x):
        tokens = []
        for patch_len in self.multi_patch_sizes:
            if patch_len <= max(self.in_features, 2):
                tokens.append(self.patch(x, patch_len=patch_len, stride=max(1, patch_len // 2)))
        return torch.cat(tokens, dim=1) if tokens else self.sequence(x)

    def conv(self, x):
        features = [conv(x.unsqueeze(1)).transpose(1, 2) for conv in self.conv_bank]
        return self.add_pos(torch.stack(features, dim=0).mean(dim=0))


class TemporalProjectionHead(nn.Module):
    def __init__(self, in_features, out_features, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.temporal_weight = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, out_features),
        )
        self.residual_proj = nn.Linear(in_features, out_features)
        self.trend_proj = nn.Linear(in_features, out_features)
        self.level_proj = nn.Linear(in_features, out_features)

    def forward(self, h, x_norm, mean, std, use_revin=True, extra=None):
        weights = torch.softmax(self.temporal_weight(h).squeeze(-1), dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        out = self.head(pooled) + self.residual_proj(x_norm)
        if extra is not None:
            out = out + extra
        if use_revin:
            out = out * std + mean
        return out
