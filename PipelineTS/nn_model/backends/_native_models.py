import math

import numpy as np

from PipelineTS.base import ForecastingMixin
from PipelineTS.nn_model._modern_ts_specs import (
    MODERN_TS_NATIVE_ATTENTION_KINDS,
    MODERN_TS_NATIVE_DEFAULTS,
    MODERN_TS_NATIVE_FEATURE_BANK_KINDS,
    MODERN_TS_NATIVE_RNN_KINDS,
)
from PipelineTS.nn_model.backends import resolve_nn_backend
from PipelineTS.nn_model.backends._linear_models import _batch_size


_NATIVE_KIND_DEFAULTS = {
    'nbeats': {'hidden': 128, 'layers': 3},
    'nhits': {'hidden': 128, 'layers': 3},
    'tide': {'hidden': 128, 'layers': 3},
    'tcn': {'hidden': 96, 'layers': 3},
    'patch_rnn': {'hidden': 96, 'layers': 3},
    'stacking_rnn': {'hidden': 96, 'layers': 3},
    'time2vec': {'hidden': 96, 'layers': 3},
    'transformer': {'hidden': 128, 'layers': 2},
    'itransformer': {'hidden': 128, 'layers': 2},
    'gau': {'hidden': 128, 'layers': 2},
    'tft': {'hidden': 128, 'layers': 2},
    'srs_net': {'hidden': 128, 'layers': 2},
    'deepar': {'hidden': 96, 'layers': 3},
    **MODERN_TS_NATIVE_DEFAULTS,
}


def _as_float_array(x):
    return np.asarray(x, dtype=np.float32)


def _flatten_target(y):
    y = _as_float_array(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    output_shape = y.shape[1:]
    return y.reshape(y.shape[0], -1), output_shape


def _feature_dim(kind, input_dim):
    if kind in {'tcn', 'patch_rnn', 'nhits'} | MODERN_TS_NATIVE_FEATURE_BANK_KINDS:
        return input_dim * 3
    if kind == 'time2vec':
        return input_dim * 10
    return input_dim


class _BaseNativeNN:
    def __init__(self, kind, in_features, out_features, n_vars=1, loss_fn='huber', learning_rate=0.001,
                 random_seed=42, weight_decay=1e-4, use_revin=True, dropout=0.1, **kwargs):
        self.kind = kind
        self.in_features = in_features
        self.out_features = out_features
        self.n_vars = n_vars
        self.loss_fn_name = loss_fn
        self.learning_rate = learning_rate
        self.random_seed = 42 if random_seed is None else random_seed
        self.weight_decay = weight_decay
        self.use_revin = use_revin
        self.dropout = dropout
        self.kwargs = kwargs
        self.training_logs = {'epochs': [], 'train_loss': [], 'test_loss': [], 'lrs': []}
        self._cqr_enabled = False
        self._residual_gate_enabled = False
        self.__pipelinets_is_fitted__ = False
        self._params = None
        defaults = _NATIVE_KIND_DEFAULTS.get(kind, {'hidden': 128, 'layers': 2})
        self.hidden_size = int(kwargs.get('hidden_size', kwargs.get('d_model', kwargs.get('layer_widths', defaults['hidden']))))
        self.hidden_size = max(8, min(self.hidden_size, 256))
        self.num_layers = int(kwargs.get('num_layers', kwargs.get('num_encoder_layers', kwargs.get('level', defaults['layers']))))
        self.num_layers = max(1, min(self.num_layers, 6))
        self._output_shape = None

    def _enable_cqr(self, alpha=0.1):
        self._cqr_enabled = True
        self._cqr_alpha = alpha

    def _enable_residual_gate(self, n_sinkhorn_iters=10, init_alpha=0.3):
        self._residual_gate_enabled = True
        self._gate_init_alpha = init_alpha

    def _input_dim_from_X(self, X):
        if X.ndim == 3:
            return int(X.shape[1] * X.shape[2])
        return int(X.shape[1])

    def _init_np_params(self, input_dim, target_dim):
        if self.kind == 'nhits':
            return self._init_nhits_np_params(input_dim, target_dim)
        if self.kind == 'nbeats':
            return self._init_nbeats_np_params(input_dim, target_dim)
        if self.kind == 'tcn':
            return self._init_tcn_np_params(input_dim, target_dim)
        if self.kind == 'transformer':
            return self._init_transformer_np_params(input_dim, target_dim)

        rng = np.random.RandomState(self.random_seed)
        output_dim = target_dim * (3 if self._cqr_enabled else 1)
        feature_dim = _feature_dim(self.kind, input_dim)
        params = {}

        def init_w(name, fan_in, fan_out):
            scale = math.sqrt(6.0 / max(fan_in + fan_out, 1))
            params[name] = rng.uniform(-scale, scale, (fan_in, fan_out)).astype(np.float32)
            params[name.replace('w', 'b', 1)] = np.zeros((fan_out,), dtype=np.float32)

        if self.kind in {'transformer', 'itransformer', 'gau', 'tft', 'srs_net'} | MODERN_TS_NATIVE_ATTENTION_KINDS:
            init_w('w_q', input_dim, self.hidden_size)
            init_w('w_k', input_dim, self.hidden_size)
            init_w('w_v', input_dim, self.hidden_size)
            init_w('w_attn', self.hidden_size, feature_dim)
        elif self.kind in {'stacking_rnn'} | MODERN_TS_NATIVE_RNN_KINDS:
            init_w('w_rnn_x', 1, self.hidden_size)
            init_w('w_rnn_h', self.hidden_size, self.hidden_size)
            init_w('w_rnn_out', self.hidden_size, feature_dim)
        elif self.kind == 'deepar':
            init_w('w_rnn_x', 1, self.hidden_size)
            init_w('w_rnn_h', self.hidden_size, self.hidden_size)
            init_w('w_rnn_out', self.hidden_size, feature_dim)
            init_w('w_sigma', self.hidden_size, output_dim)
        elif self.kind == 'time2vec':
            freqs = np.logspace(-2, 1, 4).astype(np.float32)
            params['t2v_freq'] = freqs
            params['t2v_phase'] = np.zeros((4,), dtype=np.float32)

        init_w('w_in', feature_dim, self.hidden_size)
        for i in range(max(self.num_layers - 1, 0)):
            init_w(f'w_{i}', self.hidden_size, self.hidden_size)
        init_w('w_out', self.hidden_size, output_dim)
        init_w('w_res', input_dim, output_dim)
        params['gate_alpha'] = np.array(float(getattr(self, '_gate_init_alpha', 0.3)), dtype=np.float32)
        return params

    def _xavier(self, rng, fan_in, fan_out):
        scale = math.sqrt(6.0 / max(fan_in + fan_out, 1))
        return rng.uniform(-scale, scale, (fan_in, fan_out)).astype(np.float32)

    def _linear_params(self, rng, params, prefix, fan_in, fan_out):
        params[f'{prefix}_w'] = self._xavier(rng, fan_in, fan_out)
        params[f'{prefix}_b'] = np.zeros((fan_out,), dtype=np.float32)

    def _layer_norm_params(self, params, prefix, size):
        params[f'{prefix}_g'] = np.ones((size,), dtype=np.float32)
        params[f'{prefix}_b'] = np.zeros((size,), dtype=np.float32)

    def _conv_params(self, rng, params, prefix, out_ch, in_ch, kernel_size, weight_norm=True):
        fan_in = max(in_ch * kernel_size, 1)
        v = rng.normal(0.0, math.sqrt(2.0 / fan_in), (out_ch, in_ch, kernel_size)).astype(np.float32)
        if weight_norm:
            params[f'{prefix}_v'] = v
            params[f'{prefix}_g'] = np.sqrt(np.sum(v * v, axis=(1, 2), keepdims=True)).astype(np.float32)
        else:
            params[f'{prefix}_w'] = v
        params[f'{prefix}_b'] = np.zeros((out_ch,), dtype=np.float32)

    def _init_nbeats_np_params(self, input_dim, target_dim):
        rng = np.random.RandomState(self.random_seed)
        output_dim = target_dim * (3 if self._cqr_enabled else 1)
        blocks = self._nbeats_blocks()
        params = {}
        _, _, _, num_layers, layer_widths, _, _ = self._nbeats_config()
        for block_idx, block_type, coeff_dim in blocks:
            current_dim = input_dim
            for layer_idx in range(num_layers):
                self._linear_params(rng, params, f'nb_b{block_idx}_fc{layer_idx}', current_dim, layer_widths)
                current_dim = layer_widths
            self._linear_params(rng, params, f'nb_b{block_idx}_backcast_coeff', layer_widths, coeff_dim)
            self._linear_params(rng, params, f'nb_b{block_idx}_forecast_coeff', layer_widths, coeff_dim)
            if block_type == 'generic':
                self._linear_params(rng, params, f'nb_b{block_idx}_backcast_basis', coeff_dim, input_dim)
                self._linear_params(rng, params, f'nb_b{block_idx}_forecast_basis', coeff_dim, output_dim)
        return params

    def _nbeats_config(self):
        generic_architecture = bool(self.kwargs.get('generic_architecture', True))
        num_stacks = int(self.kwargs.get('num_stacks', 2))
        num_blocks = int(self.kwargs.get('num_blocks', 3))
        num_layers = int(self.kwargs.get('num_layers', 4))
        layer_widths = int(self.kwargs.get('layer_widths', 256))
        expansion_coeff_dim = int(self.kwargs.get('expansion_coeff_dim', 32))
        trend_degree = int(self.kwargs.get('trend_degree', 3))
        return generic_architecture, num_stacks, num_blocks, num_layers, layer_widths, expansion_coeff_dim, trend_degree

    def _nbeats_blocks(self):
        generic_architecture, num_stacks, num_blocks, _, _, expansion_coeff_dim, trend_degree = self._nbeats_config()
        blocks = []
        block_idx = 0
        if generic_architecture:
            for _ in range(num_stacks):
                for _ in range(num_blocks):
                    blocks.append((block_idx, 'generic', expansion_coeff_dim))
                    block_idx += 1
        else:
            num_trend_stacks = max(1, num_stacks // 2)
            num_seasonal_stacks = num_stacks - num_trend_stacks
            for _ in range(num_trend_stacks):
                for _ in range(num_blocks):
                    blocks.append((block_idx, 'trend', trend_degree + 1))
                    block_idx += 1
            for _ in range(num_seasonal_stacks):
                for _ in range(num_blocks):
                    blocks.append((block_idx, 'seasonality', 2 * max(1, self.out_features // 2)))
                    block_idx += 1
        return blocks

    def _init_nhits_np_params(self, input_dim, target_dim):
        rng = np.random.RandomState(self.random_seed)
        output_dim = target_dim * (3 if self._cqr_enabled else 1)
        params = {}
        _, _, num_layers, layer_widths, _, _ = self._nhits_config()
        for block_idx, pool_kernel_size, n_freq_downsample in self._nhits_blocks(input_dim, output_dim):
            current_dim = int(math.ceil(input_dim / max(pool_kernel_size, 1)))
            for layer_idx in range(num_layers):
                self._linear_params(rng, params, f'nh_b{block_idx}_fc{layer_idx}', current_dim, layer_widths)
                current_dim = layer_widths
            backcast_out = int(math.ceil(input_dim / max(n_freq_downsample, 1)))
            forecast_out = int(math.ceil(output_dim / max(n_freq_downsample, 1)))
            self._linear_params(rng, params, f'nh_b{block_idx}_backcast', layer_widths, backcast_out)
            self._linear_params(rng, params, f'nh_b{block_idx}_forecast', layer_widths, forecast_out)
        return params

    def _nhits_config(self):
        num_stacks = int(self.kwargs.get('num_stacks', 3))
        num_blocks = int(self.kwargs.get('num_blocks', 1))
        num_layers = int(self.kwargs.get('num_layers', 2))
        layer_widths = int(self.kwargs.get('layer_widths', 512))
        pooling_kernel_sizes = self.kwargs.get('pooling_kernel_sizes')
        n_freq_downsample = self.kwargs.get('n_freq_downsample')
        return num_stacks, num_blocks, num_layers, layer_widths, pooling_kernel_sizes, n_freq_downsample

    def _nhits_blocks(self, input_dim, output_dim):
        num_stacks, num_blocks, _, _, pooling_kernel_sizes, n_freq_downsample = self._nhits_config()
        if pooling_kernel_sizes is None:
            pooling_kernel_sizes = [min(2 ** i, max(1, input_dim // 4)) for i in range(num_stacks)]
        else:
            pooling_kernel_sizes = list(pooling_kernel_sizes)
        if n_freq_downsample is None:
            n_freq_downsample = [max(1, output_dim // (2 ** i)) for i in range(num_stacks)]
            n_freq_downsample = list(reversed(n_freq_downsample))
        else:
            n_freq_downsample = list(n_freq_downsample)
        while len(pooling_kernel_sizes) < num_stacks:
            pooling_kernel_sizes.append(pooling_kernel_sizes[-1])
        while len(n_freq_downsample) < num_stacks:
            n_freq_downsample.append(1)
        blocks = []
        block_idx = 0
        for stack_idx in range(num_stacks):
            for _ in range(num_blocks):
                blocks.append((block_idx, int(pooling_kernel_sizes[stack_idx]), int(n_freq_downsample[stack_idx])))
                block_idx += 1
        return blocks

    def _init_tcn_np_params(self, input_dim, target_dim):
        rng = np.random.RandomState(self.random_seed)
        output_dim = target_dim * (3 if self._cqr_enabled else 1)
        num_levels, hidden_channels, kernel_size = self._tcn_config()
        params = {}
        for i in range(num_levels):
            in_ch = 1 if i == 0 else hidden_channels
            self._conv_params(rng, params, f'tcn_l{i}_c1', hidden_channels, in_ch, kernel_size, weight_norm=True)
            self._conv_params(rng, params, f'tcn_l{i}_c2', hidden_channels, hidden_channels, kernel_size, weight_norm=True)
            if in_ch != hidden_channels:
                self._conv_params(rng, params, f'tcn_l{i}_down', hidden_channels, in_ch, 1, weight_norm=False)
            self._layer_norm_params(params, f'tcn_l{i}_ln', hidden_channels)
        self._linear_params(rng, params, 'tcn_out', hidden_channels, output_dim)
        self._linear_params(rng, params, 'tcn_res', input_dim, output_dim)
        return params

    def _tcn_config(self):
        kernel_size = int(self.kwargs.get('kernel_size', 3))
        kernel_size = max(kernel_size, 1)
        num_levels = self.kwargs.get('num_levels')
        if num_levels is None:
            num_levels = max(2, int(math.ceil(math.log2(max(self.in_features, 4) / max(kernel_size - 1, 1)))) + 1)
            num_levels = min(num_levels, 6)
        hidden_channels = self.kwargs.get('hidden_channels')
        if hidden_channels is None:
            hidden_channels = min(max(self.in_features, 32), 128)
        return int(num_levels), int(hidden_channels), int(kernel_size)

    def _init_transformer_np_params(self, input_dim, target_dim):
        rng = np.random.RandomState(self.random_seed)
        output_dim = target_dim * (3 if self._cqr_enabled else 1)
        d_model, nhead, num_layers, dim_feedforward, output_strategy = self._transformer_config()
        params = {
            'tf_pos': (rng.randn(1, input_dim, d_model) * 0.02).astype(np.float32),
        }
        self._linear_params(rng, params, 'tf_input', 1, d_model)
        for i in range(num_layers):
            self._layer_norm_params(params, f'tf_l{i}_ln1', d_model)
            self._linear_params(rng, params, f'tf_l{i}_q', d_model, d_model)
            self._linear_params(rng, params, f'tf_l{i}_k', d_model, d_model)
            self._linear_params(rng, params, f'tf_l{i}_v', d_model, d_model)
            self._linear_params(rng, params, f'tf_l{i}_attn_out', d_model, d_model)
            self._layer_norm_params(params, f'tf_l{i}_ln2', d_model)
            self._linear_params(rng, params, f'tf_l{i}_ff1', d_model, dim_feedforward)
            self._linear_params(rng, params, f'tf_l{i}_ff2', dim_feedforward, d_model)
        self._layer_norm_params(params, 'tf_final_ln', d_model)
        if output_strategy == 'pooled':
            self._linear_params(rng, params, 'tf_temporal_weight', d_model, 1)
            self._linear_params(rng, params, 'tf_head1', d_model, d_model)
            self._linear_params(rng, params, 'tf_head2', d_model, output_dim)
            self._linear_params(rng, params, 'tf_res', input_dim, output_dim)
        else:
            self._linear_params(rng, params, 'tf_head1', input_dim * d_model, d_model * 2)
            self._linear_params(rng, params, 'tf_head2', d_model * 2, output_dim)
        return params

    def _transformer_config(self):
        d_model = int(self.kwargs.get('d_model', self.hidden_size))
        nhead = int(self.kwargs.get('nhead', self.kwargs.get('n_heads', 4)))
        for h in [nhead, 4, 2, 1]:
            if d_model % h == 0:
                nhead = h
                break
        num_layers = int(self.kwargs.get('num_encoder_layers', self.num_layers))
        dim_feedforward = int(self.kwargs.get('dim_feedforward', d_model * 4))
        output_strategy = self.kwargs.get('output_strategy', 'flatten')
        return d_model, nhead, num_layers, dim_feedforward, output_strategy

    def _features(self, xp, params, x_flat, original_x):
        if self.kind in {'tcn', 'patch_rnn', 'nhits'} | MODERN_TS_NATIVE_FEATURE_BANK_KINDS:
            delta = xp.concatenate([xp.zeros_like(x_flat[:, :1]), x_flat[:, 1:] - x_flat[:, :-1]], axis=1)
            rev = x_flat[:, ::-1]
            return xp.concatenate([x_flat, delta, rev], axis=1)

        if self.kind == 'time2vec':
            freq = params['t2v_freq']
            phase = params['t2v_phase']
            periodic = []
            for i in range(4):
                periodic.append(xp.sin(x_flat * freq[i] + phase[i]))
                periodic.append(xp.cos(x_flat * freq[i] + phase[i]))
            return xp.concatenate([x_flat] + periodic + [x_flat * x_flat], axis=1)

        if self.kind in {'transformer', 'itransformer', 'gau', 'tft', 'srs_net'} | MODERN_TS_NATIVE_ATTENTION_KINDS:
            q = self._linear(xp, x_flat, params, 'w_q')
            k = self._linear(xp, x_flat, params, 'w_k')
            v = self._linear(xp, x_flat, params, 'w_v')
            attn_score = xp.mean(q * k, axis=1, keepdims=True) / math.sqrt(max(self.hidden_size, 1))
            gate = 1.0 / (1.0 + xp.exp(-attn_score))
            if self.kind in {'gau', 'tft', 'srs_net'}:
                h = xp.tanh(v) * gate
            else:
                h = v * gate
            return self._linear(xp, h, params, 'w_attn')

        if self.kind in {'stacking_rnn', 'deepar'} | MODERN_TS_NATIVE_RNN_KINDS:
            h = xp.zeros((x_flat.shape[0], self.hidden_size))
            for i in range(x_flat.shape[1]):
                step = x_flat[:, i:i + 1]
                h = xp.tanh(self._linear(xp, step, params, 'w_rnn_x') + self._linear(xp, h, params, 'w_rnn_h'))
            return self._linear(xp, h, params, 'w_rnn_out')

        return x_flat

    def _linear(self, xp, x, params, prefix):
        return x @ params[prefix] + params[prefix.replace('w', 'b', 1)]

    def _linear_named(self, x, params, prefix):
        return x @ params[f'{prefix}_w'] + params[f'{prefix}_b']

    def _gelu(self, xp, x):
        return 0.5 * x * (1.0 + xp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

    def _softmax(self, xp, x, axis=-1):
        z = x - xp.max(x, axis=axis, keepdims=True)
        e = xp.exp(z)
        return e / xp.sum(e, axis=axis, keepdims=True)

    def _stop_gradient(self, xp, x):
        stop_gradient = getattr(xp, 'stop_gradient', None)
        if stop_gradient is not None:
            return stop_gradient(x)
        return x

    def _revin_normalize(self, xp, x, enabled=True):
        if not enabled:
            return x, xp.zeros((x.shape[0], 1)), xp.ones((x.shape[0], 1))
        mean = xp.mean(x, axis=1, keepdims=True)
        centered = x - mean
        denom = max(int(x.shape[1]) - 1, 1)
        std = xp.sqrt(xp.sum(centered * centered, axis=1, keepdims=True) / denom) + 1e-5
        mean = self._stop_gradient(xp, mean)
        std = self._stop_gradient(xp, std)
        return (x - mean) / std, mean, std

    def _layer_norm(self, xp, x, params, prefix, axis=-1, eps=1e-5):
        mean = xp.mean(x, axis=axis, keepdims=True)
        var = xp.mean((x - mean) ** 2, axis=axis, keepdims=True)
        return (x - mean) / xp.sqrt(var + eps) * params[f'{prefix}_g'] + params[f'{prefix}_b']

    def _conv_weight(self, xp, params, prefix):
        if f'{prefix}_v' not in params:
            return params[f'{prefix}_w']
        v = params[f'{prefix}_v']
        g = params[f'{prefix}_g']
        norm = xp.sqrt(xp.sum(xp.sum(v * v, axis=2, keepdims=True), axis=1, keepdims=True)) + 1e-8
        return v * (g / norm)

    def _conv1d_causal(self, xp, x, params, prefix, dilation=1):
        w = self._conv_weight(xp, params, prefix)
        b = params[f'{prefix}_b']
        batch, channels, length = x.shape
        kernel_size = w.shape[2]
        pad = int(dilation) * (int(kernel_size) - 1)
        if pad > 0:
            x = xp.concatenate([xp.zeros((batch, channels, pad)), x], axis=2)
        out = xp.zeros((batch, w.shape[0], length))
        for i in range(int(kernel_size)):
            segment = x[:, :, i * int(dilation):i * int(dilation) + length]
            segment = xp.transpose(segment, (0, 2, 1))
            wi = xp.transpose(w[:, :, i], (1, 0))
            out = out + xp.transpose(segment @ wi, (0, 2, 1))
        return out + b.reshape((1, -1, 1))

    def _order_cqr_output(self, xp, out):
        if self._cqr_enabled:
            f = out.shape[1] // 3
            q = xp.stack([out[:, :f], out[:, f:2 * f], out[:, 2 * f:]], axis=-1)
            q = xp.sort(q, axis=-1)
            out = xp.concatenate([q[..., 0], q[..., 1], q[..., 2]], axis=-1)
        return out

    def _nbeats_fixed_basis(self, xp, block_type, coeff_dim, input_dim, output_dim):
        if block_type == 'trend':
            degree = coeff_dim - 1
            backcast_time = np.arange(input_dim, dtype=np.float32) / input_dim
            forecast_time = np.arange(output_dim, dtype=np.float32) / output_dim
            backcast_basis = np.stack([backcast_time ** i for i in range(degree + 1)]).astype(np.float32)
            forecast_basis = np.stack([forecast_time ** i for i in range(degree + 1)]).astype(np.float32)
        else:
            num_harmonics = coeff_dim // 2
            backcast_time = np.arange(input_dim, dtype=np.float32) / input_dim
            forecast_time = np.arange(output_dim, dtype=np.float32) / output_dim
            backcast_basis = np.concatenate([
                np.stack([np.cos(2 * math.pi * k * backcast_time) for k in range(1, num_harmonics + 1)]),
                np.stack([np.sin(2 * math.pi * k * backcast_time) for k in range(1, num_harmonics + 1)])
            ], axis=0).astype(np.float32)
            forecast_basis = np.concatenate([
                np.stack([np.cos(2 * math.pi * k * forecast_time) for k in range(1, num_harmonics + 1)]),
                np.stack([np.sin(2 * math.pi * k * forecast_time) for k in range(1, num_harmonics + 1)])
            ], axis=0).astype(np.float32)
        return xp.array(backcast_basis), xp.array(forecast_basis)

    def _max_pool1d_ceil(self, xp, x, kernel_size):
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return x
        pooled = []
        for start in range(0, int(x.shape[1]), kernel_size):
            pooled.append(xp.max(x[:, start:start + kernel_size], axis=1, keepdims=True))
        return xp.concatenate(pooled, axis=1)

    def _linear_interpolate1d(self, xp, x, target_len):
        target_len = int(target_len)
        source_len = int(x.shape[1])
        if source_len == target_len:
            return x
        if source_len == 1:
            return self._tile_to(xp, x, target_len)
        scale = source_len / target_len
        values = []
        for idx in range(target_len):
            pos = max(0.0, (idx + 0.5) * scale - 0.5)
            left = int(math.floor(pos))
            right = min(left + 1, source_len - 1)
            weight = np.float32(pos - left)
            values.append(x[:, left:left + 1] * np.float32(1.0 - weight) + x[:, right:right + 1] * weight)
        return xp.concatenate(values, axis=1)

    def _forward_nhits_exact(self, xp, params, x_flat_raw):
        x_norm, mean, std = self._revin_normalize(xp, x_flat_raw, enabled=self.use_revin)
        _, _, num_layers, _, _, _ = self._nhits_config()
        output_dim = int(self.out_features) * (3 if self._cqr_enabled else 1)
        residual = x_norm
        forecast = xp.zeros((x_norm.shape[0], output_dim))
        for block_idx, pool_kernel_size, _ in self._nhits_blocks(x_flat_raw.shape[1], output_dim):
            h = self._max_pool1d_ceil(xp, residual, pool_kernel_size)
            for layer_idx in range(num_layers):
                h = self._gelu(xp, self._linear_named(h, params, f'nh_b{block_idx}_fc{layer_idx}'))
            backcast_coeff = self._linear_named(h, params, f'nh_b{block_idx}_backcast')
            forecast_coeff = self._linear_named(h, params, f'nh_b{block_idx}_forecast')
            backcast = self._linear_interpolate1d(xp, backcast_coeff, x_flat_raw.shape[1])
            block_forecast = self._linear_interpolate1d(xp, forecast_coeff, output_dim)
            residual = residual - backcast
            forecast = forecast + block_forecast
        forecast = forecast * self._tile_to(xp, std, forecast.shape[1]) + self._tile_to(xp, mean, forecast.shape[1])
        return self._order_cqr_output(xp, forecast)

    def _forward_nbeats_exact(self, xp, params, x_flat_raw):
        x_norm, mean, std = self._revin_normalize(xp, x_flat_raw, enabled=self.use_revin)
        _, _, _, num_layers, _, _, _ = self._nbeats_config()
        residual = x_norm
        output_dim = int(self.out_features) * (3 if self._cqr_enabled else 1)
        forecast = xp.zeros((x_norm.shape[0], output_dim))
        for block_idx, block_type, coeff_dim in self._nbeats_blocks():
            h = residual
            for layer_idx in range(num_layers):
                h = self._gelu(xp, self._linear_named(h, params, f'nb_b{block_idx}_fc{layer_idx}'))
            backcast_coeff = self._linear_named(h, params, f'nb_b{block_idx}_backcast_coeff')
            forecast_coeff = self._linear_named(h, params, f'nb_b{block_idx}_forecast_coeff')
            if block_type == 'generic':
                backcast = self._linear_named(backcast_coeff, params, f'nb_b{block_idx}_backcast_basis')
                block_forecast = self._linear_named(forecast_coeff, params, f'nb_b{block_idx}_forecast_basis')
            else:
                backcast_basis, forecast_basis = self._nbeats_fixed_basis(
                    xp, block_type, coeff_dim, x_flat_raw.shape[1], output_dim
                )
                backcast = backcast_coeff @ backcast_basis
                block_forecast = forecast_coeff @ forecast_basis
            residual = residual - backcast
            forecast = forecast + block_forecast
        forecast = forecast * self._tile_to(xp, std, forecast.shape[1]) + self._tile_to(xp, mean, forecast.shape[1])
        return self._order_cqr_output(xp, forecast)

    def _forward_tcn_exact(self, xp, params, x_flat_raw):
        x_norm, mean, std = self._revin_normalize(xp, x_flat_raw, enabled=self.use_revin)
        num_levels, hidden_channels, _ = self._tcn_config()
        h = x_norm.reshape((x_norm.shape[0], 1, x_norm.shape[1]))
        for i in range(num_levels):
            in_ch = 1 if i == 0 else hidden_channels
            dilation = 2 ** i
            out = self._conv1d_causal(xp, h, params, f'tcn_l{i}_c1', dilation=dilation)
            out = self._gelu(xp, out)
            out = self._conv1d_causal(xp, out, params, f'tcn_l{i}_c2', dilation=dilation)
            out = self._gelu(xp, out)
            res = h if in_ch == hidden_channels else self._conv1d_causal(xp, h, params, f'tcn_l{i}_down', dilation=1)
            h = self._gelu(xp, out + res)
            h = xp.transpose(self._layer_norm(xp, xp.transpose(h, (0, 2, 1)), params, f'tcn_l{i}_ln'), (0, 2, 1))
        pooled = xp.mean(h, axis=2)
        out = self._linear_named(pooled, params, 'tcn_out') + self._linear_named(x_norm, params, 'tcn_res')
        out = out * self._tile_to(xp, std, out.shape[1]) + self._tile_to(xp, mean, out.shape[1])
        return self._order_cqr_output(xp, out)

    def _mha(self, xp, h, params, prefix, nhead):
        q = self._linear_named(h, params, f'{prefix}_q')
        k = self._linear_named(h, params, f'{prefix}_k')
        v = self._linear_named(h, params, f'{prefix}_v')
        batch, length, d_model = q.shape
        head_dim = d_model // nhead
        q = xp.transpose(q.reshape((batch, length, nhead, head_dim)), (0, 2, 1, 3))
        k = xp.transpose(k.reshape((batch, length, nhead, head_dim)), (0, 2, 1, 3))
        v = xp.transpose(v.reshape((batch, length, nhead, head_dim)), (0, 2, 1, 3))
        scores = (q @ xp.transpose(k, (0, 1, 3, 2))) / math.sqrt(max(head_dim, 1))
        attn = self._softmax(xp, scores, axis=-1)
        out = attn @ v
        out = xp.transpose(out, (0, 2, 1, 3)).reshape((batch, length, d_model))
        return self._linear_named(out, params, f'{prefix}_attn_out')

    def _forward_transformer_exact(self, xp, params, x_flat_raw):
        x_norm, mean, std = self._revin_normalize(xp, x_flat_raw, enabled=self.use_revin)
        d_model, nhead, num_layers, _, output_strategy = self._transformer_config()
        pooled = output_strategy == 'pooled'
        h = self._linear_named(x_norm.reshape((x_norm.shape[0], x_norm.shape[1], 1)), params, 'tf_input')
        h = h + params['tf_pos'][:, :x_norm.shape[1], :]
        for i in range(num_layers):
            z = self._layer_norm(xp, h, params, f'tf_l{i}_ln1')
            h = h + self._mha(xp, z, params, f'tf_l{i}', nhead)
            z = self._layer_norm(xp, h, params, f'tf_l{i}_ln2')
            ff = self._linear_named(self._gelu(xp, self._linear_named(z, params, f'tf_l{i}_ff1')), params, f'tf_l{i}_ff2')
            h = h + ff
        h = self._layer_norm(xp, h, params, 'tf_final_ln')
        if pooled:
            weights = self._softmax(xp, self._linear_named(h, params, 'tf_temporal_weight').reshape((h.shape[0], h.shape[1])), axis=1)
            h = xp.sum(h * weights.reshape((weights.shape[0], weights.shape[1], 1)), axis=1)
            out = self._linear_named(self._gelu(xp, self._linear_named(h, params, 'tf_head1')), params, 'tf_head2')
            out = out + self._linear_named(x_norm, params, 'tf_res')
        else:
            h = h.reshape((h.shape[0], h.shape[1] * d_model))
            out = self._linear_named(self._gelu(xp, self._linear_named(h, params, 'tf_head1')), params, 'tf_head2')
        out = out * self._tile_to(xp, std, out.shape[1]) + self._tile_to(xp, mean, out.shape[1])
        return self._order_cqr_output(xp, out)

    def _forward(self, xp, params, x):
        if len(x.shape) == 3:
            x_flat_raw = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        else:
            x_flat_raw = x

        if self.kind == 'nhits':
            return self._forward_nhits_exact(xp, params, x_flat_raw)
        if self.kind == 'nbeats':
            return self._forward_nbeats_exact(xp, params, x_flat_raw)
        if self.kind == 'tcn':
            return self._forward_tcn_exact(xp, params, x_flat_raw)
        if self.kind == 'transformer':
            return self._forward_transformer_exact(xp, params, x_flat_raw)

        if self.use_revin:
            mean = xp.mean(x_flat_raw, axis=1, keepdims=True)
            std = xp.sqrt(xp.mean((x_flat_raw - mean) ** 2, axis=1, keepdims=True)) + 1e-5
            x_flat = (x_flat_raw - mean) / std
        else:
            mean = xp.zeros((x_flat_raw.shape[0], 1))
            std = xp.ones((x_flat_raw.shape[0], 1))
            x_flat = x_flat_raw

        feat = self._features(xp, params, x_flat, x)
        h = self._gelu(xp, self._linear(xp, feat, params, 'w_in'))
        for i in range(max(self.num_layers - 1, 0)):
            h_next = self._gelu(xp, self._linear(xp, h, params, f'w_{i}'))
            h = h + h_next if h_next.shape == h.shape else h_next
        out = self._linear(xp, h, params, 'w_out')
        residual = self._linear(xp, x_flat, params, 'w_res')

        if self.kind in {'nbeats', 'nhits'}:
            out = out + 0.5 * residual
        elif self._residual_gate_enabled:
            alpha = params['gate_alpha']
            out = alpha * out + (1.0 - alpha) * residual
        else:
            out = out + residual

        output_dim = out.shape[1]
        out = out * self._tile_to(xp, std, output_dim) + self._tile_to(xp, mean, output_dim)

        if self._cqr_enabled:
            f = output_dim // 3
            q = xp.stack([out[:, :f], out[:, f:2 * f], out[:, 2 * f:]], axis=-1)
            q = xp.sort(q, axis=-1)
            out = xp.concatenate([q[..., 0], q[..., 1], q[..., 2]], axis=-1)
        return out

    def _point_loss(self, xp, pred, target):
        err = pred - target
        if self.loss_fn_name == 'mse':
            return xp.mean(err ** 2)
        if self.loss_fn_name == 'mae':
            return xp.mean(xp.abs(err))
        abs_err = xp.abs(err)
        return xp.mean(xp.where(abs_err <= 1.0, 0.5 * err ** 2, abs_err - 0.5))

    def _pinball(self, xp, pred, target, tau):
        err = target - pred
        return xp.mean(xp.maximum(tau * err, (tau - 1.0) * err))

    def _loss(self, xp, params, xb, yb):
        pred = self._forward(xp, params, xb)
        if self._cqr_enabled:
            f = yb.shape[-1]
            alpha = self._cqr_alpha
            return (
                self._pinball(xp, pred[:, :f], yb, alpha / 2.0) +
                self._pinball(xp, pred[:, f:2 * f], yb, 0.5) +
                self._pinball(xp, pred[:, 2 * f:], yb, 1.0 - alpha / 2.0)
            ) / 3.0
        return self._point_loss(xp, pred, yb)

    def _format_prediction(self, flat):
        flat = np.asarray(flat)
        if self._cqr_enabled:
            return flat
        if self._output_shape is None or len(self._output_shape) <= 1:
            return flat.reshape(flat.shape[0], -1)
        return flat.reshape((flat.shape[0],) + tuple(self._output_shape))


class MLXNativeNN(_BaseNativeNN):
    backend = 'mlx'

    def _tile_to(self, xp, x, width):
        return xp.broadcast_to(x, (x.shape[0], width))

    def _tree_map(self, fn, *trees):
        return {k: fn(*(tree[k] for tree in trees)) for k in trees[0].keys()}

    def _copy_params(self, mx, params):
        return {k: mx.array(np.array(v)) for k, v in params.items()}

    def fit(self, X, y, epochs=1000, batch_size='auto', eval_set=None, loss_type='min',
            metrics_name='score', monitor='val_loss', min_delta=0, patience=10,
            lr_scheduler='CosineAnnealingLR', lr_scheduler_patience=10, lr_factor=0.7,
            restore_best_weights=True, verbose=True, **kwargs):
        import mlx.core as mx

        X_np = _as_float_array(X)
        y_np, output_shape = _flatten_target(y)
        self._output_shape = output_shape
        input_dim = self._input_dim_from_X(X_np)
        target_dim = y_np.shape[1]
        params = {k: mx.array(v) for k, v in self._init_np_params(input_dim, target_dim).items()}
        m = self._tree_map(lambda z: mx.zeros_like(z), params)
        v = self._tree_map(lambda z: mx.zeros_like(z), params)
        t = 0
        X_mx = mx.array(X_np)
        y_mx = mx.array(y_np)
        n = X_np.shape[0]
        bs = _batch_size(n, batch_size)
        eval_xy = None
        if eval_set is not None:
            if isinstance(eval_set, list):
                eval_set = eval_set[0]
            eval_xy = (mx.array(_as_float_array(eval_set[0])), mx.array(_flatten_target(eval_set[1])[0]))

        def loss_fn(p, xb, yb):
            return self._loss(mx, p, xb, yb)

        value_and_grad = mx.value_and_grad(loss_fn)
        best_params = self._copy_params(mx, params)
        best_loss = float('inf')
        wait = 0
        rng = np.random.RandomState(self.random_seed)
        for epoch in range(int(epochs)):
            lr = self.learning_rate
            if lr_scheduler == 'CosineAnnealingLR':
                lr = max(self.learning_rate * (0.5 * (1.0 + math.cos(math.pi * epoch / max(int(epochs), 1)))), 1e-7)
            order = rng.permutation(n)
            losses = []
            for start in range(0, n, bs):
                idx = mx.array(order[start:start + bs])
                xb = X_mx[idx]
                yb = y_mx[idx]
                loss, grads = value_and_grad(params, xb, yb)
                t += 1
                m = self._tree_map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
                v = self._tree_map(lambda a, g: 0.999 * a + 0.001 * (g * g), v, grads)
                m_hat = self._tree_map(lambda a: a / (1.0 - 0.9 ** t), m)
                v_hat = self._tree_map(lambda a: a / (1.0 - 0.999 ** t), v)
                params = self._tree_map(
                    lambda w, mh, vh: w - lr * (mh / (mx.sqrt(vh) + 1e-8) + self.weight_decay * w),
                    params, m_hat, v_hat
                )
                mx.eval(params)
                losses.append(float(np.array(loss)))
            train_loss = float(np.mean(losses)) if losses else float('inf')
            if eval_xy is not None and monitor == 'val_loss':
                val_loss_arr = loss_fn(params, eval_xy[0], eval_xy[1])
                mx.eval(val_loss_arr)
                val_loss = float(np.array(val_loss_arr))
                current = val_loss
            else:
                val_loss = None
                current = train_loss
            self.training_logs['epochs'].append(epoch)
            self.training_logs['train_loss'].append(train_loss)
            self.training_logs['lrs'].append(lr)
            if val_loss is not None:
                self.training_logs['test_loss'].append(val_loss)
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                print(msg)
            if current < best_loss - min_delta:
                best_loss = current
                best_params = self._copy_params(mx, params)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self._params = best_params if restore_best_weights else params
        self.__pipelinets_is_fitted__ = True
        return self

    def predict(self, X):
        if self._params is None:
            raise RuntimeError("This model is not fitted yet.")
        import mlx.core as mx
        pred = self._forward(mx, self._params, mx.array(_as_float_array(X)))
        mx.eval(pred)
        return self._format_prediction(np.array(pred))


class NativeNNModel(ForecastingMixin):
    def __init__(self, kind, device='auto', torch_cls=None, **kwargs):
        if 'backend' in kwargs:
            raise TypeError("backend selection is automatic and is not a public option.")
        unsupported_torch_only = kwargs.get('use_gtb', False)
        resolved = resolve_nn_backend(device=device, prefer_torch=unsupported_torch_only)
        if resolved == 'torch':
            if torch_cls is None:
                raise ImportError("The torch backend implementation is not available for this model.")
            torch_kwargs = dict(kwargs)
            torch_kwargs.setdefault('device', device)
            self._impl = torch_cls(**torch_kwargs)
        else:
            self._impl = MLXNativeNN(kind, **kwargs)
        self.backend = resolved

    def __getattr__(self, name):
        if name != '_impl' and '_impl' in self.__dict__:
            return getattr(self.__dict__['_impl'], name)
        raise AttributeError(name)

    def fit(self, *args, **kwargs):
        self._impl.fit(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        return self._impl.predict(*args, **kwargs)

    def _enable_cqr(self, *args, **kwargs):
        return self._impl._enable_cqr(*args, **kwargs)

    def _enable_residual_gate(self, *args, **kwargs):
        return self._impl._enable_residual_gate(*args, **kwargs)


def make_native_dispatcher(kind, torch_cls):
    class _NativeDispatcher(NativeNNModel):
        def __init__(self, *args, device='auto', **kwargs):
            if args:
                names = ['in_features', 'out_features']
                for name, value in zip(names, args):
                    kwargs.setdefault(name, value)
            super().__init__(kind, device=device, torch_cls=torch_cls, **kwargs)

    _NativeDispatcher.__name__ = getattr(torch_cls, '__name__', kind)
    return _NativeDispatcher
