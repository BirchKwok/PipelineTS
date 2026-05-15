import numpy as np

from PipelineTS.spinesTS.backends import is_torch_available
from PipelineTS.spinesTS.backends._native_models import MLXNativeNN


def count_np_params(params):
    return int(sum(np.prod(np.asarray(v).shape) for v in params.values()))


def count_torch_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def check_tcn():
    from PipelineTS.spinesTS.nn._tcn import TemporalConvNet

    cfg = {
        'in_features': 12,
        'out_features': 12,
        'kernel_size': 3,
        'dropout': 0.0,
        'num_levels': 2,
        'hidden_channels': 16,
        'use_gtb': False,
    }
    torch_model = TemporalConvNet(**cfg)
    native_model = MLXNativeNN('tcn', **cfg)
    torch_count = count_torch_params(torch_model)
    native_count = count_np_params(native_model._init_np_params(cfg['in_features'], cfg['out_features']))
    return 'tcn', torch_count, native_count


def check_nbeats(generic_architecture):
    from PipelineTS.spinesTS.nn._n_beats import NBEATSBackbone

    cfg = {
        'in_features': 12,
        'out_features': 12,
        'generic_architecture': generic_architecture,
        'num_stacks': 1 if generic_architecture else 2,
        'num_blocks': 1,
        'num_layers': 2,
        'layer_widths': 64,
        'expansion_coeff_dim': 8,
        'trend_degree': 3,
        'dropout': 0.0,
        'use_revin': True,
        'use_gtb': False,
    }
    torch_model = NBEATSBackbone(**cfg)
    native_model = MLXNativeNN('nbeats', **cfg)
    torch_count = count_torch_params(torch_model)
    native_count = count_np_params(native_model._init_np_params(cfg['in_features'], cfg['out_features']))
    suffix = 'generic' if generic_architecture else 'interpretable'
    return f'nbeats_{suffix}', torch_count, native_count


def check_nhits():
    from PipelineTS.spinesTS.nn._n_hits import NHiTSBackbone

    cfg = {
        'in_features': 12,
        'out_features': 12,
        'num_stacks': 3,
        'num_blocks': 1,
        'num_layers': 2,
        'layer_widths': 64,
        'pooling_kernel_sizes': None,
        'n_freq_downsample': None,
        'dropout': 0.0,
        'use_revin': True,
        'use_gtb': False,
    }
    torch_model = NHiTSBackbone(**cfg)
    native_model = MLXNativeNN('nhits', **cfg)
    torch_count = count_torch_params(torch_model)
    native_count = count_np_params(native_model._init_np_params(cfg['in_features'], cfg['out_features']))
    return 'nhits', torch_count, native_count


def check_transformer():
    from PipelineTS.spinesTS.nn._transformer import TransformerBackbone

    cfg = {
        'in_features': 12,
        'out_features': 12,
        'd_model': 32,
        'nhead': 2,
        'num_encoder_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.0,
        'use_revin': True,
        'output_strategy': 'flatten',
        'use_gtb': False,
    }
    torch_model = TransformerBackbone(**cfg)
    native_model = MLXNativeNN('transformer', **cfg)
    torch_count = count_torch_params(torch_model)
    native_count = count_np_params(native_model._init_np_params(cfg['in_features'], cfg['out_features']))
    return 'transformer', torch_count, native_count


def main():
    if not is_torch_available():
        raise RuntimeError('torch is required for architecture parity checks')

    failures = []
    checks = [
        check_nbeats(True),
        check_nbeats(False),
        check_nhits(),
        check_tcn(),
        check_transformer(),
    ]
    for name, torch_count, native_count in checks:
        status = 'OK' if torch_count == native_count else 'FAIL'
        print(f'{status} {name}: torch_params={torch_count} native_params={native_count}')
        if status != 'OK':
            failures.append(name)

    if failures:
        raise SystemExit(f'parameter count mismatch: {failures}')


if __name__ == '__main__':
    main()
