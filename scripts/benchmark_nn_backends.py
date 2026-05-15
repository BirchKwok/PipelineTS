import argparse
import inspect
import statistics
import time
import warnings

import numpy as np
import pandas as pd

from PipelineTS.spinesTS.backends import is_mlx_available, is_torch_available, resolve_nn_backend

warnings.filterwarnings('ignore')


MODEL_CLASS_NAMES = {
    'd_linear': 'DLinearModel',
    'n_linear': 'NLinearModel',
    'n_beats': 'NBeatsModel',
    'n_hits': 'NHitsModel',
    'tide': 'TiDEModel',
    'tcn': 'TCNModel',
    'patch_rnn': 'PatchRNNModel',
    'stacking_rnn': 'StackingRNNModel',
    'time2vec': 'Time2VecModel',
    'gau': 'GAUModel',
    'transformer': 'TransformerModel',
    'tft': 'TFTModel',
    'deepar': 'DeepARModel',
    'itransformer': 'ITransformerModel',
    'srs_net': 'SRSNetModel',
}


LIGHT_CONFIGS = {
    'd_linear': {'dropout': 0.0, 'use_gtb': False, 'use_residual_gate': False},
    'n_linear': {'dropout': 0.0, 'use_gtb': False, 'use_residual_gate': False},
    'n_beats': {'num_stacks': 1, 'num_blocks': 1, 'num_layers': 2, 'layer_widths': 64, 'expansion_coeff_dim': 8, 'dropout': 0.0, 'use_gtb': False, 'use_residual_gate': False},
    'n_hits': {'num_stacks': 1, 'num_blocks': 1, 'num_layers': 1, 'layer_widths': 64, 'dropout': 0.0, 'use_gtb': False, 'use_residual_gate': False},
    'tide': {'hidden_size': 64, 'decoder_output_dim': 16, 'temporal_width_past': 4, 'temporal_width_future': 4, 'num_encoder_layers': 1, 'num_decoder_layers': 1, 'dropout': 0.05, 'use_gtb': False, 'use_residual_gate': False},
    'tcn': {'num_levels': 2, 'hidden_channels': 16, 'dropout': 0.0, 'use_gtb': False, 'use_residual_gate': False},
    'patch_rnn': {'kernel_size': 6, 'multi_steps': True, 'dropout': 0.05, 'use_gtb': False, 'use_residual_gate': False},
    'stacking_rnn': {'blocks': 1, 'dropout': 0.05, 'use_gtb': False, 'use_residual_gate': False},
    'time2vec': {'dropout': 0.05, 'use_gtb': False, 'use_residual_gate': False},
    'gau': {'level': 1, 'dropout': 0.05, 'use_gtb': False, 'use_residual_gate': False},
    'transformer': {'d_model': 32, 'nhead': 2, 'num_encoder_layers': 1, 'dim_feedforward': 64, 'dropout': 0.0, 'use_gtb': False, 'use_residual_gate': False},
    'tft': {'hidden_size': 32, 'lstm_layers': 1, 'n_heads': 2, 'dropout': 0.05, 'use_gtb': False, 'use_residual_gate': False},
    'deepar': {'d_model': 32, 'n_blocks': 1, 'n_rwkv_blocks': 1, 'dropout': 0.05, 'use_residual_gate': False},
    'itransformer': {'d_model': 32, 'n_heads': 2, 'd_ff': 64, 'e_layers': 1, 'dropout': 0.05, 'use_residual_gate': False},
    'srs_net': {'d_model': 32, 'n_heads': 2, 'top_k_ratio': 0.35, 'dropout': 0.05, 'use_residual_gate': False},
}


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.float64)
    value = 10.0 + 0.02 * idx + 2.0 * np.sin(idx / 7.0) + rng.normal(0.0, 0.2, n)
    return pd.DataFrame({'date': pd.date_range('2024-01-01', periods=n, freq='D'), 'value': value})


def make_multivariate_data(n, seed):
    data = make_data(n, seed)
    idx = np.arange(n, dtype=np.float64)
    data['feature_a'] = np.cos(idx / 7.0)
    data['feature_b'] = np.sin(idx / 31.0)
    return data


def import_model_class(model_name):
    import PipelineTS.nn_model as nn_model
    return getattr(nn_model, MODEL_CLASS_NAMES[model_name])


def supported_kwargs(cls, kwargs):
    sig = inspect.signature(cls)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def sync_backend(backend):
    if backend == 'torch' and is_torch_available():
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize') and torch.backends.mps.is_available():
            torch.mps.synchronize()


def elapsed_seconds(backend, func):
    start = time.perf_counter()
    value = func()
    sync_backend(backend)
    return time.perf_counter() - start, value


def run_once(model_name, n, lags, epochs, accelerator, seed, predict_warmup, predict_repeat):
    cls = import_model_class(model_name)
    is_multi = model_name in {'itransformer', 'srs_net'}
    data = make_multivariate_data(n, seed) if is_multi else make_data(n, seed)
    common = {
        'time_col': 'date',
        'target_col': 'value',
        'lags': lags,
        'quantile': None,
        'epochs': epochs,
        'patience': epochs + 1,
        'batch_size': 512,
        'lr_scheduler': None,
        'restore_best_weights': False,
        'accelerator': accelerator,
        'random_state': seed,
        'verbose': False,
    }
    if is_multi:
        common['feature_cols'] = ['value', 'feature_a', 'feature_b']
    common.update(LIGHT_CONFIGS.get(model_name, {}))
    kwargs = supported_kwargs(cls, common)
    model = cls(**kwargs)
    resolved = getattr(model.model, 'backend', resolve_nn_backend())
    backend = resolved
    fit_seconds, _ = elapsed_seconds(backend, lambda: model.fit(data))
    for _ in range(predict_warmup):
        model.predict(lags)
        sync_backend(backend)
    predict_times = []
    pred = None
    for _ in range(predict_repeat):
        predict_seconds, pred = elapsed_seconds(backend, lambda: model.predict(lags))
        predict_times.append(predict_seconds)
    if len(pred) != lags or pred['value'].isna().any():
        raise RuntimeError('invalid prediction output')
    return fit_seconds, statistics.mean(predict_times), min(predict_times), resolved


def summarize(values):
    if not values:
        return None, None
    return statistics.mean(values), min(values)


def benchmark(args):
    models = [x.strip() for x in args.models.split(',') if x.strip()]
    rows = []
    for model_name in models:
        fit_values = []
        pred_values = []
        pred_min_values = []
        resolved = None
        error = None
        for repeat_idx in range(args.repeat):
            try:
                fit_s, pred_s, pred_min_s, resolved = run_once(
                    model_name, args.n, args.lags, args.epochs, args.accelerator,
                    args.seed + repeat_idx, args.predict_warmup, args.predict_repeat
                )
                fit_values.append(fit_s)
                pred_values.append(pred_s)
                pred_min_values.append(pred_min_s)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                break
        if error:
            rows.append({'model': model_name, 'backend': resolved or 'auto', 'status': 'error', 'error': error})
        else:
            fit_mean, fit_min = summarize(fit_values)
            pred_mean, _ = summarize(pred_values)
            _, pred_min = summarize(pred_min_values)
            rows.append({
                'model': model_name,
                'backend': resolved,
                'status': 'ok',
                'fit_mean': fit_mean,
                'fit_min': fit_min,
                'predict_mean': pred_mean,
                'predict_min': pred_min,
            })
    return rows


def print_rows(rows):
    print(f"available torch={is_torch_available()} mlx={is_mlx_available()} selected={resolve_nn_backend()}")
    print(f"{'model':<16} {'backend':<8} {'status':<8} {'fit_mean':>10} {'fit_min':>10} {'pred_mean':>10} {'pred_min':>10} error")
    print('-' * 104)
    for row in rows:
        if row['status'] == 'ok':
            print(
                f"{row['model']:<16} {row['backend']:<8} {row['status']:<8} "
                f"{row['fit_mean']:>10.3f} {row['fit_min']:>10.3f} "
                f"{row['predict_mean']:>10.4f} {row['predict_min']:>10.4f}"
            )
        else:
            print(f"{row['model']:<16} {row['backend']:<8} {row['status']:<8} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {row.get('error', '')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='n_linear,d_linear,tcn,n_beats,tide,transformer')
    parser.add_argument('--n', type=int, default=240)
    parser.add_argument('--lags', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--predict-warmup', type=int, default=1)
    parser.add_argument('--predict-repeat', type=int, default=3)
    parser.add_argument('--accelerator', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = benchmark(args)
    print_rows(rows)


if __name__ == '__main__':
    main()
