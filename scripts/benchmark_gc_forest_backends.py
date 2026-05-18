import argparse
import importlib.util
import statistics
import time

import numpy as np
import pandas as pd

from PipelineTS.ml_model import gcForestModel


def make_series(kind, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.float64)
    if kind == 'smooth':
        values = 10.0 + 2.0 * np.sin(idx / 8.0) + 0.15 * rng.standard_normal(n)
    elif kind == 'trend':
        values = 5.0 + 0.035 * idx + 1.5 * np.sin(idx / 12.0) + 0.25 * rng.standard_normal(n)
    elif kind == 'regime':
        shift = np.where(idx > n * 0.55, 2.5, 0.0)
        values = 8.0 + shift + 1.2 * np.sin(idx / 6.0) + 0.35 * rng.standard_normal(n)
    else:
        raise ValueError(f'unknown case: {kind}')
    return pd.DataFrame({'date': pd.date_range('2022-01-01', periods=n, freq='D'), 'value': values})


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2) ** 0.5)


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(denom == 0.0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)))


def sync_accelerator():
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize') and torch.backends.mps.is_available():
            torch.mps.synchronize()
    try:
        import mlx.core as mx
        mx.eval()
    except Exception:
        pass


def elapsed(func):
    start = time.perf_counter()
    value = func()
    sync_accelerator()
    return time.perf_counter() - start, value


def run_once(kind, accelerator, n, horizon, lags, seed, args):
    data = make_series(kind, n + horizon, seed)
    train = data.iloc[:-horizon].reset_index(drop=True)
    truth = data.iloc[-horizon:]['value'].to_numpy(dtype=np.float64)
    model = gcForestModel(
        time_col='date',
        target_col='value',
        lags=lags,
        quantile=None,
        n_layers=args.n_layers,
        n_estimators_per_layer=args.n_estimators_per_layer,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=seed,
        accelerator=accelerator,
        ridge_alpha=args.ridge_alpha,
        max_tree_depth=args.max_tree_depth,
        temperature=args.temperature,
    )
    fit_seconds, _ = elapsed(lambda: model.fit(train))
    predict_seconds, pred = elapsed(lambda: model.predict(horizon))
    pred_values = pred['value'].to_numpy(dtype=np.float64)
    resolved = getattr(model.model, 'resolved_accelerator_', accelerator)
    return {
        'case': kind,
        'accelerator': accelerator,
        'resolved': resolved,
        'fit_s': fit_seconds,
        'predict_ms': predict_seconds * 1000.0,
        'mae': mae(truth, pred_values),
        'rmse': rmse(truth, pred_values),
        'smape': smape(truth, pred_values),
    }


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row['case'], row['accelerator'], row['resolved'])
        grouped.setdefault(key, []).append(row)
    out = []
    for (case, accelerator, resolved), values in grouped.items():
        item = {'case': case, 'accelerator': accelerator, 'resolved': resolved}
        for key in ['fit_s', 'predict_ms', 'mae', 'rmse', 'smape']:
            item[key] = statistics.mean(v[key] for v in values)
        out.append(item)
    return out


def print_table(rows):
    header = f"{'case':<10} {'accelerator':<12} {'resolved':<8} {'fit_s':>8} {'pred_ms':>9} {'MAE':>10} {'RMSE':>10} {'SMAPE':>10}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f"{row['case']:<10} {row['accelerator']:<12} {row['resolved']:<8} "
            f"{row['fit_s']:>8.3f} {row['predict_ms']:>9.2f} "
            f"{row['mae']:>10.4f} {row['rmse']:>10.4f} {row['smape']:>10.4f}"
        )


def available_accelerators(requested):
    available = []
    for accelerator in requested:
        if accelerator == 'mlx' and importlib.util.find_spec('mlx') is None:
            continue
        if accelerator == 'torch' and importlib.util.find_spec('torch') is None:
            continue
        available.append(accelerator)
    return available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='smooth,trend,regime')
    parser.add_argument('--n', type=int, default=240)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--lags', type=int, default=12)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accelerators', default='auto,mlx,torch,sklearn')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-estimators-per-layer', type=int, default=48)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--min-samples-leaf', type=int, default=1)
    parser.add_argument('--ridge-alpha', type=float, default=1e-3)
    parser.add_argument('--max-tree-depth', type=int, default=6)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    cases = [case.strip() for case in args.cases.split(',') if case.strip()]
    accelerators = available_accelerators(
        [x.strip() for x in args.accelerators.split(',') if x.strip()]
    )
    rows = []
    for case in cases:
        for repeat_idx in range(args.repeat):
            seed = args.seed + repeat_idx
            for accelerator in accelerators:
                rows.append(run_once(case, accelerator, args.n, args.horizon, args.lags, seed, args))
    print_table(summarize(rows))


if __name__ == '__main__':
    main()
