"""Benchmark: GPU (TorchTree) vs CPU (native) tree models on REAL built-in datasets.

Tests on multiple real-world time series with different characteristics:
  - AirPassengers      (144 rows, monthly, strong trend+seasonality)
  - Electric_Production (396 rows, monthly, trend+seasonality)
  - Supermarket_Incoming(515 rows, daily, noisy)
  - Messages_Sent       (601 rows, daily, high-magnitude)
  - Web_Sales           (2088 rows, daily, moderate)
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from PipelineTS.spinesTS.data import BuiltInSeriesData

# ── Load all datasets ────────────────────────────────────────────────
loader = BuiltInSeriesData(print_file_list=False)

DATASETS = [
    # (name,       time_col, target_col, lags, predict_n)
    ('AirPassengers',       'Month', 'Passengers', 12, 12),
    ('Electric_Production', 'date',  'value',      16, 16),
    ('Supermarket_Incoming','date',  'goods_cnt',  14, 14),
    ('Messages_Sent',       'date',  'tc',         10, 10),
    ('Web_Sales',           'date',  'sales_cnt',  14, 14),
]

# ── Model imports ────────────────────────────────────────────────────
from PipelineTS.ml_model import LightGBMModel, XGBoostModel, RandomForestModel
try:
    from PipelineTS.ml_model import CatBoostModel
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from PipelineTS.ml_model import (
    TorchBoostingForestModel, TorchBaggingForestModel, DeepForestModel,
)

GPU_KW = dict(accelerator='cpu', random_state=42)


def bench_one(model_cls, data_df, time_col, target_col, lags, predict_n,
              extra_kw=None):
    """Train and predict, return (fit_time, pred_time, mse, mae)."""
    data_df = data_df.copy()
    data_df[time_col] = pd.to_datetime(data_df[time_col])
    kw = dict(time_col=time_col, target_col=target_col, lags=lags,
              quantile=None)
    kw.update(extra_kw or {})
    model = model_cls(**kw)

    t0 = time.perf_counter()
    model.fit(data_df)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = model.predict(predict_n)
    pred_time = time.perf_counter() - t0

    preds = result[target_col].values
    gt = data_df[target_col].values[-predict_n:]
    mse = mean_squared_error(gt, preds)
    mae = mean_absolute_error(gt, preds)
    return fit_time, pred_time, mse, mae


# ── Define model pairs: (label, model_class, extra_kwargs, group) ────
CPU_MODELS = [
    ('LightGBM',     LightGBMModel,     {}),
    ('XGBoost',      XGBoostModel,       {}),
    ('RandomForest', RandomForestModel,  {}),
]
if HAS_CATBOOST:
    CPU_MODELS.append(('CatBoost', CatBoostModel, {}))

GPU_MODELS = [
    ('TorchBoostingForest', TorchBoostingForestModel, GPU_KW),
    ('TorchBaggingForest',  TorchBaggingForestModel,  {**GPU_KW, 'dropout': 0.1}),
    ('DeepForest',          DeepForestModel,          {**GPU_KW, 'n_layers': 2}),
]


# ── Run benchmark ────────────────────────────────────────────────────
all_results = []  # list of dicts

print(f"\n{'='*90}")
print(f"  COMPREHENSIVE BENCHMARK: GPU (TorchTree) vs CPU on REAL Datasets")
print(f"{'='*90}")

for ds_name, time_col, target_col, lags, predict_n in DATASETS:
    data_df = loader[ds_name]
    n_rows = len(data_df)

    print(f"\n{'─'*90}")
    print(f"  Dataset: {ds_name}  |  rows={n_rows}  |  lags={lags}  |  predict={predict_n}")
    print(f"{'─'*90}")

    print(f"\n  {'Model':22s}  {'Fit':>7s}  {'Pred':>6s}  {'MSE':>12s}  {'MAE':>10s}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*6}  {'-'*12}  {'-'*10}")

    ds_results_cpu = {}
    ds_results_gpu = {}

    # CPU models
    for label, cls, extra in CPU_MODELS:
        try:
            ft, pt, mse, mae = bench_one(cls, data_df, time_col, target_col,
                                         lags, predict_n, extra)
            print(f"  {label:22s}  {ft:6.2f}s  {pt:5.3f}s  {mse:12.2f}  {mae:10.2f}")
            ds_results_cpu[label] = dict(fit=ft, mse=mse, mae=mae)
            all_results.append(dict(dataset=ds_name, model=label, group='CPU',
                                    fit=ft, pred=pt, mse=mse, mae=mae))
        except Exception as e:
            print(f"  {label:22s}  ERROR: {e}")

    print()

    # GPU models
    for label, cls, extra in GPU_MODELS:
        try:
            ft, pt, mse, mae = bench_one(cls, data_df, time_col, target_col,
                                         lags, predict_n, extra)
            print(f"  {label:22s}  {ft:6.2f}s  {pt:5.3f}s  {mse:12.2f}  {mae:10.2f}")
            ds_results_gpu[label] = dict(fit=ft, mse=mse, mae=mae)
            all_results.append(dict(dataset=ds_name, model=label, group='GPU',
                                    fit=ft, pred=pt, mse=mse, mae=mae))
        except Exception as e:
            print(f"  {label:22s}  ERROR: {e}")

# ── Grand summary ────────────────────────────────────────────────────
print(f"\n\n{'='*90}")
print(f"  GRAND SUMMARY  —  GPU vs CPU Comparison")
print(f"{'='*90}")

# Build comparison table: for each (dataset, model_type) pair, compare GPU vs CPU
PAIRS = [
    ('TorchBoostingForest', 'LightGBM'),
    ('TorchBoostingForest', 'XGBoost'),
    ('TorchBoostingForest', 'CatBoost'),
    ('TorchBaggingForest',  'RandomForest'),
]

df = pd.DataFrame(all_results)

print(f"\n  {'Dataset':22s}  {'Pair':30s}  {'GPU Time':>9s}  {'CPU Time':>9s}  "
      f"{'Speedup':>8s}  {'GPU MSE':>10s}  {'CPU MSE':>10s}  {'MSE Δ':>8s}")
print(f"  {'-'*22}  {'-'*30}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")

gpu_wins_speed = 0
gpu_wins_acc = 0
total_pairs = 0

for ds_name, _, _, _, _ in DATASETS:
    ds_df = df[df['dataset'] == ds_name]
    for gpu_name, cpu_name in PAIRS:
        gpu_row = ds_df[ds_df['model'] == gpu_name]
        cpu_row = ds_df[ds_df['model'] == cpu_name]
        if gpu_row.empty or cpu_row.empty:
            continue

        gpu_fit = gpu_row.iloc[0]['fit']
        cpu_fit = cpu_row.iloc[0]['fit']
        gpu_mse = gpu_row.iloc[0]['mse']
        cpu_mse = cpu_row.iloc[0]['mse']
        speedup = cpu_fit / gpu_fit
        mse_delta = (gpu_mse - cpu_mse) / cpu_mse * 100  # % change

        total_pairs += 1
        if speedup > 1.0:
            gpu_wins_speed += 1
        if gpu_mse <= cpu_mse:
            gpu_wins_acc += 1

        speed_marker = '✓' if speedup > 1.0 else '✗'
        acc_marker = '✓' if gpu_mse <= cpu_mse else '✗'

        pair_label = f"{gpu_name} vs {cpu_name}"
        print(f"  {ds_name:22s}  {pair_label:30s}  {gpu_fit:7.2f}s  {cpu_fit:7.2f}s  "
              f"{speedup:6.1f}x {speed_marker}  {gpu_mse:10.2f}  {cpu_mse:10.2f}  "
              f"{mse_delta:+6.1f}% {acc_marker}")

print(f"\n  {'─'*90}")
print(f"  Speed wins: {gpu_wins_speed}/{total_pairs}  |  "
      f"Accuracy wins (MSE ≤ CPU): {gpu_wins_acc}/{total_pairs}")
print(f"{'='*90}\n")

# ── Per-dataset best model ───────────────────────────────────────────
print(f"  Per-Dataset Best Model (lowest MSE):")
print(f"  {'-'*60}")
for ds_name, _, _, _, _ in DATASETS:
    ds_df = df[df['dataset'] == ds_name]
    if ds_df.empty:
        continue
    best = ds_df.loc[ds_df['mse'].idxmin()]
    print(f"  {ds_name:22s}  →  {best['model']:22s}  MSE={best['mse']:.2f}  "
          f"({best['group']})")
print(f"{'='*90}\n")
