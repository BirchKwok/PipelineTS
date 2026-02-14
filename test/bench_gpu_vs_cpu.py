"""Benchmark: GPU (TorchTree) vs CPU (native) tree models — speed & accuracy."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── synthetic time-series data ──────────────────────────────────────────
np.random.seed(42)
N = 200
dates = pd.date_range('2020-01-01', periods=N, freq='D')
trend = np.linspace(0, 2, N)
season = np.sin(np.linspace(0, 8 * np.pi, N))
noise = np.random.randn(N) * 0.15
values = trend + season + noise
data = pd.DataFrame({'date': dates, 'value': values})

LAGS = 10
PREDICT_N = 10
COMMON = dict(time_col='date', target_col='value', lags=LAGS, quantile=None)


def bench_model(name, model_cls, extra_kw=None):
    kw = {**COMMON, **(extra_kw or {})}
    model = model_cls(**kw)
    t0 = time.perf_counter()
    model.fit(data)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = model.predict(PREDICT_N)
    pred_time = time.perf_counter() - t0
    preds = result['value'].values
    # Use last PREDICT_N of real data as pseudo-ground-truth for relative comparison
    gt = values[-PREDICT_N:]
    mse = mean_squared_error(gt, preds)
    mae = mean_absolute_error(gt, preds)
    print(f"  {name:25s}  fit={fit_time:6.2f}s  pred={pred_time:5.3f}s  "
          f"MSE={mse:.4f}  MAE={mae:.4f}")
    return dict(name=name, fit=fit_time, pred=pred_time, mse=mse, mae=mae)


results = []
print(f"\n{'='*75}")
print(f"  Benchmark: N={N}, lags={LAGS}, predict_n={PREDICT_N}")
print(f"{'='*75}")

# ── CPU baselines ───────────────────────────────────────────────────────
print("\n── CPU baselines ──")
from PipelineTS.ml_model import LightGBMModel, XGBoostModel, RandomForestModel
try:
    from PipelineTS.ml_model import CatBoostModel
    results.append(bench_model('CatBoost (CPU)', CatBoostModel))
except Exception as e:
    print(f"  CatBoost skip: {e}")
results.append(bench_model('LightGBM (CPU)', LightGBMModel))
results.append(bench_model('XGBoost (CPU)', XGBoostModel))
results.append(bench_model('RandomForest (CPU)', RandomForestModel))

# ── GPU (torch) models ──────────────────────────────────────────────────
print("\n── GPU (torch) models ──")
from PipelineTS.ml_model import (
    TorchBoostingForestModel, TorchBaggingForestModel, DeepForestModel,
)
GPU_KW = dict(accelerator='cpu', random_state=42)  # use CPU for fair comparison

results.append(bench_model('TorchBoostingForest', TorchBoostingForestModel, GPU_KW))
results.append(bench_model('TorchBaggingForest', TorchBaggingForestModel,
                           {**GPU_KW, 'dropout': 0.1}))
results.append(bench_model('DeepForest', DeepForestModel,
                           {**GPU_KW, 'n_layers': 2}))

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  {'Model':25s}  {'Fit':>7s}  {'Pred':>6s}  {'MSE':>8s}  {'MAE':>8s}")
print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}")
for r in results:
    print(f"  {r['name']:25s}  {r['fit']:6.2f}s  {r['pred']:5.3f}s  "
          f"{r['mse']:8.4f}  {r['mae']:8.4f}")
print(f"{'='*75}\n")
