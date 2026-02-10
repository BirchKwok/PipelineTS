#!/usr/bin/env python
"""Benchmark: Static GTB vs Adaptive MoE Routing across all 12 NN models.

Compares three configurations:
1. baseline:  GTB disabled (use_gtb=False)
2. static:    GTB enabled with static routing (all 3 experts always active)
3. adaptive:  GTB enabled with MoE adaptive routing (top-2 of 3 experts per sample)

Uses Electric_Production dataset with lags=16, predict=16.
"""
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
from PipelineTS.spinesTS.data import BuiltInSeriesData

# ──────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────
bd = BuiltInSeriesData(print_file_list=False)
ds = bd['Electric_Production']
df = pd.DataFrame(ds)
values = df['value'].values.astype(np.float32)

LAGS = 16
N_PREDICT = 16

X_list, y_list = [], []
for i in range(len(values) - LAGS - N_PREDICT + 1):
    X_list.append(values[i:i + LAGS])
    y_list.append(values[i + LAGS:i + LAGS + N_PREDICT])
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Dataset: Electric_Production | train={len(X_train)}, test={len(X_test)}")
print(f"Lags={LAGS}, Predict={N_PREDICT}\n")

# ──────────────────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────────────────
from PipelineTS.spinesTS.nn import (
    TCN, StackingRNN, GAUNet, TSTransformer, DLinear, NLinear,
    Time2VecNet, TiDE, NHiTS, NBeats, PatchRNN, TFT
)

MODELS = {
    'DLinear':      (DLinear, {}),
    'NLinear':      (NLinear, {}),
    'NBeats':       (NBeats, dict(num_stacks=2, num_blocks=2, num_layers=3, layer_widths=128)),
    'NHiTS':        (NHiTS, dict(num_stacks=3, num_blocks=1, layer_widths=256)),
    'TFT':          (TFT, dict(hidden_size=32)),
    'GAUNet':       (GAUNet, dict(level=2)),
    'StackingRNN':  (StackingRNN, {}),
    'Time2VecNet':  (Time2VecNet, {}),
    'Transformer':  (TSTransformer, dict(d_model=64, num_encoder_layers=2)),
    'TiDE':         (TiDE, dict(hidden_size=128)),
    'PatchRNN':     (PatchRNN, {}),
    'TCN':          (TCN, {}),
}

CONFIGS = {
    'baseline': dict(use_gtb=False),
    'static':   dict(use_gtb=True, gtb_d_model=64, routing_mode='static'),
    'adaptive': dict(use_gtb=True, gtb_d_model=64, routing_mode='adaptive'),
}

EPOCHS = 200
SEED = 42

# ──────────────────────────────────────────────────────────
# Run benchmark
# ──────────────────────────────────────────────────────────
results = {}

for model_name, (ModelClass, model_kwargs) in MODELS.items():
    results[model_name] = {}
    for config_name, config_kwargs in CONFIGS.items():
        merged = {**model_kwargs, **config_kwargs}
        try:
            t0 = time.time()
            model = ModelClass(
                in_features=LAGS, out_features=N_PREDICT,
                random_seed=SEED, **merged
            )
            model.fit(X_train, y_train, epochs=EPOCHS, verbose=False)
            preds = model.predict(X_test)
            elapsed = time.time() - t0

            mse = float(np.mean((preds - y_test) ** 2))
            mae = float(np.mean(np.abs(preds - y_test)))

            # Get routing stats for adaptive mode
            routing_info = ''
            if config_name == 'adaptive':
                root = model.model if hasattr(model, 'model') else model
                for name, module in root.named_modules():
                    if module.__class__.__name__ == 'GlobalTemporalBlock':
                        stats = module.get_routing_stats()
                        if stats:
                            freqs = stats['expert_freq']
                            names = stats.get('expert_names', ['E0', 'E1', 'E2'])
                            routing_info = ' | '.join(
                                f"{n}={f:.1%}" for n, f in zip(names, freqs)
                            )
                        break

            results[model_name][config_name] = {
                'mse': mse, 'mae': mae, 'time': elapsed, 'routing': routing_info
            }
            route_str = f" [{routing_info}]" if routing_info else ""
            print(f"  {model_name:15s} {config_name:10s}  MSE={mse:8.1f}  MAE={mae:6.2f}  {elapsed:5.1f}s{route_str}")

        except Exception as e:
            print(f"  {model_name:15s} {config_name:10s}  FAILED: {e}")
            results[model_name][config_name] = {'mse': float('inf'), 'mae': float('inf'), 'time': 0, 'routing': ''}

# ──────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'Model':15s} | {'Baseline MSE':>12s} | {'Static MSE':>12s} | {'Adaptive MSE':>12s} | {'Best':>10s} | {'Δ vs Base':>10s}")
print("-" * 100)

for model_name in MODELS:
    base_mse = results[model_name].get('baseline', {}).get('mse', float('inf'))
    static_mse = results[model_name].get('static', {}).get('mse', float('inf'))
    adaptive_mse = results[model_name].get('adaptive', {}).get('mse', float('inf'))

    best_name = 'baseline'
    best_mse = base_mse
    if static_mse < best_mse:
        best_name, best_mse = 'static', static_mse
    if adaptive_mse < best_mse:
        best_name, best_mse = 'adaptive', adaptive_mse

    delta = ((best_mse - base_mse) / base_mse * 100) if base_mse > 0 else 0

    print(f"{model_name:15s} | {base_mse:12.1f} | {static_mse:12.1f} | {adaptive_mse:12.1f} | {best_name:>10s} | {delta:+9.1f}%")

print("=" * 100)

# Show routing distributions for adaptive mode
print("\nAdaptive Routing Distributions:")
for model_name in MODELS:
    r = results[model_name].get('adaptive', {}).get('routing', '')
    if r:
        print(f"  {model_name:15s}: {r}")

print("\nDone.")
