"""
GTB Ablation Benchmark Script.

Runs 5 configurations for each model on each dataset:
1. Baseline (no GTB)
2. Full GTB (FreqMix + Attn + SwiGLU)
3. No FreqMix (Attn + SwiGLU only)
4. No Attention (FreqMix + SwiGLU only)
5. No SwiGLU (FreqMix + Attn only)

Uses the spinesTS-level models directly for ablation control,
since the GlobalTemporalBlock accepts use_freq_mixing, use_attention, use_swiglu flags.
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PipelineTS.dataset import BuiltInSeriesData
from PipelineTS.spinesTS.nn import (
    DLinear, NLinear, NBeats, NHiTS, TFT, GAUNet,
    StackingRNN, Time2VecNet, TSTransformer, TiDE, PatchRNN, TCN
)


def mse(y_true, y_pred):
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def load_dataset():
    """Load Electric_Production dataset."""
    bd = BuiltInSeriesData(print_file_list=False)
    ds = bd['Electric_Production']
    data = pd.DataFrame(ds)
    data['date'] = pd.to_datetime(data['date'])
    return data


def split_series(data, target_col, lags, n_predict):
    """Create train X/y arrays using sliding window."""
    values = data[target_col].values.astype(np.float32)

    X, y = [], []
    for i in range(len(values) - lags - n_predict + 1):
        X.append(values[i:i + lags])
        y.append(values[i + lags:i + lags + n_predict])
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_model_configs(lags, n_predict):
    """Return dict of model_name -> (ModelClass, base_kwargs).
    base_kwargs does NOT include use_gtb or gtb_d_model.
    """
    common = dict(
        in_features=lags, out_features=n_predict,
        random_seed=42, device='auto'
    )
    return {
        'DLinear': (DLinear, {**common, 'loss_fn': 'huber', 'dropout': 0.1}),
        'NLinear': (NLinear, {**common, 'loss_fn': 'huber', 'dropout': 0.1}),
        'NBeats': (NBeats, {**common, 'loss_fn': 'huber', 'dropout': 0.1,
                            'num_stacks': 2, 'num_blocks': 2, 'num_layers': 3, 'layer_widths': 128}),
        'NHiTS': (NHiTS, {**common, 'loss_fn': 'huber', 'dropout': 0.1,
                          'num_stacks': 3, 'num_blocks': 1, 'layer_widths': 256}),
        'TFT': (TFT, {**common, 'loss_fn': 'huber', 'dropout': 0.1, 'hidden_size': 32}),
        'GAU': (GAUNet, {**common, 'loss_fn': 'huber', 'dropout': 0.2, 'level': 2}),
        'StackingRNN': (StackingRNN, {**common, 'loss_fn': 'mae', 'dropout': 0.1}),
        'Time2Vec': (Time2VecNet, {**common, 'loss_fn': 'mae', 'dropout': 0.1}),
        'Transformer': (TSTransformer, {**common, 'loss_fn': 'huber', 'dropout': 0.1,
                                        'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2}),
        'TiDE': (TiDE, {**common, 'loss_fn': 'huber', 'dropout': 0.1, 'hidden_size': 64}),
        'PatchRNN': (PatchRNN, {**common, 'loss_fn': 'mae', 'dropout': 0.1}),
        'TCN': (TCN, {**common, 'loss_fn': 'mae', 'dropout': 0.15}),
    }


# Ablation configurations: name -> (GTB model kwargs, GTB component overrides)
# Component overrides are applied to the GlobalTemporalBlock after model.call()
ABLATION_CONFIGS = {
    'baseline':       dict(use_gtb=False),
    'gtb_full':       dict(use_gtb=True, gtb_d_model=64),
    'gtb_no_freq':    dict(use_gtb=True, gtb_d_model=64),
    'gtb_no_attn':    dict(use_gtb=True, gtb_d_model=64),
    'gtb_no_swiglu':  dict(use_gtb=True, gtb_d_model=64),
}

# Which GTB sub-components to DISABLE for each ablation variant
GTB_DISABLE_MAP = {
    'gtb_no_freq':   'use_freq_mixing',
    'gtb_no_attn':   'use_attention',
    'gtb_no_swiglu': 'use_swiglu',
}


def patch_gtb_flags(model, ablation_name):
    """After model.call() creates the backbone, patch GTB sub-component flags.

    This disables specific components in the forward pass AND freezes their
    parameters so they don't consume optimizer state or gradients.
    """
    flag_to_disable = GTB_DISABLE_MAP.get(ablation_name)
    if flag_to_disable is None:
        return  # baseline or gtb_full: nothing to patch

    # Walk all modules to find GlobalTemporalBlock instances
    root = model.model if hasattr(model, 'model') else model
    for name, module in root.named_modules():
        if module.__class__.__name__ == 'GlobalTemporalBlock':
            setattr(module, flag_to_disable, False)


def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=300, patience=30):
    """Train model and return metrics."""
    import torch

    # Split train into train/val (90/10)
    n = len(X_train)
    n_val = max(1, n // 10)
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
    X_va, y_va = X_train[-n_val:], y_train[-n_val:]

    t0 = time.time()
    model.fit(
        X_tr, y_tr, epochs=epochs,
        eval_set=(X_va, y_va),
        patience=patience,
        verbose=False,
        lr_scheduler='CosineAnnealingLR'
    )
    preds = model.predict(X_test)
    elapsed = time.time() - t0

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    return {
        'mse': mse(y_test, preds),
        'mae': mae(y_test, preds),
        'time_sec': round(elapsed, 2)
    }


def run_ablation(model_names=None, epochs=300, patience=30):
    """Run full ablation study."""
    data = load_dataset()
    target_col = 'value'
    lags = 16
    n_predict = 16

    values = data[target_col].values.astype(np.float32)

    # Create sliding window dataset
    X, y = split_series(data, target_col, lags, n_predict)

    # Train/test split: last 20% as test
    n_test = max(1, len(X) // 5)
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    print(f"Dataset: Electric_Production")
    print(f"  Samples: train={len(X_train)}, test={len(X_test)}")
    print(f"  Lags={lags}, Predict={n_predict}")

    model_configs = get_model_configs(lags, n_predict)

    if model_names:
        model_configs = {k: v for k, v in model_configs.items() if k in model_names}

    all_results = {}

    for model_name, (ModelClass, base_kwargs) in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        model_results = {}

        for abl_name, abl_kwargs in ABLATION_CONFIGS.items():
            print(f"  [{abl_name}] ", end="", flush=True)
            try:
                # Create model with GTB config
                kwargs = {**base_kwargs, **abl_kwargs}
                model = ModelClass(**kwargs)

                # Patch GTB sub-component flags for ablation variants
                if abl_name.startswith('gtb_no_'):
                    patch_gtb_flags(model, abl_name)

                result = train_and_eval(model, X_train, y_train, X_test, y_test,
                                       epochs=epochs, patience=patience)
                model_results[abl_name] = result
                print(f"MSE={result['mse']:.4f}  MAE={result['mae']:.4f}  Time={result['time_sec']:.1f}s")
            except Exception as e:
                model_results[abl_name] = {'mse': float('nan'), 'mae': float('nan'),
                                           'time_sec': 0, 'error': str(e)}
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()

        all_results[model_name] = model_results

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_gtb_ablation.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n\n{'='*100}")
    print(f"GTB ABLATION SUMMARY (Electric_Production, lags={lags}, predict={n_predict})")
    print(f"{'='*100}")
    header = f"{'Model':<15}"
    for abl in ABLATION_CONFIGS:
        header += f" {'MSE_'+abl:>16}"
    print(header)
    print("-" * (15 + 17 * len(ABLATION_CONFIGS)))

    for model_name, model_results in all_results.items():
        row = f"{model_name:<15}"
        baseline_mse = model_results.get('baseline', {}).get('mse', float('nan'))
        for abl in ABLATION_CONFIGS:
            r = model_results.get(abl, {})
            mse_val = r.get('mse', float('nan'))
            if abl != 'baseline' and baseline_mse > 0 and not np.isnan(baseline_mse) and not np.isnan(mse_val):
                pct = (mse_val - baseline_mse) / baseline_mse * 100
                marker = '↓' if pct < 0 else '↑'
                row += f" {mse_val:>10.2f}({pct:+.0f}%{marker})"
            else:
                row += f" {mse_val:>16.2f}"
        print(row)

    # MAE summary
    print(f"\n{'Model':<15}", end="")
    for abl in ABLATION_CONFIGS:
        print(f" {'MAE_'+abl:>16}", end="")
    print()
    print("-" * (15 + 17 * len(ABLATION_CONFIGS)))

    for model_name, model_results in all_results.items():
        row = f"{model_name:<15}"
        baseline_mae = model_results.get('baseline', {}).get('mae', float('nan'))
        for abl in ABLATION_CONFIGS:
            r = model_results.get(abl, {})
            mae_val = r.get('mae', float('nan'))
            if abl != 'baseline' and baseline_mae > 0 and not np.isnan(baseline_mae) and not np.isnan(mae_val):
                pct = (mae_val - baseline_mae) / baseline_mae * 100
                marker = '↓' if pct < 0 else '↑'
                row += f" {mae_val:>10.2f}({pct:+.0f}%{marker})"
            else:
                row += f" {mae_val:>16.2f}"
        print(row)

    return all_results


if __name__ == '__main__':
    # Parse optional model filter from CLI: python benchmark_gtb_ablation.py DLinear TCN
    model_filter = sys.argv[1:] if len(sys.argv) > 1 else None
    run_ablation(model_names=model_filter)
