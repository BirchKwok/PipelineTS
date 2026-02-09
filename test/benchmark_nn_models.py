"""
Benchmark script for NN models (excluding SRSNet and ITransformer).
Tests on 2 built-in datasets: Electric_Production and AirPassengers.
Measures MSE, MAE, and training+prediction time.
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


def mse(y_true, y_pred):
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def load_datasets():
    """Load Electric_Production and AirPassengers datasets."""
    bd = BuiltInSeriesData(print_file_list=False)

    ds1 = bd['Electric_Production']
    ds1_data = pd.DataFrame(ds1)
    ds1_data['date'] = pd.to_datetime(ds1_data['date'])

    ds2 = bd['AirPassengers']
    ds2_data = pd.DataFrame(ds2)
    ds2_data['Month'] = pd.to_datetime(ds2_data['Month'])

    return {
        'Electric_Production': {
            'data': ds1_data, 'time_col': 'date', 'target_col': 'value'
        },
        'AirPassengers': {
            'data': ds2_data, 'time_col': 'Month', 'target_col': 'Passengers'
        }
    }


def get_models(time_col, target_col, lags=16):
    """Instantiate all NN models to benchmark (excluding SRSNet and ITransformer)."""
    from PipelineTS.nn_model import (
        DLinearModel, NLinearModel, NBeatsModel, NHitsModel,
        TFTModel, GAUModel, StackingRNNModel, Time2VecModel,
        TransformerModel, TiDEModel, PatchRNNModel, TCNModel
    )

    models = {
        'DLinear': DLinearModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False
        ),
        'NLinear': NLinearModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False
        ),
        'NBeats': NBeatsModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False,
            num_stacks=2, num_blocks=2, num_layers=3, layer_widths=128
        ),
        'NHiTS': NHitsModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False,
            num_stacks=3, num_blocks=1, layer_widths=256
        ),
        'TFT': TFTModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False,
            hidden_size=32
        ),
        'GAU': GAUModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False,
            level=2
        ),
        'StackingRNN': StackingRNNModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False
        ),
        'Time2Vec': Time2VecModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False
        ),
        'Transformer': TransformerModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False,
            d_model=32, nhead=4, num_encoder_layers=2
        ),
        'TiDE': TiDEModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False,
            hidden_size=64
        ),
        'PatchRNN': PatchRNNModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False
        ),
        'TCN': TCNModel(
            time_col=time_col, target_col=target_col, lags=lags,
            quantile=None, epochs=300, patience=30, verbose=False
        ),
    }
    return models


def benchmark_single(model, train_data, test_data, target_col, n_predict):
    """Train and evaluate a single model. Returns MSE, MAE, elapsed time."""
    t0 = time.time()
    model.fit(train_data)
    preds = model.predict(n_predict, data=train_data)
    elapsed = time.time() - t0

    y_true = test_data[target_col].values[:n_predict]
    y_pred = preds[target_col].values[:n_predict]

    return {
        'mse': mse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'time_sec': round(elapsed, 2)
    }


def run_benchmark(tag="baseline"):
    datasets = load_datasets()
    lags = 16
    n_predict = 16
    all_results = {}

    for ds_name, ds_info in datasets.items():
        data = ds_info['data']
        time_col = ds_info['time_col']
        target_col = ds_info['target_col']

        # Train/test split: last n_predict points as test
        train_data = data.iloc[:-n_predict].reset_index(drop=True)
        test_data = data.iloc[-n_predict:].reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (train={len(train_data)}, test={len(test_data)})")
        print(f"{'='*60}")

        models = get_models(time_col, target_col, lags=lags)
        ds_results = {}

        for model_name, model in models.items():
            print(f"  Testing {model_name}...", end=" ", flush=True)
            try:
                result = benchmark_single(model, train_data, test_data, target_col, n_predict)
                ds_results[model_name] = result
                print(f"MSE={result['mse']:.4f}  MAE={result['mae']:.4f}  Time={result['time_sec']:.1f}s")
            except Exception as e:
                ds_results[model_name] = {'mse': float('nan'), 'mae': float('nan'), 'time_sec': 0, 'error': str(e)}
                print(f"ERROR: {e}")

        all_results[ds_name] = ds_results

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), f'benchmark_{tag}.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table sorted by MSE
    print(f"\n\n{'='*90}")
    print(f"SUMMARY ({tag}) — sorted by MSE")
    print(f"{'='*90}")
    for ds_name, ds_results in all_results.items():
        print(f"\nDataset: {ds_name}")
        print(f"{'Model':<15} {'MSE':>12} {'MAE':>12} {'Time(s)':>10}")
        print("-" * 50)
        for model_name, r in sorted(ds_results.items(), key=lambda x: x[1].get('mse', float('inf'))):
            mse_val = r.get('mse', float('nan'))
            mae_val = r.get('mae', float('nan'))
            time_val = r.get('time_sec', 0)
            print(f"{model_name:<15} {mse_val:>12.4f} {mae_val:>12.4f} {time_val:>10.1f}")

    # Print summary table sorted by MAE
    print(f"\n{'='*90}")
    print(f"SUMMARY ({tag}) — sorted by MAE")
    print(f"{'='*90}")
    for ds_name, ds_results in all_results.items():
        print(f"\nDataset: {ds_name}")
        print(f"{'Model':<15} {'MAE':>12} {'MSE':>12} {'Time(s)':>10}")
        print("-" * 50)
        for model_name, r in sorted(ds_results.items(), key=lambda x: x[1].get('mae', float('inf'))):
            mae_val = r.get('mae', float('nan'))
            mse_val = r.get('mse', float('nan'))
            time_val = r.get('time_sec', 0)
            print(f"{model_name:<15} {mae_val:>12.4f} {mse_val:>12.4f} {time_val:>10.1f}")

    # If baseline exists and this is not baseline, print comparison
    baseline_path = os.path.join(os.path.dirname(__file__), 'benchmark_baseline.json')
    if tag != 'baseline' and os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        print(f"\n{'='*90}")
        print(f"COMPARISON: baseline → {tag}")
        print(f"{'='*90}")
        for ds_name in all_results:
            if ds_name not in baseline_results:
                continue
            print(f"\nDataset: {ds_name}")
            print(f"{'Model':<15} {'MSE_base':>10} {'MSE_new':>10} {'MSE_Δ%':>8}  {'MAE_base':>10} {'MAE_new':>10} {'MAE_Δ%':>8}  {'T_base':>7} {'T_new':>7}")
            print("-" * 105)
            for model_name in sorted(all_results[ds_name].keys()):
                r_new = all_results[ds_name].get(model_name, {})
                r_old = baseline_results.get(ds_name, {}).get(model_name, {})
                mse_old = r_old.get('mse', float('nan'))
                mse_new = r_new.get('mse', float('nan'))
                mae_old = r_old.get('mae', float('nan'))
                mae_new = r_new.get('mae', float('nan'))
                t_old = r_old.get('time_sec', 0)
                t_new = r_new.get('time_sec', 0)
                if mse_old and mse_old > 0 and not np.isnan(mse_old):
                    mse_pct = (mse_new - mse_old) / mse_old * 100
                else:
                    mse_pct = float('nan')
                if mae_old and mae_old > 0 and not np.isnan(mae_old):
                    mae_pct = (mae_new - mae_old) / mae_old * 100
                else:
                    mae_pct = float('nan')
                mse_marker = '✅' if mse_pct < -5 else ('⚠️' if mse_pct > 5 else '')
                mae_marker = '✅' if mae_pct < -5 else ('⚠️' if mae_pct > 5 else '')
                print(f"{model_name:<15} {mse_old:>10.2f} {mse_new:>10.2f} {mse_pct:>+7.1f}% {mse_marker} {mae_old:>10.2f} {mae_new:>10.2f} {mae_pct:>+7.1f}% {mae_marker} {t_old:>6.1f}s {t_new:>6.1f}s")

    return all_results


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_benchmark(tag)
