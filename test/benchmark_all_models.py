"""
Comprehensive benchmark for ALL PipelineTS models (NN + ML + Statistical).
Tests on built-in Electric_Production dataset.
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


def load_dataset():
    """Load Electric_Production dataset."""
    bd = BuiltInSeriesData(print_file_list=False)
    ds = bd['Electric_Production']
    data = pd.DataFrame(ds)
    data['date'] = pd.to_datetime(data['date'])
    return data, 'date', 'value'


def get_all_models(time_col, target_col, lags=16):
    """Instantiate ALL models to benchmark."""
    from PipelineTS.nn_model import (
        DLinearModel, NLinearModel, NBeatsModel, NHitsModel,
        TFTModel, GAUModel, StackingRNNModel, Time2VecModel,
        TransformerModel, TiDEModel, PatchRNNModel, TCNModel,
        ITransformerModel, SRSNetModel
    )
    from PipelineTS.ml_model import (
        LightGBMModel, XGBoostModel, CatBoostModel, RandomForestModel,
        WideGBRTModel, MultiOutputRegressorModel, MultiStepRegressorModel,
        RegressorChainModel
    )
    from PipelineTS.statistic_model import ProphetModel, AutoARIMAModel

    nn_common = dict(
        time_col=time_col, target_col=target_col, lags=lags,
        quantile=None, epochs=300, patience=30, verbose=False
    )

    models = {
        # ---- NN models ----
        'DLinear': DLinearModel(**nn_common),
        'NLinear': NLinearModel(**nn_common),
        'NBeats': NBeatsModel(**nn_common, num_stacks=2, num_blocks=2, num_layers=3, layer_widths=128),
        'NHiTS': NHitsModel(**nn_common, num_stacks=3, num_blocks=1, layer_widths=256),
        'TFT': TFTModel(**nn_common, hidden_size=32),
        'GAU': GAUModel(**nn_common, level=2),
        'StackingRNN': StackingRNNModel(**nn_common),
        'Time2Vec': Time2VecModel(**nn_common),
        'Transformer': TransformerModel(**nn_common, d_model=32, nhead=4, num_encoder_layers=2),
        'TiDE': TiDEModel(**nn_common, hidden_size=64),
        'PatchRNN': PatchRNNModel(**nn_common),
        'TCN': TCNModel(**nn_common),
        'ITransformer': ITransformerModel(**nn_common, d_model=64, n_heads=4, d_ff=128, e_layers=1),
        'SRSNet': SRSNetModel(**nn_common),
        # ---- ML models ----
        'LightGBM': LightGBMModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'XGBoost': XGBoostModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=0),
        'CatBoost': CatBoostModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=False),
        'RandomForest': RandomForestModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None),
        'WideGBRT': WideGBRTModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'MultiOutput': MultiOutputRegressorModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'MultiStep': MultiStepRegressorModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'RegressorChain': RegressorChainModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None),
        # ---- Statistical models ----
        'Prophet': ProphetModel(time_col=time_col, target_col=target_col, quantile=None),
        'AutoARIMA': AutoARIMAModel(time_col=time_col, target_col=target_col, quantile=None),
    }
    return models


def benchmark_single(model, train_data, test_data, target_col, n_predict):
    """Train and evaluate a single model. Returns MSE, MAE, elapsed time."""
    t0 = time.time()

    # Fit
    from spinesUtils.asserts import check_has_param
    fit_kwargs = {}
    if check_has_param(model.fit, 'data'):
        model.fit(data=train_data, **fit_kwargs)
    else:
        model.fit(train_data, **fit_kwargs)

    t_fit = time.time() - t0

    # Predict
    t1 = time.time()
    if check_has_param(model.predict, 'data'):
        preds = model.predict(n_predict, data=train_data)
    else:
        preds = model.predict(n_predict)
    t_pred = time.time() - t1

    y_true = test_data[target_col].values[:n_predict]
    y_pred = preds[target_col].values[:n_predict]

    return {
        'mse': round(mse(y_true, y_pred), 4),
        'mae': round(mae(y_true, y_pred), 4),
        'fit_sec': round(t_fit, 2),
        'pred_sec': round(t_pred, 2),
        'total_sec': round(t_fit + t_pred, 2)
    }


def run_benchmark(tag="baseline"):
    data, time_col, target_col = load_dataset()
    lags = 16
    n_predict = 16

    # Train/test split
    train_data = data.iloc[:-n_predict].reset_index(drop=True)
    test_data = data.iloc[-n_predict:].reset_index(drop=True)

    print(f"Dataset: Electric_Production (train={len(train_data)}, test={len(test_data)})")
    print(f"{'='*80}")

    models = get_all_models(time_col, target_col, lags=lags)
    results = {}
    total_start = time.time()

    for model_name, model in models.items():
        print(f"  {model_name:<20s}...", end=" ", flush=True)
        try:
            result = benchmark_single(model, train_data, test_data, target_col, n_predict)
            results[model_name] = result
            print(f"MSE={result['mse']:>10.4f}  MAE={result['mae']:>8.4f}  "
                  f"Fit={result['fit_sec']:>6.1f}s  Pred={result['pred_sec']:>5.1f}s  "
                  f"Total={result['total_sec']:>6.1f}s")
        except Exception as e:
            results[model_name] = {'mse': float('nan'), 'mae': float('nan'),
                                   'fit_sec': 0, 'pred_sec': 0, 'total_sec': 0, 'error': str(e)}
            print(f"ERROR: {e}")

    total_elapsed = time.time() - total_start

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), f'benchmark_all_{tag}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ({tag}) — sorted by Total Time")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'MSE':>10} {'MAE':>10} {'Fit(s)':>8} {'Pred(s)':>8} {'Total(s)':>8}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1].get('total_sec', float('inf'))):
        print(f"{name:<20} {r.get('mse', float('nan')):>10.4f} {r.get('mae', float('nan')):>10.4f} "
              f"{r.get('fit_sec', 0):>8.1f} {r.get('pred_sec', 0):>8.1f} {r.get('total_sec', 0):>8.1f}")
    print(f"\nTotal wall time: {total_elapsed:.1f}s")

    # Comparison with baseline if available
    baseline_path = os.path.join(os.path.dirname(__file__), 'benchmark_all_baseline.json')
    if tag != 'baseline' and os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        print(f"\n{'='*90}")
        print(f"COMPARISON: baseline → {tag}")
        print(f"{'='*90}")
        print(f"{'Model':<20} {'MSE_old':>9} {'MSE_new':>9} {'MSE_Δ%':>8}  "
              f"{'Time_old':>8} {'Time_new':>8} {'Speed_Δ%':>9}")
        print("-" * 85)
        for name in sorted(results.keys()):
            r_new = results.get(name, {})
            r_old = baseline.get(name, {})
            mse_old = r_old.get('mse', float('nan'))
            mse_new = r_new.get('mse', float('nan'))
            t_old = r_old.get('total_sec', 0)
            t_new = r_new.get('total_sec', 0)
            if mse_old and mse_old > 0 and not np.isnan(mse_old):
                mse_pct = (mse_new - mse_old) / mse_old * 100
            else:
                mse_pct = float('nan')
            if t_old and t_old > 0:
                speed_pct = (t_old - t_new) / t_old * 100
            else:
                speed_pct = float('nan')
            acc_mark = '✅' if mse_pct < -1 else ('⚠️' if mse_pct > 5 else '  ')
            spd_mark = '🚀' if speed_pct > 10 else ('🐢' if speed_pct < -10 else '  ')
            print(f"{name:<20} {mse_old:>9.2f} {mse_new:>9.2f} {mse_pct:>+7.1f}% {acc_mark} "
                  f"{t_old:>7.1f}s {t_new:>7.1f}s {speed_pct:>+7.1f}% {spd_mark}")

    return results


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_benchmark(tag)
