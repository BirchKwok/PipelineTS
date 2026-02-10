"""
Benchmark: PipelineTS DeepAR vs Darts DeepAR
Tests on Electric_Production and AirPassengers datasets.
Compares MSE, MAE, training+prediction time.
Also includes a few other PipelineTS NN models for context.
"""

import sys
import os
import time
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


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def load_datasets():
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


# ─── Darts DeepAR ───────────────────────────────────────────────────────────

def run_darts_deepar(train_data, test_data, time_col, target_col, lags, n_predict):
    """Run Darts DeepAR (RNNModel + GaussianLikelihood) and return predictions + timing.

    In Darts >=0.30, DeepAR is implemented as RNNModel with a probabilistic likelihood.
    """
    from darts import TimeSeries
    from darts.models import RNNModel
    from darts.utils.likelihood_models.torch import GaussianLikelihood

    ts_train = TimeSeries.from_dataframe(train_data, time_col=time_col, value_cols=target_col)

    t0 = time.time()
    model = RNNModel(
        input_chunk_length=lags,
        model='LSTM',
        hidden_dim=64,
        n_rnn_layers=2,
        dropout=0.1,
        training_length=lags + n_predict,
        n_epochs=300,
        random_state=42,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs={
            "enable_progress_bar": False,
            "accelerator": "cpu",
        }
    )
    model.fit(ts_train)
    pred = model.predict(n_predict, num_samples=100)
    elapsed = time.time() - t0

    # Use median of samples as point prediction
    y_pred = pred.quantile(0.5).values().flatten()[:n_predict]
    y_true = test_data[target_col].values[:n_predict]

    return y_true, y_pred, elapsed


# ─── PipelineTS DeepAR ──────────────────────────────────────────────────────

def run_pipelinets_deepar(train_data, test_data, time_col, target_col, lags, n_predict):
    """Run PipelineTS DeepAR and return predictions + timing."""
    from PipelineTS.nn_model import DeepARModel

    t0 = time.time()
    model = DeepARModel(
        time_col=time_col, target_col=target_col, lags=lags,
        d_model=64, n_blocks=3, n_rwkv_blocks=3, dropout=0.1,
        quantile=None, epochs=300, patience=30, verbose=False,
        random_state=42
    )
    model.fit(train_data)
    preds = model.predict(n_predict, data=train_data)
    elapsed = time.time() - t0

    y_pred = preds[target_col].values[:n_predict]
    y_true = test_data[target_col].values[:n_predict]

    return y_true, y_pred, elapsed


# ─── Other PipelineTS models for context ─────────────────────────────────────

def run_pipelinets_model(model_cls, model_kwargs, train_data, test_data, time_col, target_col, n_predict):
    t0 = time.time()
    model = model_cls(**model_kwargs)
    model.fit(train_data)
    preds = model.predict(n_predict, data=train_data)
    elapsed = time.time() - t0

    y_pred = preds[target_col].values[:n_predict]
    y_true = test_data[target_col].values[:n_predict]
    return y_true, y_pred, elapsed


def evaluate(y_true, y_pred):
    return {
        'mse': mse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'smape': smape(y_true, y_pred),
    }


def run_benchmark():
    datasets = load_datasets()
    lags = 16
    n_predict = 16

    # Context models from PipelineTS
    from PipelineTS.nn_model import (
        NHitsModel, GAUModel, TCNModel, TiDEModel, TransformerModel
    )

    context_models = {
        'PTS_NHiTS': (NHitsModel, {'num_stacks': 3, 'num_blocks': 1, 'layer_widths': 256}),
        'PTS_GAU': (GAUModel, {'level': 2}),
        'PTS_TCN': (TCNModel, {}),
        'PTS_TiDE': (TiDEModel, {'hidden_size': 64}),
        'PTS_Transformer': (TransformerModel, {'d_model': 32, 'nhead': 4, 'num_encoder_layers': 2}),
    }

    all_results = {}

    for ds_name, ds_info in datasets.items():
        data = ds_info['data']
        time_col = ds_info['time_col']
        target_col = ds_info['target_col']

        train_data = data.iloc[:-n_predict].reset_index(drop=True)
        test_data = data.iloc[-n_predict:].reset_index(drop=True)

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} (train={len(train_data)}, test={len(test_data)})")
        print(f"{'='*70}")
        print(f"{'Model':<22} {'MSE':>10} {'MAE':>10} {'sMAPE%':>8} {'Time(s)':>9}")
        print("-" * 62)

        ds_results = {}

        # 1. Darts DeepAR
        print(f"  {'Darts_DeepAR':<20}", end="", flush=True)
        try:
            y_true, y_pred, elapsed = run_darts_deepar(
                train_data, test_data, time_col, target_col, lags, n_predict
            )
            metrics = evaluate(y_true, y_pred)
            metrics['time_sec'] = round(elapsed, 2)
            ds_results['Darts_DeepAR'] = metrics
            print(f"{metrics['mse']:>10.2f} {metrics['mae']:>10.2f} {metrics['smape']:>8.2f} {elapsed:>8.1f}s")
        except Exception as e:
            ds_results['Darts_DeepAR'] = {'mse': float('nan'), 'mae': float('nan'), 'smape': float('nan'), 'time_sec': 0, 'error': str(e)}
            print(f" ERROR: {e}")

        # 2. PipelineTS DeepAR
        print(f"  {'PTS_DeepAR':<20}", end="", flush=True)
        try:
            y_true, y_pred, elapsed = run_pipelinets_deepar(
                train_data, test_data, time_col, target_col, lags, n_predict
            )
            metrics = evaluate(y_true, y_pred)
            metrics['time_sec'] = round(elapsed, 2)
            ds_results['PTS_DeepAR'] = metrics
            print(f"{metrics['mse']:>10.2f} {metrics['mae']:>10.2f} {metrics['smape']:>8.2f} {elapsed:>8.1f}s")
        except Exception as e:
            ds_results['PTS_DeepAR'] = {'mse': float('nan'), 'mae': float('nan'), 'smape': float('nan'), 'time_sec': 0, 'error': str(e)}
            print(f" ERROR: {e}")

        # 3. Context models
        for model_name, (model_cls, extra_kwargs) in context_models.items():
            print(f"  {model_name:<20}", end="", flush=True)
            try:
                kwargs = {
                    'time_col': time_col, 'target_col': target_col, 'lags': lags,
                    'quantile': None, 'epochs': 300, 'patience': 30, 'verbose': False,
                    **extra_kwargs
                }
                y_true, y_pred, elapsed = run_pipelinets_model(
                    model_cls, kwargs, train_data, test_data, time_col, target_col, n_predict
                )
                metrics = evaluate(y_true, y_pred)
                metrics['time_sec'] = round(elapsed, 2)
                ds_results[model_name] = metrics
                print(f"{metrics['mse']:>10.2f} {metrics['mae']:>10.2f} {metrics['smape']:>8.2f} {elapsed:>8.1f}s")
            except Exception as e:
                ds_results[model_name] = {'mse': float('nan'), 'mae': float('nan'), 'smape': float('nan'), 'time_sec': 0, 'error': str(e)}
                print(f" ERROR: {e}")

        all_results[ds_name] = ds_results

    # Final comparison
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON — sorted by MSE")
    print(f"{'='*70}")
    for ds_name, ds_results in all_results.items():
        print(f"\nDataset: {ds_name}")
        print(f"{'Rank':<6} {'Model':<22} {'MSE':>10} {'MAE':>10} {'sMAPE%':>8} {'Time(s)':>9}")
        print("-" * 68)
        sorted_models = sorted(ds_results.items(), key=lambda x: x[1].get('mse', float('inf')))
        for rank, (model_name, r) in enumerate(sorted_models, 1):
            mse_val = r.get('mse', float('nan'))
            mae_val = r.get('mae', float('nan'))
            smape_val = r.get('smape', float('nan'))
            time_val = r.get('time_sec', 0)
            marker = ' ★' if 'DeepAR' in model_name else ''
            print(f"#{rank:<5} {model_name:<22} {mse_val:>10.2f} {mae_val:>10.2f} {smape_val:>8.2f} {time_val:>8.1f}s{marker}")

    # Head-to-head comparison
    print(f"\n\n{'='*70}")
    print("HEAD-TO-HEAD: PipelineTS DeepAR vs Darts DeepAR")
    print(f"{'='*70}")
    for ds_name, ds_results in all_results.items():
        pts = ds_results.get('PTS_DeepAR', {})
        darts = ds_results.get('Darts_DeepAR', {})
        if not pts or not darts:
            continue
        print(f"\n  {ds_name}:")
        for metric_name in ['mse', 'mae', 'smape']:
            p_val = pts.get(metric_name, float('nan'))
            d_val = darts.get(metric_name, float('nan'))
            if d_val > 0 and not np.isnan(d_val) and not np.isnan(p_val):
                pct = (p_val - d_val) / d_val * 100
                winner = "PTS" if p_val < d_val else "Darts"
                print(f"    {metric_name.upper():<8} PTS={p_val:>10.2f}  Darts={d_val:>10.2f}  Δ={pct:>+7.1f}%  Winner: {winner}")
            else:
                print(f"    {metric_name.upper():<8} PTS={p_val:>10.2f}  Darts={d_val:>10.2f}")
        p_time = pts.get('time_sec', 0)
        d_time = darts.get('time_sec', 0)
        speedup = d_time / p_time if p_time > 0 else float('inf')
        print(f"    {'TIME':<8} PTS={p_time:>9.1f}s  Darts={d_time:>9.1f}s  Speedup: {speedup:.1f}x")


if __name__ == '__main__':
    run_benchmark()
