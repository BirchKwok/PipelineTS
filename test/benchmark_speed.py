"""
Speed benchmark for ALL PipelineTS models.
Measures fit + predict wall time on Electric_Production dataset.
Used to determine which models qualify as 'light'.
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


def load_dataset():
    bd = BuiltInSeriesData(print_file_list=False)
    ds = bd['Electric_Production']
    data = pd.DataFrame(ds)
    data['date'] = pd.to_datetime(data['date'])
    return data, 'date', 'value'


def get_all_models(time_col, target_col, lags=16):
    from PipelineTS.nn_model import (
        DLinearModel, NLinearModel, NBeatsModel, NHitsModel,
        TFTModel, GAUModel, StackingRNNModel, Time2VecModel,
        TransformerModel, TiDEModel, PatchRNNModel, TCNModel,
        ITransformerModel, SRSNetModel, DeepARModel
    )
    from PipelineTS.ml_model import (
        LightGBMModel, XGBoostModel, CatBoostModel, RandomForestModel,
        WideGBRTModel, MultiOutputRegressorModel, MultiStepRegressorModel,
        RegressorChainModel
    )
    from PipelineTS.statistic_model import ProphetModel, AutoARIMAModel

    nn_common = dict(
        time_col=time_col, target_col=target_col, lags=lags,
        quantile=None, epochs=100, patience=20, verbose=False
    )

    models = {
        # ---- NN models (15) ----
        'd_linear': DLinearModel(**nn_common),
        'n_linear': NLinearModel(**nn_common),
        'n_beats': NBeatsModel(**nn_common, num_stacks=2, num_blocks=2, num_layers=3, layer_widths=128),
        'n_hits': NHitsModel(**nn_common, num_stacks=3, num_blocks=1, layer_widths=256),
        'tft': TFTModel(**nn_common, hidden_size=32),
        'gau': GAUModel(**nn_common, level=2),
        'stacking_rnn': StackingRNNModel(**nn_common),
        'time2vec': Time2VecModel(**nn_common),
        'transformer': TransformerModel(**nn_common, d_model=32, nhead=4, num_encoder_layers=2),
        'tide': TiDEModel(**nn_common, hidden_size=64),
        'patch_rnn': PatchRNNModel(**nn_common),
        'tcn': TCNModel(**nn_common),
        'itransformer': ITransformerModel(**nn_common, d_model=64, n_heads=4, d_ff=128, e_layers=1),
        'srs_net': SRSNetModel(**nn_common),
        'deepar': DeepARModel(**nn_common),
        # ---- ML models (8) ----
        'lightgbm': LightGBMModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'xgboost': XGBoostModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=0),
        'catboost': CatBoostModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=False),
        'random_forest': RandomForestModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None),
        'wide_gbrt': WideGBRTModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'multi_output_model': MultiOutputRegressorModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'multi_step_model': MultiStepRegressorModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None, verbose=-1),
        'regressor_chain': RegressorChainModel(time_col=time_col, target_col=target_col, lags=lags, quantile=None),
        # ---- Statistical models (2) ----
        'prophet': ProphetModel(time_col=time_col, target_col=target_col, quantile=None),
        'auto_arima': AutoARIMAModel(time_col=time_col, target_col=target_col, quantile=None),
    }
    return models


def benchmark_model(model, train_data, test_data, target_col, n_predict):
    from spinesUtils.asserts import check_has_param

    t0 = time.time()
    if check_has_param(model.fit, 'data'):
        model.fit(data=train_data)
    else:
        model.fit(train_data)
    t_fit = time.time() - t0

    t1 = time.time()
    if check_has_param(model.predict, 'data'):
        preds = model.predict(n_predict, data=train_data)
    else:
        preds = model.predict(n_predict)
    t_pred = time.time() - t1

    y_true = test_data[target_col].values[:n_predict]
    y_pred = preds[target_col].values[:n_predict]
    mse_val = float(np.mean((y_true - y_pred) ** 2))
    mae_val = float(np.mean(np.abs(y_true - y_pred)))

    return {
        'mse': round(mse_val, 2),
        'mae': round(mae_val, 2),
        'fit_sec': round(t_fit, 2),
        'pred_sec': round(t_pred, 2),
        'total_sec': round(t_fit + t_pred, 2)
    }


def main():
    data, time_col, target_col = load_dataset()
    lags = 16
    n_predict = 16

    train_data = data.iloc[:-n_predict].reset_index(drop=True)
    test_data = data.iloc[-n_predict:].reset_index(drop=True)

    print(f"Dataset: Electric_Production (train={len(train_data)}, test={len(test_data)})")
    print(f"lags={lags}, n_predict={n_predict}, NN epochs=100, patience=20")
    print(f"{'='*85}")

    models = get_all_models(time_col, target_col, lags=lags)
    results = {}

    for name, model in models.items():
        print(f"  {name:<22s}...", end=" ", flush=True)
        try:
            r = benchmark_model(model, train_data, test_data, target_col, n_predict)
            results[name] = r
            print(f"MSE={r['mse']:>9.2f}  MAE={r['mae']:>7.2f}  "
                  f"Fit={r['fit_sec']:>6.1f}s  Pred={r['pred_sec']:>5.2f}s  "
                  f"Total={r['total_sec']:>6.1f}s")
        except Exception as e:
            results[name] = {'mse': float('nan'), 'mae': float('nan'),
                             'fit_sec': 999, 'pred_sec': 999, 'total_sec': 999, 'error': str(e)}
            print(f"ERROR: {e}")

    # Sort by total time
    print(f"\n{'='*85}")
    print(f"SPEED RANKING (sorted by total time)")
    print(f"{'='*85}")
    print(f"{'Rank':<5} {'Model':<22} {'Total(s)':>8} {'Fit(s)':>8} {'Pred(s)':>8} {'MSE':>10} {'Category':<12}")
    print("-" * 85)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_sec'])
    for rank, (name, r) in enumerate(sorted_results, 1):
        # Determine category
        if name in ('lightgbm', 'xgboost', 'catboost', 'random_forest', 'wide_gbrt',
                     'multi_output_model', 'multi_step_model', 'regressor_chain'):
            cat = 'ML'
        elif name in ('prophet', 'auto_arima'):
            cat = 'Statistical'
        else:
            cat = 'NN'

        err = r.get('error', '')
        mse_str = f"{r['mse']:>10.2f}" if not err else "ERROR"
        print(f"{rank:<5} {name:<22} {r['total_sec']:>8.1f} {r['fit_sec']:>8.1f} {r['pred_sec']:>8.2f} "
              f"{mse_str} {cat:<12}")

    # Suggest light models (total_sec < threshold)
    print(f"\n{'='*85}")
    print("LIGHT MODEL SUGGESTION")
    print(f"{'='*85}")

    # Use 15s as threshold for "light"
    threshold = 15.0
    light_candidates = [(name, r) for name, r in sorted_results
                        if r['total_sec'] < threshold and 'error' not in r]
    print(f"Threshold: < {threshold}s total time")
    print(f"Light models ({len(light_candidates)}):")
    light_names = []
    for name, r in light_candidates:
        light_names.append(name)
        print(f"  - {name:<22} {r['total_sec']:>6.1f}s")

    print(f"\nPython list for pipeline.py:")
    print(f"  {sorted(light_names)}")


if __name__ == '__main__':
    main()
