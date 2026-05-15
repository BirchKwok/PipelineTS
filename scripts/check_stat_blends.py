import itertools
import numpy as np
import pandas as pd

from PipelineTS.dataset import LoadElectric, LoadWebSales, LoadSupermarketIncoming, LoadMessagesSent
from PipelineTS.pipeline import ModelPipeline


DATASETS = {
    "electric": (LoadElectric, "date", "value"),
    "web_sales": (LoadWebSales, "date", "type_a"),
    "supermarket": (LoadSupermarketIncoming, "date", "goods_cnt"),
    "messages": (LoadMessagesSent, "date", "ta"),
}


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def run_one(name):
    loader, time_col, target_col = DATASETS[name]
    data = loader()[[time_col, target_col]].dropna().tail(240).reset_index(drop=True)
    data[time_col] = pd.to_datetime(data[time_col])
    train = data.iloc[:-12].reset_index(drop=True)
    valid = data.iloc[-12:].reset_index(drop=True)
    models = ["stat_ensemble", "theta", "ets", "seasonal_naive", "multi_step_model", "multi_output_model", "regressor_chain"]
    pipe = ModelPipeline(time_col=time_col, target_col=target_col, lags=12, include_models=models, quantile=None, scaler=True, cv=2, time_limit=60)
    lb = pipe.fit(train, valid_data=valid)
    y = valid[target_col].to_numpy(dtype=float)
    preds = {}
    for model in lb["model"].tolist():
        try:
            preds[model] = pipe.predict(12, model_name=model)[target_col].to_numpy(dtype=float)[:12]
        except Exception:
            pass
    best = min((mae(y, p), m) for m, p in preds.items())
    best_blend = (best[0], best[1], None)
    keys = list(preds)
    for r in (2, 3):
        for combo in itertools.combinations(keys, r):
            for raw in itertools.product(range(6), repeat=r):
                if sum(raw) == 0:
                    continue
                w = np.asarray(raw, dtype=float)
                w = w / w.sum()
                pred = sum(wi * preds[mi] for wi, mi in zip(w, combo))
                score = mae(y, pred)
                if score < best_blend[0]:
                    best_blend = (score, combo, w.round(3).tolist())
    print(f"{name} best_single={best[1]}:{best[0]:.4f} best_blend={best_blend[1]}:{best_blend[0]:.4f} weights={best_blend[2]}")


def main():
    for name in DATASETS:
        run_one(name)


if __name__ == "__main__":
    main()
