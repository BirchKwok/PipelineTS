import argparse
import inspect
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


MODEL_CLASS_NAMES = {
    "d_linear": "DLinearModel",
    "n_linear": "NLinearModel",
    "n_beats": "NBeatsModel",
    "n_hits": "NHitsModel",
    "tide": "TiDEModel",
    "tcn": "TCNModel",
    "patch_rnn": "PatchRNNModel",
    "stacking_rnn": "StackingRNNModel",
    "time2vec": "Time2VecModel",
    "gau": "GAUModel",
    "transformer": "TransformerModel",
    "tft": "TFTModel",
    "deepar": "DeepARModel",
    "itransformer": "ITransformerModel",
    "srs_net": "SRSNetModel",
}


BUILTIN_DATASETS = {
    "electric": ("LoadElectricDataSets", "date", "value"),
    "messages": ("LoadMessagesSentDataSets", "date", "ta"),
    "messages_hour": ("LoadMessagesSentHourDataSets", "date", "ta"),
    "web_sales": ("LoadWebSales", "date", "type_a"),
    "supermarket": ("LoadSupermarketIncoming", "date", "goods_cnt"),
}


PROFILE_CONFIGS = {
    "cap300": {"epochs": 300, "patience": 30},
    "cap800": {"epochs": 800, "patience": 30},
    "cap800p80": {"epochs": 800, "patience": 80},
    "balanced": {"epochs": 1500, "patience": 80},
    "long": {"epochs": 2500, "patience": 120},
}


MULTIVARIATE_MODELS = {"itransformer", "srs_net"}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_dataset(dataset_name, n_tail):
    if dataset_name not in BUILTIN_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    loader_name, time_col, target_col = BUILTIN_DATASETS[dataset_name]
    import PipelineTS.dataset as dataset_module

    loader = getattr(dataset_module, loader_name)
    data = loader()[[time_col, target_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[time_col, target_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    if n_tail and len(data) > n_tail:
        data = data.tail(n_tail).reset_index(drop=True)
    return data, time_col, target_col


def add_multivariate_features(data, target_col):
    data = data.copy()
    values = data[target_col].astype(float).to_numpy()
    n = len(values)
    idx = np.arange(n, dtype=float)
    scale = float(np.nanstd(values)) or 1.0
    data["__calendar_sin"] = np.sin(2.0 * np.pi * idx / 7.0) * scale * 0.05
    data["__calendar_cos"] = np.cos(2.0 * np.pi * idx / 30.0) * scale * 0.05
    return data


def import_model_class(model_name):
    import PipelineTS.nn_model as nn_model

    return getattr(nn_model, MODEL_CLASS_NAMES[model_name])


def supported_kwargs(cls, kwargs):
    sig = inspect.signature(cls)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def wmape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))


def default_lags(rows, horizon):
    return int(max(horizon, min(24, max(8, round(np.sqrt(rows))))))


def count_params(model):
    torch_module = getattr(getattr(model, "model", None), "model", None)
    if torch_module is None or not hasattr(torch_module, "parameters"):
        return None
    return int(sum(p.numel() for p in torch_module.parameters() if p.requires_grad))


def benchmark_one(model_name, profile, dataset_name, n_tail, horizon, seed, accelerator, batch_size, scheduler):
    row = {
        "model": model_name,
        "profile": profile,
        "dataset": dataset_name,
        "status": "error",
    }
    try:
        data, time_col, target_col = load_dataset(dataset_name, n_tail)
        lags = default_lags(len(data), horizon)
        if len(data) <= lags * 3 + horizon:
            raise ValueError(f"too few rows: rows={len(data)}, lags={lags}, horizon={horizon}")
        train = data.iloc[:-horizon].reset_index(drop=True)
        valid = data.iloc[-horizon:].reset_index(drop=True)
        if model_name in MULTIVARIATE_MODELS:
            train = add_multivariate_features(train, target_col)
            valid = add_multivariate_features(valid, target_col)
        config = PROFILE_CONFIGS[profile]
        kwargs = {
            "time_col": time_col,
            "target_col": target_col,
            "lags": lags,
            "quantile": None,
            "epochs": config["epochs"],
            "patience": config["patience"],
            "batch_size": batch_size,
            "accelerator": accelerator,
            "random_state": seed,
            "verbose": False,
        }
        if scheduler == "none":
            kwargs["lr_scheduler"] = None
        elif scheduler:
            kwargs["lr_scheduler"] = scheduler
        if model_name in MULTIVARIATE_MODELS:
            kwargs["feature_cols"] = [target_col, "__calendar_sin", "__calendar_cos"]
        cls = import_model_class(model_name)
        kwargs = supported_kwargs(cls, kwargs)
        model = cls(**kwargs)
        params = count_params(model)
        t0 = time.perf_counter()
        model.fit(train, valid_data=valid)
        fit_seconds = time.perf_counter() - t0
        pred = model.predict(horizon)
        y_true = valid[target_col].to_numpy(dtype=float)
        y_pred = pred[target_col].to_numpy(dtype=float)[:horizon]
        trained_epochs = len(getattr(getattr(model, "model", None), "training_logs", {}).get("epochs", []))
        train_loss = getattr(getattr(model, "model", None), "training_logs", {}).get("train_loss", [])
        row.update({
            "status": "ok",
            "rows": int(len(data)),
            "lags": int(lags),
            "horizon": int(horizon),
            "epochs": int(config["epochs"]),
            "patience": int(config["patience"]),
            "trained_epochs": int(trained_epochs),
            "params": params,
            "fit_seconds": float(fit_seconds),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "smape": smape(y_true, y_pred),
            "wmape": wmape(y_true, y_pred),
            "last_train_loss": float(train_loss[-1]) if train_loss else None,
            "kwargs": {k: v for k, v in kwargs.items() if k not in {"time_col", "target_col"}},
        })
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print("NN default epoch benchmark")
    print("=" * 150)
    print(f"{'dataset':<14} {'model':<16} {'profile':<10} {'status':<8} {'epochs':>7} {'trained':>7} {'fit(s)':>9} {'MAE':>12} {'SMAPE':>10} {'WMAPE':>10} error")
    print("-" * 150)
    for row in rows:
        if row.get("status") == "ok":
            print(
                f"{row['dataset']:<14} {row['model']:<16} {row['profile']:<10} ok       "
                f"{row['epochs']:>7} {row['trained_epochs']:>7} {row['fit_seconds']:>9.2f} "
                f"{row['mae']:>12.4f} {row['smape']:>10.4f} {row['wmape']:>10.4f}"
            )
        else:
            print(f"{row.get('dataset', ''):<14} {row.get('model', ''):<16} {row.get('profile', ''):<10} error    {'-':>7} {'-':>7} {'-':>9} {'-':>12} {'-':>10} {'-':>10} {row.get('error', '')}")
    print("=" * 150)
    if not ok_rows:
        return
    grouped = {}
    for row in ok_rows:
        key = (row["dataset"], row["model"])
        grouped.setdefault(key, []).append(row)
    winners = {}
    for key, items in grouped.items():
        best = min(items, key=lambda r: r["smape"])
        winners[best["profile"]] = winners.get(best["profile"], 0) + 1
    print("Profile wins by dataset/model on SMAPE")
    for profile, count in sorted(winners.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {profile}: {count}")
    ratios = {}
    for key, items in grouped.items():
        base = next((r for r in items if r["profile"] == "cap300"), None)
        if base is None or base["smape"] <= 0:
            continue
        for row in items:
            item = ratios.setdefault(row["profile"], [])
            item.append(row["smape"] / base["smape"])
    print("Average SMAPE ratio vs cap300")
    for profile, values in sorted(ratios.items()):
        print(f"  {profile}: {float(np.mean(values)):.4f} over {len(values)} runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="d_linear,n_linear,n_beats,n_hits,tide,tcn,patch_rnn,stacking_rnn,time2vec,gau,transformer,tft,deepar")
    parser.add_argument("--profiles", default="cap300,cap800,balanced")
    parser.add_argument("--datasets", default="electric,web_sales,supermarket,messages")
    parser.add_argument("--n-tail", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scheduler", default="")
    parser.add_argument("--json-out", default="tmp_benchmark_results/nn_default_epochs.json")
    args = parser.parse_args()

    models = parse_csv(args.models)
    profiles = parse_csv(args.profiles)
    datasets = parse_csv(args.datasets)
    rows = []
    for dataset_name in datasets:
        for model_name in models:
            for profile in profiles:
                rows.append(benchmark_one(
                    model_name, profile, dataset_name, args.n_tail, args.horizon,
                    args.seed, args.accelerator, args.batch_size, args.scheduler,
                ))
                latest = rows[-1]
                if latest.get("status") == "ok":
                    print(f"DONE {dataset_name} {model_name} {profile} smape={latest['smape']:.4f} trained={latest['trained_epochs']} fit={latest['fit_seconds']:.1f}s", flush=True)
                else:
                    print(f"FAIL {dataset_name} {model_name} {profile} {latest.get('error', '')}", flush=True)
    summarize(rows)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON results to {path}")


if __name__ == "__main__":
    main()
