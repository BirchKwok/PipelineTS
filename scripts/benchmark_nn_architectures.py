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
    "synthetic": (None, "date", "value"),
    "air_passengers": ("LoadAirPassengers", "Month", "Passengers"),
    "electric": ("LoadElectric", "date", "value"),
    "electric_production": ("LoadElectricProduction", "date", "value"),
    "etth1": ("LoadETTh1", "date", "OT"),
    "etth2": ("LoadETTh2", "date", "OT"),
    "ettm1": ("LoadETTm1", "date", "OT"),
    "ettm2": ("LoadETTm2", "date", "OT"),
    "inventory": ("LoadInventory", "complete_date", "num1"),
    "messages": ("LoadMessagesSent", "date", "ta"),
    "messages_hour": ("LoadMessagesSentHour", "date", "ta"),
    "web_sales": ("LoadWebSales", "date", "type_a"),
    "supermarket": ("LoadSupermarketIncoming", "date", "goods_cnt"),
}


LIGHT_ARCH_CONFIGS = {
    "d_linear": {"dropout": 0.0, "use_gtb": False, "use_residual_gate": False},
    "n_linear": {"dropout": 0.0, "use_gtb": False, "use_residual_gate": False},
    "n_beats": {"num_stacks": 1, "num_blocks": 1, "num_layers": 2, "layer_widths": 64, "expansion_coeff_dim": 8, "dropout": 0.05, "use_gtb": False},
    "n_hits": {"num_stacks": 1, "num_blocks": 1, "num_layers": 1, "layer_widths": 64, "dropout": 0.05, "use_gtb": False},
    "tide": {"hidden_size": 64, "decoder_output_dim": 16, "temporal_decoder_hidden": 16, "num_encoder_layers": 1, "num_decoder_layers": 1, "dropout": 0.05, "use_gtb": False},
    "tcn": {"num_levels": 2, "hidden_channels": 16, "dropout": 0.05, "use_gtb": False},
    "patch_rnn": {"kernel_size": 6, "multi_steps": True, "dropout": 0.05, "use_gtb": False},
    "stacking_rnn": {"blocks": 1, "dropout": 0.05, "use_gtb": False},
    "time2vec": {"use_gtb": False},
    "gau": {"level": 1, "dropout": 0.05, "use_gtb": False},
    "transformer": {"d_model": 32, "nhead": 2, "num_encoder_layers": 1, "dim_feedforward": 64, "dropout": 0.05, "use_gtb": False},
    "tft": {"hidden_size": 32, "lstm_layers": 1, "n_heads": 2, "dropout": 0.05, "use_gtb": False},
    "deepar": {"d_model": 32, "n_blocks": 1, "n_rwkv_blocks": 1, "dropout": 0.05},
    "itransformer": {"d_model": 32, "n_heads": 2, "d_ff": 64, "e_layers": 1, "dropout": 0.05},
    "srs_net": {"d_model": 32, "n_heads": 2, "top_k_ratio": 0.35, "dropout": 0.05},
}


EXTRA_VARIANT_CONFIGS = {
    "n_beats_tiny": ("n_beats", {"num_stacks": 1, "num_blocks": 1, "num_layers": 1, "layer_widths": 64, "expansion_coeff_dim": 8, "dropout": 0.05, "use_gtb": False}),
    "n_beats_small": ("n_beats", {"num_stacks": 1, "num_blocks": 2, "num_layers": 2, "layer_widths": 64, "expansion_coeff_dim": 8, "dropout": 0.05, "use_gtb": False}),
    "n_beats_medium": ("n_beats", {"num_stacks": 2, "num_blocks": 1, "num_layers": 2, "layer_widths": 96, "expansion_coeff_dim": 16, "dropout": 0.05, "use_gtb": False}),
    "n_beats_interpretable": ("n_beats", {"generic_architecture": False, "num_stacks": 2, "num_blocks": 1, "num_layers": 2, "layer_widths": 96, "dropout": 0.05, "use_gtb": False}),
    "gau_deep": ("gau", {"level": 3, "dropout": 0.2, "use_gtb": False}),
    "n_hits_tiny": ("n_hits", {"num_stacks": 1, "num_blocks": 1, "num_layers": 1, "layer_widths": 64, "dropout": 0.05, "use_gtb": False}),
    "n_hits_small": ("n_hits", {"num_stacks": 2, "num_blocks": 1, "num_layers": 1, "layer_widths": 64, "dropout": 0.05, "use_gtb": False}),
    "n_hits_medium": ("n_hits", {"num_stacks": 2, "num_blocks": 1, "num_layers": 2, "layer_widths": 96, "dropout": 0.05, "use_gtb": False}),
    "n_hits_compact96": ("n_hits", {"num_stacks": 3, "num_blocks": 1, "num_layers": 2, "layer_widths": 96, "dropout": 0.05, "use_gtb": False}),
    "n_hits_compact128": ("n_hits", {"num_stacks": 3, "num_blocks": 1, "num_layers": 2, "layer_widths": 128, "dropout": 0.05, "use_gtb": False}),
    "n_hits_compact256": ("n_hits", {"num_stacks": 3, "num_blocks": 1, "num_layers": 2, "layer_widths": 256, "dropout": 0.05, "use_gtb": False}),
    "deepar_tiny": ("deepar", {"d_model": 24, "n_blocks": 1, "n_rwkv_blocks": 1, "dropout": 0.05}),
    "deepar_small": ("deepar", {"d_model": 32, "n_blocks": 1, "n_rwkv_blocks": 1, "dropout": 0.05}),
    "deepar_medium": ("deepar", {"d_model": 48, "n_blocks": 1, "n_rwkv_blocks": 2, "dropout": 0.05}),
    "deepar_refine": ("deepar", {"d_model": 32, "n_blocks": 2, "n_rwkv_blocks": 1, "dropout": 0.05}),
    "tide_tiny": ("tide", {"hidden_size": 32, "decoder_output_dim": 8, "temporal_decoder_hidden": 8, "num_encoder_layers": 1, "num_decoder_layers": 1, "dropout": 0.05, "use_gtb": False}),
    "tide_small": ("tide", {"hidden_size": 64, "decoder_output_dim": 16, "temporal_decoder_hidden": 16, "num_encoder_layers": 1, "num_decoder_layers": 1, "dropout": 0.05, "use_gtb": False}),
    "tide_medium": ("tide", {"hidden_size": 96, "decoder_output_dim": 16, "temporal_decoder_hidden": 24, "num_encoder_layers": 2, "num_decoder_layers": 1, "dropout": 0.05, "use_gtb": False}),
    "tcn_tiny": ("tcn", {"num_levels": 1, "hidden_channels": 12, "dropout": 0.05, "use_gtb": False}),
    "tcn_small": ("tcn", {"num_levels": 2, "hidden_channels": 16, "dropout": 0.05, "use_gtb": False}),
    "tcn_medium": ("tcn", {"num_levels": 3, "hidden_channels": 24, "dropout": 0.1, "use_gtb": False}),
    "gau_tiny": ("gau", {"level": 1, "dropout": 0.05, "use_gtb": False}),
    "gau_medium": ("gau", {"level": 2, "dropout": 0.1, "use_gtb": False}),
    "transformer_tiny": ("transformer", {"d_model": 32, "nhead": 2, "num_encoder_layers": 1, "dim_feedforward": 64, "dropout": 0.05, "use_gtb": False}),
    "transformer_small": ("transformer", {"d_model": 48, "nhead": 2, "num_encoder_layers": 2, "dim_feedforward": 128, "dropout": 0.05, "use_gtb": False}),
    "tft_tiny": ("tft", {"hidden_size": 16, "lstm_layers": 1, "n_heads": 2, "dropout": 0.05, "use_gtb": False}),
    "tft_small": ("tft", {"hidden_size": 32, "lstm_layers": 1, "n_heads": 2, "dropout": 0.05, "use_gtb": False}),
    "patch_rnn_k6": ("patch_rnn", {"kernel_size": 6, "multi_steps": True, "dropout": 0.05, "use_gtb": False}),
    "stacking_rnn_1": ("stacking_rnn", {"blocks": 1, "dropout": 0.05, "use_gtb": False}),
    "stacking_rnn_3": ("stacking_rnn", {"blocks": 3, "dropout": 0.1, "use_gtb": False}),
}


ENHANCEMENT_VARIANT_CONFIGS = {
    "gtb_static": {"use_gtb": True, "routing_mode": "static", "gtb_d_model": 64},
    "gtb_adaptive": {"use_gtb": True, "routing_mode": "adaptive", "gtb_d_model": 64},
    "gtb_adaptive_plus": {"use_gtb": True, "routing_mode": "adaptive_plus", "gtb_d_model": 64},
    "ema": {"use_ema": True, "ema_decay": 0.999},
    "swa": {"use_swa": True, "swa_start_frac": 0.75},
    "residual_gate": {"use_residual_gate": True},
    "warmup": {"warmup_epochs": 10},
    "stability": {
        "use_ema": True,
        "ema_decay": 0.999,
        "use_swa": True,
        "swa_start_frac": 0.75,
        "warmup_epochs": 10,
        "use_residual_gate": True,
    },
    "gtb_adaptive_stability": {
        "use_gtb": True,
        "routing_mode": "adaptive",
        "gtb_d_model": 64,
        "use_ema": True,
        "ema_decay": 0.999,
        "use_swa": True,
        "swa_start_frac": 0.75,
        "warmup_epochs": 10,
        "use_residual_gate": True,
    },
}


GTB_MODELS = {
    "d_linear",
    "n_linear",
    "n_beats",
    "n_hits",
    "tcn",
    "tft",
    "gau",
    "stacking_rnn",
    "time2vec",
    "transformer",
    "tide",
    "patch_rnn",
}


def elapsed_seconds(func):
    start = time.perf_counter()
    value = func()
    return time.perf_counter() - start, value


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    values = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(values))


def wmape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def make_series(n=360, seed=42, freq="D"):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.float64)
    values = (
        10.0
        + 0.02 * idx
        + 2.5 * np.sin(idx / 7.0)
        + 1.2 * np.sin(idx / 31.0)
        + rng.normal(0.0, 0.5, n)
    )
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq=freq),
        "value": values,
    })


def load_dataset(dataset_name, n, seed):
    if dataset_name == "synthetic":
        data = make_series(n=n, seed=seed)
        return data, "date", "value"
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
    if n and len(data) > n:
        data = data.tail(n).reset_index(drop=True)
    return data, time_col, target_col


def import_model_class(model_name):
    import PipelineTS.nn_model as nn_model

    class_name = MODEL_CLASS_NAMES[model_name]
    return getattr(nn_model, class_name)


def supported_kwargs(cls, kwargs):
    sig = inspect.signature(cls)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def count_params(model):
    inner = getattr(model, "model", None)
    torch_module = getattr(inner, "model", inner)
    if torch_module is None or not hasattr(torch_module, "parameters"):
        return None, None
    total = sum(p.numel() for p in torch_module.parameters())
    trainable = sum(p.numel() for p in torch_module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def resolve_variant(model_name, variant):
    if variant == "baseline":
        return model_name, {}
    if variant == "light":
        return model_name, LIGHT_ARCH_CONFIGS.get(model_name, {})
    if variant in ENHANCEMENT_VARIANT_CONFIGS:
        config = dict(ENHANCEMENT_VARIANT_CONFIGS[variant])
        if ("use_gtb" in config or "routing_mode" in config or "gtb_d_model" in config) and model_name not in GTB_MODELS:
            return model_name, None
        if config.get("use_residual_gate") and model_name in {"d_linear", "n_linear"}:
            return model_name, None
        return model_name, config
    if variant in EXTRA_VARIANT_CONFIGS:
        expected_model, config = EXTRA_VARIANT_CONFIGS[variant]
        if expected_model != model_name:
            return model_name, None
        return model_name, config
    return model_name, None


def benchmark_model(model_name, variant, dataset_name, n, lags, epochs, patience, seed, backend):
    cls = import_model_class(model_name)
    _, variant_config = resolve_variant(model_name, variant)
    try:
        data, time_col, target_col = load_dataset(dataset_name, n, seed)
        if len(data) <= lags * 3:
            raise ValueError(f"Dataset {dataset_name} has too few rows for lags={lags}: {len(data)}")
        train = data.iloc[:-lags].reset_index(drop=True)
        valid = data.iloc[-lags:].reset_index(drop=True)
    except Exception as exc:
        return {
            "model": model_name,
            "variant": variant,
            "dataset": dataset_name,
            "n": n,
            "lags": lags,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    base_kwargs = {
        "time_col": time_col,
        "target_col": target_col,
        "lags": lags,
        "quantile": None,
        "epochs": epochs,
        "patience": patience if patience and patience > 0 else max(3, min(epochs, 8)),
        "batch_size": 512,
        "lr_scheduler": None,
        "restore_best_weights": False,
        "accelerator": "cpu",
        "random_state": seed,
        "verbose": False,
    }
    if backend:
        base_kwargs["backend"] = backend
    if variant_config is None:
        return {
            "model": model_name,
            "variant": variant,
            "dataset": dataset_name,
            "n": n,
            "lags": lags,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "status": "skipped",
            "error": "variant does not apply to model",
        }
    base_kwargs.update(variant_config)
    kwargs = supported_kwargs(cls, base_kwargs)
    ignored_kwargs = sorted(set(base_kwargs) - set(kwargs))
    row = {
        "model": model_name,
        "variant": variant,
        "dataset": dataset_name,
        "rows": int(len(data)),
        "n": n,
        "lags": lags,
        "epochs": epochs,
        "patience": kwargs.get("patience"),
        "seed": seed,
        "kwargs": {k: v for k, v in kwargs.items() if k not in {"time_col", "target_col"}},
        "ignored_kwargs": ignored_kwargs,
    }
    try:
        model = cls(**kwargs)
        total_params, trainable_params = count_params(model)
        row["params"] = total_params
        row["trainable_params"] = trainable_params
        fit_seconds, _ = elapsed_seconds(lambda: model.fit(train))
        predict_seconds, pred = elapsed_seconds(lambda: model.predict(lags))
        y_true = valid[target_col].to_numpy(dtype=np.float64)
        y_pred = pred[target_col].to_numpy(dtype=np.float64)[:lags]
        row.update({
            "status": "ok",
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "smape": smape(y_true, y_pred),
            "wmape": wmape(y_true, y_pred),
            "pred_rows": int(len(pred)),
        })
    except Exception as exc:
        row.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })
    return row


def print_summary(rows):
    print("NN architecture benchmark summary")
    print("=" * 150)
    print(f"{'dataset':<14} {'model':<16} {'variant':<22} {'seed':>5} {'status':<8} {'params':>10} {'fit(s)':>10} {'pred(s)':>10} {'mae':>12} {'smape':>10} {'wmape':>10}")
    print("-" * 150)
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row.get('dataset', ''):<14} {row['model']:<16} {row['variant']:<22} {row.get('seed', ''):>5} {row['status']:<8} "
                f"{str(row.get('trainable_params')):>10} {row['fit_seconds']:>10.3f} "
                f"{row['predict_seconds']:>10.4f} {row['mae']:>12.6f} {row['smape']:>10.4f} {row['wmape']:>10.4f}"
            )
        elif row["status"] == "skipped":
            print(f"{row.get('dataset', ''):<14} {row['model']:<16} {row['variant']:<22} {row.get('seed', ''):>5} {row['status']:<8} {row.get('error', '')}")
        else:
            print(f"{row.get('dataset', ''):<14} {row['model']:<16} {row['variant']:<22} {row.get('seed', ''):>5} {row['status']:<8} {row.get('error', '')}")
    print("=" * 150)


def print_aggregate_summary(rows):
    baselines = {
        (row.get("dataset"), row.get("model"), row.get("seed")): row
        for row in rows
        if row.get("status") == "ok" and row.get("variant") == "baseline"
    }
    stats = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("variant") == "baseline":
            continue
        key = (row.get("model"), row.get("variant"))
        base = baselines.get((row.get("dataset"), row.get("model"), row.get("seed")))
        if not base or not base.get("mae") or not base.get("trainable_params"):
            continue
        item = stats.setdefault(key, {"mae_ratios": [], "smape_ratios": [], "param_ratios": [], "fit_ratios": [], "wins": 0, "n": 0})
        mae_ratio = row["mae"] / base["mae"]
        smape_ratio = row["smape"] / base["smape"] if base.get("smape") else np.nan
        param_ratio = row["trainable_params"] / base["trainable_params"]
        fit_ratio = row["fit_seconds"] / base["fit_seconds"] if base.get("fit_seconds") else np.nan
        item["mae_ratios"].append(mae_ratio)
        item["smape_ratios"].append(smape_ratio)
        item["param_ratios"].append(param_ratio)
        item["fit_ratios"].append(fit_ratio)
        item["wins"] += int(mae_ratio < 1.0)
        item["n"] += 1
    if not stats:
        return
    print("Aggregate vs baseline")
    print("=" * 134)
    print(f"{'model':<16} {'variant':<24} {'runs':>6} {'wins':>8} {'avg_mae_ratio':>16} {'avg_smape_ratio':>17} {'avg_param_ratio':>16} {'avg_fit_ratio':>14}")
    print("-" * 134)
    for (model, variant), item in sorted(stats.items()):
        print(
            f"{model:<16} {variant:<24} {item['n']:>6} {item['wins']:>8} "
            f"{float(np.nanmean(item['mae_ratios'])):>16.4f} {float(np.nanmean(item['smape_ratios'])):>17.4f} "
            f"{float(np.nanmean(item['param_ratios'])):>16.4f} {float(np.nanmean(item['fit_ratios'])):>14.4f}"
        )
    print("=" * 134)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="d_linear,n_linear,n_beats,n_hits,tide,tcn,patch_rnn,stacking_rnn,time2vec,gau,transformer,tft,deepar")
    parser.add_argument("--variants", default="baseline,light")
    parser.add_argument("--datasets", default="synthetic")
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--lags", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--jsonl-out", default="")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    rows = []
    jsonl_path = Path(args.jsonl_out) if args.jsonl_out else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")
    for dataset_name in datasets:
        for model_name in models:
            for variant in variants:
                for seed in seeds:
                    row = benchmark_model(model_name, variant, dataset_name, args.n, args.lags, args.epochs, args.patience, seed, args.backend)
                    rows.append(row)
                    if jsonl_path:
                        with jsonl_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    metric = row.get("smape", row.get("error", "-"))
                    print(
                        f"[{len(rows)}] {dataset_name}/{model_name}/{variant}/seed={seed} "
                        f"status={row.get('status')} smape={metric} fit={row.get('fit_seconds', '-')}",
                        flush=True,
                    )

    print_summary(rows)
    print_aggregate_summary(rows)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON results to {path}")


if __name__ == "__main__":
    main()
