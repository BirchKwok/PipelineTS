import argparse
import json
import shutil
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DATASET_ALIASES = {
    "air_passengers": "AirPassengers",
    "airpassengers": "AirPassengers",
    "electric": "Electric_Production",
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "messages": "Messages_Sent",
    "messages_hour": "Messages_Sent_Hour",
    "supermarket": "Supermarket_Incoming",
    "web_sales": "Web_Sales",
}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def dataset_alias(name):
    aliases = {
        "AirPassengers": "air_passengers",
        "Electric_Production": "electric",
        "Messages_Sent": "messages",
        "Messages_Sent_Hour": "messages_hour",
        "Supermarket_Incoming": "supermarket",
        "Web_Sales": "web_sales",
    }
    return aliases.get(name, name.lower())


def discover_builtin_datasets():
    from PipelineTS.dataset import BuiltInSeriesData

    source = BuiltInSeriesData(print_file_list=False)
    datasets = {}
    for filename in source.file_list:
        if not filename.endswith(".csv"):
            continue
        name = Path(filename).stem
        wrapper = source[name]
        datasets[name] = {
            "name": name,
            "alias": dataset_alias(name),
            "time_col": wrapper.time_col,
            "target_col": wrapper.target_col,
        }
    return datasets


def resolve_dataset_names(names):
    datasets = discover_builtin_datasets()
    lookup = {}
    for name, spec in datasets.items():
        keys = {
            name,
            name.lower(),
            spec["alias"],
            spec["alias"].lower(),
            name.lower().replace("_", ""),
            spec["alias"].lower().replace("_", ""),
        }
        for key in keys:
            lookup[key] = name
    for alias, canonical in DATASET_ALIASES.items():
        if canonical in datasets:
            lookup[alias] = canonical

    if not names or any(item.lower() == "all" for item in names):
        return [datasets[name] for name in sorted(datasets)]

    resolved = []
    for item in names:
        key = item.lower()
        canonical = lookup.get(key)
        if canonical is None:
            choices = ", ".join(spec["alias"] for spec in resolve_dataset_names(["all"]))
            raise ValueError(f"Unknown dataset: {item}. Available: {choices}")
        resolved.append(datasets[canonical])
    return resolved


def elapsed_seconds(func):
    start = time.perf_counter()
    value = func()
    return time.perf_counter() - start, value


def load_dataset(dataset_spec, n_tail):
    from PipelineTS.dataset import BuiltInSeriesData

    source = BuiltInSeriesData(print_file_list=False)
    wrapper = source[dataset_spec["name"]]
    time_col = dataset_spec["time_col"]
    target_col = dataset_spec["target_col"]
    data = wrapper[[time_col, target_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[time_col, target_col])
    data = data.sort_values(time_col).reset_index(drop=True)
    if n_tail and len(data) > n_tail:
        data = data.tail(n_tail).reset_index(drop=True)
    return data, time_col, target_col


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def wmape(y_true, y_pred):
    denom = np.sum(np.abs(y_true)) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def evaluate_predictions(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    y_true = np.asarray(y_true[:n], dtype=np.float64)
    y_pred = np.asarray(y_pred[:n], dtype=np.float64)
    return {
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "eval_rows": int(n),
    }


def compact_leaderboard(df, limit=5):
    if df is None or len(df) == 0:
        return []
    out = df.head(limit).copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float).round(6)
    return out.to_dict(orient="records")


def benchmark_smartrouter(train, valid, time_col, target_col, dataset_name, args):
    from PipelineTS.pipeline import SmartRouter

    row = {
        "dataset": dataset_name,
        "method": "smartrouter",
        "status": "error",
    }
    try:
        kwargs = {
            "time_col": time_col,
            "target_col": target_col,
            "n_predict": args.horizon,
            "preset": args.smart_preset,
            "time_limit": args.time_limit,
            "accelerator": args.accelerator,
            "random_state": args.seed,
            "verbose": False,
        }
        if args.smart_max_models > 0:
            kwargs["max_models"] = args.smart_max_models
        if args.smart_cv > 0:
            kwargs["cv"] = args.smart_cv
        if args.smart_search:
            kwargs["search_strategy"] = args.smart_search
        if args.smart_ensemble:
            kwargs["ensemble_strategy"] = args.smart_ensemble
        if args.smart_hpo:
            kwargs["hpo_strategy"] = args.smart_hpo

        router = SmartRouter(**kwargs)
        fit_seconds, _ = elapsed_seconds(lambda: router.fit(train, valid_data=valid))
        predict_seconds, pred = elapsed_seconds(
            lambda: router.predict(n=args.horizon, use_ensemble=args.smart_use_ensemble)
        )
        y_true = valid[target_col].to_numpy(dtype=np.float64)
        y_pred = pred[target_col].to_numpy(dtype=np.float64)
        row.update(evaluate_predictions(y_true, y_pred))
        lb = router.leader_board_
        row.update({
            "status": "ok",
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "total_seconds": float(fit_seconds + predict_seconds),
            "best_model": None if lb is None or lb.empty else str(lb.iloc[0]["model"]),
            "leaderboard": compact_leaderboard(lb),
            "failed_models": [] if router.pipeline_ is None else router.pipeline_.failed_models,
            "skipped_models": [] if router.pipeline_ is None else router.pipeline_.skipped_models,
            "baseline_guardrail": getattr(router, "_baseline_guardrail", None),
            "strategy": getattr(router, "strategy_", None),
        })
    except Exception as exc:
        row.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        })
    return row


def make_autogluon_frame(data, time_col, target_col, dataset_name):
    return pd.DataFrame({
        "item_id": dataset_name,
        "timestamp": pd.to_datetime(data[time_col]),
        "target": data[target_col].astype(float),
    })


def benchmark_autogluon(train, valid, time_col, target_col, dataset_name, args):
    row = {
        "dataset": dataset_name,
        "method": "autogluon",
        "status": "error",
    }
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except Exception as exc:
        row["status"] = "skipped"
        row["error"] = f"AutoGluon unavailable: {type(exc).__name__}: {exc}"
        return row

    try:
        ag_train_df = make_autogluon_frame(train, time_col, target_col, dataset_name)
        ts_train = TimeSeriesDataFrame.from_data_frame(
            ag_train_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        freq = pd.infer_freq(train[time_col])
        path = Path(args.output_dir) / "autogluon_models" / dataset_name
        if path.exists() and args.clean_autogluon_path:
            shutil.rmtree(path)
        predictor_kwargs = {
            "target": "target",
            "prediction_length": args.horizon,
            "eval_metric": "MAE",
            "path": str(path),
            "verbosity": 0,
            "log_to_file": False,
        }
        if freq:
            predictor_kwargs["freq"] = freq
        predictor = TimeSeriesPredictor(**predictor_kwargs)
        fit_kwargs = {
            "train_data": ts_train,
            "time_limit": int(args.time_limit) if args.time_limit else None,
            "presets": args.autogluon_preset or None,
            "random_seed": args.seed,
            "verbosity": 0,
        }
        fit_seconds, _ = elapsed_seconds(lambda: predictor.fit(**fit_kwargs))
        predict_seconds, pred = elapsed_seconds(lambda: predictor.predict(ts_train, random_seed=args.seed))
        if "mean" in pred.columns:
            y_pred = pred["mean"].to_numpy(dtype=np.float64)
        elif "target" in pred.columns:
            y_pred = pred["target"].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"No mean/target column in AutoGluon prediction: {list(pred.columns)}")
        y_true = valid[target_col].to_numpy(dtype=np.float64)
        row.update(evaluate_predictions(y_true, y_pred))
        try:
            lb = predictor.leaderboard(data=ts_train, display=False).head(5)
        except Exception:
            lb = pd.DataFrame()
        row.update({
            "status": "ok",
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "total_seconds": float(fit_seconds + predict_seconds),
            "best_model": str(getattr(predictor, "model_best", None)),
            "leaderboard": compact_leaderboard(lb),
        })
    except Exception as exc:
        row.update({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        })
    return row


def compare_dataset(dataset_spec, args):
    dataset_name = dataset_spec["alias"]
    data, time_col, target_col = load_dataset(dataset_spec, args.n_tail)
    if len(data) <= args.horizon * 3:
        raise ValueError(f"Dataset {dataset_name} too short: rows={len(data)}, horizon={args.horizon}")
    train = data.iloc[:-args.horizon].reset_index(drop=True)
    valid = data.iloc[-args.horizon:].reset_index(drop=True)
    rows = []
    if "smartrouter" in args.methods:
        rows.append(benchmark_smartrouter(train, valid, time_col, target_col, dataset_name, args))
    if "autogluon" in args.methods:
        rows.append(benchmark_autogluon(train, valid, time_col, target_col, dataset_name, args))
    return rows


def print_progress(row):
    if row.get("status") == "ok":
        print(
            f"DONE {row['dataset']:<12} {row['method']:<12} "
            f"mae={row['mae']:.4f} smape={row['smape']:.4f} "
            f"fit={row['fit_seconds']:.1f}s pred={row['predict_seconds']:.2f}s "
            f"best={row.get('best_model')}",
            flush=True,
        )
    else:
        print(
            f"FAIL {row['dataset']:<12} {row['method']:<12} "
            f"status={row.get('status')} error={row.get('error')}",
            flush=True,
        )


def print_summary(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    print("\nAutoGluon vs SmartRouter")
    print("=" * 118)
    print(f"{'dataset':<12} {'method':<12} {'status':<8} {'best_model':<24} {'fit(s)':>9} {'pred(s)':>9} {'MAE':>14} {'SMAPE':>10} {'WMAPE':>10}")
    print("-" * 118)
    for row in rows:
        if row.get("status") == "ok":
            print(
                f"{row['dataset']:<12} {row['method']:<12} ok       "
                f"{str(row.get('best_model'))[:24]:<24} "
                f"{row['fit_seconds']:>9.2f} {row['predict_seconds']:>9.2f} "
                f"{row['mae']:>14.4f} {row['smape']:>10.4f} {row['wmape']:>10.4f}"
            )
        else:
            print(
                f"{row.get('dataset', ''):<12} {row.get('method', ''):<12} "
                f"{row.get('status', 'error'):<8} {str(row.get('error', ''))[:90]}"
            )
    print("=" * 118)
    by_dataset = {}
    for row in ok:
        by_dataset.setdefault(row["dataset"], []).append(row)
    wins = {"smartrouter": 0, "autogluon": 0, "tie": 0}
    print("Dataset winners by MAE")
    for dataset_name in sorted(by_dataset):
        items = by_dataset[dataset_name]
        if len(items) < 2:
            continue
        best = min(items, key=lambda r: r["mae"])
        other = max(items, key=lambda r: r["mae"])
        if abs(best["mae"] - other["mae"]) <= max(1e-9, abs(other["mae"]) * 1e-9):
            wins["tie"] += 1
            print(f"  {dataset_name}: tie")
        else:
            wins[best["method"]] += 1
            ratio = best["mae"] / other["mae"] if other["mae"] else np.nan
            print(f"  {dataset_name}: {best['method']} wins, mae_ratio={ratio:.4f}")
    print(f"Wins: smartrouter={wins['smartrouter']} autogluon={wins['autogluon']} tie={wins['tie']}")
    if ok:
        sr = [r for r in ok if r["method"] == "smartrouter"]
        ag = [r for r in ok if r["method"] == "autogluon"]
        if sr and ag:
            print(f"Avg fit seconds: smartrouter={np.mean([r['fit_seconds'] for r in sr]):.2f}, autogluon={np.mean([r['fit_seconds'] for r in ag]):.2f}")
            print(f"Avg SMAPE: smartrouter={np.mean([r['smape'] for r in sr]):.4f}, autogluon={np.mean([r['smape'] for r in ag]):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="electric,web_sales,supermarket,messages")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--methods", default="smartrouter,autogluon")
    parser.add_argument("--n-tail", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--smart-preset", default="fast")
    parser.add_argument("--smart-max-models", type=int, default=3)
    parser.add_argument("--smart-cv", type=int, default=2)
    parser.add_argument("--smart-search", default="basic")
    parser.add_argument("--smart-ensemble", default="none")
    parser.add_argument("--smart-hpo", default="none")
    parser.add_argument("--smart-use-ensemble", action="store_true")
    parser.add_argument("--autogluon-preset", default="fast_training")
    parser.add_argument("--output-dir", default="tmp_benchmark_results/autogluon_smartrouter")
    parser.add_argument("--json-out", default="tmp_benchmark_results/autogluon_smartrouter/results.json")
    parser.add_argument("--clean-autogluon-path", action="store_true")
    args = parser.parse_args()

    args.datasets = parse_csv(args.datasets)
    args.methods = parse_csv(args.methods)
    dataset_specs = resolve_dataset_names(args.datasets)
    if args.list_datasets:
        for spec in dataset_specs:
            print(
                f"{spec['alias']:<16} name={spec['name']:<24} "
                f"time_col={spec['time_col']:<12} target_col={spec['target_col']}"
            )
        return
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_spec in dataset_specs:
        dataset_rows = compare_dataset(dataset_spec, args)
        rows.extend(dataset_rows)
        for row in dataset_rows:
            print_progress(row)

    print_summary(rows)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Wrote JSON results to {path}")


if __name__ == "__main__":
    main()
