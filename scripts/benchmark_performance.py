import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def elapsed_seconds(func):
    start = time.perf_counter()
    value = func()
    return time.perf_counter() - start, value


def make_series(n=600, seed=42, freq="D"):
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


def bench_split_series(n, lags, repeat):
    from PipelineTS.preprocessing.sequence import (
        lag_splits,
        split_series,
        split_series_multivariate,
    )

    y = make_series(n)["value"].to_numpy(dtype=np.float64)
    features = np.column_stack([y, np.sin(np.arange(n) / 7.0), np.cos(np.arange(n) / 31.0)])
    rows = []
    for name, fn in [
        ("split_series", lambda: split_series(y, y, window_size=lags, pred_steps=lags)),
        ("lag_splits", lambda: lag_splits(y, window_size=lags)),
        ("split_series_multivariate", lambda: split_series_multivariate(features, y, window_size=lags, pred_steps=lags)),
    ]:
        times = []
        shape = None
        for _ in range(repeat):
            sec, out = elapsed_seconds(fn)
            times.append(sec)
            first = out[0] if isinstance(out, tuple) else out
            shape = tuple(first.shape)
        rows.append({
            "case": name,
            "n": n,
            "lags": lags,
            "repeat": repeat,
            "seconds_mean": float(np.mean(times)),
            "seconds_min": float(np.min(times)),
            "output_shape": shape,
        })
    return rows


def bench_nn_model(n, lags, epochs, model_name):
    from PipelineTS.pipeline import ModelPipeline

    df = make_series(n)
    train = df.iloc[:-lags * 2].reset_index(drop=True)
    valid = df.iloc[-lags * 2:].reset_index(drop=True)
    kwargs = {
        "time_col": "date",
        "target_col": "value",
        "lags": lags,
        "include_models": [model_name],
        "scaler": True,
        "quantile": None,
        "cv": 2,
        "accelerator": "cpu",
        "random_state": 42,
        f"{model_name}__epochs": epochs,
        f"{model_name}__patience": max(3, min(epochs, 10)),
        f"{model_name}__verbose": False,
    }
    pipe = ModelPipeline(**kwargs)
    fit_sec, lb = elapsed_seconds(lambda: pipe.fit(train, valid_data=valid))
    pred_sec, pred = elapsed_seconds(lambda: pipe.predict(n=lags))
    return {
        "case": f"nn_{model_name}",
        "n": n,
        "lags": lags,
        "epochs": epochs,
        "fit_seconds": fit_sec,
        "predict_seconds": pred_sec,
        "leaderboard": lb.to_dict(orient="records"),
        "pred_rows": int(len(pred)),
    }


def bench_pipeline(n, lags, models):
    from PipelineTS.pipeline import ModelPipeline

    df = make_series(n)
    train = df.iloc[:-lags * 2].reset_index(drop=True)
    valid = df.iloc[-lags * 2:].reset_index(drop=True)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    kwargs = {
        "time_col": "date",
        "target_col": "value",
        "lags": lags,
        "include_models": model_list,
        "scaler": True,
        "quantile": None,
        "cv": 2,
        "accelerator": "cpu",
        "random_state": 42,
    }
    for m in model_list:
        if m in {"d_linear", "n_linear", "tide", "tcn", "n_hits", "patch_rnn"}:
            kwargs[f"{m}__epochs"] = 30
            kwargs[f"{m}__patience"] = 5
            kwargs[f"{m}__verbose"] = False
    pipe = ModelPipeline(**kwargs)
    fit_sec, lb = elapsed_seconds(lambda: pipe.fit(train, valid_data=valid))
    pred_sec, pred = elapsed_seconds(lambda: pipe.predict(n=lags))
    return {
        "case": "pipeline",
        "n": n,
        "lags": lags,
        "models": model_list,
        "fit_seconds": fit_sec,
        "predict_seconds": pred_sec,
        "leaderboard": lb.to_dict(orient="records"),
        "pred_rows": int(len(pred)),
        "failed_models": pipe.failed_models,
        "skipped_models": pipe.skipped_models,
    }


def bench_smart_router(n, lags, max_models, preset, search_strategy, time_limit):
    from PipelineTS.pipeline import SmartRouter

    df = make_series(n)
    train = df.iloc[:-lags * 2].reset_index(drop=True)
    valid = df.iloc[-lags * 2:].reset_index(drop=True)
    router = SmartRouter(
        time_col="date",
        target_col="value",
        n_predict=lags,
        preset=preset,
        max_models=max_models,
        cv=2,
        time_limit=time_limit,
        search_strategy=search_strategy,
        ensemble_strategy="none",
        accelerator="cpu",
        random_state=42,
        verbose=False,
    )
    fit_sec, _ = elapsed_seconds(lambda: router.fit(train, valid_data=valid))
    pred_sec = 0.0
    pred = pd.DataFrame()
    predict_error = None
    if router.leader_board_ is not None and not router.leader_board_.empty:
        try:
            pred_sec, pred = elapsed_seconds(lambda: router.predict(n=lags, use_ensemble=False))
        except Exception as exc:
            predict_error = f"{type(exc).__name__}: {exc}"
    lb = router.leader_board_
    return {
        "case": "smart_router",
        "n": n,
        "lags": lags,
        "max_models": max_models,
        "preset": preset,
        "search_strategy": search_strategy,
        "time_limit": time_limit,
        "fit_seconds": fit_sec,
        "predict_seconds": pred_sec,
        "leaderboard": [] if lb is None else lb.to_dict(orient="records"),
        "pred_rows": int(len(pred)),
        "predict_error": predict_error,
        "failed_models": [] if router.pipeline_ is None else router.pipeline_.failed_models,
        "skipped_models": [] if router.pipeline_ is None else router.pipeline_.skipped_models,
        "dataset_benchmark_rows": 0 if router.dataset_benchmark_ is None else len(router.dataset_benchmark_),
    }


def print_summary(results):
    print("\nPerformance benchmark summary")
    print("=" * 80)
    for item in results:
        case = item.get("case")
        if case in {"split_series", "lag_splits", "split_series_multivariate"}:
            print(f"{case:<28} mean={item['seconds_mean']:.6f}s min={item['seconds_min']:.6f}s shape={item['output_shape']}")
        else:
            print(f"{case:<28} fit={item.get('fit_seconds', 0):.3f}s predict={item.get('predict_seconds', 0):.3f}s")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="split,pipeline,smart", help="Comma-separated: split,nn,pipeline,smart")
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--lags", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--nn-model", default="d_linear")
    parser.add_argument("--nn-epochs", type=int, default=30)
    parser.add_argument("--pipeline-models", default="multi_output_model,d_linear")
    parser.add_argument("--smart-max-models", type=int, default=2)
    parser.add_argument("--smart-preset", default="fast")
    parser.add_argument("--smart-search", default="basic")
    parser.add_argument("--smart-time-limit", type=float, default=60.0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cases = {c.strip() for c in args.cases.split(",") if c.strip()}
    results = []

    if "split" in cases:
        results.extend(bench_split_series(args.n, args.lags, args.repeat))
    if "nn" in cases:
        results.append(bench_nn_model(args.n, args.lags, args.nn_epochs, args.nn_model))
    if "pipeline" in cases:
        results.append(bench_pipeline(args.n, args.lags, args.pipeline_models))
    if "smart" in cases:
        results.append(bench_smart_router(
            args.n, args.lags, args.smart_max_models, args.smart_preset,
            args.smart_search, args.smart_time_limit,
        ))

    print_summary(results)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON results to {path}")


if __name__ == "__main__":
    main()
