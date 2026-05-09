import argparse
import builtins
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


@contextmanager
def external_backend_guard():
    original_import = builtins.__import__
    blocked = "stats" + "models"

    def guarded_import(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError("blocked external statistical backend")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import


def elapsed_ms(func):
    start = time.perf_counter()
    value = func()
    return (time.perf_counter() - start) * 1000.0, value


def make_series(n, rng):
    idx = np.arange(n, dtype=np.float64)
    return np.sin(idx / 8.0) + 0.002 * idx + rng.normal(0.0, 0.1, n)


def bench_native_stats(sizes):
    from PipelineTS.utils.native_stats import (
        acf,
        adf_test,
        kpss_test,
        ljung_box,
        pacf,
        seasonal_decompose,
    )

    rng = np.random.default_rng(42)
    print("native_stats", flush=True)
    print("n,acf_ms,pacf_ms,ljung_box_ms,adf_ms,kpss_ms,decompose_ms", flush=True)
    for n in sizes:
        y = make_series(n, rng)
        acf_ms, _ = elapsed_ms(lambda: acf(y, nlags=40))
        pacf_ms, _ = elapsed_ms(lambda: pacf(y, nlags=40))
        lb_ms, _ = elapsed_ms(lambda: ljung_box(y, lags=20))
        adf_ms, _ = elapsed_ms(lambda: adf_test(y))
        kpss_ms, _ = elapsed_ms(lambda: kpss_test(y))
        decomp_ms, _ = elapsed_ms(lambda: seasonal_decompose(y, period=24))
        print(
            f"{n},{acf_ms:.3f},{pacf_ms:.3f},{lb_ms:.3f},"
            f"{adf_ms:.3f},{kpss_ms:.3f},{decomp_ms:.3f}",
            flush=True,
        )


def bench_auto_arima(sizes, max_p, max_q):
    from PipelineTS.statistic_model.auto_arima import AutoARIMAModel

    rng = np.random.default_rng(123)
    print("auto_arima", flush=True)
    print("n,candidates,fit_ms,predict_ms,order", flush=True)
    candidates = (max_p + 1) * (max_q + 1)
    for n in sizes:
        y = make_series(n, rng)
        data = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=n, freq="D"),
            "y": y,
        })
        model = AutoARIMAModel(
            time_col="ds",
            target_col="y",
            max_p=max_p,
            max_q=max_q,
            max_d=1,
            quantile=None,
        )
        fit_ms, _ = elapsed_ms(lambda: model.fit(data))
        pred_ms, pred = elapsed_ms(lambda: model.predict(12))
        if len(pred) != 12 or pred["y"].isna().any():
            raise RuntimeError("invalid AutoARIMA prediction output")
        print(f"{n},{candidates},{fit_ms:.3f},{pred_ms:.3f},{model._order}", flush=True)


def parse_csv_ints(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-sizes", default="200,1000,5000")
    parser.add_argument("--arima-sizes", default="80,200,500")
    parser.add_argument("--max-p", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=2)
    args = parser.parse_args()

    with external_backend_guard():
        bench_native_stats(parse_csv_ints(args.stats_sizes))
        bench_auto_arima(parse_csv_ints(args.arima_sizes), args.max_p, args.max_q)


if __name__ == "__main__":
    main()
