import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
CHECKPOINTS = TUTORIALS / ".ipynb_checkpoints"


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def write_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = TUTORIALS / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    CHECKPOINTS.mkdir(exist_ok=True)
    checkpoint = CHECKPOINTS / filename.replace(".ipynb", "-checkpoint.ipynb")
    checkpoint.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


COMMON_RETAIL_FUNCTION = r'''
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

def make_retail_demand(n_days=240, n_stores=1, start="2023-01-01"):
    rows = []
    for i in range(n_stores):
        rng = np.random.default_rng(100 + i)
        dates = pd.date_range(start, periods=n_days, freq="D")
        dow = dates.dayofweek.to_numpy()
        month = dates.month.to_numpy()
        holiday = ((dow >= 5) | rng.binomial(1, 0.04, n_days).astype(bool)).astype(int)
        promotion = rng.binomial(1, 0.14 + 0.06 * (dow >= 4), n_days).astype(int)
        price_index = 1.0 + 0.04 * np.sin(np.linspace(0, 5 * np.pi, n_days)) + rng.normal(0, 0.015, n_days)
        temperature = 18 + 10 * np.sin(np.linspace(-0.8, 2.8 * np.pi, n_days)) + rng.normal(0, 1.8, n_days)
        stockout = rng.binomial(1, 0.025, n_days)
        baseline = 120 + 18 * i
        weekly = np.where(dow < 5, 8, 28)
        seasonal = 16 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 + i / 3)
        trend = 0.08 * np.arange(n_days)
        demand = (
            baseline + weekly + seasonal + trend
            + 34 * promotion + 22 * holiday
            + 0.9 * np.maximum(temperature - 20, 0)
            - 75 * (price_index - 1.0)
            - 45 * stockout
            + rng.normal(0, 7, n_days)
        )
        rows.append(pd.DataFrame({
            "date": dates,
            "store_id": f"store_{i + 1:02d}",
            "sales": np.maximum(demand, 1),
            "promotion": promotion,
            "holiday": holiday,
            "price_index": price_index,
            "temperature": temperature,
            "stockout": stockout,
            "month": month,
        }))
    return pd.concat(rows, ignore_index=True)
'''

NOTEBOOKS = {}

NOTEBOOKS["01_QuickStart_Guide.ipynb"] = [
    md('''
    # PipelineTS Quick Start: Retail Demand Forecasting
    # PipelineTS 快速开始：零售销量预测

    This tutorial uses a common industrial scenario: **daily store demand forecasting** with promotions, holidays, weather-like signals, and inventory stockout effects.

    本教程使用常见工业场景：带促销、节假日、天气类信号和缺货影响的**门店日销量预测**。

    You will cover:

    你将学习：

    - Current model names from `ModelPipeline.list_all_available_models()`
    - `ModelPipeline.fit()`, `predict()`, `predict_quantiles()`
    - `SmartRouter` fast automation
    - Forecast visualization
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    data = make_retail_demand(n_days=260, n_stores=1)
    data = data.drop(columns=["store_id"])
    train = data.iloc[:-14].reset_index(drop=True)
    valid = data.iloc[-14:].reset_index(drop=True)
    future_covariates = valid[["date", "promotion", "holiday"]].reset_index(drop=True)

    print(train.tail(3))
    print(valid.head(3))
    '''),
    code('''
    from PipelineTS.pipeline import ModelPipeline

    print("Available model names:")
    print(ModelPipeline.list_all_available_models())
    '''),
    code('''
    pipeline = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        cv=2,
        random_forest__n_estimators=120,
        extra_forest__n_estimators=120,
    )

    leaderboard = pipeline.fit(train, valid_data=valid)
    leaderboard
    '''),
    code('''
    forecast = pipeline.predict(n=14, future_covariates=future_covariates)
    forecast.head()
    '''),
    code('''
    quantiles = pipeline.predict_quantiles(
        n=14,
        levels=[0.5, 0.8, 0.9, 0.95],
        future_covariates=future_covariates,
    )
    quantiles.head()
    '''),
    code('''
    from PipelineTS.plot import plot_forecast, plot_leaderboard

    plot_forecast(train, forecast, time_col="date", target_col="sales", history_tail=90, lang="zh")
    plot_leaderboard(leaderboard, lang="zh")
    '''),
    code('''
    from PipelineTS.pipeline import SmartRouter

    router = SmartRouter(
        time_col="date",
        target_col="sales",
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        preset="fast",
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        time_limit=45,
    )
    router.fit(train, valid_data=valid)
    router.predict(14, future_covariates=future_covariates).head()
    '''),
]

NOTEBOOKS["02_All_Models_Guide.ipynb"] = [
    md('''
    # All Models Guide: Current Model Names
    # 全模型指南：当前模型名称

    PipelineTS model names used by `ModelPipeline` and `SmartRouter` are registry keys, not class names. This notebook documents the latest names and shows how to choose a practical candidate set for industrial workloads.

    PipelineTS 在 `ModelPipeline` 和 `SmartRouter` 中使用的是模型注册名，而不是类名。本教程展示最新名称，并说明如何为工业工作负载选择候选模型。
    '''),
    code('''
    from PipelineTS.pipeline import ModelPipeline

    available = ModelPipeline.list_all_available_models()
    print(f"{len(available)} models are available in this environment:")
    for name in available:
        print("-", name)
    '''),
    md('''
    ## Current core names / 当前核心名称

    | Family / 家族 | Model names / 模型名 | Typical use / 典型场景 |
    |---|---|---|
    | Statistical / 统计 | `auto_arima`, `prophet` | Strong seasonality, explainable baselines |
    | Native ML / 原生机器学习 | `catboost`, `xgboost`, `random_forest`, `extra_forest`, `gc_forest`, `wide_gbrt` | Retail demand, operations metrics, tabular covariates |
    | Sklearn wrappers / sklearn 封装 | `multi_output_model`, `multi_step_model`, `regressor_chain` | Fast baselines and robust production challengers |
    | NN light / 轻量 NN | `d_linear`, `n_linear`, `tide`, `tcn` | Fast neural baselines, trend/seasonality patterns |
    | NN medium/heavy / 中大型 NN | `n_beats`, `n_hits`, `tft`, `gau`, `stacking_rnn`, `time2vec`, `transformer`, `patch_rnn`, `deepar` | Complex patterns when enough history exists |
    | Multivariate NN / 多变量 NN | `itransformer`, `srs_net` | Multi-input or multi-output industrial sensor/load forecasting |
    | Optional foundation models / 可选基础模型 | `chronos_2`, `chronos_2_synth`, `chronos_2_small` | Zero-shot forecasting when `chronos-forecasting` is installed |
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    data = make_retail_demand(n_days=180, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-12].copy(), data.iloc[-12:].copy()
    safe_models = [m for m in ["random_forest", "extra_forest", "multi_output_model", "multi_step_model"] if m in available]
    safe_models
    '''),
    code('''
    from PipelineTS.pipeline import ModelPipeline

    pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=12,
        include_models=safe_models,
        quantile=None,
        cv=2,
        random_forest__n_estimators=80,
        extra_forest__n_estimators=80,
    )
    pipe.fit(train, valid_data=valid)
    pipe.leader_board_
    '''),
    md('''
    ## Direct model classes / 直接模型类

    For single-model experiments you can instantiate wrapper classes directly. In production tutorials we recommend registry names through `ModelPipeline`, because they work with `PipelineConfigs`, error resilience, logging, save/load, and `SmartRouter`.

    对于单模型实验，可以直接实例化包装类。生产教程更推荐通过 `ModelPipeline` 使用注册名，因为它能统一支持 `PipelineConfigs`、错误容忍、日志、保存/加载和 `SmartRouter`。
    '''),
    code('''
    from PipelineTS.ml_model import RandomForestModel

    model = RandomForestModel(
        time_col="date",
        target_col="sales",
        lags=12,
        quantile=0.9,
        n_estimators=120,
        random_state=42,
    )
    model.fit(train, cv=2)
    model.predict(12).head()
    '''),
    code('''
    nn_candidates = [m for m in ["d_linear", "n_linear", "tide", "n_hits", "itransformer", "srs_net"] if m in available]
    optional_candidates = [m for m in ["chronos_2", "chronos_2_synth", "chronos_2_small"] if m in available]

    print("NN candidates available:", nn_candidates)
    print("Optional foundation models available:", optional_candidates)
    '''),
]

NOTEBOOKS["03_Multivariate_Prediction.ipynb"] = [
    md('''
    # Multivariate Forecasting: Manufacturing Energy & Sensor Signals
    # 多变量预测：制造业能耗与传感器信号

    This notebook uses an industrial manufacturing scenario where power consumption is affected by production volume, machine temperature, vibration, and line speed.

    本教程使用制造业场景：产线能耗受产量、机器温度、振动和线速影响。

    Covered API:

    - `feature_cols` for multi-input forecasting
    - multi-output `target_col=[...]`
    - `itransformer` and `srs_net` model names
    - `ModelPipeline` with multivariate NN models
    '''),
    code('''
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(7)
    n = 240
    date = pd.date_range("2023-01-01", periods=n, freq="h")
    shift = ((date.hour >= 8) & (date.hour < 20)).astype(int)
    production = 80 + 25 * shift + 10 * np.sin(np.linspace(0, 8*np.pi, n)) + rng.normal(0, 4, n)
    line_speed = 1.0 + 0.15 * shift + rng.normal(0, 0.03, n)
    temperature = 45 + 0.08 * production + 5 * np.sin(np.linspace(0, 4*np.pi, n)) + rng.normal(0, 1.5, n)
    vibration = 0.4 + 0.006 * production + rng.normal(0, 0.05, n)
    power_kw = 120 + 1.7 * production + 16 * line_speed + 0.9 * temperature + rng.normal(0, 8, n)

    plant = pd.DataFrame({
        "date": date,
        "power_kw": power_kw,
        "production_units": production,
        "temperature_c": temperature,
        "vibration": vibration,
        "line_speed": line_speed,
        "shift": shift,
    })
    plant.head()
    '''),
    code('''
    from PipelineTS.pipeline import ModelPipeline

    feature_cols = ["power_kw", "production_units", "temperature_c", "vibration", "line_speed", "shift"]
    train, valid = plant.iloc[:-24].copy(), plant.iloc[-24:].copy()

    available = ModelPipeline.list_all_available_models()
    multivariate_models = [m for m in ["itransformer", "srs_net"] if m in available]
    multivariate_models
    '''),
    code('''
    if multivariate_models:
        selected_mv_models = multivariate_models[:1]
        mv_kwargs = {}
        if "itransformer" in selected_mv_models:
            mv_kwargs.update({"itransformer__epochs": 30, "itransformer__patience": 6})
        if "srs_net" in selected_mv_models:
            mv_kwargs.update({"srs_net__epochs": 30, "srs_net__patience": 6})
        pipe_miso = ModelPipeline(
            time_col="date",
            target_col="power_kw",
            feature_cols=feature_cols,
            lags=24,
            include_models=selected_mv_models,
            quantile=None,
            cv=2,
            **mv_kwargs,
        )
        lb_miso = pipe_miso.fit(train, valid_data=valid)
        display(lb_miso)
        display(pipe_miso.predict(24).head())
    else:
        print("No multivariate NN backend is available in this environment.")
    '''),
    code('''
    targets = ["power_kw", "temperature_c"]
    feature_cols_mimo = ["power_kw", "temperature_c", "production_units", "vibration", "line_speed", "shift"]

    if "itransformer" in available:
        pipe_mimo = ModelPipeline(
            time_col="date",
            target_col=targets,
            feature_cols=feature_cols_mimo,
            lags=24,
            include_models=["itransformer"],
            quantile=None,
            cv=2,
            itransformer__epochs=30,
            itransformer__patience=6,
        )
        pipe_mimo.fit(train, valid_data=valid)
        pipe_mimo.predict(12).head()
    '''),
    md('''
    ## Practical guidance / 实践建议

    - Use `feature_cols` when drivers are observed historically and should be used as model inputs.
    - Use `known_covariates` when future values are known at prediction time, such as calendar, planned promotions, planned shifts, or prices.
    - Use `target_col=[...]` only when you need simultaneous forecasting of multiple business KPIs.
    '''),
]

NOTEBOOKS["04_Advanced_Pipeline.ipynb"] = [
    md('''
    # Advanced ModelPipeline: Configs, Per-Model Settings, Save/Load
    # 高级 ModelPipeline：配置、单模型设置、保存/加载

    Scenario: demand planning teams often compare conservative baselines, high-variance tree models, and business-specific lag choices before promoting a champion model.

    场景：需求计划团队通常需要比较稳健基线、高方差树模型和业务特定滞后窗口，然后再上线冠军模型。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

    data = make_retail_demand(n_days=220, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-21].copy(), data.iloc[-21:].copy()
    '''),
    code('''
    configs = PipelineConfigs([
        ("random_forest", "rf_short_lag", {
            "init_configs": {"n_estimators": 120, "max_depth": 10, "random_state": 42},
            "fit_configs": {},
            "predict_configs": {},
            "pipeline_configs": {"lags": 7, "scaler": None},
        }),
        ("extra_forest", "extra_standard_scaled", {
            "init_configs": {"n_estimators": 160, "max_depth": 12, "random_state": 42},
            "fit_configs": {},
            "predict_configs": {},
            "pipeline_configs": {"lags": 21, "scaler": StandardScaler()},
        }),
        ("multi_output_model", "multi_output_diff", {
            "init_configs": {},
            "fit_configs": {},
            "predict_configs": {},
            "pipeline_configs": {"differential_n": 1},
        }),
    ])
    '''),
    code('''
    pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        configs=configs,
        include_init_config_model=False,
        quantile=0.9,
        cv=2,
        time_limit=90,
    )
    leaderboard = pipe.fit(train, valid_data=valid)
    leaderboard
    '''),
    code('''
    best_name = pipe.leader_board_.iloc[0]["model"]
    best_model = pipe.get_model(best_name)
    best_configs = pipe.get_model_all_configs(best_name)

    print("Best model:", best_name)
    print("Config keys:", sorted(best_configs.keys()))
    print("Failed models:", pipe.failed_models)
    print("Skipped models:", pipe.skipped_models)
    '''),
    code('''
    pred_best = pipe.predict(21)
    pred_named = pipe.predict(21, model_name=best_name)
    q_pred = pipe.predict_quantiles(21, levels=[0.5, 0.8, 0.9])

    display(pred_best.head())
    display(q_pred.head())
    '''),
    code('''
    model_path = Path("../tmp_retail_pipeline.pts")
    pipe.save(model_path, metadata={"scenario": "retail_demand", "owner": "planning"})

    loaded = ModelPipeline.load(model_path)
    loaded.predict(7).head()
    '''),
    code('''
    pipe.plot(n=21, history_tail=90, lang="zh")
    pipe.plot_leaderboard(lang="zh")
    '''),
]

NOTEBOOKS["05_Preprocessing_and_Data.ipynb"] = [
    md('''
    # Preprocessing & Data Diagnostics: Messy POS Data
    # 预处理与数据诊断：杂乱 POS 数据

    Industrial time series are rarely clean. This tutorial creates duplicate timestamps, missing dates, outliers, and stockout artifacts, then uses PipelineTS preprocessing and diagnostic APIs.

    工业时间序列很少天然干净。本教程构造重复时间戳、缺失日期、异常值和缺货影响，然后使用 PipelineTS 预处理与诊断 API。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    raw = make_retail_demand(n_days=120, n_stores=1).drop(columns=["store_id"])
    messy = raw.copy()
    messy = pd.concat([messy, messy.iloc[[20, 21]]], ignore_index=True)
    messy.loc[10:12, "sales"] = np.nan
    messy.loc[45, "sales"] *= 4
    messy = messy.drop(index=[30, 31, 32]).sample(frac=1, random_state=42).reset_index(drop=True)
    messy.head()
    '''),
    code('''
    from PipelineTS.preprocessing import TimeSeriesDataQualityReport

    quality = TimeSeriesDataQualityReport(time_col="date", target_col="sales")
    report = quality.fit(messy)
    report["overview"], report["issues"][:5]
    '''),
    code('''
    from PipelineTS.preprocessing import sort_and_deduplicate, resample_time_series, clip_or_winsorize, smooth_series

    clean = sort_and_deduplicate(messy, time_col="date", agg="mean")
    clean = resample_time_series(clean, time_col="date", target_col="sales", freq="D", fill_method="linear")
    clean = clip_or_winsorize(clean, target_col="sales", lower_q=0.01, upper_q=0.99)
    clean = smooth_series(clean, target_col="sales", method="rolling_mean", window=3)
    clean.head()
    '''),
    code('''
    from PipelineTS.preprocessing import TimeSeriesPreprocessor

    prep = TimeSeriesPreprocessor()
    transformed = prep.transform_target(clean, target_col="sales", method="log1p")
    differenced = prep.difference_series(clean, target_col="sales", order=1)

    display(transformed.head())
    display(differenced.head())
    '''),
    code('''
    from PipelineTS.preprocessing import (
        time_index_report,
        series_profile,
        forecastability_report,
        baseline_forecast_report,
        leakage_risk_report,
        modeling_readiness_report,
    )

    print(time_index_report(clean, time_col="date"))
    print(series_profile(clean, target_col="sales"))
    print(forecastability_report(clean, target_col="sales", horizon=14))
    print(baseline_forecast_report(clean, time_col="date", target_col="sales", horizon=14))
    print(leakage_risk_report(clean, time_col="date", target_col="sales", known_covariates=["promotion", "holiday"]))
    print(modeling_readiness_report(clean, time_col="date", target_col="sales"))
    '''),
    code('''
    from PipelineTS.plot import plot_series, plot_decomposition, plot_acf_pacf

    plot_series(clean, time_col="date", target_col="sales", title="Cleaned POS demand", lang="zh")
    plot_decomposition(clean, time_col="date", target_col="sales", lang="zh")
    plot_acf_pacf(clean["sales"].dropna().values, max_lags=30, lang="zh")
    '''),
]

NOTEBOOKS["06_Hyperparameter_Tuning.ipynb"] = [
    md('''
    # Hyperparameter Tuning: Practical Search Budgets
    # 超参数调优：实用搜索预算

    Scenario: a forecasting platform must tune within strict time budgets. PipelineTS supports explicit `PipelineConfigs`, double-underscore model kwargs, and SmartRouter HPO strategies.

    场景：预测平台需要在严格时间预算内调参。PipelineTS 支持显式 `PipelineConfigs`、双下划线模型参数和 SmartRouter HPO 策略。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    from PipelineTS.pipeline import ModelPipeline, PipelineConfigs, SmartRouter

    data = make_retail_demand(n_days=220, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-21].copy(), data.iloc[-21:].copy()
    '''),
    code('''
    pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        include_models=["random_forest", "extra_forest"],
        quantile=None,
        cv=2,
        random_forest__n_estimators=100,
        random_forest__max_depth=8,
        extra_forest__n_estimators=160,
        extra_forest__max_depth=12,
    )
    pipe.fit(train, valid_data=valid)
    pipe.leader_board_
    '''),
    code('''
    configs = PipelineConfigs([
        ("random_forest", "rf_80trees_lag7", {
            "init_configs": {"n_estimators": 80, "max_depth": 8, "random_state": 42},
            "pipeline_configs": {"lags": 7, "scaler": None},
        }),
        ("random_forest", "rf_160trees_lag21", {
            "init_configs": {"n_estimators": 160, "max_depth": 12, "random_state": 42},
            "pipeline_configs": {"lags": 21, "scaler": True},
        }),
    ])

    tuned_pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        include_models=["random_forest"],
        configs=configs,
        quantile=None,
        cv=2,
    )
    tuned_pipe.fit(train, valid_data=valid)
    tuned_pipe.leader_board_
    '''),
    code('''
    router = SmartRouter(
        time_col="date",
        target_col="sales",
        preset="medium_quality",
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        search_strategy="auto",
        hpo_strategy="auto",
        hpo_n_trials=5,
        hpo_timeout_per_model=20,
        time_limit=120,
    )
    router.fit(train, valid_data=valid)

    print("Active HPO strategy:", router._active_hpo_strategy_)
    print("HPO results:", router._hpo_results)
    router.leader_board_
    '''),
    code('''
    print("Strategy keys:", router.strategy_.keys())
    print("Autonomy summary:")
    router.autonomy_summary_
    '''),
]

NOTEBOOKS["07_Benchmarks.ipynb"] = [
    md('''
    # Benchmarking Forecast Pipelines: Champion-Challenger Evaluation
    # 预测管道基准测试：冠军-挑战者评估

    Scenario: before deployment, a team compares baseline and candidate pipelines on a holdout window using business-facing metrics.

    场景：上线前，团队在留出窗口上使用业务指标比较基线和候选管道。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    import time
    import numpy as np
    from PipelineTS.pipeline import ModelPipeline
    from PipelineTS.evaluation import ModelComparison
    from PipelineTS.metrics import mape, smape, picp, pinaw, winkler_score

    data = make_retail_demand(n_days=240, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-28].copy(), data.iloc[-28:].copy()
    horizon = len(valid)
    '''),
    code('''
    candidates = {
        "baseline_fast": ["multi_output_model", "multi_step_model"],
        "forest_ensemble": ["random_forest", "extra_forest"],
        "mixed_light": ["random_forest", "multi_output_model", "regressor_chain"],
    }

    results = {}
    timings = {}
    for label, models in candidates.items():
        t0 = time.time()
        pipe = ModelPipeline(
            time_col="date",
            target_col="sales",
            lags=14,
            include_models=models,
            quantile=0.9,
            cv=2,
            random_forest__n_estimators=80,
            extra_forest__n_estimators=80,
        )
        pipe.fit(train, valid_data=valid)
        pred = pipe.predict(horizon)
        results[label] = pred
        timings[label] = time.time() - t0

    timings
    '''),
    code('''
    comp = ModelComparison(time_col="date", target_col="sales")
    y_true = valid["sales"].values

    for name, pred in results.items():
        comp.add_result(
            name,
            y_true,
            pred["sales"].values[:horizon],
            lower=pred.get("sales_lower", None),
            upper=pred.get("sales_upper", None),
        )

    table = comp.fit(
        metrics={"MAPE": mape, "sMAPE": smape},
        interval_metrics={"PICP": picp, "PINAW": pinaw, "Winkler90": lambda y, lo, hi: winkler_score(y, lo, hi, alpha=0.1)},
    )
    table["fit_predict_seconds"] = table["model"].map(timings)
    table.sort_values("MAPE")
    '''),
    code('''
    comp.rank("MAPE")
    '''),
    code('''
    comp.plot_bar(metric_cols=["MAPE", "sMAPE", "fit_predict_seconds"])
    comp.plot_predictions(time_index=valid["date"].values)
    '''),
]

NOTEBOOKS["08_Visualization.ipynb"] = [
    md('''
    # Visualization: Forecast, Leaderboard, Diagnostics
    # 可视化：预测、排行榜与诊断

    Scenario: analysts need reusable visual checks for demand forecasts before sending results to planning systems.

    场景：分析师需要在预测结果进入计划系统前进行可复用的可视化检查。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    from PipelineTS.pipeline import ModelPipeline
    from PipelineTS.plot import (
        TSPlotter,
        plot_series,
        plot_forecast,
        plot_leaderboard,
        plot_leaderboard_detail,
        plot_model_comparison,
        plot_residuals,
        plot_acf_pacf,
        plot_decomposition,
        plot_train_test_split,
    )

    data = make_retail_demand(n_days=220, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-21].copy(), data.iloc[-21:].copy()
    '''),
    code('''
    plot_series(data, time_col="date", target_col="sales", title="Retail demand history", lang="zh")
    plot_train_test_split(train, valid, time_col="date", target_col="sales", lang="zh")
    plot_decomposition(train, time_col="date", target_col="sales", lang="zh")
    plot_acf_pacf(train["sales"].values, max_lags=30, lang="zh")
    '''),
    code('''
    pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        cv=2,
        random_forest__n_estimators=80,
        extra_forest__n_estimators=80,
    )
    leaderboard = pipe.fit(train, valid_data=valid)
    pred = pipe.predict(21)
    '''),
    code('''
    plot_forecast(train, pred, time_col="date", target_col="sales", history_tail=90, lang="zh")
    plot_leaderboard(leaderboard, lang="zh")
    plot_leaderboard_detail(leaderboard, lang="zh")
    '''),
    code('''
    predictions = {
        name: pipe.predict(21, model_name=name)
        for name in pipe.leader_board_["model"].head(3)
    }
    plot_model_comparison(train, predictions, time_col="date", target_col="sales", history_tail=90, lang="zh")
    '''),
    code('''
    y_true = valid["sales"].values[:len(pred)]
    y_pred = pred["sales"].values[:len(y_true)]
    plot_residuals(y_true, y_pred, time_index=valid["date"].values[:len(y_true)], lang="zh")
    '''),
    code('''
    plotter = TSPlotter(time_col="date", target_col="sales", lang="zh")
    plotter.plot_series(data)
    plotter.plot_forecast(train, pred, history_tail=60)
    plotter.plot_leaderboard(pipe.leader_board_)
    '''),
]

NOTEBOOKS["09_Multi_Quantile_Intervals.ipynb"] = [
    md('''
    # Multi-Quantile Intervals: Inventory Service Levels
    # 多分位区间：库存服务水平

    Scenario: inventory teams need multiple uncertainty bands to choose safety stock for 50%, 80%, 90%, and 95% service levels.

    场景：库存团队需要多个不确定性区间，为 50%、80%、90%、95% 服务水平选择安全库存。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    from PipelineTS.pipeline import ModelPipeline, SmartRouter
    from PipelineTS.metrics import picp, pinaw, winkler_score

    data = make_retail_demand(n_days=240, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-28].copy(), data.iloc[-28:].copy()
    horizon = 28
    '''),
    code('''
    pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        cv=3,
        random_forest__n_estimators=100,
        extra_forest__n_estimators=100,
    )
    pipe.fit(train, valid_data=valid)

    point_and_90 = pipe.predict(horizon)
    multi_q = pipe.predict_quantiles(horizon, levels=[0.5, 0.8, 0.9, 0.95])
    multi_q.head()
    '''),
    code('''
    y_true = valid["sales"].values[:horizon]
    rows = []
    for level in [0.5, 0.8, 0.9, 0.95]:
        label = f"{level:.2f}".rstrip("0").rstrip(".")
        lo = multi_q[f"sales_q{label}_lower"].values[:horizon]
        hi = multi_q[f"sales_q{label}_upper"].values[:horizon]
        rows.append({
            "coverage_level": level,
            "PICP": picp(y_true, lo, hi),
            "PINAW": pinaw(y_true, lo, hi),
            "Winkler": winkler_score(y_true, lo, hi, alpha=1-level),
        })
    pd.DataFrame(rows)
    '''),
    code('''
    router = SmartRouter(
        time_col="date",
        target_col="sales",
        preset="fast",
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        cv=3,
        time_limit=60,
    )
    router.fit(train, valid_data=valid)
    router.predict_quantiles(horizon, levels=[0.8, 0.9, 0.95]).head()
    '''),
    code('''
    from PipelineTS.plot import plot_forecast

    plot_forecast(train, point_and_90, time_col="date", target_col="sales", history_tail=90, lang="zh")
    '''),
]

NOTEBOOKS["10_Multi_Series_Covariates.ipynb"] = [
    md('''
    # Multi-Series + Covariates: Store-Level Demand Planning
    # 多序列 + 协变量：门店级需求计划

    This is a common production pattern: many related stores/SKUs, future-known promotions and holidays, historical-only weather/stockout signals, and per-series forecasts.

    这是常见生产模式：多门店/SKU，未来已知促销和节假日，历史天气/缺货信号，以及每序列预测。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    panel = make_retail_demand(n_days=220, n_stores=4)
    train = panel.groupby("store_id", group_keys=False).head(200).reset_index(drop=True)
    valid = panel.groupby("store_id", group_keys=False).tail(20).reset_index(drop=True)
    future_covariates = valid[["date", "store_id", "promotion", "holiday"]].reset_index(drop=True)

    print(panel.groupby("store_id").size())
    panel.head()
    '''),
    code('''
    from PipelineTS.plot import plot_series

    plot_series(panel, time_col="date", target_col="sales", id_col="store_id", title="Store-level demand", lang="zh")
    '''),
    code('''
    from PipelineTS.pipeline import ModelPipeline

    panel_pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        id_col="store_id",
        lags=14,
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        include_models=["random_forest", "extra_forest"],
        quantile=0.9,
        cv=2,
        random_forest__n_estimators=100,
        extra_forest__n_estimators=100,
    )
    panel_lb = panel_pipe.fit(train, valid_data=valid)
    panel_lb
    '''),
    code('''
    panel_pred = panel_pipe.predict(20, future_covariates=future_covariates)
    print(panel_pred.groupby("store_id").size())
    panel_pred.head(10)
    '''),
    code('''
    panel_q = panel_pipe.predict_quantiles(20, levels=[0.8, 0.9], future_covariates=future_covariates)
    panel_q.head()
    '''),
    code('''
    from PipelineTS.pipeline import SmartRouter

    router_panel = SmartRouter(
        time_col="date",
        target_col="sales",
        id_col="store_id",
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        preset="fast",
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        time_limit=90,
    )
    router_panel.fit(train, valid_data=valid)
    router_panel.predict(20, future_covariates=future_covariates).head()
    '''),
    code('''
    print("Profile series count:", router_panel.profile_.n_series)
    print("Data insights:")
    router_panel.insights_.summary()
    '''),
]

NOTEBOOKS["11_Incremental_Learning.ipynb"] = [
    md('''
    # Incremental Updates: Rolling Operations Forecast
    # 增量更新：滚动运营预测

    Scenario: a call-center or operations dashboard receives new daily observations and refreshes forecasts without rebuilding the whole workflow manually.

    场景：呼叫中心或运营看板每天收到新观测值，需要刷新预测，而不是手工重建整个流程。
    '''),
    code('''
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(2024)
    n = 260
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    campaign = rng.binomial(1, 0.10, n)
    incidents = rng.poisson(0.15, n)
    tickets = 900 + 130 * (dow < 5) + 220 * campaign + 80 * incidents + 40 * np.sin(np.linspace(0, 8*np.pi, n)) + rng.normal(0, 45, n)
    ops = pd.DataFrame({"date": dates, "tickets": np.maximum(tickets, 1), "campaign": campaign, "incidents": incidents})

    initial = ops.iloc[:200].copy()
    new_batch = ops.iloc[200:230].copy()
    holdout = ops.iloc[230:].copy()
    '''),
    code('''
    from PipelineTS.pipeline import ModelPipeline

    pipe = ModelPipeline(
        time_col="date",
        target_col="tickets",
        lags=14,
        known_covariates=["campaign"],
        past_covariates=["incidents"],
        include_models=["random_forest", "multi_output_model"],
        quantile=0.9,
        cv=2,
        random_forest__n_estimators=120,
    )
    pipe.fit(initial)
    before_update = pipe.predict(14, future_covariates=holdout[["date", "campaign"]].head(14))
    before_update.head()
    '''),
    code('''
    pipe.update(new_batch, refit_all=False)
    after_update = pipe.predict(14, future_covariates=holdout[["date", "campaign"]].head(14))
    after_update.head()
    '''),
    code('''
    from PipelineTS.pipeline import SmartRouter

    router = SmartRouter(
        time_col="date",
        target_col="tickets",
        known_covariates=["campaign"],
        past_covariates=["incidents"],
        preset="fast",
        include_models=["random_forest", "multi_output_model", "extra_forest"],
        quantile=0.9,
        time_limit=60,
    )
    router.fit(initial)
    router.update(new_batch, refit_all=False)
    router.predict(14, future_covariates=holdout[["date", "campaign"]].head(14)).head()
    '''),
    code('''
    from pathlib import Path

    path = Path("../tmp_ops_router.pts")
    router.save(path, metadata={"scenario": "operations_tickets", "refresh": "daily"})
    loaded_router = SmartRouter.load(path)
    loaded_router.predict(7, future_covariates=holdout[["date", "campaign"]].head(7)).head()
    '''),
    code('''
    router.plot(n=14, history_tail=80, lang="zh")
    router.plot_leaderboard(lang="zh")
    '''),
]

NOTEBOOKS["12_SmartRouter_and_Pipeline.ipynb"] = [
    md('''
    # SmartRouter & ModelPipeline: Production API Coverage
    # SmartRouter 与 ModelPipeline：生产 API 覆盖

    This notebook is the API coverage map for common industrial forecasting workflows: model discovery, manual pipelines, automated routing, search/HPO, ensembles, uncertainty, covariates, panel data, visualization, updates, and persistence.

    本教程覆盖常见工业预测流程的 API：模型发现、手动管道、自动路由、搜索/HPO、集成、不确定性、协变量、面板数据、可视化、更新和持久化。
    '''),
    code(COMMON_RETAIL_FUNCTION),
    code('''
    from pathlib import Path
    from PipelineTS.pipeline import ModelPipeline, PipelineConfigs, SmartRouter

    data = make_retail_demand(n_days=240, n_stores=1).drop(columns=["store_id"])
    train, valid = data.iloc[:-21].copy(), data.iloc[-21:].copy()
    future_covariates = valid[["date", "promotion", "holiday"]].copy()

    print(ModelPipeline.list_all_available_models())
    print(SmartRouter.list_all_available_models())
    '''),
    code('''
    pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        include_models="light",
        exclude_models=None,
        quantile=0.9,
        cv=2,
        time_limit=90,
    )
    pipe.fit(train, valid_data=valid)
    pipe.leader_board_.head()
    '''),
    code('''
    print("Best model configs:")
    print(pipe.get_model_all_configs())
    print("Failures:", pipe.failed_models)
    print("Skipped:", pipe.skipped_models)

    pipe_pred = pipe.predict(21, future_covariates=future_covariates)
    pipe_q = pipe.predict_quantiles(21, levels=[0.5, 0.8, 0.9], future_covariates=future_covariates)
    display(pipe_pred.head())
    display(pipe_q.head())
    '''),
    code('''
    configs = PipelineConfigs([
        ("random_forest", "rf_business_lag", {
            "init_configs": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "pipeline_configs": {"lags": 7, "scaler": None},
        }),
        ("extra_forest", "extra_long_lag", {
            "init_configs": {"n_estimators": 140, "max_depth": 12, "random_state": 42},
            "pipeline_configs": {"lags": 21, "scaler": True},
        }),
    ])

    configured_pipe = ModelPipeline(
        time_col="date",
        target_col="sales",
        lags=14,
        include_models=["random_forest", "extra_forest"],
        configs=configs,
        quantile=0.9,
        cv=2,
    )
    configured_pipe.fit(train, valid_data=valid)
    configured_pipe.leader_board_
    '''),
    code('''
    router = SmartRouter(
        time_col="date",
        target_col="sales",
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        preset="medium_quality",
        include_models=["random_forest", "extra_forest", "multi_output_model", "multi_step_model"],
        max_models=4,
        cv=2,
        search_strategy="auto",
        ensemble_strategy="auto",
        ensemble_top_k=3,
        hpo_strategy="auto",
        hpo_n_trials=5,
        hpo_timeout_per_model=20,
        quantile=0.9,
        time_limit=120,
    )
    router.fit(train, valid_data=valid)
    router.leader_board_
    '''),
    code('''
    print("Selected strategy:")
    print(router.strategy_)
    print("Data profile:")
    print(router.profile_)
    print("Data insights:")
    print(router.insights_)
    print("Autonomy summary:")
    router.autonomy_summary_
    '''),
    code('''
    router_ensemble = router.predict(21)
    router_forecast = router.predict(21, use_ensemble=False, future_covariates=future_covariates)
    router_q = router.predict_quantiles(21, levels=[0.8, 0.9, 0.95], future_covariates=future_covariates)

    display(router_ensemble.head())
    display(router_forecast.head())
    display(router_q.head())
    '''),
    code('''
    router.plot(n=21, history_tail=90, lang="zh")
    router.plot_leaderboard(lang="zh")
    '''),
    code('''
    path = Path("../tmp_router_prod.pts")
    router.save(path, metadata={"scenario": "retail_production", "service": "forecasting"})
    restored = SmartRouter.load(path)
    restored.predict(7, future_covariates=future_covariates.head(7)).head()
    '''),
    code('''
    new_observations = valid.copy()
    router.update(new_observations, refit_all=False)
    router.predict(7, future_covariates=future_covariates.head(7)).head()
    '''),
    code('''
    panel = make_retail_demand(n_days=160, n_stores=3)
    panel_train = panel.groupby("store_id", group_keys=False).head(145).reset_index(drop=True)
    panel_valid = panel.groupby("store_id", group_keys=False).tail(15).reset_index(drop=True)
    panel_future = panel_valid[["date", "store_id", "promotion", "holiday"]]

    panel_router = SmartRouter(
        time_col="date",
        target_col="sales",
        id_col="store_id",
        known_covariates=["promotion", "holiday"],
        past_covariates=["temperature", "price_index", "stockout"],
        preset="fast",
        include_models=["random_forest", "extra_forest", "multi_output_model"],
        quantile=0.9,
        time_limit=90,
    )
    panel_router.fit(panel_train, valid_data=panel_valid)
    panel_router.predict(15, future_covariates=panel_future).head()
    '''),
]

expected_checkpoints = {
    filename.replace(".ipynb", "-checkpoint.ipynb")
    for filename in NOTEBOOKS
}

for stale in CHECKPOINTS.glob("*.ipynb"):
    if stale.name not in expected_checkpoints:
        stale.unlink()

for filename, cells in NOTEBOOKS.items():
    write_notebook(filename, cells)
    print(f"updated {filename} ({len(cells)} cells)")
