# Quick Start Guide
# 快速入门指南

This guide walks through PipelineTS in order of increasing control, from a single function call to manual pipeline configuration.
本指南按控制粒度从低到高依次介绍 PipelineTS，从单次函数调用到手动管道配置。

---

## Level 0 — One-line forecast (zero config) / 一行预测（零配置）

The fastest way to get a prediction. Column names are inferred automatically.
获取预测结果的最快方式。列名自动推断，无需任何配置。

```python
from PipelineTS import forecast

pred = forecast("sales.csv", n=12)
```

PipelineTS will automatically / PipelineTS 将自动完成：
1. Detect the time and target columns / 检测时间列与目标列
2. Auto-clean the data (deduplicate, fill gaps, clip outliers) / 自动清洗数据（去重、填补缺口、裁剪异常值）
3. Run AutoML model selection (`preset="fast"` by default) / 运行 AutoML 模型选择（默认 `preset="fast"`）
4. Return a DataFrame of the next 12 predictions / 返回未来 12 步的预测 DataFrame

```python
# With prediction intervals and better quality / 带预测区间与更高质量
pred = forecast("sales.csv", n=12, quantile=0.9, preset="high_quality")

# Accept a DataFrame directly / 直接传入 DataFrame
pred = forecast(df, n=12)

# Keep the trained model for later use / 保留训练好的模型
pred, model = forecast(df, n=12, return_model=True)
model.save("forecaster.pts")
loaded = type(model).load("forecaster.pts")
```

---

## Level 0.5 — Diagnose before you forecast / 预测前先诊断

Before running a forecast, use `diagnose()` to understand your data.
在运行预测之前，使用 `diagnose()` 了解数据状况。

```python
from PipelineTS import diagnose

result = diagnose("sales.csv", horizon=12)
print(result["status"])       # 'READY' / 'WARNING' / 'NOT_READY'
print(result["reports"]["forecastability"])
print(result["next_step"])    # Ready-to-run forecast() call / 可直接运行的 forecast() 调用
```

Use `full=True` for extended reports (seasonality, trend, leakage risk).
使用 `full=True` 获取扩展报告（季节性、趋势、泄露风险）。

```python
result = diagnose(df, horizon=12, full=True)
print(result["reports"]["seasonality"])
print(result["reports"]["trend"])
```

---

## Level 1 — AutoForecast (sklearn-style, reusable) / AutoForecast（sklearn 风格，可复用）

`AutoForecast` is a reusable object that wraps the full AutoML pipeline.
Use it when you need to `fit()` once and `predict()` multiple times, or want access to the leaderboard and strategy.
`AutoForecast` 是封装了完整 AutoML 管道的可复用对象。
适用于需要 `fit()` 一次、多次 `predict()` 的场景，或需要查看排行榜和策略详情时。

```python
from PipelineTS import AutoForecast

model = AutoForecast(
    horizon=12,
    preset="medium_quality",  # 'fast' | 'medium_quality' | 'high_quality' | 'best_quality'
    quantile=0.9,             # 90% prediction intervals / 90% 预测区间
    time_limit=120,           # 2-minute training budget / 2 分钟训练预算
)
model.fit(train_df)

pred = model.predict()        # Predict next 12 steps / 预测未来 12 步
pred = model.predict(n=24)    # Override horizon / 覆盖预测步数

# Inspect results / 查看结果
print(model.leader_board_)
print(model.strategy_)
```

**With covariates / 带协变量:**

```python
model = AutoForecast(
    horizon=14,
    known_covariates=["promotion", "holiday"],   # Values known at prediction time / 预测时已知的列
    past_covariates=["temperature"],              # Historical context only / 仅提供历史上下文
)
model.fit(train_df)
pred = model.predict(future_covariates=future_df)  # future_df must have n rows / 须含 n 行
```

**For panel (multi-series) data / 面板（多序列）数据:**

```python
model = AutoForecast(horizon=12, id_col="store_id")
model.fit(panel_df)   # panel_df has columns: date, store_id, sales, ... / 含 date, store_id, sales 等列
pred = model.predict()
```

**Save and load / 保存与加载:**

```python
model.save("forecaster.pts")
loaded = AutoForecast.load("forecaster.pts")
loaded.predict()
```

---

## Level 2 — ModelPipeline (full control) / ModelPipeline（完全控制）

`ModelPipeline` gives explicit control over model selection, validation, and configuration.
`ModelPipeline` 对模型选择、验证和配置提供完全的显式控制。

```python
from PipelineTS.pipeline import ModelPipeline

print(ModelPipeline.list_all_available_models())  # see all model names / 查看所有模型名称

pipeline = ModelPipeline(
    time_col="date",
    target_col="sales",
    lags=14,
    quantile=0.9,
    include_models=["catboost", "d_linear", "auto_arima"],  # or 'light' | 'ml' | 'nn' | 'all'
    cv=3,
    random_state=42,
)

leaderboard = pipeline.fit(train_df, valid_data=val_df)
print(leaderboard)

pred = pipeline.predict(n=14)
pred = pipeline.predict(n=14, model_name="catboost_0")   # specific model / 指定模型

quantiles = pipeline.predict_quantiles(n=14, levels=[0.5, 0.8, 0.95])
```

**Visualize / 可视化:**

```python
pipeline.plot(n=14, history_tail=90, lang="zh")
pipeline.plot_leaderboard(lang="zh")
```

**Per-model configuration / 逐模型配置:**

```python
from PipelineTS.pipeline import PipelineConfigs

configs = PipelineConfigs([
    ("d_linear", "d_linear_12", {"pipeline_configs": {"lags": 12}}),
    ("d_linear", "d_linear_24", {"pipeline_configs": {"lags": 24}}),
    ("catboost", "cat_deep",    {"init_configs": {"iterations": 300},
                                 "pipeline_configs": {"differential_n": 1}}),
])
pipeline = ModelPipeline(time_col="date", target_col="sales", lags=14, configs=configs)
pipeline.fit(train_df)
```

**Save and load / 保存与加载:**

```python
pipeline.save("pipeline.pts")
loaded = ModelPipeline.load("pipeline.pts")
```

---

## Level 3 — SmartRouter (intelligent AutoML) / SmartRouter（智能 AutoML）

`SmartRouter` profiles data, selects models, runs optional screening/HPO, and builds an ensemble.
`SmartRouter` 对数据画像、选择模型、运行可选筛选/HPO 并构建集成。

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col="date",
    target_col="sales",
    n_predict=12,
    preset="medium_quality",   # 'fast' | 'medium_quality' | 'high_quality' | 'best_quality'
    ensemble_strategy="auto",  # 'auto' | 'weighted_avg' | 'none'
    verbose=True,
)
router.fit(train_df)

pred = router.predict(12)
pred = router.predict_quantiles(12, levels=[0.8, 0.95])

# Inspect what the router decided / 查看路由器的决策
print(router.strategy)
print(router.leader_board_)
print(router.profile_)         # Data characteristics / 数据特征画像

router.update(new_df)          # Incremental update / 增量更新
router.plot(n=12, lang="zh")
router.plot_leaderboard()
```

---

## Level 4 — Walk-forward backtesting / 前向回测

Evaluate before deploying.
在部署前进行评估。

```python
from PipelineTS import backtest

result = backtest("sales.csv", n=12, n_splits=5, metric="smape")
print(result["summary"])
# {'mean': 0.082, 'std': 0.011, 'min': 0.071, 'max': 0.097}

# Sliding window, custom metric / 滑动窗口，自定义指标
import numpy as np
result = backtest(
    df, n=12,
    metric=lambda y, yhat: np.median(np.abs(y - yhat)),
    mode="sliding", train_size=200,
)
```

---

## Next Steps / 下一步

- **Complete API / 完整 API** → [api_reference.md](api_reference.md)
- **All models / 所有模型** → [models.md](models.md)
- **Preprocessing / 数据预处理** → [preprocessing.md](preprocessing.md)
- **Multivariate & covariates / 多变量与协变量** → [multivariate.md](multivariate.md)
- **Advanced pipeline tricks / 高级管道用法** → [advanced.md](advanced.md)
- **Evaluation / 评估** → [evaluation.md](evaluation.md)
- **Interactive tutorials / 交互式教程** → `tutorials/00_EasyAPI.ipynb` … `tutorials/12_SmartRouter_and_Pipeline.ipynb`
