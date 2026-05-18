# API Reference
# API 参考

Complete reference for all public classes and functions in PipelineTS.
PipelineTS 所有公共类和函数的完整参考。

---

## Zero-Friction API (`PipelineTS`) / 零配置 API

The top-level package exposes a set of simple, one-call functions that handle
column inference, preprocessing, AutoML routing, and evaluation automatically.
Import them directly from `PipelineTS`.
顶层包提供了一组简单的单次调用函数，自动处理列推断、预处理、AutoML 路由和评估。
直接从 `PipelineTS` 导入即可使用。

```python
from PipelineTS import (
    load_data,
    infer_time_col, infer_target_col, infer_id_col,
    preprocess,
    diagnose,
    AutoForecast,
    forecast,
    backtest,
)
```

---

### `load_data`

```python
load_data(
    data: pd.DataFrame | str | Path,
    **read_kwargs,
) -> pd.DataFrame
```

Load time series data from a file path or return a DataFrame as-is.
Supports `.csv`, `.tsv`, `.xlsx`/`.xls`, `.parquet`, `.json`.
Files without an extension are treated as CSV.
从文件路径加载时间序列数据，或原样返回 DataFrame。
支持 `.csv`、`.tsv`、`.xlsx`/`.xls`、`.parquet`、`.json`。无扩展名文件视为 CSV。

```python
df = load_data("sales.csv")
df = load_data("/data/electric.parquet")
df = load_data(existing_dataframe)   # returned as-is / 原样返回
```

---

### `infer_time_col`

```python
infer_time_col(
    data: pd.DataFrame,
    time_col: str | None = None,
) -> str
```

Infer or validate the time column. When `time_col=None`, searches by:
推断或验证时间列。`time_col=None` 时按以下顺序搜索：
1. Common time-like column names (`date`, `ds`, `timestamp`, `datetime`, …) / 常见时间列名
2. Columns with `datetime64` dtype / `datetime64` 类型列
3. Columns whose values parse as dates with ≥ 80 % success / 超过 80% 值可解析为日期的列

```python
col = infer_time_col(df)              # auto-detect / 自动检测
col = infer_time_col(df, "ts")        # validate explicit name / 验证指定列名
```

---

### `infer_target_col`

```python
infer_target_col(
    data: pd.DataFrame,
    target_col: str | None = None,
    time_col: str | None = None,
    id_col: str | None = None,
    exclude: list | str | None = None,
) -> str
```

Infer or validate the forecast target column.
Prefers columns named `y`, `value`, `sales`, `demand`, `revenue`, etc.
Falls back to the last numeric column when multiple candidates exist.
推断或验证预测目标列。
优先选择名为 `y`、`value`、`sales`、`demand`、`revenue` 等的列。
存在多个候选列时，退而选择最后一个数值列。

```python
col = infer_target_col(df, time_col="date")
col = infer_target_col(df, target_col="sales", exclude=["promotion"])
```

---

### `infer_id_col`

```python
infer_id_col(
    data: pd.DataFrame,
    id_col: str | None = None,
) -> str | None
```

Infer or validate the series-ID column for panel (multi-series) data.
Pass `id_col='auto'` to auto-detect columns named `id`, `series_id`, `store_id`, etc.
Returns `None` for single-series mode.
推断或验证面板（多序列）数据的序列 ID 列。
传入 `id_col='auto'` 可自动检测名为 `id`、`series_id`、`store_id` 等的列。
单序列模式返回 `None`。

```python
col = infer_id_col(df, id_col="store_id")   # explicit / 显式指定
col = infer_id_col(df, id_col="auto")        # auto-detect / 自动检测
col = infer_id_col(df)                       # returns None / 返回 None
```

---

### `preprocess`

```python
preprocess(
    data: pd.DataFrame | str | Path,
    time_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    freq: str | bool | None = None,
    fill_missing: bool = True,
    fill_method: str = "linear",      # 'linear' | 'ffill' | 'bfill' | 'zero'
    deduplicate: bool = True,
    clip_outliers: bool | str = "auto",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]
```

Clean and standardize a time series DataFrame for forecasting.
清洗并标准化时间序列 DataFrame，使其适合预测。

**Steps applied (in order) / 处理步骤（按顺序）:**
1. Column inference — resolves `time_col`, `target_col`, `id_col` / 列推断——解析 `time_col`、`target_col`、`id_col`
2. Sort & deduplicate — sorts by time, removes duplicate timestamps per series / 排序去重——按时间排序，删除每条序列中重复的时间戳
3. Frequency resampling (optional) — resamples to a regular grid at `freq` / 频率重采样（可选）——按 `freq` 重采样到规则网格
4. Missing value filling — fills NaN values in numeric columns / 缺失值填充——填充数值列中的 NaN
5. Outlier clipping — clips extreme values in `target_col` / 异常值裁剪——裁剪 `target_col` 中的极端值

| Parameter / 参数 | Description / 描述 |
|---|---|
| `freq` | Pandas offset string (`'D'`, `'MS'`, `'h'`), `True`/`'auto'` for auto-detect, or `None` to skip / Pandas 频率字符串，`True`/`'auto'` 自动检测，`None` 跳过 |
| `clip_outliers` | `'auto'` clips only when IQR outliers detected; `True` always clips; `False` never / `'auto'` 仅在检测到 IQR 异常时裁剪；`True` 始终裁剪；`False` 从不裁剪 |
| `return_info` | Return `(cleaned_df, info_dict)` instead of just the DataFrame / 返回 `(cleaned_df, info_dict)` 而非仅返回 DataFrame |

```python
clean = preprocess("sales.csv")
clean, info = preprocess(df, time_col="date", return_info=True)
clean = preprocess(df, freq="D", clip_outliers=True)
```

**`info` dict keys / 返回字典键:** `time_col`, `target_col`, `id_col`, `rows`, `freq`,
`filled_missing`, `clipped_outliers`.

---

### `diagnose`

```python
diagnose(
    data: pd.DataFrame | str | Path,
    time_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    horizon: int | None = None,
    known_covariates: list | str | None = None,
    past_covariates: list | str | None = None,
    full: bool = False,
    **read_kwargs,
) -> dict
```

Run a comprehensive readiness diagnostic on a time series dataset.
Answers: *Is this data ready for forecasting?*
对时间序列数据集进行全面的就绪性诊断。
回答：*数据是否已准备好进行预测？*

**Returned dict keys / 返回字典键:**

| Key / 键 | Description / 描述 |
|---|---|
| `status` | `'READY'` / `'WARNING'` / `'NOT_READY'` |
| `rows` / `clean_rows` | Raw and cleaned row count / 原始行数与清洗后行数 |
| `time_col` / `target_col` / `id_col` | Resolved column names / 已解析的列名 |
| `freq` | Detected frequency (e.g. `'MS'`) / 检测到的频率 |
| `horizon` | Effective forecast horizon used / 实际使用的预测步数 |
| `preprocess` | Dict of preprocessing actions applied / 已执行的预处理操作字典 |
| `reports` | Sub-reports: `readiness`, `forecastability`, `baseline`, `recommendations` (+ `full=True` 时包含 `panel` 及扩展报告) |
| `next_step` | Copy-pasteable `forecast()` call / 可直接复制运行的 `forecast()` 调用 |

```python
result = diagnose("sales.csv", horizon=12)
print(result["status"])             # 'READY'
print(result["next_step"])          # forecast(data, n=12, ...)
print(result["reports"]["forecastability"])

# Extended reports / 扩展报告
result = diagnose(df, full=True)
print(result["reports"]["seasonality"])
print(result["reports"]["trend"])
```

---

### `AutoForecast`

```python
AutoForecast(
    time_col: str | None = None,
    target_col: str | None = None,
    horizon: int | None = None,         # also: n_predict
    quantile: float | None = None,
    preset: str = "fast",
    time_limit: float | None = None,
    id_col: str | None = None,
    known_covariates: list | str | None = None,
    past_covariates: list | str | None = None,
    preprocess_data: bool | dict = True,
    verbose: bool = False,
    **router_kwargs,
)
```

Scikit-learn–style AutoML forecaster backed by `SmartRouter`.
Handles column inference, preprocessing, and training in one object.
基于 `SmartRouter` 的 sklearn 风格 AutoML 预测器。
在单个对象中完成列推断、预处理和训练。

**Presets / 预设方案:**

| Preset / 预设 | Models / 模型数 | Search / 搜索 | Ensemble / 集成 | Speed / 速度 |
|---|---|---|---|---|
| `'fast'` | 3 | basic / 基础 | none / 无 | seconds / 秒级 |
| `'medium_quality'` | 5 | auto / 自动 | auto / 自动 | ~1 min / 约 1 分钟 |
| `'high_quality'` | 8 | thorough / 充分 | weighted / 加权 | ~5 min / 约 5 分钟 |
| `'best_quality'` | 15 | thorough / 充分 | top-5 weighted / 前 5 加权 | ~20 min / 约 20 分钟 |

**Key attributes (available after `fit`) / 关键属性（`fit` 后可用）:**

| Attribute / 属性 | Description / 描述 |
|---|---|
| `router_` | Fitted `SmartRouter` instance / 已拟合的 `SmartRouter` 实例 |
| `inferred_columns_` | Dict with resolved `time_col`, `target_col`, `id_col` / 已解析的列名字典 |
| `training_data_` | Preprocessed training DataFrame / 已预处理的训练数据 |
| `leader_board_` | Model ranking table / 模型排行榜 DataFrame |
| `strategy_` | Routing decision summary / 路由决策摘要 |
| `best_model_` | Best fitted model object / 最佳已拟合模型对象 |

**Methods / 方法:**

```python
model = AutoForecast(horizon=12, preset="medium_quality", quantile=0.9)

model.fit(data)                                    # Train / 训练
model.fit(train_df, valid_data=val_df)             # With explicit validation / 显式验证集

pred = model.predict()                             # Next horizon steps / 预测未来 horizon 步
pred = model.predict(n=24)                         # Override horizon / 覆盖预测步数
pred = model.predict(data=new_context)             # Fresh context window / 新上下文窗口
pred = model.predict(future_covariates=fut_df)     # With future covariates / 带未来协变量

pred = model.fit_predict(data)                     # fit + predict in one call / 一步完成

model.save("forecaster.pts")
loaded = AutoForecast.load("forecaster.pts")
```

```python
# Full example / 完整示例
from PipelineTS import AutoForecast

model = AutoForecast(
    time_col="date",
    target_col="value",
    horizon=12,
    preset="high_quality",
    quantile=0.9,
    id_col=None,                       # set for panel data / 面板数据时指定
    known_covariates=["promotion"],    # future-known columns / 未来已知列
    time_limit=300,                    # 5-minute budget / 5 分钟训练预算
)
model.fit(train_df, valid_data=val_df)
pred = model.predict()

print(model.leader_board_)
print(model.strategy_)
model.save("my_model.pts")
```

---

### `forecast`

```python
forecast(
    data: pd.DataFrame | str | Path,
    n: int | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
    horizon: int | None = None,
    quantile: float | None = None,
    preset: str = "fast",
    time_limit: float | None = None,
    id_col: str | None = None,
    known_covariates: list | str | None = None,
    past_covariates: list | str | None = None,
    future_covariates: pd.DataFrame | None = None,
    preprocess_data: bool | dict = True,
    return_model: bool = False,
    verbose: bool = False,
    valid_data: pd.DataFrame | None = None,
    **router_kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, AutoForecast]
```

One-line AutoML forecast — the simplest entry point.
Auto-infers columns, preprocesses, trains, and returns predictions.
一行 AutoML 预测——最简入口。
自动推断列名、预处理、训练并返回预测结果。

```python
from PipelineTS import forecast

# Minimal usage / 最简用法
pred = forecast("sales.csv", n=12)

# With options / 带选项
pred = forecast(df, n=12, quantile=0.9, preset="high_quality")

# Panel data / 面板数据
pred = forecast(df, n=12, id_col="store_id")

# With covariates / 带协变量
pred = forecast(
    df, n=12,
    known_covariates=["promotion", "holiday"],
    future_covariates=future_df,   # DataFrame with n rows / 含 n 行的 DataFrame
)

# Get the model back for saving / 返回模型以保存
pred, model = forecast(df, n=12, return_model=True)
model.save("forecaster.pts")
```

**Returns / 返回值:** DataFrame with `[time_col, target_col]` and optionally
`[target_col_lower, target_col_upper]` columns.
包含 `[time_col, target_col]` 列，设置 `quantile` 时还包含 `[target_col_lower, target_col_upper]`。

---

### `backtest`

```python
backtest(
    data: pd.DataFrame | str | Path,
    n: int | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    n_splits: int = 3,
    test_size: int | None = None,
    metric: str | callable = "mae",
    mode: str = "expanding",          # 'expanding' | 'sliding'
    train_size: int | None = None,
    preset: str = "fast",
    time_limit: float | None = None,
    quantile: float | None = None,
    known_covariates: list | str | None = None,
    past_covariates: list | str | None = None,
    preprocess_data: bool | dict = True,
    verbose: bool = False,
    return_backtester: bool = False,
    **router_kwargs,
) -> dict
```

Walk-forward backtesting with AutoML forecasting.
Simulates how the model would have performed in production.
基于 AutoML 预测的时间序列前向回测。
模拟模型在生产环境中的实际表现。

**Metric strings / 指标字符串:** `'mae'`, `'mse'`, `'rmse'`, `'mape'`, `'smape'`, `'wmape'`, `'medae'`
Or pass any `callable(y_true, y_pred) -> float`.
或传入任意 `callable(y_true, y_pred) -> float` 自定义指标。

**Returned dict / 返回字典:**

| Key / 键 | Description / 描述 |
|---|---|
| `results` | List of per-fold metric values / 各折指标值列表 |
| `summary` | Dict: `mean`, `std`, `min`, `max` / 统计摘要字典 |
| `metric` | Metric name / 指标名称 |
| `time_col` / `target_col` / `id_col` | Resolved column names / 已解析的列名 |
| `horizon` | Effective test window size / 实际测试窗口大小 |
| `n_splits` | Number of folds evaluated / 评估折数 |
| `backtester` | `Backtester` instance (only when `return_backtester=True`) / 仅 `return_backtester=True` 时返回 |

```python
from PipelineTS import backtest

result = backtest("sales.csv", n=12, n_splits=5)
print(result["summary"])   # {'mean': ..., 'std': ..., 'min': ..., 'max': ...}

# Sliding window / 滑动窗口
result = backtest(df, n=12, metric="smape", mode="sliding", train_size=200)

# Custom metric / 自定义指标
result = backtest(df, n=12, metric=lambda y, yhat: np.median(np.abs(y - yhat)))

# Get per-fold backtester / 获取回测器对象
result = backtest(df, n=12, return_backtester=True)
result["backtester"].summary()
```

---

## Pipeline / 管道

### `PipelineTS.pipeline.ModelPipeline`

The main pipeline class for automatic model selection.
用于自动模型选择的主管道类。

```python
ModelPipeline(
    time_col: str,
    target_col: str,
    lags: int,
    quantile: float | None = None,
    random_state: int = 0,
    metric: callable = mean_absolute_error,
    metric_less_is_better: bool = True,
    include_models: str | list | None = 'light',
    exclude_models: list | None = None,
    configs: PipelineConfigs | None = None,
    include_init_config_model: bool = True,
    scaler: bool | None | TransformerMixin = True,
    accelerator: str = 'auto',
    cv: int = 5,
    feature_cols: list | None = None,
    **model_init_kwargs,
)
```

**Methods / 方法:**

| Method / 方法 | Description / 描述 |
|---|---|
| `fit(data, valid_data=None)` | Train all models and return leaderboard / 训练所有模型并返回排行榜 |
| `predict(n, model_name=None, data=None, future_covariates=None)` | Predict n steps using best or specified model / 使用最佳或指定模型预测 n 步 |
| `predict_quantiles(n, levels, model_name=None)` | Multi-quantile prediction / 多分位数预测 |
| `update(new_data)` | Incremental learning on new data / 在新数据上增量学习 |
| `plot(n=None, lang='zh', history_tail=None)` | Plot forecast from best model / 绘制最佳模型预测图 |
| `plot_leaderboard(lang='zh')` | Plot model leaderboard chart / 绘制模型排行榜图 |
| `get_model(model_name=None)` | Get the best or specified trained model / 获取最佳或指定的已训练模型 |
| `get_model_all_configs(model_name=None)` | Get all configs for a model / 获取模型的全部配置 |
| `list_all_available_models()` | Class method: list all model names / 类方法：列出所有模型名称 |
| `save(path)` | Save pipeline to zip file / 保存管道到 zip 文件 |
| `load(path)` | Static: load pipeline from zip / 静态方法：从 zip 加载管道 |

**Attributes / 属性:**

| Attribute / 属性 | Description / 描述 |
|---|---|
| `leader_board_` | DataFrame with model performance rankings / 模型性能排名 DataFrame |
| `best_model_` | The best performing model object / 最佳模型对象 |
| `failed_models` | List of failed model details / 失败模型详情列表 |
| `skipped_models` | List of skipped model names and reasons / 跳过的模型名称和原因列表 |

---

### `PipelineTS.pipeline.SmartRouter`

Intelligent routing system for automatic data profiling, model selection, and ensemble building.
智能路由系统，自动数据画像、模型选择和集成构建。

```python
SmartRouter(
    time_col: str,
    target_col: str,
    n_predict: int | None = None,
    max_models: int = 5,
    quantile: float | None = None,
    ensemble_strategy: str = 'auto',
    ensemble_top_k: int = 3,
    id_col: str | None = None,
    known_covariates: list | None = None,
    past_covariates: list | None = None,
    include_models: str | list | None = None,
    search_strategy: str = 'auto',
    hpo_strategy: str = 'none',
    hpo_n_trials: int = 10,
    hpo_timeout_per_model: float | None = None,
    time_limit: float | None = None,
    random_state: int = 0,
    verbose: bool = True,
)
```

**Key Parameters / 关键参数:**

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 描述 |
|---|---|---|---|
| `include_models` | str, list, None | None | Pin specific model(s); skips heuristic selection and screening / 指定模型；跳过启发式选择和筛选 |
| `search_strategy` | str | 'auto' | `'basic'` (no screening/exploration), `'auto'`, or `'thorough'` / 搜索策略 |
| `hpo_strategy` | str | 'none' | `'none'`, `'quick'` (≤5 trials), or `'full'` / HPO 策略 |

**Methods / 方法:**

| Method / 方法 | Description / 描述 |
|---|---|
| `fit(data)` | Profile data, select models, train, optionally build ensemble / 数据画像、选择模型、训练、可选构建集成 |
| `predict(n, use_ensemble=True, future_covariates=None)` | Predict using ensemble or best model / 使用集成或最佳模型预测 |
| `predict_quantiles(n, levels)` | Multi-quantile prediction / 多分位数预测 |
| `update(new_data)` | Incremental learning / 增量学习 |
| `plot(n=None, lang='zh')` | Plot forecast / 绘制预测图 |
| `plot_leaderboard(lang='zh')` | Plot leaderboard / 绘制排行榜 |
| `get_model(model_name=None)` | Get fitted model / 获取已训练模型 |
| `list_all_available_models()` | Class method: list all valid model names / 类方法：列出所有有效模型名称 |

**Attributes / 属性:**

| Attribute / 属性 | Description / 描述 |
|---|---|
| `strategy` | Selected strategy dict (models, lags, scaler, etc.) / 选定策略字典 |
| `leader_board_` | Model rankings / 模型排名 |
| `ensemble_` | EnsemblePredictor (if built) / 集成预测器 |
| `pipeline_` | Underlying ModelPipeline / 底层 ModelPipeline |
| `profile_` | DataProfile with data characteristics / 数据画像 |
| `include_models` | User-pinned model list (or None) / 用户指定模型列表（或 None） |

---

### `PipelineTS.pipeline.PipelineConfigs`

Configuration class for creating model variants with per-model settings.
用于创建模型变体并支持每模型设置的配置类。

```python
PipelineConfigs(configs: list[tuple])
```

Each tuple format / 每个元组的格式:
- `(model_name, config_dict)` - Auto-named / 自动命名
- `(model_name, custom_name, config_dict)` - Custom-named / 自定义命名

**Config dict keys / 配置字典键:**

| Key / 键 | Description / 描述 |
|---|---|
| `init_configs` | Model `__init__` parameters / 模型初始化参数 |
| `fit_configs` | Parameters passed to `fit()` / 传递给 `fit()` 的参数 |
| `predict_configs` | Parameters passed to `predict()` / 传递给 `predict()` 的参数 |
| `pipeline_configs` | Pipeline-level per-model settings / 管道级别每模型设置 |

**`pipeline_configs` supported keys / `pipeline_configs` 支持的键:**

| Key / 键 | Type / 类型 | Description / 描述 |
|---|---|---|
| `lags` | int | Per-model input window size / 每模型滞后窗口大小 |
| `scaler` | bool, None, or TransformerMixin | Per-model scaler (`True`=MinMaxScaler, `None`=disabled, or custom instance) / 每模型缩放器 |
| `differential_n` | int | Per-model differencing order / 每模型差分阶数 |
| `feature_cols` | list | Per-model feature columns / 每模型特征列 |

```python
from PipelineTS.pipeline import PipelineConfigs
from sklearn.preprocessing import StandardScaler

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_std', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'lags': 20, 'scaler': StandardScaler()},
    }),
    ('torch_boosting_forest', 'boost_noscale', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'scaler': None},
    }),
])
```

---

## Neural Network Models / 神经网络模型

All located in `PipelineTS.nn_model`.
全部位于 `PipelineTS.nn_model`。

| Class / 类 | Import / 导入 |
|---|---|
| `NLinearModel` | `from PipelineTS.nn_model import NLinearModel` |
| `DLinearModel` | `from PipelineTS.nn_model import DLinearModel` |
| `NBeatsModel` | `from PipelineTS.nn_model import NBeatsModel` |
| `NHitsModel` | `from PipelineTS.nn_model import NHitsModel` |
| `TFTModel` | `from PipelineTS.nn_model import TFTModel` |
| `TransformerModel` | `from PipelineTS.nn_model import TransformerModel` |
| `TiDEModel` | `from PipelineTS.nn_model import TiDEModel` |
| `GAUModel` | `from PipelineTS.nn_model import GAUModel` |
| `StackingRNNModel` | `from PipelineTS.nn_model import StackingRNNModel` |
| `Time2VecModel` | `from PipelineTS.nn_model import Time2VecModel` |
| `PatchRNNModel` | `from PipelineTS.nn_model import PatchRNNModel` |
| `TCNModel` | `from PipelineTS.nn_model import TCNModel` |
| `ITransformerModel` | `from PipelineTS.nn_model import ITransformerModel` |
| `SRSNetModel` | `from PipelineTS.nn_model import SRSNetModel` |
| `DeepARModel` | `from PipelineTS.nn_model import DeepARModel` |
| `TimeXerModel` | `from PipelineTS.nn_model import TimeXerModel` |
| `TimeMixerModel` | `from PipelineTS.nn_model import TimeMixerModel` |
| `TimesNetModel` | `from PipelineTS.nn_model import TimesNetModel` |
| `PyraformerModel` | `from PipelineTS.nn_model import PyraformerModel` |
| `ETSformerModel` | `from PipelineTS.nn_model import ETSformerModel` |
| `LightTSModel` | `from PipelineTS.nn_model import LightTSModel` |
| `PatchTSTModel` | `from PipelineTS.nn_model import PatchTSTModel` |
| `TSMixerModel` | `from PipelineTS.nn_model import TSMixerModel` |
| `NonstationaryTransformerModel` | `from PipelineTS.nn_model import NonstationaryTransformerModel` |
| `FEDformerModel` | `from PipelineTS.nn_model import FEDformerModel` |
| `AutoformerModel` | `from PipelineTS.nn_model import AutoformerModel` |
| `InformerModel` | `from PipelineTS.nn_model import InformerModel` |
| `ReformerModel` | `from PipelineTS.nn_model import ReformerModel` |
| `MultiPatchFormerModel` | `from PipelineTS.nn_model import MultiPatchFormerModel` |
| `WPMixerModel` | `from PipelineTS.nn_model import WPMixerModel` |
| `TimeFilterModel` | `from PipelineTS.nn_model import TimeFilterModel` |
| `MSGNetModel` | `from PipelineTS.nn_model import MSGNetModel` |
| `SegRNNModel` | `from PipelineTS.nn_model import SegRNNModel` |
| `TiRexModel` | `from PipelineTS.nn_model import TiRexModel` |
| `Chronos2Model` *(optional)* | `from PipelineTS.nn_model import Chronos2Model` |
| `Chronos2SynthModel` *(optional)* | `from PipelineTS.nn_model import Chronos2SynthModel` |
| `Chronos2SmallModel` *(optional)* | `from PipelineTS.nn_model import Chronos2SmallModel` |
| `TiRexFoundationModel` *(optional)* | `from PipelineTS.nn_model import TiRexFoundationModel` |
| `SundialModel` *(optional)* | `from PipelineTS.nn_model import SundialModel` |
| `TimeMoEModel` *(optional)* | `from PipelineTS.nn_model import TimeMoEModel` |

**Common interface / 通用接口:**

```python
model = SomeModel(
    time_col: str,           # Time column name / 时间列名
    target_col: str | list,  # Target column(s) / 目标列
    lags: int,               # Input window size / 输入窗口大小
    quantile: float | None,  # Interval coverage / 区间覆盖率
    random_state: int,       # Random seed / 随机种子
    epochs: int = 1000,      # Max epochs / 最大轮数
    patience: int = 100,     # Early stopping / 早停
    verbose: bool = False,   # Show progress / 显示进度
    learning_rate: float = 0.001,  # Learning rate / 学习率
    feature_cols: list | None = None,  # For multivariate models / 多变量模型用
    use_gtb: bool = False,   # Enable GlobalTemporalBlock / 启用全局时序块
    gtb_d_model: int = 64,   # GTB hidden dimension / GTB 隐藏维度
    routing_mode: str = 'static',  # 'static' or 'adaptive' (MoE) / 静态或自适应（MoE）
)

model.fit(data, valid_data=None)  # Train / 训练
model.predict(n, data=None)       # Predict / 预测
```

The modern NN family (`timexer`, `time_mixer`, `timesnet`, `patchtst`, etc.) is implemented through shared layers and a central model spec registry, so public classes remain available without duplicating 19 separate training wrappers.
现代 NN 模型族（如 `timexer`、`time_mixer`、`timesnet`、`patchtst` 等）通过共享层和统一模型规格注册实现，在保留公开类的同时避免复制 19 份训练包装逻辑。

### Foundation Models *(optional dependencies / 可选依赖)*

Zero-shot foundation models wrapping Chronos-2, TiRex, Sundial, and Time-MoE pretrained checkpoints.
零样本基础模型，封装 Chronos-2、TiRex、Sundial 和 Time-MoE 预训练检查点。

> Optional dependencies: `pip install chronos-forecasting`, `pip install tirex-ts`, and `pip install transformers==4.40.1`.

| Class / 类 | Pipeline Key / 管道键名 | HuggingFace Path | Size / 大小 |
|---|---|---|---|
| `Chronos2Model` | `chronos_2` | `amazon/chronos-2` | 120M |
| `Chronos2SynthModel` | `chronos_2_synth` | `autogluon/chronos-2-synth` | 120M |
| `Chronos2SmallModel` | `chronos_2_small` | `autogluon/chronos-2-small` | 28M |
| `TiRexFoundationModel` | `tirex_foundation` | `NX-AI/TiRex` | 35M |
| `SundialModel` | `sundial` | `thuml/sundial-base-128m` | 128M |
| `TimeMoEModel` | `time_moe` | `Maple728/TimeMoE-50M` | 50M |

`ChronosModel` is a backward-compatible alias for `Chronos2Model`.
`ChronosModel` 是 `Chronos2Model` 的向后兼容别名。

```python
from PipelineTS.nn_model import Chronos2Model, Chronos2SmallModel, TiRexFoundationModel, SundialModel, TimeMoEModel

model = Chronos2SmallModel(
    time_col: str,
    target_col: str,
    lags: int = 1,                   # API compatibility / API 兼容
    quantile: float | None = 0.9,    # Conformal interval coverage / 共形区间覆盖率
    device_map: str = 'auto',        # 'auto', 'cpu', 'cuda', 'mps'
)

model.fit(data, cv=5)                              # Store data + calibrate intervals / 存储数据 + 校准区间
model.predict(n, future_covariates=None)            # Zero-shot predict / 零样本预测
```

**Features / 特点:** Zero-shot (no training), multi-series (`id_col`), Chronos-2 covariates support, conformal intervals.
**特点：** 零样本（无需训练）、多序列（`id_col`）、Chronos-2 协变量支持、共形预测区间。

---

## Machine Learning Models / 机器学习模型

All located in `PipelineTS.ml_model`.
全部位于 `PipelineTS.ml_model`。

### GPU-Accelerated Tree Models / GPU 加速树模型

| Class / 类 | Import / 导入 | Description / 描述 |
|---|---|---|
| `TorchBoostingForestModel` | `from PipelineTS.ml_model import TorchBoostingForestModel` | Staged gradient boosting (MART/DART) / 分阶段梯度提升 |
| `TorchBaggingForestModel` | `from PipelineTS.ml_model import TorchBaggingForestModel` | Bagging forest with dropout / 带 Dropout 的袋装森林 |
| `DeepForestModel` | `from PipelineTS.ml_model import DeepForestModel` | Cascade multi-layer ensemble (gcForest) / 级联多层集成 |

```python
TorchBoostingForestModel(
    time_col: str,
    target_col: str,
    lags: int = 1,
    quantile: float | None = 0.9,
    accelerator: str | None = None,  # 'cuda', 'mps', 'cpu', or None (auto)
    n_trees: int = 64,
    tree_depth: int = 5,
    learning_rate: float = 0.08,
    n_epochs: int = 200,
    batch_size: int = 0,             # 0 = full batch
    early_stop_patience: int = 15,
    dropout: float = 0.0,
    weight_decay: float = 1e-4,
    boosting_stages: int = 3,
    boosting_shrinkage: float = 0.5,
    random_state: int | None = None,
    verbose: bool = False,
    auto_complexity: bool = False,   # Enable adaptive depth/trees auto-tuning
)
```

```python
TorchBaggingForestModel(
    time_col: str,
    target_col: str,
    lags: int = 1,
    quantile: float | None = 0.9,
    accelerator: str | None = None,
    n_trees: int = 128,
    tree_depth: int = 5,
    learning_rate: float = 0.08,
    n_epochs: int = 300,
    batch_size: int = 0,
    early_stop_patience: int = 15,
    dropout: float = 0.15,           # Tree-level dropout for decorrelation
    weight_decay: float = 1e-4,
    random_state: int | None = None,
    verbose: bool = False,
    auto_complexity: bool = False,
)
```

```python
DeepForestModel(
    time_col: str,
    target_col: str,
    lags: int = 1,
    quantile: float | None = 0.9,
    accelerator: str | None = None,
    n_trees: int = 32,
    tree_depth: int = 4,
    n_layers: int = 3,               # Number of cascade layers
    learning_rate: float = 0.08,
    n_epochs: int = 200,
    batch_size: int = 0,
    early_stop_patience: int = 12,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    random_state: int | None = None,
    verbose: bool = False,
    auto_complexity: bool = False,
)
```

**Auto-complexity property / 自适应复杂度属性:**

```python
model.model.complexity_info  # dict or None
# Returns: {'profile', 'tree_depth', 'n_trees', 'complexity_score', 'reasons', 'stats'}
```

### Other ML Models / 其他 ML 模型

| Class / 类 | Import / 导入 |
|---|---|
| `WideGBRTModel` | `from PipelineTS.ml_model import WideGBRTModel` |
| `MultiOutputRegressorModel` | `from PipelineTS.ml_model import MultiOutputRegressorModel` |
| `MultiStepRegressorModel` | `from PipelineTS.ml_model import MultiStepRegressorModel` |
| `RegressorChainModel` | `from PipelineTS.ml_model import RegressorChainModel` |

**Common interface / 通用接口:**

```python
model = SomeMLModel(
    time_col: str,
    target_col: str,
    lags: int,
    quantile: float | None = None,
    random_state: int = 42,
    **model_specific_params,
)

model.fit(data)
model.predict(n, data=None)
```

---

## Statistical Models / 统计模型

All located in `PipelineTS.statistic_model`.
全部位于 `PipelineTS.statistic_model`。

| Class / 类 | Pipeline key | Import / 导入 |
|---|---|---|
| `ProphetModel` | `prophet` | `from PipelineTS.statistic_model import ProphetModel` |
| `AutoARIMAModel` | `auto_arima` | `from PipelineTS.statistic_model import AutoARIMAModel` |
| `ThetaModel` | `theta` | `from PipelineTS.statistic_model import ThetaModel` |
| `ETSModel` | `ets` | `from PipelineTS.statistic_model import ETSModel` |
| `NaiveModel` | `naive` | `from PipelineTS.statistic_model import NaiveModel` |
| `SeasonalNaiveModel` | `seasonal_naive` | `from PipelineTS.statistic_model import SeasonalNaiveModel` |
| `StatisticalEnsembleModel` | `stat_ensemble` | `from PipelineTS.statistic_model import StatisticalEnsembleModel` |

**Lightweight baselines** — `theta`, `ets`, `naive`, `seasonal_naive`, `stat_ensemble`
are included in `include_models='light'` and SmartRouter's safe portfolio.
They are dependency-free, train in milliseconds, and are ideal baseline champions.

```python
from PipelineTS.statistic_model import AutoARIMAModel, ProphetModel

model = AutoARIMAModel(
    time_col="date",
    target_col="value",
    lags=12,
    quantile=0.9,
    max_p=5, max_q=5, max_d=2,
)
model.fit(data)
pred = model.predict(12)

model2 = ProphetModel(
    time_col="date",
    target_col="value",
    lags=12,
    quantile=0.9,
    known_covariates=["promotion"],
    past_covariates=["temperature"],
)
model2.fit(data)
pred2 = model2.predict(12)
```

---

## Dataset / 数据集

All located in `PipelineTS.dataset`.
全部位于 `PipelineTS.dataset`。

| Function/Class / 函数/类 | Description / 描述 |
|---|---|
| `LoadElectricDataSets()` | Electric Production dataset / 电力生产数据集 |
| `LoadMessagesSentDataSets()` | Messages Sent (daily) / 消息发送量（日度） |
| `LoadMessagesSentHourDataSets()` | Messages Sent (hourly) / 消息发送量（小时） |
| `LoadWebSales()` | Web Sales dataset / 网络销售数据集 |
| `LoadSupermarketIncoming()` | Supermarket Incoming / 超市进货量 |
| `BuiltInSeriesData()` | Access all built-in datasets / 访问所有内置数据集 |
| `DataGenerator()` | Generate synthetic data / 生成合成数据 |

---

## Preprocessing / 预处理

All located in `PipelineTS.preprocessing`.
全部位于 `PipelineTS.preprocessing`。

### `Scaler`

```python
Scaler(scaler_name: str)
# scaler_name: 'min_max' | 'standard' | 'quantile' | 'gauss_rank'

scaler.fit_transform(X)      # Fit and transform / 拟合并变换
scaler.transform(X)          # Transform / 变换
scaler.inverse_transform(X)  # Inverse transform / 反向变换
```

### `TimeSeriesMissingHandler`

```python
TimeSeriesMissingHandler(time_col: str)

handler.fit(data, value_cols=None)                      # Detect missing values / 检测缺失值
handler.transform(data, method='linear', fill_implicit_gaps=True)  # Fill missing values / 填充缺失值
handler.fit_transform(data, method='linear')            # Fit + transform in one call / 一步检测并填充
# method: 'linear' | 'ffill' | 'bfill' | 'spline' | 'zero'
```

### `TimeSeriesOutlierDetector`

```python
TimeSeriesOutlierDetector(time_col: str, method: str = 'iqr', threshold: float = 1.5, window: int = 20)
# method: 'iqr' | 'zscore' | 'rolling_zscore' | 'grubbs'

detector.fit(data, target_col)                          # Returns boolean mask / 返回布尔掩码
detector.transform(data, target_col, strategy='clip')   # Handle outliers / 处理异常值
detector.fit_transform(data, target_col, strategy='clip')  # Fit + transform / 一步检测并处理
# strategy: 'clip' | 'nan' | 'median' | 'linear'
```

### `TimeSeriesDataQualityReport`

```python
TimeSeriesDataQualityReport(time_col: str, target_col: str)

reporter.fit(data)            # Returns report dict / 返回报告字典
reporter.report(data)         # Print formatted report / 打印格式化报告
```

### `StationarityTest`

```python
StationarityTest(significance_level: float = 0.05)

tester.adf_test(series)                     # ADF test / ADF 检验
tester.kpss_test(series, regression='c')    # KPSS test / KPSS 检验
tester.fit(series)                           # Combined ADF + KPSS / 联合检验
tester.suggest_differencing(series, max_d=2) # Suggest differencing order / 建议差分阶数
```

### `FrequencyDetector`

```python
from PipelineTS.preprocessing.time_series_analysis import FrequencyDetector

FrequencyDetector(time_col: str)

detector.fit(data, target_col=None)      # Detect frequency + dominant periods / 检测频率 + 主要周期
```

### `TimeSeriesSplit`

```python
TimeSeriesSplit.split(data, time_col, test_size=0.2)                              # Simple split / 简单分割
TimeSeriesSplit.expanding_window(data, time_col, min_train_size, test_size, step)  # Expanding CV / 扩展窗口
TimeSeriesSplit.sliding_window(data, time_col, train_size, test_size, step)        # Sliding CV / 滑动窗口
```

### Sequence Splitting / 序列分割

```python
from PipelineTS.preprocessing import (
    split_series,                # Univariate split / 单变量分割
    split_series_multivariate,   # Multivariate split / 多变量分割
    train_test_split_ts,         # Time-series train/test split / 时序训练/测试分割
)
```

---

## Feature Engineering / 特征工程

All located in `PipelineTS.feature_engineering`.
全部位于 `PipelineTS.feature_engineering`。

### `TimeSeriesFeatureEngineer`

```python
TimeSeriesFeatureEngineer(
    time_col, target_col=None,
    use_calendar=True, use_fourier=False, fourier_periods=None, fourier_harmonics=1,
    use_holidays=False, holiday_country=None, custom_holidays=None,
    use_lags=False, lag_window='auto', lag_features='all', drop_time_col=False,
)

engineer.fit(data)             # Fit / 拟合
engineer.transform(data)       # Transform / 转换
engineer.fit_transform(data)   # Fit + transform / 拟合 + 转换
engineer.get_feature_names()   # List generated feature names / 列出生成的特征名
```

### `FourierFeatures`

```python
FourierFeatures(time_col, periods, n_harmonics=1, prefix='fourier_')

ff.transform(data)           # Add Fourier features / 添加傅里叶特征
ff.get_feature_names()       # Feature column names / 特征列名
```

### `HolidayFeatures`

```python
HolidayFeatures(time_col, country=None, custom_holidays=None, window=3, prefix='holiday_')

hf.transform(data)           # Add holiday features / 添加节假日特征
hf.get_feature_names()       # Feature column names / 特征列名
# For country='CN': uses chinese-calendar (pip install chinesecalendar) as authoritative source
# 当 country='CN' 时：使用 chinese-calendar 作为标准数据源
# Extra CN features: holiday_is_workday, holiday_is_in_lieu, holiday_holiday_name
# 中国额外特征：holiday_is_workday（工作日）, holiday_is_in_lieu（调休）, holiday_holiday_name（节日名）
```

### `LagFeatureExtractor`

```python
LagFeatureExtractor(time_col, target_col, window='auto', features='all', prefix='lag_')
# features: 'all' or subset of: mean, std, min, max, median, skew, kurtosis,
#           trend_slope, ema, autocorr, momentum, rms, cv, iqr, energy

lf.transform(data)           # Add lag features / 添加滞后特征
lf.get_feature_names()       # Feature column names / 特征列名
```

---

## Metrics / 指标

### Point Metrics / 点指标

```python
from PipelineTS.metrics import mae, mse, rmse, wmape, mape, smape, mase, r2_score, medae
```

| Function / 函数 | Description / 描述 |
|---|---|
| `mae(y_true, y_pred)` | Mean Absolute Error / 平均绝对误差 |
| `mse(y_true, y_pred)` | Mean Squared Error / 均方误差 |
| `rmse(y_true, y_pred)` | Root Mean Squared Error / 均方根误差 |
| `wmape(y_true, y_pred)` | Weighted MAPE / 加权 MAPE |
| `mape(y_true, y_pred)` | Mean Absolute Percentage Error / 平均绝对百分比误差 |
| `smape(y_true, y_pred)` | Symmetric MAPE / 对称 MAPE |
| `mase(y_true, y_pred, y_train, seasonality=1)` | Mean Absolute Scaled Error / 平均绝对缩放误差 |
| `r2_score(y_true, y_pred)` | Coefficient of Determination / 决定系数 |
| `medae(y_true, y_pred)` | Median Absolute Error / 中位绝对误差 |

### Interval Metrics / 区间指标

```python
from PipelineTS.metrics import quantile_acc, picp, pinaw, winkler_score
```

| Function / 函数 | Description / 描述 |
|---|---|
| `quantile_acc(y_true, lower, upper)` | Interval coverage rate / 区间覆盖率 |
| `picp(y_true, lower, upper)` | Prediction Interval Coverage Probability / 预测区间覆盖概率 |
| `pinaw(y_true, lower, upper)` | Normalized Average Width / 归一化平均宽度 |
| `winkler_score(y_true, lower, upper, alpha=0.1)` | Winkler interval score / Winkler 区间分数 |

---

## Evaluation / 评估

All located in `PipelineTS.evaluation`.
全部位于 `PipelineTS.evaluation`。

### `Backtester`

```python
Backtester(model, time_col, target_col, metric, metric_name='metric', metric_less_is_better=True)

bt.fit(data, n_splits=5, test_size=10, mode='expanding', train_size=None, verbose=True)
bt.summary()  # Returns dict: mean, std, min, max, median, n_folds, n_failed
```

### `ResidualAnalyzer`

```python
ResidualAnalyzer(y_true, y_pred)

analyzer.statistics()          # Basic stats / 基本统计量
analyzer.normality_test()      # Shapiro-Wilk + Jarque-Bera / 正态性检验
analyzer.autocorrelation()     # ACF + Ljung-Box / 自相关检验
analyzer.bias_analysis()       # Systematic bias / 系统性偏差
analyzer.report()              # Formatted output / 格式化输出
analyzer.plot()                # 4-panel diagnostic / 四面板诊断图
```

### `ModelComparison`

```python
ModelComparison(time_col, target_col)

comp.add_result(model_name, y_true, y_pred, lower=None, upper=None)
comp.fit(metrics=None, interval_metrics=None)         # Returns DataFrame / 返回 DataFrame
comp.rank(metric_name, ascending=True)                # Ranked table / 排名表
comp.plot_bar()                                        # Bar chart / 柱状图
comp.plot_radar()                                      # Radar chart / 雷达图
comp.plot_predictions(time_index=None)                 # Overlay plot / 叠加图
```

---

## Training / 训练

All located in `PipelineTS.training`.
全部位于 `PipelineTS.training`。

### `AutoTune`

```python
AutoTune(model_class, time_col, target_col, lags, metric,
         metric_less_is_better=True, n_trials=20, test_size=0.2,
         fixed_params=None, random_state=0)

best_model, best_params, history = tuner.fit(data, search_space, verbose=True)
# search_space format: {'param': ('int'|'float'|'categorical', ...)}
```

### `WeightedEnsemble`

```python
WeightedEnsemble(models, time_col, target_col, weights='auto', metric=None)

ens.fit(data, valid_data=None)
ens.predict(n)
ens.get_weights()  # Returns {name: weight} / 返回 {名称: 权重}
```

### `StackingEnsemble`

```python
StackingEnsemble(models, time_col, target_col, n_folds=3)

stack.fit(data)
stack.predict(n)
```

---

## Prediction / 预测

All located in `PipelineTS.prediction`.
全部位于 `PipelineTS.prediction`。

### `RollingPredictor`

```python
RollingPredictor(model, time_col, target_col, train_size, horizon, step=1, refit=True)

results = rp.predict(data, verbose=True)            # Returns DataFrame / 返回 DataFrame
eval_results = rp.score(results, metrics=None)      # Per-window + overall metrics / 每窗口 + 总体指标
```

### `ModelExplainer`

```python
ModelExplainer(model, time_col, target_col)

explainer.feature_importance()                  # Native importance / 原生重要性
explainer.plot_importance(importance_df=None, top_k=20)  # Bar chart / 柱状图
```

---

## I/O

```python
from PipelineTS.io import save_model, load_model

save_model(path: str, model, metadata=None)  # Save model to .pts file
load_model(path: str, verify_checksum=True)  # Load model from .pts file
```

Or use the `.save()` / `.load()` methods directly on any fitted object:

```python
pipeline.save("my_pipeline.pts")
loaded = ModelPipeline.load("my_pipeline.pts")

router.save("my_router.pts", metadata={"version": "1.0"})
loaded = SmartRouter.load("my_router.pts")

model.save("forecaster.pts")
loaded = AutoForecast.load("forecaster.pts")
```

---

## Plotting / 绘图

All located in `PipelineTS.plot`.
全部位于 `PipelineTS.plot`。

### Chinese Font Configuration / 中文字体配置

```python
from PipelineTS.plot import configure_chinese_font

configure_chinese_font(force: bool = False) -> str
# Auto-detect and set Chinese font. Returns font name.
# 自动检测并设置中文字体。返回字体名称。
```

### `TSPlotter`

```python
TSPlotter(time_col: str, target_col: str, lang: str = 'zh')

plotter.plot_series(data, id_col=None, **kwargs)
plotter.plot_forecast(train_data, forecast_data, **kwargs)
plotter.plot_leaderboard(leaderboard, **kwargs)
plotter.plot_leaderboard_detail(leaderboard, **kwargs)
plotter.plot_model_comparison(train_data, predictions, **kwargs)
plotter.plot_residuals(y_true, y_pred, **kwargs)
plotter.plot_acf_pacf(series, **kwargs)
plotter.plot_decomposition(data, **kwargs)
plotter.plot_train_test_split(train_data, test_data, **kwargs)
```

### Standalone Plot Functions / 独立绑图函数

```python
from PipelineTS.plot import (
    plot_series, plot_forecast, plot_leaderboard, plot_leaderboard_detail,
    plot_model_comparison, plot_residuals, plot_acf_pacf,
    plot_decomposition, plot_train_test_split,
)
```

| Function / 函数 | Signature / 签名 |
|---|---|
| `plot_series` | `(data, time_col, target_col, id_col=None, max_series=9, lang='zh', show=True)` |
| `plot_forecast` | `(train_data, forecast_data, time_col, target_col, history_tail=None, lang='zh', show=True)` |
| `plot_leaderboard` | `(leaderboard, metric_col='metric', model_col='model', lang='zh', show=True)` |
| `plot_leaderboard_detail` | `(leaderboard, lang='zh', show=True)` |
| `plot_model_comparison` | `(train_data, predictions: dict, time_col, target_col, lang='zh', show=True)` |
| `plot_residuals` | `(y_true, y_pred, time_index=None, lang='zh', show=True)` |
| `plot_acf_pacf` | `(series, max_lags=30, lang='zh', show=True)` |
| `plot_decomposition` | `(data, time_col, target_col, period=None, model='additive', lang='zh', show=True)` |
| `plot_train_test_split` | `(train_data, test_data, time_col, target_col, lang='zh', show=True)` |

All functions return `matplotlib.figure.Figure` and accept `title`, `figsize`, `show` parameters.
所有函数返回 `matplotlib.figure.Figure`，接受 `title`、`figsize`、`show` 参数。

### Color Constants / 颜色常量

```python
from PipelineTS.plot import COLORS, MODEL_COLORS

COLORS: dict   # Named colors: primary, forecast, actual, interval, etc.
MODEL_COLORS: list  # 15-color palette for model comparison
```

### Legacy Functions / 旧版函数

```python
from PipelineTS.plot import plot_data_period

plot_data_period(
    data1: pd.DataFrame,      # First dataset (e.g., train) / 第一个数据集
    data2: pd.DataFrame,      # Second dataset (e.g., prediction) / 第二个数据集
    time_col: str,
    target_col: str,
    labels: list = None,
    date_fmt: str = '%Y-%m-%d',
)
```
