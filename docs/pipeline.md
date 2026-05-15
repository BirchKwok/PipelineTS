# Pipeline Usage
# 管道使用

`ModelPipeline` is the core class for automatic model comparison and selection in PipelineTS.
`ModelPipeline` 是 PipelineTS 中自动模型比较和选择的核心类。

It trains multiple models, evaluates them with cross-validation, and ranks them by performance.
它训练多个模型，通过交叉验证评估并按性能排名。

---

## Basic Usage / 基本用法

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    lags=12,
    quantile=0.9,
)

# Train all default models
# 训练所有默认模型
leaderboard = pipeline.fit(data)

# Predict using the best model
# 使用最佳模型预测
result = pipeline.predict(10)
```

---

## Key Parameters / 关键参数

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 描述 |
|---|---|---|---|
| `time_col` | str | — | Time column name / 时间列名 |
| `target_col` | str | — | Target column name / 目标列名 |
| `lags` | int | — | Input window size / 输入窗口大小 |
| `quantile` | float or None | None | Prediction interval coverage / 预测区间覆盖率 |
| `random_state` | int | 0 | Random seed / 随机种子 |
| `metric` | callable | MAE | Evaluation metric function / 评估指标函数 |
| `metric_less_is_better` | bool | True | Whether lower metric is better / 指标是否越低越好 |
| `include_models` | str, list, or None | 'light' | Models to include / 要包含的模型 |
| `exclude_models` | list or None | None | Models to exclude / 要排除的模型 |
| `configs` | PipelineConfigs or None | None | Custom model configurations / 自定义模型配置 |
| `scaler` | bool, None, or transformer | True | Data scaler / 数据缩放器 |
| `accelerator` | str | 'auto' | Computing device / 计算设备 |
| `cv` | int | 5 | Cross-validation folds / 交叉验证折数 |
| `feature_cols` | list or None | None | Feature columns for multivariate / 多变量的特征列 |

---

## Model Filtering / 模型筛选

### Predefined Model Sets / 预定义模型集合

```python
# Lightweight models (default, fast) / 轻量级模型（默认，快速）
pipeline = ModelPipeline(..., include_models='light')

# All available models / 所有可用模型
pipeline = ModelPipeline(..., include_models='all')

# Only neural network models / 仅神经网络模型
pipeline = ModelPipeline(..., include_models='nn')

# Only machine learning models / 仅机器学习模型
pipeline = ModelPipeline(..., include_models='ml')
```

### Custom Model List / 自定义模型列表

```python
# Specify exact model names / 指定精确的模型名称
pipeline = ModelPipeline(
    ...,
    include_models=['torch_boosting_forest', 'torch_bagging_forest', 'd_linear', 'n_linear']
)

# Or exclude specific models / 或排除特定模型
pipeline = ModelPipeline(
    ...,
    exclude_models=['prophet', 'tft']
)
```

**Note**: `include_models` and `exclude_models` cannot be used simultaneously.
**注意**：`include_models` 和 `exclude_models` 不能同时使用。

### Use a Single Model Class / 使用单个模型类

You can pass a model class directly to `include_models`.
可以直接将模型类传递给 `include_models`。

```python
from PipelineTS.nn_model import TCNModel

pipeline = ModelPipeline(
    ...,
    include_models=TCNModel,
)
```

---

## PipelineConfigs / 管道配置

`PipelineConfigs` allows you to create multiple variants of the same model with different hyperparameters.
`PipelineConfigs` 允许你创建同一模型的多个变体，每个变体使用不同的超参数。

```python
from PipelineTS.pipeline import PipelineConfigs

configs = PipelineConfigs([
    # (model_name, custom_name, config_dict)
    # (模型名称, 自定义名称, 配置字典)
    ('torch_boosting_forest', 'boost_small', {
        'init_configs': {'n_trees': 32},
    }),
    ('torch_boosting_forest', 'boost_large', {
        'init_configs': {'n_trees': 128},
    }),

    # Without custom name: auto-numbered
    # 不指定自定义名称：自动编号
    ('torch_bagging_forest', {'init_configs': {'n_trees': 96}}),
])

pipeline = ModelPipeline(
    ...,
    configs=configs,
    include_init_config_model=False,  # Only use configured models
                                       # 仅使用已配置的模型
)
```

The config dict supports four keys:
配置字典支持四个键：

| Key / 键 | Description / 描述 |
|---|---|
| `init_configs` | Model initialization parameters / 模型初始化参数 |
| `fit_configs` | Parameters passed to `fit()` / 传递给 `fit()` 的参数 |
| `predict_configs` | Parameters passed to `predict()` / 传递给 `predict()` 的参数 |
| `pipeline_configs` | Pipeline-level per-model settings (lags, scaler, etc.) / 管道级别的每模型设置（滞后窗口、缩放器等） |

---

## Per-Model Pipeline Configuration / 每模型管道配置

`pipeline_configs` allows each model variant to use different pipeline-level settings such as lags, scalers, and differencing. This is useful when you want to compare models under different preprocessing conditions.

`pipeline_configs` 允许每个模型变体使用不同的管道级别设置，如滞后窗口、缩放器和差分。当你想在不同预处理条件下比较模型时非常有用。

### Supported Keys / 支持的键

| Key / 键 | Type / 类型 | Description / 描述 |
|---|---|---|
| `lags` | int | Per-model input window size (overrides global `lags`) / 每模型输入窗口大小（覆盖全局 `lags`） |
| `scaler` | bool, None, or TransformerMixin | Per-model scaler: `True`=MinMaxScaler, `None`=no scaling, or a custom scaler instance / 每模型缩放器：`True`=MinMaxScaler，`None`=不缩放，或自定义缩放器实例 |
| `differential_n` | int | Per-model differencing order (only for models that accept it) / 每模型差分阶数（仅适用于支持该参数的模型） |
| `feature_cols` | list | Per-model feature columns / 每模型特征列 |

### Example: Different Lags / 示例：不同滞后窗口

```python
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_short', {
        'init_configs': {'n_trees': 32},
        'pipeline_configs': {'lags': 6},
    }),
    ('torch_boosting_forest', 'boost_long', {
        'init_configs': {'n_trees': 32},
        'pipeline_configs': {'lags': 24},
    }),
])

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,  # Global default
    include_models=['torch_boosting_forest'],
    configs=configs,
)
leaderboard = pipeline.fit(data)
# boost_short uses lags=6, boost_long uses lags=24
# boost_short 使用 lags=6，boost_long 使用 lags=24
```

### Example: Different Scalers / 示例：不同缩放器

```python
from sklearn.preprocessing import StandardScaler
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_standard', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'scaler': StandardScaler()},
    }),
    ('torch_boosting_forest', 'boost_noscale', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'scaler': None},  # No scaling
    }),
])

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models=['torch_boosting_forest'],
    configs=configs,
    scaler=True,  # Global default: MinMaxScaler
)
leaderboard = pipeline.fit(data)
# boost_standard uses StandardScaler, boost_noscale uses no scaler
# boost_standard 使用 StandardScaler，boost_noscale 不使用缩放器
```

### Example: Combined Settings / 示例：组合设置

```python
from sklearn.preprocessing import StandardScaler
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_custom', {
        'init_configs': {'n_trees': 128},
        'pipeline_configs': {
            'lags': 20,
            'scaler': StandardScaler(),
        },
    }),
    ('torch_boosting_forest', 'boost_default', {
        'init_configs': {'n_trees': 64},
        # No pipeline_configs: uses global lags and scaler
        # 不指定 pipeline_configs：使用全局 lags 和 scaler
    }),
])

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models=['torch_boosting_forest'],
    configs=configs,
)
leaderboard = pipeline.fit(data)
```

**Priority order / 优先级顺序**: `pipeline_configs` overrides global settings, and `init_configs` has the highest priority for model initialization parameters.

**优先级顺序**：`pipeline_configs` 覆盖全局设置，而 `init_configs` 对模型初始化参数具有最高优先级。

---

## Double-underscore Syntax / 双下划线语法

You can pass model-specific initialization parameters directly to `ModelPipeline` using double-underscore syntax.
可以使用双下划线语法直接向 `ModelPipeline` 传递模型特定的初始化参数。

```python
pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    lags=12,
    include_models=['torch_boosting_forest', 'torch_bagging_forest', 'd_linear'],

    # Model-specific parameters / 模型特定参数
    torch_boosting_forest__n_trees=64,
    torch_bagging_forest__n_trees=96,
    d_linear__lags=50,
)
```

When a parameter conflicts with a `ModelPipeline` keyword parameter, the double-underscore version takes priority.
当参数与 `ModelPipeline` 关键字参数冲突时，双下划线版本优先。

---

## Custom Scaler / 自定义缩放器

```python
from sklearn.preprocessing import StandardScaler
from PipelineTS.preprocessing import Scaler

# Use sklearn scaler / 使用 sklearn 缩放器
pipeline = ModelPipeline(..., scaler=StandardScaler())

# Use built-in Scaler / 使用内置缩放器
# Options: 'min_max', 'standard', 'quantile', 'gauss_rank'
# 选项：'min_max', 'standard', 'quantile', 'gauss_rank'
pipeline = ModelPipeline(..., scaler=Scaler('gauss_rank'))

# True = MinMaxScaler (default) / True = MinMaxScaler（默认）
pipeline = ModelPipeline(..., scaler=True)

# None = no scaling / None = 不缩放
pipeline = ModelPipeline(..., scaler=None)
```

---

## Custom Evaluation Metric / 自定义评估指标

```python
from PipelineTS.metrics import rmse, wmape

pipeline = ModelPipeline(
    ...,
    metric=rmse,                # Custom metric function / 自定义指标函数
    metric_less_is_better=True, # Lower is better / 越低越好
)
```

---

## Accessing Results / 获取结果

### Get the Best Model / 获取最佳模型

```python
best_model = pipeline.get_model()
```

### Get a Specific Model / 获取指定模型

```python
model = pipeline.get_model(model_name='torch_boosting_forest')
```

### Get Model Configurations / 获取模型配置

```python
configs = pipeline.get_model_all_configs()           # Best model / 最佳模型
configs = pipeline.get_model_all_configs('torch_boosting_forest')  # Specific model / 指定模型
```

### Predict with a Specific Model / 使用指定模型预测

```python
result = pipeline.predict(10, model_name='torch_boosting_forest')
```

### Predict from a Specific Series / 从指定序列预测

```python
# Predict starting from a specific data segment
# 从指定数据段开始预测
result = pipeline.predict(10, data=some_data_segment)
```

---

## Training with Validation Data / 使用验证数据训练

You can provide explicit validation data for the leaderboard evaluation.
可以提供显式的验证数据用于排行榜评估。

```python
valid_data = data.iloc[-30:, :]
train_data = data.iloc[:-30, :]

pipeline.fit(train_data, valid_data=valid_data)
```

---

## Multi-Series (Panel Data) / 多序列（面板数据）

Use `id_col` to train on multiple time series simultaneously. Each series gets its own scaler.

使用 `id_col` 同时在多条时间序列上训练。每条序列拥有独立的缩放器。

```python
pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    lags=12,
    id_col='store_id',
    include_models=['torch_boosting_forest', 'torch_bagging_forest'],
)
pipeline.fit(panel_data)

# Returns DataFrame with store_id column / 返回带 store_id 列的 DataFrame
result = pipeline.predict(n=10)
```

---

## Covariate Support / 协变量支持

Pass known future covariates and past covariates to GBDT, Prophet, and AutoARIMA models.

向 GBDT、Prophet 和 AutoARIMA 模型传递已知未来协变量和历史协变量。

```python
pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    known_covariates=['holiday', 'promotion'],
    past_covariates=['temperature'],
    include_models=['torch_boosting_forest', 'prophet'],
)
pipeline.fit(data)

import pandas as pd
future_cov = pd.DataFrame({
    'holiday': [0, 0, 1, 0, 0],
    'promotion': [1, 0, 0, 0, 0],
})
result = pipeline.predict(n=5, future_covariates=future_cov)
```

---

## Incremental Learning / 增量学习

Use `update()` to incrementally train on new data without full retraining.

使用 `update()` 在新数据上进行增量训练，无需完全重新训练。

- **Neural networks**: Warm-start with fewer epochs on combined data.
- **神经网络**：在合并数据上以更少轮次热启动。

- **Other models**: Efficiently refitted on combined old + new data.
- **其他模型**：在合并的旧 + 新数据上高效重新拟合。

```python
pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models=['torch_boosting_forest'],
)
pipeline.fit(initial_data)

# When new data arrives / 当新数据到达时
pipeline.update(new_data)
result = pipeline.predict(10)
```

**Note**: `update()` raises `ValueError` if the pipeline has not been fitted yet.
**注意**：如果管道尚未拟合，`update()` 会抛出 `ValueError`。

---

## Multi-Quantile Prediction / 多分位数预测

Output prediction intervals at multiple coverage levels simultaneously.

同时输出多个覆盖水平的预测区间。

```python
pipeline.fit(data)

result = pipeline.predict_quantiles(n=10, levels=[0.5, 0.8, 0.95])
# Columns: date, value, value_q0.5_lower, value_q0.5_upper, ...
```

---

## Visualization / 可视化

Pipeline provides built-in `plot()` and `plot_leaderboard()` methods with Chinese font support.

管道提供内置 `plot()` 和 `plot_leaderboard()` 方法，支持中文字体。

```python
# Forecast plot (best model) / 预测图（最佳模型）
pipeline.plot(n=12, lang='zh')

# Use specific model / 使用指定模型
pipeline.plot(n=12, model_name='torch_boosting_forest', history_tail=60, lang='en')

# Leaderboard chart / 排行榜图
pipeline.plot_leaderboard(lang='zh')
```

For more visualization functions, see [Visualization / 可视化](visualization.md).

更多可视化函数，请参阅 [可视化](visualization.md)。

---

## Save and Load Pipeline / 保存与加载管道

```python
from PipelineTS.io import save_model, load_model

# Save the entire pipeline (all trained models)
# 保存整个管道（所有已训练模型）
save_model('pipeline.zip', pipeline)

# Load the pipeline
# 加载管道
loaded_pipeline = load_model('pipeline.zip')
result = loaded_pipeline.predict(10)
```
