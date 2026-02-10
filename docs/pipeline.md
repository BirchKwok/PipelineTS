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
    include_models=['lightgbm', 'xgboost', 'd_linear', 'n_linear']
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
    ('lightgbm', 'lgbm_small', {
        'init_configs': {'n_estimators': 50},
    }),
    ('lightgbm', 'lgbm_large', {
        'init_configs': {'n_estimators': 300},
    }),

    # Without custom name: auto-numbered
    # 不指定自定义名称：自动编号
    ('xgboost', {'init_configs': {'n_estimators': 200}}),
])

pipeline = ModelPipeline(
    ...,
    configs=configs,
    include_init_config_model=False,  # Only use configured models
                                       # 仅使用已配置的模型
)
```

The config dict supports three keys:
配置字典支持三个键：

| Key / 键 | Description / 描述 |
|---|---|
| `init_configs` | Model initialization parameters / 模型初始化参数 |
| `fit_configs` | Parameters passed to `fit()` / 传递给 `fit()` 的参数 |
| `predict_configs` | Parameters passed to `predict()` / 传递给 `predict()` 的参数 |

---

## Double-underscore Syntax / 双下划线语法

You can pass model-specific initialization parameters directly to `ModelPipeline` using double-underscore syntax.
可以使用双下划线语法直接向 `ModelPipeline` 传递模型特定的初始化参数。

```python
pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    lags=12,
    include_models=['lightgbm', 'xgboost', 'd_linear'],

    # Model-specific parameters / 模型特定参数
    lightgbm__n_estimators=200,
    lightgbm__verbose=-1,
    xgboost__n_estimators=150,
    xgboost__verbose=0,
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
from PipelineTS.spinesTS.metrics import rmse, wmape

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
model = pipeline.get_model(model_name='lightgbm')
```

### Get Model Configurations / 获取模型配置

```python
configs = pipeline.get_model_all_configs()           # Best model / 最佳模型
configs = pipeline.get_model_all_configs('xgboost')  # Specific model / 指定模型
```

### Predict with a Specific Model / 使用指定模型预测

```python
result = pipeline.predict(10, model_name='lightgbm')
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
