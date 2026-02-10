# Multivariate Prediction
# 多变量预测

PipelineTS supports multivariate time series forecasting through `ITransformerModel` and `SRSNetModel`.
PipelineTS 通过 `ITransformerModel` 和 `SRSNetModel` 支持多变量时间序列预测。

These models can leverage multiple input features to improve prediction accuracy, and can predict multiple target variables simultaneously.
这些模型可以利用多个输入特征提高预测精度，并能同时预测多个目标变量。

---

## Three Prediction Modes / 三种预测模式

| Mode / 模式 | `target_col` | `feature_cols` | Description / 描述 |
|---|---|---|---|
| Univariate / 单变量 | `'y'` | `None` | Classic single-variable prediction (default) / 经典单变量预测（默认） |
| Multi-input Single-output / 多输入单输出 | `'y'` | `['a', 'b', 'y']` | Multiple features predict one target / 多个特征预测单个目标 |
| Multi-input Multi-output / 多输入多输出 | `['a', 'b']` | `['a', 'b', 'c']` | Multiple features predict multiple targets / 多个特征预测多个目标 |

---

## Mode 1: Univariate (Default) / 模式一：单变量（默认）

When `feature_cols` is not specified, the model uses only the target column for prediction. This is the standard behavior.
当不指定 `feature_cols` 时，模型仅使用目标列进行预测。这是标准行为。

```python
from PipelineTS.nn_model import ITransformerModel

model = ITransformerModel(
    time_col='date',
    target_col='value',
    lags=12,
    d_model=32, n_heads=2, d_ff=64, e_layers=1,
    quantile=None, epochs=50, verbose=False
)
model.fit(data)
result = model.predict(10)
```

---

## Mode 2: Multi-input Single-output / 模式二：多输入单输出

Specify `feature_cols` to use multiple columns as input features while predicting a single target column.
指定 `feature_cols` 使用多个列作为输入特征，同时预测单个目标列。

```python
import numpy as np
import pandas as pd

# Prepare multivariate data / 准备多变量数据
np.random.seed(42)
n = 200
dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
data = pd.DataFrame({
    'date': dates,
    'value': np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1,
    'feature_a': np.cos(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1,
    'feature_b': np.sin(np.linspace(0, 2 * np.pi, n)) * 0.5 + np.random.randn(n) * 0.05,
})
```

```python
from PipelineTS.nn_model import ITransformerModel

model = ITransformerModel(
    time_col='date',
    target_col='value',                              # Single target / 单目标
    feature_cols=['value', 'feature_a', 'feature_b'], # Multiple inputs / 多输入
    lags=12,
    d_model=32, n_heads=2, d_ff=64, e_layers=1,
    quantile=None, epochs=50, verbose=False
)
model.fit(data)
result = model.predict(10)
# result contains: date, value
# result 包含：date, value
```

---

## Mode 3: Multi-input Multi-output / 模式三：多输入多输出

Pass a list of column names to `target_col` to predict multiple target variables simultaneously.
将列名列表传递给 `target_col`，同时预测多个目标变量。

```python
from PipelineTS.nn_model import SRSNetModel

model = SRSNetModel(
    time_col='date',
    target_col=['value', 'feature_a'],                # Multiple targets / 多目标
    feature_cols=['value', 'feature_a', 'feature_b'],  # Multiple inputs / 多输入
    lags=12,
    d_model=32, n_heads=2,
    quantile=None, epochs=50, verbose=False
)
model.fit(data)
result = model.predict(10)
# result contains: date, value, feature_a
# result 包含：date, value, feature_a
```

---

## Supported Models / 支持的模型

| Model / 模型 | Multi-input Single-output / 多输入单输出 | Multi-input Multi-output / 多输入多输出 |
|---|---|---|
| ITransformerModel | ✅ | ✅ |
| SRSNetModel | ✅ | ✅ |

**Note**: Other models (ML, statistical, and other NN models) currently support univariate mode only.
**注意**：其他模型（ML、统计和其他 NN 模型）目前仅支持单变量模式。

---

## ITransformerModel vs SRSNetModel / 两种模型的区别

| Feature / 特性 | ITransformerModel | SRSNetModel |
|---|---|---|
| Architecture / 架构 | Inverted Transformer / 反转 Transformer | Multi-scale adaptive patch + selective representation / 多尺度自适应 patch + 选择性表征 |
| Internal training / 内部训练 | Trains on all variates, extracts targets at prediction / 训练所有变量，预测时提取目标 | Trains directly on target outputs / 直接训练目标输出 |
| Best for / 适用于 | Datasets with strong inter-variable correlations / 变量间相关性强的数据集 | Datasets with complex temporal patterns / 时序模式复杂的数据集 |

---

## Using with ModelPipeline / 在 ModelPipeline 中使用

`ModelPipeline` supports the `feature_cols` parameter and automatically passes it to models that support multivariate prediction.
`ModelPipeline` 支持 `feature_cols` 参数，并自动传递给支持多变量预测的模型。

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    feature_cols=['value', 'feature_a', 'feature_b'],
    lags=12,
    include_models=['itransformer', 'srs_net', 'lightgbm'],
    quantile=None,
    cv=2,
)

leaderboard = pipeline.fit(data)
result = pipeline.predict(10)
```

**Note**: ML models in the pipeline will automatically ignore `feature_cols` and use univariate mode.
**注意**：管道中的 ML 模型会自动忽略 `feature_cols`，使用单变量模式。

---

## Tips / 使用提示

- Include the target column in `feature_cols` for best results.
- 在 `feature_cols` 中包含目标列以获得最佳效果。

- Ensure all `feature_cols` columns are numeric and have no missing values.
- 确保所有 `feature_cols` 列都是数值类型且没有缺失值。

- For multi-output mode, the prediction result will contain all target columns.
- 对于多输出模式，预测结果将包含所有目标列。

- Multivariate interval prediction (CQR) is only supported in univariate mode for NN models.
- 多变量区间预测（CQR）目前仅在 NN 模型的单变量模式下支持。
