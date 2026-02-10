# Quick Start Guide
# 快速入门指南

This guide will walk you through the core workflow of PipelineTS in just a few steps.
本指南将通过几个简单步骤带你了解 PipelineTS 的核心工作流。

---

## Step 1: Load Data / 第一步：加载数据

PipelineTS provides several built-in datasets for quick experimentation.
PipelineTS 提供了多个内置数据集，方便快速实验。

```python
from PipelineTS.dataset import LoadElectricDataSets
import pandas as pd

# Load the Electric Production dataset
# 加载电力生产数据集
data = LoadElectricDataSets()
time_col = 'date'
target_col = 'value'
data[time_col] = pd.to_datetime(data[time_col])

print(f"Data shape: {data.shape}")
print(f"Time range: {data[time_col].min()} ~ {data[time_col].max()}")
```

You can also load your own data as a pandas DataFrame:
你也可以加载自己的数据作为 pandas DataFrame：

```python
data = pd.read_csv('your_data.csv')
data['date'] = pd.to_datetime(data['date'])
```

**Important**: The time column must be in `datetime64[ns]` format.
**重要**：时间列必须为 `datetime64[ns]` 格式。

---

## Step 2: Train a Single Model / 第二步：训练单个模型

All models share the same `fit()` / `predict()` interface.
所有模型共享相同的 `fit()` / `predict()` 接口。

```python
from PipelineTS.ml_model import LightGBMModel

model = LightGBMModel(
    time_col=time_col,
    target_col=target_col,
    lags=12,           # Use 12 past time steps as features
                       # 使用过去 12 个时间步作为特征
    quantile=0.9,      # 90% prediction interval
                       # 90% 预测区间
    verbose=-1
)

# Train the model
# 训练模型
model.fit(data)

# Predict the next 10 steps
# 预测未来 10 个时间步
result = model.predict(10)
print(result)
```

The result is a DataFrame containing:
结果是一个 DataFrame，包含：

- `date`: Predicted timestamps / 预测的时间戳
- `value`: Point predictions / 点预测值
- `value_lower`: Lower bound of the prediction interval (when `quantile` is set) / 预测区间下界（设置了 `quantile` 时）
- `value_upper`: Upper bound of the prediction interval (when `quantile` is set) / 预测区间上界（设置了 `quantile` 时）

---

## Step 3: Visualize Results / 第三步：可视化结果

```python
from PipelineTS.plot import plot_data_period

plot_data_period(
    data,
    result,
    time_col=time_col,
    target_col=target_col
)
```

---

## Step 4: Use ModelPipeline for Auto Selection / 第四步：使用 ModelPipeline 自动选择模型

`ModelPipeline` automatically trains multiple models and selects the best one.
`ModelPipeline` 自动训练多个模型并选出最佳模型。

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col=time_col,
    target_col=target_col,
    lags=12,
    quantile=0.9,
    include_models='ml',  # 'light', 'all', 'nn', 'ml', or a list
                          # 'light', 'all', 'nn', 'ml', 或模型名称列表
    cv=3,                 # Cross-validation folds for interval estimation
                          # 用于区间估计的交叉验证折数
)

# Train all models and get leaderboard
# 训练所有模型并获取排行榜
leaderboard = pipeline.fit(data)
print(leaderboard)

# Predict using the best model
# 使用最佳模型进行预测
result = pipeline.predict(10)
```

You can also predict using a specific model:
你也可以使用指定的模型进行预测：

```python
result = pipeline.predict(10, model_name='xgboost')
```

---

## Step 5: Save and Load / 第五步：保存与加载

```python
from PipelineTS.io import save_model, load_model

# Save the pipeline or a single model
# 保存管道或单个模型
save_model('my_pipeline.zip', pipeline)

# Load it back
# 重新加载
loaded_pipeline = load_model('my_pipeline.zip')
result = loaded_pipeline.predict(10)
```

---

## Next Steps / 下一步

- Learn about all 24 models: [Model Reference](models.md)
- 了解所有 24 个模型：[模型参考](models.md)

- Master the pipeline: [Pipeline Usage](pipeline.md)
- 掌握管道使用：[管道使用](pipeline.md)

- Explore multivariate prediction: [Multivariate Prediction](multivariate.md)
- 探索多变量预测：[多变量预测](multivariate.md)

- Advanced features: [Advanced Features](advanced.md)
- 高级功能：[高级功能](advanced.md)
