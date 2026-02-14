# Prediction Utilities
# 预测工具

PipelineTS provides rolling (sliding window) prediction and model explainability tools.
PipelineTS 提供滚动（滑动窗口）预测和模型可解释性工具。

---

## Rolling Prediction / 滚动预测

`RollingPredictor` re-fits the model on a sliding window of recent data, producing predictions that adapt to distribution shifts.
`RollingPredictor` 在最新数据的滑动窗口上重新训练模型，产生能适应分布变化的预测。

At each step: train on the most recent `train_size` observations → forecast `horizon` steps → advance the window by `step` observations.
每一步：在最近 `train_size` 个观测上训练 → 预测 `horizon` 步 → 窗口前进 `step` 个观测。

```python
from PipelineTS.prediction import RollingPredictor
from PipelineTS.ml_model import TorchBoostingForestModel

model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)

rp = RollingPredictor(
    model,
    time_col='date',
    target_col='value',
    train_size=100,   # Training window size / 训练窗口大小
    horizon=10,       # Forecast steps per window / 每个窗口的预测步数
    step=10,          # Window advance per iteration / 每次迭代窗口前进步数
    refit=True,       # Re-fit at each step (True) or fit once (False) / 每步重新训练或仅训练一次
)

# Run rolling prediction / 运行滚动预测
results = rp.predict(data, verbose=True)
# Returns DataFrame with: date, value (predicted), value_actual, window_id
# 返回 DataFrame，包含：date, value（预测值）, value_actual（真实值）, window_id
```

### Evaluation / 评估

```python
# Evaluate with default metrics (MAE, RMSE) / 使用默认指标（MAE、RMSE）评估
eval_results = rp.score(results)
print(f"Overall MAE:  {eval_results['MAE']['overall']:.4f}")
print(f"Overall RMSE: {eval_results['RMSE']['overall']:.4f}")

# Per-window metrics / 每个窗口的指标
print(f"Per-window MAE: {eval_results['MAE']['per_window']}")

# Custom metrics / 自定义指标
from PipelineTS.metrics import mape, r2_score
eval_results = rp.score(results, metrics={'MAPE': mape, 'R²': r2_score})
```

### Refit vs No-Refit / 重新训练 vs 不重新训练

| Mode / 模式 | `refit=True` | `refit=False` |
|---|---|---|
| Speed / 速度 | Slower (re-trains each window) / 较慢（每个窗口重新训练） | Fast (train once) / 较快（仅训练一次） |
| Adaptivity / 适应性 | Adapts to distribution shifts / 适应分布变化 | Fixed model / 固定模型 |
| Use case / 适用场景 | Non-stationary data / 非平稳数据 | Stable patterns / 稳定模式 |

### Parameters / 参数

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 描述 |
|---|---|---|---|
| `model` | PipelineTS model | required | Model instance (deep-copied per window) / 模型实例（每窗口深拷贝） |
| `time_col` | str | required | Time column name / 时间列名 |
| `target_col` | str | required | Target column name / 目标列名 |
| `train_size` | int | required | Training window size / 训练窗口大小 |
| `horizon` | int | required | Forecast steps per window / 每窗口预测步数 |
| `step` | int | `1` | Window advance / 窗口前进步数 |
| `refit` | bool | `True` | Re-fit model at each step / 每步重新训练 |

---

## Model Explainability / 模型可解释性

`ModelExplainer` extracts and visualizes feature importance from fitted PipelineTS models.
`ModelExplainer` 从已训练的 PipelineTS 模型中提取并可视化特征重要性。

### Native Feature Importance / 原生特征重要性

For GPU tree models (TorchBoostingForest, TorchBaggingForest, DeepForest), native feature importance is extracted from the underlying tree-based model.
对于 GPU 树模型（TorchBoostingForest、TorchBaggingForest、DeepForest），从底层树模型提取原生特征重要性。

```python
from PipelineTS.prediction import ModelExplainer
from PipelineTS.ml_model import TorchBoostingForestModel

# Fit a model first / 先训练模型
model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)
model.fit(data)

# Create explainer / 创建解释器
explainer = ModelExplainer(model, time_col='date', target_col='value')

# Get feature importance table / 获取特征重要性表
importance = explainer.feature_importance()
# Returns DataFrame with columns: feature, importance (sorted descending)
# 返回 DataFrame，列为：feature, importance（降序排列）
print(importance.head(10))
```

### Visualization / 可视化

```python
# Horizontal bar chart of top features / 顶部特征的水平柱状图
explainer.plot_importance(top_k=15, figsize=(10, 6))

# Plot from a custom importance DataFrame / 从自定义重要性 DataFrame 绘图
explainer.plot_importance(importance_df=importance, top_k=20)
```

### Supported Models / 支持的模型

| Model / 模型 | Native Importance / 原生重要性 | Notes / 备注 |
|---|---|---|
| TorchBoostingForestModel | No | Use permutation importance / 使用置换重要性 |
| TorchBaggingForestModel | No | Use permutation importance / 使用置换重要性 |
| DeepForestModel | No | Use permutation importance / 使用置换重要性 |
| WideGBRTModel | Yes | `feature_importances_` |
| NN Models / 神经网络模型 | No | Use permutation importance instead / 使用置换重要性替代 |
| Statistical Models / 统计模型 | No | Use residual analysis instead / 使用残差分析替代 |

### Parameters / 参数

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `model` | fitted PipelineTS model | A model that has been `fit()` / 已训练的模型 |
| `time_col` | str | Time column name / 时间列名 |
| `target_col` | str | Target column name / 目标列名 |
