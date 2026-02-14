# Training Utilities
# 训练工具

PipelineTS provides built-in hyperparameter tuning and ensemble methods to improve forecasting performance.
PipelineTS 提供内置的超参数调优和集成方法，以提升预测性能。

---

## AutoTune / 自动调参

`AutoTune` wraps any PipelineTS model class for hyperparameter optimization. It uses Optuna (TPE sampler) when available, and falls back to random search otherwise.
`AutoTune` 封装任意 PipelineTS 模型类进行超参数优化。当 Optuna 可用时使用 TPE 采样器，否则回退到随机搜索。

```python
from PipelineTS.training import AutoTune
from PipelineTS.ml_model import TorchBoostingForestModel
from PipelineTS.spinesTS.metrics import mae

tuner = AutoTune(
    model_class=TorchBoostingForestModel,
    time_col='date',
    target_col='value',
    lags=12,
    metric=mae,
    metric_less_is_better=True,
    n_trials=30,
    test_size=0.2,                 # Last 20% for validation / 最后 20% 作为验证集
    fixed_params={'verbose': -1},  # Params not tuned / 不调优的参数
    random_state=42,
)

best_model, best_params, history = tuner.fit(data, search_space={
    'n_estimators': ('int', 50, 500),
    'learning_rate': ('float', 0.01, 0.3, True),  # True = log scale / True = 对数刻度
    'max_depth': ('int', 3, 10),
    'num_leaves': ('int', 15, 63),
})

# best_model: fitted model with optimal hyperparameters / 使用最优超参数训练的模型
# best_params: dict of best hyperparameters / 最优超参数字典
# history: DataFrame with all trial results / 包含所有试验结果的 DataFrame
```

### Search Space Format / 搜索空间格式

```python
search_space = {
    'param_name': ('int', low, high),                 # Integer range / 整数范围
    'param_name': ('float', low, high),               # Float range / 浮点数范围
    'param_name': ('float', low, high, True),         # Float log-scale / 浮点数对数刻度
    'param_name': ('categorical', ['a', 'b', 'c']),   # Categorical / 分类
}
```

### Tuning NN Models / 调优神经网络模型

```python
from PipelineTS.nn_model import TCNModel

tuner = AutoTune(
    model_class=TCNModel,
    time_col='date', target_col='value', lags=12,
    metric=mae, n_trials=20,
    fixed_params={'verbose': False, 'patience': 50, 'quantile': None},
)

best_model, best_params, history = tuner.fit(data, search_space={
    'epochs': ('int', 100, 500),
    'learning_rate': ('float', 1e-4, 1e-2, True),
    'kernel_size': ('categorical', [3, 5, 7]),
})
```

### Parameters / 参数

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 描述 |
|---|---|---|---|
| `model_class` | class | required | PipelineTS model class / 模型类 |
| `time_col` | str | required | Time column name / 时间列名 |
| `target_col` | str | required | Target column name / 目标列名 |
| `lags` | int | required | Number of lag steps / 滞后步数 |
| `metric` | callable | required | `metric(y_true, y_pred) -> float` |
| `metric_less_is_better` | bool | `True` | Lower is better? / 越低越好？ |
| `n_trials` | int | `20` | Number of tuning trials / 调优试验次数 |
| `test_size` | float or int | `0.2` | Validation set size / 验证集大小 |
| `fixed_params` | dict or None | `None` | Non-tuned parameters / 不调优的参数 |
| `random_state` | int | `0` | Random seed / 随机种子 |

### Optional Dependency / 可选依赖

```bash
pip install optuna  # Recommended for TPE sampling / 推荐安装以使用 TPE 采样
```

Without Optuna, `AutoTune` uses random search (no installation needed).
未安装 Optuna 时，`AutoTune` 使用随机搜索（无需额外安装）。

---

## Weighted Ensemble / 加权集成

`WeightedEnsemble` combines predictions from multiple models using weighted averaging.
`WeightedEnsemble` 使用加权平均组合多个模型的预测。

```python
from PipelineTS.training import WeightedEnsemble
from PipelineTS.ml_model import TorchBoostingForestModel, TorchBaggingForestModel

models = [
    ('boost', TorchBoostingForestModel(time_col='date', target_col='value', lags=12)),
    ('bag',   TorchBaggingForestModel(time_col='date', target_col='value', lags=12)),
]

# Auto weights: inverse-error weighting from validation set
# 自动权重：基于验证集的逆误差加权
ens = WeightedEnsemble(models, time_col='date', target_col='value', weights='auto')
ens.fit(data)

# Or manual weights / 或手动权重
ens = WeightedEnsemble(models, time_col='date', target_col='value', weights=[0.5, 0.3, 0.2])
ens.fit(data)

result = ens.predict(10)  # Weighted average prediction / 加权平均预测
print(ens.get_weights())  # {'lgbm': 0.45, 'xgb': 0.32, 'cat': 0.23}
```

**Auto weight computation / 自动权重计算:**

When `weights='auto'`, the ensemble:
当 `weights='auto'` 时：

1. Fits all models on training data. / 在训练数据上训练所有模型。
2. Predicts on a validation set (last 20% of data by default). / 在验证集上预测（默认为数据的最后 20%）。
3. Computes inverse-error weights: `w_i = 1/err_i`, then normalizes. / 计算逆误差权重，然后归一化。

### Parameters / 参数

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `models` | list of (name, model) | Named model instances / 命名的模型实例 |
| `time_col` | str | Time column name / 时间列名 |
| `target_col` | str | Target column name / 目标列名 |
| `weights` | list or `'auto'` | Manual weights or auto-compute / 手动权重或自动计算 |
| `metric` | callable or None | Metric for auto weights (default: MAE) / 自动权重的指标 |

---

## Stacking Ensemble / 堆叠集成

`StackingEnsemble` trains a meta-learner (ridge regression) on base model predictions from temporal cross-validation.
`StackingEnsemble` 在基础模型的时序交叉验证预测上训练元学习器（岭回归）。

```python
from PipelineTS.training import StackingEnsemble
from PipelineTS.ml_model import TorchBoostingForestModel, TorchBaggingForestModel

models = [
    ('boost', TorchBoostingForestModel(time_col='date', target_col='value', lags=12)),
    ('bag',   TorchBaggingForestModel(time_col='date', target_col='value', lags=12)),
]

stack = StackingEnsemble(
    models,
    time_col='date',
    target_col='value',
    n_folds=3,  # Temporal CV folds for meta-feature generation / 生成元特征的时序交叉验证折数
)
stack.fit(data)
result = stack.predict(10)
```

**How it works / 工作原理:**

1. **Meta-feature generation / 元特征生成**: Temporal CV produces out-of-fold predictions from each base model. / 时序交叉验证从每个基模型生成折外预测。
2. **Meta-learner training / 元学习器训练**: Ridge regression is trained on stacked base model predictions vs. actual values. / 在堆叠的基模型预测与真实值上训练岭回归。
3. **Final prediction / 最终预测**: All base models are re-trained on full data; their predictions are combined by the meta-learner. / 所有基模型在全量数据上重新训练；元学习器组合其预测。

### Parameters / 参数

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `models` | list of (name, model) | Named model instances / 命名的模型实例 |
| `time_col` | str | Time column name / 时间列名 |
| `target_col` | str | Target column name / 目标列名 |
| `n_folds` | int | Number of temporal CV folds (default: 3) / 时序交叉验证折数 |
