# Advanced Features
# 高级功能

This guide covers advanced usage patterns including hyperparameter tuning, differencing, custom estimators, and more.
本指南涵盖高级用法，包括超参数调优、差分、自定义估计器等。

---

## Hyperparameter Tuning with Optuna / 使用 Optuna 进行超参数调优

PipelineTS integrates seamlessly with Optuna for hyperparameter optimization.
PipelineTS 与 Optuna 无缝集成，支持超参数优化。

### Installation / 安装

```bash
pip install optuna
```

### Example / 示例

```python
import optuna
from sklearn.metrics import mean_absolute_error
from PipelineTS.pipeline import ModelPipeline
from PipelineTS.ml_model import WideGBRTModel

def objective(trial):
    # Suggest hyperparameters / 建议超参数
    lags = trial.suggest_int('lags', 8, 60, step=2)
    n_estimators = trial.suggest_int('n_estimators', 100, 500, log=True)
    differential_n = trial.suggest_int('differential_n', 1, 3)

    pipeline = ModelPipeline(
        time_col='date',
        target_col='value',
        lags=lags,
        random_state=42,
        include_models=WideGBRTModel,
        metric=mean_absolute_error,
        metric_less_is_better=True,
        scaler=None,
        WideGBRTModel__n_estimators=n_estimators,
        WideGBRTModel__differential_n=differential_n,
    )

    pipeline.fit(train_data, valid_data=valid_data)
    prediction = pipeline.predict(n)
    return mean_absolute_error(
        valid_data['value'].values,
        prediction['value'].values
    )

# Run optimization / 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Best parameters / 最佳参数
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

### Tuning NN Models / 调优神经网络模型

```python
def objective_nn(trial):
    lags = trial.suggest_int('lags', 8, 48, step=4)
    epochs = trial.suggest_int('epochs', 100, 1000, step=100)
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    from PipelineTS.nn_model import TCNModel
    model = TCNModel(
        time_col='date', target_col='value',
        lags=lags, epochs=epochs, learning_rate=learning_rate,
        patience=50, verbose=False
    )
    model.fit(train_data)
    prediction = model.predict(n)
    return mean_absolute_error(
        valid_data['value'].values,
        prediction['value'].values
    )
```

---

## Differencing / 差分

Differencing removes trends from the time series, which can improve prediction accuracy for non-stationary data.
差分可以去除时间序列中的趋势项，对非平稳数据可提高预测精度。

### WideGBRTModel with Differencing / WideGBRTModel 差分

```python
from PipelineTS.ml_model import WideGBRTModel

# differential_n=0: No differencing / 不使用差分
# differential_n=1: First-order differencing (default) / 一阶差分（默认）
# differential_n=2: Second-order differencing / 二阶差分

model = WideGBRTModel(
    time_col='date', target_col='value', lags=12,
    differential_n=1,
    verbose=-1
)
model.fit(data)
result = model.predict(10)
```

Higher-order differencing can help with quadratic trends but may introduce noise.
更高阶的差分可以处理二次趋势，但可能引入噪声。

---

## Custom Estimators / 自定义估计器

Multi-output regression models support custom base estimators.
多输出回归模型支持自定义基础估计器。

```python
from PipelineTS.ml_model import MultiOutputRegressorModel
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Use XGBoost as base estimator / 使用 XGBoost 作为基础估计器
model = MultiOutputRegressorModel(
    time_col='date', target_col='value', lags=12,
    estimator=XGBRegressor,
    kwargs={'verbosity': 0},
)

# Use CatBoost as base estimator / 使用 CatBoost 作为基础估计器
model = MultiOutputRegressorModel(
    time_col='date', target_col='value', lags=12,
    estimator=CatBoostRegressor,
    verbose=False,
)
```

The same works for `WideGBRTModel`:
`WideGBRTModel` 也同样适用：

```python
from PipelineTS.ml_model import WideGBRTModel
from xgboost import XGBRegressor

model = WideGBRTModel(
    time_col='date', target_col='value', lags=12,
    estimator=XGBRegressor,
    verbose=0,
)
```

---

## Prediction Intervals Deep Dive / 预测区间详解

### Conformal Prediction (ML/Statistical Models) / 保形预测（ML/统计模型）

All ML and statistical models use Conformal Prediction for interval estimation:
所有 ML 和统计模型使用保形预测进行区间估计：

1. During cross-validation, per-point signed residuals `(y_true - y_pred)` are collected.
1. 在交叉验证期间，收集逐点有符号残差 `(y_true - y_pred)`。

2. Asymmetric conformal quantiles are computed with finite-sample correction.
2. 使用有限样本校正计算非对称保形分位数。

3. Intervals are additive: `pred + q_lower`, `pred + q_upper`.
3. 区间是加法形式：`pred + q_lower`, `pred + q_upper`。

This provides distribution-free marginal coverage guarantee.
这提供了无分布假设的边际覆盖率保证。

### CQR (Neural Network Models) / CQR（神经网络模型）

Neural network models use Conformalized Quantile Regression (CQR):
神经网络模型使用保形分位数回归（CQR）：

1. The model is wrapped with a CQR head that outputs lower, median, and upper quantiles.
1. 模型被包装一个 CQR 头部，输出下分位、中位和上分位。

2. During CV, nonconformity scores are computed: `E_i = max(q_lo - y, y - q_hi)`.
2. 在交叉验证期间，计算不一致性分数：`E_i = max(q_lo - y, y - q_hi)`。

3. A conformal quantile Q_hat is computed from these scores.
3. 从这些分数计算保形分位数 Q_hat。

4. Final intervals: `q_lower(x) - Q_hat`, `q_upper(x) + Q_hat`.
4. 最终区间：`q_lower(x) - Q_hat`, `q_upper(x) + Q_hat`。

CQR provides adaptive intervals that are wider where the model is uncertain.
CQR 提供自适应区间，在模型不确定的区域区间更宽。

---

## GlobalTemporalBlock (GTB) / 全局时序块

GlobalTemporalBlock is an optional plug-in module available for all 12 univariate NN models. It combines three expert components with residual connections and RevIN normalization.
GlobalTemporalBlock 是所有 12 个单变量 NN 模型的可选插件模块。它组合三个专家组件，带残差连接和 RevIN 归一化。

### Three Expert Components / 三个专家组件

| Expert / 专家 | Description / 描述 |
|---|---|
| **FreqMixingBlock** | Frequency-domain mixing via FFT → learnable complex weights → iFFT / 通过 FFT → 可学习复数权重 → iFFT 的频域混合 |
| **GatedLinearAttention** | Efficient gated linear attention (no softmax) / 高效门控线性注意力（无 softmax） |
| **SwiGLU** | SwiGLU feed-forward network / SwiGLU 前馈网络 |

### Static Routing Mode / 静态路由模式

In static mode (default), all three experts are always active:
在静态模式（默认）下，三个专家始终全部激活：

```python
from PipelineTS.nn_model import DLinearModel

model = DLinearModel(
    time_col='date', target_col='value', lags=12,
    use_gtb=True,              # Enable GTB / 启用 GTB
    gtb_d_model=64,            # GTB hidden dimension / GTB 隐藏维度
    routing_mode='static',     # Default: all experts active / 默认：所有专家激活
    quantile=0.9, epochs=50,
)
model.fit(data)
```

### Adaptive MoE Routing Mode / 自适应 MoE 路由模式

In adaptive mode, a lightweight router network dynamically selects top-K experts per sample (inspired by DeepSeek-V2 / Switch Transformer):
在自适应模式下，轻量级路由网络动态选择每个样本的 top-K 专家（灵感来自 DeepSeek-V2 / Switch Transformer）：

```python
model = DLinearModel(
    time_col='date', target_col='value', lags=12,
    use_gtb=True,
    gtb_d_model=64,
    routing_mode='adaptive',   # MoE routing: top-2 of 3 experts per sample
                               # MoE 路由：每个样本从 3 个专家中选 2 个
    quantile=0.9, epochs=50,
)
model.fit(data)
```

Key features of adaptive routing:
自适应路由的关键特性：

- **Sparse top-K gating**: Only K experts (default 2 of 3) are activated per sample, reducing computation.
- **稀疏 top-K 门控**：每个样本仅激活 K 个专家（默认 3 选 2），减少计算量。

- **Load-balancing loss**: An auxiliary loss `L_balance = n · Σ(f_i · P_i)` prevents routing collapse and encourages balanced expert usage. It is automatically added to the training loss.
- **负载均衡损失**：辅助损失 `L_balance = n · Σ(f_i · P_i)` 防止路由崩塌，鼓励均衡使用专家。自动加入训练损失。

- **Exploration noise**: Gaussian noise on router logits during training for better exploration.
- **探索噪声**：训练时在路由 logits 上注入高斯噪声，促进更好的探索。

### GTB via ModelPipeline / 通过 ModelPipeline 使用 GTB

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models='nn',
    # Enable GTB with adaptive routing for all NN models
    # 为所有 NN 模型启用 GTB 自适应路由
    d_linear__use_gtb=True,
    d_linear__routing_mode='adaptive',
    tcn__use_gtb=True,
    tcn__routing_mode='adaptive',
)
pipeline.fit(data)
```

### Supported Models / 支持的模型

GTB is available for all 12 univariate NN models: DLinear, NLinear, NBeats, NHiTS, TFT, Transformer, TiDE, GAU, StackingRNN, Time2Vec, PatchRNN, TCN.
GTB 可用于所有 12 个单变量 NN 模型：DLinear、NLinear、NBeats、NHiTS、TFT、Transformer、TiDE、GAU、StackingRNN、Time2Vec、PatchRNN、TCN。

---

## Computing Backends / 计算后端

Neural network models support multiple computing backends:
神经网络模型支持多种计算后端：

```python
from PipelineTS.pipeline import ModelPipeline

# Auto-detect best available backend / 自动检测最佳可用后端
pipeline = ModelPipeline(..., accelerator='auto')

# Force CPU / 强制使用 CPU
pipeline = ModelPipeline(..., accelerator='cpu')

# Override per model / 对单个模型覆盖
pipeline = ModelPipeline(
    ...,
    accelerator='auto',
    n_hits__accelerator='cpu',  # Force CPU for NHits / 对 NHits 强制使用 CPU
    tft__accelerator='cpu',     # Force CPU for TFT / 对 TFT 强制使用 CPU
)
```

---

## Prophet Advanced Features / Prophet 高级功能

`ProphetModel` is a custom Prophet-like decomposable model (not Facebook Prophet). It uses piecewise linear trend + Fourier seasonality + ridge regression, 100x+ faster than Facebook Prophet.
`ProphetModel` 是自定义的类 Prophet 可分解模型（非 Facebook Prophet）。使用分段线性趋势 + 傅里叶季节性 + 岭回归，比 Facebook Prophet 快 100 倍以上。

```python
from PipelineTS.statistic_model import ProphetModel

model = ProphetModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9,
    auto_seasonality=True,       # Auto-detect seasonality / 自动检测季节性
    use_lag_features=True,       # Enable rolling lag features / 启用滚动滞后特征
    lag_window='auto',           # Auto-determine window size / 自动确定窗口大小
    lag_prior_scale=5.0,         # Regularization for lag features / 滞后特征的正则化
)
model.fit(data, cv=3)
result = model.predict(10)
```

The lag features include 7 causal rolling statistics:
滞后特征包括 7 个因果滚动统计量：

- rolling_mean, rolling_std, trend_slope, momentum, half_ratio, EMA, autocorr
- 滚动均值、滚动标准差、趋势斜率、动量、半比率、EMA、自相关

---

## Saving and Loading with Scaler / 带缩放器的保存与加载

When using manual scaling, you can save the scaler alongside the model.
当使用手动缩放时，可以将缩放器与模型一起保存。

```python
from PipelineTS.io import save_model, load_model
from PipelineTS.preprocessing import Scaler

# Save model with scaler / 保存模型和缩放器
save_model('model.zip', model, scaler=scaler)

# Load model and scaler / 加载模型和缩放器
model, scaler = load_model('model.zip')
```
