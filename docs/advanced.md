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

## SmartRouter / 智能路由器

`SmartRouter` is an intelligent routing system that automatically analyzes time series data characteristics and makes optimal decisions for preprocessing, model selection, lag window size, feature engineering, and hyperparameters. It also supports automatic weighted ensemble of top-performing models.

`SmartRouter` 是一个智能路由系统，自动分析时间序列数据特征，为预处理、模型选择、滞后窗口大小、特征工程和超参数做出最优决策。它还支持顶级模型的自动加权集成。

### How SmartRouter Works / SmartRouter 工作原理

1. **Data Profiling**: Analyzes data for stationarity, seasonality, trend, noise, autocorrelation, multi-seasonality, and regime changes
2. **Strategy Building**: Selects preprocessing steps, models, lags, scaler, and differencing order based on data profile
3. **Feature Engineering Routing**: Decides whether to enable adaptive MoE routing for NN models and Prophet lag features
4. **Adaptive Hyperparameters**: Adjusts model parameters (GBDT n_estimators/learning_rate/max_depth, NN routing_mode) based on data
5. **Pipeline Execution**: Trains selected models and generates leaderboard
6. **Ensemble Building**: Optionally builds weighted ensemble from top-K models

### Basic Usage / 基本用法

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col='date',
    target_col='value',
    n_predict=12,
    max_models=5,
    ensemble_strategy='auto',
    verbose=True,
)

router.fit(data)
result = router.predict(12)
```

### Ensemble Strategies / 集成策略

```python
# 'auto' mode: builds ensemble only when top models are competitive (within 30% of best)
# 'auto' 模式：仅在顶级模型具有竞争力时构建集成（在最佳模型的 30% 范围内）
router = SmartRouter(..., ensemble_strategy='auto', ensemble_top_k=3)

# 'weighted_avg' mode: always builds ensemble
# 'weighted_avg' 模式：始终构建集成
router = SmartRouter(..., ensemble_strategy='weighted_avg', ensemble_top_k=3)

# 'none' mode: disables ensemble
# 'none' 模式：禁用集成
router = SmartRouter(..., ensemble_strategy='none')

# Predict with ensemble (default) or force single model
# 使用集成（默认）或强制单模型预测
result = router.predict(12)                # Uses ensemble if available
result = router.predict(12, use_ensemble=False)  # Force best single model
```

### Accessing Results and Strategy / 访问结果和策略

```python
# Full strategy dict with all decisions
# 包含所有决策的完整策略字典
strategy = router.strategy
print(f"Selected models: {strategy['models']}")
print(f"Lags: {strategy['lags']}")
print(f"Scaler: {strategy['scaler']}")
print(f"Feature engineering: {strategy['feature_engineering']}")
print(f"Hyperparameters: {strategy['model_hyperparams']}")

# Leaderboard with model rankings
# 模型排名的排行榜
print(router.leader_board_)

# Ensemble information (if built)
# 集成信息（如果已构建）
if router.ensemble_:
    print(router.ensemble_)
    print(router.ensemble_.all_configs)

# Get specific fitted model
# 获取特定的拟合模型
model = router.get_model('lightgbm')
```

### Data Profile Details / 数据画像详情

The `DataProfile` object contains comprehensive data characteristics:

```python
from PipelineTS.pipeline import SmartRouter, DataProfile

router = SmartRouter(time_col='date', target_col='value')
profile = router._profile_data(data)

print(f"Rows: {profile.n_rows}")
print(f"Frequency: {profile.freq}")
print(f"Stationarity: {profile.stationarity}")
print(f"Trend strength: {profile.trend_strength:.3f}")
print(f"Seasonality strength: {profile.seasonality_strength:.3f}")
print(f"Autocorr lag-1: {profile.autocorr_lag1:.3f}")
print(f"Number of seasonalities: {profile.n_seasonalities}")
print(f"Regime changes: {profile.regime_changes}")
print(f"Noise ratio: {profile.noise_ratio:.3f}")
print(f"Skewness: {profile.skewness:.3f}")
print(f"Missing: {profile.pct_missing:.2%}")
print(f"Outliers: {profile.pct_outlier:.2%}")
```

### Model Scoring Factors / 模型评分因素

SmartRouter scores each model based on:

| Factor / 因素 | Impact / 影响 |
|---|---|
| **Series length** / 序列长度 | Small data favors statistical + ML; large data favors NN / 小数据偏好统计+ML；大数据偏好NN |
| **Stationarity** / 平稳性 | Non-stationary data favors Prophet, ARIMA, DLinear, GBDT with differencing / 非平稳数据偏好 Prophet、ARIMA、DLinear、带差分的 GBDT |
| **Seasonality** / 季节性 | Strong seasonality favors Prophet, NBeats, NHiTS, TFT / 强季节性偏好 Prophet、NBeats、NHiTS、TFT |
| **Trend strength** / 趋势强度 | Strong trend favors Prophet, DLinear, NLinear, TiDE / 强趋势偏好 Prophet、DLinear、NLinear、TiDE |
| **Autocorrelation** / 自相关 | High autocorr favors ARIMA, RNN, TCN; low favors tree models / 高自相关偏好 ARIMA、RNN、TCN；低偏好树模型 |
| **Multi-seasonality** / 多季节性 | Multiple periods favor Prophet, TFT, NBeats / 多周期偏好 Prophet、TFT、NBeats |
| **Forecast horizon** / 预测范围 | Long horizon favors extrapolation models; short allows complex models / 长范围偏好外推模型；短范围允许复杂模型 |
| **Regime changes** / 机制变化 | Many changes favor tree models; penalizes smooth models / 多变化偏好树模型；惩罚平滑模型 |
| **Noise level** / 噪声水平 | High noise favors regularized GBDT (LightGBM, XGBoost) / 高噪声偏好正则化 GBDT |

### Customizing SmartRouter / 自定义 SmartRouter

While SmartRouter makes automatic decisions, you can influence its behavior:

```python
# Force specific models by using max_models with a list
# 通过使用 max_models 和列表强制特定模型
from PipelineTS.pipeline import ModelPipeline

# Custom pipeline with SmartRouter-selected lags but specific models
# 使用 SmartRouter 选择的滞后窗口但指定模型的自定义管道
router = SmartRouter(time_col='date', target_col='value')
router.fit(data)

# Use SmartRouter's lag selection with custom ModelPipeline
# 将 SmartRouter 的滞后选择用于自定义 ModelPipeline
custom_pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    lags=router.strategy['lags'],  # Use SmartRouter's suggested lags
    include_models=['lightgbm', 'prophet'],
)
```

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
