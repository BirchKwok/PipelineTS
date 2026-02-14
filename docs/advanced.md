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
model = router.get_model('torch_boosting_forest')
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
| **Noise level** / 噪声水平 | High noise favors regularized GBDT (TorchBoostingForest) / 高噪声偏好正则化 GBDT |

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
    include_models=['torch_boosting_forest', 'prophet'],
)
```

---

## Computing Backends / 计算后端

Neural network models and GPU tree models support multiple computing backends:
神经网络模型和 GPU 树模型支持多种计算后端：

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

## GPU-Accelerated Tree Models / GPU 加速树模型

All tree models in PipelineTS (`TorchBoostingForestModel`, `TorchBaggingForestModel`, `DeepForestModel`) are implemented as **GPU-accelerated differentiable tree ensembles** built in PyTorch.

PipelineTS 中所有树模型（`TorchBoostingForestModel`、`TorchBaggingForestModel`、`DeepForestModel`）均实现为基于 PyTorch 的 **GPU 加速可微分树集成**。

### Architecture / 架构

The core `_DifferentiableTreeEnsemble` stores all tree parameters as **batched tensors** `(n_trees, ...)`. The forward pass uses `torch.einsum` and element-wise ops — **zero Python loops** over individual trees.

核心 `_DifferentiableTreeEnsemble` 将所有树参数存储为**批量张量** `(n_trees, ...)`。前向传播使用 `torch.einsum` 和逐元素操作 —— 对单棵树**零 Python 循环**。

```
Input → Feature Normalization → Oblivious Trees (batched einsum) + Linear Skip → Output
                                      ↑
                          Temperature-annealed feature selection (1.0→0.1)
```

**Three ensemble modes / 三种集成模式:**

| Mode / 模式 | Model / 模型 | Description / 描述 |
|---|---|---|
| `additive` | TorchBoostingForest | Staged gradient boosting with GrowNet corrective step / 带 GrowNet 修正的分阶段梯度提升 |
| `additive` + dropout | TorchBaggingForest | Bagging with tree-level dropout for decorrelation / 带树级 Dropout 的袋装集成 |
| `cascade` | DeepForest | Multi-layer gcForest; each layer augments features / 多层级联，每层增强特征 |

### GPU Optimization Details / GPU 优化详情

These optimizations are **automatic** — they activate when CUDA is available and data is large enough:

这些优化是**自动的** —— 当 CUDA 可用且数据足够大时自动激活：

| Optimization / 优化 | Condition / 条件 | Benefit / 收益 |
|---|---|---|
| **AMP (Mixed Precision)** | CUDA + n ≥ 128 | FP16 forward pass with FP32 loss scaling — 1.5–2× throughput / FP16 前向 + FP32 损失缩放 |
| **torch.compile** | CUDA + n ≥ 256 + PyTorch 2.0+ | Fused CUDA kernels via `reduce-overhead` mode / 融合 CUDA 内核 |
| **pin_memory + non_blocking** | CUDA | Overlapped CPU→GPU data transfer / CPU→GPU 数据传输重叠 |
| **torch.inference_mode** | Always (predict) | Disables autograd + version counting for faster inference / 禁用自动求导加速推理 |
| **On-device randperm** | CUDA | `torch.randperm(n, device=cuda)` avoids CPU→GPU transfer per epoch / 避免每轮 CPU→GPU 传输 |

```python
from PipelineTS.ml_model import TorchBoostingForestModel

# GPU acceleration is automatic / GPU 加速是自动的
model = TorchBoostingForestModel(
    time_col='date', target_col='value', lags=16,
    accelerator='cuda',  # or None for auto-detect / None 为自动检测
)
model.fit(data)
result = model.predict(10)
```

### Staged Gradient Boosting / 分阶段梯度提升

`TorchBoostingForestModel` supports true sequential residual boosting via `boosting_stages > 1`:

`TorchBoostingForestModel` 通过 `boosting_stages > 1` 支持真正的顺序残差提升：

1. **Stage 1**: Train a tree ensemble on the target.
2. **Stage 2**: Train a new ensemble on the residual error from Stage 1.
3. **Stage N**: Each stage learns the residual from all previous stages.
4. **GrowNet corrective step**: After all stages, jointly fine-tune all stage models for a few epochs.

```python
model = TorchBoostingForestModel(
    time_col='date', target_col='value', lags=16,
    boosting_stages=3,         # 3 sequential residual stages / 3 个顺序残差阶段
    boosting_shrinkage=0.5,    # Shrinkage per stage / 每阶段收缩率
)
```

### Adaptive Complexity Auto-Tuning / 自适应复杂度自动调优

When `auto_complexity=True`, an `_AdaptiveComplexityController` analyzes the normalized training data and automatically selects optimal `tree_depth` and `n_trees`.

当 `auto_complexity=True` 时，`_AdaptiveComplexityController` 分析归一化训练数据并自动选择最优的 `tree_depth` 和 `n_trees`。

**Data statistics analyzed / 分析的数据统计量:**

| Statistic / 统计量 | Method / 方法 |
|---|---|
| **Noise ratio** | Residual variance after OLS fit / OLS 拟合后的残差方差比 |
| **Nonlinearity** | Running-mean residual vs linear residual / 滑动均值残差 vs 线性残差 |
| **Autocorrelation** | Lag-1 autocorrelation coefficient / 滞后-1 自相关系数 |
| **Feature concentration** | Entropy-based importance concentration / 基于熵的重要性集中度 |

**Complexity profiles / 复杂度配置:**

| Profile / 配置 | Depth / 深度 | Trees / 树数 | Data size / 数据规模 |
|---|---|---|---|
| `minimal` | 2–3 | 8–24 | n < 60 |
| `light` | 3–4 | 16–48 | n < 150 |
| `moderate` | 4–5 | 32–64 | n < 400 |
| `heavy` | 5–6 | 48–96 | n < 1000 |
| `maximal` | 6–7 | 64–128 | n ≥ 1000 |

**Adjustment factors / 调整因子:**

- High noise (>0.7) → reduce complexity / 高噪声 → 降低复杂度
- High nonlinearity (>0.6) → deeper trees / 高非线性 → 更深的树
- Strong autocorrelation (>0.8) → moderate complexity / 强自相关 → 适中复杂度
- Cascade mode → lighter per-layer trees (−0.15) / 级联模式 → 每层更轻量的树

```python
from PipelineTS.ml_model import TorchBoostingForestModel

model = TorchBoostingForestModel(
    time_col='date', target_col='value', lags=16,
    auto_complexity=True,     # Enable auto-tuning / 启用自动调优
    verbose=True,             # Print selection reasons / 打印选择原因
)
model.fit(data)

# Access complexity selection results / 访问复杂度选择结果
info = model.model.complexity_info
print(f"Profile: {info['profile']}")              # e.g., 'moderate'
print(f"Selected: depth={info['tree_depth']}, trees={info['n_trees']}")
print(f"Score: {info['complexity_score']}")        # 0.0 – 1.0
print(f"Reasons: {info['reasons']}")               # ['medium_data(n=350)->moderate', ...]
print(f"Data stats: {info['stats']}")              # noise, nonlinearity, autocorr, etc.
```

**Verbose output example / 详细输出示例:**

```
[AdaptiveComplexity] profile=moderate score=0.550 depth=5 trees=64
  reasons: medium_data(n=350)->moderate, low_noise(0.04)->increase, strong_autocorr(0.85)->moderate
  stats: noise=0.04 nonlin=0.50 autocorr=0.85 n=350 f=42
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

## Multi-Quantile Prediction / 多分位数预测

Output prediction intervals at multiple coverage levels simultaneously. The `predict_quantiles()` method is available on `ModelPipeline` and `SmartRouter`.

同时输出多个覆盖水平的预测区间。`predict_quantiles()` 方法在 `ModelPipeline` 和 `SmartRouter` 上均可用。

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, include_models=['torch_boosting_forest', 'torch_bagging_forest'],
)
pipeline.fit(data)

# Single quantile (standard) / 单分位数（标准）
result = pipeline.predict(10)
# Columns: date, value, value_lower, value_upper

# Multi-quantile output / 多分位数输出
result = pipeline.predict_quantiles(n=10, levels=[0.5, 0.8, 0.95])
# Columns: date, value, value_q0.5_lower, value_q0.5_upper,
#          value_q0.8_lower, value_q0.8_upper, value_q0.95_lower, value_q0.95_upper
```

The intervals are guaranteed to be monotonic: wider coverage levels always produce wider intervals.
区间保证单调性：更宽的覆盖水平始终产生更宽的区间。

---

## Multi-Series (Panel Data) / 多序列（面板数据）

PipelineTS natively supports panel data with multiple time series via the `id_col` parameter.

PipelineTS 通过 `id_col` 参数原生支持包含多条时间序列的面板数据。

### Key Design Decisions / 关键设计决策

- **Per-series scaling**: Each series gets its own fitted `MinMaxScaler` stored in `_panel_scalers`.
- **每序列缩放**：每条序列拥有独立的 `MinMaxScaler`，存储在 `_panel_scalers` 中。

- **Per-series prediction**: Each series is predicted independently with correct inverse scaling.
- **每序列预测**：每条序列独立预测，正确执行逆缩放。

- **Full backward compatibility**: All single-series behavior is unchanged when `id_col=None`.
- **完全向后兼容**：`id_col=None` 时所有单序列行为不变。

```python
from PipelineTS.pipeline import ModelPipeline, SmartRouter

# ModelPipeline with panel data
pipe = ModelPipeline(
    time_col='date', target_col='value', lags=10,
    id_col='series_id',
    include_models=['torch_boosting_forest', 'torch_bagging_forest'],
)
pipe.fit(panel_data)
result = pipe.predict(n=5)  # Returns DataFrame with series_id column
                             # 返回带 series_id 列的 DataFrame

# SmartRouter with panel data (profiles longest series)
# SmartRouter 使用面板数据（对最长序列做数据画像）
router = SmartRouter(
    time_col='date', target_col='value',
    id_col='series_id',
)
router.fit(panel_data)
result = router.predict(10)
```

---

## Covariate Support / 协变量支持

GBDT, Prophet, and AutoARIMA models support external covariates for improved forecasts.

GBDT、Prophet 和 AutoARIMA 模型支持外部协变量以改善预测。

### Types of Covariates / 协变量类型

| Type / 类型 | Description / 描述 | Supported Models / 支持的模型 |
|---|---|---|
| `known_covariates` | Future values known at prediction time (holidays, promotions) / 预测时已知的未来值 | GBDT, Prophet, AutoARIMA, NN (via feature_cols) |
| `past_covariates` | Historical-only features (weather, sensor data) / 仅历史特征 | GBDT |

```python
from PipelineTS.pipeline import ModelPipeline
import pandas as pd

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    known_covariates=['holiday', 'promotion'],
    past_covariates=['temperature'],
    include_models=['torch_boosting_forest', 'prophet', 'auto_arima'],
)
pipeline.fit(data)  # data must contain all covariate columns
                    # data 必须包含所有协变量列

# At prediction time, provide future values of known covariates
# 预测时提供已知协变量的未来值
future_cov = pd.DataFrame({
    'holiday': [0, 0, 0, 1, 0],
    'promotion': [1, 0, 0, 0, 0],
})
result = pipeline.predict(n=5, future_covariates=future_cov)
```

If `future_covariates` is not provided at prediction time but the model was trained with covariates, zero placeholders are used automatically.

如果预测时未提供 `future_covariates` 但模型训练时使用了协变量，将自动使用零占位符。

---

## Incremental Learning / 增量学习

The `update()` method enables incremental training on new data without full retraining from scratch.

`update()` 方法支持在新数据上进行增量训练，无需从头完全重新训练。

### How It Works / 工作原理

| Model Type / 模型类型 | Strategy / 策略 |
|---|---|
| **Neural Networks** / 神经网络 | Warm-start: continue training with fewer epochs on combined data / 热启动：在合并数据上以更少轮次继续训练 |
| **GBDT / Statistical** / GBDT/统计模型 | Full refit on combined old + new data (efficient due to cached parameters) / 在合并数据上完全重新拟合 |

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models=['torch_boosting_forest', 'tide'],
)
pipeline.fit(initial_data)

# When new data arrives / 当新数据到达时
pipeline.update(new_data)

# Training data is now: initial_data + new_data / 训练数据现在是：初始数据 + 新数据
result = pipeline.predict(10)
```

```python
# SmartRouter also supports update() / SmartRouter 也支持 update()
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(time_col='date', target_col='value')
router.fit(initial_data)
router.update(new_data)
```

**Note**: `update()` raises `ValueError` if the pipeline/router has not been fitted yet.
**注意**：如果管道/路由器尚未拟合，`update()` 会抛出 `ValueError`。

---

## SmartRouter HPO / SmartRouter 超参数优化

SmartRouter has built-in Optuna hyperparameter optimization that runs between lag exploration and full training.

SmartRouter 内置 Optuna 超参数优化，在滞后窗口探索和正式训练之间运行。

### HPO Strategies / HPO 策略

| Strategy / 策略 | Description / 描述 |
|---|---|
| `'none'` (default) | No HPO, use default or adaptive hyperparameters / 不使用 HPO，使用默认或自适应超参数 |
| `'quick'` | Capped at 5 trials per model, fast exploration / 每模型最多 5 次试验，快速探索 |
| `'full'` | Full search with `hpo_n_trials` trials per model / 每模型完整搜索 `hpo_n_trials` 次试验 |

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col='date',
    target_col='value',
    hpo_strategy='quick',           # 'none', 'quick', or 'full'
    hpo_n_trials=10,                # Trials per model (for 'full') / 每模型试验数
    hpo_timeout_per_model=60,       # Seconds per model (None=no limit) / 每模型秒数
)
router.fit(data)

# Access HPO results / 访问 HPO 结果
print(router._hpo_results)  # {model: {best_params, best_value, n_trials, time}}
```

### Search Spaces / 搜索空间

| Model Type / 模型类型 | Parameters / 参数 |
|---|---|
| **TorchBoostingForest** | n_trees, tree_depth, learning_rate, boosting_stages |
| **TorchBaggingForest** | n_trees, tree_depth, dropout |
| **NN light** | learning_rate, epochs |
| **NN heavy** | learning_rate, epochs |
| **Prophet** | changepoint_prior_scale |

---

## Multi-Layer Stacking Ensemble / 多层堆叠集成

SmartRouter supports `'multi_stack'` ensemble strategy that trains a two-layer meta-learner.

SmartRouter 支持 `'multi_stack'` 集成策略，训练两层元学习器。

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col='date',
    target_col='value',
    ensemble_strategy='multi_stack',   # Two-layer stacking / 两层堆叠
    ensemble_top_k=3,
)
router.fit(data)
result = router.predict(12)
```

**How it works / 工作原理:**

1. **Layer 1**: Ridge + ElasticNet meta-learners trained on expanding-window OOF predictions.
1. **第 1 层**：Ridge + ElasticNet 元学习器在扩展窗口的 OOF 预测上训练。

2. **Layer 2**: Blends Layer 1 predictions with weights inversely proportional to validation MSE.
2. **第 2 层**：以与验证 MSE 成反比的权重混合第 1 层预测。

Falls back gracefully to simpler stacking or weighted_avg if multi-layer stacking fails.
如果多层堆叠失败，会优雅回退到简单堆叠或加权平均。

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
