# PipelineTS

![PyPI](https://img.shields.io/pypi/v/PipelineTS)
![PyPI - License](https://img.shields.io/pypi/l/PipelineTS)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PipelineTS)
[![Downloads](https://pepy.tech/badge/pipelinets)](https://pepy.tech/project/pipelinets)
[![Downloads](https://pepy.tech/badge/pipelinets/month)](https://pepy.tech/project/pipelinets)
[![Downloads](https://pepy.tech/badge/pipelinets/week)](https://pepy.tech/project/pipelinets)

One-stop time series analysis tool, supporting data preprocessing, feature engineering, model training, model evaluation, and forecasting.
一站式时间序列分析工具，支持数据预处理、特征工程、模型训练、模型评估与预测。

Built on top of spinesTS, it provides a unified interface for 26 time series models with automatic model selection, conformal prediction intervals, multivariate forecasting, and rich visualization with Chinese font support.
基于 spinesTS 构建，提供 26 种时间序列模型的统一接口，支持自动模型选择、保形预测区间、多变量预测，以及支持中文字体的丰富可视化。

---

## Table of Contents / 目录

- [Features / 特性](#features--特性)
- [Installation / 安装](#installation--安装)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Available Models / 可用模型](#available-models--可用模型)
- [ModelPipeline / 模型管道](#modelpipeline--模型管道)
- [SmartRouter / 智能路由器](#smartrouter--智能路由器)
- [Data Preprocessing / 数据预处理](#data-preprocessing--数据预处理)
- [Feature Engineering / 特征工程](#feature-engineering--特征工程)
- [Evaluation Metrics / 评估指标](#evaluation-metrics--评估指标)
- [Model Evaluation / 模型评估](#model-evaluation--模型评估)
- [Training Utilities / 训练工具](#training-utilities--训练工具)
- [Prediction Utilities / 预测工具](#prediction-utilities--预测工具)
- [Visualization / 可视化](#visualization--可视化)
- [Interval Prediction / 区间预测](#interval-prediction--区间预测)
- [Multi-Quantile Prediction / 多分位数预测](#multi-quantile-prediction--多分位数预测)
- [Multivariate Prediction / 多变量预测](#multivariate-prediction--多变量预测)
- [Multi-Series (Panel Data) / 多序列（面板数据）](#multi-series-panel-data--多序列面板数据)
- [Covariate Support / 协变量支持](#covariate-support--协变量支持)
- [Incremental Learning / 增量学习](#incremental-learning--增量学习)
- [Save and Load / 保存与加载](#save-and-load--保存与加载)
- [Documentation / 文档](#documentation--文档)
- [Tutorials / 教程](#tutorials--教程)
- [License / 许可证](#license--许可证)

---

## Features / 特性

- **29 built-in models**: 15 neural network, 8 machine learning, 2 statistical, and 4 foundation (Chronos-2) models.
- **29 个内置模型**：15 个神经网络、8 个机器学习、2 个统计模型和 4 个基础（Chronos-2）模型。

- **Automatic model selection**: `ModelPipeline` trains and compares all models, automatically selecting the best one.
- **自动模型选择**：`ModelPipeline` 训练并比较所有模型，自动选出最佳模型。

- **Intelligent SmartRouter**: `SmartRouter` intelligently analyzes data characteristics (stationarity, seasonality, trend, noise, autocorrelation) and automatically selects optimal preprocessing, models, lags, and hyperparameters. Supports weighted ensemble of top models with 'auto' or 'weighted_avg' strategies.
- **智能 SmartRouter**：`SmartRouter` 智能分析数据特征（平稳性、季节性、趋势、噪声、自相关），自动选择最优预处理、模型、滞后窗口和超参数。支持顶级模型的加权集成，提供 'auto' 和 'weighted_avg' 策略。

- **Conformal prediction intervals**: Industry-standard distribution-free prediction intervals with coverage guarantees.
- **保形预测区间**：行业标准的无分布预测区间，具有覆盖率保证。

- **CQR for neural networks**: Conformalized Quantile Regression provides adaptive, input-dependent intervals for NN models.
- **神经网络 CQR**：保形分位数回归为神经网络模型提供自适应的、依赖输入的预测区间。

- **Multi-quantile prediction**: Output prediction intervals at multiple coverage levels simultaneously (e.g., 50%, 80%, 95%).
- **多分位数预测**：同时输出多个覆盖水平的预测区间（如 50%、80%、95%）。

- **Multivariate forecasting**: ITransformer and SRSNet support multi-input/multi-output prediction modes.
- **多变量预测**：ITransformer 和 SRSNet 支持多输入/多输出预测模式。

- **Multi-series (panel data)**: Native support for multiple time series via `id_col`, with per-series scaling and prediction.
- **多序列（面板数据）**：通过 `id_col` 原生支持多条时间序列，每条序列独立缩放和预测。

- **Covariate support**: Known future covariates and past covariates for GBDT, Prophet, and AutoARIMA models.
- **协变量支持**：GBDT、Prophet 和 AutoARIMA 模型支持已知未来协变量和历史协变量。

- **Incremental learning**: `update()` method for warm-start training on new data without full retraining.
- **增量学习**：`update()` 方法支持在新数据上热启动训练，无需完全重新训练。

- **Visualization with Chinese font support**: Comprehensive plotting toolkit with automatic Chinese font detection, supporting single/multi-series plots, forecast visualization, leaderboard charts, residual diagnostics, ACF/PACF, and time series decomposition.
- **支持中文字体的可视化**：全面的绘图工具包，自动检测中文字体，支持单/多序列图、预测可视化、排行榜图表、残差诊断、ACF/PACF 和时间序列分解。

- **GlobalTemporalBlock (GTB)**: Optional plug-in module for all 12 NN models combining frequency mixing, gated linear attention, and SwiGLU FFN with residual connections and RevIN normalization. Supports both static (manual) and adaptive MoE (Mixture-of-Experts) routing modes.
- **GlobalTemporalBlock (GTB)**：所有 12 个 NN 模型的可选插件模块，组合频率混合、门控线性注意力和 SwiGLU FFN，带残差连接和 RevIN 归一化。支持静态（手动）和自适应 MoE（混合专家）路由模式。

- **MoE Adaptive Routing**: Learned sparse top-K expert selection (inspired by DeepSeek-V2 / Switch Transformer) with load-balancing auxiliary loss. The router dynamically activates 2 of 3 GTB experts per sample for compute-efficient inference.
- **MoE 自适应路由**：学习型稀疏 top-K 专家选择（灵感来自 DeepSeek-V2 / Switch Transformer），带负载均衡辅助损失。路由器动态激活每个样本 3 个 GTB 专家中的 2 个，实现高效推理。

- **Rich feature engineering**: Automatic lag feature extraction (26+ features per window) for GBDT/ML models and Prophet.
- **丰富的特征工程**：为 GBDT/ML 模型和 Prophet 自动提取滞后特征（每个窗口 26+ 个特征）。

- **Data preprocessing toolkit**: Missing value detection & interpolation, outlier detection & handling, data quality reporting, stationarity tests, frequency auto-detection, and time series train/test splitting.
- **数据预处理工具箱**：缺失值检测与插值、异常值检测与处理、数据质量报告、平稳性检验、频率自动检测、时间序列训练/测试分割。

- **Comprehensive evaluation metrics**: MAPE, sMAPE, MASE, R², MedAE for point forecasts; PICP, PINAW, Winkler score for prediction intervals.
- **全面的评估指标**：点预测指标（MAPE、sMAPE、MASE、R²、MedAE）；区间预测指标（PICP、PINAW、Winkler 分数）。

- **Model evaluation framework**: Walk-forward backtesting, residual analysis with diagnostics, and multi-model comparison visualization.
- **模型评估框架**：前向回测、残差诊断分析、多模型对比可视化。

- **User-facing feature engineering**: Unified API composing Fourier features, holiday features, rolling lag features, and calendar features.
- **面向用户的特征工程**：统一 API，组合傅里叶特征、节假日特征、滚动滞后特征和日历特征。

- **Training utilities**: Built-in AutoTune (Optuna / random search), weighted ensemble, stacking ensemble, and multi-layer stacking.
- **训练工具**：内置 AutoTune（Optuna / 随机搜索）、加权集成、堆叠集成、多层堆叠集成。

- **SmartRouter HPO**: Built-in Optuna hyperparameter optimization in SmartRouter with 'quick' and 'full' strategies.
- **SmartRouter HPO**：SmartRouter 内置 Optuna 超参数优化，支持 'quick' 和 'full' 策略。

- **Prediction utilities**: Rolling (sliding window) predictor and model explainability (feature importance).
- **预测工具**：滚动（滑动窗口）预测器和模型可解释性（特征重要性）。

- **Unified API**: All models share the same `fit()` / `predict()` interface.
- **统一 API**：所有模型共享相同的 `fit()` / `predict()` 接口。

- **Built-in datasets**: Multiple time series datasets for quick experimentation.
- **内置数据集**：多个时间序列数据集，方便快速实验。

---

## Installation / 安装

Install via pip:
通过 pip 安装：

```bash
pip install PipelineTS
```

Python >= 3.9 is required.
需要 Python >= 3.9。

---

## Quick Start / 快速开始

### Load Data / 加载数据

```python
from PipelineTS.dataset import LoadElectricDataSets
import pandas as pd

# Load a built-in dataset
# 加载内置数据集
data = LoadElectricDataSets()
time_col = 'date'
target_col = 'value'
data[time_col] = pd.to_datetime(data[time_col])
```

### Train a Single Model / 训练单个模型

```python
from PipelineTS.ml_model import TorchBoostingForestModel

# Initialize and train a model
# 初始化并训练模型
model = TorchBoostingForestModel(
    time_col=time_col,
    target_col=target_col,
    lags=12,
    quantile=0.9,
)
model.fit(data)

# Predict the next 10 steps
# 预测未来 10 个时间步
result = model.predict(10)
```

### Use ModelPipeline for Auto Model Selection / 使用 ModelPipeline 自动选择模型

```python
from PipelineTS.pipeline import ModelPipeline

# Create pipeline and train all models
# 创建管道并训练所有模型
pipeline = ModelPipeline(
    time_col=time_col,
    target_col=target_col,
    lags=12,
    quantile=0.9,
    include_models='ml',  # Options: 'light', 'all', 'nn', 'ml', or a list of model names
                          # 选项：'light', 'all', 'nn', 'ml', 或模型名称列表
)

# Train and get leaderboard
# 训练并获取排行榜
leaderboard = pipeline.fit(data)

# Predict using the best model
# 使用最佳模型进行预测
result = pipeline.predict(10)
```

### Use SmartRouter for Intelligent Auto-Selection / 使用 SmartRouter 智能自动选择

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col=time_col,
    target_col=target_col,
    n_predict=12,
    max_models=5,
    ensemble_strategy='auto',  # 'auto', 'weighted_avg', or 'none'
)

router.fit(data)
result = router.predict(12)  # Uses ensemble if built, else best model
```

### Visualize Results / 可视化结果

```python
# One-line plot from pipeline (supports Chinese labels)
# 管道一键绘图（支持中文标签）
pipeline.plot(n=10, lang='zh')
pipeline.plot_leaderboard(lang='zh')

# Or use standalone functions / 或使用独立函数
from PipelineTS.plot import plot_forecast, plot_series, configure_chinese_font

configure_chinese_font()  # Auto-detect Chinese font / 自动检测中文字体
plot_forecast(train_data, result, time_col=time_col, target_col=target_col)
plot_series(data, time_col=time_col, target_col=target_col)
```

---

## Available Models / 可用模型

### Neural Network Models / 神经网络模型 (15)

| Model / 模型 | Key / 键名 | Description / 描述 |
|---|---|---|
| NLinearModel | `n_linear` | Simple linear mapping / 简单线性映射 |
| DLinearModel | `d_linear` | Decomposition linear / 分解线性模型 |
| NBeatsModel | `n_beats` | N-BEATS architecture / N-BEATS 架构 |
| NHitsModel | `n_hits` | Hierarchical interpolation / 分层插值 |
| TFTModel | `tft` | Temporal Fusion Transformer / 时序融合 Transformer |
| TransformerModel | `transformer` | Transformer encoder / Transformer 编码器 |
| TiDEModel | `tide` | Time-series Dense Encoder / 时序密集编码器 |
| GAUModel | `gau` | Gated Attention Unit / 门控注意力单元 |
| StackingRNNModel | `stacking_rnn` | RWKV linear RNN + gated residual blocks / RWKV 线性 RNN + 门控残差块 |
| Time2VecModel | `time2vec` | Trend-seasonal decomposition + Time2Vec + RWKV / 趋势-季节分解 + Time2Vec + RWKV |
| PatchRNNModel | `patch_rnn` | Patch-based RNN / 基于 Patch 的 RNN |
| TCNModel | `tcn` | Temporal Convolutional Network / 时序卷积网络 |
| ITransformerModel | `itransformer` | Inverted Transformer (multivariate) / 反转 Transformer（多变量） |
| SRSNetModel | `srs_net` | Selective Representation Space Network (multivariate) / 选择性表征空间网络（多变量） |
| DeepARModel | `deepar` | Probabilistic forecasting with RWKV encoder + Gaussian head / 基于 RWKV 编码器 + 高斯输出头的概率预测 |

### Machine Learning Models / 机器学习模型 (8)

| Model / 模型 | Key / 键名 | Description / 描述 |
|---|---|---|
| TorchBoostingForestModel | `torch_boosting_forest` | GPU-accelerated gradient boosting forest (MART/DART) / GPU 加速梯度提升森林 |
| TorchBaggingForestModel | `torch_bagging_forest` | GPU-accelerated bagging forest with dropout / GPU 加速装袋森林 |
| DeepForestModel | `deep_forest` | Cascade (gcForest) multi-layer ensemble / 级联（gcForest）多层集成 |
| WideGBRTModel | `wide_gbrt` | Wide-table GBRT with rich features / 宽表 GBRT + 丰富特征 |
| MultiOutputRegressorModel | `multi_output_model` | Multi-output regressor / 多输出回归 |
| MultiStepRegressorModel | `multi_step_model` | Multi-step regressor / 多步回归 |
| RegressorChainModel | `regressor_chain` | Regressor chain / 回归链 |

### Statistical Models / 统计模型 (2)

| Model / 模型 | Key / 键名 | Description / 描述 |
|---|---|---|
| ProphetModel | `prophet` | Custom Prophet-like model with ridge regression / 自定义类 Prophet 岭回归模型 |
| AutoARIMAModel | `auto_arima` | Auto ARIMA parameter search / 自动 ARIMA 参数搜索 |

### Foundation Models / 基础模型 (4, optional / 可选)

> Requires: `pip install chronos-forecasting`

| Model / 模型 | Key / 键名 | Description / 描述 |
|---|---|---|
| Chronos2Model | `chronos_2` | Amazon Chronos-2 (120M params, covariate support) / Amazon Chronos-2（120M 参数，支持协变量） |
| Chronos2SynthModel | `chronos_2_synth` | Chronos-2-Synth trained on synthetic data (120M) / Chronos-2-Synth 合成数据训练（120M） |
| Chronos2SmallModel | `chronos_2_small` | Chronos-2-Small lightweight variant (28M) / Chronos-2-Small 轻量版（28M） |
| ChronosBoltModel | `chronos_bolt_small` | Amazon Chronos-Bolt Small (tiny, zero-shot) / Amazon Chronos-Bolt Small（微型，零样本） |

All Chronos-2 models are **zero-shot** — no training needed, they use pretrained weights from large-scale time series corpora.
所有 Chronos-2 模型都是**零样本**的 —— 无需训练，使用大规模时序语料库的预训练权重。

---

## ModelPipeline / 模型管道

`ModelPipeline` is the core class for automatic model comparison and selection.
`ModelPipeline` 是自动模型比较和选择的核心类。

### Model Filtering / 模型筛选

```python
from PipelineTS.pipeline import ModelPipeline

# List all available models
# 列出所有可用模型
ModelPipeline.list_all_available_models()

# Use predefined model sets / 使用预定义模型集合
pipeline = ModelPipeline(..., include_models='light')  # 'light', 'all', 'nn', 'ml'

# Or specify a list of model names / 或指定模型名称列表
pipeline = ModelPipeline(..., include_models=['torch_boosting_forest', 'torch_bagging_forest', 'd_linear'])
```

### PipelineConfigs / 管道配置

Use `PipelineConfigs` to create multiple model variants with different hyperparameters.
使用 `PipelineConfigs` 创建具有不同超参数的多个模型变体。

```python
from PipelineTS.pipeline import PipelineConfigs

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_v1', {'init_configs': {'n_trees': 32}}),
    ('torch_boosting_forest', 'boost_v2', {'init_configs': {'n_trees': 128}}),
])

pipeline = ModelPipeline(..., configs=configs)
```

### Per-Model Pipeline Settings / 每模型管道设置

Use `pipeline_configs` to give each model variant its own lags, scaler, or differencing settings.
使用 `pipeline_configs` 为每个模型变体指定独立的滞后窗口、缩放器或差分设置。

```python
from sklearn.preprocessing import StandardScaler
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_short_std', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'lags': 6, 'scaler': StandardScaler()},
    }),
    ('torch_boosting_forest', 'boost_long_none', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'lags': 24, 'scaler': None},
    }),
])

pipeline = ModelPipeline(
    time_col=time_col, target_col=target_col, lags=12,
    include_models=['torch_boosting_forest'],
    configs=configs,
)
leaderboard = pipeline.fit(data)
```

Supported `pipeline_configs` keys: `lags`, `scaler`, `differential_n`, `feature_cols`.
See [Pipeline Usage](docs/pipeline.md) for details.

支持的 `pipeline_configs` 键：`lags`、`scaler`、`differential_n`、`feature_cols`。
详见 [管道使用](docs/pipeline.md)。

### Double-underscore Syntax / 双下划线语法

Pass model-specific parameters directly via double-underscore syntax.
通过双下划线语法直接传递模型特定参数。

```python
pipeline = ModelPipeline(
    ...,
    torch_boosting_forest__n_trees=64,
    torch_bagging_forest__n_trees=128,
    d_linear__lags=50,
)
```

---

## SmartRouter / 智能路由器

`SmartRouter` is an intelligent routing system that automatically analyzes time series data characteristics and makes optimal decisions for preprocessing, model selection, lag window size, and hyperparameters. It also supports automatic weighted ensemble of top-performing models.

`SmartRouter` 是一个智能路由系统，自动分析时间序列数据特征，为预处理、模型选择、滞后窗口大小和超参数做出最优决策。它还支持顶级模型的自动加权集成。

### Key Capabilities / 核心能力

| Feature / 特性 | Description / 描述 |
|---|---|
| **Automatic Data Profiling** / 自动数据画像 | Detects stationarity, seasonality, trend strength, noise level, autocorrelation, multi-seasonality, and regime changes / 检测平稳性、季节性、趋势强度、噪声水平、自相关、多季节性和机制变化 |
| **Intelligent Model Scoring** / 智能模型评分 | Scores 25+ models based on data characteristics (length, seasonality, trend, noise, autocorrelation, forecast horizon) / 基于数据特征（长度、季节性、趋势、噪声、自相关、预测范围）对 25+ 模型评分 |
| **Adaptive Feature Engineering** / 自适应特征工程 | Auto-enables adaptive MoE routing for NN models; Prophet lag features when autocorrelation is strong / 为 NN 模型自动启用自适应 MoE 路由；自相关强时启用 Prophet 滞后特征 |
| **Adaptive Hyperparameters** / 自适应超参数 | Auto-adjusts GBDT (n_estimators, learning_rate, max_depth) and NN (routing_mode) based on data profile / 根据数据画像自动调整 GBDT 和 NN 超参数 |
| **Weighted Ensemble** / 加权集成 | `ensemble_strategy='auto'` builds ensemble when top models are competitive; `'weighted_avg'` always builds / `ensemble_strategy='auto'` 在顶级模型具有竞争力时构建集成；`'weighted_avg'` 始终构建 |
| **Model Pinning** / 模型指定 | `include_models` pins specific models; SmartRouter optimizes lags, scaler, hyperparams, and ensemble for them / `include_models` 指定特定模型；SmartRouter 为其优化滞后窗口、缩放器、超参数和集成 |

### Usage / 用法

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col='date',
    target_col='value',
    n_predict=12,               # Forecast horizon / 预测范围
    max_models=5,               # Number of candidate models / 候选模型数量
    ensemble_strategy='auto',   # 'auto', 'weighted_avg', or 'none'
    ensemble_top_k=3,           # Max models in ensemble / 集成中最大模型数
    random_state=42,
    verbose=True,
)

router.fit(data)

# Predict (uses ensemble if built, else best single model)
# 预测（使用集成如果已构建，否则使用最佳单模型）
result = router.predict(n=12)

# Force using best single model (bypass ensemble)
# 强制使用最佳单模型（绕过集成）
result = router.predict(n=12, use_ensemble=False)

# Access selected strategy
# 查看选择的策略
print(router.strategy)        # Full strategy dict / 完整策略字典
print(router.leader_board_)   # Model rankings / 模型排名
print(router.ensemble_)       # Ensemble info (if built) / 集成信息（如果已构建）
```

### Pinning Models (`include_models`) / 指定模型

Pin specific models and let SmartRouter optimize everything else for them:

指定特定模型，让 SmartRouter 为其优化其他所有环节：

```python
# Pin specific models — SmartRouter handles preprocessing, lags, hyperparams, ensemble
# 指定模型 — SmartRouter 处理预处理、滞后窗口、超参数、集成
router = SmartRouter(
    time_col='date',
    target_col='value',
    include_models=['prophet', 'torch_boosting_forest'],
    hpo_strategy='quick',   # HPO still works for pinned models / HPO 仍然适用于指定模型
)
router.fit(data)

# Single model with full optimization / 单模型全面优化
router = SmartRouter(
    time_col='date',
    target_col='value',
    include_models='torch_boosting_forest',  # str accepted / 字符串也可以
)
router.fit(data)
```

### Data Profile Fields / 数据画像字段

The `DataProfile` object contains these characteristics:

| Field / 字段 | Description / 描述 |
|---|---|
| `n_rows` | Number of observations / 观测值数量 |
| `freq` | Detected frequency (MS, D, h, etc.) / 检测到的频率 |
| `stationarity` | 'stationary', 'trend_stationary', 'non_stationary' / 平稳性结论 |
| `trend_strength` | R² of linear fit (0-1) / 线性拟合 R² |
| `seasonality_strength` | Spectral power at dominant frequency (0-1) / 主导频率的谱功率 |
| `autocorr_lag1`, `autocorr_lag2` | Lag-1 and lag-2 autocorrelation / 一阶和二阶自相关 |
| `n_seasonalities` | Number of detected seasonal periods / 检测到的季节周期数 |
| `regime_changes` | Count of trend direction changes / 趋势方向变化次数 |
| `noise_ratio` | Std of residuals / total std / 残差标准差 / 总标准差 |
| `skewness`, `kurtosis` | Distribution shape metrics / 分布形状指标 |
| `pct_missing`, `pct_outlier` | Missing and outlier percentages / 缺失值和异常值百分比 |

### Ensemble Strategies / 集成策略

- **`ensemble_strategy='auto'` (default)**: Builds ensemble only when multiple top models have similar performance (within 30% of best metric). This avoids ensemble with one dominant model.
- **`ensemble_strategy='weighted_avg'`**: Always builds ensemble of top-K models with inverse-metric weighting.
- **`ensemble_strategy='none'`**: Disables ensemble, always uses single best model.

---

## Data Preprocessing / 数据预处理

PipelineTS provides a comprehensive data preprocessing toolkit for time series data.
PipelineTS 提供全面的时间序列数据预处理工具箱。

### Missing Value Handling / 缺失值处理

```python
from PipelineTS.preprocessing import TimeSeriesMissingHandler

handler = TimeSeriesMissingHandler(time_col='date')

# Detect missing values (explicit NaNs + implicit time gaps)
# 检测缺失值（显式 NaN + 隐式时间间隔缺失）
report = handler.fit(data)
print(f"Implicit gaps: {report['n_implicit_gaps']}")
print(f"Explicit NaN: {report['n_explicit_nan']}")

# Fill missing values / 填充缺失值
# Methods: 'linear', 'ffill', 'bfill', 'spline', 'zero'
# 方法：'linear'（线性插值）, 'ffill'（前向填充）, 'bfill'（后向填充）, 'spline'（样条插值）, 'zero'（零填充）
filled = handler.transform(data, method='linear')
```

### Outlier Detection & Handling / 异常值检测与处理

```python
from PipelineTS.preprocessing import TimeSeriesOutlierDetector

# Methods: 'iqr', 'zscore', 'rolling_zscore', 'grubbs'
# 方法：'iqr'（四分位距）, 'zscore'（Z 分数）, 'rolling_zscore'（滚动 Z 分数）, 'grubbs'（Grubbs 检验）
detector = TimeSeriesOutlierDetector(time_col='date', method='iqr')

# Detect outliers / 检测异常值
mask = detector.fit(data, target_col='value')

# Handle outliers / 处理异常值
# Strategies: 'clip', 'nan', 'median', 'linear'
# 策略：'clip'（截断）, 'nan'（置空）, 'median'（中位数替换）, 'linear'（线性插值替换）
cleaned = detector.transform(data, target_col='value', strategy='clip')
```

### Data Quality Report / 数据质量报告

```python
from PipelineTS.preprocessing import TimeSeriesDataQualityReport

reporter = TimeSeriesDataQualityReport(time_col='date', target_col='value')

# Generate a comprehensive report / 生成全面的数据质量报告
report = reporter.fit(data)

# Print a formatted report / 打印格式化报告
reporter.report(data)
```

### Stationarity Tests / 平稳性检验

```python
from PipelineTS.preprocessing import StationarityTest

tester = StationarityTest(significance_level=0.05)

# Run ADF + KPSS combined test / 运行 ADF + KPSS 联合检验
result = tester.fit(data['value'].values)
print(result['conclusion'])       # 'stationary', 'trend_stationary', etc.
print(result['suggested_action']) # Recommended action / 建议操作

# Auto-suggest differencing order / 自动建议差分阶数
d = tester.suggest_differencing(data['value'].values)
```

### Frequency Detection / 频率检测

```python
from PipelineTS.preprocessing.time_series_analysis import FrequencyDetector

detector = FrequencyDetector(time_col='date')
info = detector.fit(data, target_col='value')
print(f"Frequency: {info['freq']}")
print(f"Regular: {info['is_regular']}")
print(f"Dominant periods: {info['dominant_periods']}")  # via FFT / 基于 FFT
```

### Time Series Split / 时间序列分割

```python
from PipelineTS.preprocessing import TimeSeriesSplit

# Simple temporal split / 简单时间分割
train, test = TimeSeriesSplit.split(data, time_col='date', test_size=0.2)

# Expanding window CV / 扩展窗口交叉验证
for train_df, test_df in TimeSeriesSplit.expanding_window(
    data, time_col='date', min_train_size=100, test_size=20, step=10
):
    pass  # train and evaluate / 训练和评估

# Sliding window CV / 滑动窗口交叉验证
for train_df, test_df in TimeSeriesSplit.sliding_window(
    data, time_col='date', train_size=100, test_size=20, step=10
):
    pass
```

---

## Feature Engineering / 特征工程

PipelineTS provides a unified feature engineering pipeline with multiple composable feature extractors.
PipelineTS 提供统一的特征工程管道，包含多个可组合的特征提取器。

### Unified Feature Pipeline / 统一特征管道

```python
from PipelineTS.feature_engineering import TimeSeriesFeatureEngineer

engineer = TimeSeriesFeatureEngineer(
    time_col='date',
    target_col='value',
    use_calendar=True,              # Calendar features (weekday, month, etc.) / 日历特征（星期、月份等）
    use_fourier=True,               # Fourier periodic features / 傅里叶周期特征
    fourier_periods=[7, 365],       # Weekly + yearly cycles / 周 + 年周期
    fourier_harmonics=2,            # Harmonics per period / 每个周期的谐波数
    use_holidays=True,              # Holiday indicators / 节假日指示符
    holiday_country='US',           # Country-specific holidays / 国家特定节假日
    use_lags=True,                  # Rolling lag features / 滚动滞后特征
    lag_window=12,                  # Window size / 窗口大小
    lag_features=['mean', 'std', 'trend_slope', 'ema'],
)

df_enriched = engineer.fit_transform(data)
```

### Individual Feature Extractors / 单独的特征提取器

```python
from PipelineTS.feature_engineering import FourierFeatures, HolidayFeatures, LagFeatureExtractor

# Fourier features / 傅里叶特征
ff = FourierFeatures(time_col='date', periods={'weekly': 7, 'yearly': 365}, n_harmonics=2)
df = ff.transform(data)

# Holiday features (CN uses chinese-calendar for official data)
# 节假日特征（中国使用 chinese-calendar 获取官方数据，pip install chinesecalendar）
hf = HolidayFeatures(time_col='date', country='CN')
df = hf.transform(data)  # includes is_workday, is_in_lieu for CN / 中国额外包含工作日、调休特征

# Lag features (15 rolling statistics) / 滞后特征（15 种滚动统计量）
lf = LagFeatureExtractor(time_col='date', target_col='value', window=12, features='all')
df = lf.transform(data)
```

---

## Evaluation Metrics / 评估指标

### Point Forecast Metrics / 点预测指标

```python
from PipelineTS.metrics import mape, smape, mase, r2_score, medae
from PipelineTS.spinesTS.metrics import mae, mse, rmse, wmape
import numpy as np

y_true = np.array([100, 200, 300, 400, 500], dtype=np.float64)
y_pred = np.array([110, 190, 310, 390, 510], dtype=np.float64)

print(f"MAE:   {mae(y_true, y_pred):.4f}")
print(f"RMSE:  {rmse(y_true, y_pred):.4f}")
print(f"MAPE:  {mape(y_true, y_pred):.4f}")    # Mean Absolute Percentage Error / 平均绝对百分比误差
print(f"sMAPE: {smape(y_true, y_pred):.4f}")   # Symmetric MAPE / 对称 MAPE
print(f"R²:    {r2_score(y_true, y_pred):.4f}") # Coefficient of determination / 决定系数
print(f"MedAE: {medae(y_true, y_pred):.4f}")   # Median Absolute Error / 中位绝对误差

# MASE requires training data / MASE 需要训练数据
y_train = np.array([50, 80, 120, 160, 200, 250, 300], dtype=np.float64)
print(f"MASE:  {mase(y_true, y_pred, y_train):.4f}")  # Mean Absolute Scaled Error / 平均绝对缩放误差
```

### Interval Prediction Metrics / 区间预测指标

```python
from PipelineTS.metrics import picp, pinaw, winkler_score, quantile_acc

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
lower  = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
upper  = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

print(f"PICP:    {picp(y_true, lower, upper):.4f}")             # Coverage probability / 覆盖概率
print(f"PINAW:   {pinaw(y_true, lower, upper):.4f}")            # Normalized average width / 归一化平均宽度
print(f"Winkler: {winkler_score(y_true, lower, upper):.4f}")    # Winkler interval score / Winkler 区间分数
```

---

## Model Evaluation / 模型评估

### Backtesting / 回测

Walk-forward backtesting evaluates model performance by simulating sequential real-world forecasts.
前向回测通过模拟顺序的真实预测来评估模型性能。

```python
from PipelineTS.evaluation import Backtester
from PipelineTS.ml_model import TorchBoostingForestModel
from PipelineTS.spinesTS.metrics import mae

model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)
bt = Backtester(model, time_col='date', target_col='value', metric=mae, metric_name='MAE')

# Run expanding window backtesting / 运行扩展窗口回测
results = bt.fit(data, n_splits=5, test_size=12, mode='expanding')

# Summary statistics / 汇总统计
summary = bt.summary()
print(f"Mean MAE: {summary['mean']:.4f} ± {summary['std']:.4f}")
```

### Residual Analysis / 残差分析

```python
from PipelineTS.evaluation import ResidualAnalyzer

analyzer = ResidualAnalyzer(y_true, y_pred)

# Statistics, normality, autocorrelation, and bias analysis
# 统计量、正态性检验、自相关分析和偏差分析
stats = analyzer.statistics()
norm = analyzer.normality_test()     # Shapiro-Wilk + Jarque-Bera
acorr = analyzer.autocorrelation()   # ACF + Ljung-Box test / ACF + Ljung-Box 检验
bias = analyzer.bias_analysis()      # Systematic bias detection / 系统性偏差检测

analyzer.report()  # Formatted report / 格式化报告
analyzer.plot()          # 4-panel diagnostic plot / 四面板诊断图
```

### Model Comparison / 模型对比

```python
from PipelineTS.evaluation import ModelComparison
from PipelineTS.metrics import mape, r2_score, picp

comp = ModelComparison(time_col='date', target_col='value')
comp.add_result('BoostForest', y_true, y_pred_boost, lower=lower_boost, upper=upper_boost)
comp.add_result('BagForest', y_true, y_pred_bag, lower=lower_bag, upper=upper_bag)

# Evaluate on multiple metrics / 多指标评估
table = comp.fit(
    metrics={'MAPE': mape, 'R²': r2_score},
    interval_metrics={'PICP': picp}
)

comp.rank('MAPE', ascending=True)  # Rank by metric / 按指标排名
comp.plot_bar()                     # Bar chart / 柱状图
comp.plot_radar()                   # Radar chart / 雷达图
comp.plot_predictions()             # Prediction overlay plot / 预测叠加图
```

---

## Training Utilities / 训练工具

### AutoTune / 自动调参

Built-in hyperparameter tuning using Optuna (with random search fallback).
内置超参数调优，使用 Optuna（支持随机搜索回退）。

```python
from PipelineTS.training import AutoTune
from PipelineTS.ml_model import TorchBoostingForestModel
from PipelineTS.spinesTS.metrics import mae

tuner = AutoTune(
    model_class=TorchBoostingForestModel,
    time_col='date', target_col='value', lags=12,
    metric=mae, n_trials=30,
)

best_model, best_params, history = tuner.fit(data, search_space={
    'n_trees': ('int', 16, 128),
    'learning_rate': ('float', 0.01, 0.3, True),  # True = log scale / True = 对数刻度
    'tree_depth': ('int', 3, 7),
})
```

### Ensemble Methods / 集成方法

```python
from PipelineTS.training import WeightedEnsemble, StackingEnsemble
from PipelineTS.ml_model import TorchBoostingForestModel, TorchBaggingForestModel

models = [
    ('boost', TorchBoostingForestModel(time_col='date', target_col='value', lags=12)),
    ('bag', TorchBaggingForestModel(time_col='date', target_col='value', lags=12)),
]

# Weighted ensemble (auto-computes inverse-error weights)
# 加权集成（自动计算逆误差权重）
ens = WeightedEnsemble(models, time_col='date', target_col='value', weights='auto')
ens.fit(data)
result = ens.predict(10)
print(ens.get_weights())

# Stacking ensemble (ridge meta-learner on CV predictions)
# 堆叠集成（在交叉验证预测上使用岭回归元学习器）
stack = StackingEnsemble(models, time_col='date', target_col='value', n_folds=3)
stack.fit(data)
result = stack.predict(10)
```

---

## Prediction Utilities / 预测工具

### Rolling Prediction / 滚动预测

Re-fits the model on a sliding window of recent data for adaptive forecasting.
在滑动窗口的最新数据上重新训练模型，实现自适应预测。

```python
from PipelineTS.prediction import RollingPredictor
from PipelineTS.ml_model import TorchBoostingForestModel

model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)
rp = RollingPredictor(
    model, time_col='date', target_col='value',
    train_size=100,   # Training window / 训练窗口
    horizon=10,       # Forecast steps per window / 每个窗口的预测步数
    step=10,          # Window advance / 窗口前进步数
    refit=True,       # Re-fit each window / 每个窗口重新训练
)

results = rp.predict(data)
eval_results = rp.score(results)
```

### Model Explainability / 模型可解释性

```python
from PipelineTS.prediction import ModelExplainer

explainer = ModelExplainer(model, time_col='date', target_col='value')

# Native feature importance (GBDT models) / 原生特征重要性（GBDT 模型）
importance = explainer.feature_importance()
explainer.plot_importance(top_k=15)
```

---

## Visualization / 可视化

PipelineTS provides a comprehensive visualization toolkit with automatic Chinese font detection. All plot functions support bilingual labels (`lang='zh'` or `lang='en'`).

PipelineTS 提供全面的可视化工具包，自动检测中文字体。所有绑图函数支持双语标签（`lang='zh'` 或 `lang='en'`）。

### Chinese Font Configuration / 中文字体配置

```python
from PipelineTS.plot import configure_chinese_font

# Auto-detect and configure Chinese font (macOS/Windows/Linux)
# 自动检测并配置中文字体（macOS/Windows/Linux）
font_name = configure_chinese_font()
print(f"Using font: {font_name}")  # e.g., 'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC'
```

### Plot Functions / 绑图函数

| Function / 函数 | Description / 描述 |
|---|---|
| `plot_series()` | Single or multi-series (panel) visualization / 单序列或多序列（面板）可视化 |
| `plot_forecast()` | Actual vs forecast with prediction intervals / 实际值 vs 预测值 + 预测区间 |
| `plot_leaderboard()` | Model ranking horizontal bar chart / 模型排名水平柱状图 |
| `plot_leaderboard_detail()` | Leaderboard with training/eval cost / 排行榜 + 训练/评估耗时 |
| `plot_model_comparison()` | Multi-model forecast overlay / 多模型预测叠加对比 |
| `plot_residuals()` | 4-panel residual diagnostics / 四面板残差诊断 |
| `plot_acf_pacf()` | ACF + PACF side by side / ACF + PACF 并排图 |
| `plot_decomposition()` | Trend / seasonal / residual decomposition / 趋势/季节性/残差分解 |
| `plot_train_test_split()` | Visualize train/test partition / 训练集/测试集分割可视化 |

### Quick Examples / 快速示例

```python
from PipelineTS.plot import plot_series, plot_forecast, plot_decomposition, TSPlotter

# Single series / 单序列
plot_series(data, time_col='date', target_col='value', title='销量趋势')

# Multi-series panel / 多序列面板
plot_series(panel_data, time_col='date', target_col='value', id_col='store_id')

# Forecast with intervals / 预测 + 区间
plot_forecast(train_data, pred_data, time_col='date', target_col='value')

# Time series decomposition / 时间序列分解
plot_decomposition(data, time_col='date', target_col='value')
```

### TSPlotter Class / TSPlotter 类

```python
# Reusable plotter with fixed column names / 可复用的绑图器
plotter = TSPlotter(time_col='date', target_col='value', lang='zh')
plotter.plot_series(data)
plotter.plot_forecast(train, pred)
plotter.plot_leaderboard(leaderboard)
plotter.plot_decomposition(data)
plotter.plot_residuals(y_true, y_pred)
```

### Pipeline Integration / 管道集成

```python
# One-line plot from Pipeline or SmartRouter
# 管道或智能路由器一键绘图
pipeline.plot(n=12, lang='zh')            # Forecast plot / 预测图
pipeline.plot_leaderboard(lang='zh')      # Leaderboard chart / 排行榜图

router.plot(n=12, lang='zh')              # SmartRouter forecast / 智能路由器预测图
router.plot_leaderboard(lang='zh')        # SmartRouter leaderboard / 智能路由器排行榜
```

---

## Interval Prediction / 区间预测

PipelineTS uses Conformal Prediction for distribution-free prediction intervals with coverage guarantees.
PipelineTS 使用保形预测（Conformal Prediction）生成无分布假设的预测区间，具有覆盖率保证。

For neural network models, Conformalized Quantile Regression (CQR) provides adaptive, input-dependent intervals.
对于神经网络模型，保形分位数回归（CQR）提供自适应的、依赖输入的预测区间。

```python
# Single model with interval prediction
# 单模型区间预测
from PipelineTS.ml_model import TorchBoostingForestModel

model = TorchBoostingForestModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9,  # 90% prediction interval / 90% 预测区间
)
model.fit(data)
result = model.predict(10)
# result contains: value, value_lower, value_upper
# result 包含：value, value_lower, value_upper
```

```python
# Pipeline with interval prediction
# 管道区间预测
pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    quantile=0.9,
    include_models='ml',
)
pipeline.fit(data)
result = pipeline.predict(10)
```

---

## Multi-Quantile Prediction / 多分位数预测

Output prediction intervals at multiple coverage levels simultaneously. This is useful when you need to visualize uncertainty at different confidence levels.

同时输出多个覆盖水平的预测区间。当需要在不同置信水平下可视化不确定性时非常有用。

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, include_models=['torch_boosting_forest'],
)
pipeline.fit(data)

# Multi-quantile output / 多分位数输出
result = pipeline.predict_quantiles(n=10, levels=[0.5, 0.8, 0.95])
# Columns: date, value, value_q0.5_lower, value_q0.5_upper,
#          value_q0.8_lower, value_q0.8_upper, value_q0.95_lower, value_q0.95_upper
# 列: date, value, value_q0.5_lower, value_q0.5_upper, ...
```

```python
# SmartRouter also supports multi-quantile / SmartRouter 也支持多分位数
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(time_col='date', target_col='value', quantile=0.9)
router.fit(data)
result = router.predict_quantiles(n=12, levels=[0.5, 0.9])
```

---

## Multivariate Prediction / 多变量预测

ITransformerModel and SRSNetModel support three prediction modes:
ITransformerModel 和 SRSNetModel 支持三种预测模式：

| Mode / 模式 | target_col | feature_cols | Description / 描述 |
|---|---|---|---|
| Univariate / 单变量 | `'y'` | `None` | Classic single-variable / 经典单变量预测 |
| Multi-input Single-output / 多输入单输出 | `'y'` | `['a','b','y']` | Multiple features → one target / 多特征 → 单目标 |
| Multi-input Multi-output / 多输入多输出 | `['a','b']` | `['a','b','c']` | Multiple features → multiple targets / 多特征 → 多目标 |

```python
from PipelineTS.nn_model import ITransformerModel

model = ITransformerModel(
    time_col='date',
    target_col='value',
    feature_cols=['value', 'feature_a', 'feature_b'],
    lags=12,
    quantile=None,
    epochs=50
)
model.fit(data)
result = model.predict(10)
```

---

## Multi-Series (Panel Data) / 多序列（面板数据）

Native support for multiple time series via the `id_col` parameter. Each series gets its own scaler and predictions.

通过 `id_col` 参数原生支持多条时间序列。每条序列拥有独立的缩放器和预测结果。

```python
from PipelineTS.pipeline import ModelPipeline
import pandas as pd

# Panel data with multiple series / 包含多条序列的面板数据
# data has columns: date, value, store_id
# data 包含列: date, value, store_id

pipeline = ModelPipeline(
    time_col='date',
    target_col='value',
    lags=12,
    id_col='store_id',  # Enable multi-series / 启用多序列
    include_models=['torch_boosting_forest', 'torch_bagging_forest'],
)
pipeline.fit(data)

# Returns DataFrame with store_id column / 返回带有 store_id 列的 DataFrame
result = pipeline.predict(n=10)
```

```python
# SmartRouter with multi-series / SmartRouter 多序列
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col='date', target_col='value',
    id_col='store_id',
)
router.fit(data)
result = router.predict(10)
```

---

## Covariate Support / 协变量支持

GBDT, Prophet, and AutoARIMA models support known future covariates and past covariates.

GBDT、Prophet 和 AutoARIMA 模型支持已知未来协变量和历史协变量。

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    known_covariates=['holiday', 'promotion'],   # Known future values / 已知未来值
    past_covariates=['temperature'],              # Historical only / 仅历史值
    include_models=['torch_boosting_forest', 'prophet'],
)
pipeline.fit(data)  # data must contain covariate columns / data 必须包含协变量列

# Provide future covariates at prediction time / 预测时提供未来协变量
future_cov = pd.DataFrame({
    'holiday': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'promotion': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
})
result = pipeline.predict(n=10, future_covariates=future_cov)
```

---

## Incremental Learning / 增量学习

The `update()` method enables incremental training on new data without full retraining. Neural network models use warm-start (fewer epochs), while other models are efficiently refitted on combined data.

`update()` 方法支持在新数据上进行增量训练，无需完全重新训练。神经网络模型使用热启动（更少的轮次），其他模型在合并数据上高效重新拟合。

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models=['torch_boosting_forest', 'torch_bagging_forest'],
)
pipeline.fit(initial_data)

# Later, when new data arrives / 当新数据到达时
pipeline.update(new_data)

# Predictions now reflect the new data / 预测结果现在反映新数据
result = pipeline.predict(10)
```

```python
# SmartRouter also supports update() / SmartRouter 也支持 update()
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(time_col='date', target_col='value')
router.fit(initial_data)
router.update(new_data)
result = router.predict(12)
```

---

## Save and Load / 保存与加载

PipelineTS uses a custom binary format (.pts) with built-in integrity verification for saving and loading models, pipelines, and SmartRouters.

PipelineTS 使用自定义二进制格式（.pts）保存和加载模型、管道和智能路由器，具有内置完整性验证。

### Basic Usage / 基本用法

```python
from PipelineTS.io import save_model, load_model

# Save a model, pipeline, or SmartRouter (default .pts format)
# 保存模型、管道或智能路由器（默认 .pts 格式）
save_model('model.pts', model)
save_model('pipeline.pts', pipeline)
save_model('router.pts', router)

# Load back / 重新加载
model = load_model('model.pts')
pipeline = load_model('pipeline.pts')
router = load_model('router.pts')
```

### Advanced Features / 高级功能

```python
# Save with metadata and custom scaler
# 保存时包含元数据和自定义缩放器
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
metadata = {'version': '1.0', 'description': 'Electric production forecast'}

save_model('model.pts', model, scaler=scaler, metadata=metadata)

# Load with checksum verification (default enabled)
# 加载时校验和验证（默认启用）
model = load_model('model.pts', verify_checksum=True)

# Get file information without loading
# 获取文件信息而不加载
from PipelineTS.io import get_file_info

info = get_file_info('model.pts')
print(f"Model type: {info['model_type']}")
print(f"Format version: {info['version']}")
print(f"Metadata: {info['metadata']}")

# Verify file integrity
# 验证文件完整性
from PipelineTS.io import verify_file

is_valid = verify_file('model.pts')
print(f"File integrity: {is_valid}")
```

### Security Features / 安全特性

The .pts format includes multiple security layers:

.pts 格式包含多层安全保护：

- **SHA-256 checksums**: Global checksum over entire payload + per-section checksums
- **SHA-256 校验和**：整个载荷的全局校验和 + 每段校验和
- **Magic number validation**: Quick file type identification
- **魔数验证**：快速文件类型识别
- **Format versioning**: Forward/backward compatibility support
- **格式版本控制**：前向/后向兼容性支持
- **Atomic writes**: Temp file + os.replace for crash-safe writes
- **原子写入**：临时文件 + os.replace 确保崩溃安全写入


---

## Documentation / 文档

For detailed documentation, see the [docs/](docs/) directory:
详细文档请参阅 [docs/](docs/) 目录：

- [Installation Guide / 安装指南](docs/installation.md)
- [Quick Start Guide / 快速入门指南](docs/quickstart.md)
- [Model Reference / 模型参考](docs/models.md)
- [Pipeline Usage / 管道使用](docs/pipeline.md)
- [Preprocessing & Data / 数据预处理](docs/preprocessing.md)
- [Feature Engineering / 特征工程](docs/feature_engineering.md)
- [Evaluation & Metrics / 评估与指标](docs/evaluation.md)
- [Training Utilities / 训练工具](docs/training.md)
- [Prediction Utilities / 预测工具](docs/prediction.md)
- [Visualization / 可视化](docs/visualization.md)
- [Multivariate Prediction / 多变量预测](docs/multivariate.md)
- [Advanced Features / 高级功能](docs/advanced.md)
- [API Reference / API 参考](docs/api_reference.md)
- [Changelog / 更新日志](CHANGELOG.md)

---

## Tutorials / 教程

Interactive Jupyter notebook tutorials are available in the [tutorials/](tutorials/) directory:
交互式 Jupyter Notebook 教程位于 [tutorials/](tutorials/) 目录：

| # | Tutorial / 教程 | Description / 描述 |
|---|---|---|
| 01 | [Quick Start Guide](tutorials/01_QuickStart_Guide.ipynb) | Basic usage and core workflow / 基本用法和核心工作流 |
| 02 | [All Models Guide](tutorials/02_All_Models_Guide.ipynb) | Usage of all 24 models / 所有 24 个模型的用法 |
| 03 | [Multivariate Prediction](tutorials/03_Multivariate_Prediction.ipynb) | Multi-input/multi-output forecasting / 多输入/多输出预测 |
| 04 | [Advanced Pipeline](tutorials/04_Advanced_Pipeline.ipynb) | PipelineConfigs, scalers, metrics / 管道配置、缩放器、指标 |
| 05 | [Preprocessing & Data](tutorials/05_Preprocessing_and_Data.ipynb) | Datasets, scalers, sequence splitting / 数据集、缩放器、序列分割 |
| 06 | [Hyperparameter Tuning](tutorials/06_Hyperparameter_Tuning.ipynb) | Optuna integration for tuning / 使用 Optuna 进行超参数调优 |
| 07 | [Benchmarks](tutorials/07_Benchmarks.ipynb) | Model benchmarking across datasets / 跨数据集的模型基准测试 |
| 08 | [Visualization](tutorials/08_Visualization.ipynb) | Full visualization toolkit with Chinese fonts / 全面可视化工具包（含中文字体） |
| 09 | [Multi-Quantile Intervals](tutorials/09_Multi_Quantile_Intervals.ipynb) | Multi-level prediction intervals / 多分位数预测区间 |
| 10 | [Multi-Series & Covariates](tutorials/10_Multi_Series_Covariates.ipynb) | Panel data and external covariates / 面板数据与外部协变量 |
| 11 | [Incremental Learning](tutorials/11_Incremental_Learning.ipynb) | Update models with new data / 增量学习更新模型 |
| 12 | [SmartRouter & Pipeline](tutorials/12_SmartRouter_and_Pipeline.ipynb) | Core engines: ModelPipeline & SmartRouter deep dive / 核心引擎深度指南 |

---

## License / 许可证

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。
