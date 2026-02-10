# PipelineTS

![PyPI](https://img.shields.io/pypi/v/PipelineTS)
![PyPI - License](https://img.shields.io/pypi/l/PipelineTS)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PipelineTS)
[![Downloads](https://pepy.tech/badge/pipelinets)](https://pepy.tech/project/pipelinets)
[![Downloads](https://pepy.tech/badge/pipelinets/month)](https://pepy.tech/project/pipelinets)
[![Downloads](https://pepy.tech/badge/pipelinets/week)](https://pepy.tech/project/pipelinets)

One-stop time series analysis tool, supporting data preprocessing, feature engineering, model training, model evaluation, and forecasting.
一站式时间序列分析工具，支持数据预处理、特征工程、模型训练、模型评估与预测。

Built on top of spinesTS, it provides a unified interface for 24 time series models with automatic model selection, conformal prediction intervals, and multivariate forecasting.
基于 spinesTS 构建，提供 24 种时间序列模型的统一接口，支持自动模型选择、保形预测区间和多变量预测。

---

## Table of Contents / 目录

- [Features / 特性](#features--特性)
- [Installation / 安装](#installation--安装)
- [Quick Start / 快速开始](#quick-start--快速开始)
- [Available Models / 可用模型](#available-models--可用模型)
- [ModelPipeline / 模型管道](#modelpipeline--模型管道)
- [Interval Prediction / 区间预测](#interval-prediction--区间预测)
- [Multivariate Prediction / 多变量预测](#multivariate-prediction--多变量预测)
- [Save and Load / 保存与加载](#save-and-load--保存与加载)
- [Documentation / 文档](#documentation--文档)
- [Tutorials / 教程](#tutorials--教程)
- [License / 许可证](#license--许可证)

---

## Features / 特性

- **24 built-in models**: 14 neural network, 7 machine learning, 2 statistical, and 1 ensemble pipeline model.
- **24 个内置模型**：14 个神经网络、7 个机器学习、2 个统计模型和 1 个集成管道模型。

- **Automatic model selection**: `ModelPipeline` trains and compares all models, automatically selecting the best one.
- **自动模型选择**：`ModelPipeline` 训练并比较所有模型，自动选出最佳模型。

- **Conformal prediction intervals**: Industry-standard distribution-free prediction intervals with coverage guarantees.
- **保形预测区间**：行业标准的无分布预测区间，具有覆盖率保证。

- **CQR for neural networks**: Conformalized Quantile Regression provides adaptive, input-dependent intervals for NN models.
- **神经网络 CQR**：保形分位数回归为神经网络模型提供自适应的、依赖输入的预测区间。

- **Multivariate forecasting**: ITransformer and SRSNet support multi-input/multi-output prediction modes.
- **多变量预测**：ITransformer 和 SRSNet 支持多输入/多输出预测模式。

- **Rich feature engineering**: Automatic lag feature extraction (26+ features per window) for GBDT/ML models and Prophet.
- **丰富的特征工程**：为 GBDT/ML 模型和 Prophet 自动提取滞后特征（每个窗口 26+ 个特征）。

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
from PipelineTS.ml_model import LightGBMModel

# Initialize and train a LightGBM model
# 初始化并训练 LightGBM 模型
model = LightGBMModel(
    time_col=time_col,
    target_col=target_col,
    lags=12,
    quantile=0.9,
    verbose=-1
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

### Visualize Results / 可视化结果

```python
from PipelineTS.plot import plot_data_period

plot_data_period(
    data, result,
    time_col=time_col,
    target_col=target_col
)
```

---

## Available Models / 可用模型

### Neural Network Models / 神经网络模型 (14)

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

### Machine Learning Models / 机器学习模型 (7)

| Model / 模型 | Key / 键名 | Description / 描述 |
|---|---|---|
| LightGBMModel | `lightgbm` | LightGBM gradient boosting / LightGBM 梯度提升 |
| XGBoostModel | `xgboost` | XGBoost gradient boosting / XGBoost 梯度提升 |
| CatBoostModel | `catboost` | CatBoost gradient boosting / CatBoost 梯度提升 |
| RandomForestModel | `random_forest` | Random Forest regressor / 随机森林回归 |
| WideGBRTModel | `wide_gbrt` | Wide-table GBRT with rich features / 宽表 GBRT + 丰富特征 |
| MultiOutputRegressorModel | `multi_output_model` | Multi-output regressor / 多输出回归 |
| MultiStepRegressorModel | `multi_step_model` | Multi-step regressor / 多步回归 |
| RegressorChainModel | `regressor_chain` | Regressor chain / 回归链 |

### Statistical Models / 统计模型 (2)

| Model / 模型 | Key / 键名 | Description / 描述 |
|---|---|---|
| ProphetModel | `prophet` | Custom Prophet-like model with ridge regression / 自定义类 Prophet 岭回归模型 |
| AutoARIMAModel | `auto_arima` | Auto ARIMA parameter search / 自动 ARIMA 参数搜索 |

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
pipeline = ModelPipeline(..., include_models=['lightgbm', 'xgboost', 'd_linear'])
```

### PipelineConfigs / 管道配置

Use `PipelineConfigs` to create multiple model variants with different hyperparameters.
使用 `PipelineConfigs` 创建具有不同超参数的多个模型变体。

```python
from PipelineTS.pipeline import PipelineConfigs

configs = PipelineConfigs([
    ('lightgbm', 'lgbm_v1', {'init_configs': {'n_estimators': 100}}),
    ('lightgbm', 'lgbm_v2', {'init_configs': {'n_estimators': 300}}),
])

pipeline = ModelPipeline(..., configs=configs)
```

### Double-underscore Syntax / 双下划线语法

Pass model-specific parameters directly via double-underscore syntax.
通过双下划线语法直接传递模型特定参数。

```python
pipeline = ModelPipeline(
    ...,
    lightgbm__n_estimators=200,
    xgboost__verbose=0,
    d_linear__lags=50,
)
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
from PipelineTS.ml_model import LightGBMModel

model = LightGBMModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9,  # 90% prediction interval / 90% 预测区间
    verbose=-1
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

## Save and Load / 保存与加载

```python
from PipelineTS.io import save_model, load_model

# Save a model or pipeline / 保存模型或管道
save_model('model.zip', model)

# Load a model or pipeline / 加载模型或管道
model = load_model('model.zip')
```

---

## Documentation / 文档

For detailed documentation, see the [docs/](docs/) directory:
详细文档请参阅 [docs/](docs/) 目录：

- [Installation Guide / 安装指南](docs/installation.md)
- [Quick Start Guide / 快速入门指南](docs/quickstart.md)
- [Model Reference / 模型参考](docs/models.md)
- [Pipeline Usage / 管道使用](docs/pipeline.md)
- [Preprocessing & Data / 数据预处理](docs/preprocessing.md)
- [Multivariate Prediction / 多变量预测](docs/multivariate.md)
- [Advanced Features / 高级功能](docs/advanced.md)
- [API Reference / API 参考](docs/api_reference.md)

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

---

## License / 许可证

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。
