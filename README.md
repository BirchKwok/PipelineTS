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
- [Data Preprocessing / 数据预处理](#data-preprocessing--数据预处理)
- [Feature Engineering / 特征工程](#feature-engineering--特征工程)
- [Evaluation Metrics / 评估指标](#evaluation-metrics--评估指标)
- [Model Evaluation / 模型评估](#model-evaluation--模型评估)
- [Training Utilities / 训练工具](#training-utilities--训练工具)
- [Prediction Utilities / 预测工具](#prediction-utilities--预测工具)
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

- **Data preprocessing toolkit**: Missing value detection & interpolation, outlier detection & handling, data quality reporting, stationarity tests, frequency auto-detection, and time series train/test splitting.
- **数据预处理工具箱**：缺失值检测与插值、异常值检测与处理、数据质量报告、平稳性检验、频率自动检测、时间序列训练/测试分割。

- **Comprehensive evaluation metrics**: MAPE, sMAPE, MASE, R², MedAE for point forecasts; PICP, PINAW, Winkler score for prediction intervals.
- **全面的评估指标**：点预测指标（MAPE、sMAPE、MASE、R²、MedAE）；区间预测指标（PICP、PINAW、Winkler 分数）。

- **Model evaluation framework**: Walk-forward backtesting, residual analysis with diagnostics, and multi-model comparison visualization.
- **模型评估框架**：前向回测、残差诊断分析、多模型对比可视化。

- **User-facing feature engineering**: Unified API composing Fourier features, holiday features, rolling lag features, and calendar features.
- **面向用户的特征工程**：统一 API，组合傅里叶特征、节假日特征、滚动滞后特征和日历特征。

- **Training utilities**: Built-in AutoTune (Optuna / random search), weighted ensemble, and stacking ensemble.
- **训练工具**：内置 AutoTune（Optuna / 随机搜索）、加权集成、堆叠集成。

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
from PipelineTS.ml_model import LightGBMModel
from PipelineTS.spinesTS.metrics import mae

model = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
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
comp.add_result('LightGBM', y_true, y_pred_lgbm, lower=lower_lgbm, upper=upper_lgbm)
comp.add_result('XGBoost', y_true, y_pred_xgb, lower=lower_xgb, upper=upper_xgb)

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
from PipelineTS.ml_model import LightGBMModel
from PipelineTS.spinesTS.metrics import mae

tuner = AutoTune(
    model_class=LightGBMModel,
    time_col='date', target_col='value', lags=12,
    metric=mae, n_trials=30,
    fixed_params={'verbose': -1},
)

best_model, best_params, history = tuner.fit(data, search_space={
    'n_estimators': ('int', 50, 500),
    'learning_rate': ('float', 0.01, 0.3, True),  # True = log scale / True = 对数刻度
    'max_depth': ('int', 3, 10),
    'num_leaves': ('int', 15, 63),
})
```

### Ensemble Methods / 集成方法

```python
from PipelineTS.training import WeightedEnsemble, StackingEnsemble
from PipelineTS.ml_model import LightGBMModel, XGBoostModel

models = [
    ('lgbm', LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)),
    ('xgb', XGBoostModel(time_col='date', target_col='value', lags=12, verbose=0)),
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
from PipelineTS.ml_model import LightGBMModel

model = LightGBMModel(time_col='date', target_col='value', lags=12, verbose=-1)
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
- [Feature Engineering / 特征工程](docs/feature_engineering.md)
- [Evaluation & Metrics / 评估与指标](docs/evaluation.md)
- [Training Utilities / 训练工具](docs/training.md)
- [Prediction Utilities / 预测工具](docs/prediction.md)
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
