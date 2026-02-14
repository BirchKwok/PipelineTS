# Preprocessing & Data
# 数据预处理

PipelineTS provides built-in datasets, data scalers, sequence splitting utilities, and evaluation metrics.
PipelineTS 提供内置数据集、数据缩放器、序列分割工具和评估指标。

---

## Built-in Datasets / 内置数据集

### Quick-load Functions / 快速加载函数

```python
from PipelineTS.dataset import (
    LoadElectricDataSets,
    LoadMessagesSentDataSets,
    LoadMessagesSentHourDataSets,
    LoadWebSales,
    LoadSupermarketIncoming,
)

# Each function returns a pandas DataFrame
# 每个函数返回一个 pandas DataFrame
data = LoadElectricDataSets()
print(data.shape, data.columns.tolist())
```

| Function / 函数 | Dataset / 数据集 | Columns / 列 |
|---|---|---|
| `LoadElectricDataSets()` | US Electric Production (monthly) / 美国电力生产（月度） | date, value |
| `LoadMessagesSentDataSets()` | Messages Sent (daily) / 消息发送量（日度） | date, ta, tb, tc |
| `LoadMessagesSentHourDataSets()` | Messages Sent (hourly) / 消息发送量（小时） | date, ta, tb, tc |
| `LoadWebSales()` | Web Sales (daily) / 网络销售（日度） | date, type_a, type_b, sales_cnt |
| `LoadSupermarketIncoming()` | Supermarket Incoming (daily) / 超市进货量（日度） | date, goods_cnt |

### BuiltInSeriesData / 全部内置数据

`BuiltInSeriesData` provides access to all built-in datasets, including ETT, M3, AirPassengers, etc.
`BuiltInSeriesData` 提供对所有内置数据集的访问，包括 ETT、M3、AirPassengers 等。

```python
from PipelineTS.dataset import BuiltInSeriesData

# List all datasets / 列出所有数据集
series_data = BuiltInSeriesData()

# Access by name or index / 通过名称或索引访问
etth1 = series_data['ETTh1']
air = series_data['AirPassengers']
```

### DataGenerator / 数据生成器

Generate synthetic time series data for testing.
生成合成时间序列数据，用于测试。

```python
from PipelineTS.dataset import DataGenerator

dg = DataGenerator()

# Trigonometric series / 三角函数序列
trig = dg.trigonometry_ds(size=100, random_state=42)

# White noise / 白噪声
noise = dg.white_noise(size=100, random_state=42)

# Random walk / 随机游走
walk = dg.random_walk(size=100, started_zero=True, random_state=42)
```

---

## Data Scalers / 数据缩放器

PipelineTS provides a unified `Scaler` interface supporting 4 scaling methods.
PipelineTS 提供统一的 `Scaler` 接口，支持 4 种缩放方法。

```python
from PipelineTS.preprocessing import Scaler
import numpy as np

X = np.random.randn(100, 1)

# Available methods / 可用方法：
# 'min_max'    - MinMaxScaler (scales to [0, 1])
#                MinMaxScaler（缩放到 [0, 1]）
# 'standard'   - StandardScaler (zero mean, unit variance)
#                StandardScaler（零均值，单位方差）
# 'quantile'   - QuantileTransformer
#                分位数变换
# 'gauss_rank' - GaussRankScaler (maps to Gaussian distribution)
#                高斯排名缩放（映射到高斯分布）

scaler = Scaler('min_max')
X_scaled = scaler.fit_transform(X)
X_recovered = scaler.inverse_transform(X_scaled)
```

### Using Scalers with Pipeline / 在管道中使用缩放器

```python
from PipelineTS.pipeline import ModelPipeline
from sklearn.preprocessing import StandardScaler

# Use built-in Scaler / 使用内置缩放器
pipeline = ModelPipeline(..., scaler=Scaler('gauss_rank'))

# Use sklearn scaler / 使用 sklearn 缩放器
pipeline = ModelPipeline(..., scaler=StandardScaler())

# True = MinMaxScaler (default) / True = MinMaxScaler（默认）
# None = no scaling / None = 不缩放
```

### Manual Scaling with Models / 手动缩放配合模型使用

```python
from PipelineTS.preprocessing import Scaler
from PipelineTS.ml_model import TorchBoostingForestModel

scaler = Scaler('min_max')
data['value'] = scaler.fit_transform(data['value'].values.reshape(-1, 1)).squeeze()

model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)
model.fit(data)
result = model.predict(10)

# Inverse transform predictions / 反向变换预测结果
for col in result.columns:
    if col != 'date':
        result[col] = scaler.inverse_transform(result[col].values.reshape(-1, 1)).squeeze()
```

---

## Sequence Splitting / 序列分割

Convert time series data into supervised learning format (X, y).
将时间序列数据转换为监督学习格式（X, y）。

### Univariate Splitting / 单变量分割

```python
from PipelineTS.spinesTS.preprocessing import split_series, train_test_split_ts
import numpy as np

series = np.sin(np.linspace(0, 4 * np.pi, 100))

# Split into (X, y) pairs / 分割为 (X, y) 对
X, y = split_series(series, in_features=10, out_features=5)
print(f"X.shape={X.shape}, y.shape={y.shape}")
# X.shape=(85, 10), y.shape=(85, 5)

# Time-series aware train/test split (preserves temporal order)
# 时序感知的训练/测试分割（保持时间顺序）
X_train, X_test, y_train, y_test = train_test_split_ts(X, y, train_size=0.8)
```

### Multivariate Splitting / 多变量分割

```python
from PipelineTS.spinesTS.preprocessing import split_series_multivariate

# 3D input: (timesteps, n_variables)
# 三维输入：（时间步数，变量数）
multi_series = np.random.randn(100, 3).astype(np.float32)
X, y = split_series_multivariate(multi_series, in_features=10, out_features=5)
print(f"X.shape={X.shape}, y.shape={y.shape}")
# X.shape=(85, 10, 3), y.shape=(85, 5, 3)
# Dimensions: (samples, timesteps, variables)
# 维度含义：（样本数，时间步，变量数）
```

---

## Missing Value Handling / 缺失值处理

`TimeSeriesMissingHandler` detects and fills missing values in time series data.
`TimeSeriesMissingHandler` 检测并填充时间序列数据中的缺失值。

It handles two types of missing data:
它处理两种类型的缺失数据：

- **Explicit NaN**: Actual NaN values in value columns. / 值列中的实际 NaN 值。
- **Implicit gaps**: Missing timestamps in the time column. / 时间列中缺失的时间戳。

```python
from PipelineTS.preprocessing import TimeSeriesMissingHandler

handler = TimeSeriesMissingHandler(time_col='date')

# Detect missing values / 检测缺失值
report = handler.fit(data, value_cols=['value'])
print(f"Implicit gaps: {report['n_implicit_gaps']}")
print(f"Explicit NaN:  {report['n_explicit_nan']}")
print(f"Completeness:  {report['completeness_ratio']:.2%}")
print(f"Gap locations:  {report['gap_timestamps'][:5]}")

# Fill missing values / 填充缺失值
filled = handler.transform(data, method='linear', fill_implicit_gaps=True)
```

**Available fill methods / 可用填充方法：**

| Method / 方法 | Description / 描述 |
|---|---|
| `'linear'` | Linear interpolation (default) / 线性插值（默认） |
| `'ffill'` | Forward fill (last observation carried forward) / 前向填充（用上一个观测值） |
| `'bfill'` | Backward fill (next observation carried backward) / 后向填充（用下一个观测值） |
| `'spline'` | Cubic spline interpolation / 三次样条插值 |
| `'zero'` | Fill with zeros / 零填充 |

---

## Outlier Detection & Handling / 异常值检测与处理

`TimeSeriesOutlierDetector` detects and handles anomalous values in time series data.
`TimeSeriesOutlierDetector` 检测并处理时间序列数据中的异常值。

```python
from PipelineTS.preprocessing import TimeSeriesOutlierDetector

# Choose a detection method / 选择检测方法
detector = TimeSeriesOutlierDetector(time_col='date', method='iqr', threshold=1.5)

# Detect outliers (returns boolean mask) / 检测异常值（返回布尔掩码）
mask = detector.fit(data, target_col='value')
print(f"Outliers found: {mask['value'].sum()}")

# Handle outliers with a strategy / 使用策略处理异常值
cleaned = detector.transform(data, target_col='value', strategy='clip')
```

**Detection methods / 检测方法：**

| Method / 方法 | Description / 描述 |
|---|---|
| `'iqr'` | Interquartile range: values outside Q1 - threshold×IQR to Q3 + threshold×IQR / 四分位距法 |
| `'zscore'` | Global z-score: \|z\| > threshold / 全局 Z 分数 |
| `'rolling_zscore'` | Rolling window z-score (detects local anomalies) / 滚动窗口 Z 分数（检测局部异常） |
| `'grubbs'` | Grubbs' test for statistical outlier detection / Grubbs 检验 |

**Handling strategies / 处理策略：**

| Strategy / 策略 | Description / 描述 |
|---|---|
| `'clip'` | Clip outliers to the boundary values / 将异常值截断到边界值 |
| `'nan'` | Replace outliers with NaN / 将异常值替换为 NaN |
| `'median'` | Replace with rolling median / 用滚动中位数替换 |
| `'linear'` | Replace with linear interpolation / 用线性插值替换 |

---

## Data Quality Report / 数据质量报告

`TimeSeriesDataQualityReport` generates a comprehensive health check of time series data.
`TimeSeriesDataQualityReport` 生成时间序列数据的全面健康检查报告。

```python
from PipelineTS.preprocessing import TimeSeriesDataQualityReport

reporter = TimeSeriesDataQualityReport(time_col='date', target_col='value')

# Generate report dict / 生成报告字典
report = reporter.fit(data)
# Keys: 'overview', 'time_analysis', 'value_analysis', 'missing_analysis', 'issues'

# Print formatted report / 打印格式化报告
reporter.report(data)
```

The report includes:
报告包含：

- **Overview / 概览**: row count, column count, time range, duration.
- **Time analysis / 时间分析**: frequency, regularity, gap detection.
- **Value analysis / 值分析**: statistics, distribution, outlier count.
- **Missing analysis / 缺失分析**: NaN count per column, completeness ratio.
- **Issues / 问题**: automatically detected data quality issues with severity levels.

---

## Stationarity Tests / 平稳性检验

`StationarityTest` wraps ADF and KPSS tests with a unified interface and actionable output.
`StationarityTest` 封装了 ADF 和 KPSS 检验，提供统一接口和可操作的输出。

```python
from PipelineTS.preprocessing import StationarityTest

tester = StationarityTest(significance_level=0.05)

# Individual tests / 单独检验
adf_result = tester.adf_test(data['value'].values)
kpss_result = tester.kpss_test(data['value'].values)

# Combined test with conclusion / 联合检验并给出结论
result = tester.fit(data['value'].values)
print(result['conclusion'])        # 'stationary' / 'trend_stationary' / 'difference_stationary' / 'non_stationary'
print(result['suggested_action'])  # e.g. 'No differencing needed.' / 例如 '不需要差分。'

# Auto-suggest differencing order / 自动建议差分阶数
d = tester.suggest_differencing(data['value'].values, max_d=2)
print(f"Suggested d={d}")
```

**Combined test interpretation / 联合检验解读：**

| ADF | KPSS | Conclusion / 结论 |
|---|---|---|
| Stationary / 平稳 | Stationary / 平稳 | `'stationary'` — No action needed / 无需操作 |
| Stationary / 平稳 | Non-stationary / 非平稳 | `'trend_stationary'` — Consider detrending / 考虑去趋势 |
| Non-stationary / 非平稳 | Stationary / 平稳 | `'difference_stationary'` — Apply d=1 / 应用一阶差分 |
| Non-stationary / 非平稳 | Non-stationary / 非平稳 | `'non_stationary'` — Apply d=1 or d=2 / 应用一阶或二阶差分 |

---

## Frequency Detection / 频率检测

`FrequencyDetector` auto-detects sampling frequency and dominant seasonal periods.
`FrequencyDetector` 自动检测采样频率和主要季节性周期。

```python
from PipelineTS.preprocessing.time_series_analysis import FrequencyDetector

detector = FrequencyDetector(time_col='date')
info = detector.fit(data, target_col='value')

print(f"Frequency: {info['freq']}")                   # e.g. 'D', 'h' / 例如 'D', 'h'
print(f"Timedelta: {info['freq_timedelta']}")          # e.g. Timedelta('1 days')
print(f"Regular:   {info['is_regular']}")              # True / False
print(f"Dominant periods: {info['dominant_periods']}")  # e.g. [30, 7, 365] (via FFT)
```

---

## Time Series Split / 时间序列分割

`TimeSeriesSplit` provides time-aware train/test splitting that preserves temporal ordering (unlike sklearn's random split).
`TimeSeriesSplit` 提供保持时间顺序的训练/测试分割（不同于 sklearn 的随机分割）。

```python
from PipelineTS.preprocessing import TimeSeriesSplit

# Simple split (last 20% as test) / 简单分割（最后 20% 为测试集）
train, test = TimeSeriesSplit.split(data, time_col='date', test_size=0.2)

# Expanding window cross-validation / 扩展窗口交叉验证
# Training set grows from the start / 训练集从开头持续增长
for train_df, test_df in TimeSeriesSplit.expanding_window(
    data, time_col='date', min_train_size=100, test_size=20, step=10
):
    model.fit(train_df)
    pred = model.predict(len(test_df))

# Sliding window cross-validation / 滑动窗口交叉验证
# Fixed-size training window that slides forward / 固定大小的训练窗口向前滑动
for train_df, test_df in TimeSeriesSplit.sliding_window(
    data, time_col='date', train_size=100, test_size=20, step=10
):
    model.fit(train_df)
    pred = model.predict(len(test_df))
```

---

## Evaluation Metrics / 评估指标

### Point Prediction Metrics / 点预测指标

```python
from PipelineTS.spinesTS.metrics import mae, mse, rmse, wmape
from PipelineTS.metrics import mape, smape, mase, r2_score, medae
import numpy as np

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

print(f"MAE:   {mae(y_true, y_pred):.4f}")    # Mean Absolute Error / 平均绝对误差
print(f"MSE:   {mse(y_true, y_pred):.4f}")    # Mean Squared Error / 均方误差
print(f"RMSE:  {rmse(y_true, y_pred):.4f}")   # Root Mean Squared Error / 均方根误差
print(f"WMAPE: {wmape(y_true, y_pred):.4f}")  # Weighted MAPE / 加权 MAPE
print(f"MAPE:  {mape(y_true, y_pred):.4f}")   # Mean Absolute Percentage Error / 平均绝对百分比误差
print(f"sMAPE: {smape(y_true, y_pred):.4f}")  # Symmetric MAPE / 对称 MAPE
print(f"R²:    {r2_score(y_true, y_pred):.4f}") # Coefficient of determination / 决定系数
print(f"MedAE: {medae(y_true, y_pred):.4f}")  # Median Absolute Error / 中位绝对误差

# MASE requires training data / MASE 需要训练数据
y_train = np.arange(10, dtype=np.float64)
print(f"MASE:  {mase(y_true, y_pred, y_train, seasonality=1):.4f}")
```

| Metric / 指标 | Import / 导入 | Description / 描述 |
|---|---|---|
| MAE | `spinesTS.metrics` | Mean Absolute Error / 平均绝对误差 |
| MSE | `spinesTS.metrics` | Mean Squared Error / 均方误差 |
| RMSE | `spinesTS.metrics` | Root Mean Squared Error / 均方根误差 |
| WMAPE | `spinesTS.metrics` | Weighted Mean Absolute Percentage Error / 加权平均绝对百分比误差 |
| MAPE | `PipelineTS.metrics` | Mean Absolute Percentage Error / 平均绝对百分比误差 |
| sMAPE | `PipelineTS.metrics` | Symmetric MAPE / 对称 MAPE |
| MASE | `PipelineTS.metrics` | Mean Absolute Scaled Error / 平均绝对缩放误差 |
| R² | `PipelineTS.metrics` | Coefficient of Determination / 决定系数 |
| MedAE | `PipelineTS.metrics` | Median Absolute Error / 中位绝对误差 |

### Interval Prediction Metrics / 区间预测指标

```python
from PipelineTS.metrics import quantile_acc, picp, pinaw, winkler_score

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
lower  = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
upper  = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

# Coverage rate / 覆盖率
print(f"Coverage:    {quantile_acc(y_true, lower, upper):.2%}")
print(f"PICP:        {picp(y_true, lower, upper):.4f}")       # Prediction Interval Coverage Probability
print(f"PINAW:       {pinaw(y_true, lower, upper):.4f}")      # Normalized Average Width / 归一化平均宽度
print(f"Winkler:     {winkler_score(y_true, lower, upper, alpha=0.1):.4f}")  # Lower is better / 越低越好
```

| Metric / 指标 | Description / 描述 |
|---|---|
| `quantile_acc` / `picp` | Prediction interval coverage rate (higher = better) / 预测区间覆盖率（越高越好） |
| `pinaw` | Normalized average width (lower = better, tighter intervals) / 归一化平均宽度（越低越好，区间越紧） |
| `winkler_score` | Rewards narrow intervals, penalizes missed coverage (lower = better) / 奖励窄区间，惩罚遗漏覆盖（越低越好） |

---

## Visualization / 可视化

### Plot Time Series / 绘制时间序列

```python
from PipelineTS.plot import plot_data_period

# Plot train data and predictions / 绘制训练数据和预测结果
plot_data_period(
    train_data, prediction,
    time_col='date',
    target_col='value',
    labels=['Train', 'Prediction']
)
```
