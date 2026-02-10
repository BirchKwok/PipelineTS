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
from PipelineTS.ml_model import LightGBMModel

scaler = Scaler('min_max')
data['value'] = scaler.fit_transform(data['value'].values.reshape(-1, 1)).squeeze()

model = LightGBMModel(time_col='date', target_col='value', lags=12)
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

## Evaluation Metrics / 评估指标

### Point Prediction Metrics / 点预测指标

```python
from PipelineTS.spinesTS.metrics import mae, mse, rmse, wmape
import numpy as np

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

print(f"MAE:   {mae(y_true, y_pred):.4f}")    # Mean Absolute Error / 平均绝对误差
print(f"MSE:   {mse(y_true, y_pred):.4f}")    # Mean Squared Error / 均方误差
print(f"RMSE:  {rmse(y_true, y_pred):.4f}")   # Root Mean Squared Error / 均方根误差
print(f"WMAPE: {wmape(y_true, y_pred):.4f}")  # Weighted Mean Absolute Percentage Error / 加权平均绝对百分比误差
```

### Interval Prediction Metrics / 区间预测指标

```python
from PipelineTS.metrics import quantile_acc

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
lower  = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
upper  = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

# Compute interval coverage rate / 计算区间覆盖率
acc = quantile_acc(y_true, lower, upper)
print(f"Coverage: {acc:.2%}")  # 100.00%
```

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
