# Model Reference
# 模型参考

PipelineTS includes 24 built-in time series forecasting models across three categories.
PipelineTS 包含 24 个内置时间序列预测模型，分为三大类。

All models share a unified API: `fit(data)` for training and `predict(n)` for forecasting.
所有模型共享统一的 API：`fit(data)` 用于训练，`predict(n)` 用于预测。

---

## Common Parameters / 通用参数

The following parameters are shared by all models:
以下参数为所有模型共有：

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `time_col` | str | Name of the time column / 时间列名 |
| `target_col` | str or list | Name of the target column(s) / 目标列名（或列表） |
| `lags` | int | Number of past time steps used as input features / 用作输入特征的历史时间步数 |
| `quantile` | float or None | Coverage level for prediction intervals (e.g., 0.9 for 90%). None = point prediction only / 预测区间覆盖率（如 0.9 表示 90%）。None = 仅点预测 |
| `random_state` | int | Random seed for reproducibility / 随机种子，用于可复现性 |

---

## Neural Network Models / 神经网络模型

All NN models additionally support:
所有 NN 模型还支持以下参数：

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `epochs` | 1000 | Maximum training epochs / 最大训练轮数 |
| `patience` | 100 | Early stopping patience / 早停耐心值 |
| `verbose` | False | Whether to show training progress / 是否显示训练进度 |
| `learning_rate` | 0.001 | Learning rate / 学习率 |

### NLinearModel

Simple linear mapping model. The fastest NN model, suitable as a baseline.
简单线性映射模型。最快的神经网络模型，适合作为基准。

```python
from PipelineTS.nn_model import NLinearModel

model = NLinearModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, epochs=50, patience=10, verbose=False
)
model.fit(data)
result = model.predict(10)
```

### DLinearModel

Decomposition linear model that separates trend and seasonal components.
分解线性模型，将序列分解为趋势和季节性分量。

```python
from PipelineTS.nn_model import DLinearModel

model = DLinearModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, epochs=50, patience=10, verbose=False
)
```

### NBeatsModel

N-BEATS architecture supporting both generic and interpretable modes.
N-BEATS 架构，支持通用和可解释两种模式。

```python
from PipelineTS.nn_model import NBeatsModel

model = NBeatsModel(
    time_col='date', target_col='value', lags=12,
    generic_architecture=True,  # True: generic, False: interpretable
                                # True: 通用模式, False: 可解释模式
    num_stacks=2, num_blocks=1, num_layers=2, layer_widths=64,
    quantile=0.9, epochs=50, verbose=False
)
```

### NHitsModel

N-HiTS model with hierarchical interpolation for efficient multi-step forecasting.
N-HiTS 模型，使用分层插值结构提高多步预测效率。

```python
from PipelineTS.nn_model import NHitsModel

model = NHitsModel(
    time_col='date', target_col='value', lags=12,
    num_stacks=2, num_blocks=1, num_layers=2, layer_widths=64,
    quantile=0.9, epochs=50, verbose=False
)
```

### TFTModel

Temporal Fusion Transformer combining LSTM and multi-head attention.
时序融合 Transformer，结合 LSTM 和多头注意力机制。

```python
from PipelineTS.nn_model import TFTModel

model = TFTModel(
    time_col='date', target_col='value', lags=12,
    hidden_size=32, lstm_layers=1, n_heads=2,
    quantile=0.9, epochs=50, verbose=False
)
```

### TransformerModel

Classic Transformer encoder architecture for time series.
经典 Transformer 编码器架构。

```python
from PipelineTS.nn_model import TransformerModel

model = TransformerModel(
    time_col='date', target_col='value', lags=12,
    d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64,
    quantile=0.9, epochs=50, verbose=False
)
```

### TiDEModel

Time-series Dense Encoder with fully-connected encoder-decoder structure.
时序密集编码器，基于全连接的编解码器结构。

```python
from PipelineTS.nn_model import TiDEModel

model = TiDEModel(
    time_col='date', target_col='value', lags=12,
    num_encoder_layers=2, num_decoder_layers=2,
    hidden_size=64, decoder_output_dim=16,
    quantile=0.9, epochs=50, verbose=False
)
```

### GAUModel

Gated Attention Unit model with gated attention mechanism.
门控注意力单元模型，使用门控注意力机制。

```python
from PipelineTS.nn_model import GAUModel

model = GAUModel(
    time_col='date', target_col='value', lags=12,
    level=3,
    quantile=0.9, epochs=50, verbose=False
)
```

### StackingRNNModel

RWKV (linear RNN) encoder with gated residual blocks and RevIN normalization. Uses parallel linear temporal mixing (no sequential recurrence), followed by gated residual refinement with SiLU activation, plus a direct residual shortcut.
RWKV（线性 RNN）编码器 + 门控残差块 + RevIN 归一化。使用并行线性时序混合（无顺序递归），经过带 SiLU 激活的门控残差精炼，加上直接残差快捷连接。

```python
from PipelineTS.nn_model import StackingRNNModel

model = StackingRNNModel(
    time_col='date', target_col='value', lags=12,
    blocks=3,           # Number of gated residual blocks / 门控残差块数量
    d_model=48,         # Hidden dimension / 隐藏层维度
    quantile=0.9, epochs=50, verbose=False
)
```

### Time2VecModel

Trend-seasonal decomposition combined with StableTime2Vec periodic encoding and RWKV temporal mixing. The input is decomposed via moving average into trend and seasonal components; the trend path uses a linear projection while the seasonal path applies log-spaced Time2Vec periodic features followed by RWKV encoder blocks. Includes RevIN normalization and direct residual shortcut.
趋势-季节分解 + StableTime2Vec 周期编码 + RWKV 时序混合。输入通过移动平均分解为趋势和季节分量；趋势路径使用线性投影，季节路径使用对数间距的 Time2Vec 周期特征 + RWKV 编码器。包含 RevIN 归一化和直接残差快捷连接。

```python
from PipelineTS.nn_model import Time2VecModel

model = Time2VecModel(
    time_col='date', target_col='value', lags=12,
    num_layers=2,       # Number of RWKV blocks / RWKV 块数量
    quantile=0.9, epochs=50, verbose=False
)
```

### PatchRNNModel

Patch-based RNN that segments the input sequence into patches before feeding to LSTM.
基于 Patch 的 RNN，将输入序列分块后输入 LSTM。

```python
from PipelineTS.nn_model import PatchRNNModel

model = PatchRNNModel(
    time_col='date', target_col='value', lags=12,
    kernel_size=4,
    quantile=0.9, epochs=50, verbose=False
)
```

### TCNModel

Temporal Convolutional Network with dilated causal convolutions.
时序卷积网络，使用膨胀因果卷积。

```python
from PipelineTS.nn_model import TCNModel

model = TCNModel(
    time_col='date', target_col='value', lags=12,
    kernel_size=3,
    quantile=0.9, epochs=50, verbose=False
)
```

### ITransformerModel

Inverted Transformer that treats each variable as a token. Supports multivariate prediction.
反转 Transformer，将每个变量视为一个 token。支持多变量预测。

```python
from PipelineTS.nn_model import ITransformerModel

model = ITransformerModel(
    time_col='date', target_col='value', lags=12,
    d_model=32, n_heads=2, d_ff=64, e_layers=1,
    feature_cols=None,  # Set for multivariate mode / 设置为多变量模式
    quantile=0.9, epochs=50, verbose=False
)
```

### SRSNetModel

Selective Representation Space Network with multi-scale adaptive patches. Supports multivariate prediction.
选择性表征空间网络，使用多尺度自适应 patch。支持多变量预测。

```python
from PipelineTS.nn_model import SRSNetModel

model = SRSNetModel(
    time_col='date', target_col='value', lags=12,
    d_model=32, n_heads=2,
    feature_cols=None,  # Set for multivariate mode / 设置为多变量模式
    quantile=0.9, epochs=50, verbose=False
)
```

---

## Machine Learning Models / 机器学习模型

ML models are based on gradient boosting and ensemble methods. They typically train faster than NN models.
ML 模型基于梯度提升和集成方法。通常比神经网络模型训练更快。

All ML models automatically build rich lag features (26+ features per window) including statistics, trends, and autocorrelation.
所有 ML 模型自动构建丰富的滞后特征（每个窗口 26+ 个特征），包括统计量、趋势和自相关。

### LightGBMModel

```python
from PipelineTS.ml_model import LightGBMModel

model = LightGBMModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, n_estimators=200, verbose=-1
)
model.fit(data)
result = model.predict(10)
```

### XGBoostModel

```python
from PipelineTS.ml_model import XGBoostModel

model = XGBoostModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, n_estimators=200, verbose=0
)
```

### CatBoostModel

```python
from PipelineTS.ml_model import CatBoostModel

model = CatBoostModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, iterations=200, verbose=False
)
```

### RandomForestModel

```python
from PipelineTS.ml_model import RandomForestModel

model = RandomForestModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, n_estimators=200, random_state=42
)
```

### WideGBRTModel

Wide-table GBRT with automatically constructed rich time series features. Supports differencing.
宽表 GBRT 模型，自动构建丰富的时序特征。支持差分操作。

```python
from PipelineTS.ml_model import WideGBRTModel

model = WideGBRTModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, n_estimators=200, verbose=-1,
    differential_n=1,  # Order of differencing / 差分阶数
)
```

### MultiOutputRegressorModel / MultiStepRegressorModel / RegressorChainModel

Multi-output regression wrappers for multi-step forecasting.
多输出回归封装器，用于多步预测。

```python
from PipelineTS.ml_model import (
    MultiOutputRegressorModel,
    MultiStepRegressorModel,
    RegressorChainModel
)

# All share the same interface / 都共享相同接口
model = MultiOutputRegressorModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, verbose=-1
)
model.fit(data)
result = model.predict(10)
```

You can specify a custom estimator:
可以指定自定义估计器：

```python
from xgboost import XGBRegressor

model = MultiOutputRegressorModel(
    time_col='date', target_col='value', lags=12,
    estimator=XGBRegressor, verbose=0
)
```

---

## Statistical Models / 统计模型

### ProphetModel

Custom Prophet-like decomposable time series model (not Facebook Prophet). Uses piecewise linear trend with automatic changepoint detection, Fourier-based seasonality with FFT auto-detection, and optional causal rolling lag features (7 features: rolling mean, std, trend slope, momentum, half-ratio, EMA, autocorrelation). All parameters solved via ridge regression (closed-form), making it 100x+ faster than Facebook Prophet.
自定义类 Prophet 可分解时序模型（非 Facebook Prophet）。使用分段线性趋势 + 自动变点检测、基于傅里叶的季节性 + FFT 自动检测，以及可选的因果滚动滞后特征（7 个特征：滚动均值、标准差、趋势斜率、动量、半比率、EMA、自相关）。所有参数通过岭回归（解析解）求解，比 Facebook Prophet 快 100 倍以上。

```python
from PipelineTS.statistic_model import ProphetModel

model = ProphetModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9,
    auto_seasonality=True,       # Auto-detect seasonality via FFT / 通过 FFT 自动检测季节性
    use_lag_features=True,       # Enable causal rolling lag features / 启用因果滚动滞后特征
    lag_window='auto',           # Auto-determine window size / 自动确定窗口大小
    changepoint_prior_scale=0.05, # Smaller = smoother trend / 越小趋势越平滑
)
model.fit(data, cv=2)
result = model.predict(10)
```

### AutoARIMAModel

Automatic ARIMA parameter search.
自动搜索最佳 ARIMA 参数。

```python
from PipelineTS.statistic_model import AutoARIMAModel

model = AutoARIMAModel(
    time_col='date', target_col='value', lags=12,
    start_p=0, max_p=3, start_q=0, max_q=3,
    seasonal=False, quantile=0.9
)
model.fit(data, cv=2)
result = model.predict(10)
```

---

## Prediction Intervals / 预测区间

All models support prediction intervals via the `quantile` parameter.
所有模型通过 `quantile` 参数支持预测区间。

- **ML and Statistical models**: Use Conformal Prediction with asymmetric intervals.
- **ML 和统计模型**：使用保形预测，生成非对称区间。

- **NN models**: Use Conformalized Quantile Regression (CQR) for adaptive, input-dependent intervals.
- **NN 模型**：使用保形分位数回归（CQR），生成自适应的、依赖输入的区间。

Set `quantile=None` for point predictions only.
设置 `quantile=None` 仅进行点预测。
