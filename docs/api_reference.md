# API Reference
# API 参考

Complete reference for all public classes and functions in PipelineTS.
PipelineTS 所有公共类和函数的完整参考。

---

## Pipeline / 管道

### `PipelineTS.pipeline.ModelPipeline`

The main pipeline class for automatic model selection.
用于自动模型选择的主管道类。

```python
ModelPipeline(
    time_col: str,
    target_col: str,
    lags: int,
    quantile: float | None = None,
    random_state: int = 0,
    metric: callable = mean_absolute_error,
    metric_less_is_better: bool = True,
    include_models: str | list | None = 'light',
    exclude_models: list | None = None,
    configs: PipelineConfigs | None = None,
    include_init_config_model: bool = True,
    scaler: bool | None | TransformerMixin = True,
    accelerator: str = 'auto',
    cv: int = 5,
    feature_cols: list | None = None,
    **model_init_kwargs,
)
```

**Methods / 方法:**

| Method / 方法 | Description / 描述 |
|---|---|
| `fit(data, valid_data=None)` | Train all models and return leaderboard / 训练所有模型并返回排行榜 |
| `predict(n, model_name=None, data=None)` | Predict n steps using best or specified model / 使用最佳或指定模型预测 n 步 |
| `get_model(model_name=None)` | Get the best or specified trained model / 获取最佳或指定的已训练模型 |
| `get_model_all_configs(model_name=None)` | Get all configs for a model / 获取模型的全部配置 |
| `list_all_available_models()` | Class method: list all model names / 类方法：列出所有模型名称 |

**Attributes / 属性:**

| Attribute / 属性 | Description / 描述 |
|---|---|
| `leader_board_` | DataFrame with model performance rankings / 模型性能排名 DataFrame |
| `best_model_` | The best performing model object / 最佳模型对象 |

---

### `PipelineTS.pipeline.PipelineConfigs`

Configuration class for creating model variants.
用于创建模型变体的配置类。

```python
PipelineConfigs(configs: list[tuple])
```

Each tuple format / 每个元组的格式:
- `(model_name, config_dict)` - Auto-named / 自动命名
- `(model_name, custom_name, config_dict)` - Custom-named / 自定义命名

---

## Neural Network Models / 神经网络模型

All located in `PipelineTS.nn_model`.
全部位于 `PipelineTS.nn_model`。

| Class / 类 | Import / 导入 |
|---|---|
| `NLinearModel` | `from PipelineTS.nn_model import NLinearModel` |
| `DLinearModel` | `from PipelineTS.nn_model import DLinearModel` |
| `NBeatsModel` | `from PipelineTS.nn_model import NBeatsModel` |
| `NHitsModel` | `from PipelineTS.nn_model import NHitsModel` |
| `TFTModel` | `from PipelineTS.nn_model import TFTModel` |
| `TransformerModel` | `from PipelineTS.nn_model import TransformerModel` |
| `TiDEModel` | `from PipelineTS.nn_model import TiDEModel` |
| `GAUModel` | `from PipelineTS.nn_model import GAUModel` |
| `StackingRNNModel` | `from PipelineTS.nn_model import StackingRNNModel` |
| `Time2VecModel` | `from PipelineTS.nn_model import Time2VecModel` |
| `PatchRNNModel` | `from PipelineTS.nn_model import PatchRNNModel` |
| `TCNModel` | `from PipelineTS.nn_model import TCNModel` |
| `ITransformerModel` | `from PipelineTS.nn_model import ITransformerModel` |
| `SRSNetModel` | `from PipelineTS.nn_model import SRSNetModel` |

**Common interface / 通用接口:**

```python
model = SomeModel(
    time_col: str,           # Time column name / 时间列名
    target_col: str | list,  # Target column(s) / 目标列
    lags: int,               # Input window size / 输入窗口大小
    quantile: float | None,  # Interval coverage / 区间覆盖率
    random_state: int,       # Random seed / 随机种子
    epochs: int = 1000,      # Max epochs / 最大轮数
    patience: int = 100,     # Early stopping / 早停
    verbose: bool = False,   # Show progress / 显示进度
    learning_rate: float = 0.001,  # Learning rate / 学习率
    feature_cols: list | None = None,  # For multivariate models / 多变量模型用
)

model.fit(data, valid_data=None)  # Train / 训练
model.predict(n, data=None)       # Predict / 预测
```

---

## Machine Learning Models / 机器学习模型

All located in `PipelineTS.ml_model`.
全部位于 `PipelineTS.ml_model`。

| Class / 类 | Import / 导入 |
|---|---|
| `LightGBMModel` | `from PipelineTS.ml_model import LightGBMModel` |
| `XGBoostModel` | `from PipelineTS.ml_model import XGBoostModel` |
| `CatBoostModel` | `from PipelineTS.ml_model import CatBoostModel` |
| `RandomForestModel` | `from PipelineTS.ml_model import RandomForestModel` |
| `WideGBRTModel` | `from PipelineTS.ml_model import WideGBRTModel` |
| `MultiOutputRegressorModel` | `from PipelineTS.ml_model import MultiOutputRegressorModel` |
| `MultiStepRegressorModel` | `from PipelineTS.ml_model import MultiStepRegressorModel` |
| `RegressorChainModel` | `from PipelineTS.ml_model import RegressorChainModel` |

**Common interface / 通用接口:**

```python
model = SomeMLModel(
    time_col: str,
    target_col: str,
    lags: int,
    quantile: float | None = None,
    random_state: int = 42,
    **model_specific_params,
)

model.fit(data)
model.predict(n, data=None)
```

---

## Statistical Models / 统计模型

All located in `PipelineTS.statistic_model`.
全部位于 `PipelineTS.statistic_model`。

| Class / 类 | Import / 导入 |
|---|---|
| `ProphetModel` | `from PipelineTS.statistic_model import ProphetModel` |
| `AutoARIMAModel` | `from PipelineTS.statistic_model import AutoARIMAModel` |

---

## Dataset / 数据集

All located in `PipelineTS.dataset`.
全部位于 `PipelineTS.dataset`。

| Function/Class / 函数/类 | Description / 描述 |
|---|---|
| `LoadElectricDataSets()` | Electric Production dataset / 电力生产数据集 |
| `LoadMessagesSentDataSets()` | Messages Sent (daily) / 消息发送量（日度） |
| `LoadMessagesSentHourDataSets()` | Messages Sent (hourly) / 消息发送量（小时） |
| `LoadWebSales()` | Web Sales dataset / 网络销售数据集 |
| `LoadSupermarketIncoming()` | Supermarket Incoming / 超市进货量 |
| `BuiltInSeriesData()` | Access all built-in datasets / 访问所有内置数据集 |
| `DataGenerator()` | Generate synthetic data / 生成合成数据 |

---

## Preprocessing / 预处理

### `PipelineTS.preprocessing.Scaler`

```python
Scaler(scaler_name: str)
# scaler_name: 'min_max' | 'standard' | 'quantile' | 'gauss_rank'

scaler.fit_transform(X)      # Fit and transform / 拟合并变换
scaler.transform(X)          # Transform / 变换
scaler.inverse_transform(X)  # Inverse transform / 反向变换
```

### Sequence Splitting / 序列分割

```python
from PipelineTS.spinesTS.preprocessing import (
    split_series,                # Univariate split / 单变量分割
    split_series_multivariate,   # Multivariate split / 多变量分割
    train_test_split_ts,         # Time-series train/test split / 时序训练/测试分割
)
```

---

## Metrics / 指标

### Point Metrics / 点指标

```python
from PipelineTS.spinesTS.metrics import mae, mse, rmse, wmape
```

| Function / 函数 | Description / 描述 |
|---|---|
| `mae(y_true, y_pred)` | Mean Absolute Error / 平均绝对误差 |
| `mse(y_true, y_pred)` | Mean Squared Error / 均方误差 |
| `rmse(y_true, y_pred)` | Root Mean Squared Error / 均方根误差 |
| `wmape(y_true, y_pred)` | Weighted Mean Absolute Percentage Error / 加权平均绝对百分比误差 |

### Interval Metrics / 区间指标

```python
from PipelineTS.metrics import quantile_acc

quantile_acc(y_true, lower, upper)  # Interval coverage rate / 区间覆盖率
```

---

## I/O

```python
from PipelineTS.io import save_model, load_model

save_model(path: str, model, scaler=None)  # Save model / 保存模型
load_model(path: str)                       # Load model / 加载模型
```

---

## Plotting / 绑图

```python
from PipelineTS.plot import plot_data_period

plot_data_period(
    data1: pd.DataFrame,      # First dataset (e.g., train) / 第一个数据集（如训练集）
    data2: pd.DataFrame,      # Second dataset (e.g., prediction) / 第二个数据集（如预测集）
    time_col: str,             # Time column name / 时间列名
    target_col: str,           # Target column name / 目标列名
    labels: list = None,       # Legend labels / 图例标签
    date_fmt: str = '%Y-%m-%d', # Date format / 日期格式
)
```
