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
| `predict(n, model_name=None, data=None, future_covariates=None)` | Predict n steps using best or specified model / 使用最佳或指定模型预测 n 步 |
| `predict_quantiles(n, levels, model_name=None)` | Multi-quantile prediction / 多分位数预测 |
| `update(new_data)` | Incremental learning on new data / 在新数据上增量学习 |
| `plot(n=None, lang='zh', history_tail=None)` | Plot forecast from best model / 绘制最佳模型预测图 |
| `plot_leaderboard(lang='zh')` | Plot model leaderboard chart / 绘制模型排行榜图 |
| `get_model(model_name=None)` | Get the best or specified trained model / 获取最佳或指定的已训练模型 |
| `get_model_all_configs(model_name=None)` | Get all configs for a model / 获取模型的全部配置 |
| `list_all_available_models()` | Class method: list all model names / 类方法：列出所有模型名称 |
| `save(path)` | Save pipeline to zip file / 保存管道到 zip 文件 |
| `load(path)` | Static: load pipeline from zip / 静态方法：从 zip 加载管道 |

**Attributes / 属性:**

| Attribute / 属性 | Description / 描述 |
|---|---|
| `leader_board_` | DataFrame with model performance rankings / 模型性能排名 DataFrame |
| `best_model_` | The best performing model object / 最佳模型对象 |
| `failed_models` | List of failed model details / 失败模型详情列表 |
| `skipped_models` | List of skipped model names and reasons / 跳过的模型名称和原因列表 |

---

### `PipelineTS.pipeline.SmartRouter`

Intelligent routing system for automatic data profiling, model selection, and ensemble building.
智能路由系统，自动数据画像、模型选择和集成构建。

```python
SmartRouter(
    time_col: str,
    target_col: str,
    n_predict: int | None = None,
    max_models: int = 5,
    quantile: float | None = None,
    ensemble_strategy: str = 'auto',
    ensemble_top_k: int = 3,
    id_col: str | None = None,
    known_covariates: list | None = None,
    past_covariates: list | None = None,
    include_models: str | list | None = None,
    search_strategy: str = 'auto',
    hpo_strategy: str = 'none',
    hpo_n_trials: int = 10,
    hpo_timeout_per_model: float | None = None,
    time_limit: float | None = None,
    random_state: int = 0,
    verbose: bool = True,
)
```

**Key Parameters / 关键参数:**

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 描述 |
|---|---|---|---|
| `include_models` | str, list, None | None | Pin specific model(s); skips heuristic selection and screening / 指定模型；跳过启发式选择和筛选 |
| `search_strategy` | str | 'auto' | `'basic'` (no screening/exploration), `'auto'`, or `'thorough'` / 搜索策略 |
| `hpo_strategy` | str | 'none' | `'none'`, `'quick'` (≤5 trials), or `'full'` / HPO 策略 |

**Methods / 方法:**

| Method / 方法 | Description / 描述 |
|---|---|
| `fit(data)` | Profile data, select models, train, optionally build ensemble / 数据画像、选择模型、训练、可选构建集成 |
| `predict(n, use_ensemble=True, future_covariates=None)` | Predict using ensemble or best model / 使用集成或最佳模型预测 |
| `predict_quantiles(n, levels)` | Multi-quantile prediction / 多分位数预测 |
| `update(new_data)` | Incremental learning / 增量学习 |
| `plot(n=None, lang='zh')` | Plot forecast / 绘制预测图 |
| `plot_leaderboard(lang='zh')` | Plot leaderboard / 绘制排行榜 |
| `get_model(model_name=None)` | Get fitted model / 获取已训练模型 |
| `list_all_available_models()` | Class method: list all valid model names / 类方法：列出所有有效模型名称 |

**Attributes / 属性:**

| Attribute / 属性 | Description / 描述 |
|---|---|
| `strategy` | Selected strategy dict (models, lags, scaler, etc.) / 选定策略字典 |
| `leader_board_` | Model rankings / 模型排名 |
| `ensemble_` | EnsemblePredictor (if built) / 集成预测器 |
| `pipeline_` | Underlying ModelPipeline / 底层 ModelPipeline |
| `profile_` | DataProfile with data characteristics / 数据画像 |
| `include_models` | User-pinned model list (or None) / 用户指定模型列表（或 None） |

---

### `PipelineTS.pipeline.PipelineConfigs`

Configuration class for creating model variants with per-model settings.
用于创建模型变体并支持每模型设置的配置类。

```python
PipelineConfigs(configs: list[tuple])
```

Each tuple format / 每个元组的格式:
- `(model_name, config_dict)` - Auto-named / 自动命名
- `(model_name, custom_name, config_dict)` - Custom-named / 自定义命名

**Config dict keys / 配置字典键:**

| Key / 键 | Description / 描述 |
|---|---|
| `init_configs` | Model `__init__` parameters / 模型初始化参数 |
| `fit_configs` | Parameters passed to `fit()` / 传递给 `fit()` 的参数 |
| `predict_configs` | Parameters passed to `predict()` / 传递给 `predict()` 的参数 |
| `pipeline_configs` | Pipeline-level per-model settings / 管道级别每模型设置 |

**`pipeline_configs` supported keys / `pipeline_configs` 支持的键:**

| Key / 键 | Type / 类型 | Description / 描述 |
|---|---|---|
| `lags` | int | Per-model input window size / 每模型滞后窗口大小 |
| `scaler` | bool, None, or TransformerMixin | Per-model scaler (`True`=MinMaxScaler, `None`=disabled, or custom instance) / 每模型缩放器 |
| `differential_n` | int | Per-model differencing order / 每模型差分阶数 |
| `feature_cols` | list | Per-model feature columns / 每模型特征列 |

```python
from PipelineTS.pipeline import PipelineConfigs
from sklearn.preprocessing import StandardScaler

configs = PipelineConfigs([
    ('torch_boosting_forest', 'boost_std', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'lags': 20, 'scaler': StandardScaler()},
    }),
    ('torch_boosting_forest', 'boost_noscale', {
        'init_configs': {'n_trees': 64},
        'pipeline_configs': {'scaler': None},
    }),
])
```

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
| `DeepARModel` | `from PipelineTS.nn_model import DeepARModel` |
| `Chronos2Model` *(optional)* | `from PipelineTS.nn_model import Chronos2Model` |
| `Chronos2SynthModel` *(optional)* | `from PipelineTS.nn_model import Chronos2SynthModel` |
| `Chronos2SmallModel` *(optional)* | `from PipelineTS.nn_model import Chronos2SmallModel` |

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
    use_gtb: bool = False,   # Enable GlobalTemporalBlock / 启用全局时序块
    gtb_d_model: int = 64,   # GTB hidden dimension / GTB 隐藏维度
    routing_mode: str = 'static',  # 'static' or 'adaptive' (MoE) / 静态或自适应（MoE）
)

model.fit(data, valid_data=None)  # Train / 训练
model.predict(n, data=None)       # Predict / 预测
```

### Chronos-2 Family *(optional dependency / 可选依赖)*

Zero-shot foundation models wrapping Amazon/AutoGluon Chronos-2 pretrained models.
零样本基础模型，封装 Amazon/AutoGluon Chronos-2 预训练模型。

> Requires: `pip install chronos-forecasting`

| Class / 类 | Pipeline Key / 管道键名 | HuggingFace Path | Size / 大小 |
|---|---|---|---|
| `Chronos2Model` | `chronos_2` | `amazon/chronos-2` | 120M |
| `Chronos2SynthModel` | `chronos_2_synth` | `autogluon/chronos-2-synth` | 120M |
| `Chronos2SmallModel` | `chronos_2_small` | `autogluon/chronos-2-small` | 28M |

`ChronosModel` is a backward-compatible alias for `Chronos2Model`.
`ChronosModel` 是 `Chronos2Model` 的向后兼容别名。

```python
from PipelineTS.nn_model import Chronos2Model, Chronos2SynthModel, Chronos2SmallModel

model = Chronos2SmallModel(
    time_col: str,
    target_col: str,
    lags: int = 1,                   # API compatibility / API 兼容
    quantile: float | None = 0.9,    # Conformal interval coverage / 共形区间覆盖率
    device_map: str = 'auto',        # 'auto', 'cpu', 'cuda', 'mps'
)

model.fit(data, cv=5)                              # Store data + calibrate intervals / 存储数据 + 校准区间
model.predict(n, future_covariates=None)            # Zero-shot predict / 零样本预测
```

**Features / 特点:** Zero-shot (no training), multi-series (`id_col`), covariates support, conformal intervals.
**特点：** 零样本（无需训练）、多序列（`id_col`）、协变量支持、共形预测区间。

---

## Machine Learning Models / 机器学习模型

All located in `PipelineTS.ml_model`.
全部位于 `PipelineTS.ml_model`。

### GPU-Accelerated Tree Models / GPU 加速树模型

| Class / 类 | Import / 导入 | Description / 描述 |
|---|---|---|
| `TorchBoostingForestModel` | `from PipelineTS.ml_model import TorchBoostingForestModel` | Staged gradient boosting (MART/DART) / 分阶段梯度提升 |
| `TorchBaggingForestModel` | `from PipelineTS.ml_model import TorchBaggingForestModel` | Bagging forest with dropout / 带 Dropout 的袋装森林 |
| `DeepForestModel` | `from PipelineTS.ml_model import DeepForestModel` | Cascade multi-layer ensemble (gcForest) / 级联多层集成 |

```python
TorchBoostingForestModel(
    time_col: str,
    target_col: str,
    lags: int = 1,
    quantile: float | None = 0.9,
    accelerator: str | None = None,  # 'cuda', 'mps', 'cpu', or None (auto)
    n_trees: int = 64,
    tree_depth: int = 5,
    learning_rate: float = 0.08,
    n_epochs: int = 200,
    batch_size: int = 0,             # 0 = full batch
    early_stop_patience: int = 15,
    dropout: float = 0.0,
    weight_decay: float = 1e-4,
    boosting_stages: int = 3,
    boosting_shrinkage: float = 0.5,
    random_state: int | None = None,
    verbose: bool = False,
    auto_complexity: bool = False,   # Enable adaptive depth/trees auto-tuning
)
```

```python
TorchBaggingForestModel(
    time_col: str,
    target_col: str,
    lags: int = 1,
    quantile: float | None = 0.9,
    accelerator: str | None = None,
    n_trees: int = 128,
    tree_depth: int = 5,
    learning_rate: float = 0.08,
    n_epochs: int = 300,
    batch_size: int = 0,
    early_stop_patience: int = 15,
    dropout: float = 0.15,           # Tree-level dropout for decorrelation
    weight_decay: float = 1e-4,
    random_state: int | None = None,
    verbose: bool = False,
    auto_complexity: bool = False,
)
```

```python
DeepForestModel(
    time_col: str,
    target_col: str,
    lags: int = 1,
    quantile: float | None = 0.9,
    accelerator: str | None = None,
    n_trees: int = 32,
    tree_depth: int = 4,
    n_layers: int = 3,               # Number of cascade layers
    learning_rate: float = 0.08,
    n_epochs: int = 200,
    batch_size: int = 0,
    early_stop_patience: int = 12,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    random_state: int | None = None,
    verbose: bool = False,
    auto_complexity: bool = False,
)
```

**Auto-complexity property / 自适应复杂度属性:**

```python
model.model.complexity_info  # dict or None
# Returns: {'profile', 'tree_depth', 'n_trees', 'complexity_score', 'reasons', 'stats'}
```

### Other ML Models / 其他 ML 模型

| Class / 类 | Import / 导入 |
|---|---|
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

All located in `PipelineTS.preprocessing`.
全部位于 `PipelineTS.preprocessing`。

### `Scaler`

```python
Scaler(scaler_name: str)
# scaler_name: 'min_max' | 'standard' | 'quantile' | 'gauss_rank'

scaler.fit_transform(X)      # Fit and transform / 拟合并变换
scaler.transform(X)          # Transform / 变换
scaler.inverse_transform(X)  # Inverse transform / 反向变换
```

### `TimeSeriesMissingHandler`

```python
TimeSeriesMissingHandler(time_col: str)

handler.fit(data, value_cols=None)                      # Detect missing values / 检测缺失值
handler.transform(data, method='linear', fill_implicit_gaps=True)  # Fill missing values / 填充缺失值
handler.fit_transform(data, method='linear')            # Fit + transform in one call / 一步检测并填充
# method: 'linear' | 'ffill' | 'bfill' | 'spline' | 'zero'
```

### `TimeSeriesOutlierDetector`

```python
TimeSeriesOutlierDetector(time_col: str, method: str = 'iqr', threshold: float = 1.5, window: int = 20)
# method: 'iqr' | 'zscore' | 'rolling_zscore' | 'grubbs'

detector.fit(data, target_col)                          # Returns boolean mask / 返回布尔掩码
detector.transform(data, target_col, strategy='clip')   # Handle outliers / 处理异常值
detector.fit_transform(data, target_col, strategy='clip')  # Fit + transform / 一步检测并处理
# strategy: 'clip' | 'nan' | 'median' | 'linear'
```

### `TimeSeriesDataQualityReport`

```python
TimeSeriesDataQualityReport(time_col: str, target_col: str)

reporter.fit(data)            # Returns report dict / 返回报告字典
reporter.report(data)         # Print formatted report / 打印格式化报告
```

### `StationarityTest`

```python
StationarityTest(significance_level: float = 0.05)

tester.adf_test(series)                     # ADF test / ADF 检验
tester.kpss_test(series, regression='c')    # KPSS test / KPSS 检验
tester.fit(series)                           # Combined ADF + KPSS / 联合检验
tester.suggest_differencing(series, max_d=2) # Suggest differencing order / 建议差分阶数
```

### `FrequencyDetector`

```python
from PipelineTS.preprocessing.time_series_analysis import FrequencyDetector

FrequencyDetector(time_col: str)

detector.fit(data, target_col=None)      # Detect frequency + dominant periods / 检测频率 + 主要周期
```

### `TimeSeriesSplit`

```python
TimeSeriesSplit.split(data, time_col, test_size=0.2)                              # Simple split / 简单分割
TimeSeriesSplit.expanding_window(data, time_col, min_train_size, test_size, step)  # Expanding CV / 扩展窗口
TimeSeriesSplit.sliding_window(data, time_col, train_size, test_size, step)        # Sliding CV / 滑动窗口
```

### Sequence Splitting / 序列分割

```python
from PipelineTS.preprocessing import (
    split_series,                # Univariate split / 单变量分割
    split_series_multivariate,   # Multivariate split / 多变量分割
    train_test_split_ts,         # Time-series train/test split / 时序训练/测试分割
)
```

---

## Feature Engineering / 特征工程

All located in `PipelineTS.feature_engineering`.
全部位于 `PipelineTS.feature_engineering`。

### `TimeSeriesFeatureEngineer`

```python
TimeSeriesFeatureEngineer(
    time_col, target_col=None,
    use_calendar=True, use_fourier=False, fourier_periods=None, fourier_harmonics=1,
    use_holidays=False, holiday_country=None, custom_holidays=None,
    use_lags=False, lag_window='auto', lag_features='all', drop_time_col=False,
)

engineer.fit(data)             # Fit / 拟合
engineer.transform(data)       # Transform / 转换
engineer.fit_transform(data)   # Fit + transform / 拟合 + 转换
engineer.get_feature_names()   # List generated feature names / 列出生成的特征名
```

### `FourierFeatures`

```python
FourierFeatures(time_col, periods, n_harmonics=1, prefix='fourier_')

ff.transform(data)           # Add Fourier features / 添加傅里叶特征
ff.get_feature_names()       # Feature column names / 特征列名
```

### `HolidayFeatures`

```python
HolidayFeatures(time_col, country=None, custom_holidays=None, window=3, prefix='holiday_')

hf.transform(data)           # Add holiday features / 添加节假日特征
hf.get_feature_names()       # Feature column names / 特征列名
# For country='CN': uses chinese-calendar (pip install chinesecalendar) as authoritative source
# 当 country='CN' 时：使用 chinese-calendar 作为标准数据源
# Extra CN features: holiday_is_workday, holiday_is_in_lieu, holiday_holiday_name
# 中国额外特征：holiday_is_workday（工作日）, holiday_is_in_lieu（调休）, holiday_holiday_name（节日名）
```

### `LagFeatureExtractor`

```python
LagFeatureExtractor(time_col, target_col, window='auto', features='all', prefix='lag_')
# features: 'all' or subset of: mean, std, min, max, median, skew, kurtosis,
#           trend_slope, ema, autocorr, momentum, rms, cv, iqr, energy

lf.transform(data)           # Add lag features / 添加滞后特征
lf.get_feature_names()       # Feature column names / 特征列名
```

---

## Metrics / 指标

### Point Metrics / 点指标

```python
from PipelineTS.metrics import mae, mse, rmse, wmape, mape, smape, mase, r2_score, medae
```

| Function / 函数 | Description / 描述 |
|---|---|
| `mae(y_true, y_pred)` | Mean Absolute Error / 平均绝对误差 |
| `mse(y_true, y_pred)` | Mean Squared Error / 均方误差 |
| `rmse(y_true, y_pred)` | Root Mean Squared Error / 均方根误差 |
| `wmape(y_true, y_pred)` | Weighted MAPE / 加权 MAPE |
| `mape(y_true, y_pred)` | Mean Absolute Percentage Error / 平均绝对百分比误差 |
| `smape(y_true, y_pred)` | Symmetric MAPE / 对称 MAPE |
| `mase(y_true, y_pred, y_train, seasonality=1)` | Mean Absolute Scaled Error / 平均绝对缩放误差 |
| `r2_score(y_true, y_pred)` | Coefficient of Determination / 决定系数 |
| `medae(y_true, y_pred)` | Median Absolute Error / 中位绝对误差 |

### Interval Metrics / 区间指标

```python
from PipelineTS.metrics import quantile_acc, picp, pinaw, winkler_score
```

| Function / 函数 | Description / 描述 |
|---|---|
| `quantile_acc(y_true, lower, upper)` | Interval coverage rate / 区间覆盖率 |
| `picp(y_true, lower, upper)` | Prediction Interval Coverage Probability / 预测区间覆盖概率 |
| `pinaw(y_true, lower, upper)` | Normalized Average Width / 归一化平均宽度 |
| `winkler_score(y_true, lower, upper, alpha=0.1)` | Winkler interval score / Winkler 区间分数 |

---

## Evaluation / 评估

All located in `PipelineTS.evaluation`.
全部位于 `PipelineTS.evaluation`。

### `Backtester`

```python
Backtester(model, time_col, target_col, metric, metric_name='metric', metric_less_is_better=True)

bt.fit(data, n_splits=5, test_size=10, mode='expanding', train_size=None, verbose=True)
bt.summary()  # Returns dict: mean, std, min, max, median, n_folds, n_failed
```

### `ResidualAnalyzer`

```python
ResidualAnalyzer(y_true, y_pred)

analyzer.statistics()          # Basic stats / 基本统计量
analyzer.normality_test()      # Shapiro-Wilk + Jarque-Bera / 正态性检验
analyzer.autocorrelation()     # ACF + Ljung-Box / 自相关检验
analyzer.bias_analysis()       # Systematic bias / 系统性偏差
analyzer.report()              # Formatted output / 格式化输出
analyzer.plot()                # 4-panel diagnostic / 四面板诊断图
```

### `ModelComparison`

```python
ModelComparison(time_col, target_col)

comp.add_result(model_name, y_true, y_pred, lower=None, upper=None)
comp.fit(metrics=None, interval_metrics=None)         # Returns DataFrame / 返回 DataFrame
comp.rank(metric_name, ascending=True)                # Ranked table / 排名表
comp.plot_bar()                                        # Bar chart / 柱状图
comp.plot_radar()                                      # Radar chart / 雷达图
comp.plot_predictions(time_index=None)                 # Overlay plot / 叠加图
```

---

## Training / 训练

All located in `PipelineTS.training`.
全部位于 `PipelineTS.training`。

### `AutoTune`

```python
AutoTune(model_class, time_col, target_col, lags, metric,
         metric_less_is_better=True, n_trials=20, test_size=0.2,
         fixed_params=None, random_state=0)

best_model, best_params, history = tuner.fit(data, search_space, verbose=True)
# search_space format: {'param': ('int'|'float'|'categorical', ...)}
```

### `WeightedEnsemble`

```python
WeightedEnsemble(models, time_col, target_col, weights='auto', metric=None)

ens.fit(data, valid_data=None)
ens.predict(n)
ens.get_weights()  # Returns {name: weight} / 返回 {名称: 权重}
```

### `StackingEnsemble`

```python
StackingEnsemble(models, time_col, target_col, n_folds=3)

stack.fit(data)
stack.predict(n)
```

---

## Prediction / 预测

All located in `PipelineTS.prediction`.
全部位于 `PipelineTS.prediction`。

### `RollingPredictor`

```python
RollingPredictor(model, time_col, target_col, train_size, horizon, step=1, refit=True)

results = rp.predict(data, verbose=True)            # Returns DataFrame / 返回 DataFrame
eval_results = rp.score(results, metrics=None)      # Per-window + overall metrics / 每窗口 + 总体指标
```

### `ModelExplainer`

```python
ModelExplainer(model, time_col, target_col)

explainer.feature_importance()                  # Native importance / 原生重要性
explainer.plot_importance(importance_df=None, top_k=20)  # Bar chart / 柱状图
```

---

## I/O

```python
from PipelineTS.io import save_model, load_model

save_model(path: str, model, scaler=None)  # Save model / 保存模型
load_model(path: str)                       # Load model / 加载模型
```

---

## Plotting / 绘图

All located in `PipelineTS.plot`.
全部位于 `PipelineTS.plot`。

### Chinese Font Configuration / 中文字体配置

```python
from PipelineTS.plot import configure_chinese_font

configure_chinese_font(force: bool = False) -> str
# Auto-detect and set Chinese font. Returns font name.
# 自动检测并设置中文字体。返回字体名称。
```

### `TSPlotter`

```python
TSPlotter(time_col: str, target_col: str, lang: str = 'zh')

plotter.plot_series(data, id_col=None, **kwargs)
plotter.plot_forecast(train_data, forecast_data, **kwargs)
plotter.plot_leaderboard(leaderboard, **kwargs)
plotter.plot_leaderboard_detail(leaderboard, **kwargs)
plotter.plot_model_comparison(train_data, predictions, **kwargs)
plotter.plot_residuals(y_true, y_pred, **kwargs)
plotter.plot_acf_pacf(series, **kwargs)
plotter.plot_decomposition(data, **kwargs)
plotter.plot_train_test_split(train_data, test_data, **kwargs)
```

### Standalone Plot Functions / 独立绑图函数

```python
from PipelineTS.plot import (
    plot_series, plot_forecast, plot_leaderboard, plot_leaderboard_detail,
    plot_model_comparison, plot_residuals, plot_acf_pacf,
    plot_decomposition, plot_train_test_split,
)
```

| Function / 函数 | Signature / 签名 |
|---|---|
| `plot_series` | `(data, time_col, target_col, id_col=None, max_series=9, lang='zh', show=True)` |
| `plot_forecast` | `(train_data, forecast_data, time_col, target_col, history_tail=None, lang='zh', show=True)` |
| `plot_leaderboard` | `(leaderboard, metric_col='metric', model_col='model', lang='zh', show=True)` |
| `plot_leaderboard_detail` | `(leaderboard, lang='zh', show=True)` |
| `plot_model_comparison` | `(train_data, predictions: dict, time_col, target_col, lang='zh', show=True)` |
| `plot_residuals` | `(y_true, y_pred, time_index=None, lang='zh', show=True)` |
| `plot_acf_pacf` | `(series, max_lags=30, lang='zh', show=True)` |
| `plot_decomposition` | `(data, time_col, target_col, period=None, model='additive', lang='zh', show=True)` |
| `plot_train_test_split` | `(train_data, test_data, time_col, target_col, lang='zh', show=True)` |

All functions return `matplotlib.figure.Figure` and accept `title`, `figsize`, `show` parameters.
所有函数返回 `matplotlib.figure.Figure`，接受 `title`、`figsize`、`show` 参数。

### Color Constants / 颜色常量

```python
from PipelineTS.plot import COLORS, MODEL_COLORS

COLORS: dict   # Named colors: primary, forecast, actual, interval, etc.
MODEL_COLORS: list  # 15-color palette for model comparison
```

### Legacy Functions / 旧版函数

```python
from PipelineTS.plot import plot_data_period

plot_data_period(
    data1: pd.DataFrame,      # First dataset (e.g., train) / 第一个数据集
    data2: pd.DataFrame,      # Second dataset (e.g., prediction) / 第二个数据集
    time_col: str,
    target_col: str,
    labels: list = None,
    date_fmt: str = '%Y-%m-%d',
)
```
