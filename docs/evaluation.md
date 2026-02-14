# Evaluation & Metrics
# 评估与指标

PipelineTS provides a comprehensive model evaluation framework including backtesting, residual analysis, and multi-model comparison visualization.
PipelineTS 提供全面的模型评估框架，包括回测、残差分析和多模型对比可视化。

---

## Backtesting / 回测

`Backtester` performs walk-forward backtesting by simulating sequential real-world forecasts: train on past data, predict forward, slide the window, repeat.
`Backtester` 通过模拟顺序的真实预测进行前向回测：在历史数据上训练，向前预测，滑动窗口，重复。

The model is deep-copied per fold to avoid state leakage between folds.
模型在每个折叠中深拷贝，以避免折叠之间的状态泄漏。

```python
from PipelineTS.evaluation import Backtester
from PipelineTS.ml_model import TorchBoostingForestModel
from PipelineTS.spinesTS.metrics import mae

model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)

bt = Backtester(
    model,
    time_col='date',
    target_col='value',
    metric=mae,
    metric_name='MAE',
    metric_less_is_better=True,
)

# Expanding window: training set grows from the beginning
# 扩展窗口：训练集从头开始持续增长
results = bt.fit(data, n_splits=5, test_size=12, mode='expanding', verbose=True)

# Sliding window: fixed-size training window
# 滑动窗口：固定大小的训练窗口
results = bt.fit(data, n_splits=5, test_size=12, mode='sliding', train_size=100)

# Summary statistics / 汇总统计
summary = bt.summary()
print(f"Mean MAE:   {summary['mean']:.4f}")
print(f"Std MAE:    {summary['std']:.4f}")
print(f"Min MAE:    {summary['min']:.4f}")
print(f"Max MAE:    {summary['max']:.4f}")
print(f"Median MAE: {summary['median']:.4f}")
print(f"Failed:     {summary['n_failed']}/{summary['n_folds']}")
```

**Parameters / 参数：**

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `model` | PipelineTS model | Any model with `fit()` and `predict()` / 任意有 `fit()` 和 `predict()` 的模型 |
| `time_col` | str | Datetime column name / 日期时间列名 |
| `target_col` | str | Target column name / 目标列名 |
| `metric` | callable | `metric(y_true, y_pred) -> float` |
| `metric_name` | str | Display name for the metric / 指标显示名 |
| `metric_less_is_better` | bool | Whether lower metric is better / 指标是否越低越好 |

**`run()` parameters / `run()` 参数：**

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `n_splits` | int | Number of folds (default: 5) / 折叠数 |
| `test_size` | int | Forecast horizon per fold / 每折预测步数 |
| `mode` | str | `'expanding'` or `'sliding'` / 扩展窗口或滑动窗口 |
| `train_size` | int or None | Fixed training size for sliding mode / 滑动模式的固定训练大小 |

---

## Residual Analysis / 残差分析

`ResidualAnalyzer` provides comprehensive residual diagnostics to evaluate model quality.
`ResidualAnalyzer` 提供全面的残差诊断，用于评估模型质量。

```python
from PipelineTS.evaluation import ResidualAnalyzer

analyzer = ResidualAnalyzer(y_true, y_pred)
```

### Statistics / 统计量

```python
stats = analyzer.statistics()
# Keys: mean, std, min, max, median, skewness, kurtosis, mean_abs (MAE), rmse
```

### Normality Tests / 正态性检验

```python
norm = analyzer.normality_test()
# norm['shapiro']:     Shapiro-Wilk test (statistic, p_value, is_normal)
# norm['jarque_bera']: Jarque-Bera test (statistic, p_value, is_normal)
```

Well-behaved residuals should be approximately normally distributed.
良好的残差应近似正态分布。

### Autocorrelation Analysis / 自相关分析

```python
acorr = analyzer.autocorrelation(max_lags=20)
# acorr['acf_values']:       ACF values at each lag / 每个滞后的 ACF 值
# acorr['significant_lags']: Lags with significant autocorrelation / 显著自相关的滞后
# acorr['ljung_box']:        Ljung-Box test (statistic, p_value, has_autocorrelation)
```

Significant residual autocorrelation suggests the model is not fully capturing temporal patterns.
残差的显著自相关表明模型没有完全捕捉时间模式。

### Bias Analysis / 偏差分析

```python
bias = analyzer.bias_analysis()
# bias['mean_bias']:       Mean residual value / 残差均值
# bias['bias_direction']:  'under-predicting', 'over-predicting', or 'unbiased'
# bias['bias_significant']: True if t-test rejects H0: mean=0 at 5% / t 检验是否拒绝均值=0
# bias['pct_positive']:    Fraction of positive residuals / 正残差比例
# bias['pct_negative']:    Fraction of negative residuals / 负残差比例
```

### Report and Visualization / 报告与可视化

```python
# Print formatted report / 打印格式化报告
analyzer.report()

# 4-panel diagnostic plot / 四面板诊断图
# Panels: residuals over time, histogram, Q-Q plot, ACF
# 面板：残差时间图、直方图、Q-Q 图、ACF 图
analyzer.plot(figsize=(14, 10))
```

---

## Model Comparison / 模型对比

`ModelComparison` evaluates and visualizes multiple models side-by-side on multiple metrics.
`ModelComparison` 在多个指标上并排评估和可视化多个模型。

```python
from PipelineTS.evaluation import ModelComparison
from PipelineTS.metrics import mape, r2_score, picp, pinaw

comp = ModelComparison(time_col='date', target_col='value')

# Register model predictions / 注册模型预测
comp.add_result('TorchBoostingForest', y_true, y_pred_boost, lower=lower_boost, upper=upper_boost)
comp.add_result('TorchBaggingForest',  y_true, y_pred_bag,   lower=lower_bag,   upper=upper_bag)
comp.add_result('Prophet',  y_true, y_pred_prophet)

# Evaluate on metrics / 按指标评估
table = comp.fit(
    metrics={'MAPE': mape, 'R²': r2_score},
    interval_metrics={'PICP': picp, 'PINAW': pinaw},  # Only applied to models with intervals
                                                        # 仅应用于有区间的模型
)
print(table)
```

### Ranking / 排名

```python
# Rank models by a metric / 按指标排名
ranked = comp.rank('MAPE', ascending=True)
print(ranked)
# Output: rank, model, MAPE, R², PICP, PINAW
```

### Visualization / 可视化

```python
# Grouped bar chart comparing metrics / 分组柱状图对比指标
comp.plot_bar(figsize=(12, 5))

# Radar chart (metrics normalized to [0,1]) / 雷达图（指标归一化到 [0,1]）
comp.plot_radar(figsize=(8, 8))

# Prediction overlay plot / 预测叠加图
comp.plot_predictions(time_index=test_dates, figsize=(14, 5))
```

**`add_result()` parameters / `add_result()` 参数：**

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `model_name` | str | Display name / 显示名称 |
| `y_true` | array | True values / 真实值 |
| `y_pred` | array | Predicted values / 预测值 |
| `lower` | array or None | Lower interval bound / 区间下界 |
| `upper` | array or None | Upper interval bound / 区间上界 |
