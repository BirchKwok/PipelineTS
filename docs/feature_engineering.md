# Feature Engineering
# 特征工程

PipelineTS provides a unified feature engineering pipeline with multiple composable feature extractors.
PipelineTS 提供统一的特征工程管道，包含多个可组合的特征提取器。

All feature extractors are vectorized with numpy/pandas for performance.
所有特征提取器均使用 numpy/pandas 向量化实现，以确保性能。

---

## Unified Feature Pipeline / 统一特征管道

`TimeSeriesFeatureEngineer` composes multiple feature extractors into a single `transform()` call.
`TimeSeriesFeatureEngineer` 将多个特征提取器组合为一个 `transform()` 调用。

```python
from PipelineTS.feature_engineering import TimeSeriesFeatureEngineer

engineer = TimeSeriesFeatureEngineer(
    time_col='date',
    target_col='value',
    use_calendar=True,              # Calendar features (weekday, month, etc.) / 日历特征
    use_fourier=True,               # Fourier periodic features / 傅里叶周期特征
    fourier_periods=[7, 365],       # Weekly + yearly cycles / 周 + 年周期
    fourier_harmonics=2,            # Harmonics per period / 每个周期的谐波数
    use_holidays=True,              # Holiday indicators / 节假日指示符
    holiday_country='US',           # Country-specific holidays / 国家特定节假日
    custom_holidays=['2024-07-04'], # Custom holiday dates / 自定义节假日
    use_lags=True,                  # Rolling lag features / 滚动滞后特征
    lag_window=12,                  # Window size / 窗口大小
    lag_features=['mean', 'std', 'trend_slope', 'ema'],
    drop_time_col=False,            # Keep time column / 保留时间列
)

# Fit and transform / 拟合并转换
df_enriched = engineer.fit_transform(data)

# Get generated feature names / 获取生成的特征名
print(engineer.get_feature_names())
```

Each component can be enabled/disabled independently via boolean flags.
每个组件都可以通过布尔标志独立启用/禁用。

---

## Fourier Features / 傅里叶特征

`FourierFeatures` generates deterministic sin/cos periodic basis functions from a datetime column.
`FourierFeatures` 从日期时间列生成确定性的 sin/cos 周期基函数。

Fourier features capture cyclical patterns (e.g., day-of-week, month-of-year) without one-hot encoding overhead.
傅里叶特征捕捉周期性模式（如周几、月份），避免了独热编码的开销。

```python
from PipelineTS.feature_engineering import FourierFeatures

# Using period lengths (in number of time steps)
# 使用周期长度（以时间步数为单位）
ff = FourierFeatures(
    time_col='date',
    periods=[7, 30, 365],    # Weekly, monthly, yearly / 周、月、年
    n_harmonics=2,            # 2 sin/cos pairs per period / 每个周期 2 对 sin/cos
    prefix='fourier_',
)
df = ff.transform(data)

# Using named periods for readable column names
# 使用命名周期以获得可读的列名
ff = FourierFeatures(
    time_col='date',
    periods={'weekly': 7, 'monthly': 30, 'yearly': 365},
    n_harmonics=3,
)
df = ff.transform(data)
# Columns: fourier_weekly_sin_1, fourier_weekly_cos_1, ..., fourier_yearly_cos_3

# Get feature names / 获取特征名
print(ff.get_feature_names())
```

**Parameters / 参数:**

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `time_col` | str | Datetime column name / 日期时间列名 |
| `periods` | list or dict | Period lengths or {name: length} / 周期长度或 {名称: 长度} |
| `n_harmonics` | int | Number of sin/cos pairs per period (default: 1) / 每个周期的谐波数 |
| `prefix` | str | Column name prefix (default: `'fourier_'`) / 列名前缀 |

---

## Holiday Features / 节假日特征

`HolidayFeatures` generates holiday-related binary indicators and distance features.
`HolidayFeatures` 生成节假日相关的二值指示符和距离特征。

```python
from PipelineTS.feature_engineering import HolidayFeatures

# China holidays (recommended: uses chinese-calendar for official data)
# 中国节假日（推荐：使用 chinese-calendar 获取官方数据）
hf = HolidayFeatures(
    time_col='date',
    country='CN',                                  # ISO country code / ISO 国家代码
    custom_holidays=['2024-10-01', '2024-02-10'],  # Custom dates / 自定义日期
    window=3,                                       # ±3 days = "near holiday" / ±3天 = "临近节假日"
)
df = hf.transform(data)
```

**Generated features (all countries) / 生成的特征（所有国家）：**

| Feature / 特征 | Description / 描述 |
|---|---|
| `holiday_is_holiday` | Binary: 1 if the date is a holiday / 二值：1 表示该日期为节假日 |
| `holiday_days_to_nearest` | Signed distance (days) to nearest holiday / 到最近节假日的有符号距离（天数） |
| `holiday_near_holiday` | Binary: 1 if within ±window days of a holiday / 二值：1 表示在节假日 ±window 天内 |

### China-specific features / 中国特有特征

When `country='CN'` and the [`chinese-calendar`](https://github.com/LKI/chinese-calendar) package is installed,
PipelineTS uses it as the **authoritative source** for Chinese holidays.
This provides official government holiday schedules (2004–2026), including make-up workdays (调休/补班) and holiday names.

当 `country='CN'` 且安装了 [`chinese-calendar`](https://github.com/LKI/chinese-calendar) 包时，
PipelineTS 将其作为中国节假日的**标准数据源**。
提供国务院官方节假日安排（2004–2026），包括调休/补班日和节假日名称。

```bash
pip install chinesecalendar  # Recommended for CN / 中国节假日推荐安装
```

**Extra features for China / 中国额外特征：**

| Feature / 特征 | Description / 描述 |
|---|---|
| `holiday_is_workday` | Binary: 1 if official workday (includes make-up days) / 二值：1 表示官方工作日（含调休补班） |
| `holiday_is_in_lieu` | Binary: 1 if make-up workday (调休) / 二值：1 表示调休日 |
| `holiday_holiday_name` | Holiday name string (e.g. 'National Day') or '' / 节假日名称字符串（如 'National Day'）或空 |

```python
# Example: 2024 National Day / 示例：2024 年国庆节
import pandas as pd
from PipelineTS.feature_engineering import HolidayFeatures

data = pd.DataFrame({
    'date': pd.date_range('2024-09-28', '2024-10-08', freq='D'),
    'value': range(11),
})
hf = HolidayFeatures(time_col='date', country='CN')
result = hf.transform(data)

# Oct 1-7: is_holiday=1, is_workday=0
# Oct 4, 7: is_in_lieu=1 (调休日)
# Sep 29: is_workday=1 (周日补班)
# Oct 1: holiday_name='National Day'
```

### Other countries / 其他国家

For non-CN countries, the `holidays` package is used when installed, supporting 100+ countries (US, DE, JP, KR, etc.).
对于非中国国家，安装 `holidays` 包后可支持 100+ 个国家（US、DE、JP、KR 等）。

Without any holiday package, built-in generic holidays (New Year, Christmas, etc.) are used as fallback.
未安装任何节假日包时，使用内置通用节假日（元旦、圣诞节等）作为回退。

```bash
pip install holidays  # Optional for non-CN countries / 非中国国家可选
```

### Priority order for CN / 中国节假日优先级

1. `chinese-calendar` (推荐，官方数据) → 2. `holidays` 库 → 3. 内置通用节假日

---

## Lag Features / 滞后特征

`LagFeatureExtractor` extracts rolling-window statistical features from the target column.
`LagFeatureExtractor` 从目标列提取滚动窗口统计特征。

All features are strictly causal — each row's features are computed from past values only (no data leakage).
所有特征严格因果——每行的特征仅从过去的值计算（无数据泄漏）。

```python
from PipelineTS.feature_engineering import LagFeatureExtractor

extractor = LagFeatureExtractor(
    time_col='date',
    target_col='value',
    window=12,             # Rolling window size / 滚动窗口大小
    features='all',        # All 15 features / 全部 15 个特征
    prefix='lag_',
)
df = extractor.transform(data)
print(extractor.get_feature_names())
```

**Available features (15 total) / 可用特征（共 15 个）：**

| Feature / 特征 | Description / 描述 |
|---|---|
| `mean` | Rolling mean / 滚动均值 |
| `std` | Rolling standard deviation / 滚动标准差 |
| `min` | Rolling minimum / 滚动最小值 |
| `max` | Rolling maximum / 滚动最大值 |
| `median` | Rolling median / 滚动中位数 |
| `skew` | Rolling skewness / 滚动偏度 |
| `kurtosis` | Rolling kurtosis / 滚动峰度 |
| `trend_slope` | Linear regression slope / 线性回归斜率 |
| `ema` | Exponential moving average / 指数移动平均 |
| `autocorr` | Lag-1 autocorrelation / 滞后 1 阶自相关 |
| `momentum` | Value change over window / 窗口内的值变化量 |
| `rms` | Root mean square / 均方根 |
| `cv` | Coefficient of variation / 变异系数 |
| `iqr` | Interquartile range / 四分位距 |
| `energy` | Sum of squared values / 平方和 |

You can select a subset of features:
可以选择特征子集：

```python
extractor = LagFeatureExtractor(
    time_col='date', target_col='value', window=12,
    features=['mean', 'std', 'trend_slope', 'ema', 'autocorr'],
)
```

---

## Calendar Features / 日历特征

Calendar features are generated by the internal `DateExtendFeatures` class, accessible via `TimeSeriesFeatureEngineer(use_calendar=True)`.
日历特征由内部的 `DateExtendFeatures` 类生成，可通过 `TimeSeriesFeatureEngineer(use_calendar=True)` 访问。

Generated features include: hour, day of week, day of month, day of year, week of year, month, quarter, year, is_weekend, etc.
生成的特征包括：小时、星期几、月中日、年中日、年中周、月份、季度、年份、是否周末等。

```python
from PipelineTS.feature_engineering import TimeSeriesFeatureEngineer

engineer = TimeSeriesFeatureEngineer(
    time_col='date',
    use_calendar=True,  # Only calendar features / 仅日历特征
)
df = engineer.fit_transform(data)
```
