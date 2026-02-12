# Visualization
# 可视化

PipelineTS provides a comprehensive visualization toolkit with **automatic Chinese font detection**, bilingual labels, and modern chart styling. All plot functions work seamlessly with matplotlib and support both Chinese (`lang='zh'`) and English (`lang='en'`) labels.

PipelineTS 提供全面的可视化工具包，支持**自动中文字体检测**、双语标签和现代图表样式。所有绑图函数与 matplotlib 无缝配合，支持中文（`lang='zh'`）和英文（`lang='en'`）标签。

---

## Chinese Font Support / 中文字体支持

matplotlib does not natively support Chinese characters. PipelineTS automatically detects and configures Chinese fonts across all major platforms.

matplotlib 默认不支持中文字符。PipelineTS 自动检测并配置所有主流平台上的中文字体。

### Auto-Detection / 自动检测

```python
from PipelineTS.plot import configure_chinese_font

# Auto-detect and configure (called automatically on first plot)
# 自动检测并配置（首次绑图时自动调用）
font_name = configure_chinese_font()
print(f"Using font: {font_name}")
```

### Supported Fonts by Platform / 各平台支持的字体

| Platform / 平台 | Fonts (ordered by priority) / 字体（按优先级排列） |
|---|---|
| **macOS** | PingFang SC, Heiti SC, STHeiti, Songti SC, Hiragino Sans GB |
| **Windows** | Microsoft YaHei, SimHei, SimSun, DengXian |
| **Linux** | WenQuanYi Micro Hei, Noto Sans CJK SC, Source Han Sans SC |

If no Chinese font is found, a warning is issued with installation instructions. The `axes.unicode_minus` is always set to `False` to prevent minus-sign rendering issues.

如果未找到中文字体，会发出警告并附带安装说明。`axes.unicode_minus` 始终设为 `False` 以防止负号渲染问题。

### Manual Font Configuration / 手动字体配置

```python
import matplotlib as mpl

# Override with a specific font / 手动指定字体
mpl.rcParams['font.sans-serif'] = ['Your Font Name'] + mpl.rcParams['font.sans-serif']
mpl.rcParams['axes.unicode_minus'] = False
```

---

## Plot Functions Overview / 绑图函数概览

| Function / 函数 | Description / 描述 |
|---|---|
| `plot_series()` | Single or multi-series (panel) visualization / 单序列或多序列（面板）可视化 |
| `plot_forecast()` | Actual vs forecast with prediction intervals / 实际值 vs 预测值 + 预测区间 |
| `plot_leaderboard()` | Model ranking horizontal bar chart / 模型排名水平柱状图 |
| `plot_leaderboard_detail()` | Leaderboard + training/eval cost side by side / 排行榜 + 训练/评估耗时 |
| `plot_model_comparison()` | Multi-model forecast overlay / 多模型预测叠加对比 |
| `plot_residuals()` | 4-panel residual diagnostics / 四面板残差诊断 |
| `plot_acf_pacf()` | ACF + PACF side by side / ACF + PACF 并排图 |
| `plot_decomposition()` | Trend / seasonal / residual decomposition / 趋势/季节性/残差分解 |
| `plot_train_test_split()` | Visualize train/test partition / 训练集/测试集分割可视化 |

All functions share these common parameters:
所有函数共享以下通用参数：

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 描述 |
|---|---|---|---|
| `lang` | str | `'zh'` | Label language: `'zh'` or `'en'` / 标签语言 |
| `figsize` | tuple | varies | Figure size `(width, height)` / 图表尺寸 |
| `show` | bool | `True` | Whether to call `plt.show()` / 是否调用 `plt.show()` |
| `title` | str or None | None | Custom title (auto-generated if None) / 自定义标题 |

---

## plot_series — Time Series Visualization / 时间序列可视化

Visualize one or more time series. When `id_col` is provided, creates a panel of subplots.

可视化单条或多条时间序列。提供 `id_col` 时，创建子图面板。

### Single Series / 单序列

```python
from PipelineTS.plot import plot_series

plot_series(
    data,
    time_col='date',
    target_col='value',
    title='销量趋势',
    lang='zh',
)
```

### Multi-Series Panel / 多序列面板

```python
plot_series(
    panel_data,
    time_col='date',
    target_col='value',
    id_col='store_id',       # Triggers panel mode / 触发面板模式
    max_series=9,             # Max subplots / 最大子图数
    title='各门店销量',
)
```

**Parameters / 参数:**

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `data` | pd.DataFrame | Input data / 输入数据 |
| `time_col` | str | Datetime column / 日期时间列 |
| `target_col` | str | Target value column / 目标值列 |
| `id_col` | str or None | Series identifier for panel mode / 面板模式的序列标识符 |
| `max_series` | int | Max series to show in panel (default: 9) / 面板中最大序列数 |
| `ax` | Axes or None | Existing axes for single-series mode / 单序列模式的已有坐标轴 |

---

## plot_forecast — Forecast Visualization / 预测可视化

Plot historical data and forecast with optional prediction intervals and multi-quantile bands.

绘制历史数据和预测值，可选预测区间和多分位数带。

```python
from PipelineTS.plot import plot_forecast

plot_forecast(
    train_data,           # Historical data / 历史数据
    forecast_data,        # Predictions (may include _lower, _upper columns)
                          # 预测值（可能包含 _lower, _upper 列）
    time_col='date',
    target_col='value',
    history_tail=60,      # Show only last 60 history points / 仅显示最后 60 个历史点
    title='预测结果',
    lang='zh',
)
```

The function automatically detects:
函数自动检测：

- **Standard intervals**: `{target_col}_lower` and `{target_col}_upper` columns → shaded region
- **标准区间**：`{target_col}_lower` 和 `{target_col}_upper` 列 → 阴影区域

- **Multi-quantile intervals**: `{target_col}_q{level}_lower/upper` columns → nested shaded bands
- **多分位数区间**：`{target_col}_q{level}_lower/upper` 列 → 嵌套阴影带

**Parameters / 参数:**

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `train_data` | pd.DataFrame | Historical/training data / 历史/训练数据 |
| `forecast_data` | pd.DataFrame | Forecast output / 预测输出 |
| `history_tail` | int or None | Show only last N history points / 仅显示最后 N 个历史点 |
| `ax` | Axes or None | Existing axes / 已有坐标轴 |

---

## plot_leaderboard — Model Ranking Chart / 模型排名图

Plot model leaderboard as a horizontal bar chart. The best model is highlighted in green.

以水平柱状图展示模型排行榜。最佳模型以绿色高亮。

```python
from PipelineTS.plot import plot_leaderboard

leaderboard = pipeline.leader_board_

plot_leaderboard(
    leaderboard,
    metric_col='metric',
    model_col='model',
    title='模型排行榜',
    lang='zh',
)
```

---

## plot_leaderboard_detail — Detailed Leaderboard / 详细排行榜

Two-panel chart: metric bars + training/eval cost breakdown.

双面板图表：指标柱状图 + 训练/评估耗时分解。

```python
from PipelineTS.plot import plot_leaderboard_detail

plot_leaderboard_detail(
    leaderboard,   # Must have: model, metric, train_cost(s), eval_cost(s)
                   # 必须包含列: model, metric, train_cost(s), eval_cost(s)
    title='模型详细排行',
    lang='zh',
)
```

---

## plot_model_comparison — Multi-Model Overlay / 多模型叠加对比

Overlay predictions from multiple models on the same chart.

在同一图表上叠加显示多个模型的预测结果。

```python
from PipelineTS.plot import plot_model_comparison

predictions = {
    'LightGBM': pred_lgbm,
    'Prophet':  pred_prophet,
    'TFT':      pred_tft,
}

plot_model_comparison(
    train_data,
    predictions,          # {model_name: forecast_df}
    time_col='date',
    target_col='value',
    history_tail=30,
    title='模型预测对比',
)
```

---

## plot_residuals — Residual Diagnostics / 残差诊断

Four-panel diagnostic plot: residual time series, histogram with normal overlay, Q-Q plot, and ACF plot.

四面板诊断图：残差时序图、带正态拟合的直方图、Q-Q 图、ACF 图。

```python
from PipelineTS.plot import plot_residuals

plot_residuals(
    y_true,
    y_pred,
    time_index=dates,     # Optional time axis / 可选时间轴
    title='残差分析',
    lang='zh',
)
```

---

## plot_acf_pacf — Autocorrelation Analysis / 自相关分析

ACF and PACF side by side with significance bands (95% confidence).

ACF 和 PACF 并排显示，带 95% 置信区间显著性带。

```python
from PipelineTS.plot import plot_acf_pacf

plot_acf_pacf(
    data['value'].values,
    max_lags=30,
    title='自相关分析',
    lang='zh',
)
```

**Note**: Requires `statsmodels`. Falls back gracefully if not installed.
**注意**：需要 `statsmodels`。未安装时优雅回退。

---

## plot_decomposition — Time Series Decomposition / 时间序列分解

Decompose time series into trend, seasonal, and residual components. Uses `statsmodels.seasonal_decompose` if available, with a moving-average fallback.

将时间序列分解为趋势、季节性和残差分量。优先使用 `statsmodels.seasonal_decompose`，不可用时使用移动平均回退。

```python
from PipelineTS.plot import plot_decomposition

plot_decomposition(
    data,
    time_col='date',
    target_col='value',
    period=None,          # Auto-detect if None / 为 None 时自动检测
    model='additive',     # 'additive' or 'multiplicative'
    title='时间序列分解',
    lang='zh',
)
```

---

## plot_train_test_split — Data Split Visualization / 数据分割可视化

Visualize the train/test partition with a vertical separator line.

可视化训练集/测试集分割，带有垂直分隔线。

```python
from PipelineTS.plot import plot_train_test_split

plot_train_test_split(
    train_data,
    test_data,
    time_col='date',
    target_col='value',
    title='训练集/测试集',
    lang='zh',
)
```

---

## TSPlotter Class / TSPlotter 类

A high-level reusable plotting interface that binds `time_col`, `target_col`, and `lang` once for repeated use.

高层可复用绑图接口，一次绑定 `time_col`、`target_col` 和 `lang`，方便反复使用。

```python
from PipelineTS.plot import TSPlotter

plotter = TSPlotter(time_col='date', target_col='value', lang='zh')

# All methods mirror the standalone functions / 所有方法与独立函数对应
plotter.plot_series(data)
plotter.plot_series(panel_data, id_col='store_id')
plotter.plot_forecast(train, pred)
plotter.plot_leaderboard(leaderboard)
plotter.plot_leaderboard_detail(leaderboard)
plotter.plot_model_comparison(train, predictions_dict)
plotter.plot_residuals(y_true, y_pred)
plotter.plot_acf_pacf(series)
plotter.plot_decomposition(data)
plotter.plot_train_test_split(train, test)
```

---

## Pipeline / SmartRouter Integration / 管道/智能路由器集成

`ModelPipeline` and `SmartRouter` provide built-in `plot()` and `plot_leaderboard()` methods for one-line visualization.

`ModelPipeline` 和 `SmartRouter` 提供内置 `plot()` 和 `plot_leaderboard()` 方法，一行代码即可可视化。

### ModelPipeline

```python
from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(time_col='date', target_col='value', lags=12)
pipeline.fit(data)

# Forecast plot (best model) / 预测图（最佳模型）
pipeline.plot(n=12, lang='zh')

# With options / 带选项
pipeline.plot(
    n=12,
    model_name='lightgbm',   # Use specific model / 使用指定模型
    history_tail=60,          # Show last 60 points / 显示最后 60 个点
    lang='en',                # English labels / 英文标签
)

# Leaderboard chart / 排行榜图
pipeline.plot_leaderboard(lang='zh')
```

### SmartRouter

```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(time_col='date', target_col='value')
router.fit(data)

router.plot(n=12, lang='zh')
router.plot_leaderboard(lang='zh')
```

---

## Color Palette / 颜色调色板

The visualization module uses a modern, accessible color palette. You can customize it by importing the color constants.

可视化模块使用现代、易读的颜色调色板。可以通过导入颜色常量进行自定义。

```python
from PipelineTS.plot import COLORS, MODEL_COLORS

print(COLORS)
# {'primary': '#2563EB', 'secondary': '#7C3AED', 'success': '#059669',
#  'actual': '#1F2937', 'forecast': '#2563EB', 'interval': '#93C5FD', ...}

print(MODEL_COLORS)
# ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626', ...]
```

---

## Return Values / 返回值

All plot functions return a `matplotlib.figure.Figure` object, allowing further customization:

所有绑图函数返回 `matplotlib.figure.Figure` 对象，允许进一步自定义：

```python
fig = plot_forecast(train, pred, 'date', 'value', show=False)

# Customize / 自定义
ax = fig.axes[0]
ax.set_ylim(0, 200)
ax.set_title('My Custom Title / 自定义标题')
fig.savefig('forecast.png', dpi=150, bbox_inches='tight')
```

Set `show=False` to prevent automatic display (useful for saving or embedding in reports).

设置 `show=False` 可防止自动显示（适用于保存或嵌入报告中）。
