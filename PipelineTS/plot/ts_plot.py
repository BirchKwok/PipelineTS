"""Comprehensive time-series visualization toolkit with Chinese font support.

Provides:
- Automatic Chinese font detection and configuration
- Single-series and multi-series (panel) plotting
- Forecast visualization with prediction intervals
- Leaderboard / model comparison charts
- Residual diagnostics
- Trend-seasonality decomposition
- ACF / PACF plots

All functions accept ``lang='zh'`` or ``lang='en'`` to switch labels.
"""

import warnings
import platform
import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Tuple

# ---------------------------------------------------------------------------
#  Chinese Font Auto-Detection
# ---------------------------------------------------------------------------

_FONT_CONFIGURED = False
_CHINESE_FONT_NAME = None


def _find_chinese_font() -> Optional[str]:
    """Search for a usable Chinese font on the current system."""
    try:
        from matplotlib import font_manager as fm
    except ImportError:
        return None

    system = platform.system()

    # Candidate font names ordered by preference
    if system == 'Darwin':  # macOS
        candidates = [
            'PingFang SC', 'Heiti SC', 'Heiti TC',
            'STHeiti', 'STSong', 'STFangsong',
            'Songti SC', 'Hiragino Sans GB',
            'Apple LiGothic', 'Apple LiSung',
            'Arial Unicode MS',
        ]
    elif system == 'Windows':
        candidates = [
            'Microsoft YaHei', 'SimHei', 'SimSun',
            'NSimSun', 'FangSong', 'KaiTi',
            'Microsoft JhengHei', 'DengXian',
            'Arial Unicode MS',
        ]
    else:  # Linux
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK TC',
            'Noto Sans CJK JP', 'Noto Sans Mono CJK SC',
            'Source Han Sans SC', 'Source Han Sans CN',
            'AR PL UMing CN', 'AR PL UKai CN',
            'Droid Sans Fallback',
            'Arial Unicode MS',
        ]

    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name

    # Fallback: scan for any font whose name contains CJK keywords
    cjk_keywords = ['cjk', 'chinese', 'hei', 'song', 'fang', 'kai',
                     'ming', 'gothic', 'pingfang', 'yahei', 'wenquan']
    for f in fm.fontManager.ttflist:
        low = f.name.lower()
        if any(kw in low for kw in cjk_keywords):
            return f.name

    return None


def configure_chinese_font(force: bool = False) -> str:
    """Configure matplotlib to use a Chinese-capable font.

    Parameters
    ----------
    force : bool
        Re-detect even if already configured.

    Returns
    -------
    str
        The font family name that was set, or '' if none found.
    """
    global _FONT_CONFIGURED, _CHINESE_FONT_NAME

    if _FONT_CONFIGURED and not force:
        return _CHINESE_FONT_NAME or ''

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    font_name = _find_chinese_font()
    if font_name:
        mpl.rcParams['font.sans-serif'] = [font_name] + mpl.rcParams.get(
            'font.sans-serif', ['DejaVu Sans'])
        mpl.rcParams['axes.unicode_minus'] = False
        _CHINESE_FONT_NAME = font_name
    else:
        warnings.warn(
            "未找到中文字体。中文标签可能无法正常显示。"
            "请安装中文字体（如 Noto Sans CJK SC）或手动设置 "
            "matplotlib rcParams['font.sans-serif']。",
            UserWarning,
        )
        mpl.rcParams['axes.unicode_minus'] = False
        _CHINESE_FONT_NAME = ''

    _FONT_CONFIGURED = True
    return _CHINESE_FONT_NAME or ''


def _ensure_font():
    """Call once before any plotting to ensure Chinese font is ready."""
    if not _FONT_CONFIGURED:
        configure_chinese_font()


# ---------------------------------------------------------------------------
#  Color Palette
# ---------------------------------------------------------------------------

# Modern, accessible color palette
COLORS = {
    'primary': '#2563EB',      # blue
    'secondary': '#7C3AED',    # violet
    'success': '#059669',      # emerald
    'warning': '#D97706',      # amber
    'danger': '#DC2626',       # red
    'info': '#0891B2',         # cyan
    'dark': '#1F2937',         # gray-800
    'light': '#F3F4F6',        # gray-100
    'actual': '#1F2937',       # dark gray for actual data
    'forecast': '#2563EB',     # blue for forecasts
    'interval': '#93C5FD',     # light blue for intervals
    'grid': '#E5E7EB',        # gray-200
}

MODEL_COLORS = [
    '#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626',
    '#0891B2', '#DB2777', '#4F46E5', '#65A30D', '#EA580C',
    '#6366F1', '#14B8A6', '#F59E0B', '#EF4444', '#8B5CF6',
]

# ---------------------------------------------------------------------------
#  Bilingual Labels
# ---------------------------------------------------------------------------

_LABELS = {
    'zh': {
        'actual': '实际值',
        'forecast': '预测值',
        'upper': '上界',
        'lower': '下界',
        'interval': '预测区间',
        'time': '时间',
        'value': '值',
        'model': '模型',
        'metric': '指标',
        'rank': '排名',
        'leaderboard': '模型排行榜',
        'comparison': '模型预测对比',
        'residual': '残差',
        'residuals': '残差分析',
        'histogram': '残差分布',
        'qq': 'Q-Q 图',
        'acf': '自相关函数 (ACF)',
        'pacf': '偏自相关函数 (PACF)',
        'trend': '趋势',
        'seasonal': '季节性',
        'remainder': '残差项',
        'decomposition': '时间序列分解',
        'series': '序列',
        'train': '训练集',
        'test': '测试集',
        'forecast_plot': '预测结果',
        'train_cost': '训练耗时(秒)',
        'eval_cost': '评估耗时(秒)',
        'best': '(最优)',
    },
    'en': {
        'actual': 'Actual',
        'forecast': 'Forecast',
        'upper': 'Upper Bound',
        'lower': 'Lower Bound',
        'interval': 'Prediction Interval',
        'time': 'Time',
        'value': 'Value',
        'model': 'Model',
        'metric': 'Metric',
        'rank': 'Rank',
        'leaderboard': 'Model Leaderboard',
        'comparison': 'Model Prediction Comparison',
        'residual': 'Residual',
        'residuals': 'Residual Analysis',
        'histogram': 'Residual Distribution',
        'qq': 'Q-Q Plot',
        'acf': 'Autocorrelation (ACF)',
        'pacf': 'Partial Autocorrelation (PACF)',
        'trend': 'Trend',
        'seasonal': 'Seasonal',
        'remainder': 'Remainder',
        'decomposition': 'Time Series Decomposition',
        'series': 'Series',
        'train': 'Train',
        'test': 'Test',
        'forecast_plot': 'Forecast',
        'train_cost': 'Train Cost (s)',
        'eval_cost': 'Eval Cost (s)',
        'best': '(best)',
    },
}


def _L(key: str, lang: str = 'zh') -> str:
    """Get a label string in the specified language."""
    return _LABELS.get(lang, _LABELS['zh']).get(key, key)


# ---------------------------------------------------------------------------
#  Style Helpers
# ---------------------------------------------------------------------------

def _apply_style(ax, title: str = '', xlabel: str = '', ylabel: str = '',
                 grid: bool = True, legend: bool = True,
                 legend_loc: str = 'best'):
    """Apply consistent styling to an axes object."""
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if grid:
        ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=9, framealpha=0.8)


# ===========================================================================
#  1. plot_series — single or multi-series raw data visualization
# ===========================================================================

def plot_series(
    data: pd.DataFrame,
    time_col: str,
    target_col: str,
    id_col: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    max_series: int = 9,
    lang: str = 'zh',
    show: bool = True,
    ax=None,
):
    """Plot one or more time series.

    Parameters
    ----------
    data : pd.DataFrame
    time_col, target_col : str
    id_col : str or None
        If provided, plot one sub-series per unique value (panel mode).
    title : str or None
    figsize : tuple
    max_series : int
        Maximum number of series to show in panel mode.
    lang : 'zh' or 'en'
    show : bool
        Whether to call plt.show().
    ax : matplotlib Axes or None
        Existing axes for single-series mode.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _ensure_font()

    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])

    if id_col and id_col in data.columns:
        # --- Multi-series panel ---
        series_ids = data[id_col].unique()[:max_series]
        n = len(series_ids)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows / 1.5),
                                 squeeze=False, sharex=False)
        for idx, sid in enumerate(series_ids):
            r, c = divmod(idx, ncols)
            ax_i = axes[r][c]
            sdf = data[data[id_col] == sid].sort_values(time_col)
            ax_i.plot(sdf[time_col], sdf[target_col],
                      color=MODEL_COLORS[idx % len(MODEL_COLORS)], linewidth=1.2)
            _apply_style(ax_i, title=f'{_L("series", lang)}: {sid}',
                         xlabel=_L('time', lang), ylabel=_L('value', lang))
            ax_i.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_i.tick_params(axis='x', rotation=30, labelsize=8)

        # Hide unused subplots
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle(title or _L('series', lang), fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout()
    else:
        # --- Single series ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.plot(data[time_col], data[target_col],
                color=COLORS['actual'], linewidth=1.2, label=target_col)
        _apply_style(ax, title=title or target_col,
                     xlabel=_L('time', lang), ylabel=_L('value', lang))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        fig.tight_layout()

    if show:
        plt.show()

    return fig


# ===========================================================================
#  2. plot_forecast — actual vs forecast with prediction intervals
# ===========================================================================

def plot_forecast(
    train_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    time_col: str,
    target_col: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    history_tail: Optional[int] = None,
    lang: str = 'zh',
    show: bool = True,
    ax=None,
):
    """Plot historical data and forecast with optional prediction intervals.

    Parameters
    ----------
    train_data : pd.DataFrame
        Historical/training data.
    forecast_data : pd.DataFrame
        Forecast output (may contain ``target_col_lower``, ``target_col_upper``).
    time_col, target_col : str
    title : str or None
    figsize : tuple
    history_tail : int or None
        Show only the last N points of history for clarity.
    lang : 'zh' or 'en'
    show : bool
    ax : Axes or None

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _ensure_font()

    train = train_data.copy()
    pred = forecast_data.copy()
    train[time_col] = pd.to_datetime(train[time_col])
    pred[time_col] = pd.to_datetime(pred[time_col])

    if history_tail is not None:
        train = train.iloc[-history_tail:]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # History
    ax.plot(train[time_col], train[target_col],
            color=COLORS['actual'], linewidth=1.3, label=_L('actual', lang))

    # Forecast
    ax.plot(pred[time_col], pred[target_col],
            color=COLORS['forecast'], linewidth=1.8, marker='o', markersize=3,
            label=_L('forecast', lang))

    # Prediction intervals
    lower_col = f'{target_col}_lower'
    upper_col = f'{target_col}_upper'
    if lower_col in pred.columns and upper_col in pred.columns:
        ax.fill_between(
            pred[time_col], pred[lower_col], pred[upper_col],
            color=COLORS['interval'], alpha=0.35,
            label=_L('interval', lang),
        )
        ax.plot(pred[time_col], pred[lower_col],
                color=COLORS['interval'], linewidth=0.8, linestyle='--', alpha=0.6)
        ax.plot(pred[time_col], pred[upper_col],
                color=COLORS['interval'], linewidth=0.8, linestyle='--', alpha=0.6)

    # Multi-quantile intervals (e.g. value_q0.5_lower, value_q0.95_upper)
    q_cols = [c for c in pred.columns if c.startswith(f'{target_col}_q') and c.endswith('_lower')]
    if q_cols:
        alphas = np.linspace(0.15, 0.35, len(q_cols))
        for i, lc in enumerate(sorted(q_cols)):
            uc = lc.replace('_lower', '_upper')
            if uc in pred.columns:
                q_label = lc.replace(f'{target_col}_', '').replace('_lower', '')
                ax.fill_between(
                    pred[time_col], pred[lc], pred[uc],
                    alpha=alphas[i], color=COLORS['secondary'],
                    label=q_label,
                )

    # Vertical line at forecast start
    ax.axvline(x=pred[time_col].iloc[0], color=COLORS['dark'],
               linestyle=':', linewidth=0.8, alpha=0.5)

    _apply_style(ax, title=title or _L('forecast_plot', lang),
                 xlabel=_L('time', lang), ylabel=_L('value', lang))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ===========================================================================
#  3. plot_leaderboard — model ranking bar chart
# ===========================================================================

def plot_leaderboard(
    leaderboard: pd.DataFrame,
    metric_col: str = 'metric',
    model_col: str = 'model',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
    lang: str = 'zh',
    show: bool = True,
):
    """Plot model leaderboard as a horizontal bar chart.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        Pipeline leaderboard with at least ``model`` and ``metric`` columns.
    metric_col, model_col : str
    title : str or None
    figsize : tuple
    lang : 'zh' or 'en'
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    _ensure_font()

    df = leaderboard.copy().sort_values(metric_col, ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [COLORS['primary'] if i > 0 else COLORS['success'] for i in range(n)]
    colors = colors[::-1]  # best model (lowest metric) at bottom gets green

    bars = ax.barh(
        np.arange(n), df[metric_col].values,
        color=colors, edgecolor='white', linewidth=0.5, height=0.6,
    )

    # Labels
    labels = list(df[model_col].values)
    labels[0] = f'{labels[0]} {_L("best", lang)}'
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=10)

    # Value annotations
    for i, (bar, val) in enumerate(zip(bars, df[metric_col].values)):
        ax.text(bar.get_width() + df[metric_col].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color=COLORS['dark'])

    _apply_style(ax, title=title or _L('leaderboard', lang),
                 xlabel=_L('metric', lang), legend=False)
    ax.invert_yaxis()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ===========================================================================
#  4. plot_model_comparison — multi-model forecast overlay
# ===========================================================================

def plot_model_comparison(
    train_data: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    time_col: str,
    target_col: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    history_tail: Optional[int] = None,
    lang: str = 'zh',
    show: bool = True,
):
    """Overlay predictions from multiple models on the same chart.

    Parameters
    ----------
    train_data : pd.DataFrame
    predictions : dict
        {model_name: forecast_df} mapping.
    time_col, target_col : str
    title : str or None
    figsize : tuple
    history_tail : int or None
    lang : 'zh' or 'en'
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _ensure_font()

    train = train_data.copy()
    train[time_col] = pd.to_datetime(train[time_col])
    if history_tail:
        train = train.iloc[-history_tail:]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train[time_col], train[target_col],
            color=COLORS['actual'], linewidth=1.3, label=_L('actual', lang))

    for i, (name, pred_df) in enumerate(predictions.items()):
        pred = pred_df.copy()
        pred[time_col] = pd.to_datetime(pred[time_col])
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(pred[time_col], pred[target_col],
                color=color, linewidth=1.5, marker='o', markersize=2,
                label=name, linestyle='--')

    _apply_style(ax, title=title or _L('comparison', lang),
                 xlabel=_L('time', lang), ylabel=_L('value', lang))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ===========================================================================
#  5. plot_residuals — 4-panel residual diagnostics
# ===========================================================================

def plot_residuals(
    y_true,
    y_pred,
    time_index=None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 9),
    lang: str = 'zh',
    show: bool = True,
):
    """Plot 4-panel residual diagnostics.

    Panels: residual time-plot, histogram, Q-Q, ACF.

    Parameters
    ----------
    y_true, y_pred : array-like
    time_index : array-like or None
    title : str or None
    figsize : tuple
    lang : 'zh' or 'en'
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    _ensure_font()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_true - y_pred
    x = time_index if time_index is not None else np.arange(len(residuals))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title or _L('residuals', lang), fontsize=14, fontweight='bold')

    # 1) Residuals over time
    ax = axes[0, 0]
    ax.plot(x, residuals, color=COLORS['primary'], linewidth=0.8, alpha=0.8)
    ax.axhline(0, color=COLORS['danger'], linestyle='--', linewidth=0.8)
    _apply_style(ax, title=_L('residual', lang),
                 xlabel=_L('time', lang), ylabel=_L('residual', lang),
                 legend=False)

    # 2) Histogram + KDE
    ax = axes[0, 1]
    n_bins = min(50, max(10, len(residuals) // 5))
    ax.hist(residuals, bins=n_bins, density=True, color=COLORS['primary'],
            alpha=0.6, edgecolor='white', linewidth=0.5)
    # Normal fit overlay
    from scipy import stats as sp_stats
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xr, sp_stats.norm.pdf(xr, residuals.mean(), residuals.std()),
            color=COLORS['danger'], linewidth=1.5, label='Normal')
    _apply_style(ax, title=_L('histogram', lang), xlabel=_L('residual', lang))

    # 3) Q-Q plot
    ax = axes[1, 0]
    sp_stats.probplot(residuals, dist='norm', plot=ax)
    ax.get_lines()[0].set(color=COLORS['primary'], markersize=3)
    ax.get_lines()[1].set(color=COLORS['danger'])
    _apply_style(ax, title=_L('qq', lang), legend=False)

    # 4) ACF
    ax = axes[1, 1]
    n_lags = min(30, len(residuals) // 2 - 1)
    if n_lags > 1:
        from PipelineTS.utils.native_stats import acf as native_acf
        acf_vals = native_acf(residuals, nlags=n_lags, fft=True)
        lags = np.arange(len(acf_vals))
        bound = 1.96 / np.sqrt(len(residuals))
        ax.bar(lags, acf_vals, color=COLORS['primary'], width=0.35, alpha=0.8)
        ax.axhline(bound, color=COLORS['danger'], linestyle='--', linewidth=0.8)
        ax.axhline(-bound, color=COLORS['danger'], linestyle='--', linewidth=0.8)
        ax.axhline(0, color=COLORS['dark'], linewidth=0.4)
    _apply_style(ax, title=_L('acf', lang), xlabel='Lag', legend=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


# ===========================================================================
#  6. plot_acf_pacf — ACF and PACF side by side
# ===========================================================================

def plot_acf_pacf(
    series,
    max_lags: int = 30,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
    lang: str = 'zh',
    show: bool = True,
):
    """Plot ACF and PACF side by side.

    Parameters
    ----------
    series : array-like
        Time series values.
    max_lags : int
    title : str or None
    figsize : tuple
    lang : 'zh' or 'en'
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    _ensure_font()

    y = np.asarray(series, dtype=np.float64)
    n_lags = min(max_lags, len(y) // 2 - 1)
    if n_lags < 2:
        warnings.warn("Series too short for ACF/PACF plot.")
        return None

    from PipelineTS.utils.native_stats import acf as native_acf
    from PipelineTS.utils.native_stats import pacf as native_pacf
    acf_vals = native_acf(y, nlags=n_lags, fft=True)
    pacf_vals = native_pacf(y, nlags=n_lags)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    bound = 1.96 / np.sqrt(len(y))

    for ax, vals, label in [(ax1, acf_vals, _L('acf', lang)),
                            (ax2, pacf_vals, _L('pacf', lang))]:
        lags = np.arange(len(vals))
        ax.bar(lags, vals, color=COLORS['primary'], width=0.35, alpha=0.8)
        ax.axhline(bound, color=COLORS['danger'], linestyle='--', linewidth=0.8)
        ax.axhline(-bound, color=COLORS['danger'], linestyle='--', linewidth=0.8)
        ax.axhline(0, color=COLORS['dark'], linewidth=0.4)
        _apply_style(ax, title=label, xlabel='Lag', legend=False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ===========================================================================
#  7. plot_decomposition — trend / seasonal / residual
# ===========================================================================

def plot_decomposition(
    data: pd.DataFrame,
    time_col: str,
    target_col: str,
    period: Optional[int] = None,
    model: str = 'additive',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    lang: str = 'zh',
    show: bool = True,
):
    """Decompose and plot trend, seasonal, and residual components.

    Parameters
    ----------
    data : pd.DataFrame
    time_col, target_col : str
    period : int or None
        Seasonal period. If None, auto-detects.
    model : 'additive' or 'multiplicative'
    title : str or None
    figsize : tuple
    lang : 'zh' or 'en'
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _ensure_font()

    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    y = df[target_col].values.astype(np.float64)
    t = df[time_col].values

    # Auto-detect period
    if period is None:
        try:
            freq = pd.infer_freq(df[time_col])
            freq_map = {'D': 7, 'W': 52, 'MS': 12, 'M': 12, 'h': 24, 'H': 24,
                        'T': 60, 'min': 60, 'B': 5, 'Q': 4, 'QS': 4, 'Y': 1, 'YS': 1}
            period = freq_map.get(freq, max(2, len(y) // 10))
        except Exception:
            period = max(2, len(y) // 10)

    from PipelineTS.utils.native_stats import seasonal_decompose
    result = seasonal_decompose(y, model=model, period=period)
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(title or _L('decomposition', lang), fontsize=14, fontweight='bold')

    components = [
        (y, target_col, COLORS['actual']),
        (trend, _L('trend', lang), COLORS['primary']),
        (seasonal, _L('seasonal', lang), COLORS['secondary']),
        (resid, _L('remainder', lang), COLORS['warning']),
    ]

    for ax, (vals, label, color) in zip(axes, components):
        ax.plot(t, vals, color=color, linewidth=1.0)
        _apply_style(ax, ylabel=label, legend=False)
        if vals is resid:
            ax.axhline(0, color=COLORS['danger'], linestyle='--', linewidth=0.6)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    return fig


# ===========================================================================
#  8. plot_train_test_split — visualize train/test partition
# ===========================================================================

def plot_train_test_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    time_col: str,
    target_col: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
    lang: str = 'zh',
    show: bool = True,
):
    """Visualize train/test data split.

    Parameters
    ----------
    train_data, test_data : pd.DataFrame
    time_col, target_col : str
    title, figsize, lang, show : standard params

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _ensure_font()

    train = train_data.copy()
    test = test_data.copy()
    train[time_col] = pd.to_datetime(train[time_col])
    test[time_col] = pd.to_datetime(test[time_col])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(train[time_col], train[target_col],
            color=COLORS['primary'], linewidth=1.2, label=_L('train', lang))
    ax.plot(test[time_col], test[target_col],
            color=COLORS['warning'], linewidth=1.2, label=_L('test', lang))
    ax.axvline(x=test[time_col].iloc[0], color=COLORS['dark'],
               linestyle=':', linewidth=1.0, alpha=0.7)

    _apply_style(ax, title=title or f'{_L("train", lang)} / {_L("test", lang)}',
                 xlabel=_L('time', lang), ylabel=_L('value', lang))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ===========================================================================
#  9. plot_leaderboard_detail — detailed leaderboard with costs
# ===========================================================================

def plot_leaderboard_detail(
    leaderboard: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    lang: str = 'zh',
    show: bool = True,
):
    """Plot detailed leaderboard with metric + training/eval cost.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        Must have columns: model, metric, train_cost(s), eval_cost(s).
    title : str or None
    figsize : tuple
    lang : 'zh' or 'en'
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    _ensure_font()

    df = leaderboard.copy().sort_values('metric', ascending=True).reset_index(drop=True)
    n = len(df)

    has_cost = 'train_cost(s)' in df.columns and 'eval_cost(s)' in df.columns

    if has_cost:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                        gridspec_kw={'width_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # Metric bars
    colors = [COLORS['success'] if i == 0 else COLORS['primary'] for i in range(n)]
    bars = ax1.barh(np.arange(n), df['metric'].values, color=colors,
                    edgecolor='white', height=0.6)
    labels = df['model'].values.copy()
    labels[0] = f'★ {labels[0]}'
    ax1.set_yticks(np.arange(n))
    ax1.set_yticklabels(labels, fontsize=10)
    for bar, val in zip(bars, df['metric'].values):
        ax1.text(bar.get_width() + df['metric'].max() * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=9)
    _apply_style(ax1, title=_L('leaderboard', lang), xlabel=_L('metric', lang),
                 legend=False)
    ax1.invert_yaxis()

    # Cost bars
    if has_cost:
        x = np.arange(n)
        w = 0.35
        ax2.barh(x - w / 2, df['train_cost(s)'].values, height=w,
                 color=COLORS['info'], label=_L('train_cost', lang))
        ax2.barh(x + w / 2, df['eval_cost(s)'].values, height=w,
                 color=COLORS['warning'], label=_L('eval_cost', lang))
        ax2.set_yticks(x)
        ax2.set_yticklabels(df['model'].values, fontsize=9)
        _apply_style(ax2, title=_L('train_cost', lang), xlabel='秒 (s)' if lang == 'zh' else 'Seconds')
        ax2.invert_yaxis()

    fig.suptitle(title or '', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ===========================================================================
#  TSPlotter — high-level class integrating with Pipeline / SmartRouter
# ===========================================================================

class TSPlotter:
    """High-level plotting interface for PipelineTS.

    Parameters
    ----------
    time_col : str
    target_col : str
    lang : 'zh' or 'en'
        Default language for labels.

    Examples
    --------
    >>> plotter = TSPlotter(time_col='date', target_col='value', lang='zh')
    >>> plotter.plot_series(data)
    >>> plotter.plot_forecast(train, forecast)
    >>> plotter.plot_leaderboard(pipeline.leader_board_)
    """

    def __init__(self, time_col: str, target_col: str, lang: str = 'zh'):
        self.time_col = time_col
        self.target_col = target_col
        self.lang = lang
        _ensure_font()

    def plot_series(self, data, id_col=None, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_series(data, self.time_col, self.target_col,
                           id_col=id_col, **kwargs)

    def plot_forecast(self, train_data, forecast_data, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_forecast(train_data, forecast_data,
                             self.time_col, self.target_col, **kwargs)

    def plot_leaderboard(self, leaderboard, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_leaderboard(leaderboard, **kwargs)

    def plot_leaderboard_detail(self, leaderboard, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_leaderboard_detail(leaderboard, **kwargs)

    def plot_model_comparison(self, train_data, predictions, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_model_comparison(train_data, predictions,
                                     self.time_col, self.target_col, **kwargs)

    def plot_residuals(self, y_true, y_pred, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_residuals(y_true, y_pred, **kwargs)

    def plot_acf_pacf(self, series, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_acf_pacf(series, **kwargs)

    def plot_decomposition(self, data, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_decomposition(data, self.time_col, self.target_col, **kwargs)

    def plot_train_test_split(self, train_data, test_data, **kwargs):
        kwargs.setdefault('lang', self.lang)
        return plot_train_test_split(train_data, test_data,
                                      self.time_col, self.target_col, **kwargs)
