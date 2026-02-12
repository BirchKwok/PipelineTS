"""Tests for PipelineTS visualization module."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd
import numpy as np


# ---------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------

def _make_data(n=200, freq='D'):
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq=freq)
    y = 50 + np.cumsum(np.random.randn(n) * 0.5) + np.sin(np.arange(n) * 2 * np.pi / 30) * 5
    return pd.DataFrame({'date': dates, 'value': y})


def _make_panel(n=200, n_series=3):
    base = _make_data(n)
    frames = []
    for i in range(n_series):
        s = base.copy()
        s['value'] = s['value'] * (1 + 0.2 * i) + i * 10
        s['series_id'] = f'S{i}'
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _make_forecast(data, n_pred=20):
    train = data.iloc[:-n_pred].copy()
    last_date = train['date'].iloc[-1]
    freq = pd.infer_freq(data['date'])
    pred_dates = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq),
                               periods=n_pred, freq=freq)
    y_pred = data['value'].iloc[-n_pred:].values
    pred = pd.DataFrame({
        'date': pred_dates,
        'value': y_pred,
        'value_lower': y_pred - 3,
        'value_upper': y_pred + 3,
    })
    return train, pred


# ---------------------------------------------------------------
#  1. Chinese font detection
# ---------------------------------------------------------------

def test_chinese_font_detection():
    from PipelineTS.plot.ts_plot import _find_chinese_font, configure_chinese_font
    font = _find_chinese_font()
    # Should find a font on most systems; on CI it may be None
    print(f"  Detected font: {font}")

    configured = configure_chinese_font(force=True)
    assert isinstance(configured, str)

    # axes.unicode_minus should be False
    assert matplotlib.rcParams['axes.unicode_minus'] is False
    print("[PASS] test_chinese_font_detection")


def test_font_idempotent():
    from PipelineTS.plot.ts_plot import configure_chinese_font
    f1 = configure_chinese_font(force=True)
    f2 = configure_chinese_font()  # should be cached
    assert f1 == f2
    print("[PASS] test_font_idempotent")


# ---------------------------------------------------------------
#  2. plot_series
# ---------------------------------------------------------------

def test_plot_series_single():
    from PipelineTS.plot.ts_plot import plot_series
    data = _make_data()
    fig = plot_series(data, 'date', 'value', title='单序列', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_series_single")


def test_plot_series_panel():
    from PipelineTS.plot.ts_plot import plot_series
    panel = _make_panel()
    fig = plot_series(panel, 'date', 'value', id_col='series_id',
                      title='多序列面板', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_series_panel")


def test_plot_series_english():
    from PipelineTS.plot.ts_plot import plot_series
    data = _make_data()
    fig = plot_series(data, 'date', 'value', lang='en', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_series_english")


# ---------------------------------------------------------------
#  3. plot_forecast
# ---------------------------------------------------------------

def test_plot_forecast_basic():
    from PipelineTS.plot.ts_plot import plot_forecast
    data = _make_data()
    train, pred = _make_forecast(data)
    fig = plot_forecast(train, pred, 'date', 'value', title='预测结果', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_forecast_basic")


def test_plot_forecast_no_interval():
    from PipelineTS.plot.ts_plot import plot_forecast
    data = _make_data()
    train, pred = _make_forecast(data)
    pred_no_int = pred[['date', 'value']].copy()
    fig = plot_forecast(train, pred_no_int, 'date', 'value', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_forecast_no_interval")


def test_plot_forecast_history_tail():
    from PipelineTS.plot.ts_plot import plot_forecast
    data = _make_data()
    train, pred = _make_forecast(data)
    fig = plot_forecast(train, pred, 'date', 'value',
                        history_tail=30, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_forecast_history_tail")


# ---------------------------------------------------------------
#  4. plot_leaderboard
# ---------------------------------------------------------------

def test_plot_leaderboard():
    from PipelineTS.plot.ts_plot import plot_leaderboard
    lb = pd.DataFrame({
        'model': ['lightgbm', 'catboost', 'prophet', 'tide', 'tft'],
        'metric': [2.1, 2.5, 3.2, 4.1, 5.0],
    })
    fig = plot_leaderboard(lb, title='模型排行榜', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_leaderboard")


def test_plot_leaderboard_detail():
    from PipelineTS.plot.ts_plot import plot_leaderboard_detail
    lb = pd.DataFrame({
        'model': ['lightgbm', 'catboost', 'prophet'],
        'metric': [2.1, 2.5, 3.2],
        'train_cost(s)': [5.2, 8.1, 0.01],
        'eval_cost(s)': [0.3, 0.4, 0.1],
    })
    fig = plot_leaderboard_detail(lb, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_leaderboard_detail")


# ---------------------------------------------------------------
#  5. plot_model_comparison
# ---------------------------------------------------------------

def test_plot_model_comparison():
    from PipelineTS.plot.ts_plot import plot_model_comparison
    data = _make_data()
    train, pred = _make_forecast(data)
    preds = {
        'Model A': pred[['date', 'value']].copy(),
        'Model B': pred[['date', 'value']].assign(value=pred['value'] + 2),
        'Model C': pred[['date', 'value']].assign(value=pred['value'] - 1),
    }
    fig = plot_model_comparison(train, preds, 'date', 'value',
                                title='模型预测对比', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_model_comparison")


# ---------------------------------------------------------------
#  6. plot_residuals
# ---------------------------------------------------------------

def test_plot_residuals():
    from PipelineTS.plot.ts_plot import plot_residuals
    np.random.seed(0)
    y_true = np.random.randn(100) * 5 + 50
    y_pred = y_true + np.random.randn(100) * 2
    fig = plot_residuals(y_true, y_pred, title='残差分析', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_residuals")


def test_plot_residuals_with_time():
    from PipelineTS.plot.ts_plot import plot_residuals
    np.random.seed(0)
    t = pd.date_range('2020-01-01', periods=80, freq='D')
    y_true = np.random.randn(80) * 5 + 50
    y_pred = y_true + np.random.randn(80)
    fig = plot_residuals(y_true, y_pred, time_index=t, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_residuals_with_time")


# ---------------------------------------------------------------
#  7. plot_acf_pacf
# ---------------------------------------------------------------

def test_plot_acf_pacf():
    from PipelineTS.plot.ts_plot import plot_acf_pacf
    data = _make_data()
    fig = plot_acf_pacf(data['value'].values, title='自相关分析', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_acf_pacf")


# ---------------------------------------------------------------
#  8. plot_decomposition
# ---------------------------------------------------------------

def test_plot_decomposition():
    from PipelineTS.plot.ts_plot import plot_decomposition
    data = _make_data()
    fig = plot_decomposition(data, 'date', 'value', title='序列分解', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_decomposition")


def test_plot_decomposition_auto_period():
    from PipelineTS.plot.ts_plot import plot_decomposition
    data = _make_data(freq='MS')
    fig = plot_decomposition(data, 'date', 'value', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_decomposition_auto_period")


# ---------------------------------------------------------------
#  9. plot_train_test_split
# ---------------------------------------------------------------

def test_plot_train_test_split():
    from PipelineTS.plot.ts_plot import plot_train_test_split
    data = _make_data()
    train = data.iloc[:160]
    test = data.iloc[160:]
    fig = plot_train_test_split(train, test, 'date', 'value',
                                title='训练集/测试集', show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close('all')
    print("[PASS] test_plot_train_test_split")


# ---------------------------------------------------------------
# 10. TSPlotter class
# ---------------------------------------------------------------

def test_tsplotter_class():
    from PipelineTS.plot.ts_plot import TSPlotter
    data = _make_data()
    train, pred = _make_forecast(data)
    lb = pd.DataFrame({'model': ['a', 'b'], 'metric': [1.0, 2.0]})

    plotter = TSPlotter(time_col='date', target_col='value', lang='zh')

    fig = plotter.plot_series(data, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)

    fig = plotter.plot_forecast(train, pred, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)

    fig = plotter.plot_leaderboard(lb, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)

    fig = plotter.plot_decomposition(data, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)

    fig = plotter.plot_residuals(np.ones(10), np.zeros(10), show=False)
    assert isinstance(fig, matplotlib.figure.Figure)

    plt.close('all')
    print("[PASS] test_tsplotter_class")


# ---------------------------------------------------------------
# 11. Chinese label rendering (smoke test)
# ---------------------------------------------------------------

def test_chinese_labels_render():
    from PipelineTS.plot.ts_plot import plot_forecast, configure_chinese_font
    configure_chinese_font(force=True)

    data = _make_data()
    train, pred = _make_forecast(data)
    fig = plot_forecast(train, pred, 'date', 'value', lang='zh',
                        title='中文预测标题', show=False)
    ax = fig.axes[0]
    assert '实际值' in [t.get_text() for t in ax.get_legend().get_texts()]
    assert '预测值' in [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close('all')
    print("[PASS] test_chinese_labels_render")


# ---------------------------------------------------------------
# 12. Import from PipelineTS.plot
# ---------------------------------------------------------------

def test_import_from_plot_package():
    from PipelineTS.plot import (
        configure_chinese_font,
        TSPlotter,
        plot_series,
        plot_forecast,
        plot_leaderboard,
        plot_model_comparison,
        plot_residuals,
        plot_acf_pacf,
        plot_decomposition,
        plot_train_test_split,
        COLORS,
        MODEL_COLORS,
    )
    assert callable(configure_chinese_font)
    assert callable(plot_series)
    assert isinstance(COLORS, dict)
    assert isinstance(MODEL_COLORS, list)
    print("[PASS] test_import_from_plot_package")


# ---------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------

if __name__ == '__main__':
    test_chinese_font_detection()
    test_font_idempotent()
    test_plot_series_single()
    test_plot_series_panel()
    test_plot_series_english()
    test_plot_forecast_basic()
    test_plot_forecast_no_interval()
    test_plot_forecast_history_tail()
    test_plot_leaderboard()
    test_plot_leaderboard_detail()
    test_plot_model_comparison()
    test_plot_residuals()
    test_plot_residuals_with_time()
    test_plot_acf_pacf()
    test_plot_decomposition()
    test_plot_decomposition_auto_period()
    test_plot_train_test_split()
    test_tsplotter_class()
    test_chinese_labels_render()
    test_import_from_plot_package()

    print(f"\n=== All {20} visualization tests passed! ===")
