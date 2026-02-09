"""
Comprehensive test suite for plot functions in PipelineTS.

Tests:
- plot_data_period: two-series visualization
- plot_single_series: single series visualization

Note: Uses matplotlib's non-interactive backend to avoid display issues in CI.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def series_data():
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, 50))
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def pred_data():
    dates = pd.date_range('2020-02-20', periods=10, freq='D')
    values = np.sin(np.linspace(0, np.pi, 10))
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def pred_data_with_interval():
    dates = pd.date_range('2020-02-20', periods=10, freq='D')
    values = np.sin(np.linspace(0, np.pi, 10))
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'value_upper': values + 0.2,
        'value_lower': values - 0.2
    })


# ─── plot_data_period ─────────────────────────────────────────────────────────

class TestPlotDataPeriod:
    def test_basic_plot(self, series_data, pred_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_data_period
        plot_data_period(series_data, pred_data, time_col='date', target_col='value')
        plt.close('all')

    def test_plot_with_interval(self, series_data, pred_data_with_interval):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_data_period
        plot_data_period(series_data, pred_data_with_interval,
                         time_col='date', target_col='value')
        plt.close('all')

    def test_plot_with_labels(self, series_data, pred_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_data_period
        plot_data_period(series_data, pred_data,
                         time_col='date', target_col='value',
                         labels=('Train', 'Test'))
        plt.close('all')

    def test_plot_with_interval_labels(self, series_data, pred_data_with_interval):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_data_period
        plot_data_period(series_data, pred_data_with_interval,
                         time_col='date', target_col='value',
                         labels=('Train', 'Pred', 'Upper', 'Lower'))
        plt.close('all')

    def test_wrong_labels_count_raises(self, series_data, pred_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_data_period
        with pytest.raises(ValueError):
            plot_data_period(series_data, pred_data,
                             time_col='date', target_col='value',
                             labels=('A', 'B', 'C'))
        plt.close('all')

    def test_empty_data_raises(self, pred_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_data_period
        empty_df = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'),
                                  'value': pd.Series(dtype='float64')})
        with pytest.raises(ValueError):
            plot_data_period(empty_df, pred_data,
                             time_col='date', target_col='value')
        plt.close('all')


# ─── plot_single_series ───────────────────────────────────────────────────────

class TestPlotSingleSeries:
    def test_basic_plot(self, series_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_single_series
        plot_single_series(series_data, time_col='date', target_col='value')
        plt.close('all')

    def test_plot_with_label(self, series_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_single_series
        plot_single_series(series_data, time_col='date', target_col='value',
                           label='My Series')
        plt.close('all')

    def test_custom_date_fmt(self, series_data):
        import matplotlib.pyplot as plt
        from PipelineTS.plot import plot_single_series
        plot_single_series(series_data, time_col='date', target_col='value',
                           date_fmt='%Y/%m/%d')
        plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
