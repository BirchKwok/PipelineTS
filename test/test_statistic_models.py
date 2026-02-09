"""
Comprehensive test suite for all statistic models in PipelineTS.

Tests:
- ProphetModel: fit, predict, quantile, auto_seasonality, lag_features
- AutoARIMAModel: fit, predict, quantile, seasonal options
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture(scope="module")
def sample_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


LAGS = 6
PREDICT_N = 3


def _check_prediction(result, target_col='value', time_col='date',
                       n=PREDICT_N, check_interval=True):
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) == n, f"Expected {n} rows, got {len(result)}"
    assert target_col in result.columns, f"Missing column: {target_col}"
    assert time_col in result.columns, f"Missing column: {time_col}"
    if check_interval:
        assert f"{target_col}_lower" in result.columns, "Missing lower bound"
        assert f"{target_col}_upper" in result.columns, "Missing upper bound"
    assert not result[target_col].isna().any(), "Predictions contain NaN"


# ─── ProphetModel ─────────────────────────────────────────────────────────────

class TestProphetModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            n_changepoints=10, yearly_seasonality=False,
            weekly_seasonality=True, quantile=0.9,
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            n_changepoints=10, yearly_seasonality=False,
            weekly_seasonality=True, quantile=None,
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_auto_seasonality(self, sample_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            auto_seasonality=True, quantile=None,
        )
        model.fit(sample_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_all_configs(self, small_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None,
        )
        model.fit(small_data, cv=2)
        configs = model.all_configs
        assert isinstance(configs, dict)

    def test_lag_features(self, small_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, use_lag_features=True,
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── AutoARIMAModel ───────────────────────────────────────────────────────────

class TestAutoARIMAModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.statistic_model import AutoARIMAModel
        model = AutoARIMAModel(
            time_col='date', target_col='value', lags=LAGS,
            start_p=0, max_p=2, start_q=0, max_q=2,
            seasonal=False, quantile=0.9
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.statistic_model import AutoARIMAModel
        model = AutoARIMAModel(
            time_col='date', target_col='value', lags=LAGS,
            start_p=0, max_p=2, start_q=0, max_q=2,
            seasonal=False, quantile=None
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_all_configs(self, small_data):
        from PipelineTS.statistic_model import AutoARIMAModel
        model = AutoARIMAModel(
            time_col='date', target_col='value', lags=LAGS,
            start_p=0, max_p=2, start_q=0, max_q=2,
            seasonal=False, quantile=None
        )
        model.fit(small_data, cv=2)
        configs = model.all_configs
        assert isinstance(configs, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
