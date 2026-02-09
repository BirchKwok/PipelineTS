"""
Comprehensive test suite for all ML models in PipelineTS.

Tests all 7 ML models:
- CatBoostModel, LightGBMModel, XGBoostModel, RandomForestModel
- WideGBRTModel
- MultiOutputRegressorModel, MultiStepRegressorModel, RegressorChainModel

Each test verifies:
1. Model instantiation with default and custom parameters
2. fit() runs without error
3. predict() returns a DataFrame with correct shape and columns
4. predict(n, data=...) works with explicit data
5. Prediction interval columns exist when quantile is set
6. No NaN in predictions
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
    """Small dataset for ML model tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


LAGS = 6
PREDICT_N = 3


def _check_prediction(result, target_col='value', time_col='date',
                       n=PREDICT_N, check_interval=True):
    """Helper to validate prediction output."""
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert len(result) == n, f"Expected {n} rows, got {len(result)}"
    assert target_col in result.columns, f"Missing column: {target_col}"
    assert time_col in result.columns, f"Missing column: {time_col}"
    if check_interval:
        assert f"{target_col}_lower" in result.columns, "Missing lower bound"
        assert f"{target_col}_upper" in result.columns, "Missing upper bound"
    assert not result[target_col].isna().any(), "Predictions contain NaN"


# ─── CatBoostModel ───────────────────────────────────────────────────────────

class TestCatBoostModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, iterations=50, verbose=False
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, iterations=50, verbose=False
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_predict_with_data(self, small_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, iterations=50, verbose=False
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N, data=small_data)
        _check_prediction(result, check_interval=False)

    def test_all_configs(self, small_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, iterations=50, verbose=False
        )
        model.fit(small_data)
        configs = model.all_configs
        assert isinstance(configs, dict), "all_configs should be a dict"


# ─── LightGBMModel ───────────────────────────────────────────────────────────

class TestLightGBMModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import LightGBMModel
        model = LightGBMModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_estimators=50, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import LightGBMModel
        model = LightGBMModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_predict_with_data(self, small_data):
        from PipelineTS.ml_model import LightGBMModel
        model = LightGBMModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N, data=small_data)
        _check_prediction(result, check_interval=False)


# ─── XGBoostModel ────────────────────────────────────────────────────────────

class TestXGBoostModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import XGBoostModel
        model = XGBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_estimators=50, verbose=0
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import XGBoostModel
        model = XGBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, verbose=0
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── RandomForestModel ───────────────────────────────────────────────────────

class TestRandomForestModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import RandomForestModel
        model = RandomForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_estimators=50, random_state=42
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import RandomForestModel
        model = RandomForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, random_state=42
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── WideGBRTModel ───────────────────────────────────────────────────────────

class TestWideGBRTModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import WideGBRTModel
        model = WideGBRTModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_estimators=50, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import WideGBRTModel
        model = WideGBRTModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_predict_with_data(self, small_data):
        from PipelineTS.ml_model import WideGBRTModel
        model = WideGBRTModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N, data=small_data)
        _check_prediction(result, check_interval=False)

    def test_differential(self, small_data):
        from PipelineTS.ml_model import WideGBRTModel
        model = WideGBRTModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_estimators=50, verbose=-1,
            differential_n=2
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── MultiOutputRegressorModel ───────────────────────────────────────────────

class TestMultiOutputRegressorModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import MultiOutputRegressorModel
        model = MultiOutputRegressorModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import MultiOutputRegressorModel
        model = MultiOutputRegressorModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── MultiStepRegressorModel ─────────────────────────────────────────────────

class TestMultiStepRegressorModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import MultiStepRegressorModel
        model = MultiStepRegressorModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import MultiStepRegressorModel
        model = MultiStepRegressorModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── RegressorChainModel ─────────────────────────────────────────────────────

class TestRegressorChainModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import RegressorChainModel
        model = RegressorChainModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import RegressorChainModel
        model = RegressorChainModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, verbose=-1
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
