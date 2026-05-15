"""
Comprehensive test suite for all models.

Tests:
- 7 NN models: NLinear, DLinear, NBeats, NHits, TFT, Transformer, TiDE
- 2 ML tree models: CatBoost, RandomForest
- 1 Statistic model: AutoARIMA

Each test verifies:
1. Model instantiation
2. fit() runs without error
3. predict() returns a DataFrame with correct shape and columns
4. Prediction interval columns exist when quantile is set
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_data():
    """Create a simple time series DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture(scope="module")
def small_data():
    """Smaller dataset for faster ML model tests."""
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
    # No NaN in predictions
    assert not result[target_col].isna().any(), "Predictions contain NaN"


# ─── NN Model Tests ──────────────────────────────────────────────────────────

class TestNLinearModel:
    def test_fit_predict(self, sample_data):
        from PipelineTS.nn_model import NLinearModel
        model = NLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_no_quantile(self, sample_data):
        from PipelineTS.nn_model import NLinearModel
        model = NLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


class TestDLinearModel:
    def test_fit_predict(self, sample_data):
        from PipelineTS.nn_model import DLinearModel
        model = DLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestNBeatsModel:
    def test_fit_predict_generic(self, sample_data):
        from PipelineTS.nn_model import NBeatsModel
        model = NBeatsModel(
            time_col='date', target_col='value', lags=LAGS,
            generic_architecture=True, num_stacks=1, num_blocks=1,
            num_layers=2, layer_widths=32,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_interpretable(self, sample_data):
        from PipelineTS.nn_model import NBeatsModel
        model = NBeatsModel(
            time_col='date', target_col='value', lags=LAGS,
            generic_architecture=False, num_stacks=2, num_blocks=1,
            num_layers=2, layer_widths=32,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestNHitsModel:
    def test_fit_predict(self, sample_data):
        from PipelineTS.nn_model import NHitsModel
        model = NHitsModel(
            time_col='date', target_col='value', lags=LAGS,
            num_stacks=2, num_blocks=1, num_layers=2, layer_widths=64,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestTFTModel:
    def test_fit_predict(self, sample_data):
        from PipelineTS.nn_model import TFTModel
        model = TFTModel(
            time_col='date', target_col='value', lags=LAGS,
            hidden_size=16, lstm_layers=1, n_heads=2,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestTransformerModel:
    def test_fit_predict(self, sample_data):
        from PipelineTS.nn_model import TransformerModel
        model = TransformerModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestTiDEModel:
    def test_fit_predict(self, sample_data):
        from PipelineTS.nn_model import TiDEModel
        model = TiDEModel(
            time_col='date', target_col='value', lags=LAGS,
            num_encoder_layers=1, num_decoder_layers=1,
            hidden_size=32, decoder_output_dim=8,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


# ─── ML Model Tests ──────────────────────────────────────────────────────────

class TestCatBoostModel:
    def test_fit_predict(self, small_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, iterations=16
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_predict_with_data(self, small_data):
        from PipelineTS.ml_model import CatBoostModel
        model = CatBoostModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, iterations=16
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N, data=small_data)
        _check_prediction(result, check_interval=False)


class TestRandomForestModel:
    def test_fit_predict(self, small_data):
        from PipelineTS.ml_model import RandomForestModel
        model = RandomForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_estimators=16, random_state=42
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


# ─── Statistic Model Tests ───────────────────────────────────────────────────

class TestProphetModel:
    def test_fit_predict(self, small_data):
        from PipelineTS.statistic_model import ProphetModel
        model = ProphetModel(
            time_col='date', target_col='value', lags=LAGS,
            n_changepoints=10, yearly_seasonality=False,
            weekly_seasonality=True, quantile=0.9,
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_no_quantile(self, small_data):
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


class TestAutoARIMAModel:
    def test_fit_predict(self, small_data):
        from PipelineTS.statistic_model import AutoARIMAModel
        model = AutoARIMAModel(
            time_col='date', target_col='value', lags=LAGS,
            start_p=0, max_p=2, start_q=0, max_q=2,
            seasonal=False, quantile=0.9
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_no_quantile(self, small_data):
        from PipelineTS.statistic_model import AutoARIMAModel
        model = AutoARIMAModel(
            time_col='date', target_col='value', lags=LAGS,
            start_p=0, max_p=2, start_q=0, max_q=2,
            seasonal=False, quantile=None
        )
        model.fit(small_data, cv=2)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── PipelineTS NN Backbone Model Tests ──────────────────────────────────────

class TestBackboneNLinear:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import NLinear
        model = NLinear(in_features=10, out_features=10, loss_fn='mae', random_seed=42)
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


class TestBackboneDLinear:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import DLinear
        model = DLinear(in_features=10, out_features=10, loss_fn='mae', random_seed=42)
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


class TestBackboneNBeats:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import NBeats
        model = NBeats(
            in_features=10, out_features=10, num_stacks=1,
            num_blocks=1, num_layers=2, layer_widths=32,
            loss_fn='mae', random_seed=42
        )
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


class TestBackboneNHiTS:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import NHiTS
        model = NHiTS(
            in_features=10, out_features=10, num_stacks=2,
            num_blocks=1, layer_widths=32,
            loss_fn='mae', random_seed=42
        )
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


class TestBackboneTransformer:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import TSTransformer
        model = TSTransformer(
            in_features=10, out_features=10, d_model=16,
            nhead=2, num_encoder_layers=1,
            loss_fn='mae', random_seed=42
        )
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


class TestBackboneTFT:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import TFT
        model = TFT(
            in_features=10, out_features=10, hidden_size=16,
            n_heads=2, loss_fn='mae', random_seed=42
        )
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


class TestBackboneTiDE:
    def test_fit_predict(self):
        from PipelineTS.nn_model.backbones import TiDE
        model = TiDE(
            in_features=10, out_features=10, hidden_size=32,
            loss_fn='mae', random_seed=42
        )
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10)


# ─── No legacy import test ───────────────────────────────────────────────────

class TestNoLegacyImport:
    """Ensure legacy libraries are not imported at runtime by any active module."""

    def test_no_mapie(self):
        import importlib
        mod = importlib.import_module('PipelineTS.base.base')
        src = open(mod.__file__).read()
        assert 'import mapie' not in src and 'from mapie' not in src, \
            "base.py still imports mapie!"

    def test_no_pmdarima(self):
        import importlib
        mod = importlib.import_module('PipelineTS.statistic_model.auto_arima')
        src = open(mod.__file__).read()
        assert 'import pmdarima' not in src and 'from pmdarima' not in src, \
            "auto_arima.py still imports pmdarima!"

    def test_no_external_statistical_backend_dependency(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        blocked = "stats" + "models"
        for path in list((root / "PipelineTS").rglob("*.py")) + [root / "pyproject.toml"]:
            src = path.read_text(encoding="utf-8")
            assert blocked not in src, f"{path} still references external statistical backend!"

    def test_no_facebook_prophet(self):
        import importlib
        mod = importlib.import_module('PipelineTS.statistic_model.prophet')
        src = open(mod.__file__).read()
        assert 'from prophet import' not in src, \
            "prophet.py still imports facebook prophet!"
        mod2 = importlib.import_module('PipelineTS.pipeline.pipeline_models')
        src2 = open(mod2.__file__).read()
        assert 'from prophet import' not in src2, \
            "pipeline_models.py still imports facebook prophet!"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
