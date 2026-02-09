"""
Comprehensive test suite for all NN models in PipelineTS.

Tests all 14 NN models:
- NLinearModel, DLinearModel, NBeatsModel, NHitsModel
- TFTModel, TransformerModel, TiDEModel
- GAUModel, StackingRNNModel, Time2VecModel
- PatchRNNModel, TCNModel
- ITransformerModel (multivariate), SRSNetModel (multivariate)

Each test verifies:
1. Model instantiation with default and custom parameters
2. fit() runs without error
3. predict() returns a DataFrame with correct shape and columns
4. Prediction interval columns exist when quantile is set
5. No NaN in predictions
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

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
def multivariate_data():
    """Create a multivariate time series DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    v1 = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1
    v2 = np.cos(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1
    v3 = np.sin(np.linspace(0, 2 * np.pi, n)) * 0.5 + np.random.randn(n) * 0.05
    return pd.DataFrame({'date': dates, 'value': v1, 'feature_a': v2, 'feature_b': v3})


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


# ─── NLinearModel ─────────────────────────────────────────────────────────────

class TestNLinearModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import NLinearModel
        model = NLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import NLinearModel
        model = NLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_all_configs(self, sample_data):
        from PipelineTS.nn_model import NLinearModel
        model = NLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        configs = model.all_configs
        assert isinstance(configs, dict), "all_configs should be a dict"


# ─── DLinearModel ─────────────────────────────────────────────────────────────

class TestDLinearModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import DLinearModel
        model = DLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import DLinearModel
        model = DLinearModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── NBeatsModel ──────────────────────────────────────────────────────────────

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


# ─── NHitsModel ───────────────────────────────────────────────────────────────

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


# ─── TFTModel ─────────────────────────────────────────────────────────────────

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


# ─── TransformerModel ─────────────────────────────────────────────────────────

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


# ─── TiDEModel ────────────────────────────────────────────────────────────────

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


# ─── GAUModel ─────────────────────────────────────────────────────────────────

class TestGAUModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import GAUModel
        model = GAUModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False,
            level=2
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import GAUModel
        model = GAUModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False,
            level=2
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── StackingRNNModel ─────────────────────────────────────────────────────────

class TestStackingRNNModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import StackingRNNModel
        model = StackingRNNModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False,
            blocks=1
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import StackingRNNModel
        model = StackingRNNModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False,
            blocks=1
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── Time2VecModel ────────────────────────────────────────────────────────────

class TestTime2VecModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import Time2VecModel
        model = Time2VecModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import Time2VecModel
        model = Time2VecModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── PatchRNNModel ────────────────────────────────────────────────────────────

class TestPatchRNNModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import PatchRNNModel
        model = PatchRNNModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False,
            kernel_size=2
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import PatchRNNModel
        model = PatchRNNModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False,
            kernel_size=2
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── TCNModel ─────────────────────────────────────────────────────────────────

class TestTCNModel:
    def test_fit_predict_with_quantile(self, sample_data):
        from PipelineTS.nn_model import TCNModel
        model = TCNModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, epochs=5, patience=3, verbose=False,
            kernel_size=3
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        from PipelineTS.nn_model import TCNModel
        model = TCNModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, epochs=5, patience=3, verbose=False,
            kernel_size=3
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── ITransformerModel (Multivariate) ─────────────────────────────────────────

class TestITransformerModel:
    def test_univariate_fit_predict(self, sample_data):
        from PipelineTS.nn_model import ITransformerModel
        model = ITransformerModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=16, n_heads=2, d_ff=32, e_layers=1,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_multivariate_multi_input_single_output(self, multivariate_data):
        from PipelineTS.nn_model import ITransformerModel
        model = ITransformerModel(
            time_col='date', target_col='value',
            feature_cols=['value', 'feature_a', 'feature_b'],
            lags=LAGS, d_model=16, n_heads=2, d_ff=32, e_layers=1,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(multivariate_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_no_quantile(self, sample_data):
        from PipelineTS.nn_model import ITransformerModel
        model = ITransformerModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=16, n_heads=2, d_ff=32, e_layers=1,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


# ─── SRSNetModel (Multivariate) ──────────────────────────────────────────────

class TestSRSNetModel:
    def test_univariate_fit_predict(self, sample_data):
        from PipelineTS.nn_model import SRSNetModel
        model = SRSNetModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=16, n_heads=2,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_multivariate_multi_input_single_output(self, multivariate_data):
        from PipelineTS.nn_model import SRSNetModel
        model = SRSNetModel(
            time_col='date', target_col='value',
            feature_cols=['value', 'feature_a', 'feature_b'],
            lags=LAGS, d_model=16, n_heads=2,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(multivariate_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_no_quantile(self, sample_data):
        from PipelineTS.nn_model import SRSNetModel
        model = SRSNetModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=16, n_heads=2,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
