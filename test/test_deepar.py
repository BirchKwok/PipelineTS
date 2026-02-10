"""
Test suite for DeepAR model (spinesTS backend + PipelineTS wrapper).

Tests:
1. SpinesTS DeepAR: low-level fit/predict with Gaussian NLL
2. PipelineTS DeepARModel: full pipeline with interval prediction
3. Point prediction (quantile=None)
4. Predict with external data
5. Pipeline integration (include_models=['deepar'])
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LAGS = 6
PREDICT_N = 3


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
    """Smaller dataset for faster tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


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


# ─── SpinesTS Core DeepAR Tests ─────────────────────────────────────────────

class TestSpinesTSDeepAR:
    def test_fit_predict(self):
        """Test basic fit and predict at spinesTS level."""
        from PipelineTS.spinesTS.nn import DeepAR
        model = DeepAR(
            in_features=10, out_features=10,
            d_model=32, n_blocks=2, n_rwkv_blocks=2,
            loss_fn='mae', random_seed=42
        )
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 10).astype(np.float32)
        model.fit(X, y, epochs=5, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape == (1, 10), f"Expected (1, 10), got {pred.shape}"

    def test_gaussian_output_during_training(self):
        """Test that model outputs [mu | sigma] during training forward pass."""
        from PipelineTS.spinesTS.nn._deepar import DeepARBlock
        import torch
        block = DeepARBlock(in_features=10, out_features=5, d_model=16)
        block._return_distribution = True
        x = torch.randn(4, 10)
        out = block(x)
        # Should output [mu | sigma] = 2 * out_features
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_point_prediction_mode(self):
        """Test that setting _return_distribution=False gives point predictions."""
        from PipelineTS.spinesTS.nn._deepar import DeepARBlock
        import torch
        block = DeepARBlock(in_features=10, out_features=5, d_model=16)
        block._return_distribution = False
        x = torch.randn(4, 10)
        out = block(x)
        assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"

    def test_gaussian_nll_loss(self):
        """Test GaussianNLLLossFn computes valid loss."""
        from PipelineTS.spinesTS.nn._deepar import GaussianNLLLossFn
        import torch
        loss_fn = GaussianNLLLossFn()
        # pred = [mu | sigma], target = y
        pred = torch.cat([torch.randn(8, 5), torch.ones(8, 5)], dim=-1)
        target = torch.randn(8, 5)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Gaussian NLL should be positive"


# ─── PipelineTS DeepARModel Tests ───────────────────────────────────────────

class TestDeepARModel:
    def test_fit_predict_with_quantile(self, sample_data):
        """Test full pipeline with interval prediction."""
        from PipelineTS.nn_model import DeepARModel
        model = DeepARModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=32, n_blocks=2, n_rwkv_blocks=2,
            quantile=0.9, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, sample_data):
        """Test point prediction without interval."""
        from PipelineTS.nn_model import DeepARModel
        model = DeepARModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=32, n_blocks=2, n_rwkv_blocks=2,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_predict_with_data(self, sample_data):
        """Test prediction with external data input."""
        from PipelineTS.nn_model import DeepARModel
        model = DeepARModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=32, n_blocks=2, n_rwkv_blocks=2,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        result = model.predict(PREDICT_N, data=sample_data)
        _check_prediction(result, check_interval=False)

    def test_longer_horizon(self, sample_data):
        """Test prediction horizon > lags (autoregressive extrapolation)."""
        from PipelineTS.nn_model import DeepARModel
        model = DeepARModel(
            time_col='date', target_col='value', lags=LAGS,
            d_model=32, n_blocks=2, n_rwkv_blocks=2,
            quantile=None, epochs=5, patience=3, verbose=False
        )
        model.fit(sample_data)
        n_predict = LAGS + 3  # longer than lags
        result = model.predict(n_predict)
        _check_prediction(result, n=n_predict, check_interval=False)


# ─── Pipeline Integration Test ──────────────────────────────────────────────

class TestDeepARPipelineIntegration:
    def test_pipeline_include_deepar(self, small_data):
        """Test that DeepAR can be included and initialized in ModelPipeline."""
        from PipelineTS.pipeline import ModelPipeline
        # Split data into train/valid with no overlap
        train_data = small_data.iloc[:80].copy()
        valid_data = small_data.iloc[80:].copy()
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['deepar'],
            random_state=42,
            deepar__epochs=5,
            deepar__patience=3,
            deepar__verbose=False,
        )
        pipe.fit(train_data, valid_data=valid_data)
        result = pipe.predict(PREDICT_N, data=valid_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N

    def test_deepar_in_available_models(self):
        """Test that DeepAR is registered in available models."""
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        models = get_all_available_models()
        assert 'deepar' in models, "DeepAR should be in available models"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
