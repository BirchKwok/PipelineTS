"""
Comprehensive test suite for spinesTS preprocessing and metrics utilities.

Tests:
- split_series, train_test_split_ts, lag_splits, split_series_multivariate
- GaussRankScaler, MultiDimScaler
- moving_average
- spinesTS metrics: wmape, rmse, mae, mse
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── split_series ─────────────────────────────────────────────────────────────

class TestSplitSeries:
    def test_basic_split(self):
        from PipelineTS.spinesTS.preprocessing import split_series
        data = np.arange(20, dtype=np.float32)
        X, y = split_series(data, data, 5, 3)
        assert X.shape[1] == 5
        assert y.shape[1] == 3
        assert X.shape[0] == y.shape[0]

    def test_split_sizes(self):
        from PipelineTS.spinesTS.preprocessing import split_series
        data = np.arange(50, dtype=np.float32)
        X, y = split_series(data, data, 10, 5)
        assert X.shape[0] > 0
        assert y.shape[0] > 0


# ─── train_test_split_ts ──────────────────────────────────────────────────────

class TestTrainTestSplitTS:
    def test_basic_split(self):
        from PipelineTS.spinesTS.preprocessing import train_test_split_ts
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split_ts(X, y, train_size=0.8)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_preserves_order(self):
        from PipelineTS.spinesTS.preprocessing import train_test_split_ts
        X = np.arange(50).reshape(50, 1).astype(np.float32)
        y = np.arange(50).reshape(50, 1).astype(np.float32)
        X_train, X_test, _, _ = train_test_split_ts(X, y, train_size=0.8)
        assert X_train[-1, 0] < X_test[0, 0], "Time series split should preserve order"


# ─── lag_splits ───────────────────────────────────────────────────────────────

class TestLagSplits:
    def test_basic(self):
        from PipelineTS.spinesTS.preprocessing import lag_splits
        data = np.arange(20, dtype=np.float32)
        X = lag_splits(data, 5)
        assert X.shape[0] > 0
        assert X.shape[1] == 5


# ─── split_series_multivariate ────────────────────────────────────────────────

class TestSplitSeriesMultivariate:
    def test_multivariate_split(self):
        from PipelineTS.spinesTS.preprocessing import split_series_multivariate
        features = np.random.randn(50, 3).astype(np.float32)
        targets = np.random.randn(50, 3).astype(np.float32)
        X, y = split_series_multivariate(features, targets, 5, 3)
        assert X.ndim == 3
        assert X.shape[1] == 5
        assert X.shape[2] == 3
        assert y.ndim == 3
        assert y.shape[1] == 3
        assert y.shape[2] == 3


# ─── GaussRankScaler ──────────────────────────────────────────────────────────

class TestGaussRankScaler:
    def test_fit_transform(self):
        from PipelineTS.spinesTS.preprocessing import GaussRankScaler
        scaler = GaussRankScaler()
        X = np.random.randn(100, 1).astype(np.float64)
        transformed = scaler.fit_transform(X)
        assert transformed.shape == X.shape

    def test_inverse_transform(self):
        from PipelineTS.spinesTS.preprocessing import GaussRankScaler
        scaler = GaussRankScaler()
        X = np.random.randn(100, 1).astype(np.float64)
        transformed = scaler.fit_transform(X)
        recovered = scaler.inverse_transform(transformed)
        assert recovered.shape == X.shape


# ─── moving_average ───────────────────────────────────────────────────────────

class TestMovingAverage:
    def test_basic(self):
        from PipelineTS.spinesTS.preprocessing import moving_average
        data = np.arange(20, dtype=np.float64)
        result = moving_average(data, window_size=3)
        assert result is not None
        assert len(result) > 0


# ─── spinesTS metrics ─────────────────────────────────────────────────────────

class TestSpinesTSMetrics:
    def test_mae(self):
        from PipelineTS.spinesTS.metrics import mae
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = mae(y_true, y_pred)
        assert abs(result - 0.5) < 1e-6

    def test_mse(self):
        from PipelineTS.spinesTS.metrics import mse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mse(y_true, y_pred)
        assert result == 0.0

    def test_rmse(self):
        from PipelineTS.spinesTS.metrics import rmse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        result = rmse(y_true, y_pred)
        assert abs(result - 1.0) < 1e-6

    def test_wmape(self):
        from PipelineTS.spinesTS.metrics import wmape
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 22.0, 27.0])
        result = wmape(y_true, y_pred)
        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_wmape_loss_torch(self):
        import torch
        from PipelineTS.spinesTS.metrics import WMAPELoss
        loss_fn = WMAPELoss()
        inputs = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0

    def test_rmse_loss_torch(self):
        import torch
        from PipelineTS.spinesTS.metrics import RMSELoss
        loss_fn = RMSELoss()
        inputs = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(inputs, targets)
        assert abs(loss.item()) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
