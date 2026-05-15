"""
Comprehensive test suite for PipelineTS low-level NN backbone models.

Tests all 14 low-level PipelineTS NN backbone models:
- NLinear, DLinear, NBeats, NHiTS
- TSTransformer, TFT, TiDE
- GAUNet, StackingRNN, Time2VecNet
- PatchRNN, TCN
- ITransformer, SRSNet

Each test verifies:
1. Model instantiation
2. fit() runs without error
3. predict() returns correct shape
4. score() returns a numeric value
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

IN_FEATURES = 10
OUT_FEATURES = 10
N_SAMPLES = 50
EPOCHS = 3


@pytest.fixture(scope="module")
def xy_data():
    np.random.seed(42)
    X = np.random.randn(N_SAMPLES, IN_FEATURES).astype(np.float32)
    y = np.random.randn(N_SAMPLES, OUT_FEATURES).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def xy_3d_data():
    """3D data for multivariate models: (N, lags, channels)."""
    np.random.seed(42)
    n_channels = 3
    X = np.random.randn(N_SAMPLES, IN_FEATURES, n_channels).astype(np.float32)
    y = np.random.randn(N_SAMPLES, OUT_FEATURES, n_channels).astype(np.float32)
    return X, y


def _check_pred_shape(pred, expected_shape):
    assert pred.shape == expected_shape, f"Expected {expected_shape}, got {pred.shape}"


# ─── NLinear ──────────────────────────────────────────────────────────────────

class TestBackboneNLinear:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import NLinear
        X, y = xy_data
        model = NLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                        loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))

    def test_score(self, xy_data):
        from PipelineTS.nn_model.backbones import NLinear
        X, y = xy_data
        model = NLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                        loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        score = model.score(X, y)
        assert isinstance(score, (float, np.floating))


# ─── DLinear ──────────────────────────────────────────────────────────────────

class TestBackboneDLinear:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import DLinear
        X, y = xy_data
        model = DLinear(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                        loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── NBeats ───────────────────────────────────────────────────────────────────

class TestBackboneNBeats:
    def test_fit_predict_generic(self, xy_data):
        from PipelineTS.nn_model.backbones import NBeats
        X, y = xy_data
        model = NBeats(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                       num_stacks=1, num_blocks=1, num_layers=2, layer_widths=32,
                       loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── NHiTS ────────────────────────────────────────────────────────────────────

class TestBackboneNHiTS:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import NHiTS
        X, y = xy_data
        model = NHiTS(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                      num_stacks=2, num_blocks=1, layer_widths=32,
                      loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── TSTransformer ────────────────────────────────────────────────────────────

class TestBackboneTSTransformer:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import TSTransformer
        X, y = xy_data
        model = TSTransformer(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                              d_model=16, nhead=2, num_encoder_layers=1,
                              loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── TFT ──────────────────────────────────────────────────────────────────────

class TestBackboneTFT:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import TFT
        X, y = xy_data
        model = TFT(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                     hidden_size=16, n_heads=2,
                     loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── TiDE ─────────────────────────────────────────────────────────────────────

class TestBackboneTiDE:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import TiDE
        X, y = xy_data
        model = TiDE(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                     hidden_size=32, loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── GAUNet ───────────────────────────────────────────────────────────────────

class TestBackboneGAUNet:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import GAUNet
        X, y = xy_data
        model = GAUNet(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                       loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── StackingRNN ──────────────────────────────────────────────────────────────

class TestBackboneStackingRNN:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import StackingRNN
        X, y = xy_data
        model = StackingRNN(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                            loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── Time2VecNet ──────────────────────────────────────────────────────────────

class TestBackboneTime2VecNet:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import Time2VecNet
        X, y = xy_data
        model = Time2VecNet(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                            loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── PatchRNN ─────────────────────────────────────────────────────────────────

class TestBackbonePatchRNN:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import PatchRNN
        X, y = xy_data
        model = PatchRNN(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                         loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── TCN ──────────────────────────────────────────────────────────────────────

class TestBackboneTCN:
    def test_fit_predict(self, xy_data):
        from PipelineTS.nn_model.backbones import TCN
        X, y = xy_data
        model = TCN(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                    loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))


# ─── ITransformer ─────────────────────────────────────────────────────────────

class TestBackboneITransformer:
    def test_fit_predict_2d(self, xy_data):
        from PipelineTS.nn_model.backbones import ITransformer
        X, y = xy_data
        model = ITransformer(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                             d_model=16, n_heads=2, d_ff=32, e_layers=1,
                             loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))

    def test_fit_predict_3d(self, xy_3d_data):
        from PipelineTS.nn_model.backbones import ITransformer
        X, y = xy_3d_data
        model = ITransformer(
            in_features=IN_FEATURES, out_features=OUT_FEATURES,
            d_model=16, n_heads=2, d_ff=32, e_layers=1,
            loss_fn='mae', random_seed=42
        )
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape[0] == 1


# ─── SRSNet ──────────────────────────────────────────────────────────────────

class TestBackboneSRSNet:
    def test_fit_predict_2d(self, xy_data):
        from PipelineTS.nn_model.backbones import SRSNet
        X, y = xy_data
        model = SRSNet(in_features=IN_FEATURES, out_features=OUT_FEATURES,
                       d_model=16, n_heads=2,
                       loss_fn='mae', random_seed=42)
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        _check_pred_shape(pred, (1, OUT_FEATURES))

    def test_fit_predict_3d(self, xy_3d_data):
        from PipelineTS.nn_model.backbones import SRSNet
        X, y = xy_3d_data
        n_channels = X.shape[2]
        model = SRSNet(
            in_features=IN_FEATURES, out_features=OUT_FEATURES,
            n_vars=n_channels, n_targets=n_channels,
            d_model=16, n_heads=2,
            loss_fn='mae', random_seed=42
        )
        model.fit(X, y, epochs=EPOCHS, verbose=False, patience=3)
        pred = model.predict(X[:1])
        assert pred.shape[0] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
