"""
Comprehensive test suite for IO (save_model / load_model) in PipelineTS.

Tests:
- save_model / load_model for a single model
- save_model / load_model for a pipeline
- Error handling for invalid paths
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


LAGS = 6
PREDICT_N = 3


# ─── save_model / load_model for single model ────────────────────────────────

class TestSaveLoadSingleModel:
    def test_save_and_load_ml_model(self, small_data, tmp_dir):
        from PipelineTS.ml_model import TorchBoostingForestModel
        from PipelineTS.io import save_model, load_model

        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=16, n_epochs=50
        )
        model.fit(small_data)

        path = os.path.join(tmp_dir, 'test_model.zip')
        save_model(path, model)
        assert os.path.exists(path), "Saved model file should exist"

        loaded = load_model(path)
        assert loaded is not None, "Loaded model should not be None"

        result = loaded.predict(PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N
        assert 'value' in result.columns

    def test_save_and_load_with_scaler(self, small_data, tmp_dir):
        from PipelineTS.ml_model import TorchBoostingForestModel
        from PipelineTS.io import save_model, load_model
        from sklearn.preprocessing import MinMaxScaler

        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=16, n_epochs=50
        )
        model.fit(small_data)

        scaler = MinMaxScaler()
        scaler.fit(small_data[['value']])

        path = os.path.join(tmp_dir, 'test_model_scaler.zip')
        save_model(path, model, scaler=scaler)
        assert os.path.exists(path)

        loaded_model, loaded_scaler = load_model(path)
        assert loaded_model is not None
        assert loaded_scaler is not None

    def test_save_invalid_path_raises(self, small_data):
        from PipelineTS.ml_model import TorchBoostingForestModel
        from PipelineTS.io import save_model

        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=16, n_epochs=50
        )
        model.fit(small_data)

        with pytest.raises(ValueError):
            save_model('/tmp/test_model.txt', model)


# ─── save_model / load_model for pipeline ────────────────────────────────────

class TestSaveLoadPipeline:
    def test_save_and_load_pipeline(self, small_data, tmp_dir):
        from PipelineTS.pipeline import ModelPipeline
        from PipelineTS.io import save_model, load_model

        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['torch_boosting_forest', 'torch_bagging_forest'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)

        path = os.path.join(tmp_dir, 'test_pipeline.zip')
        save_model(path, pipeline)
        assert os.path.exists(path)

        loaded_pipeline = load_model(path)
        assert loaded_pipeline is not None
        assert loaded_pipeline.leader_board_ is not None
        assert loaded_pipeline.best_model_ is not None

        result = loaded_pipeline.predict(PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
