"""
Comprehensive test suite for IO (save_model / load_model) in PipelineTS.

Tests:
- .pts binary format: single model, pipeline, scaler, metadata
- Security: checksum verification, corruption detection
- Utilities: get_file_info, verify_file
- Backward compatibility: legacy .zip format
- Error handling: invalid paths, missing files
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


@pytest.fixture(scope="module")
def fitted_model(small_data):
    from PipelineTS.ml_model import TorchBoostingForestModel
    model = TorchBoostingForestModel(
        time_col='date', target_col='value', lags=LAGS,
        quantile=None, n_trees=16, n_epochs=50
    )
    model.fit(small_data)
    return model


@pytest.fixture(scope="module")
def fitted_pipeline(small_data):
    from PipelineTS.pipeline import ModelPipeline
    pipeline = ModelPipeline(
        time_col='date', target_col='value', lags=LAGS,
        include_models=['torch_boosting_forest', 'torch_bagging_forest'],
        quantile=None, cv=2
    )
    pipeline.fit(small_data)
    return pipeline


@pytest.fixture(scope="module")
def fitted_router(small_data):
    from PipelineTS.pipeline.smart_router import SmartRouter
    router = SmartRouter(
        time_col='date', target_col='value',
        preset='fast', max_models=2, quantile=None,
        search_strategy='basic',
    )
    router.fit(small_data)
    return router


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


LAGS = 6
PREDICT_N = 3


# ─── .pts format: single model ───────────────────────────────────────────────

class TestPtsSingleModel:
    def test_save_and_load(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, load_model

        path = os.path.join(tmp_dir, 'model.pts')
        save_model(path, fitted_model)
        assert os.path.exists(path)

        loaded = load_model(path)
        assert loaded is not None
        result = loaded.predict(PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N
        assert 'value' in result.columns

    def test_save_and_load_with_scaler(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, load_model
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit(np.random.randn(10, 1))

        path = os.path.join(tmp_dir, 'model_scaler.pts')
        save_model(path, fitted_model, scaler=scaler)

        loaded_model, loaded_scaler = load_model(path)
        assert loaded_model is not None
        assert loaded_scaler is not None
        assert hasattr(loaded_scaler, 'transform')

    def test_save_with_metadata(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, get_file_info

        path = os.path.join(tmp_dir, 'model_meta.pts')
        meta = {'author': 'test', 'dataset': 'v2', 'notes': 'experiment 1'}
        save_model(path, fitted_model, metadata=meta)

        info = get_file_info(path)
        assert info['metadata']['author'] == 'test'
        assert info['metadata']['dataset'] == 'v2'
        assert info['model_type'] == 'single_model'

    def test_skip_checksum(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, load_model

        path = os.path.join(tmp_dir, 'model_nocheck.pts')
        save_model(path, fitted_model)
        loaded = load_model(path, verify_checksum=False)
        assert loaded is not None


# ─── .pts format: pipeline ────────────────────────────────────────────────────

class TestPtsPipeline:
    def test_save_and_load_pipeline(self, fitted_pipeline, tmp_dir):
        from PipelineTS.io import save_model, load_model

        path = os.path.join(tmp_dir, 'pipeline.pts')
        save_model(path, fitted_pipeline)
        assert os.path.exists(path)

        loaded = load_model(path)
        assert loaded is not None
        assert loaded.leader_board_ is not None
        assert loaded.best_model_ is not None

        result = loaded.predict(PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N

    def test_pipeline_save_load_method(self, fitted_pipeline, tmp_dir):
        path = os.path.join(tmp_dir, 'pipe_method.pts')
        fitted_pipeline.save(path)

        from PipelineTS.pipeline import ModelPipeline
        loaded = ModelPipeline.load(path)
        assert loaded is not None
        assert loaded.best_model_ is not None

    def test_pipeline_with_metadata(self, fitted_pipeline, tmp_dir):
        from PipelineTS.io import save_model, get_file_info

        path = os.path.join(tmp_dir, 'pipe_meta.pts')
        save_model(path, fitted_pipeline, metadata={'version': '1.0'})

        info = get_file_info(path)
        assert info['model_type'] == 'pipeline'
        assert info['metadata']['version'] == '1.0'
        assert info['metadata']['n_models'] == 2


# ─── .pts format: SmartRouter ─────────────────────────────────────────────────

class TestPtsSmartRouter:
    def test_save_and_load_router(self, fitted_router, tmp_dir):
        from PipelineTS.io import save_model, load_model

        path = os.path.join(tmp_dir, 'router.pts')
        save_model(path, fitted_router)
        assert os.path.exists(path)

        loaded = load_model(path)
        assert loaded is not None
        assert loaded.pipeline_ is not None

        result = loaded.predict(PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N

    def test_router_save_load_method(self, fitted_router, tmp_dir):
        from PipelineTS.pipeline.smart_router import SmartRouter

        path = os.path.join(tmp_dir, 'router_method.pts')
        fitted_router.save(path)

        loaded = SmartRouter.load(path)
        assert loaded is not None
        assert loaded.pipeline_ is not None

    def test_router_with_metadata(self, fitted_router, tmp_dir):
        from PipelineTS.io import save_model, get_file_info

        path = os.path.join(tmp_dir, 'router_meta.pts')
        save_model(path, fitted_router, metadata={'env': 'prod'})

        info = get_file_info(path)
        assert info['model_type'] == 'smart_router'
        assert info['metadata']['env'] == 'prod'

    def test_router_zip_save_rejected(self, fitted_router):
        from PipelineTS.io import save_model
        with pytest.raises(ValueError, match="must end with"):
            save_model('/tmp/router.zip', fitted_router)


# ─── Security: checksum and corruption ────────────────────────────────────────

class TestSecurity:
    def test_verify_file_valid(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, verify_file

        path = os.path.join(tmp_dir, 'valid.pts')
        save_model(path, fitted_model)

        result = verify_file(path)
        assert result['valid'] is True
        assert result['global_checksum_ok'] is True
        assert 'model' in result['section_checksums']
        assert result['section_checksums']['model'] is True

    def test_corrupted_file_detected(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, load_model

        path = os.path.join(tmp_dir, 'corrupt.pts')
        save_model(path, fitted_model)

        # Corrupt a byte in the middle of the file
        with open(path, 'r+b') as f:
            f.seek(100)
            original = f.read(1)
            f.seek(100)
            f.write(bytes([(original[0] + 1) % 256]))

        with pytest.raises(ValueError, match="checksum mismatch"):
            load_model(path)

    def test_corrupted_footer_detected(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, load_model

        path = os.path.join(tmp_dir, 'bad_footer.pts')
        save_model(path, fitted_model)

        with open(path, 'r+b') as f:
            f.seek(-2, 2)  # 2 bytes before end
            f.write(b'\x00\x00')

        with pytest.raises(ValueError, match="footer magic mismatch"):
            load_model(path)

    def test_wrong_magic_detected(self, tmp_dir):
        from PipelineTS.io import load_model

        path = os.path.join(tmp_dir, 'fake.pts')
        with open(path, 'wb') as f:
            f.write(b'\x00' * 100 + b'PTSE')

        with pytest.raises(ValueError, match="magic number mismatch"):
            load_model(path)

    def test_get_file_info(self, fitted_model, tmp_dir):
        from PipelineTS.io import save_model, get_file_info

        path = os.path.join(tmp_dir, 'info.pts')
        save_model(path, fitted_model)

        info = get_file_info(path)
        assert 'model_type' in info
        assert 'created_at' in info
        assert 'pipelinets_version' in info
        assert 'python_version' in info
        assert 'sections' in info
        assert 'file_size_bytes' in info
        assert info['checksum_algo'] == 'sha256'
        assert info['format_version'] == 1


# ─── Backward compatibility: legacy .zip (read-only) ─────────────────────────

class TestLegacyZip:
    def test_save_zip_rejected(self, fitted_model):
        from PipelineTS.io import save_model
        with pytest.raises(ValueError, match="must end with"):
            save_model('/tmp/model.zip', fitted_model)


# ─── Error handling ───────────────────────────────────────────────────────────

class TestErrors:
    def test_invalid_extension_raises(self, fitted_model):
        from PipelineTS.io import save_model
        with pytest.raises(ValueError, match="must end with"):
            save_model('/tmp/model.txt', fitted_model)

    def test_load_nonexistent_raises(self):
        from PipelineTS.io import load_model
        with pytest.raises(ValueError, match="does not exist"):
            load_model('/tmp/nonexistent_model_12345.pts')

    def test_load_directory_raises(self, tmp_dir):
        from PipelineTS.io import load_model
        with pytest.raises(ValueError, match="must be a file name"):
            load_model(tmp_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
