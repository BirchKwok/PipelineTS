"""
Test suite for DeepForestModel (GPU-accelerated gcForest cascade).

Tests:
1. Standalone model: fit/predict with and without quantile
2. Predict with explicit data
3. Multi-series (id_col) support
4. Pipeline integration
5. Pipeline with multi-series
6. Model registry
7. Custom hyperparameters
8. Variable forecast horizons
9. Backward compatibility (TorchDeepForestModel alias)
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
def small_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture(scope="module")
def panel_data():
    np.random.seed(42)
    dfs = []
    for sid in ['A', 'B']:
        n = 80
        dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
        values = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.randn(n) * 0.1
        df = pd.DataFrame({'date': dates, 'value': values, 'series_id': sid})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


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


class TestDeepForestModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, n_trees=8, tree_depth=3, n_layers=2,
            n_epochs=50, early_stop_patience=10, accelerator='cpu',
            random_state=42,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=8, tree_depth=3, n_layers=2,
            n_epochs=50, early_stop_patience=10, accelerator='cpu',
            random_state=42,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_predict_with_data(self, small_data):
        from PipelineTS.ml_model import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=8, tree_depth=3, n_layers=2,
            n_epochs=50, early_stop_patience=10, accelerator='cpu',
            random_state=42,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N, data=small_data)
        _check_prediction(result, check_interval=False)

    def test_multi_series(self, panel_data):
        from PipelineTS.ml_model import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=8, tree_depth=3, n_layers=2,
            n_epochs=50, early_stop_patience=10, accelerator='cpu',
            random_state=42,
        )
        model.all_configs['id_col'] = 'series_id'
        model.fit(panel_data)
        result = model.predict(PREDICT_N)
        assert 'series_id' in result.columns
        assert len(result) == PREDICT_N * 2  # 2 series

    def test_variable_horizons(self, small_data):
        from PipelineTS.ml_model import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=8, tree_depth=3, n_layers=2,
            n_epochs=50, early_stop_patience=10, accelerator='cpu',
            random_state=42,
        )
        model.fit(small_data)
        for h in [1, 5, 10]:
            result = model.predict(h)
            assert len(result) == h, f"Expected {h} rows, got {len(result)}"
            assert not result['value'].isna().any()

    def test_custom_hyperparams(self, small_data):
        from PipelineTS.ml_model import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_trees=16, tree_depth=4, n_layers=3,
            learning_rate=0.05, n_epochs=80, dropout=0.15,
            accelerator='cpu', random_state=42,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


class TestDeepForestPipeline:
    def test_pipeline_integration(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['deep_forest'],
            deep_forest__n_trees=8,
            deep_forest__n_layers=2,
            deep_forest__n_epochs=50,
            deep_forest__accelerator='cpu',
            quantile=None,
        )
        pipe.fit(small_data)
        result = pipe.predict(n=PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert 'value' in result.columns

    def test_pipeline_with_quantile(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['deep_forest'],
            deep_forest__n_trees=8,
            deep_forest__n_layers=2,
            deep_forest__n_epochs=50,
            deep_forest__accelerator='cpu',
            quantile=0.9,
        )
        pipe.fit(small_data)
        result = pipe.predict(n=PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert 'value_lower' in result.columns
        assert 'value_upper' in result.columns

    def test_pipeline_multi_series(self, panel_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            id_col='series_id',
            include_models=['deep_forest'],
            deep_forest__n_trees=8,
            deep_forest__n_layers=2,
            deep_forest__n_epochs=50,
            deep_forest__accelerator='cpu',
            quantile=None,
        )
        pipe.fit(panel_data)
        result = pipe.predict(n=PREDICT_N)
        assert 'series_id' in result.columns


class TestDeepForestRegistry:
    def test_model_in_registry(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        models = get_all_available_models()
        assert 'deep_forest' in models

    def test_model_class(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        from PipelineTS.ml_model.deep_forest import DeepForestModel
        models = get_all_available_models()
        assert models['deep_forest'] is DeepForestModel

    def test_backward_compat_alias(self):
        from PipelineTS.ml_model import TorchDeepForestModel, DeepForestModel
        assert TorchDeepForestModel is DeepForestModel
