"""
Test suite for GPU-accelerated differentiable tree models.

Tests all 3 TorchTree models:
- TorchBoostingForestModel  (staged gradient boosting)
- TorchBaggingForestModel   (bagging with tree dropout)
- TorchDeepForestModel      (cascade multi-layer)

Each test verifies:
1. Model instantiation with custom parameters
2. fit() runs without error
3. predict() returns correct shape and columns
4. No NaN in predictions
5. Pipeline integration
6. Registry presence
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


# Small configs to keep tests fast (CPU, few trees, few epochs)
FAST_KWARGS = dict(
    n_trees=8, tree_depth=3, n_epochs=50,
    early_stop_patience=10, accelerator='cpu', random_state=42,
)


class TestTorchBoostingForestModel:
    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, **FAST_KWARGS,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, **FAST_KWARGS,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


class TestTorchBaggingForestModel:
    def test_fit_predict(self, small_data):
        from PipelineTS.ml_model import TorchBaggingForestModel
        model = TorchBaggingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, **FAST_KWARGS, dropout=0.2,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model import TorchBaggingForestModel
        model = TorchBaggingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, **FAST_KWARGS, dropout=0.15,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestTorchTreeMultiSeries:
    def test_multi_series(self, panel_data):
        from PipelineTS.ml_model import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, **FAST_KWARGS,
        )
        model.all_configs['id_col'] = 'series_id'
        model.fit(panel_data)
        result = model.predict(PREDICT_N)
        assert 'series_id' in result.columns
        assert len(result) == PREDICT_N * 2


class TestTorchTreeVariableHorizons:
    def test_variable_horizons(self, small_data):
        from PipelineTS.ml_model import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, **FAST_KWARGS,
        )
        model.fit(small_data)
        for h in [1, 5, 10]:
            result = model.predict(h)
            assert len(result) == h, f"Expected {h} rows, got {len(result)}"
            assert not result['value'].isna().any()


class TestTorchTreePipeline:
    def test_pipeline_torch_boosting_forest(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['torch_boosting_forest'],
            torch_boosting_forest__n_trees=8,
            torch_boosting_forest__tree_depth=3,
            torch_boosting_forest__n_epochs=50,
            torch_boosting_forest__accelerator='cpu',
            quantile=None,
        )
        pipe.fit(small_data)
        result = pipe.predict(n=PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert 'value' in result.columns

class TestTorchTreeRegistry:
    def test_all_models_in_registry(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        models = get_all_available_models()
        for name in ['torch_boosting_forest', 'torch_bagging_forest']:
            assert name in models, f"Missing from registry: {name}"

    def test_model_classes(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        from PipelineTS.ml_model.torch_tree_models import (
            TorchBoostingForestModel, TorchBaggingForestModel,
        )
        models = get_all_available_models()
        assert models['torch_boosting_forest'] is TorchBoostingForestModel
        assert models['torch_bagging_forest'] is TorchBaggingForestModel

    def test_old_names_removed(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        models = get_all_available_models()
        for old in ['torch_lightgbm', 'torch_catboost', 'torch_xgboost',
                     'torch_random_forest']:
            assert old not in models, f"Old name still in registry: {old}"


class TestDifferentiableTreeCore:
    """Unit tests for the core PyTorch tree ensemble components."""

    def test_oblivious_tree_forward(self):
        import torch
        from PipelineTS.ml_model._torch_tree import _ObliviousDecisionTree
        tree = _ObliviousDecisionTree(in_features=10, depth=3, out_features=1)
        x = torch.randn(5, 10)
        out = tree(x)
        assert out.shape == (5, 1)

    def test_ensemble_additive_forward(self):
        import torch
        from PipelineTS.ml_model._torch_tree import _DifferentiableTreeEnsemble
        model = _DifferentiableTreeEnsemble(
            in_features=10, out_features=1, n_trees=4, tree_depth=3,
            ensemble_mode='additive',
        )
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 1)

    def test_ensemble_bagging_forward(self):
        import torch
        from PipelineTS.ml_model._torch_tree import _DifferentiableTreeEnsemble
        model = _DifferentiableTreeEnsemble(
            in_features=10, out_features=1, n_trees=4, tree_depth=3,
            ensemble_mode='bagging', dropout=0.2,
        )
        model.train()
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 1)

    def test_ensemble_cascade_forward(self):
        import torch
        from PipelineTS.ml_model._torch_tree import _DifferentiableTreeEnsemble
        model = _DifferentiableTreeEnsemble(
            in_features=10, out_features=1, n_trees=4, tree_depth=3,
            ensemble_mode='cascade', n_layers=2,
        )
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 1)

    def test_torch_tree_wrapper_fit_predict(self):
        from PipelineTS.ml_model._torch_tree import _TorchTreeWrapper
        X = np.random.randn(50, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(50).astype(np.float32) * 0.1

        wrapper = _TorchTreeWrapper(
            n_trees=4, tree_depth=3, n_epochs=30,
            accelerator='cpu', random_state=42,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (50,)
        assert not np.isnan(preds).any()

    def test_torch_tree_wrapper_multi_output(self):
        from PipelineTS.ml_model._torch_tree import _TorchTreeWrapper
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.column_stack([X[:, 0] * 2, X[:, 1] - 1]).astype(np.float32)

        wrapper = _TorchTreeWrapper(
            n_trees=4, tree_depth=3, n_epochs=30,
            accelerator='cpu', random_state=42,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (50, 2)
        assert not np.isnan(preds).any()
