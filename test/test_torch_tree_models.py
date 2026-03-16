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
        from PipelineTS.ml_model.torch_tree_models import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, **FAST_KWARGS,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)

    def test_fit_predict_no_quantile(self, small_data):
        from PipelineTS.ml_model.torch_tree_models import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, **FAST_KWARGS,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)


class TestTorchBaggingForestModel:
    def test_fit_predict(self, small_data):
        from PipelineTS.ml_model.torch_tree_models import TorchBaggingForestModel
        model = TorchBaggingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, **FAST_KWARGS, dropout=0.2,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.ml_model.torch_tree_models import TorchBaggingForestModel
        model = TorchBaggingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=0.9, **FAST_KWARGS, dropout=0.15,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result)


class TestTorchTreeMultiSeries:
    def test_multi_series(self, panel_data):
        from PipelineTS.ml_model.torch_tree_models import TorchBoostingForestModel
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
        from PipelineTS.ml_model.torch_tree_models import TorchBoostingForestModel
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
    def test_pipeline_catboost(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipe = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            catboost__iterations=16,
            quantile=None,
        )
        pipe.fit(small_data)
        result = pipe.predict(n=PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert 'value' in result.columns

class TestNativeTreeRegistry:
    def test_all_native_models_in_registry(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        models = get_all_available_models()
        for name in ['catboost', 'random_forest', 'xgboost', 'extra_forest', 'gc_forest']:
            assert name in models, f"Missing from registry: {name}"

    def test_model_classes(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        from PipelineTS.ml_model.native_tree_models import (
            CatBoostModel, RandomForestModel, XGBoostModel,
            ExtraForestModel, gcForestModel,
        )
        models = get_all_available_models()
        assert models['catboost'] is CatBoostModel
        assert models['random_forest'] is RandomForestModel
        assert models['xgboost'] is XGBoostModel
        assert models['extra_forest'] is ExtraForestModel
        assert models['gc_forest'] is gcForestModel

    def test_old_torch_names_removed(self):
        from PipelineTS.pipeline.pipeline_models import get_all_available_models
        models = get_all_available_models()
        for old in ['torch_boosting_forest', 'torch_bagging_forest',
                     'deep_forest']:
            assert old not in models, f"Old name still in registry: {old}"

    def test_backward_compat_aliases(self):
        from PipelineTS.ml_model import (
            TorchBoostingForestModel, CatBoostModel,
            TorchBaggingForestModel, RandomForestModel,
            DeepForestModel, gcForestModel,
        )
        assert TorchBoostingForestModel is CatBoostModel
        assert TorchBaggingForestModel is RandomForestModel
        assert DeepForestModel is gcForestModel


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


class TestAdaptiveComplexityController:
    """Tests for the _AdaptiveComplexityController."""

    def test_analyze_returns_all_stats(self):
        from PipelineTS.ml_model._torch_tree import _AdaptiveComplexityController
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = (X[:, 0] * 2 + np.sin(X[:, 1]) + np.random.randn(200) * 0.1).astype(np.float32)
        ctrl = _AdaptiveComplexityController()
        stats = ctrl.analyze(X, y)
        for key in ['n_samples', 'n_features', 'noise_ratio', 'nonlinearity',
                     'autocorr', 'feat_concentration']:
            assert key in stats, f"Missing stat: {key}"
        assert stats['n_samples'] == 200
        assert stats['n_features'] == 10
        assert 0 <= stats['noise_ratio'] <= 1
        assert 0 <= stats['nonlinearity'] <= 1
        assert 0 <= stats['autocorr'] <= 1

    def test_select_complexity_small_data(self):
        from PipelineTS.ml_model._torch_tree import _AdaptiveComplexityController
        np.random.seed(42)
        X = np.random.randn(40, 5).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        ctrl = _AdaptiveComplexityController()
        result = ctrl.select_complexity(X, y)
        assert result['profile'] in ('minimal', 'light')
        assert result['tree_depth'] <= 4
        assert result['n_trees'] <= 48

    def test_select_complexity_large_data(self):
        from PipelineTS.ml_model._torch_tree import _AdaptiveComplexityController
        np.random.seed(42)
        X = np.random.randn(500, 20).astype(np.float32)
        y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2]).astype(np.float32)
        ctrl = _AdaptiveComplexityController()
        result = ctrl.select_complexity(X, y)
        assert result['profile'] in ('moderate', 'heavy', 'maximal')
        assert result['tree_depth'] >= 4
        assert result['n_trees'] >= 32

    def test_select_complexity_cascade_lighter(self):
        from PipelineTS.ml_model._torch_tree import _AdaptiveComplexityController
        np.random.seed(42)
        X = np.random.randn(300, 10).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        ctrl = _AdaptiveComplexityController()
        add_result = ctrl.select_complexity(X, y, ensemble_mode='additive')
        cas_result = ctrl.select_complexity(X, y, ensemble_mode='cascade')
        # Cascade should select equal or lighter complexity
        assert cas_result['complexity_score'] <= add_result['complexity_score']

    def test_select_complexity_user_override(self):
        from PipelineTS.ml_model._torch_tree import _AdaptiveComplexityController
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        ctrl = _AdaptiveComplexityController()
        result = ctrl.select_complexity(X, y, user_depth=7, user_n_trees=100)
        assert result['tree_depth'] == 7
        assert result['n_trees'] == 100
        # auto values should still be computed
        assert 'auto_depth' in result
        assert 'auto_trees' in result

    def test_select_complexity_reasons_populated(self):
        from PipelineTS.ml_model._torch_tree import _AdaptiveComplexityController
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = X[:, 0].astype(np.float32)
        ctrl = _AdaptiveComplexityController()
        result = ctrl.select_complexity(X, y)
        assert isinstance(result['reasons'], list)
        assert len(result['reasons']) > 0


class TestAutoComplexityIntegration:
    """Tests for auto_complexity parameter in model wrappers."""

    def test_wrapper_auto_complexity(self):
        from PipelineTS.ml_model._torch_tree import _TorchTreeWrapper
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = (X[:, 0] * 2 + np.sin(X[:, 1])).astype(np.float32)

        wrapper = _TorchTreeWrapper(
            n_epochs=30, accelerator='cpu', random_state=42,
            auto_complexity=True,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (100,)
        assert not np.isnan(preds).any()
        # complexity_info should be populated
        assert wrapper.complexity_info is not None
        assert 'profile' in wrapper.complexity_info
        assert 'tree_depth' in wrapper.complexity_info
        assert 'n_trees' in wrapper.complexity_info

    def test_boosting_model_auto_complexity(self, small_data):
        from PipelineTS.ml_model.torch_tree_models import TorchBoostingForestModel
        model = TorchBoostingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_epochs=50, accelerator='cpu',
            random_state=42, auto_complexity=True,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)
        # Verify complexity info is accessible
        assert model.model.complexity_info is not None

    def test_bagging_model_auto_complexity(self, small_data):
        from PipelineTS.ml_model.torch_tree_models import TorchBaggingForestModel
        model = TorchBaggingForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_epochs=50, accelerator='cpu',
            random_state=42, auto_complexity=True, dropout=0.1,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_deep_forest_auto_complexity(self, small_data):
        from PipelineTS.ml_model.deep_forest import DeepForestModel
        model = DeepForestModel(
            time_col='date', target_col='value', lags=LAGS,
            quantile=None, n_epochs=50, accelerator='cpu',
            random_state=42, auto_complexity=True, n_layers=2,
        )
        model.fit(small_data)
        result = model.predict(PREDICT_N)
        _check_prediction(result, check_interval=False)

    def test_auto_complexity_get_params(self):
        from PipelineTS.ml_model._torch_tree import _TorchTreeWrapper
        wrapper = _TorchTreeWrapper(auto_complexity=True)
        params = wrapper.get_params()
        assert 'auto_complexity' in params
        assert params['auto_complexity'] is True
