"""
Comprehensive test suite for ModelPipeline and PipelineConfigs.

Tests:
- ModelPipeline: list_all_available_models, fit, predict, get_model, get_model_all_configs
- PipelineConfigs: creation, get_configs
- Pipeline with include_models / exclude_models options
- Pipeline with custom scaler
- Pipeline with quantile prediction
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_data():
    """Small dataset for pipeline tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture(scope="module")
def valid_data():
    """Validation dataset for pipeline tests."""
    np.random.seed(123)
    n = 30
    dates = pd.date_range(start='2020-04-10', periods=n, freq='D')
    values = np.sin(np.linspace(0, np.pi, n)) + np.random.randn(n) * 0.1
    return pd.DataFrame({'date': dates, 'value': values})


LAGS = 6
PREDICT_N = 3


# ─── ModelPipeline.list_all_available_models ──────────────────────────────────

class TestListAllAvailableModels:
    def test_returns_list(self):
        from PipelineTS.pipeline import ModelPipeline
        models = ModelPipeline.list_all_available_models()
        assert isinstance(models, list), "Should return a list"
        assert len(models) > 0, "Should have at least one model"

    def test_contains_known_models(self):
        from PipelineTS.pipeline import ModelPipeline
        models = ModelPipeline.list_all_available_models()
        for expected in ['catboost', 'random_forest', 'd_linear', 'n_linear']:
            assert expected in models, f"Expected '{expected}' in available models"

    def test_sorted(self):
        from PipelineTS.pipeline import ModelPipeline
        models = ModelPipeline.list_all_available_models()
        assert models == sorted(models), "Model list should be sorted"


# ─── ModelPipeline fit/predict with include_models ────────────────────────────

class TestPipelineFitPredict:
    def test_fit_predict_light(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost', 'random_forest'],
            quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert isinstance(leaderboard, pd.DataFrame), "Leaderboard should be a DataFrame"
        assert len(leaderboard) == 2, "Should have 2 models"
        assert 'model' in leaderboard.columns
        assert 'metric' in leaderboard.columns

        result = pipeline.predict(PREDICT_N)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N
        assert 'value' in result.columns
        assert 'date' in result.columns

    def test_fit_predict_with_quantile(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=0.9, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'quantile_acc' in leaderboard.columns

        result = pipeline.predict(PREDICT_N)
        assert 'value_lower' in result.columns
        assert 'value_upper' in result.columns

    def test_fit_with_valid_data(self, small_data, valid_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data, valid_data=valid_data)
        assert isinstance(leaderboard, pd.DataFrame)

    def test_include_models_ml(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models='ml', quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) > 0

    def test_include_single_model_string(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models='catboost', quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) == 1


# ─── ModelPipeline get_model / get_model_all_configs ──────────────────────────

class TestPipelineGetModel:
    def test_get_best_model(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost', 'random_forest'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)
        best = pipeline.get_model()
        assert best is not None, "Best model should not be None"

    def test_get_specific_model(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)
        model_name = pipeline.leader_board_.iloc[0]['model']
        model = pipeline.get_model(model_name)
        assert model is not None

    def test_get_model_all_configs(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)
        configs = pipeline.get_model_all_configs()
        assert isinstance(configs, dict)

    def test_predict_with_model_name(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)
        model_name = pipeline.leader_board_.iloc[0]['model']
        result = pipeline.predict(PREDICT_N, model_name=model_name)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == PREDICT_N


# ─── Pipeline with custom scaler ─────────────────────────────────────────────

class TestPipelineScaler:
    def test_no_scaler(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2, scaler=None
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) > 0

    def test_custom_scaler(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2, scaler=StandardScaler()
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) > 0


# ─── Pipeline with model_init_kwargs ─────────────────────────────────────────

class TestPipelineModelKwargs:
    def test_model_kwargs(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            quantile=None, cv=2,
            catboost__iterations=16
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) > 0


# ─── Pipeline exclude_models ─────────────────────────────────────────────────

class TestPipelineExcludeModels:
    def test_exclude_models(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        # Use ML-only models list and exclude catboost from it
        ml_models = ['catboost', 'random_forest', 'gc_forest']
        excluded = ['catboost']
        remaining = [m for m in ml_models if m not in excluded]
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=remaining,
            quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'catboost' not in [m.split('_0')[0] for m in leaderboard['model'].tolist()]


# ─── Pipeline error handling ─────────────────────────────────────────────────

class TestPipelineErrors:
    def test_include_and_exclude_raises(self):
        from PipelineTS.pipeline import ModelPipeline
        with pytest.raises(ValueError):
            ModelPipeline(
                time_col='date', target_col='value', lags=LAGS,
                include_models=['catboost'], exclude_models=['random_forest']
            )

    def test_data_too_short_raises(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=200,
            include_models=['catboost'], quantile=None, cv=2
        )
        with pytest.raises(ValueError):
            pipeline.fit(small_data)


# ─── PipelineConfigs ──────────────────────────────────────────────────────────

class TestPipelineConfigs:
    def test_create_configs(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', {'init_configs': {'iterations': 32}, 'fit_configs': {}}),
        ])
        assert configs is not None
        assert len(configs.configs) == 1

    def test_get_configs(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', {'init_configs': {'iterations': 32}, 'fit_configs': {}}),
        ])
        model_name = configs.configs[0][1]
        result = configs.get_configs(model_name)
        assert result is not None
        assert 'init_configs' in result
        assert result['init_configs']['iterations'] == 32

    def test_get_configs_not_found(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', {'init_configs': {'iterations': 32}, 'fit_configs': {}}),
        ])
        result = configs.get_configs('non_existent_model')
        assert result is None

    def test_rename_model(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'my_boosting', {'init_configs': {'iterations': 32}, 'fit_configs': {}}),
        ])
        result = configs.get_configs('my_boosting')
        assert result is not None

    def test_multiple_configs(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_v1', {'init_configs': {'iterations': 16}, 'fit_configs': {}}),
            ('catboost', 'boost_v2', {'init_configs': {'iterations': 32}, 'fit_configs': {}}),
        ])
        assert len(configs.configs) == 2
        r1 = configs.get_configs('boost_v1')
        r2 = configs.get_configs('boost_v2')
        assert r1['init_configs']['iterations'] == 16
        assert r2['init_configs']['iterations'] == 32

    def test_pipeline_with_configs(self, small_data):
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_fast', {'init_configs': {'iterations': 16}, 'fit_configs': {}}),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'boost_fast' in leaderboard['model'].tolist()


# ─── PipelineConfigs with pipeline_configs ───────────────────────────────────

class TestPipelineConfigsPipelineLevel:
    """Tests for per-model lags, scaler, and feature engineering via pipeline_configs."""

    def test_pipeline_configs_with_lags(self, small_data):
        """Different models can use different lags via pipeline_configs."""
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_lag4', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'lags': 4},
            }),
            ('catboost', 'boost_lag8', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'lags': 8},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        model_names = leaderboard['model'].tolist()
        assert 'boost_lag4' in model_names
        assert 'boost_lag8' in model_names
        # Verify models were initialized with different lags
        m4 = pipeline.get_model('boost_lag4')
        m8 = pipeline.get_model('boost_lag8')
        assert m4.all_configs['lags'] == 4
        assert m8.all_configs['lags'] == 8

    def test_pipeline_configs_with_scaler_none(self, small_data):
        """A model can opt out of scaling via pipeline_configs scaler=None."""
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_noscale', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'scaler': None},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2, scaler=True
        )
        leaderboard = pipeline.fit(small_data)
        assert 'boost_noscale' in leaderboard['model'].tolist()
        # Model should have scaler=None stored
        assert 'boost_noscale' in pipeline._model_scalers
        assert pipeline._model_scalers['boost_noscale'] is None
        # Predict should work
        pred = pipeline.predict(PREDICT_N, model_name='boost_noscale')
        assert len(pred) == PREDICT_N

    def test_pipeline_configs_with_custom_scaler(self, small_data):
        """A model can use a custom scaler via pipeline_configs."""
        from sklearn.preprocessing import StandardScaler
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_std', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'scaler': StandardScaler()},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2, scaler=True
        )
        leaderboard = pipeline.fit(small_data)
        assert 'boost_std' in leaderboard['model'].tolist()
        # Check the stored scaler is a StandardScaler
        assert 'boost_std' in pipeline._model_scalers
        assert isinstance(pipeline._model_scalers['boost_std'], StandardScaler)
        # Predict should work
        pred = pipeline.predict(PREDICT_N, model_name='boost_std')
        assert len(pred) == PREDICT_N

    def test_pipeline_configs_mixed_scalers(self, small_data):
        """Two models with different scalers should both work correctly."""
        from sklearn.preprocessing import StandardScaler
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_std', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'scaler': StandardScaler()},
            }),
            ('catboost', 'boost_none', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'scaler': None},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2, scaler=True
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) == 2
        # Both should predict
        pred_std = pipeline.predict(PREDICT_N, model_name='boost_std')
        pred_none = pipeline.predict(PREDICT_N, model_name='boost_none')
        assert len(pred_std) == PREDICT_N
        assert len(pred_none) == PREDICT_N
        # Predictions should differ since different scaling
        assert not np.allclose(pred_std['value'].values, pred_none['value'].values, atol=1e-6)

    def test_pipeline_configs_with_differential_n(self, small_data):
        """Per-model differential_n via pipeline_configs (only for models that accept it)."""
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('multi_output_model', 'mo_diff0', {
                'init_configs': {},
                'pipeline_configs': {'differential_n': 0},
            }),
            ('multi_output_model', 'mo_diff1', {
                'init_configs': {},
                'pipeline_configs': {'differential_n': 1},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['multi_output_model'],
            configs=configs, quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'mo_diff0' in leaderboard['model'].tolist()
        assert 'mo_diff1' in leaderboard['model'].tolist()

    def test_pipeline_configs_lags_and_scaler_combined(self, small_data):
        """Combine per-model lags and scaler in one pipeline_configs."""
        from sklearn.preprocessing import StandardScaler
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_custom', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'lags': 8, 'scaler': StandardScaler()},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'boost_custom' in leaderboard['model'].tolist()
        m = pipeline.get_model('boost_custom')
        assert m.all_configs['lags'] == 8
        assert isinstance(pipeline._model_scalers['boost_custom'], StandardScaler)

    def test_pipeline_configs_key_accepted(self):
        """PipelineConfigs should accept pipeline_configs as a valid key."""
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'test_model', {
                'init_configs': {'iterations': 16},
                'pipeline_configs': {'lags': 10, 'scaler': None},
            }),
        ])
        result = configs.get_configs('test_model')
        assert result is not None
        assert 'pipeline_configs' in result
        assert result['pipeline_configs']['lags'] == 10
        assert result['pipeline_configs']['scaler'] is None

    def test_backward_compat_no_pipeline_configs(self, small_data):
        """Models without pipeline_configs should work as before (use global scaler/lags)."""
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('catboost', 'boost_default', {
                'init_configs': {'iterations': 16},
            }),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['catboost'],
            configs=configs, quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'boost_default' in leaderboard['model'].tolist()
        # Should NOT be in _model_scalers (uses global scaler)
        assert 'boost_default' not in pipeline._model_scalers
        pred = pipeline.predict(PREDICT_N, model_name='boost_default')
        assert len(pred) == PREDICT_N


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
