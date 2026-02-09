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
        for expected in ['lightgbm', 'xgboost', 'd_linear', 'n_linear']:
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
            include_models=['lightgbm', 'random_forest'],
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
            include_models=['lightgbm'],
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
            include_models=['lightgbm'],
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
            include_models='xgboost', quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) == 1


# ─── ModelPipeline get_model / get_model_all_configs ──────────────────────────

class TestPipelineGetModel:
    def test_get_best_model(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['lightgbm', 'xgboost'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)
        best = pipeline.get_model()
        assert best is not None, "Best model should not be None"

    def test_get_specific_model(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['lightgbm'],
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
            include_models=['lightgbm'],
            quantile=None, cv=2
        )
        pipeline.fit(small_data)
        configs = pipeline.get_model_all_configs()
        assert isinstance(configs, dict)

    def test_predict_with_model_name(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['lightgbm'],
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
            include_models=['lightgbm'],
            quantile=None, cv=2, scaler=None
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) > 0

    def test_custom_scaler(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        from sklearn.preprocessing import StandardScaler
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['lightgbm'],
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
            include_models=['lightgbm'],
            quantile=None, cv=2,
            lightgbm__n_estimators=30
        )
        leaderboard = pipeline.fit(small_data)
        assert len(leaderboard) > 0


# ─── Pipeline exclude_models ─────────────────────────────────────────────────

class TestPipelineExcludeModels:
    def test_exclude_models(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        all_models = ModelPipeline.list_all_available_models()
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            exclude_models=['lightgbm'],
            quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'lightgbm' not in leaderboard['model'].tolist()


# ─── Pipeline error handling ─────────────────────────────────────────────────

class TestPipelineErrors:
    def test_include_and_exclude_raises(self):
        from PipelineTS.pipeline import ModelPipeline
        with pytest.raises(ValueError):
            ModelPipeline(
                time_col='date', target_col='value', lags=LAGS,
                include_models=['lightgbm'], exclude_models=['xgboost']
            )

    def test_data_too_short_raises(self, small_data):
        from PipelineTS.pipeline import ModelPipeline
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=200,
            include_models=['lightgbm'], quantile=None, cv=2
        )
        with pytest.raises(ValueError):
            pipeline.fit(small_data)


# ─── PipelineConfigs ──────────────────────────────────────────────────────────

class TestPipelineConfigs:
    def test_create_configs(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('lightgbm', {'init_configs': {'n_estimators': 100}, 'fit_configs': {}}),
        ])
        assert configs is not None
        assert len(configs.configs) == 1

    def test_get_configs(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('lightgbm', {'init_configs': {'n_estimators': 100}, 'fit_configs': {}}),
        ])
        model_name = configs.configs[0][1]
        result = configs.get_configs(model_name)
        assert result is not None
        assert 'init_configs' in result
        assert result['init_configs']['n_estimators'] == 100

    def test_get_configs_not_found(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('lightgbm', {'init_configs': {'n_estimators': 100}, 'fit_configs': {}}),
        ])
        result = configs.get_configs('non_existent_model')
        assert result is None

    def test_rename_model(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('lightgbm', 'my_lgbm', {'init_configs': {'n_estimators': 100}, 'fit_configs': {}}),
        ])
        result = configs.get_configs('my_lgbm')
        assert result is not None

    def test_multiple_configs(self):
        from PipelineTS.pipeline import PipelineConfigs
        configs = PipelineConfigs([
            ('lightgbm', 'lgbm_v1', {'init_configs': {'n_estimators': 50}, 'fit_configs': {}}),
            ('lightgbm', 'lgbm_v2', {'init_configs': {'n_estimators': 100}, 'fit_configs': {}}),
        ])
        assert len(configs.configs) == 2
        r1 = configs.get_configs('lgbm_v1')
        r2 = configs.get_configs('lgbm_v2')
        assert r1['init_configs']['n_estimators'] == 50
        assert r2['init_configs']['n_estimators'] == 100

    def test_pipeline_with_configs(self, small_data):
        from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
        configs = PipelineConfigs([
            ('lightgbm', 'lgbm_fast', {'init_configs': {'n_estimators': 30}, 'fit_configs': {}}),
        ])
        pipeline = ModelPipeline(
            time_col='date', target_col='value', lags=LAGS,
            include_models=['lightgbm'],
            configs=configs, quantile=None, cv=2
        )
        leaderboard = pipeline.fit(small_data)
        assert 'lgbm_fast' in leaderboard['model'].tolist()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
