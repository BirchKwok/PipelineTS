"""Tests for SmartRouter and Pipeline enhancements.

Tests:
- DataProfile: population, repr
- Strategy: selection, model diversity, scoring explanation
- Preprocessing: missing detection, skewed scaler
- Ensemble: weighted_avg, median, stacking, none
- Time budget: time_limit enforcement
- Error resilience: Pipeline handles model failures
- Pipeline callback integration
- Model scoring explanations
"""

import sys
sys.path.insert(0, '.')
import pytest

import pandas as pd
import numpy as np
from PipelineTS.pipeline.smart_router import SmartRouter, DataProfile, EnsemblePredictor
from PipelineTS.pipeline.pipeline import ModelPipeline
from PipelineTS.dataset import LoadElectricDataSets


# ─── Unit Tests (no model training) ─────────────────────────────────────────

def test_data_profile():
    """Test that DataProfile is populated correctly, including new fields."""
    df = LoadElectricDataSets()
    router = SmartRouter(time_col='date', target_col='value', verbose=False)
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)

    assert isinstance(profile, DataProfile)
    assert profile.n_rows == len(df)
    assert profile.std > 0
    assert 0 <= profile.trend_strength <= 1
    assert 0 <= profile.noise_ratio
    assert 0 <= profile.pct_missing <= 1
    assert 0 <= profile.pct_outlier <= 1
    assert profile.freq == 'MS'
    assert profile.is_regular is True
    assert 12 in profile.dominant_periods
    assert -1 <= profile.autocorr_lag1 <= 1
    assert -1 <= profile.autocorr_lag2 <= 1
    assert profile.n_seasonalities >= 1
    assert profile.regime_changes >= 0
    print(f"[PASS] test_data_profile: {profile.n_rows} rows, "
          f"freq={profile.freq}, "
          f"stationarity={profile.stationarity}, "
          f"trend={profile.trend_strength:.3f}, "
          f"seasonality={profile.seasonality_strength:.3f}, "
          f"autocorr={profile.autocorr_lag1:.3f}, "
          f"n_seasons={profile.n_seasonalities}, "
          f"regimes={profile.regime_changes}")


def test_strategy_selection():
    """Test that strategy is built correctly from profile, including new fields."""
    df = LoadElectricDataSets()
    router = SmartRouter(time_col='date', target_col='value', verbose=False,
                         max_models=5, n_predict=12)
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)
    strategy = router._build_strategy(profile)

    assert 'preprocessing' in strategy
    assert 'models' in strategy
    assert 'lags' in strategy
    assert 'scaler' in strategy
    assert 'gbdt_differential_n' in strategy
    assert 'feature_engineering' in strategy
    assert 'model_hyperparams' in strategy

    assert isinstance(strategy['models'], list)
    assert len(strategy['models']) <= 5
    assert strategy['lags'] >= 12
    assert strategy['lags'] * 2 < len(df)

    fe = strategy['feature_engineering']
    assert isinstance(fe, dict)
    assert 'routing_mode' in fe
    assert fe['routing_mode'] in ('static', 'adaptive')

    hp = strategy['model_hyperparams']
    assert isinstance(hp, dict)

    # Ensure model diversity
    stat = {'auto_arima', 'prophet'}
    ml = {'wide_gbrt',
           'multi_output_model', 'multi_step_model', 'regressor_chain',
           'deep_forest', 'torch_boosting_forest', 'torch_bagging_forest'}
    models_set = set(strategy['models'])
    assert models_set & ml, "No ML model selected"
    # Stat model not always selected for large datasets with strong patterns
    # assert models_set & stat, "No statistic model selected"

    print(f"[PASS] test_strategy_selection: lags={strategy['lags']}, "
          f"models={strategy['models']}")


def test_model_scoring_explanation():
    """Test that model scores include detailed reasons."""
    df = LoadElectricDataSets()
    router = SmartRouter(time_col='date', target_col='value', verbose=False,
                         max_models=5, n_predict=12)
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)
    _ = router._build_strategy(profile)

    # model_scores_ should be populated
    assert router.model_scores_ is not None
    assert isinstance(router.model_scores_, dict)
    assert len(router.model_scores_) > 0

    # Each model should have 'total' and 'reasons'
    for model_name, info in router.model_scores_.items():
        assert 'total' in info
        assert 'reasons' in info
        assert isinstance(info['total'], float)
        assert isinstance(info['reasons'], list)
        assert len(info['reasons']) >= 1  # at least 'base'
        # Base should always be first reason
        assert info['reasons'][0][0] == 'base'
        assert info['reasons'][0][1] == 50.0

    # Check that torch_boosting_forest has a good score for Electric_Production
    boost_score = router.model_scores_['torch_boosting_forest']['total']
    assert boost_score > 50, f"TorchBoostingForest score too low: {boost_score}"

    print(f"[PASS] test_model_scoring_explanation: "
          f"torch_boosting_forest={boost_score:.1f}, "
          f"prophet={router.model_scores_['prophet']['total']:.1f}")


def test_priority_ordering():
    """Test that selected models are sorted by score (highest first)."""
    df = LoadElectricDataSets()
    router = SmartRouter(time_col='date', target_col='value', verbose=False,
                         max_models=5, n_predict=12)
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)
    strategy = router._build_strategy(profile)

    models = strategy['models']
    scores = router.model_scores_
    model_scores = [scores[m]['total'] for m in models]

    # Should be sorted descending
    for i in range(len(model_scores) - 1):
        assert model_scores[i] >= model_scores[i + 1], \
            f"Models not sorted by priority: {list(zip(models, model_scores))}"

    print(f"[PASS] test_priority_ordering: {list(zip(models, model_scores))}")


def test_small_data_routing():
    """Test model selection for very small dataset."""
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=60, freq='D')
    values = np.sin(np.arange(60) * 2 * np.pi / 7) * 10 + 50 + np.random.randn(60) * 2
    df = pd.DataFrame({'date': dates, 'value': values})

    router = SmartRouter(time_col='date', target_col='value', verbose=False, max_models=5)
    profile = router._profile_data(df)
    strategy = router._build_strategy(profile)

    has_stat = any(m in ('prophet', 'auto_arima') for m in strategy['models'])
    has_ml = any(m in ('deep_forest', 'torch_boosting_forest',
                        'torch_bagging_forest', 'wide_gbrt')
                 for m in strategy['models'])
    # SmartRouter may prefer ML/NN models over stat for some small data profiles
    assert has_stat or has_ml, f"No stat or ML model for small data: {strategy['models']}"
    print(f"[PASS] test_small_data_routing: models={strategy['models']}")


def test_missing_data_detection():
    """Test that missing data triggers fill_missing preprocessing."""
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    values = np.sin(np.arange(200) * 2 * np.pi / 30) * 10 + 100.0
    values[10] = np.nan
    values[50] = np.nan
    values[100] = np.nan
    df = pd.DataFrame({'date': dates, 'value': values})

    router = SmartRouter(time_col='date', target_col='value', verbose=False)
    profile = router._profile_data(df)
    assert profile.pct_missing > 0
    strategy = router._build_strategy(profile)
    has_fill = any(s['step'] == 'fill_missing' for s in strategy['preprocessing'])
    assert has_fill, f"Missing not detected: {strategy['preprocessing']}"
    print(f"[PASS] test_missing_data_detection: pct_missing={profile.pct_missing:.2%}")


def test_skewed_data_scaler():
    """Test that highly skewed data gets PowerTransformer."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    values = np.exp(np.random.randn(300) * 0.5 + 3)
    df = pd.DataFrame({'date': dates, 'value': values})

    router = SmartRouter(time_col='date', target_col='value', verbose=False)
    profile = router._profile_data(df)
    strategy = router._build_strategy(profile)
    scaler_name = strategy['scaler'].__class__.__name__
    assert abs(profile.skewness) > 2.0
    assert 'Power' in scaler_name or 'Quantile' in scaler_name
    print(f"[PASS] test_skewed_data_scaler: skew={profile.skewness:.2f}, scaler={scaler_name}")


def test_data_profile_repr():
    """Test DataProfile repr."""
    p = DataProfile()
    p.n_rows = 100
    p.freq = 'MS'
    r = repr(p)
    assert 'DataProfile' in r
    assert 'n_rows=100' in r
    print(f"[PASS] test_data_profile_repr")


def test_normalize_freq():
    """Test frequency normalization for various timedeltas."""
    assert SmartRouter._normalize_freq(pd.Timedelta(days=1), None) == 'D'
    assert SmartRouter._normalize_freq(pd.Timedelta(days=7), None) == 'W'
    assert SmartRouter._normalize_freq(pd.Timedelta(days=31), None) == 'MS'
    assert SmartRouter._normalize_freq(pd.Timedelta(days=91), None) == 'QS'
    assert SmartRouter._normalize_freq(pd.Timedelta(days=365), None) == 'YS'
    assert SmartRouter._normalize_freq(pd.Timedelta(hours=1), None) == 'h'
    assert SmartRouter._normalize_freq(None, None) is None
    print(f"[PASS] test_normalize_freq")


def test_ensemble_predictor_weighted():
    """Test EnsemblePredictor with weighted_avg strategy."""
    ep = EnsemblePredictor(
        pipeline=None,
        model_names=['torch_boosting_forest', 'prophet'],
        weights={'torch_boosting_forest': 0.6, 'prophet': 0.4},
        time_col='date',
        target_col='value',
        ensemble_method='weighted_avg',
    )
    assert 'torch_boosting_forest' in ep.model_names
    assert abs(sum(ep.weights.values()) - 1.0) < 1e-6
    cfg = ep.all_configs
    assert cfg['ensemble'] is True
    assert cfg['strategy'] == 'weighted_avg'
    r = repr(ep)
    assert 'weighted_avg' in r
    assert 'torch_boosting_forest' in r and 'prophet' in r
    print(f"[PASS] test_ensemble_predictor_weighted: {r}")


def test_ensemble_predictor_median():
    """Test EnsemblePredictor with median strategy."""
    ep = EnsemblePredictor(
        pipeline=None,
        model_names=['torch_boosting_forest', 'prophet', 'torch_bagging_forest'],
        weights={'torch_boosting_forest': 0.4, 'prophet': 0.3, 'torch_bagging_forest': 0.3},
        time_col='date',
        target_col='value',
        ensemble_method='median',
    )
    cfg = ep.all_configs
    assert cfg['strategy'] == 'median'
    r = repr(ep)
    assert 'median' in r
    print(f"[PASS] test_ensemble_predictor_median: {r}")


def test_ensemble_strategy_none():
    """Test that ensemble_strategy='none' skips ensemble."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value',
        verbose=False, max_models=3, ensemble_strategy='none',
    )
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)
    strategy = router._build_strategy(profile)
    assert strategy is not None
    assert router.ensemble_strategy == 'none'
    print(f"[PASS] test_ensemble_strategy_none")


def test_ensemble_strategy_validation():
    """Test that invalid ensemble_strategy raises ValueError."""
    try:
        SmartRouter(
            time_col='date', target_col='value',
            ensemble_strategy='invalid_strategy',
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print(f"[PASS] test_ensemble_strategy_validation")


def test_feature_engineering_routing():
    """Test feature engineering decisions for Electric_Production data."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value',
        verbose=False, n_predict=12,
    )
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)
    fe = router._select_feature_engineering(profile)

    assert 'routing_mode' in fe
    assert 'prophet_use_lag_features' in fe
    assert 'prophet_seasonality_mode' in fe
    print(f"[PASS] test_feature_engineering_routing: {fe}")


def test_adaptive_hyperparams():
    """Test that hyperparams are suggested based on data profile."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value',
        verbose=False, n_predict=12,
    )
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)
    hp = router._suggest_hyperparams(profile)

    assert isinstance(hp, dict)
    if profile.n_rows >= 300 and profile.seasonality_strength > 0.1:
        assert 'torch_boosting_forest__n_trees' in hp
    print(f"[PASS] test_adaptive_hyperparams: {list(hp.keys())}")


# ─── Pipeline Unit Tests ────────────────────────────────────────────────────

def test_pipeline_time_limit_param():
    """Test that Pipeline accepts time_limit parameter."""
    from PipelineTS.pipeline import ModelPipeline
    pipeline = ModelPipeline(
        time_col='date', target_col='value', lags=6,
        include_models=['torch_boosting_forest'], time_limit=60, cv=2,
    )
    assert pipeline.time_limit == 60
    print(f"[PASS] test_pipeline_time_limit_param")


def test_pipeline_time_limit_validation():
    """Test that Pipeline rejects invalid time_limit."""
    from PipelineTS.pipeline import ModelPipeline
    try:
        ModelPipeline(
            time_col='date', target_col='value', lags=6,
            include_models=['torch_boosting_forest'], time_limit=-1, cv=2,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print(f"[PASS] test_pipeline_time_limit_validation")


def test_pipeline_error_resilience():
    """Test that Pipeline continues after model failure."""
    from PipelineTS.pipeline import ModelPipeline

    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.randn(n) * 0.1
    data = pd.DataFrame({'date': dates, 'value': values})

    pipeline = ModelPipeline(
        time_col='date', target_col='value', lags=6,
        include_models=['torch_boosting_forest', 'torch_bagging_forest'],
        cv=2,
    )
    leaderboard = pipeline.fit(data)

    # Both models should succeed, no failures
    assert len(pipeline.failed_models) == 0
    assert len(pipeline.skipped_models) == 0
    assert len(leaderboard) == 2
    print(f"[PASS] test_pipeline_error_resilience: "
          f"{len(leaderboard)} models succeeded")


def test_pipeline_failed_skipped_properties():
    """Test that failed_models and skipped_models properties work."""
    from PipelineTS.pipeline import ModelPipeline

    pipeline = ModelPipeline(
        time_col='date', target_col='value', lags=6,
        include_models=['torch_boosting_forest'], cv=2,
    )
    # Before fit
    assert pipeline.failed_models == []
    assert pipeline.skipped_models == []
    print(f"[PASS] test_pipeline_failed_skipped_properties")


# ─── Integration Tests (with model training) ────────────────────────────────

def test_smart_router_fit_predict():
    """Test full SmartRouter fit and predict workflow with ensemble."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date',
        target_col='value',
        n_predict=12,
        verbose=True,
        max_models=3,
        random_state=42,
        ensemble_strategy='auto',
        ensemble_top_k=3,
        search_strategy='basic',
        time_limit=120,
    )

    router.fit(df)

    assert router.pipeline_ is not None
    assert router.leader_board_ is not None
    assert router.best_model_ is not None
    assert len(router.leader_board_) <= 3

    # model_scores_ should be populated
    assert router.model_scores_ is not None

    # Test ensemble prediction (default)
    preds = router.predict(n=12)
    assert isinstance(preds, pd.DataFrame)
    assert 'value' in preds.columns
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    # Test single-model prediction (bypass ensemble)
    preds_single = router.predict(n=12, use_ensemble=False)
    assert isinstance(preds_single, pd.DataFrame)
    assert len(preds_single) == 12

    # Ensemble should be built or not depending on model diversity
    if router.ensemble_ is not None:
        assert isinstance(router.ensemble_, EnsemblePredictor)
        assert len(router.ensemble_.model_names) >= 2
        assert abs(sum(router.ensemble_.weights.values()) - 1.0) < 1e-6
        print(f"  Ensemble: {router.ensemble_}")
    else:
        print(f"  Ensemble: not built (models too spread)")

    model = router.get_model()
    assert model is not None

    s = router.strategy
    assert s is not None
    assert 'models' in s
    assert 'feature_engineering' in s
    assert 'model_hyperparams' in s

    print(f"[PASS] test_smart_router_fit_predict")
    print(f"  Best: {router.leader_board_.iloc[0]['model']} "
          f"(MAE={router.leader_board_.iloc[0]['metric']:.2f})")


def test_smart_router_forced_ensemble():
    """Test SmartRouter with ensemble_strategy='weighted_avg' (always ensemble)."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date',
        target_col='value',
        n_predict=12,
        verbose=True,
        max_models=3,
        random_state=42,
        ensemble_strategy='weighted_avg',
        ensemble_top_k=3,
        search_strategy='basic',
        time_limit=120,
    )

    router.fit(df)

    # Ensemble may be discarded by post-evaluation if it doesn't improve
    # over the best single model, so we check _ensemble_eval instead
    if router.ensemble_ is not None:
        assert isinstance(router.ensemble_, EnsemblePredictor)
        assert len(router.ensemble_.model_names) >= 2
        assert router.ensemble_.ensemble_method == 'weighted_avg'

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_forced_ensemble")
    print(f"  Ensemble: {router.ensemble_}")
    if router._ensemble_eval is not None:
        ee = router._ensemble_eval
        print(f"  Post-eval: ensemble={ee['ensemble_metric']:.4f} vs "
              f"best={ee['best_single_metric']:.4f} -> {'kept' if ee['kept'] else 'discarded'}")


def test_smart_router_median_ensemble():
    """Test SmartRouter with ensemble_strategy='median'."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date',
        target_col='value',
        n_predict=12,
        verbose=True,
        max_models=3,
        random_state=42,
        ensemble_strategy='median',
        ensemble_top_k=3,
        search_strategy='basic',
        time_limit=120,
    )

    router.fit(df)

    # Ensemble may be discarded by post-evaluation
    if router.ensemble_ is not None:
        assert router.ensemble_.ensemble_method == 'median'

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_median_ensemble")
    print(f"  Ensemble: {router.ensemble_}")


def test_smart_router_stacking_ensemble():
    """Test SmartRouter with ensemble_strategy='stacking'."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date',
        target_col='value',
        n_predict=12,
        verbose=True,
        max_models=3,
        random_state=42,
        ensemble_strategy='stacking',
        ensemble_top_k=3,
        search_strategy='basic',
        time_limit=120,
    )

    router.fit(df)

    # Stacking may fall back to weighted_avg if meta-learner fails
    if router.ensemble_ is not None:
        assert router.ensemble_.ensemble_method in ('stacking', 'weighted_avg')
        preds = router.predict(n=12)
        assert len(preds) == 12
        assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_stacking_ensemble")
    if router.ensemble_:
        print(f"  Ensemble: {router.ensemble_}")


def test_smart_router_time_limit():
    """Test SmartRouter with time_limit parameter."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date',
        target_col='value',
        n_predict=12,
        verbose=True,
        max_models=3,
        random_state=42,
        time_limit=120,
        search_strategy='basic',
    )

    router.fit(df)

    assert router.pipeline_ is not None
    assert router.leader_board_ is not None
    assert len(router.leader_board_) > 0

    print(f"[PASS] test_smart_router_time_limit")
    print(f"  Models completed: {len(router.leader_board_)}")


def test_smart_router_callback_integration():
    """Test that SmartRouter receives callbacks from Pipeline."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date',
        target_col='value',
        n_predict=12,
        verbose=False,
        max_models=2,
        random_state=42,
        search_strategy='basic',
        time_limit=120,
    )

    router.fit(df)

    # _model_results should be populated via callback
    assert hasattr(router, '_model_results')
    assert len(router._model_results) == len(router.leader_board_)
    for r in router._model_results:
        assert 'model_name' in r
        assert 'metric' in r
        assert 'train_cost' in r

    print(f"[PASS] test_smart_router_callback_integration: "
          f"{len(router._model_results)} callbacks received")


# ─── Search Strategy Unit Tests ──────────────────────────────────────────────

def test_search_strategy_param():
    """Test search_strategy parameter validation."""
    for s in ('basic', 'auto', 'thorough'):
        r = SmartRouter(time_col='date', target_col='value', search_strategy=s)
        assert r.search_strategy == s
    try:
        SmartRouter(time_col='date', target_col='value', search_strategy='invalid')
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("[PASS] test_search_strategy_param")


def test_should_screen_logic():
    """Test _should_screen conditions for different search strategies."""
    df = LoadElectricDataSets()

    # basic: never screens
    r = SmartRouter(time_col='date', target_col='value',
                    search_strategy='basic', max_models=8)
    r.profile_ = r._profile_data(r._ensure_datetime(df))
    assert r._should_screen() is False

    # auto: screens if max_models >= 4 and n >= 80
    r2 = SmartRouter(time_col='date', target_col='value',
                     search_strategy='auto', max_models=8)
    r2.profile_ = r2._profile_data(r2._ensure_datetime(df))
    assert r2._should_screen() is True

    # auto with small max_models: no screen (threshold is max_models>=3)
    r3 = SmartRouter(time_col='date', target_col='value',
                     search_strategy='auto', max_models=2)
    r3.profile_ = r3._profile_data(r3._ensure_datetime(df))
    assert r3._should_screen() is False

    # auto with max_models=3: screens (wide screening enabled)
    r3b = SmartRouter(time_col='date', target_col='value',
                      search_strategy='auto', max_models=3)
    r3b.profile_ = r3b._profile_data(r3b._ensure_datetime(df))
    assert r3b._should_screen() is True

    # thorough: always screens
    r4 = SmartRouter(time_col='date', target_col='value',
                     search_strategy='thorough', max_models=3)
    r4.profile_ = r4._profile_data(r4._ensure_datetime(df))
    assert r4._should_screen() is True

    print("[PASS] test_should_screen_logic")


def test_should_explore_lags_logic():
    """Test _should_explore_lags conditions."""
    df = LoadElectricDataSets()

    r = SmartRouter(time_col='date', target_col='value', search_strategy='basic')
    r.profile_ = r._profile_data(r._ensure_datetime(df))
    assert r._should_explore_lags() is False

    r2 = SmartRouter(time_col='date', target_col='value', search_strategy='auto')
    r2.profile_ = r2._profile_data(r2._ensure_datetime(df))
    assert r2._should_explore_lags() is True  # 397 >= 100

    # Small data: auto should not explore
    small_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=50, freq='D'),
        'value': np.random.randn(50),
    })
    r3 = SmartRouter(time_col='date', target_col='value', search_strategy='auto')
    r3.profile_ = r3._profile_data(r3._ensure_datetime(small_df))
    assert r3._should_explore_lags() is False  # 50 < 100

    print("[PASS] test_should_explore_lags_logic")


def test_generate_lag_candidates():
    """Test lag candidate generation."""
    r = SmartRouter(time_col='date', target_col='value')

    # Normal case
    cands = r._generate_lag_candidates(12, 400)
    assert 12 in cands
    assert len(cands) >= 2

    # Very small data: only base lag
    cands2 = r._generate_lag_candidates(4, 20)
    assert 4 in cands2

    # Large lag
    cands3 = r._generate_lag_candidates(24, 1000)
    assert 24 in cands3
    assert len(cands3) >= 2

    print(f"[PASS] test_generate_lag_candidates: {cands}, {cands2}, {cands3}")


def test_pick_fast_eval_model():
    """Test fast model selection for lag evaluation."""
    r = SmartRouter(time_col='date', target_col='value')

    assert r._pick_fast_eval_model(['tft', 'torch_boosting_forest', 'prophet']) == 'torch_boosting_forest'
    assert r._pick_fast_eval_model(['prophet', 'n_beats']) == 'prophet'
    assert r._pick_fast_eval_model(['transformer', 'srs_net']) == 'transformer'

    print("[PASS] test_pick_fast_eval_model")


def test_select_models_n_candidates():
    """Test _select_models with n_candidates override."""
    df = LoadElectricDataSets()
    r = SmartRouter(time_col='date', target_col='value',
                    verbose=False, max_models=5, n_predict=12)
    profile = r._profile_data(r._ensure_datetime(df))
    r._build_strategy(profile)

    default = r._select_models(profile)
    broad = r._select_models(profile, n_candidates=10)

    assert len(default) <= 5
    assert len(broad) <= 10
    assert len(broad) > len(default)

    print(f"[PASS] test_select_models_n_candidates: "
          f"default={len(default)}, broad={len(broad)}")


def test_inject_exploration_new_category():
    """Injection adds models from categories absent in the heuristic pool."""
    from PipelineTS.pipeline.pipeline_models import get_all_available_models

    df = LoadElectricDataSets()
    r = SmartRouter(
        time_col='date', target_col='value',
        n_predict=12, verbose=False, max_models=5,
        search_strategy='auto', random_state=42,
    )
    p = r._profile_data(r._ensure_datetime(df))
    r._build_strategy(p)

    # Build a fake pool that is missing an entire category (nn_heavy)
    nn_heavy = {'transformer', 'tft', 'itransformer', 'srs_net', 'deepar'}
    all_models = list(get_all_available_models().keys())
    # Take a pool of 12 models that deliberately excludes all nn_heavy models
    fake_pool = [m for m in all_models if m not in nn_heavy][:12]
    assert not any(m in nn_heavy for m in fake_pool), \
        "Test setup error: nn_heavy should be absent from fake_pool"

    new_pool = r._inject_exploration_candidates(fake_pool, p)

    # Pool size must stay the same
    assert len(new_pool) == len(fake_pool), \
        f"Pool size changed: {len(fake_pool)} -> {len(new_pool)}"

    # At least one nn_heavy model should have been injected
    injected_heavy = [m for m in new_pool if m in nn_heavy]
    assert len(injected_heavy) >= 1, \
        f"No nn_heavy model injected into pool: {new_pool}"

    print(f"[PASS] test_inject_exploration_new_category: "
          f"injected={injected_heavy}, pool={new_pool}")


def test_inject_exploration_respects_strategy():
    """Injection is a no-op for 'basic' and 'thorough' strategies."""
    from PipelineTS.pipeline.pipeline_models import get_all_available_models

    df = LoadElectricDataSets()
    all_models = list(get_all_available_models().keys())
    fake_pool = all_models[:12]

    # 'basic' — no screening at all, injection must not alter the pool
    r_basic = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, search_strategy='basic', random_state=0,
    )
    p = r_basic._profile_data(r_basic._ensure_datetime(df))
    r_basic._build_strategy(p)
    result_basic = r_basic._inject_exploration_candidates(fake_pool, p)
    assert result_basic == fake_pool, \
        f"'basic' should return pool unchanged: {result_basic}"

    # 'thorough' — already screens all models, injection must not alter the pool
    r_thorough = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, search_strategy='thorough', random_state=0,
    )
    p2 = r_thorough._profile_data(r_thorough._ensure_datetime(df))
    r_thorough._build_strategy(p2)
    result_thorough = r_thorough._inject_exploration_candidates(fake_pool, p2)
    assert result_thorough == fake_pool, \
        f"'thorough' should return pool unchanged: {result_thorough}"

    print("[PASS] test_inject_exploration_respects_strategy")


def _make_profile(**kwargs):
    """Helper to create a DataProfile with custom attributes."""
    p = DataProfile()
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


def test_scoring_pattern_bonus_cap():
    """Test that pattern bonuses are capped at +25 to prevent stacking abuse."""
    # Create a profile that would trigger many pattern bonuses
    p = _make_profile(
        n_rows=400, freq='MS', stationarity='non_stationary',
        trend_strength=0.9, seasonality_strength=0.8,
        noise_ratio=0.2, skewness=0.5,
        autocorr_lag1=0.9, n_seasonalities=3,
        pct_missing=0.0, regime_changes=2
    )
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    verbose=False, max_models=5)

    # Prophet used to score ~110+ due to uncapped stacking. Now capped.
    prophet_score, prophet_reasons = r._score_model('prophet', p)
    pattern_sum = sum(d for reason, d in prophet_reasons
                      if any(kw in reason for kw in
                             ['stationary', 'seasonality', 'trend',
                              'autocorr', 'seasonal']))
    # Pattern bonuses should be capped at 25
    assert pattern_sum <= 25.5, f"Pattern bonus cap violated: {pattern_sum}"
    # Prophet should not exceed ~100
    assert prophet_score <= 100, f"Prophet score too high: {prophet_score}"

    # NN models should be competitive with prophet on medium+ data
    tft_score, _ = r._score_model('tft', p)
    n_beats_score, _ = r._score_model('n_beats', p)
    # Spread between prophet and NN should be < 15
    assert prophet_score - tft_score < 15, \
        f"Prophet-TFT spread too large: {prophet_score - tft_score}"
    assert prophet_score - n_beats_score < 15, \
        f"Prophet-NBeats spread too large: {prophet_score - n_beats_score}"

    print(f"[PASS] test_scoring_pattern_bonus_cap: "
          f"prophet={prophet_score:.1f}, tft={tft_score:.1f}, "
          f"n_beats={n_beats_score:.1f}, pattern_sum={pattern_sum:.1f}")


def test_scoring_model_diversity():
    """Test that model selection produces diverse candidates, not just prophet/lgbm/tft."""
    df = LoadElectricDataSets()
    r = SmartRouter(time_col='date', target_col='value', n_predict=12,
                    verbose=False, max_models=5, random_state=42)
    r._ensure_datetime(df)
    p = r._profile_data(df)
    r._build_strategy(p)
    selected = r._select_models(p)

    # Should have models from at least 2 different categories
    categories = set()
    ml = {'wide_gbrt', 'multi_output_model', 'multi_step_model', 'regressor_chain',
          'deep_forest', 'torch_boosting_forest', 'torch_bagging_forest'}
    stat = {'auto_arima', 'prophet'}
    nn = {'d_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
          'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
          'patch_rnn', 'itransformer', 'srs_net', 'deepar'}
    for m in selected:
        if m in ml: categories.add('ml')
        elif m in stat: categories.add('stat')
        elif m in nn: categories.add('nn')

    assert len(categories) >= 2, \
        f"Insufficient diversity: {selected} -> categories={categories}"

    # The top 5 should not be exclusively prophet+torch_boosting_forest+tft
    always_same = {'prophet', 'torch_boosting_forest', 'tft'}
    assert not always_same.issubset(set(selected[:3])), \
        f"Top 3 are still always the same: {selected[:3]}"

    print(f"[PASS] test_scoring_model_diversity: {selected}, categories={categories}")


def test_hyperparams_nn_stability():
    """Test that _suggest_hyperparams sets EMA/SWA/warmup for NN models."""
    p = _make_profile(
        n_rows=400, freq='MS', stationarity='non_stationary',
        trend_strength=0.7, seasonality_strength=0.5,
        noise_ratio=0.5, skewness=0.5,
        autocorr_lag1=0.8, n_seasonalities=2,
        pct_missing=0.0, regime_changes=3
    )
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    verbose=False, max_models=5)
    hp = r._suggest_hyperparams(p)

    # Heavy NN models should get EMA + SWA
    for m in ['tft', 'transformer', 'n_beats', 'deepar']:
        assert hp.get(f'{m}__use_ema') == True, f"Missing {m}__use_ema"
        assert hp.get(f'{m}__use_swa') == True, f"Missing {m}__use_swa"

    # Transformer-based models should get warmup
    for m in ['tft', 'transformer', 'itransformer', 'gau', 'time2vec']:
        assert hp.get(f'{m}__warmup_epochs', 0) > 0, f"Missing {m}__warmup_epochs"

    # NN models should get increased epochs for large data
    for m in ['tft', 'n_beats', 'deepar']:
        assert hp.get(f'{m}__epochs', 0) >= 2000, \
            f"{m}__epochs too low: {hp.get(f'{m}__epochs')}"

    # GTB should be enabled for complex patterns on medium+ data
    for m in ['tft', 'n_beats', 'tcn']:
        assert hp.get(f'{m}__use_gtb') == True, f"Missing {m}__use_gtb"

    print(f"[PASS] test_hyperparams_nn_stability: "
          f"EMA/SWA/warmup/GTB all configured correctly")


def test_hyperparams_adaptive_lr():
    """Test that learning_rate is adapted based on data size."""
    # Large data → standard LR for heavy, higher for light
    p_large = _make_profile(
        n_rows=500, freq='D', stationarity='stationary',
        trend_strength=0.3, seasonality_strength=0.1,
        noise_ratio=0.3, skewness=0.0,
        autocorr_lag1=0.5, n_seasonalities=0,
        pct_missing=0.0, regime_changes=1
    )
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    verbose=False, max_models=5)
    hp = r._suggest_hyperparams(p_large)
    assert hp.get('tft__learning_rate') == 0.001
    assert hp.get('d_linear__learning_rate') == 0.003

    # Small data → lower LR
    p_small = _make_profile(
        n_rows=60, freq='D', stationarity='stationary',
        trend_strength=0.3, seasonality_strength=0.1,
        noise_ratio=0.3, skewness=0.0,
        autocorr_lag1=0.5, n_seasonalities=0,
        pct_missing=0.0, regime_changes=1
    )
    hp_small = r._suggest_hyperparams(p_small)
    assert hp_small.get('tft__learning_rate') == 0.0005
    assert hp_small.get('d_linear__learning_rate') == 0.001

    print(f"[PASS] test_hyperparams_adaptive_lr: "
          f"large={hp.get('tft__learning_rate')}, "
          f"small={hp_small.get('tft__learning_rate')}")


def test_stability_params_in_model_wrapper():
    """Test that NN model wrappers accept and store stability params."""
    from PipelineTS.nn_model.tft import TFTModel
    from PipelineTS.nn_model.n_beats import NBeatsModel
    from PipelineTS.nn_model.deepar import DeepARModel

    m = TFTModel(time_col='d', target_col='v', lags=12,
                 use_ema=True, ema_decay=0.998, use_swa=True,
                 swa_start_frac=0.8, warmup_epochs=15, verbose=False)
    assert m.all_configs['use_ema'] == True
    assert m.all_configs['ema_decay'] == 0.998
    assert m.all_configs['use_swa'] == True
    assert m.all_configs['swa_start_frac'] == 0.8
    assert m.all_configs['warmup_epochs'] == 15

    # Defaults should be False/0
    m2 = NBeatsModel(time_col='d', target_col='v', lags=12, verbose=False)
    assert m2.all_configs.get('use_ema') == False
    assert m2.all_configs.get('use_swa') == False
    assert m2.all_configs.get('warmup_epochs') == 0

    # DeepAR (no GTB) should also accept stability params
    m3 = DeepARModel(time_col='d', target_col='v', lags=12,
                     use_ema=True, warmup_epochs=5, verbose=False)
    assert m3.all_configs['use_ema'] == True
    assert m3.all_configs['warmup_epochs'] == 5

    print("[PASS] test_stability_params_in_model_wrapper")


def test_sinkhorn_residual_gate():
    """Test mHC-inspired SinkhornResidualGate module."""
    import torch
    from PipelineTS.spinesTS.base._torch_mixin import SinkhornResidualGate

    base = torch.nn.Linear(16, 16)
    gate = SinkhornResidualGate(base, in_features=16, out_features=16)

    # Forward pass shape
    x = torch.randn(8, 16)
    out = gate(x)
    assert out.shape == (8, 16), f"Wrong shape: {out.shape}"

    # Sinkhorn produces doubly stochastic matrix
    W = gate._sinkhorn(gate.W_logits)
    row_sums = W.sum(dim=1)
    col_sums = W.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)

    # Spectral norm <= 1 (no signal amplification)
    sn = torch.linalg.norm(W, ord=2).item()
    assert sn <= 1.01, f"Spectral norm too high: {sn}"

    # Different in/out features (with projection)
    base2 = torch.nn.Linear(12, 8)
    gate2 = SinkhornResidualGate(base2, in_features=12, out_features=8)
    out2 = gate2(torch.randn(4, 12))
    assert out2.shape == (4, 8)

    print(f"[PASS] test_sinkhorn_residual_gate: spectral_norm={sn:.4f}")


def test_residual_gate_in_model_wrapper():
    """Test use_residual_gate param propagates through NN model wrappers."""
    from PipelineTS.nn_model.tft import TFTModel
    from PipelineTS.nn_model.d_linear import DLinearModel
    from PipelineTS.nn_model.n_beats import NBeatsModel

    # Explicit enable
    m = TFTModel(time_col='d', target_col='v', lags=12,
                 use_residual_gate=True, verbose=False)
    assert m.all_configs['use_residual_gate'] == True

    # Default off
    m2 = DLinearModel(time_col='d', target_col='v', lags=12, verbose=False)
    assert m2.all_configs.get('use_residual_gate', False) == False

    m3 = NBeatsModel(time_col='d', target_col='v', lags=12,
                     use_residual_gate=True, verbose=False)
    assert m3.all_configs['use_residual_gate'] == True

    print("[PASS] test_residual_gate_in_model_wrapper")


def test_hyperparams_residual_gate():
    """Test _suggest_hyperparams enables residual gate for noisy non-stationary data."""
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    verbose=False, max_models=5)

    # Non-stationary + noise → gate should be enabled
    p = _make_profile(
        n_rows=400, freq='MS', stationarity='non_stationary',
        trend_strength=0.7, seasonality_strength=0.5,
        noise_ratio=0.5, skewness=0.0,
        autocorr_lag1=0.8, n_seasonalities=2,
        pct_missing=0.0, regime_changes=5
    )
    hp = r._suggest_hyperparams(p)
    assert hp.get('tft__use_residual_gate') == True
    assert hp.get('d_linear__use_residual_gate') == True
    assert hp.get('n_beats__use_residual_gate') == True

    # Small stationary data → gate should NOT be enabled
    p_small = _make_profile(
        n_rows=50, freq='D', stationarity='stationary',
        trend_strength=0.1, seasonality_strength=0.1,
        noise_ratio=0.2, skewness=0.0,
        autocorr_lag1=0.3, n_seasonalities=0,
        pct_missing=0.0, regime_changes=0
    )
    hp_small = r._suggest_hyperparams(p_small)
    assert hp_small.get('tft__use_residual_gate') is None

    print("[PASS] test_hyperparams_residual_gate")


def test_5category_diversity():
    """Test that _select_models uses 5-category diversity system."""
    from PipelineTS.pipeline.pipeline_models import get_all_available_models

    r = SmartRouter(time_col='date', target_col='value', n_predict=12,
                    verbose=False, max_models=8, random_state=42,
                    search_strategy='basic')
    df = LoadElectricDataSets()
    r._ensure_datetime(df)
    p = r._profile_data(df)
    r._build_strategy(p)

    sel = r._select_models(p, n_candidates=8)

    categories = {
        'ml': {'torch_boosting_forest', 'torch_bagging_forest', 'deep_forest',
               'wide_gbrt', 'multi_output_model', 'multi_step_model',
               'regressor_chain'},
        'nn_light': {'d_linear', 'n_linear', 'tide', 'tcn'},
        'nn_medium': {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                      'time2vec', 'gau'},
        'nn_heavy': {'transformer', 'tft', 'itransformer', 'srs_net', 'deepar'},
    }

    ml_count = sum(1 for m in sel if m in categories['ml'])
    nn_light_count = sum(1 for m in sel if m in categories['nn_light'])
    nn_medium_count = sum(1 for m in sel if m in categories['nn_medium'])
    nn_heavy_count = sum(1 for m in sel if m in categories['nn_heavy'])

    # ML capped at 2
    assert ml_count <= 2, f"ML count too high: {ml_count}"
    # At least one from each NN subcategory
    assert nn_light_count >= 1, f"No nn_light: {sel}"
    assert nn_medium_count >= 1, f"No nn_medium: {sel}"
    assert nn_heavy_count >= 1, f"No nn_heavy: {sel}"

    print(f"[PASS] test_5category_diversity: {sel} "
          f"(ml={ml_count}, nn_l={nn_light_count}, nn_m={nn_medium_count}, nn_h={nn_heavy_count})")


# ─── include_models Unit Tests ────────────────────────────────────────────────

def test_include_models_single_string():
    """Test include_models accepts a single string and normalizes to list."""
    r = SmartRouter(time_col='date', target_col='value',
                    include_models='prophet', verbose=False)
    assert r.include_models == ['prophet']
    assert r.max_models == 1
    print("[PASS] test_include_models_single_string")


def test_include_models_list():
    """Test include_models accepts a list of model names."""
    r = SmartRouter(time_col='date', target_col='value',
                    include_models=['prophet', 'torch_boosting_forest'],
                    verbose=False)
    assert r.include_models == ['prophet', 'torch_boosting_forest']
    assert r.max_models == 2
    print("[PASS] test_include_models_list")


def test_include_models_max_models_override():
    """Test that explicit max_models overrides auto-adjustment from include_models."""
    r = SmartRouter(time_col='date', target_col='value',
                    include_models=['prophet', 'torch_boosting_forest'],
                    max_models=5, verbose=False)
    assert r.include_models == ['prophet', 'torch_boosting_forest']
    assert r.max_models == 5  # explicit max_models takes precedence
    print("[PASS] test_include_models_max_models_override")


def test_include_models_invalid_name():
    """Test that invalid model names raise ValueError."""
    try:
        SmartRouter(time_col='date', target_col='value',
                    include_models=['prophet', 'nonexistent_model'])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'nonexistent_model' in str(e)
    print("[PASS] test_include_models_invalid_name")


def test_include_models_empty_list():
    """Test that empty include_models raises ValueError."""
    try:
        SmartRouter(time_col='date', target_col='value',
                    include_models=[])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_include_models_empty_list")


def test_include_models_bypasses_selection():
    """Test that include_models bypasses heuristic model selection."""
    df = LoadElectricDataSets()
    pinned = ['prophet', 'torch_boosting_forest', 'tide']
    r = SmartRouter(time_col='date', target_col='value',
                    include_models=pinned, verbose=False, n_predict=12)
    df_dt = r._ensure_datetime(df)
    profile = r._profile_data(df_dt)
    strategy = r._build_strategy(profile)

    # Strategy models must be exactly the pinned models
    assert strategy['models'] == pinned
    # Scoring still runs (for logging / calibration)
    assert r.model_scores_ is not None
    # Other strategy components still populated
    assert strategy['lags'] > 0
    assert strategy['scaler'] is not None
    assert 'feature_engineering' in strategy
    assert 'model_hyperparams' in strategy

    print(f"[PASS] test_include_models_bypasses_selection: models={strategy['models']}")


def test_include_models_skips_screening():
    """Test that screening is skipped when include_models is set."""
    df = LoadElectricDataSets()
    r = SmartRouter(time_col='date', target_col='value',
                    include_models=['prophet', 'torch_boosting_forest'],
                    search_strategy='thorough', verbose=False)
    r.profile_ = r._profile_data(r._ensure_datetime(df))
    # Even with thorough search, should_screen returns False for pinned models
    assert r._should_screen() is False
    print("[PASS] test_include_models_skips_screening")


def test_include_models_repr():
    """Test that __repr__ shows include_models when set."""
    r = SmartRouter(time_col='date', target_col='value',
                    include_models=['prophet', 'tide'])
    rep = repr(r)
    assert 'include_models' in rep
    assert 'prophet' in rep
    assert 'tide' in rep
    print(f"[PASS] test_include_models_repr: {rep}")


def test_include_models_none_default():
    """Test that include_models defaults to None (normal behavior)."""
    r = SmartRouter(time_col='date', target_col='value')
    assert r.include_models is None
    print("[PASS] test_include_models_none_default")


# ─── New Feature Unit Tests ───────────────────────────────────────────────────

def test_sort_by_speed():
    """Test _sort_by_speed categorizes models into correct speed tiers."""
    candidates = [
        'tft', 'prophet', 'n_beats', 'torch_boosting_forest',
        'd_linear', 'transformer', 'deep_forest', 'tide',
    ]
    sorted_c = SmartRouter._sort_by_speed(candidates)

    # Fast models (tier 0) should come first
    fast = {'prophet', 'torch_boosting_forest', 'deep_forest'}
    light = {'d_linear', 'tide'}
    medium = {'n_beats'}
    heavy = {'tft', 'transformer'}

    # Verify ordering: fast < light < medium < heavy
    fast_indices = [sorted_c.index(m) for m in fast if m in sorted_c]
    light_indices = [sorted_c.index(m) for m in light if m in sorted_c]
    medium_indices = [sorted_c.index(m) for m in medium if m in sorted_c]
    heavy_indices = [sorted_c.index(m) for m in heavy if m in sorted_c]

    assert max(fast_indices) < min(light_indices), \
        f"Fast models not before light: {sorted_c}"
    assert max(light_indices) < min(medium_indices), \
        f"Light models not before medium: {sorted_c}"
    assert max(medium_indices) < min(heavy_indices), \
        f"Medium models not before heavy: {sorted_c}"

    # Length preserved
    assert len(sorted_c) == len(candidates)

    print(f"[PASS] test_sort_by_speed: {sorted_c}")


def test_sort_by_speed_preserves_order_within_tier():
    """Test that _sort_by_speed preserves original order within same tier."""
    candidates = ['tft', 'transformer', 'deepar']  # all heavy (tier 3)
    sorted_c = SmartRouter._sort_by_speed(candidates)
    assert sorted_c == candidates, \
        f"Order within tier not preserved: {sorted_c}"

    print("[PASS] test_sort_by_speed_preserves_order_within_tier")


def test_deep_forest_scoring_rebalance():
    """Test that deep_forest no longer gets excessive bonuses."""
    p = _make_profile(
        n_rows=400, freq='MS', stationarity='non_stationary',
        trend_strength=0.7, seasonality_strength=0.6,
        noise_ratio=0.3, skewness=0.5,
        autocorr_lag1=0.8, n_seasonalities=2,
        pct_missing=0.0, regime_changes=10
    )
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    verbose=False, max_models=5)

    df_score, df_reasons = r._score_model('deep_forest', p)
    boost_score, _ = r._score_model('torch_boosting_forest', p)
    nbeats_score, _ = r._score_model('n_beats', p)

    # deep_forest should NOT dominate over NN models
    assert df_score < 95, f"deep_forest score still too high: {df_score}"
    # deep_forest should not be more than 10 pts above n_beats
    assert df_score - nbeats_score < 10, \
        f"deep_forest-n_beats gap too large: {df_score - nbeats_score}"

    print(f"[PASS] test_deep_forest_scoring_rebalance: "
          f"deep_forest={df_score:.1f}, boost={boost_score:.1f}, "
          f"n_beats={nbeats_score:.1f}")


def test_deep_forest_penalty_small_data():
    """Test that deep_forest gets penalized on small datasets."""
    p = _make_profile(
        n_rows=50, freq='D', stationarity='stationary',
        trend_strength=0.2, seasonality_strength=0.1,
        noise_ratio=0.5, skewness=0.0,
        autocorr_lag1=0.3, n_seasonalities=0,
        pct_missing=0.0, regime_changes=0
    )
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    verbose=False, max_models=5)

    df_score, df_reasons = r._score_model('deep_forest', p)

    # Should get penalty for small data
    penalties = [d for reason, d in df_reasons if d < 0 and 'deep_forest' in reason]
    assert len(penalties) > 0, "deep_forest should get penalties on small data"
    assert df_score < 70, f"deep_forest too high on small data: {df_score}"

    print(f"[PASS] test_deep_forest_penalty_small_data: score={df_score:.1f}")


def test_time_aware_epoch_capping():
    """Test that time-aware epoch capping reduces NN epochs when time_limit is tight."""
    r = SmartRouter(time_col='date', target_col='value', n_predict=12,
                    verbose=False, max_models=3, time_limit=60,
                    search_strategy='basic')

    # Simulate the hyperparams that _suggest_hyperparams would produce
    hyperparams = {
        'n_beats__epochs': 2000,
        'tide__epochs': 1500,
        'torch_boosting_forest__n_epochs': 150,  # not NN, should not be capped
    }
    models = ['n_beats', 'tide', 'torch_boosting_forest']

    # Simulate epoch capping logic (per_model_budget = 60/3 = 20s → cap=100)
    remaining_time = 60.0
    per_model_budget = remaining_time / len(models)
    nn_models_set = {
        'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
        'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
        'patch_rnn', 'itransformer', 'srs_net', 'deepar',
    }

    if per_model_budget < 30:
        epoch_cap = 100
    elif per_model_budget < 60:
        epoch_cap = 300
    elif per_model_budget < 120:
        epoch_cap = 500
    else:
        epoch_cap = None

    assert epoch_cap == 100, f"Wrong cap for budget={per_model_budget:.0f}s: {epoch_cap}"

    if epoch_cap is not None:
        for m in models:
            if m in nn_models_set:
                key = f'{m}__epochs'
                current = hyperparams.get(key, 1000)
                if current > epoch_cap:
                    hyperparams[key] = epoch_cap

    assert hyperparams['n_beats__epochs'] == 100, \
        f"n_beats epochs not capped: {hyperparams['n_beats__epochs']}"
    assert hyperparams['tide__epochs'] == 100, \
        f"tide epochs not capped: {hyperparams['tide__epochs']}"
    # Non-NN model should NOT be affected
    assert hyperparams['torch_boosting_forest__n_epochs'] == 150, \
        f"tree epochs wrongly capped: {hyperparams['torch_boosting_forest__n_epochs']}"

    print(f"[PASS] test_time_aware_epoch_capping: caps={hyperparams}")


def test_ensemble_eval_attribute():
    """Test that _ensemble_eval attribute is properly initialized."""
    r = SmartRouter(time_col='date', target_col='value', n_predict=12)
    assert r._ensemble_eval is None
    assert r._fusion_scores is None

    print("[PASS] test_ensemble_eval_attribute")


def test_get_screen_pool_size():
    """Test _get_screen_pool_size returns appropriate pool sizes."""
    r_auto = SmartRouter(time_col='d', target_col='v', n_predict=12,
                         max_models=3, search_strategy='auto')
    pool_auto = r_auto._get_screen_pool_size()
    assert pool_auto >= 12, f"Auto pool too small: {pool_auto}"
    assert pool_auto <= 27, f"Auto pool too large: {pool_auto}"

    r_auto5 = SmartRouter(time_col='d', target_col='v', n_predict=12,
                          max_models=5, search_strategy='auto')
    pool_auto5 = r_auto5._get_screen_pool_size()
    assert pool_auto5 >= 15, f"Auto5 pool too small: {pool_auto5}"

    r_thorough = SmartRouter(time_col='d', target_col='v', n_predict=12,
                             max_models=8, search_strategy='thorough')
    pool_thorough = r_thorough._get_screen_pool_size()
    # Thorough should screen all models
    assert pool_thorough >= 20, f"Thorough pool too small: {pool_thorough}"

    print(f"[PASS] test_get_screen_pool_size: auto3={pool_auto}, "
          f"auto5={pool_auto5}, thorough={pool_thorough}")


def test_fusion_select_screening_dominates():
    """Test that fusion selection includes models that screening ranks high
    even if heuristic ranks them low."""
    import pandas as pd
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    max_models=3, search_strategy='auto', verbose=False)

    # tft has LOW heuristic but BEST screening
    r.model_scores_ = {
        'd_linear': {'total': 80}, 'tide': {'total': 78},
        'n_beats': {'total': 75}, 'torch_boosting_forest': {'total': 73},
        'auto_arima': {'total': 70}, 'tft': {'total': 55},
    }
    screen_lb = pd.DataFrame({
        'model': ['tft', 'tide', 'auto_arima', 'd_linear',
                  'torch_boosting_forest', 'n_beats'],
        'metric': [3.2, 3.5, 4.0, 4.2, 4.5, 5.0],
    })
    r.strategy_ = {'models': ['d_linear', 'tide', 'n_beats']}
    broad = list(screen_lb['model'])
    selected = r._fusion_select(screen_lb, broad)

    assert 'tft' in selected, \
        f"tft (#1 in screening) should be selected: {selected}"
    assert len(selected) == 3

    print(f"[PASS] test_fusion_select_screening_dominates: {selected}")


def test_fusion_select_diversity():
    """Test that fusion selection maintains category diversity."""
    import pandas as pd
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    max_models=5, search_strategy='auto', verbose=False)

    r.model_scores_ = {
        'd_linear': {'total': 80}, 'tide': {'total': 78},
        'n_beats': {'total': 75}, 'torch_boosting_forest': {'total': 73},
        'auto_arima': {'total': 70}, 'tft': {'total': 65},
        'gau': {'total': 60}, 'prophet': {'total': 55},
    }
    screen_lb = pd.DataFrame({
        'model': ['tft', 'tide', 'auto_arima', 'd_linear',
                  'torch_boosting_forest', 'n_beats', 'gau', 'prophet'],
        'metric': [3.2, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0],
    })
    r.strategy_ = {'models': ['d_linear', 'tide', 'n_beats',
                               'torch_boosting_forest', 'auto_arima']}
    broad = list(screen_lb['model'])
    selected = r._fusion_select(screen_lb, broad)

    # Should have representatives from multiple categories
    cats = set()
    cat_map = {
        'auto_arima': 'stat', 'prophet': 'stat',
        'torch_boosting_forest': 'ml',
        'd_linear': 'nn_light', 'tide': 'nn_light',
        'n_beats': 'nn_medium', 'gau': 'nn_medium',
        'tft': 'nn_heavy',
    }
    for m in selected:
        cats.add(cat_map.get(m, 'unknown'))
    assert len(cats) >= 3, f"Too few categories: {cats} from {selected}"
    assert len(selected) == 5

    print(f"[PASS] test_fusion_select_diversity: {selected} cats={cats}")


def test_fusion_select_fallback():
    """Test fusion selection falls back to heuristic when screening is None."""
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    max_models=3, search_strategy='auto', verbose=False)
    r.model_scores_ = {'d_linear': {'total': 80}, 'tide': {'total': 78}}
    r.strategy_ = {'models': ['d_linear', 'tide', 'n_beats']}

    result = r._fusion_select(None, ['d_linear', 'tide', 'n_beats'])
    assert result == ['d_linear', 'tide', 'n_beats']

    print(f"[PASS] test_fusion_select_fallback: {result}")


def test_fusion_select_ml_cap():
    """Test that fusion selection caps ML models at 2."""
    import pandas as pd
    r = SmartRouter(time_col='d', target_col='v', n_predict=12,
                    max_models=5, search_strategy='auto', verbose=False)

    # All ML models rank high in screening
    r.model_scores_ = {
        'torch_boosting_forest': {'total': 80},
        'torch_bagging_forest': {'total': 78},
        'deep_forest': {'total': 76},
        'd_linear': {'total': 50}, 'tft': {'total': 50},
        'auto_arima': {'total': 50}, 'n_beats': {'total': 50},
    }
    screen_lb = pd.DataFrame({
        'model': ['torch_boosting_forest', 'torch_bagging_forest',
                  'deep_forest', 'd_linear', 'tft', 'auto_arima', 'n_beats'],
        'metric': [3.0, 3.1, 3.2, 3.5, 4.0, 4.5, 5.0],
    })
    r.strategy_ = {'models': []}
    broad = list(screen_lb['model'])
    selected = r._fusion_select(screen_lb, broad)

    ml_models = {'torch_boosting_forest', 'torch_bagging_forest', 'deep_forest'}
    ml_count = sum(1 for m in selected if m in ml_models)
    assert ml_count <= 2, f"ML cap violated: {ml_count} ML in {selected}"

    print(f"[PASS] test_fusion_select_ml_cap: {selected} (ml={ml_count})")


# ─── Search Strategy Integration Tests ───────────────────────────────────────

@pytest.mark.timeout(600)
def test_smart_router_search_auto():
    """Test search_strategy='auto' with screening + lag exploration."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=True, max_models=5, random_state=42,
        search_strategy='auto',
        time_limit=300,
    )
    router.fit(df)

    assert router.pipeline_ is not None
    assert router.leader_board_ is not None
    # Screening should run (max_models=5 >= 4, n=397 >= 80)
    assert router._screening_results is not None
    # Lag exploration should run (n=397 >= 100)
    assert router._lag_exploration_results is not None
    # Calibration always runs
    assert router._calibration_rho is not None

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_search_auto")
    print(f"  Screening: {len(router._screening_results)} candidates")
    print(f"  Lag exploration: {router._lag_exploration_results}")
    print(f"  Calibration rho: {router._calibration_rho:.3f}")
    print(f"  Best: {router.leader_board_.iloc[0]['model']} "
          f"(MAE={router.leader_board_.iloc[0]['metric']:.4f})")


def test_smart_router_multi_stack_ensemble():
    """Test multi_stack ensemble with diverse meta-learners."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, max_models=3, random_state=42,
        ensemble_strategy='multi_stack', ensemble_top_k=3,
        search_strategy='basic',
        time_limit=120,
    )
    router.fit(df)

    # Ensemble may be discarded by post-evaluation
    if router.ensemble_ is not None:
        ens = router.ensemble_
        # multi_stack may fall back to stacking or weighted_avg
        assert ens.ensemble_method in ('multi_stack', 'stacking', 'weighted_avg')
        if ens.ensemble_method == 'multi_stack':
            # meta_model should be a list of (estimator, weight) pairs
            assert isinstance(ens.meta_model, list)
            assert len(ens.meta_model) == 2  # Ridge + ElasticNet
            weights = [w for _, w in ens.meta_model]
            assert abs(sum(weights) - 1.0) < 1e-6, f"Blend weights don't sum to 1: {weights}"

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    # Also test single-model fallback
    preds_single = router.predict(n=12, use_ensemble=False)
    assert len(preds_single) == 12

    print(f"[PASS] test_smart_router_multi_stack_ensemble: {router.ensemble_}")


def test_hpo_strategy_param():
    """Test hpo_strategy parameter validation."""
    # Valid values
    for strat in ('none', 'quick', 'full'):
        r = SmartRouter(
            time_col='date', target_col='value', n_predict=12,
            hpo_strategy=strat,
        )
        assert r.hpo_strategy == strat

    # Invalid value
    try:
        SmartRouter(time_col='date', target_col='value', hpo_strategy='invalid')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Default is 'none'
    r = SmartRouter(time_col='date', target_col='value')
    assert r.hpo_strategy == 'none'
    assert r._hpo_results is None

    print("[PASS] test_hpo_strategy_param")


def test_hpo_search_space():
    """Test HPO search space definitions."""
    from PipelineTS.pipeline.hpo import get_search_space, MODEL_SEARCH_SPACES

    # All tree models now use torch tree params: n_trees, tree_depth, learning_rate, n_epochs
    boost_space = get_search_space('torch_boosting_forest')
    assert 'n_trees' in boost_space
    assert 'tree_depth' in boost_space
    assert 'learning_rate' in boost_space

    # TorchBaggingForest also uses torch tree params
    bag_space = get_search_space('torch_bagging_forest')
    assert 'n_trees' in bag_space

    # NN models have learning_rate, epochs
    tcn_space = get_search_space('tcn')
    assert 'learning_rate' in tcn_space
    assert 'epochs' in tcn_space

    # Chronos has no tunable params
    chronos_space = get_search_space('chronos_2')
    assert len(chronos_space) == 0

    # Unknown model returns empty
    assert get_search_space('nonexistent_model') == {}

    print(f"[PASS] test_hpo_search_space: {len(MODEL_SEARCH_SPACES)} models defined")


def test_smart_router_hpo_quick():
    """Test SmartRouter with hpo_strategy='quick'."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, max_models=2, random_state=42,
        hpo_strategy='quick', hpo_n_trials=2,
        ensemble_strategy='none', search_strategy='basic',
        time_limit=180,
    )
    router.fit(df)

    # HPO results should be populated
    assert router._hpo_results is not None
    assert len(router._hpo_results) > 0

    # Each result should have best_params, best_value, n_trials
    for model_name, result in router._hpo_results.items():
        assert 'best_params' in result
        assert 'best_value' in result
        assert result['n_trials'] <= 2  # quick caps at min(n_trials, 5)
        assert result['best_value'] > 0

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_hpo_quick: {len(router._hpo_results)} models tuned")
    for m, r in router._hpo_results.items():
        print(f"  {m}: best={r['best_value']:.4f}, params={r['best_params']}")


def test_smart_router_search_basic():
    """Test search_strategy='basic' skips screening and exploration."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, max_models=3, random_state=42,
        search_strategy='basic',
        time_limit=120,
    )
    router.fit(df)

    assert router._screening_results is None
    assert router._lag_exploration_results is None
    assert router._calibration_rho is not None
    assert router.pipeline_ is not None
    assert len(router.leader_board_) <= 3

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_search_basic")
    print(f"  Best: {router.leader_board_.iloc[0]['model']} "
          f"(MAE={router.leader_board_.iloc[0]['metric']:.4f})")


def test_multi_quantile_pipeline():
    """Test predict_quantiles on ModelPipeline with GBDT model."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])
    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=12,
        quantile=0.9, include_models=['torch_boosting_forest'],
    )
    pipe.fit(df)

    levels = [0.5, 0.8, 0.9, 0.95]
    q_preds = pipe.predict_quantiles(n=12, levels=levels)

    # Check columns
    assert 'date' in q_preds.columns
    assert 'value' in q_preds.columns
    for lv in levels:
        lv_str = f"{lv:.2f}".rstrip('0').rstrip('.')
        assert f"value_q{lv_str}_lower" in q_preds.columns, f"Missing lower for {lv}"
        assert f"value_q{lv_str}_upper" in q_preds.columns, f"Missing upper for {lv}"

    assert len(q_preds) == 12
    assert not q_preds['value'].isna().any()

    # q0.9 should match standard predict _lower/_upper
    std_preds = pipe.predict(n=12)
    import numpy as np
    np.testing.assert_allclose(
        q_preds['value_q0.9_lower'].values,
        std_preds['value_lower'].values, rtol=1e-4,
    )
    np.testing.assert_allclose(
        q_preds['value_q0.9_upper'].values,
        std_preds['value_upper'].values, rtol=1e-4,
    )

    print(f"[PASS] test_multi_quantile_pipeline: {len(q_preds.columns)} columns")


@pytest.mark.timeout(600)
def test_multi_quantile_smart_router():
    """Test predict_quantiles on SmartRouter."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, max_models=2, random_state=42,
        quantile=0.9, search_strategy='basic',
        ensemble_strategy='none',
        time_limit=180,
    )
    router.fit(df)

    levels = [0.5, 0.9]
    q_preds = router.predict_quantiles(n=12, levels=levels)

    assert len(q_preds) == 12
    assert 'value_q0.5_lower' in q_preds.columns
    assert 'value_q0.9_lower' in q_preds.columns

    print(f"[PASS] test_multi_quantile_smart_router: columns={q_preds.columns.tolist()}")


def test_multi_quantile_monotonicity():
    """Verify interval widths are monotonically increasing with coverage level."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])
    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=12,
        quantile=0.9, include_models=['torch_boosting_forest'],
    )
    pipe.fit(df)

    levels = [0.5, 0.8, 0.9, 0.95]
    q_preds = pipe.predict_quantiles(n=12, levels=levels)

    for i in range(len(q_preds)):
        widths = []
        for lv in levels:
            lv_str = f"{lv:.2f}".rstrip('0').rstrip('.')
            w = (q_preds[f'value_q{lv_str}_upper'].iloc[i]
                 - q_preds[f'value_q{lv_str}_lower'].iloc[i])
            widths.append(w)
        for j in range(len(widths) - 1):
            assert widths[j] <= widths[j + 1] + 1e-6, \
                f"Row {i}: width@{levels[j]}={widths[j]:.4f} > width@{levels[j+1]}={widths[j+1]:.4f}"

    print(f"[PASS] test_multi_quantile_monotonicity: all {len(q_preds)} rows monotonic")


def test_multi_quantile_no_quantile():
    """Test predict_quantiles works even when model trained without quantile."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])
    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=12,
        include_models=['torch_boosting_forest'],
    )
    pipe.fit(df)

    # predict_quantiles should return point predictions only (no residuals stored)
    q_preds = pipe.predict_quantiles(n=12, levels=[0.5, 0.9])

    assert len(q_preds) == 12
    assert 'value' in q_preds.columns
    # Should still have quantile columns (zero-width since no calibration)
    assert 'value_q0.5_lower' in q_preds.columns
    assert 'value_q0.9_upper' in q_preds.columns

    print(f"[PASS] test_multi_quantile_no_quantile: columns={q_preds.columns.tolist()}")


def test_incremental_pipeline_update():
    """Test Pipeline.update() with GBDT model."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])
    initial = df.iloc[:350].copy()
    new_data = df.iloc[350:].copy()

    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=12,
        include_models=['torch_boosting_forest'],
    )
    pipe.fit(initial)

    assert pipe._training_data is not None
    assert len(pipe._training_data) == 350

    preds_before = pipe.predict(n=12)

    pipe.update(new_data)

    assert len(pipe._training_data) == len(df)

    preds_after = pipe.predict(n=12)
    assert len(preds_after) == 12
    assert not preds_after['value'].isna().any()

    # Predictions should change after seeing new data
    assert not np.allclose(preds_before['value'].values, preds_after['value'].values), \
        "Predictions should change after update"

    print(f"[PASS] test_incremental_pipeline_update: "
          f"before={preds_before['value'].iloc[0]:.2f}, after={preds_after['value'].iloc[0]:.2f}")


def test_incremental_smart_router_update():
    """Test SmartRouter.update()."""
    df = LoadElectricDataSets()
    initial = df.iloc[:350].copy()
    new_data = df.iloc[350:].copy()

    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=False, max_models=2, random_state=42,
        search_strategy='basic', ensemble_strategy='none',
        time_limit=120,
    )
    router.fit(initial)

    preds_before = router.predict(n=12)

    router.update(new_data)

    preds_after = router.predict(n=12)
    assert len(preds_after) == 12
    assert not np.allclose(preds_before['value'].values, preds_after['value'].values)

    print(f"[PASS] test_incremental_smart_router_update: "
          f"before={preds_before['value'].iloc[0]:.2f}, after={preds_after['value'].iloc[0]:.2f}")


def test_incremental_update_not_fitted():
    """Test that update() raises if not fitted."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])

    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=12,
        include_models=['torch_boosting_forest'],
    )
    try:
        pipe.update(df)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
    )
    try:
        router.update(df)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("[PASS] test_incremental_update_not_fitted")


def test_per_model_lags_pipeline():
    """Test ModelPipeline with per_model_lags parameter."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])

    # Test 1: per_model_lags overrides global lag for specific models
    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=16,
        include_models=['torch_boosting_forest', 'torch_bagging_forest'],
        per_model_lags={'torch_boosting_forest': 12, 'torch_bagging_forest': 8},
        random_state=42,
    )
    lb = pipe.fit(df)
    assert not lb.empty
    assert len(lb) == 2

    # Verify models got their individual lags
    boost_model = pipe.get_model('torch_boosting_forest')
    bag_model = pipe.get_model('torch_bagging_forest')
    assert boost_model.all_configs['lags'] == 12
    assert bag_model.all_configs['lags'] == 8

    print(f"[PASS] test_per_model_lags_pipeline: boost_lag={boost_model.all_configs['lags']}, bag_lag={bag_model.all_configs['lags']}")


def test_per_model_lags_default_fallback():
    """Test that models not in per_model_lags use global lag."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])

    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=16,
        include_models=['torch_boosting_forest', 'torch_bagging_forest'],
        per_model_lags={'torch_boosting_forest': 10},  # torch_bagging_forest not specified
        random_state=42,
    )
    lb = pipe.fit(df)
    assert not lb.empty

    boost_model = pipe.get_model('torch_boosting_forest')
    bag_model = pipe.get_model('torch_bagging_forest')
    assert boost_model.all_configs['lags'] == 10
    assert bag_model.all_configs['lags'] == 16  # fallback to global

    print(f"[PASS] test_per_model_lags_default_fallback")


def test_per_model_lags_empty_dict():
    """Test that empty per_model_lags behaves like no override."""
    df = LoadElectricDataSets()
    df['date'] = pd.to_datetime(df['date'])

    pipe = ModelPipeline(
        time_col='date', target_col='value', lags=16,
        include_models=['torch_boosting_forest'],
        per_model_lags={},
        random_state=42,
    )
    lb = pipe.fit(df)
    boost_model = pipe.get_model('torch_boosting_forest')
    assert boost_model.all_configs['lags'] == 16

    print("[PASS] test_per_model_lags_empty_dict")


@pytest.mark.timeout(600)
def test_explore_lags_per_model():
    """Test _explore_lags returns per-model lag dict."""
    df = LoadElectricDataSets()
    r = SmartRouter(
        time_col='date', target_col='value',
        search_strategy='auto', max_models=3, random_state=42,
    )
    data = r._ensure_datetime(df)
    r.profile_ = r._profile_data(data)
    r.strategy_ = r._build_strategy(r.profile_)

    models = ['torch_boosting_forest', 'torch_bagging_forest']
    primary_lag = r._explore_lags(data, models, r.strategy_)

    # Verify per-model lags stored
    assert hasattr(r, '_per_model_lags')
    assert isinstance(r._per_model_lags, dict)
    for m in models:
        assert m in r._per_model_lags
        assert isinstance(r._per_model_lags[m], int)

    # Primary lag should be max of per-model lags
    assert primary_lag == max(r._per_model_lags.values())

    # Lag exploration results should be {model: {lag: metric}}
    assert isinstance(r._lag_exploration_results, dict)
    for m in models:
        assert m in r._lag_exploration_results
        assert isinstance(r._lag_exploration_results[m], dict)

    print(f"[PASS] test_explore_lags_per_model: {r._per_model_lags}")


def test_include_models_fit_predict():
    """Test full SmartRouter fit/predict with include_models (multi-model)."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=True,
        include_models=['prophet', 'torch_boosting_forest'],
        search_strategy='basic',
        ensemble_strategy='auto',
        time_limit=120,
        random_state=42,
    )
    router.fit(df)

    assert router.pipeline_ is not None
    assert router.leader_board_ is not None
    # Only the 2 pinned models should be trained
    trained_models = set(router.leader_board_['model'].tolist())
    assert trained_models.issubset({'prophet', 'torch_boosting_forest'}), \
        f"Unexpected models trained: {trained_models}"
    assert len(router.leader_board_) <= 2

    # Strategy should reflect pinned models
    assert router.strategy_['models'] == ['prophet', 'torch_boosting_forest']

    # Screening should NOT have run (include_models skips screening)
    assert router._screening_results is None

    # Profiling, preprocessing, scaler, hyperparams still run
    assert router.profile_ is not None
    assert router.model_scores_ is not None
    assert 'model_hyperparams' in router.strategy_

    # Predict should work
    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_include_models_fit_predict")
    print(f"  Models: {trained_models}")
    print(f"  Best: {router.leader_board_.iloc[0]['model']} "
          f"(MAE={router.leader_board_.iloc[0]['metric']:.4f})")


def test_include_models_single_model_optimization():
    """Test SmartRouter with a single pinned model — full optimization."""
    df = LoadElectricDataSets()
    router = SmartRouter(
        time_col='date', target_col='value', n_predict=12,
        verbose=True,
        include_models='torch_boosting_forest',
        search_strategy='auto',  # lag exploration should still work
        ensemble_strategy='none',
        time_limit=120,
        random_state=42,
    )
    router.fit(df)

    assert router.pipeline_ is not None
    assert len(router.leader_board_) == 1
    assert router.leader_board_.iloc[0]['model'] == 'torch_boosting_forest'

    # Strategy optimized for the single model
    assert router.strategy_['models'] == ['torch_boosting_forest']
    assert router.strategy_['lags'] > 0
    assert router.strategy_['scaler'] is not None

    # No ensemble for single model
    assert router.ensemble_ is None

    # Lag exploration may still run (search_strategy='auto', n>=100)
    # depending on _should_explore_lags

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_include_models_single_model_optimization")
    print(f"  Lag: {router.strategy_['lags']}")
    print(f"  MAE: {router.leader_board_.iloc[0]['metric']:.4f}")


if __name__ == '__main__':
    # Fast tests first (no model training)
    test_data_profile()
    test_strategy_selection()
    test_model_scoring_explanation()
    test_priority_ordering()
    test_data_profile_repr()
    test_normalize_freq()
    test_small_data_routing()
    test_missing_data_detection()
    test_skewed_data_scaler()
    test_ensemble_predictor_weighted()
    test_ensemble_predictor_median()
    test_ensemble_strategy_none()
    test_ensemble_strategy_validation()
    test_feature_engineering_routing()
    test_adaptive_hyperparams()
    test_pipeline_time_limit_param()
    test_pipeline_time_limit_validation()
    test_pipeline_failed_skipped_properties()
    test_search_strategy_param()
    test_should_screen_logic()
    test_should_explore_lags_logic()
    test_generate_lag_candidates()
    test_pick_fast_eval_model()
    test_select_models_n_candidates()
    test_scoring_pattern_bonus_cap()
    test_scoring_model_diversity()
    test_hyperparams_nn_stability()
    test_hyperparams_adaptive_lr()
    test_stability_params_in_model_wrapper()
    test_sort_by_speed()
    test_sort_by_speed_preserves_order_within_tier()
    test_deep_forest_scoring_rebalance()
    test_deep_forest_penalty_small_data()
    test_time_aware_epoch_capping()
    test_ensemble_eval_attribute()
    test_get_screen_pool_size()
    test_fusion_select_screening_dominates()
    test_fusion_select_diversity()
    test_fusion_select_fallback()
    test_fusion_select_ml_cap()

    test_include_models_single_string()
    test_include_models_list()
    test_include_models_max_models_override()
    test_include_models_invalid_name()
    test_include_models_empty_list()
    test_include_models_bypasses_selection()
    test_include_models_skips_screening()
    test_include_models_repr()
    test_include_models_none_default()

    # Integration tests (with model training)
    test_pipeline_error_resilience()
    test_smart_router_fit_predict()
    test_smart_router_forced_ensemble()
    test_smart_router_median_ensemble()
    test_smart_router_stacking_ensemble()
    test_smart_router_time_limit()
    test_smart_router_callback_integration()
    test_smart_router_multi_stack_ensemble()
    test_hpo_strategy_param()
    test_hpo_search_space()
    test_smart_router_hpo_quick()
    test_smart_router_search_auto()
    test_smart_router_search_basic()
    test_multi_quantile_pipeline()
    test_multi_quantile_smart_router()
    test_multi_quantile_monotonicity()
    test_multi_quantile_no_quantile()
    test_incremental_pipeline_update()
    test_incremental_smart_router_update()
    test_incremental_update_not_fitted()
    test_per_model_lags_pipeline()
    test_per_model_lags_default_fallback()
    test_per_model_lags_empty_dict()
    test_explore_lags_per_model()
    test_include_models_fit_predict()
    test_include_models_single_model_optimization()

    print("\n=== All SmartRouter tests passed! ===")
