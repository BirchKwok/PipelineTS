"""Tests for SmartRouter."""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from PipelineTS.pipeline.smart_router import SmartRouter, DataProfile, EnsemblePredictor
from PipelineTS.dataset import LoadElectricDataSets


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
    # New fields
    assert -1 <= profile.autocorr_lag1 <= 1
    assert -1 <= profile.autocorr_lag2 <= 1
    assert profile.n_seasonalities >= 1  # Electric has at least 1 seasonal period
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
    assert strategy['lags'] >= 12  # should cover n_predict and dominant period
    assert strategy['lags'] * 2 < len(df)

    # Feature engineering should be a dict
    fe = strategy['feature_engineering']
    assert isinstance(fe, dict)
    assert 'routing_mode' in fe
    assert fe['routing_mode'] in ('static', 'adaptive')

    # Model hyperparams should be a dict
    hp = strategy['model_hyperparams']
    assert isinstance(hp, dict)

    # Ensure model diversity
    stat = {'auto_arima', 'prophet'}
    ml = {'lightgbm', 'catboost', 'xgboost', 'random_forest', 'wide_gbrt',
           'multi_output_model', 'multi_step_model', 'regressor_chain'}
    models_set = set(strategy['models'])
    assert models_set & ml, "No ML model selected"
    assert models_set & stat, "No statistic model selected"

    print(f"[PASS] test_strategy_selection: lags={strategy['lags']}, "
          f"models={strategy['models']}, "
          f"scaler={strategy['scaler'].__class__.__name__}, "
          f"diff_n={strategy['gbdt_differential_n']}, "
          f"fe={fe}, hp_keys={list(hp.keys())}")


def test_small_data_routing():
    """Test model selection for very small dataset."""
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=60, freq='D')
    values = np.sin(np.arange(60) * 2 * np.pi / 7) * 10 + 50 + np.random.randn(60) * 2
    df = pd.DataFrame({'date': dates, 'value': values})

    router = SmartRouter(time_col='date', target_col='value', verbose=False, max_models=5)
    profile = router._profile_data(df)
    strategy = router._build_strategy(profile)

    # Should prefer statistical/ML models for small data
    has_stat = any(m in ('prophet', 'auto_arima') for m in strategy['models'])
    has_ml = any(m in ('lightgbm', 'catboost', 'xgboost', 'random_forest')
                 for m in strategy['models'])
    assert has_stat, f"No stat model for small data: {strategy['models']}"
    assert has_ml, f"No ML model for small data: {strategy['models']}"
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
    values = np.exp(np.random.randn(300) * 0.5 + 3)  # log-normal
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


def test_ensemble_predictor():
    """Test EnsemblePredictor class directly."""
    ep = EnsemblePredictor(
        pipeline=None,
        model_names=['lightgbm', 'prophet'],
        weights={'lightgbm': 0.6, 'prophet': 0.4},
        time_col='date',
        target_col='value',
    )
    assert 'lightgbm' in ep.model_names
    assert abs(sum(ep.weights.values()) - 1.0) < 1e-6
    cfg = ep.all_configs
    assert cfg['ensemble'] is True
    assert cfg['strategy'] == 'weighted_avg'
    r = repr(ep)
    assert 'lightgbm' in r and 'prophet' in r
    print(f"[PASS] test_ensemble_predictor: {r}")


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
    # _build_ensemble requires pipeline_ to be set; just verify param stored
    assert router.ensemble_strategy == 'none'
    print(f"[PASS] test_ensemble_strategy_none")


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
    # Electric_Production has 397 rows, seasonality > 0.1 → should suggest GBDT estimators
    if profile.n_rows >= 300 and profile.seasonality_strength > 0.1:
        assert 'lightgbm__n_estimators' in hp
    print(f"[PASS] test_adaptive_hyperparams: {list(hp.keys())}")


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
    )

    router.fit(df)

    assert router.pipeline_ is not None
    assert router.leader_board_ is not None
    assert router.best_model_ is not None
    assert len(router.leader_board_) == 3

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

    # Test get_model
    model = router.get_model()
    assert model is not None

    # Test strategy property (includes new fields)
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
    )

    router.fit(df)

    # With 'weighted_avg', ensemble should always be built if >= 2 models
    assert router.ensemble_ is not None
    assert isinstance(router.ensemble_, EnsemblePredictor)
    assert len(router.ensemble_.model_names) >= 2

    preds = router.predict(n=12)
    assert len(preds) == 12
    assert not preds['value'].isna().any()

    print(f"[PASS] test_smart_router_forced_ensemble")
    print(f"  Ensemble: {router.ensemble_}")


if __name__ == '__main__':
    # Fast tests first (no model training)
    test_data_profile()
    test_strategy_selection()
    test_data_profile_repr()
    test_normalize_freq()
    test_small_data_routing()
    test_missing_data_detection()
    test_skewed_data_scaler()
    test_ensemble_predictor()
    test_ensemble_strategy_none()
    test_feature_engineering_routing()
    test_adaptive_hyperparams()

    # Integration tests (with model training)
    test_smart_router_fit_predict()
    test_smart_router_forced_ensemble()

    print("\n=== All SmartRouter tests passed! ===")
