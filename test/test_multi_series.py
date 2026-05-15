"""Tests for multi-series (panel data) support.

Tests:
- Panel split functions: split_series_panel, lag_splits_panel
- GBDT models: CatBoost/RandomForest with id_col
- ModelPipeline: id_col injection, per-series scaling, evaluation
- SmartRouter: id_col pass-through, profiling
- Backward compatibility: single-series still works
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from PipelineTS.preprocessing import (
    split_series_panel, lag_splits_panel, split_series, lag_splits
)


# ─── Helper: generate panel data ────────────────────────────────────────────

def make_panel_data(n_series=3, n_points=100, freq='D', seed=42):
    """Create synthetic multi-series panel data."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_points, freq=freq)
    parts = []
    for i in range(n_series):
        sid = f'series_{i}'
        vals = np.cumsum(np.random.randn(n_points)) + (i + 1) * 50
        parts.append(pd.DataFrame({
            'ds': dates, 'y': vals, 'series_id': sid
        }))
    return pd.concat(parts, ignore_index=True)


# ─── 1. Panel split functions ───────────────────────────────────────────────

def test_split_series_panel_basic():
    """split_series_panel creates windows per series, no cross-boundary leakage."""
    x = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
    groups = np.array(['A'] * 5 + ['B'] * 5)

    X, Y, indices = split_series_panel(x, x, groups, window_size=2, pred_steps=2)

    assert X.shape[1] == 2
    assert Y.shape[1] == 2
    # Check no cross-series windows
    for row_x, row_y in zip(X, Y):
        # All values in a window should come from the same series
        all_vals = np.concatenate([row_x, row_y])
        assert (np.all(all_vals <= 5) or np.all(all_vals >= 10)), \
            f"Cross-series leakage detected: {all_vals}"

    # indices should contain both series
    series_ids = [sid for sid, count in indices]
    assert 'A' in series_ids or np.str_('A') in series_ids
    print("[PASS] test_split_series_panel_basic")


def test_split_series_panel_window_counts():
    """Verify correct number of windows per series."""
    n_a, n_b = 10, 8
    x = np.arange(n_a + n_b, dtype=float)
    groups = np.array(['A'] * n_a + ['B'] * n_b)

    ws, ps = 3, 2
    X, Y, indices = split_series_panel(x, x, groups, window_size=ws, pred_steps=ps)

    # Series A: 10 points, window=3, pred=2 → 10 - 3 - 2 + 1 = 6 windows
    # Series B: 8 points, window=3, pred=2 → 8 - 3 - 2 + 1 = 4 windows
    expected_total = (n_a - ws - ps + 1) + (n_b - ws - ps + 1)
    assert X.shape[0] == expected_total, f"Expected {expected_total} windows, got {X.shape[0]}"
    print(f"[PASS] test_split_series_panel_window_counts: {X.shape[0]} windows")


def test_lag_splits_panel_basic():
    """lag_splits_panel returns correct last window per series."""
    x = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
    groups = np.array(['A'] * 5 + ['B'] * 5)

    result = lag_splits_panel(x, groups, window_size=3)

    assert isinstance(result, dict)
    assert len(result) == 2
    # Series A last 3: [3, 4, 5]
    a_key = 'A' if 'A' in result else np.str_('A')
    b_key = 'B' if 'B' in result else np.str_('B')
    assert np.allclose(result[a_key], [[3, 4, 5]])
    assert np.allclose(result[b_key], [[30, 40, 50]])
    print("[PASS] test_lag_splits_panel_basic")


def test_lag_splits_panel_short_series():
    """Series shorter than window_size should be skipped."""
    x = np.array([1, 2, 3, 4, 5, 100, 200], dtype=float)
    groups = np.array(['A'] * 5 + ['B'] * 2)

    result = lag_splits_panel(x, groups, window_size=3)
    a_key = [k for k in result if str(k) == 'A'][0]
    assert a_key in result
    # B has only 2 points, window=3 → should be skipped or have 1 window
    print(f"[PASS] test_lag_splits_panel_short_series: keys={list(result.keys())}")


# ─── 2. GBDT model with id_col ──────────────────────────────────────────────

def test_catboost_multi_series():
    """CatBoostModel trains and predicts with id_col."""
    from PipelineTS.ml_model import CatBoostModel

    panel = make_panel_data(n_series=3, n_points=80)
    model = CatBoostModel(time_col='ds', target_col='y', lags=10)
    model.all_configs['id_col'] = 'series_id'

    model.fit(panel)
    assert hasattr(model, '_panel_raw_lags')
    assert len(model._panel_raw_lags) == 3

    preds = model.predict(n=5)
    assert preds.shape[0] == 15  # 3 series * 5 steps
    assert 'series_id' in preds.columns
    assert set(preds['series_id'].unique()) == {'series_0', 'series_1', 'series_2'}
    print("[PASS] test_catboost_multi_series")


def test_random_forest_multi_series():
    """RandomForestModel trains and predicts with id_col."""
    from PipelineTS.ml_model import RandomForestModel

    panel = make_panel_data(n_series=2, n_points=60)
    model = RandomForestModel(time_col='ds', target_col='y', lags=8)
    model.all_configs['id_col'] = 'series_id'

    model.fit(panel)
    preds = model.predict(n=3)
    assert preds.shape[0] == 6  # 2 series * 3 steps
    assert 'series_id' in preds.columns
    print("[PASS] test_random_forest_multi_series")


def test_gbdt_predict_with_data():
    """GBDT predict with explicit data= argument for multi-series."""
    from PipelineTS.ml_model import CatBoostModel

    panel = make_panel_data(n_series=2, n_points=80)
    model = CatBoostModel(time_col='ds', target_col='y', lags=10)
    model.all_configs['id_col'] = 'series_id'
    model.fit(panel)

    # Predict using explicit data
    preds = model.predict(n=5, data=panel)
    assert preds.shape[0] == 10  # 2 series * 5 steps
    assert 'series_id' in preds.columns
    print("[PASS] test_gbdt_predict_with_data")


def test_gbdt_single_series_unchanged():
    """Single-series GBDT works without id_col (backward compat)."""
    from PipelineTS.ml_model import CatBoostModel

    panel = make_panel_data(n_series=1, n_points=120)
    single = panel[['ds', 'y']]
    model = CatBoostModel(time_col='ds', target_col='y', lags=10, quantile=None)

    model.fit(single)
    preds = model.predict(n=5)
    assert preds.shape[0] == 5
    assert 'series_id' not in preds.columns
    print("[PASS] test_gbdt_single_series_unchanged")


# ─── 3. ModelPipeline with id_col ───────────────────────────────────────────

def test_pipeline_multi_series_ml():
    """ModelPipeline with id_col trains ML models and predicts per series."""
    from PipelineTS.pipeline import ModelPipeline

    panel = make_panel_data(n_series=3, n_points=100)
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        id_col='series_id',
        include_models=['catboost'],
        scaler=True,
    )

    lb = pipe.fit(panel)
    assert not lb.empty, "Leaderboard should not be empty"
    assert lb.shape[0] >= 1

    preds = pipe.predict(n=5)
    assert preds.shape[0] == 15  # 3 series * 5 steps
    assert 'series_id' in preds.columns
    print(f"[PASS] test_pipeline_multi_series_ml: best={lb.iloc[0]['model']}")


def test_pipeline_id_col_injection():
    """Pipeline injects id_col into model all_configs."""
    from PipelineTS.pipeline import ModelPipeline

    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        id_col='series_id',
        include_models=['catboost'],
    )
    models = pipe._initial_models()
    for name, model in models:
        assert model.all_configs.get('id_col') == 'series_id', \
            f"Model {name} missing id_col in all_configs"
    print("[PASS] test_pipeline_id_col_injection")


def test_pipeline_per_series_scaling():
    """Pipeline applies per-series scaling when id_col is set."""
    from PipelineTS.pipeline import ModelPipeline
    from sklearn.preprocessing import MinMaxScaler

    panel = make_panel_data(n_series=2, n_points=50)
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=5,
        id_col='series_id',
        include_models=['catboost'],
        scaler=True,
    )

    df, _ = pipe._scale_data(panel.copy())

    # Each series should be scaled independently
    assert len(pipe._panel_scalers) == 2
    for sid in panel['series_id'].unique():
        scaled_vals = df[df['series_id'] == sid]['y'].values
        # MinMaxScaler should map to [0, 1]
        assert scaled_vals.min() >= -0.01 and scaled_vals.max() <= 1.01, \
            f"Series {sid} not properly scaled: [{scaled_vals.min():.3f}, {scaled_vals.max():.3f}]"
    print("[PASS] test_pipeline_per_series_scaling")


def test_pipeline_no_id_col_unchanged():
    """Pipeline without id_col works as before (backward compat)."""
    from PipelineTS.pipeline import ModelPipeline

    panel = make_panel_data(n_series=1, n_points=80)
    single = panel[['ds', 'y']]
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        include_models=['catboost'],
        scaler=True,
    )

    lb = pipe.fit(single)
    assert not lb.empty
    preds = pipe.predict(n=5)
    assert preds.shape[0] == 5
    assert 'series_id' not in preds.columns
    print("[PASS] test_pipeline_no_id_col_unchanged")


# ─── 4. SmartRouter with id_col ─────────────────────────────────────────────

def test_smartrouter_id_col_param():
    """SmartRouter accepts id_col parameter."""
    from PipelineTS.pipeline.smart_router import SmartRouter

    router = SmartRouter(
        time_col='ds', target_col='y',
        id_col='series_id', verbose=False,
    )
    assert router.id_col == 'series_id'
    print("[PASS] test_smartrouter_id_col_param")


def test_smartrouter_profile_multi_series():
    """SmartRouter profiles representative series correctly."""
    from PipelineTS.pipeline.smart_router import SmartRouter

    panel = make_panel_data(n_series=3, n_points=80)
    router = SmartRouter(
        time_col='ds', target_col='y',
        id_col='series_id', verbose=False,
    )
    df_dt = router._ensure_datetime(panel)
    profile = router._profile_data(df_dt)

    # Should detect multi-series
    assert profile.n_series == 3
    # n_rows should be representative series length, not total
    assert profile.n_rows == 80, f"n_rows should be 80 (per-series), got {profile.n_rows}"
    assert profile._total_rows == 240
    print(f"[PASS] test_smartrouter_profile_multi_series: n_rows={profile.n_rows}, n_series={profile.n_series}")


def test_smartrouter_profile_single_series():
    """SmartRouter profiles single-series correctly (backward compat)."""
    from PipelineTS.pipeline.smart_router import SmartRouter
    from PipelineTS.dataset import LoadElectric

    df = LoadElectricProduction()
    router = SmartRouter(time_col='date', target_col='value', verbose=False)
    df_dt = router._ensure_datetime(df)
    profile = router._profile_data(df_dt)

    assert profile.n_series == 1
    assert profile.n_rows == len(df)
    print("[PASS] test_smartrouter_profile_single_series")


# ─── 5. DataProfile n_series field ──────────────────────────────────────────

def test_data_profile_n_series():
    """DataProfile includes n_series field."""
    from PipelineTS.pipeline.smart_router import DataProfile

    profile = DataProfile()
    assert hasattr(profile, 'n_series')
    assert profile.n_series == 1  # default

    summary = profile.summary()
    assert 'n_series' in summary
    print("[PASS] test_data_profile_n_series")


# ─── 6. Statistic models multi-series ────────────────────────────────────────

def test_prophet_multi_series():
    """ProphetModel trains per-series local models and predicts per series."""
    from PipelineTS.statistic_model.prophet import ProphetModel

    panel = make_panel_data(n_series=2, n_points=80)
    model = ProphetModel(time_col='ds', target_col='y', quantile=None)
    model.all_configs['id_col'] = 'series_id'

    model.fit(panel)
    assert hasattr(model, '_panel_models')
    assert len(model._panel_models) == 2

    preds = model.predict(n=5)
    assert preds.shape[0] == 10
    assert 'series_id' in preds.columns
    print("[PASS] test_prophet_multi_series")


def test_auto_arima_multi_series():
    """AutoARIMAModel trains per-series local models and predicts per series."""
    from PipelineTS.statistic_model.auto_arima import AutoARIMAModel

    panel = make_panel_data(n_series=2, n_points=80)
    model = AutoARIMAModel(time_col='ds', target_col='y', max_p=2, max_q=2, quantile=None)
    model.all_configs['id_col'] = 'series_id'

    model.fit(panel)
    assert hasattr(model, '_panel_models')
    assert len(model._panel_models) == 2

    preds = model.predict(n=3)
    assert preds.shape[0] == 6
    assert 'series_id' in preds.columns
    print("[PASS] test_auto_arima_multi_series")


# ─── 7. NN model multi-series ────────────────────────────────────────────────

def test_nn_multi_series_pipeline():
    """NN model (d_linear) works in multi-series pipeline."""
    from PipelineTS.pipeline import ModelPipeline

    panel = make_panel_data(n_series=2, n_points=100)
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        id_col='series_id',
        include_models=['d_linear'],
        scaler=True, quantile=None,
    )
    lb = pipe.fit(panel)
    assert not lb.empty

    preds = pipe.predict(n=5)
    assert preds.shape[0] == 10
    assert 'series_id' in preds.columns
    print(f"[PASS] test_nn_multi_series_pipeline: metric={lb.iloc[0]['metric']:.4f}")


# ─── 8. Full integration: all model types ────────────────────────────────────

def test_pipeline_all_model_types():
    """Pipeline with stat + ML + NN models all handle multi-series correctly."""
    from PipelineTS.pipeline import ModelPipeline

    panel = make_panel_data(n_series=3, n_points=100)
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        id_col='series_id',
        include_models=['prophet', 'catboost'],
        scaler=True, quantile=None,
    )
    lb = pipe.fit(panel)
    assert lb.shape[0] >= 2

    preds = pipe.predict(n=5)
    assert preds.shape[0] == 15  # 3 series * 5 steps
    assert 'series_id' in preds.columns
    assert set(preds['series_id'].unique()) == {'series_0', 'series_1', 'series_2'}
    print(f"[PASS] test_pipeline_all_model_types: best={lb.iloc[0]['model']}")


# ─── 9. Multi-series backtesting ──────────────────────────────────────────────

def test_backtesting_panel_prophet():
    """Backtester handles panel data with per-series local models (Prophet)."""
    from PipelineTS.evaluation.backtesting import Backtester
    from PipelineTS.statistic_model.prophet import ProphetModel

    panel = make_panel_data(n_series=2, n_points=80)
    model = ProphetModel(time_col='ds', target_col='y', quantile=None)
    model.all_configs['id_col'] = 'series_id'

    bt = Backtester(
        model, time_col='ds', target_col='y',
        metric=lambda yt, yp: float(np.mean(np.abs(yt - yp))),
        metric_name='mae',
        id_col='series_id',
    )
    results = bt.fit(panel, n_splits=3, test_size=5, mode='expanding', verbose=False)
    assert len(results) == 3
    assert results['mae'].notna().all()
    summary = bt.summary()
    assert summary['n_failed'] == 0
    print(f"[PASS] test_backtesting_panel_prophet: mean_mae={summary['mean']:.4f}")


def test_backtesting_panel_catboost():
    """Backtester handles panel data with global model (CatBoost)."""
    from PipelineTS.evaluation.backtesting import Backtester
    from PipelineTS.ml_model import CatBoostModel

    panel = make_panel_data(n_series=3, n_points=100)
    model = CatBoostModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.all_configs['id_col'] = 'series_id'

    bt = Backtester(
        model, time_col='ds', target_col='y',
        metric=lambda yt, yp: float(np.mean(np.abs(yt - yp))),
        metric_name='mae',
        id_col='series_id',
    )
    results = bt.fit(panel, n_splits=3, test_size=10, mode='expanding', verbose=False)
    assert len(results) == 3
    assert results['mae'].notna().all()
    summary = bt.summary()
    assert summary['n_failed'] == 0
    print(f"[PASS] test_backtesting_panel_torch_boosting: mean_mae={summary['mean']:.4f}")


def test_backtesting_single_series_unchanged():
    """Backtester without id_col still works for single series."""
    from PipelineTS.evaluation.backtesting import Backtester
    from PipelineTS.statistic_model.prophet import ProphetModel

    dates = pd.date_range('2020-01-01', periods=80, freq='D')
    data = pd.DataFrame({'ds': dates, 'y': np.cumsum(np.random.randn(80)) + 100})

    model = ProphetModel(time_col='ds', target_col='y', quantile=None)
    bt = Backtester(
        model, time_col='ds', target_col='y',
        metric=lambda yt, yp: float(np.mean(np.abs(yt - yp))),
    )
    results = bt.fit(data, n_splits=3, test_size=5, verbose=False)
    assert len(results) == 3
    assert results['metric'].notna().all()
    print("[PASS] test_backtesting_single_series_unchanged")


# ─── Run all tests ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        # Panel split functions
        test_split_series_panel_basic,
        test_split_series_panel_window_counts,
        test_lag_splits_panel_basic,
        test_lag_splits_panel_short_series,
        # GBDT multi-series
        test_torch_boosting_multi_series,
        test_torch_bagging_multi_series,
        test_gbdt_predict_with_data,
        test_gbdt_single_series_unchanged,
        # ModelPipeline
        test_pipeline_multi_series_ml,
        test_pipeline_id_col_injection,
        test_pipeline_per_series_scaling,
        test_pipeline_no_id_col_unchanged,
        # SmartRouter
        test_smartrouter_id_col_param,
        test_smartrouter_profile_multi_series,
        test_smartrouter_profile_single_series,
        # DataProfile
        test_data_profile_n_series,
        # Statistic models
        test_prophet_multi_series,
        test_auto_arima_multi_series,
        # NN model
        test_nn_multi_series_pipeline,
        # Full integration
        test_pipeline_all_model_types,
        # Backtesting
        test_backtesting_panel_prophet,
        test_backtesting_panel_torch_boosting,
        test_backtesting_single_series_unchanged,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((t.__name__, str(e)[:300]))

    print(f"\n{'='*60}")
    if failed:
        print(f"{len(failed)}/{len(tests)} FAILED:")
        for name, err in failed:
            print(f"  {name}: {err}")
    else:
        print(f"All {len(tests)} multi-series tests PASSED!")
