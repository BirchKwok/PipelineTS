"""Tests for covariate support (known/past covariates) across all model types."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _make_data(n=120, seed=42):
    """Create test data with known and past covariates."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    holiday = np.zeros(n)
    holiday[::7] = 1.0  # weekly holiday
    temperature = np.sin(np.arange(n) * 2 * np.pi / 30) * 10 + 20
    y = np.cumsum(np.random.randn(n)) + holiday * 5 + temperature * 0.3
    return pd.DataFrame({
        'ds': dates, 'y': y,
        'holiday': holiday,
        'temperature': temperature,
    })


def _make_future(n=5, holiday_on_last=True):
    """Create future covariates DataFrame."""
    holiday = np.zeros(n)
    if holiday_on_last:
        holiday[-1] = 1.0
    return pd.DataFrame({'holiday': holiday})


def _make_panel_data(n_per_series=80, n_series=3, seed=42):
    """Create multi-series panel data with covariates."""
    np.random.seed(seed)
    parts = []
    for i in range(n_series):
        dates = pd.date_range('2020-01-01', periods=n_per_series, freq='D')
        holiday = np.zeros(n_per_series)
        holiday[::7] = 1.0
        temperature = np.sin(np.arange(n_per_series) * 2 * np.pi / 30) * 10 + 20
        y = np.cumsum(np.random.randn(n_per_series)) + holiday * (3 + i) + temperature * 0.2
        parts.append(pd.DataFrame({
            'ds': dates, 'y': y,
            'holiday': holiday, 'temperature': temperature,
            'series_id': f'series_{i}',
        }))
    return pd.concat(parts, ignore_index=True)


# ============================================================
# GBDT Covariate Tests
# ============================================================

def test_torch_boosting_known_covariates():
    """TorchBoostingForest with known covariates - single series."""
    from PipelineTS.ml_model import TorchBoostingForestModel
    data = _make_data()
    model = TorchBoostingForestModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.all_configs['known_covariates'] = ['holiday']
    model.fit(data)
    
    future = _make_future(5)
    preds_fc = model.predict(5, future_covariates=future)
    preds_no = model.predict(5)
    
    assert preds_fc.shape == (5, 2), f"Expected (5,2), got {preds_fc.shape}"
    assert preds_no.shape == (5, 2), f"Expected (5,2), got {preds_no.shape}"
    # Predictions should differ when covariates differ
    assert not np.allclose(preds_fc['y'].values, preds_no['y'].values, atol=0.01), \
        "Predictions should differ with/without future covariates"
    print("[PASS] test_torch_boosting_known_covariates")


def test_torch_boosting_past_covariates():
    """TorchBoostingForest with past covariates - single series."""
    from PipelineTS.ml_model import TorchBoostingForestModel
    data = _make_data()
    model = TorchBoostingForestModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.all_configs['past_covariates'] = ['temperature']
    model.fit(data)
    
    preds = model.predict(5)
    assert preds.shape == (5, 2)
    assert not preds['y'].isna().any(), "Predictions should not contain NaN"
    print("[PASS] test_torch_boosting_past_covariates")


def test_torch_boosting_known_and_past_covariates():
    """TorchBoostingForest with both known and past covariates."""
    from PipelineTS.ml_model import TorchBoostingForestModel
    data = _make_data()
    model = TorchBoostingForestModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.all_configs['known_covariates'] = ['holiday']
    model.all_configs['past_covariates'] = ['temperature']
    model.fit(data)

    future = _make_future(5)
    preds = model.predict(5, future_covariates=future)
    assert preds.shape == (5, 2)
    print("[PASS] test_torch_boosting_known_and_past_covariates")


def test_torch_boosting_covariates_autoregressive():
    """TorchBoostingForest autoregressive prediction (n > lags) with covariates."""
    from PipelineTS.ml_model import TorchBoostingForestModel
    data = _make_data()
    model = TorchBoostingForestModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.all_configs['known_covariates'] = ['holiday']
    model.all_configs['past_covariates'] = ['temperature']
    model.fit(data)

    future = pd.DataFrame({'holiday': np.zeros(20)})
    preds = model.predict(20, future_covariates=future)
    assert preds.shape == (20, 2), f"Expected (20,2), got {preds.shape}"
    print("[PASS] test_torch_boosting_covariates_autoregressive")


def test_torch_boosting_panel_covariates():
    """TorchBoostingForest with covariates on multi-series panel data."""
    from PipelineTS.ml_model import TorchBoostingForestModel
    data = _make_panel_data()
    model = TorchBoostingForestModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.all_configs['id_col'] = 'series_id'
    model.all_configs['known_covariates'] = ['holiday']
    model.all_configs['past_covariates'] = ['temperature']
    model.fit(data)

    # Build per-series future covariates
    future_parts = []
    for sid in data['series_id'].unique():
        fc = pd.DataFrame({'holiday': np.zeros(5), 'series_id': sid})
        future_parts.append(fc)
    future = pd.concat(future_parts, ignore_index=True)

    preds = model.predict(5, future_covariates=future)
    assert 'series_id' in preds.columns
    for sid in data['series_id'].unique():
        sid_preds = preds[preds['series_id'] == sid]
        assert len(sid_preds) == 5, f"Series {sid}: expected 5 preds, got {len(sid_preds)}"
    print("[PASS] test_torch_boosting_panel_covariates")


def test_torch_boosting_no_covariates_backward_compat():
    """TorchBoostingForest without covariates - backward compatibility."""
    from PipelineTS.ml_model import TorchBoostingForestModel
    data = _make_data()[['ds', 'y']]
    model = TorchBoostingForestModel(time_col='ds', target_col='y', lags=10, quantile=None)
    model.fit(data)
    preds = model.predict(5)
    assert preds.shape == (5, 2)
    print("[PASS] test_torch_boosting_no_covariates_backward_compat")


# ============================================================
# Prophet Covariate Tests
# ============================================================

def test_prophet_known_covariates():
    """Prophet with known covariates as regressors."""
    from PipelineTS.statistic_model.prophet import ProphetModel
    data = _make_data()
    model = ProphetModel(time_col='ds', target_col='y', quantile=None)
    model.all_configs['known_covariates'] = ['holiday']
    model.fit(data)

    future = _make_future(5, holiday_on_last=True)
    preds_fc = model.predict(5, future_covariates=future)
    preds_no = model.predict(5)

    assert preds_fc.shape == (5, 2)
    assert preds_no.shape == (5, 2)
    # Holiday on last day should shift prediction
    diff = abs(preds_fc['y'].iloc[-1] - preds_no['y'].iloc[-1])
    assert diff > 0.1, f"Holiday effect should be visible, diff={diff}"
    print("[PASS] test_prophet_known_covariates")


def test_prophet_no_covariates_backward_compat():
    """Prophet without covariates - backward compatibility."""
    from PipelineTS.statistic_model.prophet import ProphetModel
    data = _make_data()[['ds', 'y']]
    model = ProphetModel(time_col='ds', target_col='y', quantile=None)
    model.fit(data)
    preds = model.predict(5)
    assert preds.shape == (5, 2)
    print("[PASS] test_prophet_no_covariates_backward_compat")


# ============================================================
# AutoARIMA Covariate Tests
# ============================================================

def test_autoarima_known_covariates():
    """AutoARIMA with known covariates as exogenous variables."""
    from PipelineTS.statistic_model.auto_arima import AutoARIMAModel
    data = _make_data()
    model = AutoARIMAModel(time_col='ds', target_col='y', quantile=None, max_p=2, max_q=2)
    model.all_configs['known_covariates'] = ['holiday']
    model.fit(data)

    future = _make_future(5, holiday_on_last=True)
    preds_fc = model.predict(5, future_covariates=future)
    preds_no = model.predict(5)

    assert preds_fc.shape == (5, 2)
    assert preds_no.shape == (5, 2)
    diff = abs(preds_fc['y'].iloc[-1] - preds_no['y'].iloc[-1])
    assert diff > 0.1, f"Holiday effect should be visible, diff={diff}"
    print("[PASS] test_autoarima_known_covariates")


def test_autoarima_no_covariates_backward_compat():
    """AutoARIMA without covariates - backward compatibility."""
    from PipelineTS.statistic_model.auto_arima import AutoARIMAModel
    data = _make_data()[['ds', 'y']]
    model = AutoARIMAModel(time_col='ds', target_col='y', quantile=None, max_p=2, max_q=2)
    model.fit(data)
    preds = model.predict(5)
    assert preds.shape == (5, 2)
    print("[PASS] test_autoarima_no_covariates_backward_compat")


# ============================================================
# Pipeline Covariate Tests
# ============================================================

def test_pipeline_covariates_torch_boosting():
    """Pipeline with covariates using TorchBoostingForest."""
    from PipelineTS.pipeline import ModelPipeline
    data = _make_data()
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        known_covariates=['holiday'],
        past_covariates=['temperature'],
        include_models=['torch_boosting_forest'],
        scaler=True, quantile=None,
    )
    pipe.fit(data)

    future = _make_future(5)
    preds = pipe.predict(n=5, future_covariates=future)
    assert preds.shape[0] == 5
    assert 'y' in preds.columns
    print("[PASS] test_pipeline_covariates_torch_boosting")


def test_pipeline_covariates_prophet():
    """Pipeline with covariates using Prophet."""
    from PipelineTS.pipeline import ModelPipeline
    data = _make_data()
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        known_covariates=['holiday'],
        include_models=['prophet'],
        scaler=False, quantile=None,
    )
    pipe.fit(data)

    future = _make_future(5)
    preds = pipe.predict(n=5, future_covariates=future)
    assert preds.shape[0] == 5
    print("[PASS] test_pipeline_covariates_prophet")


def test_pipeline_no_covariates_backward_compat():
    """Pipeline without covariates - backward compatibility."""
    from PipelineTS.pipeline import ModelPipeline
    data = _make_data()[['ds', 'y']]
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        include_models=['torch_boosting_forest'],
        scaler=True, quantile=None,
    )
    pipe.fit(data)
    preds = pipe.predict(n=5)
    assert preds.shape[0] == 5
    print("[PASS] test_pipeline_no_covariates_backward_compat")


# ============================================================
# Main
# ============================================================

ALL_TESTS = [
    # GBDT (TorchTree)
    test_torch_boosting_known_covariates,
    test_torch_boosting_past_covariates,
    test_torch_boosting_known_and_past_covariates,
    test_torch_boosting_covariates_autoregressive,
    test_torch_boosting_panel_covariates,
    test_torch_boosting_no_covariates_backward_compat,
    # Prophet
    test_prophet_known_covariates,
    test_prophet_no_covariates_backward_compat,
    # AutoARIMA
    test_autoarima_known_covariates,
    test_autoarima_no_covariates_backward_compat,
    # Pipeline
    test_pipeline_covariates_torch_boosting,
    test_pipeline_covariates_prophet,
    test_pipeline_no_covariates_backward_compat,
]

if __name__ == '__main__':
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"All {passed} covariate tests PASSED!")
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
