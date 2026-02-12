"""Comprehensive tests for ChronosModel integration."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '.')


def _make_data(n=100, freq='D', seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq=freq)
    y = np.cumsum(np.random.randn(n)) + np.sin(np.arange(n) * 2 * np.pi / 30) * 5
    return pd.DataFrame({'ds': dates, 'y': y})


def _make_panel(n_series=3, n_per=60, seed=42):
    parts = []
    for i in range(n_series):
        np.random.seed(seed + i)
        dates = pd.date_range('2020-01-01', periods=n_per, freq='D')
        y = np.cumsum(np.random.randn(n_per)) + i * 10
        parts.append(pd.DataFrame({'ds': dates, 'y': y, 'sid': f'series_{i}'}))
    return pd.concat(parts, ignore_index=True)


# ============================================================
# 1. Basic single-series point prediction
# ============================================================
def test_chronos_basic():
    from PipelineTS.nn_model.chronos import ChronosModel
    data = _make_data()
    model = ChronosModel(time_col='ds', target_col='y', quantile=None,
                         model_name='chronos-bolt-tiny')
    model.fit(data)
    preds = model.predict(5)
    assert preds.shape == (5, 2), f"Expected (5,2), got {preds.shape}"
    assert 'ds' in preds.columns and 'y' in preds.columns
    # Predictions should be finite
    assert preds['y'].notna().all()
    print("[PASS] test_chronos_basic")


# ============================================================
# 2. Quantile / conformal intervals
# ============================================================
def test_chronos_quantile():
    from PipelineTS.nn_model.chronos import ChronosModel
    data = _make_data()
    model = ChronosModel(time_col='ds', target_col='y', quantile=0.9,
                         model_name='chronos-bolt-tiny')
    model.fit(data)
    preds = model.predict(5)
    assert preds.shape == (5, 4), f"Expected (5,4), got {preds.shape}"
    assert 'y_lower' in preds.columns and 'y_upper' in preds.columns
    # Intervals should be ordered: lower <= point <= upper
    assert (preds['y_lower'] <= preds['y']).all()
    assert (preds['y'] <= preds['y_upper']).all()
    print("[PASS] test_chronos_quantile")


# ============================================================
# 3. Multi-series (panel) prediction
# ============================================================
def test_chronos_multi_series():
    from PipelineTS.nn_model.chronos import ChronosModel
    panel = _make_panel()
    model = ChronosModel(time_col='ds', target_col='y', quantile=None,
                         model_name='chronos-bolt-tiny')
    model.all_configs['id_col'] = 'sid'
    model.fit(panel)
    preds = model.predict(5)
    assert 'sid' in preds.columns
    for sid in panel['sid'].unique():
        sid_preds = preds[preds['sid'] == sid]
        assert len(sid_preds) == 5, f"{sid}: expected 5, got {len(sid_preds)}"
    print(f"[PASS] test_chronos_multi_series: {len(preds)} total rows")


# ============================================================
# 4. Multi-series with quantile
# ============================================================
def test_chronos_multi_series_quantile():
    from PipelineTS.nn_model.chronos import ChronosModel
    panel = _make_panel(n_series=2, n_per=80)
    model = ChronosModel(time_col='ds', target_col='y', quantile=0.9,
                         model_name='chronos-bolt-tiny')
    model.all_configs['id_col'] = 'sid'
    model.fit(panel)
    preds = model.predict(5)
    assert 'y_lower' in preds.columns and 'y_upper' in preds.columns
    assert 'sid' in preds.columns
    assert len(preds) == 10  # 2 series * 5 steps
    print("[PASS] test_chronos_multi_series_quantile")


# ============================================================
# 5. Pipeline integration
# ============================================================
def test_chronos_pipeline():
    from PipelineTS.pipeline import ModelPipeline
    data = _make_data()
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        include_models=['chronos'],
        scaler=False, quantile=None,
        chronos__model_name='chronos-bolt-tiny',
    )
    pipe.fit(data)
    preds = pipe.predict(n=5)
    assert preds.shape == (5, 2)
    assert pipe.best_model_ is not None
    print("[PASS] test_chronos_pipeline")


# ============================================================
# 6. Pipeline multi-series
# ============================================================
def test_chronos_pipeline_multi_series():
    from PipelineTS.pipeline import ModelPipeline
    panel = _make_panel()
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        id_col='sid', include_models=['chronos'],
        scaler=False, quantile=None,
        chronos__model_name='chronos-bolt-tiny',
    )
    pipe.fit(panel)
    preds = pipe.predict(n=5)
    assert 'sid' in preds.columns
    assert len(preds) == 15  # 3 series * 5 steps
    print("[PASS] test_chronos_pipeline_multi_series")


# ============================================================
# 7. Model name resolution
# ============================================================
def test_chronos_model_names():
    from PipelineTS.nn_model.chronos import ChronosModel, CHRONOS_MODELS
    # All preset names should resolve
    for name, path in CHRONOS_MODELS.items():
        m = ChronosModel(time_col='ds', target_col='y', quantile=None, model_name=name)
        assert m.all_configs['hf_path'] == path, f"{name} -> {m.all_configs['hf_path']}"
    # Custom HF path should pass through
    m = ChronosModel(time_col='ds', target_col='y', model_name='my-org/my-model')
    assert m.all_configs['hf_path'] == 'my-org/my-model'
    print(f"[PASS] test_chronos_model_names: {len(CHRONOS_MODELS)} presets")


# ============================================================
# 8. Optional import — verify graceful degradation
# ============================================================
def test_chronos_registry():
    from PipelineTS.pipeline.pipeline_models import get_all_available_models
    models = get_all_available_models()
    assert 'chronos' in models, "ChronosModel should be registered"
    print(f"[PASS] test_chronos_registry: chronos in {len(models)} models")


# ============================================================
# 9. Backward compat — existing models unaffected
# ============================================================
def test_chronos_no_regression():
    from PipelineTS.pipeline import ModelPipeline
    data = _make_data()
    pipe = ModelPipeline(
        time_col='ds', target_col='y', lags=10,
        include_models=['prophet'],
        scaler=False, quantile=None,
    )
    pipe.fit(data)
    preds = pipe.predict(n=5)
    assert preds.shape == (5, 2)
    print("[PASS] test_chronos_no_regression")


# ============================================================
# 10. Different prediction horizons
# ============================================================
def test_chronos_variable_horizon():
    from PipelineTS.nn_model.chronos import ChronosModel
    data = _make_data()
    model = ChronosModel(time_col='ds', target_col='y', quantile=None,
                         model_name='chronos-bolt-tiny')
    model.fit(data)
    for n in [1, 3, 10, 20]:
        preds = model.predict(n)
        assert len(preds) == n, f"n={n}: expected {n}, got {len(preds)}"
    print("[PASS] test_chronos_variable_horizon")


if __name__ == '__main__':
    tests = [
        test_chronos_basic,
        test_chronos_quantile,
        test_chronos_multi_series,
        test_chronos_multi_series_quantile,
        test_chronos_pipeline,
        test_chronos_pipeline_multi_series,
        test_chronos_model_names,
        test_chronos_registry,
        test_chronos_no_regression,
        test_chronos_variable_horizon,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"All {passed} Chronos tests PASSED!")
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
