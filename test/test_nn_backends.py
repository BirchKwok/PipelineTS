import numpy as np
import pandas as pd
import pytest

from PipelineTS.nn_model.backends import resolve_nn_backend


@pytest.fixture(scope='module')
def backend_data():
    dates = pd.date_range('2024-01-01', periods=48, freq='D')
    values = np.sin(np.linspace(0, 4 * np.pi, 48)).astype(float)
    return pd.DataFrame({'date': dates, 'value': values})


def _fit_predict(cls, data, quantile=None):
    model = cls(
        time_col='date', target_col='value', lags=6,
        quantile=quantile, epochs=3, patience=2, verbose=False,
    )
    model.fit(data)
    pred = model.predict(3)
    assert len(pred) == 3
    assert pred['value'].notna().all()
    if quantile is not None:
        assert 'value_lower' in pred.columns
        assert 'value_upper' in pred.columns
        assert pred['value_lower'].notna().all()
        assert pred['value_upper'].notna().all()
    return model


def test_resolve_nn_backend_auto_available():
    backend = resolve_nn_backend()
    assert backend in {'torch', 'mlx'}


def test_nlinear_auto_backend(backend_data):
    from PipelineTS.nn_model import NLinearModel
    model = _fit_predict(NLinearModel, backend_data)
    assert model.model.backend in {'torch', 'mlx'}


def test_dlinear_auto_backend_with_quantile(backend_data):
    from PipelineTS.nn_model import DLinearModel
    model = _fit_predict(DLinearModel, backend_data, quantile=0.9)
    assert model.model.backend in {'torch', 'mlx'}


def test_backend_choice_is_not_public(backend_data):
    from PipelineTS.nn_model import NLinearModel
    with pytest.raises(TypeError):
        NLinearModel(
            time_col='date', target_col='value', lags=6,
            quantile=None, epochs=3, patience=2, verbose=False,
            backend='torch',
        )
