import importlib.util
import time

import numpy as np
import pandas as pd

from PipelineTS.ml_model import gcForestModel


def make_data(n=90):
    rng = np.random.default_rng(42)
    dates = pd.date_range('2021-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 5 * np.pi, n)) + 0.05 * rng.standard_normal(n)
    return pd.DataFrame({'date': dates, 'value': values})


def run_case(accelerator):
    data = make_data()
    model = gcForestModel(
        time_col='date', target_col='value', lags=6,
        quantile=None, n_layers=2, n_estimators_per_layer=16,
        max_depth=4, random_state=42, accelerator=accelerator,
    )
    t0 = time.perf_counter()
    model.fit(data)
    fit_seconds = time.perf_counter() - t0
    pred = model.predict(4)
    assert len(pred) == 4
    assert pred['value'].notna().all()
    return getattr(model.model, 'resolved_accelerator_', accelerator), fit_seconds


def main():
    auto_backend, auto_seconds = run_case('auto')
    parts = [
        f'auto={auto_backend} auto_fit={auto_seconds:.3f}s',
    ]
    if importlib.util.find_spec('mlx') is not None:
        mlx_backend, mlx_seconds = run_case('mlx')
        parts.append(f'mlx={mlx_backend} mlx_fit={mlx_seconds:.3f}s')
    if importlib.util.find_spec('torch') is not None:
        torch_backend, torch_seconds = run_case('torch')
        parts.append(f'torch={torch_backend} torch_fit={torch_seconds:.3f}s')
    sklearn_backend, sklearn_seconds = run_case('sklearn')
    parts.append(f'sklearn={sklearn_backend} sklearn_fit={sklearn_seconds:.3f}s')
    print('PASS gc_forest ' + ' '.join(parts))


if __name__ == '__main__':
    main()
