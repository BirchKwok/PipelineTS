import numpy as np
import pandas as pd

from PipelineTS.nn_model import DLinearModel, NLinearModel
from PipelineTS.nn_model.backends import is_mlx_available, is_torch_available


def make_data(n=48):
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    values = np.sin(np.linspace(0, 4 * np.pi, n)).astype(float)
    return pd.DataFrame({'date': dates, 'value': values})


def run_model(cls, quantile=None):
    data = make_data()
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
    print(f"{cls.__name__} backend={model.model.backend} quantile={quantile} ok")


def main():
    print(f"available torch={is_torch_available()} mlx={is_mlx_available()}")
    for cls in (NLinearModel, DLinearModel):
        run_model(cls)
        run_model(cls, quantile=0.9)


if __name__ == '__main__':
    main()
