import numpy as np
import pandas as pd

from PipelineTS.nn_model import (
    DeepARModel,
    DLinearModel,
    GAUModel,
    ITransformerModel,
    NBeatsModel,
    NHitsModel,
    NLinearModel,
    PatchRNNModel,
    SRSNetModel,
    StackingRNNModel,
    TCNModel,
    TFTModel,
    TiDEModel,
    Time2VecModel,
    TransformerModel,
)
from PipelineTS.pipeline.pipeline_models import get_all_available_models
from PipelineTS.spinesTS.backends import is_mlx_available, is_torch_available, resolve_nn_backend


def make_data(n=56):
    return pd.DataFrame(
        {
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'value': np.sin(np.linspace(0, 4 * np.pi, n)).astype(float),
        }
    )


def make_multivariate_data(n=72):
    return pd.DataFrame(
        {
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'value': np.sin(np.linspace(0, 4 * np.pi, n)).astype(float),
            'feature_a': np.cos(np.linspace(0, 4 * np.pi, n)).astype(float),
            'feature_b': (0.5 * np.sin(np.linspace(0, 2 * np.pi, n))).astype(float),
        }
    )


def check_univariate(cls):
    data = make_data()
    model = cls(
        time_col='date',
        target_col='value',
        lags=6,
        quantile=None,
        epochs=2,
        patience=1,
        verbose=False,
    )
    model.fit(data)
    pred = model.predict(3)
    assert len(pred) == 3
    assert pred['value'].notna().all()
    print(f'OK {cls.__name__} backend={getattr(model.model, "backend", None)}')


def check_multivariate(cls):
    data = make_multivariate_data()
    model = cls(
        time_col='date',
        target_col='value',
        feature_cols=['value', 'feature_a', 'feature_b'],
        lags=6,
        quantile=None,
        epochs=2,
        patience=1,
        verbose=False,
    )
    model.fit(data)
    pred = model.predict(3)
    assert len(pred) == 3
    assert pred['value'].notna().all()
    print(f'OK {cls.__name__} backend={getattr(model.model, "backend", None)}')


def check_registry():
    registered = get_all_available_models()
    names = [
        'd_linear',
        'n_linear',
        'n_beats',
        'n_hits',
        'tcn',
        'tft',
        'gau',
        'stacking_rnn',
        'time2vec',
        'transformer',
        'tide',
        'patch_rnn',
        'itransformer',
        'srs_net',
        'deepar',
    ]
    for name in names:
        assert name in registered, name
    print(f'OK registry nn_models={len(names)} total={len(registered)}')


def main():
    backend = resolve_nn_backend()
    print(f'available torch={is_torch_available()} mlx={is_mlx_available()} selected={backend}')
    for cls in (
        NLinearModel,
        DLinearModel,
        Time2VecModel,
        GAUModel,
        StackingRNNModel,
        PatchRNNModel,
        TCNModel,
        NBeatsModel,
        NHitsModel,
        TransformerModel,
        TFTModel,
        TiDEModel,
        DeepARModel,
    ):
        check_univariate(cls)
    for cls in (ITransformerModel, SRSNetModel):
        check_multivariate(cls)
    check_registry()


if __name__ == '__main__':
    main()
