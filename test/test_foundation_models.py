import pandas as pd
import pytest

from PipelineTS.nn_model._foundation_specs import FOUNDATION_HF_PATHS, FOUNDATION_MODEL_KEYS


def test_foundation_models_registered():
    from PipelineTS.pipeline import ModelPipeline

    available = set(ModelPipeline.list_all_available_models())
    assert set(FOUNDATION_MODEL_KEYS).issubset(available)


def test_foundation_model_init_without_loading_dependencies():
    from PipelineTS.nn_model import TiRexFoundationModel, SundialModel, TimeMoEModel

    models = [
        TiRexFoundationModel(time_col='ds', target_col='y', quantile=None),
        SundialModel(time_col='ds', target_col='y', quantile=None),
        TimeMoEModel(time_col='ds', target_col='y', quantile=None),
    ]
    for model in models:
        assert model._pipeline is None
        assert model.all_configs['hf_path'] in FOUNDATION_HF_PATHS.values()


def test_foundation_missing_dependency_errors_are_actionable(monkeypatch):
    import PipelineTS.nn_model.foundation as foundation
    from PipelineTS.nn_model import SundialModel, TiRexFoundationModel

    data = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=16), 'y': range(16)})

    monkeypatch.setattr(foundation, '_import_transformers', lambda: (_ for _ in ()).throw(ImportError('install transformers')))
    with pytest.raises(ImportError, match='install transformers'):
        SundialModel(time_col='ds', target_col='y', quantile=None).fit(data)

    monkeypatch.setattr(foundation, '_import_tirex', lambda: (_ for _ in ()).throw(ImportError('install tirex-ts')))
    with pytest.raises(ImportError, match='install tirex-ts'):
        TiRexFoundationModel(time_col='ds', target_col='y', quantile=None).fit(data)
