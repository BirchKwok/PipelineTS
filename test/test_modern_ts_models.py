import numpy as np
import pytest

from PipelineTS.nn_model._modern_ts_specs import MODERN_TS_MODEL_KEYS, MODERN_TS_MODEL_SPECS
from PipelineTS.nn_model._nn_specs import CORE_NN_MODEL_SPECS, NN_KEYS_BY_CATEGORY, NN_MODEL_KEYS


def test_modern_ts_specs_are_registered():
    from PipelineTS.nn_model.modern_ts import MODERN_TS_MODEL_CLASSES
    from PipelineTS.pipeline import ModelPipeline

    available = set(ModelPipeline.list_all_available_models())
    assert set(MODERN_TS_MODEL_KEYS).issubset(available)
    assert set(MODERN_TS_MODEL_KEYS) == set(MODERN_TS_MODEL_CLASSES)


def test_unified_nn_specs_are_registered():
    from PipelineTS.pipeline import ModelPipeline

    available = set(ModelPipeline.list_all_available_models())
    assert set(NN_MODEL_KEYS).issubset(available)
    assert set(NN_MODEL_KEYS) == (
        NN_KEYS_BY_CATEGORY['light'] |
        NN_KEYS_BY_CATEGORY['medium'] |
        NN_KEYS_BY_CATEGORY['heavy']
    )


@pytest.mark.parametrize('spec', CORE_NN_MODEL_SPECS)
def test_core_nn_classes_use_shared_wrapper(spec):
    import PipelineTS.nn_model as nn_model
    from PipelineTS.nn_model._wrapper import NNBackboneForecastingMixin, MultivariateNNBackboneForecastingMixin

    cls = getattr(nn_model, spec.wrapper_class)
    assert issubclass(cls, (NNBackboneForecastingMixin, MultivariateNNBackboneForecastingMixin))


@pytest.mark.parametrize('model_key', ['d_linear', 'n_linear', 'tide', 'tcn', 'gau'])
def test_core_nn_shared_wrapper_init_smoke(model_key):
    from PipelineTS.pipeline.pipeline_models import get_all_available_models

    cls = get_all_available_models()[model_key]
    model = cls(
        time_col='date', target_col='value', lags=8,
        quantile=None, epochs=1, patience=1, verbose=False,
        accelerator='cpu',
    )
    assert model.all_configs['lags'] == 8
    assert 'model_configs' in model.all_configs
    assert model.model is not None


@pytest.mark.parametrize('spec', MODERN_TS_MODEL_SPECS)
def test_modern_ts_dynamic_classes_exist(spec):
    from PipelineTS.nn_model import backbones
    import PipelineTS.nn_model as nn_model

    assert hasattr(backbones, spec.backbone_class)
    assert hasattr(nn_model, spec.wrapper_class)


@pytest.mark.parametrize('model_key', ['lightts', 'patchtst', 'seg_rnn', 'nonstationary_transformer'])
def test_modern_ts_backbone_forward_smoke(model_key):
    torch = pytest.importorskip('torch')
    from PipelineTS.nn_model.backbones._modern_ts import ModernTSBackbone

    model = ModernTSBackbone(
        in_features=8, out_features=8, variant=model_key,
        d_model=16, n_heads=2, e_layers=1, d_ff=32,
    )
    x = torch.as_tensor(np.random.randn(2, 8).astype(np.float32))
    y = model(x)
    assert tuple(y.shape) == (2, 8)
