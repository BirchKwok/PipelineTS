from importlib import import_module

from PipelineTS.nn_model._modern_ts_specs import MODERN_TS_MODEL_SPECS
from PipelineTS.nn_model.backends._native_models import make_native_dispatcher


def _missing_model(name, error):
    class _MissingModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} is unavailable because its backend dependency could not be imported: {error}"
            )

    _MissingModel.__name__ = name
    return _MissingModel


def _load(module_name, attr_name):
    try:
        module = import_module(module_name, package=__name__)
        return getattr(module, attr_name)
    except ImportError as exc:
        return _missing_model(attr_name, exc)


Time2VecNet = make_native_dispatcher('time2vec', _load('._time2vec_net', 'Time2VecNet'))
GAUNet = make_native_dispatcher('gau', _load('._gau', 'GAUNet'))
StackingRNN = make_native_dispatcher('stacking_rnn', _load('._rnn', 'StackingRNN'))
PatchRNN = make_native_dispatcher('patch_rnn', _load('._patch_rnn', 'PatchRNN'))
TCN = make_native_dispatcher('tcn', _load('._tcn', 'TCN'))
ITransformer = make_native_dispatcher('itransformer', _load('._itransformer', 'ITransformer'))
SRSNet = make_native_dispatcher('srs_net', _load('._srs_net', 'SRSNet'))
NLinear = _load('._n_linear', 'NLinear')
DLinear = _load('._d_linear', 'DLinear')
NBeats = make_native_dispatcher('nbeats', _load('._n_beats', 'NBeats'))
NHiTS = make_native_dispatcher('nhits', _load('._n_hits', 'NHiTS'))
TSTransformer = make_native_dispatcher('transformer', _load('._transformer', 'TSTransformer'))
TFT = make_native_dispatcher('tft', _load('._tft', 'TFT'))
TiDE = make_native_dispatcher('tide', _load('._tide', 'TiDE'))
DeepAR = make_native_dispatcher('deepar', _load('._deepar', 'DeepAR'))

for _spec in MODERN_TS_MODEL_SPECS:
    globals()[_spec.backbone_class] = make_native_dispatcher(
        _spec.key,
        _load('._modern_ts', _spec.backbone_class),
    )
