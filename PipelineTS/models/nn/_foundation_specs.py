from collections import namedtuple


FoundationModelSpec = namedtuple('FoundationModelSpec', 'key wrapper_class family hf_path')

FOUNDATION_MODEL_SPECS = (
    FoundationModelSpec('chronos_2', 'Chronos2Model', 'chronos', 'amazon/chronos-2'),
    FoundationModelSpec('chronos_2_synth', 'Chronos2SynthModel', 'chronos', 'autogluon/chronos-2-synth'),
    FoundationModelSpec('chronos_2_small', 'Chronos2SmallModel', 'chronos', 'autogluon/chronos-2-small'),
    FoundationModelSpec('tirex_foundation', 'TiRexFoundationModel', 'tirex', 'NX-AI/TiRex'),
    FoundationModelSpec('sundial', 'SundialModel', 'sundial', 'thuml/sundial-base-128m'),
    FoundationModelSpec('time_moe', 'TimeMoEModel', 'time_moe', 'Maple728/TimeMoE-50M'),
)

FOUNDATION_MODEL_KEYS = tuple(spec.key for spec in FOUNDATION_MODEL_SPECS)
FOUNDATION_WRAPPER_CLASS_NAMES = tuple(spec.wrapper_class for spec in FOUNDATION_MODEL_SPECS)
FOUNDATION_HF_PATHS = {spec.key: spec.hf_path for spec in FOUNDATION_MODEL_SPECS}
