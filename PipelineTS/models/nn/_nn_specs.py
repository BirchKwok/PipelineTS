from collections import namedtuple

from PipelineTS.models.nn._modern_ts_specs import MODERN_TS_MODEL_KEYS, MODERN_TS_MODEL_SPECS, MODERN_TS_KEYS_BY_CATEGORY
from PipelineTS.models.nn._foundation_specs import FOUNDATION_MODEL_KEYS


NNModelSpec = namedtuple('NNModelSpec', 'key wrapper_class category family')

CORE_NN_MODEL_SPECS = (
    NNModelSpec('d_linear', 'DLinearModel', 'light', 'linear'),
    NNModelSpec('n_linear', 'NLinearModel', 'light', 'linear'),
    NNModelSpec('tide', 'TiDEModel', 'light', 'mlp'),
    NNModelSpec('tcn', 'TCNModel', 'light', 'conv'),
    NNModelSpec('n_beats', 'NBeatsModel', 'medium', 'basis'),
    NNModelSpec('n_hits', 'NHitsModel', 'medium', 'basis'),
    NNModelSpec('stacking_rnn', 'StackingRNNModel', 'medium', 'rnn'),
    NNModelSpec('patch_rnn', 'PatchRNNModel', 'medium', 'rnn'),
    NNModelSpec('time2vec', 'Time2VecModel', 'medium', 'embedding'),
    NNModelSpec('gau', 'GAUModel', 'medium', 'attention'),
    NNModelSpec('transformer', 'TransformerModel', 'heavy', 'attention'),
    NNModelSpec('tft', 'TFTModel', 'heavy', 'attention'),
    NNModelSpec('itransformer', 'ITransformerModel', 'heavy', 'multivariate'),
    NNModelSpec('srs_net', 'SRSNetModel', 'heavy', 'multivariate'),
    NNModelSpec('deepar', 'DeepARModel', 'heavy', 'probabilistic'),
)

CORE_NN_MODEL_KEYS = tuple(spec.key for spec in CORE_NN_MODEL_SPECS)
CORE_NN_WRAPPER_CLASS_NAMES = tuple(spec.wrapper_class for spec in CORE_NN_MODEL_SPECS)
CORE_NN_KEYS_BY_CATEGORY = {
    category: {spec.key for spec in CORE_NN_MODEL_SPECS if spec.category == category}
    for category in ('light', 'medium', 'heavy')
}
NN_KEYS_BY_CATEGORY = {
    category: CORE_NN_KEYS_BY_CATEGORY[category] | MODERN_TS_KEYS_BY_CATEGORY[category]
    for category in ('light', 'medium', 'heavy')
}
NN_MODEL_KEYS = CORE_NN_MODEL_KEYS + MODERN_TS_MODEL_KEYS
NN_LIGHT_MODEL_KEYS = tuple(spec.key for spec in CORE_NN_MODEL_SPECS if spec.category == 'light') + tuple(spec.key for spec in MODERN_TS_MODEL_SPECS if spec.category == 'light')
NN_MEDIUM_MODEL_KEYS = tuple(spec.key for spec in CORE_NN_MODEL_SPECS if spec.category == 'medium') + tuple(spec.key for spec in MODERN_TS_MODEL_SPECS if spec.category == 'medium')
NN_HEAVY_MODEL_KEYS = tuple(spec.key for spec in CORE_NN_MODEL_SPECS if spec.category == 'heavy') + tuple(spec.key for spec in MODERN_TS_MODEL_SPECS if spec.category == 'heavy')
NN_GTB_MODEL_KEYS = tuple(
    spec.key for spec in CORE_NN_MODEL_SPECS
    if spec.family not in {'multivariate', 'probabilistic'}
) + MODERN_TS_MODEL_KEYS
NN_TRANSFORMER_LIKE_MODEL_KEYS = (
    'transformer', 'tft', 'itransformer', 'gau', 'time2vec',
) + MODERN_TS_MODEL_KEYS
NN_FOUNDATION_MODEL_KEYS = FOUNDATION_MODEL_KEYS
