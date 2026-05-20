from collections import namedtuple


ModernTSModelSpec = namedtuple(
    'ModernTSModelSpec',
    'key backbone_class wrapper_class variant category hidden layers native_family'
)

MODERN_TS_MODEL_SPECS = (
    ModernTSModelSpec('timexer', 'TimeXer', 'TimeXerModel', 'timexer', 'heavy', 128, 2, 'feature_bank'),
    ModernTSModelSpec('time_mixer', 'TimeMixer', 'TimeMixerModel', 'time_mixer', 'light', 96, 2, 'plain'),
    ModernTSModelSpec('timesnet', 'TimesNet', 'TimesNetModel', 'timesnet', 'medium', 128, 2, 'feature_bank'),
    ModernTSModelSpec('pyraformer', 'Pyraformer', 'PyraformerModel', 'pyraformer', 'heavy', 128, 2, 'attention'),
    ModernTSModelSpec('etsformer', 'ETSformer', 'ETSformerModel', 'etsformer', 'medium', 96, 2, 'plain'),
    ModernTSModelSpec('lightts', 'LightTS', 'LightTSModel', 'lightts', 'light', 64, 2, 'plain'),
    ModernTSModelSpec('patchtst', 'PatchTST', 'PatchTSTModel', 'patchtst', 'medium', 128, 2, 'feature_bank'),
    ModernTSModelSpec('tsmixer', 'TSMixer', 'TSMixerModel', 'tsmixer', 'light', 96, 2, 'plain'),
    ModernTSModelSpec('nonstationary_transformer', 'NonstationaryTransformer', 'NonstationaryTransformerModel', 'nonstationary_transformer', 'heavy', 128, 2, 'attention'),
    ModernTSModelSpec('fedformer', 'FEDformer', 'FEDformerModel', 'fedformer', 'heavy', 128, 2, 'feature_bank'),
    ModernTSModelSpec('autoformer', 'Autoformer', 'AutoformerModel', 'autoformer', 'medium', 128, 2, 'attention'),
    ModernTSModelSpec('informer', 'Informer', 'InformerModel', 'informer', 'heavy', 128, 2, 'attention'),
    ModernTSModelSpec('reformer', 'Reformer', 'ReformerModel', 'reformer', 'heavy', 128, 2, 'attention'),
    ModernTSModelSpec('multi_patch_former', 'MultiPatchFormer', 'MultiPatchFormerModel', 'multi_patch_former', 'heavy', 128, 2, 'feature_bank'),
    ModernTSModelSpec('wpmixer', 'WPMixer', 'WPMixerModel', 'wpmixer', 'medium', 96, 2, 'feature_bank'),
    ModernTSModelSpec('timefilter', 'TimeFilter', 'TimeFilterModel', 'timefilter', 'light', 96, 2, 'feature_bank'),
    ModernTSModelSpec('msgnet', 'MSGNet', 'MSGNetModel', 'msgnet', 'medium', 128, 2, 'feature_bank'),
    ModernTSModelSpec('seg_rnn', 'SegRNN', 'SegRNNModel', 'seg_rnn', 'light', 96, 2, 'rnn'),
    ModernTSModelSpec('tirex', 'TiRex', 'TiRexModel', 'tirex', 'medium', 128, 2, 'attention'),
)

MODERN_TS_MODEL_KEYS = tuple(spec.key for spec in MODERN_TS_MODEL_SPECS)
MODERN_TS_BACKBONE_CLASS_NAMES = tuple(spec.backbone_class for spec in MODERN_TS_MODEL_SPECS)
MODERN_TS_WRAPPER_CLASS_NAMES = tuple(spec.wrapper_class for spec in MODERN_TS_MODEL_SPECS)
MODERN_TS_KEYS_BY_CATEGORY = {
    category: {spec.key for spec in MODERN_TS_MODEL_SPECS if spec.category == category}
    for category in ('light', 'medium', 'heavy')
}
MODERN_TS_NATIVE_DEFAULTS = {
    spec.key: {'hidden': spec.hidden, 'layers': spec.layers}
    for spec in MODERN_TS_MODEL_SPECS
}
MODERN_TS_NATIVE_FEATURE_BANK_KINDS = {
    spec.key for spec in MODERN_TS_MODEL_SPECS if spec.native_family == 'feature_bank'
}
MODERN_TS_NATIVE_ATTENTION_KINDS = {
    spec.key for spec in MODERN_TS_MODEL_SPECS if spec.native_family == 'attention'
}
MODERN_TS_NATIVE_RNN_KINDS = {
    spec.key for spec in MODERN_TS_MODEL_SPECS if spec.native_family == 'rnn'
}
