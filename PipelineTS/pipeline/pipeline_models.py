from frozendict import frozendict

from PipelineTS.models.statistical import *
from PipelineTS.models.ml import *
from PipelineTS.models.nn import *
from PipelineTS.models.nn.backends import is_mlx_available, is_torch_available
from PipelineTS.models.nn._foundation_specs import FOUNDATION_MODEL_SPECS
from PipelineTS.models.nn._nn_specs import CORE_NN_MODEL_SPECS

from PipelineTS.base.base_utils import get_model_name_before_initial


def get_all_available_models():
    """
    Retrieve a dictionary of all available model classes in the pipeline.

    Returns
    -------
    models : frozendict
        A frozendict containing model names as keys and corresponding model class references as values.

    Notes
    -----
    - The function attempts to import external dependencies to check for additional models.
    - If the 'prophet' package is installed, a 'prophet' model will be added to the available models.
    """
    models = {
        'auto_arima': AutoARIMAModel,
        'prophet': ProphetModel,
        'naive': NaiveModel,
        'seasonal_naive': SeasonalNaiveModel,
        'theta': ThetaModel,
        'ets': ETSModel,
        'short_trend_slot_blend': ShortTrendSlotBlendModel,
        'long_slot_trend_blend': LongSlotTrendBlendModel,
        'stat_ensemble': StatisticalEnsembleModel,
        # ML tree models (native implementations)
        'catboost': CatBoostModel,
        'xgboost': XGBoostModel,
        'random_forest': RandomForestModel,
        'extra_forest': ExtraForestModel,
        'gc_forest': gcForestModel,
        'wide_gbrt': WideGBRTModel,
        'multi_output_model': MultiOutputRegressorModel,
        'multi_step_model': MultiStepRegressorModel,
        'regressor_chain': RegressorChainModel,
    }

    if is_torch_available() or is_mlx_available():
        models.update({spec.key: globals()[spec.wrapper_class] for spec in CORE_NN_MODEL_SPECS})
        models.update(MODERN_TS_MODEL_CLASSES)

    models.update({spec.key: globals()[spec.wrapper_class] for spec in FOUNDATION_MODEL_SPECS})

    return frozendict(models)

def get_all_model_class_name():
    models = dict(get_all_available_models())

    res = {}

    for k, v in models.items():
        res[get_model_name_before_initial(v)] = v

    return frozendict(res)