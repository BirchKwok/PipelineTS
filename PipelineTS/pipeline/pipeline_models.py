from frozendict import frozendict

from PipelineTS.statistic_model import *
from PipelineTS.ml_model import *
from PipelineTS.nn_model import *

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
    try:
        from prophet import Prophet
        extra_pkg_installed = True
    except ImportError:
        extra_pkg_installed = False

    MODELS = frozendict({
        'auto_arima': AutoARIMAModel,
        'catboost': CatBoostModel,
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'wide_gbrt': WideGBRTModel,
        'd_linear': DLinearModel,
        'n_linear': NLinearModel,
        'n_beats': NBeatsModel,
        'n_hits': NHitsModel,
        'tcn': TCNModel,
        'tft': TFTModel,
        'gau': GAUModel,
        'stacking_rnn': StackingRNNModel,
        'time2vec': Time2VecModel,
        'multi_output_model': MultiOutputRegressorModel,
        'multi_step_model': MultiStepRegressorModel,
        'transformer': TransformerModel,
        'random_forest': RandomForestModel,
        'tide': TiDEModel,
        'patch_rnn': PatchRNNModel,
        'regressor_chain': RegressorChainModel
    })

    if extra_pkg_installed:
        MODELS = MODELS.set('prophet', ProphetModel)

    return MODELS


def get_all_model_class_name():
    models = dict(get_all_available_models())

    res = {}

    for k, v in models.items():
        res[get_model_name_before_initial(v)] = v

    return frozendict(res)