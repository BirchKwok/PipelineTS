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
    models = {
        'auto_arima': AutoARIMAModel,
        'prophet': ProphetModel,
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
        # NN models
        'd_linear': DLinearModel,
        'n_linear': NLinearModel,
        'n_beats': NBeatsModel,
        'n_hits': NHitsModel,
        'tcn': TCNModel,
        'tft': TFTModel,
        'gau': GAUModel,
        'stacking_rnn': StackingRNNModel,
        'time2vec': Time2VecModel,
        'transformer': TransformerModel,
        'tide': TiDEModel,
        'patch_rnn': PatchRNNModel,
        'itransformer': ITransformerModel,
        'srs_net': SRSNetModel,
        'deepar': DeepARModel,
    }

    # Chronos-2 family is optional — only register if chronos-forecasting is installed
    try:
        from PipelineTS.nn_model.chronos import (
            Chronos2Model, Chronos2SynthModel, Chronos2SmallModel,
        )
        models['chronos_2'] = Chronos2Model
        models['chronos_2_synth'] = Chronos2SynthModel
        models['chronos_2_small'] = Chronos2SmallModel
    except ImportError:
        pass

    return frozendict(models)

def get_all_model_class_name():
    models = dict(get_all_available_models())

    res = {}

    for k, v in models.items():
        res[get_model_name_before_initial(v)] = v

    return frozendict(res)