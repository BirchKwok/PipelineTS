from frozendict import frozendict

from PipelineTS.statistic_model import *
from PipelineTS.ml_model import *
from PipelineTS.nn_model import *


def get_all_available_models():
    """获取所有可用的模型"""
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
        'seg_rnn': SegRNNModel
    })

    if extra_pkg_installed:
        MODELS = MODELS.set('prophet', ProphetModel)

    return MODELS
