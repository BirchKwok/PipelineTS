import importlib
import sys

__version__ = '1.3.0'

from PipelineTS.easy import (
    AutoForecast,
    backtest,
    diagnose,
    forecast,
    infer_id_col,
    infer_target_col,
    infer_time_col,
    load_data,
    preprocess,
)
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs, SmartRouter


_COMPAT_MODULE_ALIASES = {
    "dataset": "datasets",
    "statistic_model": "models.statistical",
    "ml_model": "models.ml",
    "nn_model": "models.nn",
    "feature_engineering": "preprocessing.features",
    "evaluation": "metrics.evaluation",
    "plot": "utils.plot",
    "prediction": "pipeline.prediction",
    "training": "pipeline.training",
}


def _install_compat_aliases():
    for old_name, new_name in _COMPAT_MODULE_ALIASES.items():
        module = importlib.import_module(f"{__name__}.{new_name}")
        sys.modules.setdefault(f"{__name__}.{old_name}", module)


_install_compat_aliases()

__all__ = [
    "__version__",
    "AutoForecast",
    "forecast",
    "preprocess",
    "diagnose",
    "backtest",
    "load_data",
    "infer_time_col",
    "infer_target_col",
    "infer_id_col",
    "ModelPipeline",
    "PipelineConfigs",
    "SmartRouter",
]
