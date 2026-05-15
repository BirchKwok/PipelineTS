from PipelineTS.ml_model.native_tree_models import (
    CatBoostModel,
    XGBoostModel,
    RandomForestModel,
    ExtraForestModel,
    gcForestModel,
)
from PipelineTS.ml_model.wide_gbrt import WideGBRTModel
from PipelineTS.ml_model.multi_output_model import (
    MultiOutputRegressorModel,
    MultiStepRegressorModel,
    RegressorChainModel
)

# Backward compatibility aliases
TorchBoostingForestModel = CatBoostModel
TorchBaggingForestModel = RandomForestModel
DeepForestModel = gcForestModel
TorchDeepForestModel = gcForestModel

from PipelineTS.ml_model.regressor_wrappers import MultiStepRegressor, MultiOutputRegressor
from PipelineTS.ml_model.wide_gbrt_preprocessing import GBRTPreprocessing
from PipelineTS.ml_model.dummy import DummyModel
from PipelineTS.ml_model.estimator_pipeline import Pipeline
