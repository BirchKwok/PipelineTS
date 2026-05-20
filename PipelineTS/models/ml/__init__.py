from PipelineTS.models.ml.native_tree_models import (
    CatBoostModel,
    XGBoostModel,
    RandomForestModel,
    ExtraForestModel,
    gcForestModel,
)
from PipelineTS.models.ml.wide_gbrt import WideGBRTModel
from PipelineTS.models.ml.multi_output_model import (
    MultiOutputRegressorModel,
    MultiStepRegressorModel,
    RegressorChainModel
)

# Backward compatibility aliases
TorchBoostingForestModel = CatBoostModel
TorchBaggingForestModel = RandomForestModel
DeepForestModel = gcForestModel
TorchDeepForestModel = gcForestModel

from PipelineTS.models.ml.regressor_wrappers import MultiStepRegressor, MultiOutputRegressor
from PipelineTS.models.ml.wide_gbrt_preprocessing import GBRTPreprocessing
from PipelineTS.models.ml.dummy import DummyModel
from PipelineTS.models.ml.estimator_pipeline import Pipeline
