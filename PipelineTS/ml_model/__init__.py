from PipelineTS.ml_model.torch_tree_models import (
    TorchBoostingForestModel,
    TorchBaggingForestModel,
)
from PipelineTS.ml_model.deep_forest import DeepForestModel

# Backward-compatible aliases: old native-library classes now point to Torch models
CatBoostModel = TorchBoostingForestModel
LightGBMModel = TorchBoostingForestModel
XGBoostModel = TorchBoostingForestModel
RandomForestModel = TorchBaggingForestModel

# Backward compatibility alias
from PipelineTS.ml_model.torch_tree_models import TorchDeepForestModel
from PipelineTS.ml_model.wide_gbrt import WideGBRTModel
from PipelineTS.ml_model.multi_output_model import (
    MultiOutputRegressorModel,
    MultiStepRegressorModel,
    RegressorChainModel
)
