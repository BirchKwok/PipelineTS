from ._constants import seed_everything
try:
    from ._torch_ops import one_dim_tensor_del_elements
except ImportError as _torch_ops_import_error:
    def one_dim_tensor_del_elements(*args, **kwargs):
        raise ImportError(
            "Torch tensor utilities require the torch backend. Install it with `pip install PipelineTS[torch]`."
        ) from _torch_ops_import_error
from ._validation import check_is_fitted
