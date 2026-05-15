from ._base_mixin import ForecastingMixin
try:
    from ._torch_mixin import TorchModelMixin, detect_available_device
except ImportError as _torch_import_error:
    class TorchModelMixin:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The torch backend is not installed. Install it with `pip install PipelineTS[torch]`."
            ) from _torch_import_error

    def detect_available_device(*args, **kwargs):
        raise ImportError(
            "The torch backend is not installed. Install it with `pip install PipelineTS[torch]`."
        ) from _torch_import_error
