import importlib.util
import sys


_BACKEND_INSTALL_HINTS = {
    'torch': "Install PyTorch with `pip install PipelineTS[torch]` or the platform-specific PyTorch wheel.",
    'mlx': "Install MLX on Apple Silicon with `pip install PipelineTS[mlx]`.",
}


def is_backend_available(name):
    if name == 'torch':
        return importlib.util.find_spec('torch') is not None
    if name == 'mlx':
        return importlib.util.find_spec('mlx') is not None
    return False


def is_torch_available():
    return is_backend_available('torch')


def is_mlx_available():
    return is_backend_available('mlx')


def require_backend(name):
    if not is_backend_available(name):
        raise ImportError(_BACKEND_INSTALL_HINTS.get(name, f"Backend {name!r} is not available."))


def resolve_nn_backend(device='auto', prefer_torch=False):
    if prefer_torch and is_torch_available():
        return 'torch'
    if sys.platform == 'darwin' and is_mlx_available():
        return 'mlx'
    if is_torch_available():
        return 'torch'

    raise ImportError(
        "No neural network backend is available. Install one of `PipelineTS[mlx]` or `PipelineTS[torch]`."
    )
