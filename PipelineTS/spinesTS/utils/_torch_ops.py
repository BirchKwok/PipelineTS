import torch
from spinesUtils.asserts import raise_if_not


def one_dim_tensor_del_elements(tensor, start_index, end_index=None):
    """Delete elements from a one dim torch tensor.
    
    Parameters
    ----------
    tensor: torch.Tensor, with one dim
    start_index: int, index of the value to be dropped
    end_index: int, index of the value to be dropped, if not None,
    the tensor will be dropped from the start_index to the end_index
    
    Returns
    -------
    torch.Tensor
    """
    raise_if_not(ValueError, isinstance(tensor, torch.Tensor), "tensor must be a torch.Tensor")
    raise_if_not(ValueError, tensor.ndim == 1, "tensor must be a one dim tensor")
    raise_if_not(ValueError, isinstance(start_index, int), "start_index must be int")
    raise_if_not(ValueError, isinstance(end_index, int) or end_index is None, "end_index must be int or None")
    if end_index is None:
        if start_index == 0:
            return tensor[1:]
        elif start_index == -1 or start_index == len(tensor):
            return tensor[:-1]
        else:
            return torch.cat((tensor[:start_index], tensor[start_index+1:]))
    else:
        raise_if_not(ValueError, end_index >= start_index, "end_index must be larger than start_index")
        return torch.cat((tensor[:start_index], tensor[end_index+1:]))
