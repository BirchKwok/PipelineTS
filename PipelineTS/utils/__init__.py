# utility functions
import numpy as np


def update_dict_without_conflict(dict_a, dict_b):
    for i in dict_b:
        if i not in dict_a:
            dict_a[i] = dict_b[i]
    return dict_a


def check_time_col_is_timestamp(data, time_col):
    import pandas as pd
    from spinesUtils.asserts import raise_if_not

    raise_if_not(TypeError, pd.api.types.is_datetime64_any_dtype(data[time_col]),
                 'The time column must be of datetime type, '
                 'consider use pandas.to_datetime to convert it.')


def compute_time_interval(data, time_col):
    return data[time_col].diff().mode().values[0]


def time_diff(a, b):
    return np.timedelta64(a - b)


def infer_freq(data, time_col):
    """Infer pandas frequency string from a datetime column.

    Returns a frequency string compatible with ``pd.date_range``.
    Falls back to the modal timedelta if ``pd.infer_freq`` fails.
    """
    import pandas as pd
    ts = pd.to_datetime(data[time_col]).sort_values()
    freq = pd.infer_freq(ts)
    if freq is not None:
        return freq
    # Fallback: use modal diff
    delta = ts.diff().mode().values[0]
    return delta


def make_future_dates(last_dt, n, freq):
    """Generate *n* future timestamps starting after *last_dt*.

    Parameters
    ----------
    last_dt : datetime-like
        Last observed timestamp.
    n : int
        Number of future steps.
    freq : str or timedelta
        Frequency (e.g. ``'MS'``, ``'D'``, or a timedelta).

    Returns
    -------
    pd.DatetimeIndex of length *n*.
    """
    import pandas as pd
    try:
        return pd.date_range(start=last_dt, periods=n + 1, freq=freq)[1:]
    except Exception:
        # Ultimate fallback: daily
        return last_dt + pd.to_timedelta(range(n + 1), unit='D')[1:]

from PipelineTS.utils.random import seed_everything, check_if_datats
from PipelineTS.utils.validation import check_is_fitted
try:
    from PipelineTS.utils.torch_ops import one_dim_tensor_del_elements
except ImportError as _torch_ops_import_error:
    def one_dim_tensor_del_elements(*args, **kwargs):
        raise ImportError(
            "Torch tensor utilities require the torch backend. Install it with `pip install PipelineTS[torch]`."
        ) from _torch_ops_import_error
