import pandas as pd
from spinesUtils.asserts import raise_if_not


def moving_average(x, window_size=3):
    raise_if_not(ValueError, isinstance(window_size, int), "window_size must be an integer")
    raise_if_not(ValueError, window_size > 0, "window_size must be greater than 0")
    if x.ndim == 1:
        x = pd.Series(x)
        return x.rolling(window_size).mean().values[window_size-1:]
    elif x.ndim == 2:
        x = pd.DataFrame(x.T)
        return x.rolling(window_size).mean().values.T[:, window_size-1:]
    else:
        raise ValueError("x must be one dim or two dim sequence.")
    