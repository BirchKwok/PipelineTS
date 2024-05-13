from functools import partial
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from spinesUtils.asserts import raise_if_not


def split_series(x_seq, y_seq, window_size: int, pred_steps: int, train_size: Union[None, float] = None,
                 shuffle: bool = False, skip_steps: int = 1):
    """Returns two-dimensional array cut by the specified window size.

    Parameters
    ----------
    x_seq : numpy.ndarray or pandas.Series or list, the series which needed to split
    y_seq : numpy.ndarray or pandas.Series or list, the series which needed to split
    window_size : int, sliding window size
    pred_steps : int, the number of steps predicted forward by the series
    train_size : float,
    shuffle : bool, whether to shuffle the split results
    skip_steps : int, the number of skipped steps per slide

    Returns
    -------
    numpy.ndarray.

    when train_size is not None, return X_train, X_test, y_train, y_test
    otherwise return X, y
    """
    raise_if_not(TypeError, isinstance(window_size, int), "window_size must be an integer")
    raise_if_not(ValueError, window_size > 0, "window_size must be greater than 0")
    raise_if_not(TypeError, isinstance(skip_steps, int), "skip_steps must be an integer")
    raise_if_not(ValueError, skip_steps > 0, "skip_steps must be greater than 0")
    raise_if_not(TypeError, isinstance(pred_steps, int), "pred_steps must be an integer")
    raise_if_not(ValueError, pred_steps > 0, "pred_steps must be greater than 0")
    raise_if_not(TypeError, isinstance(shuffle, bool), "shuffle must be a bool")
    raise_if_not(ValueError, train_size is None or (0 < train_size < 1 and isinstance(train_size, float)),
                 "train_size must be None or a float between 0 and 1")
    raise_if_not(TypeError, isinstance(x_seq, (pd.Series, np.ndarray, list)),
                 "x_seq must be a pandas.Series or numpy.ndarray or list")
    raise_if_not(TypeError, isinstance(y_seq, (pd.Series, np.ndarray, list)),
                 "y_seq must be a pandas.Series or numpy.ndarray or list")
    raise_if_not(ValueError, len(x_seq) == len(y_seq), "x_seq and y_seq must have the same length")
    raise_if_not(ValueError, len(x_seq) >= window_size and len(y_seq) >= window_size,
                 "x_seq and y_seq must be longer than or equal to window_size")
    raise_if_not(TypeError, isinstance(shuffle, bool), "shuffle must be a bool")

    if isinstance(x_seq, pd.Series):
        x_seq = x_seq.values
    elif isinstance(x_seq, list):
        x_seq = np.array(x_seq)
    if isinstance(y_seq, pd.Series):
        y_seq = y_seq.values
    elif isinstance(y_seq, list):
        y_seq = np.array(y_seq)
    X, y = [], []

    for i in range(len(x_seq)):
        end_index = i * skip_steps + window_size
        out_end_index = end_index + pred_steps

        if out_end_index > len(x_seq):
            break

        seq_x, seq_y = list(x_seq[i * skip_steps:end_index]), list(y_seq[end_index:out_end_index])

        X.append(seq_x)
        y.append(seq_y)

    if train_size is None:
        if pred_steps > 1:
            return np.array(X), np.array(y)
        return np.array(X), np.squeeze(np.array(y))
    elif pred_steps == 1 and train_size is not None:
        return train_test_split(np.array(X), np.squeeze(np.array(y)), train_size=train_size, shuffle=shuffle)
    else:
        return train_test_split(np.array(X), np.array(y), train_size=train_size, shuffle=shuffle)


def lag_splits(x_seq, window_size, skip_steps=1, pred_steps=1):
    """Returns a two-dimensional array cut by the specified window size.

    Parameters
    ----------
    x_seq : numpy.ndarray or pandas.Series or list, the series which needed to split
    window_size : int, sliding window size
    skip_steps : int, the number of skipped steps per slide
    pred_steps : int, the number of steps predicted forward by the series

    Return
    -------
    numpy.ndarray
    """
    raise_if_not(TypeError, isinstance(window_size, int), "window_size must be an integer")
    raise_if_not(ValueError, window_size > 0, "window_size must be greater than 0")
    raise_if_not(TypeError, isinstance(skip_steps, int), "skip_steps must be an integer")
    raise_if_not(ValueError, skip_steps > 0, "skip_steps must be greater than 0")
    raise_if_not(TypeError, isinstance(pred_steps, int), "pred_steps must be an integer")
    raise_if_not(ValueError, pred_steps > 0, "pred_steps must be greater than 0")
    raise_if_not(TypeError, isinstance(x_seq, (pd.Series, np.ndarray, list)),
                 "x_seq must be a pandas.Series or numpy.ndarray or list")
    raise_if_not(ValueError, len(x_seq) >= window_size, "x_seq must be longer than or equal to window_size")

    if isinstance(x_seq, pd.Series):
        x_seq = x_seq.values
    elif isinstance(x_seq, list):
        x_seq = np.array(x_seq)

    X = []
    p, _ = 0, 0
    for i in range(len(x_seq)):
        if p == pred_steps:
            p, _ = 1, i
        else:
            p += 1
        end_index = _ * skip_steps + window_size
        if end_index > len(x_seq):
            break

        seq_x = list(x_seq[_ * skip_steps:end_index])

        X.append(seq_x)

    return np.array(X)


train_test_split_ts = partial(train_test_split, shuffle=False)  # train_test_split for time series
