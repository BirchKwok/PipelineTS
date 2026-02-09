# utility functions
import numpy as np


def update_dict_without_conflict(dict_a, dict_b):
    for i in dict_b:
        if i not in dict_a:
            dict_a[i] = dict_b[i]
    return dict_a


def check_time_col_is_timestamp(data, time_col):
    from spinesUtils.asserts import raise_if_not

    raise_if_not(TypeError, data[time_col].dtype == 'datetime64[ns]',
                 'The time column must be of type datetime64[ns], '
                 'consider use pandas.to_datetime to convert it.')


def compute_time_interval(data, time_col):
    return data[time_col].diff().mode().values[0]


def time_diff(a, b):
    return np.timedelta64(a - b)
