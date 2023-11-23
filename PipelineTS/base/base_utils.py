import pandas as pd
from spinesUtils.asserts import raise_if

from PipelineTS.utils import time_diff, compute_time_interval


def generate_valid_data(data, valid_data, lags, time_col, target_col):
    if valid_data.shape[0] < lags * 2:
        raise_if(ValueError, time_diff(valid_data[time_col].min(),
                                       data[time_col].max())
                 != compute_time_interval(data, time_col),
                 "The validation data must start at the next time point of the training data.")

        v_df = data.iloc[-(2 * lags - valid_data.shape[0]):, :][
            [time_col, target_col]
        ]
        valid_data = pd.concat([v_df, valid_data], axis=0)

    return valid_data

