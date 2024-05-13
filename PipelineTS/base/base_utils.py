import pandas as pd
from frozendict import frozendict
from spinesUtils.asserts import ParameterTypeAssert
from spinesUtils.asserts import raise_if

from PipelineTS.base.base import IntervalEstimationMixin
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


def get_model_name_before_initial(model):
    import re
    model_name = list(filter(lambda s: len(s) > 0,
                             re.split("'|<|>", str(model))))[-1].split('.')[-1]

    return model_name


@ParameterTypeAssert({
    'builtin_available_models': (dict, frozendict),
    'include_models': (list, None)
})
def generate_models_set(
    builtin_available_models,
    include_models=None
):
    """
    Generate a set of models based on input parameters.

    Parameters
    ----------
    builtin_available_models : dict or frozendict
        A dictionary containing available models.

    include_models : list or None, optional
        A list of models to include in the set. If None, include all available models.

    Returns
    -------
    models_set : tuple
        A tuple containing sorted key-value pairs of models based on the model names.

    Notes
    -----
    The function sorts the models based on their names and handles both model classes and model names as strings.
    """
    if include_models is None:
        return tuple(sorted(builtin_available_models.items(), key=lambda s: s[0]))
    else:
        include_models = drop_duplicates_with_order(include_models)

        ms = {}

        for model in include_models:
            if not isinstance(model, (list, str)) and issubclass(model, IntervalEstimationMixin):
                ms[get_model_name_before_initial(model)] = model
            else:
                # At this point, the 'model' is essentially the name of the model as a string.
                ms[model] = builtin_available_models[model]

        return tuple(sorted(ms.items(), key=lambda s: s[0]))


def drop_duplicates_with_order(x):
    """
    Drop duplicates in a list while preserving the order.

    Parameters
    ----------
    x : list or tuple
        The list to be processed.

    Returns
    -------
    x : list
        The processed list.

    """
    if isinstance(x, tuple):
        x = list(x)

    result = []
    for item in x:
        if item not in result:
            result.append(item)

    return result
