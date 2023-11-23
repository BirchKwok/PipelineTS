from spinesUtils import ParameterTypeAssert
from spinesUtils.utils import drop_duplicates_with_order
from frozendict import frozendict

from PipelineTS.base import IntervalEstimationMixin


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
