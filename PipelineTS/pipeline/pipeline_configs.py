from tabulate import tabulate
from IPython.display import display

from spinesUtils.utils import (
    reindex_iterable_object,
    is_in_ipython
)
from spinesUtils.asserts import ParameterTypeAssert, ParameterValuesAssert

from PipelineTS.pipeline.pipeline_models import get_all_available_models
from PipelineTS.base.base_utils import drop_duplicates_with_order


MODELS = get_all_available_models()


def configs_check(configs):
    """
    Check the validity of the given configuration format.

    Parameters
    ----------
    configs : list
        List of configuration tuples for models.

    Returns
    -------
    valid : bool
        True if the configuration format is valid, False otherwise.

    Notes
    -----
    - Each tuple in the list must have 2 or 3 elements.
    - The last element of each tuple must be a dictionary.
    - The first element of each tuple must be a string or tuple of two strings.
    """
    condition_a = all(
        isinstance(i, tuple) and 2 <= len(i) <= 3 and isinstance(i[-1], dict) for i in configs
    )

    condition_b = all(
       isinstance(i[0], str) if len(i) == 2 else isinstance(i[0], str) and isinstance(i[1], str) for i in configs
    )

    return condition_a and condition_b


class PipelineConfigs:
    """Per-model configuration for :class:`~PipelineTS.pipeline.ModelPipeline`.

    Provides fine-grained control over each model's initialization, training,
    prediction, and pipeline-level settings (lags, scaler, differencing).
    Multiple instances of the same model can be registered under different
    names to compare configurations side-by-side in the leaderboard.

    Parameters
    ----------
    configs : list of tuple
        Each tuple has the form::

            (model_name, alias, config_dict)          # 3-element form
            (model_name, config_dict)                 # 2-element form (auto-alias)

        - ``model_name`` : str — must be a valid model key (see
          :meth:`~ModelPipeline.list_all_available_models`).
        - ``alias`` : str — custom name for this instance in the leaderboard.
          When omitted, a numeric suffix is appended (e.g. ``'d_linear_1'``).
        - ``config_dict`` : dict — any combination of the four sub-config keys:

          * ``'init_configs'`` : dict — keyword arguments passed to the model
            constructor (e.g. ``{'hidden_size': 128}``).
          * ``'fit_configs'`` : dict — keyword arguments passed to ``model.fit()``.
          * ``'predict_configs'`` : dict — keyword arguments passed to
            ``model.predict()``.
          * ``'pipeline_configs'`` : dict — pipeline-level overrides:

            - ``'lags'`` : int — per-model lag window (overrides global ``lags``).
            - ``'scaler'`` : bool, None, or ``TransformerMixin`` — ``True`` for
              ``MinMaxScaler``, ``None`` for no scaling, or a custom fitted
              scaler instance.
            - ``'differential_n'`` : int — differencing order (applied only
              when the model supports it).
            - ``'feature_cols'`` : list of str — multivariate feature columns.

    Raises
    ------
    KeyError
        If ``model_name`` is not a registered model, or if ``config_dict``
        contains unrecognised keys.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
    >>>
    >>> configs = PipelineConfigs([
    ...     # Two DLinear instances with different lags and scalers
    ...     ('d_linear', 'd_linear_24',  {'pipeline_configs': {'lags': 24}}),
    ...     ('d_linear', 'd_linear_std', {'pipeline_configs': {'lags': 12,
    ...                                                         'scaler': StandardScaler()}}),
    ...     # CatBoost with model-specific hyperparams and differencing
    ...     ('catboost', 'catboost_diff', {
    ...         'init_configs': {'iterations': 200},
    ...         'pipeline_configs': {'differential_n': 1},
    ...     }),
    ... ])
    >>> pipeline = ModelPipeline(time_col='date', target_col='value',
    ...                          lags=12, configs=configs)
    >>> pipeline.fit(data)
    """
    @ParameterTypeAssert({
        'configs': list
    }, 'PipelineConfigs')
    @ParameterValuesAssert({
        'configs': configs_check
    }, 'PipelineConfigs')
    def __init__(self, configs):
        # Initialize sub-configurations
        self.sub_configs = {'init_configs': {}, 'fit_configs': {}, 'predict_configs': {}, 'pipeline_configs': {}}

        # Remove duplicates and maintain order
        _to_process_configs = drop_duplicates_with_order(configs)

        self.configs = []
        for models_group in reindex_iterable_object(_to_process_configs, key=lambda s: s[0], index_start=1):
            for (index, (*model_name_list, model_configs)) in models_group:
                model_name = model_name_list[0]
                if len(model_name_list) == 2:
                    model_name_after_rename = model_name_list[1]
                else:
                    model_name_after_rename = f"{model_name}_{index}"

                # Check if required sub-configurations are present
                if not all(i in self.sub_configs for i in model_configs):
                    raise KeyError

                # Check if the model_name is predefined
                if model_name not in MODELS:
                    raise KeyError

                # Update with default sub-configurations if necessary
                if len(model_configs) < 3:
                    model_configs.update({k: self.sub_configs[k]
                                          for k in self.sub_configs if k not in model_configs})

                self.configs.append((model_name, model_name_after_rename, model_configs))

        # Display the configurations in a tabular format
        if is_in_ipython():
            display(
                tabulate(self.configs, headers=['model_name', 'model_name_after_rename', 'model_configs'],
                         tablefmt='html', colalign=("right", "left", "left"), showindex='always')
            )
        else:
            print(
                tabulate(self.configs, headers=['model_name', 'model_name_after_rename', 'model_configs'],
                         tablefmt='pretty', colalign=("right", "left", "left"), showindex='always')
            )

    def get_configs(self, model_name_after_rename):
        """
        Retrieve the configuration details for a specific model.

        Parameters
        ----------
        model_name_after_rename : str
            The name of the model for which to retrieve configuration details.

        Returns
        -------
        model_configs : dict or None
            A dictionary containing the configuration details for the specified model.

        Example
        -------
        >>> pipeline_configs = PipelineConfigs(configs=[('catboost', 'my_catboost', {'init_configs': {...}, 'fit_configs': {...}})])
        >>> configs_model_a = pipeline_configs.get_configs('my_boosting')
        >>> print(configs_model_a)
        {'init_configs': {...}, 'fit_configs': {...}, 'predict_configs': {}}
        """
        for (model_name, mnar, model_configs) in self.configs:
            if model_name_after_rename == mnar:
                return model_configs
        return None
