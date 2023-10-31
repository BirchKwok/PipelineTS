from tabulate import tabulate
from IPython.display import display

from spinesUtils.utils import (
    drop_duplicates_with_order,
    reindex_iterable_object,
    is_in_ipython
)
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert

from PipelineTS.pipeline.pipeline_models import get_all_available_models


MODELS = get_all_available_models()


def configs_check(configs):
    condition_a = all(
        isinstance(i, tuple) and 2 <= len(i) <= 3 and isinstance(i[-1], dict) for i in configs
    )

    condition_b = all(
       isinstance(i[0], str) if len(i) == 2 else isinstance(i[0], str) and isinstance(i[1], str) for i in configs
    )

    return condition_a and condition_b


class PipelineConfigs:
    @ParameterTypeAssert({
        'configs': list
    }, 'PipelineConfigs')
    @ParameterValuesAssert({
        'configs': configs_check
    }, 'PipelineConfigs')
    def __init__(self, configs):
        self.sub_configs = {'init_configs': {}, 'fit_configs': {}, 'predict_configs': {}}

        _to_process_configs = drop_duplicates_with_order(configs)

        self.configs = []
        for models_group in reindex_iterable_object(_to_process_configs, key=lambda s: s[0], index_start=1):
            for (index, (*model_name_list, model_configs)) in models_group:
                model_name = model_name_list[0]
                if len(model_name_list) == 2:
                    model_name_after_rename = model_name_list[1]
                else:
                    model_name_after_rename = f"{model_name}_{index}"

                if not all(i in self.sub_configs for i in model_configs):
                    raise KeyError

                if model_name not in MODELS:
                    raise KeyError

                if len(model_configs) < 3:
                    model_configs.update({k: self.sub_configs[k]
                                          for k in self.sub_configs if k not in model_configs})

                self.configs.append((model_name, model_name_after_rename, model_configs))

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
        for (model_name, mnar, model_configs) in self.configs:
            if model_name_after_rename == mnar:
                return model_configs
