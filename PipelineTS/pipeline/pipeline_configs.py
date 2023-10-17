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


class PipelineConfigs:
    @ParameterTypeAssert({
        'configs': list
    })
    @ParameterValuesAssert({
        'configs': lambda s: all(isinstance(i, tuple) and len(i) == 2 and isinstance(i[1], dict) for i in s)
    })
    def __init__(self, configs):
        self.sub_configs = {'init_configs': {}, 'fit_configs': {}, 'predict_configs': {}}

        _to_process_configs = drop_duplicates_with_order(configs)

        self.configs = []
        for models_group in reindex_iterable_object(_to_process_configs, key=lambda s: s[0], index_start=1):
            for (index, (model_name, model_configs)) in models_group:
                if not all(i in self.sub_configs for i in model_configs):
                    raise KeyError

                if model_name not in MODELS:
                    raise KeyError

                if len(model_configs) < 3:
                    model_configs.update({k: self.sub_configs[k]
                                          for k in self.sub_configs if k not in model_configs})

                self.configs.append((model_name, f"{model_name}_{index}", model_configs))

        if is_in_ipython():
            display(
                tabulate(self.configs, headers=['model_name', 'model_name_with_index', 'model_configs'],
                         tablefmt='html', colalign=("right", "left", "left"), showindex='always')
            )
        else:
            print(
                tabulate(self.configs, headers=['model_name', 'model_name_with_index', 'model_configs'],
                         tablefmt='pretty', colalign=("right", "left", "left"), showindex='always')
            )

    def get_configs(self, model_name_after_rename):
        for (model_name, mnwi, model_configs) in self.configs:
            if model_name_after_rename == mnwi:
                return model_configs
