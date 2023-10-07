import time

from frozendict import frozendict
from spinesTS.metrics import mae
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert, generate_function_kwargs
from spinesUtils.asserts import check_obj_is_function
from spinesUtils.utils import Logger
import pandas as pd

from PipelineTS.statistic_model import *
from PipelineTS.ml_model import *
from PipelineTS.nn_model import *

# TODO: 传入数据，进行数据采集周期检验，看看是否有漏数据，如果有，进行插值（可选），如果有异常值，进行噪音去除（可选）
# TODO: 模型依次对数据进行预测，并将结果进行评估，默认评价函数为MAE和WMAPE，评价函数可选
# TODO: 输出预测模型评价结果，按评价结果排序


MODELS = frozendict({
    'prophet': ProphetModel,
    'auto_arima': AutoARIMAModel,
    'catboost': CatBoostModel,
    'lightgbm': LightGBMModel,
    'xgboost': XGBoostModel,
    'wide_gbrt': WideGBRTModel,
    'd_linear': DLinearModel,
    'n_linear': NLinearModel,
    'n_beats': NBeatsModel,
    'n_hits': NHitsModel,
    'tcn': TCNModel,
    'tft': TFTModel
})


class PipelineConfigs:
    @ParameterTypeAssert({
        'configs': dict
    })
    def __init__(self, configs):
        self.configs = configs
        self.check_configs()

    def check_configs(self):
        assert all(i in MODELS for i in self.configs)

        configs_level2_keys = ('init_configs', 'fit_configs', 'predict_configs')
        assert all(i in configs_level2_keys for i in j for j in self.configs)

    def get_configs(self, model_name):
        return self.configs.get(model_name)


class PipelineTS:
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'configs': (None, PipelineConfigs),
        'random_state': int,
        'verbose': bool
    }, 'Pipeline')
    @ParameterValuesAssert({
        'metric': lambda s: check_obj_is_function(s)
    }, 'Pipeline')
    def __init__(
            self,
            time_col,
            target_col,
            lags,
            metric=mae,
            metric_less_is_better=True,
            configs=None,
            random_state=0,
            verbose=True
    ):
        self.logger = Logger(name='PipelineTS', verbose=verbose)

        self.target_col = target_col
        self.time_col = time_col
        self.lags = lags
        self.metric = metric
        self.metric_less_is_better = metric_less_is_better
        self.random_state = random_state
        self.configs = configs

        self.models = dict(MODELS)

        for model_name, model in self.models.items():
            model_kwargs = generate_function_kwargs(
                model,
                time_col=self.time_col,
                target_col=self.target_col,
                lags=self.lags,
                random_state=random_state
            )

            if self.configs is not None and \
                    self.configs.get_configs(model_name).get('init_configs') is not None:
                model_kwargs.update(self.configs.get_configs(model_name).get('init_configs'))

            self.models[model_name] = model(**model_kwargs)

        self.leader_board = None

    def fit_and_eval(self, df, valid_df=None):
        res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])

        if valid_df is not None:
            assert df.columns.tolist() == valid_df.columns.tolist()

        for model_name, model in self.models.items():
            self.logger.print(f"fitting and evaluating {model_name}...")

            tik = time.time()
            model.fit(df)
            tok = time.time()
            train_cost = tok - tik

            tik = time.time()
            eval_res = model.predict(valid_df.shape[0])
            metric = self.metric(valid_df[self.target_col].values, eval_res[self.target_col].values)
            tok = time.time()
            eval_cost = tok - tik
            res = pd.concat(
                (res, pd.DataFrame([[model_name, train_cost, eval_cost, metric]],
                                   columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])),
                axis=0, ignore_index=True)

        self.leader_board = res.sort_values(by='metric', ascending=self.metric_less_is_better).reset_index(drop=True)
        return self.leader_board

    def predict(self, model, n):
        return self.models[model].predict(n)

