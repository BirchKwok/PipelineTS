import time
from copy import deepcopy

from frozendict import frozendict
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from spinesTS.metrics import mae
from spinesTS.utils import func_has_params
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.asserts import check_obj_is_function, augmented_isinstance
from spinesUtils.utils import (
    Logger,
    drop_duplicates_with_order,
    reindex_iterable_object,
    is_in_ipython
)

from tabulate import tabulate
from IPython.display import display

from PipelineTS.statistic_model import *
from PipelineTS.ml_model import *
from PipelineTS.nn_model import *
from PipelineTS.metrics import quantile_acc
from PipelineTS.nn_model.sps_nn_model_base import SpinesNNModelMixin

# TODO: 传入数据，进行数据采集周期检验，看看是否有漏数据，如果有，进行插值（可选），如果有异常值，进行噪音去除（可选）

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
    'tft': TFTModel,
    'gau': GAUModel,
    'stacking_rnn': StackingRNNModel,
    'time2vec': Time2VecModel,
    'multi_output_model': MultiOutputRegressorModel,
    'multi_step_model': MultiStepRegressorModel,
    'transformer': TransformerModel,
    'random_forest': RandomForestModel,
    'tide': TiDEModel
})


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


class ModelPipeline:
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'with_quantile_prediction': bool,
        'models': (None, list),
        'metric_less_is_better': bool,
        'configs': (None, PipelineConfigs),
        'random_state': (int, None),
        'verbose': (bool, int),
        'include_init_config_model': bool,
        'use_standard_scale': (bool, None),
        'device': (str, None),
        'cv': int
    }, 'Pipeline')
    @ParameterValuesAssert({
        'metric': lambda s: check_obj_is_function(s),
        'device': (
                lambda s: s in ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto", "cuda")
                          or augmented_isinstance(s, None)
        )
    }, 'Pipeline')
    def __init__(
            self,
            time_col,
            target_col,
            lags,
            with_quantile_prediction=False,  # the quantile prediction switch
            models=None,
            metric=mae,
            metric_less_is_better=True,
            configs=None,
            random_state=0,
            verbose=True,
            include_init_config_model=False,
            use_standard_scale=False,  # otherwise, MinMaxScaler
            device=None,
            cv=5
    ):
        if with_quantile_prediction:
            assert cv > 1, "if with_quantile_prediction is True, cv must be greater than 1."

        self.logger = Logger(name='PipelineTS', verbose=verbose)

        self.target_col = target_col
        self.time_col = time_col
        self.lags = lags
        self.metric = metric
        self.metric_less_is_better = metric_less_is_better
        self.random_state = random_state
        self.configs = configs
        self._given_models = models
        self.with_quantile_prediction = with_quantile_prediction

        self.include_init_config_model = include_init_config_model

        if use_standard_scale is not None:
            self.scaler = StandardScaler() if use_standard_scale else MinMaxScaler()
        else:
            self.scaler = use_standard_scale

        self._temp_scaler = deepcopy(self.scaler)

        self.models_ = []
        self.leader_board_ = None
        self.best_model_ = None
        self.device = device
        self.cv = cv

    @staticmethod
    def _device_setting(model, device):
        if isinstance(model, SpinesNNModelMixin):
            if device in ('cuda', 'cpu', 'mps'):
                device = device
            else:
                device = 'cpu'
        else:
            device = device

        return device

    def _initial_models(self):
        initial_models = []
        ms = tuple(sorted(MODELS.items(), key=lambda s: s[0])) if self._given_models is None else (
            drop_duplicates_with_order([(k, MODELS[k]) for k in self._given_models]))

        # 模型训练顺序
        for (model_name, model) in ms:
            model_kwargs = self._fill_func_params(
                func=model,
                time_col=self.time_col,
                target_col=self.target_col,
                lags=self.lags,
                random_state=self.random_state,
                quantile=0.9 if self.with_quantile_prediction else None,
                device=self._device_setting(model, self.device)
            )

            continue_signal = False  # 是否跳过添加默认模型
            if self.configs is not None:
                for (model_name_in_config, model_name_after_rename_in_config, model_configs_in_config) \
                        in self.configs.configs:
                    if model_name_in_config == model_name:
                        new_model_kwargs = deepcopy(model_kwargs)

                        new_model_kwargs.update(
                            self.configs.get_configs(model_name_after_rename_in_config).get('init_configs')
                        )

                        if not self.include_init_config_model:
                            initial_models.append([model_name_after_rename_in_config, model(**new_model_kwargs)])

                        else:
                            if [model_name, model(**new_model_kwargs)] not in initial_models:
                                initial_models.append([model_name, model(**new_model_kwargs)])
                            initial_models.append([model_name_after_rename_in_config, model(**new_model_kwargs)])

                        continue_signal = True

                if continue_signal:
                    continue

                initial_models.append([model_name, model(**model_kwargs)])
            else:
                initial_models.append([model_name, model(**model_kwargs)])

        return initial_models

    @classmethod
    def list_models(cls):
        return list(MODELS.keys())

    def _scale_data(self, df, valid_df, refit_scaler=True):
        if refit_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        if scaler is not None:
            df[self.target_col] = scaler.fit_transform(
                df[self.target_col].values.reshape(-1, 1)
            ).squeeze()

            if valid_df is not None:
                valid_df[self.target_col] = scaler.transform(
                    valid_df[self.target_col].values.reshape(-1, 1)).squeeze()

        return df, valid_df

    def _inverse_data(self, df, columns=None, use_scaler=True):
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        if columns is None:
            columns = self.target_col

        if scaler is not None:
            df[columns] = scaler.inverse_transform(
                df[columns].values.reshape(-1, 1)
            ).squeeze()

        return df

    def _fill_func_params(self, func, **kwargs):
        init_kwargs = {}

        for i in kwargs:
            if func_has_params(func, i):
                init_kwargs.update({i: kwargs[i]})

        return init_kwargs

    def _fit(self, model_name_after_rename, model, train_df, valid_df, res_df, use_scaler=True):
        tik = time.time()

        # -------------------- fitting -------------------------
        if self.configs is not None:
            if self.configs.get_configs(model_name_after_rename):
                fit_kwargs = self.configs.get_configs(model_name_after_rename).get('fit_configs')
            else:
                fit_kwargs = {}
        else:
            fit_kwargs = {}

        model_kwargs = self._fill_func_params(func=model.fit, data=train_df, fit_kwargs=fit_kwargs, cv=self.cv)
        model.fit(**model_kwargs)

        tok = time.time()
        train_cost = tok - tik

        # -------------------- predicting -------------------------
        tik = time.time()

        if self.configs is not None:
            if self.configs.get_configs(model_name_after_rename):
                predict_kwargs = self.configs.get_configs(model_name_after_rename).get('predict_configs')
            else:
                predict_kwargs = {}
        else:
            predict_kwargs = {}

        if func_has_params(model.predict, 'predict_kwargs'):
            eval_res = model.predict(valid_df.shape[0], predict_kwargs=predict_kwargs)
        else:
            eval_res = model.predict(valid_df.shape[0])

        if use_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        metric = self.metric(
            scaler.inverse_transform(
                valid_df[self.target_col].values.reshape(-1, 1)
            ).squeeze(),
            scaler.inverse_transform(
                eval_res[self.target_col].values.reshape(-1, 1)
            ).squeeze()

        )

        if self.with_quantile_prediction:
            res_quantile_acc = quantile_acc(
                yt=scaler.inverse_transform(
                    valid_df[self.target_col].values.reshape(-1, 1)
                ).squeeze(),
                left_pred=scaler.inverse_transform(
                    eval_res[f"{self.target_col}_lower"].values.reshape(-1, 1)
                ).squeeze(),
                right_pred=scaler.inverse_transform(
                    eval_res[f"{self.target_col}_upper"].values.reshape(-1, 1)
                ).squeeze()
            )

        tok = time.time()
        eval_cost = tok - tik

        if self.with_quantile_prediction:
            res_df = pd.concat(
                (res_df, pd.DataFrame(
                    [[model_name_after_rename, train_cost, eval_cost, metric, res_quantile_acc]],
                    columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric', 'quantile_acc'])),
                axis=0, ignore_index=True)
        else:
            res_df = pd.concat(
                (res_df, pd.DataFrame([[model_name_after_rename, train_cost, eval_cost, metric]],
                                      columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])),
                axis=0, ignore_index=True)

        return model_name_after_rename, model, res_df

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'valid_data': (None, pd.DataFrame),
        'update_leaderboard': bool
    })
    def fit(self, data, valid_data=None, update_leaderboard=True):
        """fit all models"""

        if data.shape[0] <= self.lags:
            raise ValueError(f'length of df must be greater than lags, df length = {df.shape[0]}, lags = {self.lags}')

        if valid_data is not None:
            assert data.columns.tolist() == valid_data.columns.tolist()
            df, valid_df = data.copy(), valid_data.copy()
        else:
            df, valid_df = data.iloc[:-self.lags, :], data.iloc[-self.lags:, :]
            df, valid_df = df.copy(), valid_df.copy()

        # 如果指定use_standard_scale，此语句会对数据缩放
        df, valid_df = self._scale_data(df, valid_df, refit_scaler=True)

        if self.with_quantile_prediction:
            res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric', 'quantile_acc'])
        else:
            res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])

        models = self._initial_models()
        self.logger.print(f"There are a total of {len(models)} models to be trained.")

        for idx, (model_name_after_rename, model) in enumerate(models):
            self.logger.print(f"[model {idx:>{len(str(len(models)))}d}] fitting and "
                              f"evaluating {model_name_after_rename}...")

            model_name_after_rename, model, res = self._fit(
                model_name_after_rename, model, df, valid_df, res,
                use_scaler=True
            )

            self.models_.append((model_name_after_rename, model))

        if update_leaderboard:
            self.leader_board_ = res.sort_values(
                by='metric', ascending=self.metric_less_is_better
            ).reset_index(drop=True)

            self.leader_board_.columns.name = 'Leaderboard'

        self.best_model_ = self.get_models(self.leader_board_.iloc[0, :]['model'])
        return self.leader_board_

    @ParameterTypeAssert({
        'model_name': (str, None)
    })
    def get_models(self, model_name=None):
        if model_name is None:
            return self.best_model_
        else:
            for (md_name, model) in self.models_:
                if model_name == md_name:
                    return model

    @ParameterTypeAssert({
        'n': int,
        'model_name': (None, str)
    })
    def predict(self, n, model_name=None):
        """By default, the best model is used for prediction

        :parameter
        n: predict steps
        model_name: str, model's name, specifying the model used, default None

        :return
        pd.DataFrame
        """
        if model_name is not None:
            res = self.get_models(model_name).predict(n)
        else:
            res = self.best_model_.predict(n)

        for i in res.columns:
            if i.startswith(self.target_col):
                res = self._inverse_data(res, columns=i)

        return res
