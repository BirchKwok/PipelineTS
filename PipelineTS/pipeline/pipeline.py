import sys
from copy import deepcopy
import gc

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from frozendict import frozendict

from spinesTS.base._torch_mixin import detect_available_device
from spinesTS.metrics import mae
from spinesTS.utils import func_has_params
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.preprocessing import gc_collector
from spinesUtils.asserts import (
    check_obj_is_function,
    augmented_isinstance,
    raise_if
)
from spinesUtils.utils import (
    Logger,
    drop_duplicates_with_order,
    Timer
)

from PipelineTS.metrics import quantile_acc
from PipelineTS.pipeline.pipeline_models import get_all_available_models
from PipelineTS.pipeline.pipeline_configs import PipelineConfigs


# TODO: 传入数据，进行数据采集周期检验，看看是否有漏数据，如果有，进行插值（可选），如果有异常值，进行噪音去除（可选）

def update_dict_without_conflict(dict_a, dict_b):
    for i in dict_b:
        if i not in dict_a:
            dict_a[i] = dict_b[i]
    return dict_a


class ModelPipeline:
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'with_quantile_prediction': bool,
        'include_models': (None, list),
        'exclude_models': (None, list),
        'metric_less_is_better': bool,
        'configs': (None, PipelineConfigs),
        'random_state': (int, None),
        'verbose': (bool, int),
        'include_init_config_model': bool,
        'use_standard_scale': (bool, None),
        'accelerator': (str, None),
        'cv': int,
    }, 'Pipeline')
    @ParameterValuesAssert({
        'metric': lambda s: check_obj_is_function(s),
        'accelerator': (
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
            include_models=None,
            exclude_models=None,
            metric=mae,
            metric_less_is_better=True,
            configs=None,
            random_state=0,
            verbose=True,
            include_init_config_model=False,
            use_standard_scale=False,  # False for MinMaxScaler, True for StandardScaler, None means no data be scaled
            accelerator='auto',
            cv=5,
            **model_init_kwargs
    ):
        raise_if(ValueError, include_models is not None and exclude_models is not None,
                 "include_models and exclude_models can not be set at the same time.")

        if with_quantile_prediction:
            raise_if(ValueError, cv <= 1, "if with_quantile_prediction is True, cv must be greater than 1.")

        self._available_models = get_all_available_models()

        raise_if(ValueError, exclude_models is not None and
                 (not all([i in self._available_models for i in exclude_models])),
                 "exclude_models must be None or in the list of models.")

        raise_if(ValueError, include_models is not None and
                 (not all([i in self._available_models for i in include_models])),
                 "include_models must be None or in the list of models.")

        if exclude_models is not None:
            self._available_models = dict(self._available_models)

            for em in exclude_models:
                del self._available_models[em]
            self._available_models = frozendict(self._available_models)

        self.logger = Logger(name='PipelineTS', verbose=verbose)

        self.target_col = target_col
        self.time_col = time_col
        self.lags = lags
        self.metric = metric
        self.metric_less_is_better = metric_less_is_better
        self.random_state = random_state
        self.configs = configs
        self._given_models = include_models
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
        self.accelerator = accelerator
        self.cv = cv

        self._timer = Timer()

        self._model_init_kwargs = {}

        model_init_kwargs = update_dict_without_conflict(model_init_kwargs,
                                                         {
                                                             'multi_output_model__verbose': -1,
                                                             'multi_step_model__verbose': -1,
                                                             'lightgbm__verbose': -1,
                                                             'wide_gbrt__verbose': -1,
                                                             'catboost__verbose': False,
                                                             'xgboost__verbose': 0
                                                         })

        for k, v in model_init_kwargs.items():
            raise_if(ValueError, '__' not in k,
                     f"{k} must has double underline.")

            raise_if(ValueError, k.split('__')[0] not in self._available_models,
                     f"{k.split('__')[0]} is not a valid model name")
            self._model_init_kwargs[k] = v

        sys.stderr.write(detect_available_device(self.accelerator)[1]+'\n\n')

    def _initial_models(self):
        initial_models = []
        ms = tuple(sorted(self._available_models.items(), key=lambda s: s[0])) if self._given_models is None else (
            drop_duplicates_with_order([(k, self._available_models[k]) for k in self._given_models]))

        # 模型训练顺序
        for (model_name, model) in ms:
            model_kwargs = self._fill_func_params(
                func=model,
                time_col=self.time_col,
                target_col=self.target_col,
                lags=self.lags,
                random_state=self.random_state,
                quantile=0.9 if self.with_quantile_prediction else None,
                accelerator=self.accelerator
            )

            if len(self._model_init_kwargs) > 0:
                for k, v in self._model_init_kwargs.items():
                    if k.split('__')[0] == model_name:
                        model_kwargs[k[len(model_name) + 2:]] = v

            if self.configs is not None:
                include_in_configs = False
                for (model_name_in_config, model_name_after_rename_in_config, model_configs_in_config) \
                        in self.configs.configs:
                    if model_name_in_config == model_name:
                        include_in_configs = True
                        new_model_kwargs = deepcopy(model_kwargs)

                        new_model_kwargs.update(
                            self.configs.get_configs(model_name_after_rename_in_config).get('init_configs')
                        )

                        initial_models.append([model_name_after_rename_in_config, model(**new_model_kwargs)])

                        if self.include_init_config_model:
                            if [model_name, model(**model_kwargs)] not in initial_models:
                                initial_models.append([model_name, model(**model_kwargs)])

                if not include_in_configs:
                    initial_models.append([model_name, model(**model_kwargs)])
            else:
                initial_models.append([model_name, model(**model_kwargs)])

        return initial_models

    @classmethod
    def list_all_available_models(cls):
        return list(get_all_available_models().keys())

    def _scale_data(self, data, valid_data=None, refit_scaler=True):
        df, valid_df = data.copy(), valid_data  # valid_data will not be deep copy in this step

        if refit_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        if scaler is not None:
            df[self.target_col] = scaler.fit_transform(
                df[self.target_col].values.reshape(-1, 1)
            ).squeeze()

            if valid_data is not None:
                valid_df = valid_data.copy()
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

    @staticmethod
    def _fill_func_params(func, **kwargs):
        init_kwargs = {}

        for i in kwargs:
            if func_has_params(func, i):
                init_kwargs.update({i: kwargs[i]})

        return init_kwargs

    @gc_collector(3)
    def _fit(self, model_name_after_rename, model, train_df, valid_df, res_df, use_scaler=True):
        self._timer.start()

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

        train_cost = self._timer.last_timestamp_diff()

        self._timer.middle_point()
        gc.collect()
        gc.garbage.clear()

        self._timer.sleep(3)
        # -------------------- predicting -------------------------
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

        eval_cost = self._timer.last_timestamp_diff()

        del eval_res

        gc.collect()
        self._timer.sleep(3)

        self._timer.clear()  # 重置计时器

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
        'valid_data': (None, pd.DataFrame)
    })
    def fit(self, data, valid_data=None):
        """fit all models"""

        if data.shape[0] <= self.lags:
            raise ValueError(f'length of df must be greater than lags, df length = {data.shape[0]}, lags = {self.lags}')

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

        self.leader_board_ = res.sort_values(
            by='metric', ascending=self.metric_less_is_better
        ).reset_index(drop=True)

        self.leader_board_.columns.name = 'Leaderboard'

        self.best_model_ = self.get_model(self.leader_board_.iloc[0, :]['model'])

        del data, valid_data, df, valid_df, res
        gc.collect()
        gc.garbage.clear()

        return self.leader_board_

    @ParameterTypeAssert({
        'model_name': (str, None)
    })
    def get_model(self, model_name=None):
        """By default, return the best model"""
        if model_name is None:
            return self.best_model_
        else:
            for (md_name, model) in self.models_:
                if model_name == md_name:
                    return model

    @ParameterTypeAssert({
        'model_name': (str, None)
    })
    def get_model_all_configs(self, model_name=None):
        """By default, return the best model's configure"""
        if model_name is None:
            return self.best_model_.all_configs
        else:
            for (md_name, model) in self.models_:
                if model_name == md_name:
                    return model.all_configs

    @ParameterTypeAssert({
        'n': int,
        'model_name': (None, str)
    })
    def predict(self, n, series=None, model_name=None):
        """By default, the best model is used for prediction

        :parameter
        n: predict steps
        series: the sequence to predict from the last time point in the sequence,
                accepting only a pandas DataFrame type
        model_name: str, model's name, specifying the model used, default None

        :return
        pd.DataFrame
        """
        if model_name is not None:
            res = self.get_model(model_name).predict(n, series=series)
        else:
            res = self.best_model_.predict(n, series=series)

        for i in res.columns:
            if i.startswith(self.target_col):
                res = self._inverse_data(res, columns=i)

        return res
