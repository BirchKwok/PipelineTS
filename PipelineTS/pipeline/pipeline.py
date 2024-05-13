import time
from copy import deepcopy
import gc

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
from frozendict import frozendict

from PipelineTS.spinesTS.base import detect_available_device
from PipelineTS.spinesTS.metrics import mae
from spinesUtils.preprocessing import gc_collector
from spinesUtils.asserts import (
    ParameterTypeAssert,
    ParameterValuesAssert,
    check_obj_is_function,
    augmented_isinstance,
    raise_if,
    raise_if_not,
    check_has_param
)
from spinesUtils.logging import Logger
from spinesUtils.timer import Timer

# All model classes in PipelineTS are subclasses of the IntervalEstimationMixin class.
from PipelineTS.base.base import IntervalEstimationMixin
from PipelineTS.metrics import quantile_acc
from PipelineTS.pipeline.pipeline_models import get_all_available_models, get_all_model_class_name
from PipelineTS.pipeline.pipeline_configs import PipelineConfigs
from PipelineTS.utils import update_dict_without_conflict, check_time_col_is_timestamp
from PipelineTS.base.base_utils import generate_models_set


# TODO: 传入数据，进行数据采集周期检验，看看是否有漏数据，如果有，进行插值（可选），如果有异常值，进行噪音去除（可选）


class ModelPipeline:
    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'lags': int,
        'quantile': (None, float),
        'exclude_models': (None, list, str),
        'metric_less_is_better': bool,
        'configs': (None, PipelineConfigs),
        'random_state': (int, None),
        'include_init_config_model': bool,
        'accelerator': (str, None),
        'cv': int,
        'gbdt_differential_n': int
    }, 'Pipeline')
    @ParameterValuesAssert({
        'metric': lambda s: check_obj_is_function(s),
        'scaler': lambda s: augmented_isinstance(s, (TransformerMixin, None, bool)),
        'accelerator': (
                lambda s:
                s in ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto", "cuda")
                or augmented_isinstance(s, None)
        ),
        'include_models': (
                lambda s:
                s in ('light', 'all', 'nn', 'ml') or
                augmented_isinstance(s, (None, list, str)) or
                issubclass(s, IntervalEstimationMixin)
        )
    }, 'Pipeline')
    def __init__(
            self,
            time_col,
            target_col,
            lags,
            quantile=None,  # the quantile prediction switch
            include_models='light',
            exclude_models=None,
            metric=mae,
            metric_less_is_better=True,
            configs=None,
            random_state=0,
            include_init_config_model=False,
            scaler=True,  # whether to use the scaler, default is True, use MinMaxScaler
            accelerator='auto',
            cv=5,
            gbdt_differential_n=1,
            **model_init_kwargs
    ):
        """
        Initialize the ModelPipeline.

        Parameters
        ----------
        time_col : str
            Name of the column representing time.
        target_col : str
            Name of the column containing the target variable.
        lags : int
            Number of lagged time steps for modeling.
        quantile : float, optional, default: None
            Quantile value of interval prediction.
        include_models : {'light', 'all', 'nn', 'ml'} or list or None or a available model of PipelineTS, optional, default: 'light'
            Models to include in the pipeline.
        exclude_models : list or None or str, optional, default: None
            Models to exclude from the pipeline.
        metric : callable, optional, default: Mean Absolute Error (mae)
            Evaluation metric function.
        metric_less_is_better : bool, optional, default: True
            Whether lower metric values are better.
        configs : PipelineConfigs or None, optional, default: None
            Configuration object for the pipeline.
        random_state : int, optional, default: 0
            Seed for random number generation.
        verbose : bool or int, optional, default: True
            Verbosity level.
        include_init_config_model : bool, optional, default: False
            Include models with initial configuration.
        scaler : bool or None or transformer that has the type of sklearn.base.TransformerMixin, optional, default: True
            Use scaler for data scaling, True for MinMaxScaler, None means no scaling.
            Alternatively, you can specify your own transformer.
        accelerator : {'cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto', 'cuda'} or None, optional, default: 'auto'
            Hardware accelerator type.
        cv : int, optional, default: 5
            Number of cross-validation folds.
        gbdt_differential_n : int, optional, default: 1
            The number of differencing operations to apply to the target variable.
        **model_init_kwargs
            Additional keyword arguments for model initialization.

        Raises
        ------
        ValueError
            If include_models and exclude_models are set simultaneously.
            If quantile is not None and cv is not greater than 1.
            If exclude_models contain invalid model names.
            If include_models contain invalid model names.
            If model names in model_init_kwargs do not match available models.

        Notes
        -----
        The include_models parameter supports predefined sets ('light', 'all', 'nn', 'ml') or a custom list of model names.
        The accelerator parameter supports values ('cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps', 'auto', 'cuda') or None.
        """
        raise_if(ValueError, include_models is not None and exclude_models is not None,
                 "include_models and exclude_models can not be set at the same time.")

        if augmented_isinstance(exclude_models, str):
            exclude_models = [exclude_models]

        if include_models == 'light':
            include_models = ['d_linear', 'lightgbm', 'multi_step_model', 'n_hits', 'n_linear',
                              'random_forest', 'regressor_chain', 'tcn', 'xgboost']
        elif include_models == 'all':
            include_models = None
        elif include_models == 'nn':
            include_models = ['d_linear', 'gau', 'n_beats', 'n_hits', 'n_linear', 'tcn', 'tft',
                              'patch_rnn', 'stacking_rnn', 'tide', 'time2vec', 'transformer']
        elif include_models == 'ml':
            include_models = ['catboost', 'lightgbm', 'multi_output_model',
                              'multi_step_model', 'random_forest', 'wide_gbrt', 'xgboost']
        elif isinstance(include_models, str):
            raise_if_not(ValueError, include_models in ModelPipeline.list_all_available_models(),
                         f"{include_models} is not a available model name. ")
            include_models = [include_models]
        elif not isinstance(include_models, (list, str)) and issubclass(include_models, IntervalEstimationMixin):
            include_models = [include_models]
        else:
            include_models = include_models

        if quantile:
            raise_if(ValueError, cv <= 1, "if quantile is not None, cv must be greater than 1.")

        self._available_models = get_all_available_models()

        raise_if(ValueError, exclude_models is not None and
                 (not all([i in self._available_models for i in exclude_models])),
                 "exclude_models must be None or in the list of models.")

        raise_if(ValueError, include_models is not None and
                 (not all([i in self._available_models or
                           issubclass(i, IntervalEstimationMixin) for i in include_models])),
                 "include_models must be None or in the list of models or a available PipelineTS model.")

        if exclude_models is not None:
            self._available_models = dict(self._available_models)

            for em in exclude_models:
                del self._available_models[em]
            self._available_models = frozendict(self._available_models)

        self.logger = Logger(name='PipelineTS')

        self.target_col = target_col
        self.time_col = time_col
        self.lags = lags
        self.metric = metric
        self.metric_less_is_better = metric_less_is_better
        self.random_state = random_state
        self.configs = configs
        self._given_models = include_models
        self.quantile = quantile

        self.include_init_config_model = include_init_config_model

        if augmented_isinstance(scaler, bool) and scaler is True:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = scaler if scaler is not False else None

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

            raise_if(ValueError, k.split('__')[0] not in self._available_models and k.split('__')[0]
                     not in get_all_model_class_name(),
                     f"{k.split('__')[0]} is not a valid model name")
            self._model_init_kwargs[k] = v

        self._compute_device_msg = detect_available_device(self.accelerator)[1] + '\n\n'

        self.gbdt_differential_n = gbdt_differential_n

    def _initial_models(self):
        initial_models = []
        ms = generate_models_set(self._available_models, self._given_models)

        # 模型训练顺序
        for (model_name, model) in ms:
            model_kwargs = self._fill_func_params(
                func=model,
                time_col=self.time_col,
                target_col=self.target_col,
                lags=self.lags,
                random_state=self.random_state,
                quantile=self.quantile,
                accelerator=self.accelerator,
                differential_n=self.gbdt_differential_n
            )

            # Populate model initialization parameters specified in double underscore format.
            # This scenario takes precedence over keyword arguments.
            if len(self._model_init_kwargs) > 0:
                for k, v in self._model_init_kwargs.items():
                    if k.split('__')[0] == model_name:
                        model_kwargs[k[len(model_name) + 2:]] = v

            # The PipelineConfigs class has the highest configuration authority.
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
        """
        Get a list of all available model names in the ModelPipeline.

        Returns
        -------
        models : list of str
            List of model names available for use in the pipeline.

        Example
        -------
        >>> ModelPipeline.list_all_available_models()
        ['catboost',
         'd_linear',
         'gau',
         'lightgbm',
         'multi_output_model',
         'multi_step_model',
         'n_beats',
         'n_hits',
         'n_linear',
         'random_forest',
         'regressor_chain',
         'patch_rnn',
         'stacking_rnn',
         'tcn',
         'tft',
         'tide',
         'time2vec',
         'transformer',
         'xgboost']
        """
        return sorted(list(get_all_available_models().keys()))

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
            if check_has_param(func, i):
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

        model_kwargs = self._fill_func_params(func=model.fit, data=train_df, fit_kwargs=fit_kwargs, cv=self.cv,
                                              valid_data=valid_df)
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

        if check_has_param(model.predict, 'predict_kwargs'):
            eval_res = model.predict(valid_df.shape[0], data=valid_df, predict_kwargs=predict_kwargs)
        else:
            eval_res = model.predict(valid_df.shape[0])

        if use_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        yt = valid_df[self.target_col].values
        yp = eval_res[self.target_col].values

        if self.quantile:
            left_pred = eval_res[f"{self.target_col}_lower"].values
            right_pred = eval_res[f"{self.target_col}_upper"].values

        if scaler is not None:
            yt = scaler.inverse_transform(yt.reshape(-1, 1)).squeeze()
            yp = scaler.inverse_transform(yp.reshape(-1, 1)).squeeze()

            if self.quantile:
                left_pred = scaler.inverse_transform(left_pred.reshape(-1, 1)).squeeze()
                right_pred = scaler.inverse_transform(right_pred.reshape(-1, 1)).squeeze()
                res_quantile_acc = quantile_acc(yt, left_pred, right_pred)

        metric = self.metric(yt, yp)

        eval_cost = self._timer.last_timestamp_diff()

        del eval_res

        gc.collect()
        self._timer.sleep(3)

        self._timer.clear()  # 重置计时器

        if self.quantile:
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
        """
        Fit all models in the ModelPipeline to the provided training data.

        Parameters
        ----------
        data : pd.DataFrame
            The training data containing historical information.
        valid_data : pd.DataFrame or None, optional, default: None
            Validation data for evaluating model performance.

        Returns
        -------
        leaderboard : pd.DataFrame
            Leaderboard containing model evaluation metrics, sorted by model performance.

        Raises
        ------
        ValueError
            If the length of data is less than or equal to lags.
        AssertionError
            If columns of data and valid_data do not match.

        Example
        -------
        >>> pipeline = ModelPipeline(time_col='timestamp', target_col='value', lags=10)
        >>> leaderboard = pipeline.fit(train_data, valid_data)
        >>> print(leaderboard)
           Leaderboard         model  train_cost(s)  eval_cost(s)    metric
        0           0    lightgbm_0       2.567801      0.978624  0.123456
        1           1    xgboost_1       3.123456      1.234567  0.456789
        2           2  random_forest       1.987654      0.876543  0.987654
        ...         ...            ...            ...           ...       ...

        Notes
        -----
        - The fit function trains all models in the pipeline using the provided training data.
        - The optional valid_data parameter allows for model evaluation on a separate validation dataset.
        - The resulting leaderboard provides a ranked list of models based on the specified evaluation metric.
        """
        self.logger.info('Information about the device used for computation:\n'+self._compute_device_msg)
        time.sleep(0.5)

        check_time_col_is_timestamp(data, self.time_col)

        if data.shape[0] <= self.lags:
            raise ValueError(f'length of df must be greater than lags, df length = {data.shape[0]}, lags = {self.lags}')

        if valid_data is not None:
            raise_if_not(AssertionError, data.columns.tolist() == valid_data.columns.tolist(),
                         "columns of data and valid_data do not match.")
            check_time_col_is_timestamp(valid_data, self.time_col)

            df, valid_df = data.copy(), valid_data.copy()
        else:
            df, valid_df = data.copy(), data.iloc[-(2 * self.lags):, :]

        # 如果指定scaler，此语句会对数据缩放
        df, valid_df = self._scale_data(df, valid_df, refit_scaler=True)

        res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])
        if self.quantile:
            res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric', 'quantile_acc'])

        models = self._initial_models()
        self.logger.info(f"There are a total of {len(models)} models to be trained.")

        for idx, (model_name_after_rename, model) in enumerate(models):
            self.logger.info(f"[model {idx:>{len(str(len(models)))}d}] fitting and "
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
        """
        Retrieve a trained model from the ModelPipeline.

        Parameters
        ----------
        model_name : str or None, optional, default: None
            Name of the model to retrieve. If None, returns the best model.

        Returns
        -------
        model : Model
            The trained model corresponding to the specified model_name. If model_name is None, returns the best model.

        Example
        -------
        >>> pipeline = ModelPipeline(time_col='timestamp', target_col='value', lags=10)
        >>> pipeline.fit(train_data, valid_data)
        >>> best_model = pipeline.get_model()
        >>> specific_model = pipeline.get_model('lightgbm_0')

        Notes
        -----
        - If model_name is not provided, the function returns the best-performing model based on the leaderboard.
        - The function allows retrieving a specific trained model by providing its unique name (e.g., 'lightgbm_0').
        """
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
        """
        Retrieve the configuration details of a trained model from the ModelPipeline.

        Parameters
        ----------
        model_name : str or None, optional, default: None
            Name of the model to retrieve configuration details. If None, returns the configuration details of the best model.

        Returns
        -------
        configs : dict or None
            A dictionary containing the configuration details of the specified model. If model_name is None, returns the configuration details of the best model.

        Example
        -------
        >>> pipeline = ModelPipeline(time_col='timestamp', target_col='value', lags=10)
        >>> pipeline.fit(train_data, valid_data)
        >>> best_model_configs = pipeline.get_model_all_configs()
        >>> specific_model_configs = pipeline.get_model_all_configs('lightgbm_0')

        Notes
        -----
        - If model_name is not provided, the function returns the configuration details of the best-performing model.
        - The function allows retrieving configuration details for a specific trained model by providing its unique name (e.g., 'lightgbm_0').
        """
        if model_name is None:
            return self.best_model_.all_configs
        else:
            for (md_name, model) in self.models_:
                if model_name == md_name:
                    return model.all_configs

    @ParameterTypeAssert({
        'n': int,
        'data': (pd.DataFrame, None),
        'model_name': (None, str)
    })
    def predict(self, n, data=None, model_name=None):
        """
        Generate predictions using the trained models in the ModelPipeline.

        Parameters
        ----------
        n : int
            Predictive steps, indicating the number of time steps to forecast into the future.
        data : pd.DataFrame or None, optional, default: None
            The input data for making predictions. If None, the last available data in the pipeline will be used.
        model_name : str or None, optional, default: None
            Model name to use for predictions. If None, the best model will be used.

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame containing the predicted values for the specified model or the best model.

        Example
        -------
        >>> pipeline = ModelPipeline(time_col='timestamp', target_col='value', lags=10)
        >>> pipeline.fit(train_data, valid_data)
        >>> predictions_best_model = pipeline.predict(n=5)
        >>> predictions_specific_model = pipeline.predict(n=5, model_name='lightgbm_0', data=test_data)

        Notes
        -----
        - The predict function generates future predictions using the trained models in the pipeline.
        - If data is not provided, the function uses the last available data in the pipeline.
        - If model_name is not provided, the function uses the best-performing model based on the leaderboard.
        """
        df = None
        if data is not None:
            df = data.copy()
            if self.scaler is not None:
                df[self.target_col] = self.scaler.transform(df[self.target_col].values.reshape(-1, 1)).squeeze()

        if model_name is not None:
            if check_has_param(self.get_model(model_name).predict, 'data'):

                res = self.get_model(model_name).predict(n, data=df)
            else:
                res = self.get_model(model_name).predict(n)
        else:
            if check_has_param(self.get_model(model_name).predict, 'data'):

                res = self.best_model_.predict(n, data=df)
            else:
                res = self.best_model_.predict(n)

        for i in res.columns:
            if i.startswith(self.target_col):
                res = self._inverse_data(res, columns=i)

        return res
