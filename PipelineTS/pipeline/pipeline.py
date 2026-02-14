from copy import deepcopy
import gc
import time
import traceback

import numpy as np
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
        'gbdt_differential_n': int,
        'time_limit': (int, float, None),
        'id_col': (str, None),
        'per_model_lags': (dict, None),
    }, 'ModelPipeline')
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
    }, 'ModelPipeline')
    def __init__(
            self,
            time_col,
            target_col,
            lags,
            quantile=None,  # the quantile prediction switch
            feature_cols=None,  # input feature columns for multivariate models
            id_col=None,  # series identifier column for multi-series (panel) data
            known_covariates=None,  # columns available for both history and future
            past_covariates=None,  # columns available only historically
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
            gbdt_differential_n=0,
            time_limit=None,
            per_model_lags=None,
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
            include_models = ['d_linear', 'itransformer', 'multi_output_model', 'multi_step_model',
                              'n_hits', 'n_linear', 'patch_rnn', 'torch_bagging_forest',
                              'regressor_chain', 'tide', 'transformer']
        elif include_models == 'all':
            include_models = None
        elif include_models == 'nn':
            include_models = ['d_linear', 'deepar', 'gau', 'n_beats', 'n_hits', 'n_linear', 'tcn', 'tft',
                              'patch_rnn', 'stacking_rnn', 'tide', 'time2vec', 'transformer',
                              'itransformer', 'srs_net']
        elif include_models == 'ml':
            include_models = ['torch_boosting_forest', 'torch_bagging_forest', 'multi_output_model',
                              'multi_step_model', 'deep_forest', 'wide_gbrt']
        elif isinstance(include_models, str):
            raise_if_not(ValueError, include_models in ModelPipeline.list_all_available_models(),
                         f"{include_models} is not a available model name. ")
            include_models = [include_models]
        elif include_models is not None and not isinstance(include_models, (list, str)) and issubclass(include_models, IntervalEstimationMixin):
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

        self.logger = Logger(name='ModelPipeline')

        self.target_col = target_col
        self.time_col = time_col
        self.lags = lags
        self.feature_cols = feature_cols
        self.id_col = id_col
        self.known_covariates = known_covariates or []
        self.past_covariates = past_covariates or []
        self._panel_scalers = {}  # per-series scalers for multi-series mode
        self._temp_panel_scalers = {}  # temp scalers for CV folds
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
        self.time_limit = time_limit

        self._timer = Timer()
        self._fit_start_time = None
        self._failed_models = []
        self._skipped_models = []
        self._on_model_complete_callback = None
        self._device_info_logged = False

        self._model_init_kwargs = {}
        self._training_data = None

        model_init_kwargs = update_dict_without_conflict(model_init_kwargs,
                                                         {
                                                             'multi_output_model__verbose': -1,
                                                             'multi_step_model__verbose': -1,
                                                             'wide_gbrt__verbose': -1
                                                         })

        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be a positive number or None.")

        for k, v in model_init_kwargs.items():
            raise_if(ValueError, '__' not in k,
                     f"{k} must has double underline.")

            all_models = get_all_available_models()
            raise_if(ValueError, k.split('__')[0] not in all_models and k.split('__')[0]
                     not in get_all_model_class_name(),
                     f"{k.split('__')[0]} is not a valid model name")
            if k.split('__')[0] in self._available_models:
                self._model_init_kwargs[k] = v

        # Build compact single-line device description
        _cpu_only_models = {'auto_arima', 'prophet',
                            'wide_gbrt', 'multi_output_model', 'multi_step_model',
                            'regressor_chain'}
        _effective = self._given_models or list(self._available_models.keys())
        _all_cpu = all(isinstance(m, str) and m in _cpu_only_models for m in _effective)
        if _all_cpu:
            _active = 'CPU'
        else:
            _device, _device_detail = detect_available_device(self.accelerator)
            _active = _device.upper().replace(':', ' ').split()[0]  # 'mps', 'cuda:0' -> 'MPS', 'CUDA'
        self._compute_device_msg = f"Accelerator: {_active}"

        self.per_model_lags = per_model_lags or {}
        self.gbdt_differential_n = gbdt_differential_n

    def _check_time_budget(self):
        """Check if time budget is exhausted. Returns remaining seconds or None."""
        if self.time_limit is None or self._fit_start_time is None:
            return None
        elapsed = time.time() - self._fit_start_time
        remaining = self.time_limit - elapsed
        return remaining

    def _is_time_exhausted(self):
        """Return True if time budget is used up."""
        remaining = self._check_time_budget()
        if remaining is None:
            return False
        return remaining <= 0

    def _format_leaderboard_table(self, res_df):
        """Format leaderboard as a readable string table for logging."""
        if res_df.empty:
            return "  (no models completed)"

        sorted_df = res_df.sort_values(
            by='metric', ascending=self.metric_less_is_better
        ).reset_index(drop=True)

        lines = []
        # Header
        cols = sorted_df.columns.tolist()
        header = f"  {'Rank':<5} "
        for c in cols:
            if c == 'model':
                header += f"{'Model':<25} "
            elif c == 'metric':
                header += f"{'Metric':>12} "
            elif 'cost' in c:
                header += f"{c:>15} "
            elif c == 'quantile_acc':
                header += f"{'QAcc':>8} "
            else:
                header += f"{c:>12} "
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        # Rows
        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            marker = " *" if rank == 1 else "  "
            line = f"{marker}{rank:<4} "
            for c in cols:
                val = row[c]
                if c == 'model':
                    line += f"{str(val):<25} "
                elif isinstance(val, float):
                    line += f"{val:>12.4f} "
                else:
                    line += f"{str(val):>12} "
            lines.append(line)

        return "\n".join(lines)

    def _inject_pipeline_configs(self, m):
        """Inject pipeline-level configs (id_col, covariates) into model all_configs."""
        if not hasattr(m, 'all_configs'):
            return
        if self.id_col is not None:
            m.all_configs['id_col'] = self.id_col
        if self.known_covariates:
            m.all_configs['known_covariates'] = list(self.known_covariates)
        if self.past_covariates:
            m.all_configs['past_covariates'] = list(self.past_covariates)

    def _initial_models(self):
        initial_models = []
        ms = generate_models_set(self._available_models, self._given_models)

        # Auto-construct feature_cols from known_covariates for multivariate NN models
        effective_feature_cols = self.feature_cols
        if effective_feature_cols is None and self.known_covariates:
            target = self.target_col if isinstance(self.target_col, str) else self.target_col[0]
            effective_feature_cols = [target] + list(self.known_covariates)

        # 模型训练顺序
        for (model_name, model) in ms:
            # Use per-model lag if available, otherwise global lag
            effective_lags = self.per_model_lags.get(model_name, self.lags)
            model_kwargs = self._fill_func_params(
                func=model,
                time_col=self.time_col,
                target_col=self.target_col,
                lags=effective_lags,
                random_state=self.random_state,
                quantile=self.quantile,
                accelerator=self.accelerator,
                differential_n=self.gbdt_differential_n,
                feature_cols=effective_feature_cols
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

                        m = model(**new_model_kwargs)
                        self._inject_pipeline_configs(m)
                        initial_models.append([model_name_after_rename_in_config, m])

                        if self.include_init_config_model:
                            if [model_name, model(**model_kwargs)] not in initial_models:
                                m2 = model(**model_kwargs)
                                self._inject_pipeline_configs(m2)
                                initial_models.append([model_name, m2])

                if not include_in_configs:
                    m = model(**model_kwargs)
                    self._inject_pipeline_configs(m)
                    initial_models.append([model_name, m])
            else:
                m = model(**model_kwargs)
                self._inject_pipeline_configs(m)
                initial_models.append([model_name, m])

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
        ['auto_arima',
         'd_linear',
         'deep_forest',
         'gau',
         'multi_output_model',
         'multi_step_model',
         'n_beats',
         'n_hits',
         'n_linear',
         'regressor_chain',
         'patch_rnn',
         'stacking_rnn',
         'tcn',
         'tft',
         'tide',
         'time2vec',
         'torch_bagging_forest',
         'torch_boosting_forest',
         'transformer']
        """
        return sorted(list(get_all_available_models().keys()))

    def _scale_data(self, data, valid_data=None, refit_scaler=True):
        df, valid_df = data.copy(), valid_data  # valid_data will not be deep copy in this step

        if refit_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        if scaler is not None:
            if self.id_col is not None and self.id_col in df.columns:
                # Per-series scaling: each series gets its own scaler
                scalers = self._panel_scalers if refit_scaler else self._temp_panel_scalers
                for sid, idx in df.groupby(self.id_col).groups.items():
                    s = deepcopy(scaler) if sid not in scalers else scalers[sid]
                    if refit_scaler or sid not in scalers:
                        df.loc[idx, self.target_col] = s.fit_transform(
                            df.loc[idx, self.target_col].values.reshape(-1, 1)
                        ).squeeze()
                        scalers[sid] = s
                    else:
                        df.loc[idx, self.target_col] = s.transform(
                            df.loc[idx, self.target_col].values.reshape(-1, 1)
                        ).squeeze()

                if valid_data is not None:
                    valid_df = valid_data.copy()
                    for sid, idx in valid_df.groupby(self.id_col).groups.items():
                        if sid in scalers:
                            valid_df.loc[idx, self.target_col] = scalers[sid].transform(
                                valid_df.loc[idx, self.target_col].values.reshape(-1, 1)
                            ).squeeze()
            else:
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
            if self.id_col is not None and self.id_col in df.columns:
                scalers = self._panel_scalers if use_scaler else self._temp_panel_scalers
                for sid, idx in df.groupby(self.id_col).groups.items():
                    if sid in scalers:
                        df.loc[idx, columns] = scalers[sid].inverse_transform(
                            df.loc[idx, columns].values.reshape(-1, 1)
                        ).squeeze()
            else:
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

        self._timer.sleep(0.1)
        # -------------------- predicting -------------------------
        if self.configs is not None:
            if self.configs.get_configs(model_name_after_rename):
                predict_kwargs = self.configs.get_configs(model_name_after_rename).get('predict_configs')
            else:
                predict_kwargs = {}
        else:
            predict_kwargs = {}

        # For multi-series, predict per-series steps, not total rows
        _is_panel = self.id_col is not None and self.id_col in valid_df.columns
        if _is_panel:
            n_predict = int(valid_df.groupby(self.id_col).size().min())
        else:
            n_predict = valid_df.shape[0]

        if check_has_param(model.predict, 'predict_kwargs'):
            eval_res = model.predict(n_predict, data=valid_df, predict_kwargs=predict_kwargs)
        else:
            eval_res = model.predict(n_predict)

        if use_scaler:
            scaler = self.scaler
        else:
            scaler = self._temp_scaler

        yt = valid_df[self.target_col].values
        yp = eval_res[self.target_col].values

        res_quantile_acc = None
        if self.quantile:
            left_pred = eval_res[f"{self.target_col}_lower"].values
            right_pred = eval_res[f"{self.target_col}_upper"].values

        if scaler is not None:
            if _is_panel and self._panel_scalers:
                # Per-series inverse transform for both actuals and predictions
                yt_inv, yp_inv = [], []
                lp_inv, rp_inv = [], []
                for sid in valid_df[self.id_col].unique():
                    s = self._panel_scalers.get(sid)
                    if s is None:
                        continue
                    v_mask = valid_df[self.id_col] == sid
                    yt_s = valid_df.loc[v_mask, self.target_col].values
                    yt_inv.append(s.inverse_transform(yt_s.reshape(-1, 1)).squeeze())

                    if self.id_col in eval_res.columns:
                        p_mask = eval_res[self.id_col] == sid
                        yp_s = eval_res.loc[p_mask, self.target_col].values
                    else:
                        yp_s = yp[:len(yt_s)]
                        yp = yp[len(yt_s):]
                    yp_inv.append(s.inverse_transform(yp_s.reshape(-1, 1)).squeeze())

                    if self.quantile:
                        if self.id_col in eval_res.columns:
                            lp_s = eval_res.loc[p_mask, f"{self.target_col}_lower"].values
                            rp_s = eval_res.loc[p_mask, f"{self.target_col}_upper"].values
                        else:
                            lp_s = left_pred[:len(yt_s)]
                            rp_s = right_pred[:len(yt_s)]
                            left_pred = left_pred[len(yt_s):]
                            right_pred = right_pred[len(yt_s):]
                        lp_inv.append(s.inverse_transform(lp_s.reshape(-1, 1)).squeeze())
                        rp_inv.append(s.inverse_transform(rp_s.reshape(-1, 1)).squeeze())

                yt = np.concatenate(yt_inv)
                yp = np.concatenate(yp_inv)
                if self.quantile:
                    left_pred = np.concatenate(lp_inv)
                    right_pred = np.concatenate(rp_inv)
                    res_quantile_acc = quantile_acc(yt, left_pred, right_pred)
            else:
                yt = scaler.inverse_transform(yt.reshape(-1, 1)).squeeze()
                yp = scaler.inverse_transform(yp.reshape(-1, 1)).squeeze()

                if self.quantile:
                    left_pred = scaler.inverse_transform(left_pred.reshape(-1, 1)).squeeze()
                    right_pred = scaler.inverse_transform(right_pred.reshape(-1, 1)).squeeze()
                    res_quantile_acc = quantile_acc(yt, left_pred, right_pred)

        metric_val = self.metric(yt, yp)

        eval_cost = self._timer.last_timestamp_diff()

        del eval_res

        gc.collect()
        self._timer.sleep(0.1)

        self._timer.clear()  # 重置计时器

        if self.quantile:
            res_df = pd.concat(
                (res_df, pd.DataFrame(
                    [[model_name_after_rename, train_cost, eval_cost, metric_val, res_quantile_acc]],
                    columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric', 'quantile_acc'])),
                axis=0, ignore_index=True)
        else:
            res_df = pd.concat(
                (res_df, pd.DataFrame([[model_name_after_rename, train_cost, eval_cost, metric_val]],
                                      columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])),
                axis=0, ignore_index=True)

        return model_name_after_rename, model, res_df, {
            'train_cost': train_cost, 'eval_cost': eval_cost,
            'metric': metric_val, 'quantile_acc': res_quantile_acc,
        }

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
        0           0  torch_boosting_forest_0  2.567801  0.978624  0.123456
        1           1  torch_bagging_forest_0   3.123456  1.234567  0.456789
        2           2  deep_forest_0            1.987654  0.876543  0.987654
        ...         ...            ...            ...           ...       ...

        Notes
        -----
        - The fit function trains all models in the pipeline using the provided training data.
        - The optional valid_data parameter allows for model evaluation on a separate validation dataset.
        - The resulting leaderboard provides a ranked list of models based on the specified evaluation metric.
        """
        self._fit_start_time = time.time()
        self._failed_models = []
        self._skipped_models = []

        if not self._device_info_logged:
            self.logger.info(self._compute_device_msg)
            self._device_info_logged = True
        if self.time_limit is not None:
            self.logger.info(f"Time budget: {self.time_limit:.0f}s")

        check_time_col_is_timestamp(data, self.time_col)

        if self.id_col is not None and self.id_col in data.columns:
            # Multi-series: validate each series has enough data
            for sid, sdf in data.groupby(self.id_col):
                if len(sdf) <= self.lags:
                    self.logger.warning(
                        f"Series '{sid}' has only {len(sdf)} rows (<= lags={self.lags}), will be skipped."
                    )
            if self.id_col not in data.columns:
                raise ValueError(f"id_col '{self.id_col}' not found in data columns.")
            n_series = data[self.id_col].nunique()
            self.logger.info(f"Multi-series mode: {n_series} series detected (id_col='{self.id_col}')")
        else:
            if data.shape[0] <= self.lags:
                raise ValueError(f'length of df must be greater than lags, df length = {data.shape[0]}, lags = {self.lags}')

        if valid_data is not None:
            raise_if_not(AssertionError, data.columns.tolist() == valid_data.columns.tolist(),
                         "columns of data and valid_data do not match.")
            check_time_col_is_timestamp(valid_data, self.time_col)

            df, valid_df = data.copy(), valid_data.copy()
        else:
            if self.id_col is not None and self.id_col in data.columns:
                # Multi-series: take last 2*lags rows per series as validation
                parts = []
                for sid, sdf in data.groupby(self.id_col):
                    sdf = sdf.sort_values(self.time_col)
                    parts.append(sdf.iloc[-(2 * self.lags):])
                df, valid_df = data.copy(), pd.concat(parts, ignore_index=True)
            else:
                df, valid_df = data.copy(), data.iloc[-(2 * self.lags):, :]

        # 如果指定scaler，此语句会对数据缩放
        df, valid_df = self._scale_data(df, valid_df, refit_scaler=True)

        res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric'])
        if self.quantile:
            res = pd.DataFrame(columns=['model', 'train_cost(s)', 'eval_cost(s)', 'metric', 'quantile_acc'])

        models = self._initial_models()
        n_models = len(models)
        model_names = [name for name, _ in models]
        self.logger.info(f"Training {n_models} models: {model_names}")
        if self.per_model_lags:
            lag_info = ', '.join(f"{m}={l}" for m, l in sorted(self.per_model_lags.items()) if m in model_names)
            if lag_info:
                self.logger.info(f"Per-model lags: {lag_info} (primary={self.lags})")

        for idx, (model_name_after_rename, model) in enumerate(models):
            # Check time budget before starting next model
            if self._is_time_exhausted():
                remaining_names = [name for name, _ in models[idx:]]
                self._skipped_models.extend(
                    [(name, 'time_limit_exceeded') for name in remaining_names]
                )
                self.logger.warning(
                    f"Time budget exhausted after {time.time() - self._fit_start_time:.1f}s. "
                    f"Skipping {len(remaining_names)} remaining model(s): {remaining_names}"
                )
                break

            remaining = self._check_time_budget()
            budget_str = f" | remaining {remaining:.0f}s" if remaining is not None else ""
            self.logger.info(
                f"[{idx + 1}/{n_models}] Fitting {model_name_after_rename}...{budget_str}"
            )

            try:
                model_name_after_rename, model, res, fit_info = self._fit(
                    model_name_after_rename, model, df, valid_df, res,
                    use_scaler=True
                )
                self.models_.append((model_name_after_rename, model))

                # Log per-model result
                metric_val = fit_info['metric']
                train_t = fit_info['train_cost']
                eval_t = fit_info['eval_cost']
                qacc_str = ""
                if fit_info.get('quantile_acc') is not None:
                    qacc_str = f", quantile_acc={fit_info['quantile_acc']:.3f}"
                self.logger.info(
                    f"  => {model_name_after_rename}: metric={metric_val:.6f}, "
                    f"train={train_t:.2f}s, eval={eval_t:.2f}s{qacc_str}"
                )

                # Invoke callback for SmartRouter integration
                if self._on_model_complete_callback is not None:
                    self._on_model_complete_callback(
                        model_name=model_name_after_rename,
                        model=model,
                        fit_info=fit_info,
                        idx=idx,
                        total=n_models,
                    )

            except Exception as e:
                self._failed_models.append({
                    'model': model_name_after_rename,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })
                self.logger.error(
                    f"  => {model_name_after_rename} FAILED: {type(e).__name__}: {e}"
                )

        # Summary logging
        total_time = time.time() - self._fit_start_time
        n_success = len([r for r in res.itertuples()]) if not res.empty else 0
        n_failed = len(self._failed_models)
        n_skipped = len(self._skipped_models)

        self.logger.info(
            f"\nTraining complete: {n_success} succeeded, "
            f"{n_failed} failed, {n_skipped} skipped "
            f"({total_time:.1f}s total)"
        )

        if n_failed > 0:
            self.logger.warning(
                f"Failed models: {[f['model'] for f in self._failed_models]}"
            )

        if res.empty:
            self.logger.error("No models completed successfully.")
            self.leader_board_ = res
            return self.leader_board_

        self.leader_board_ = res.sort_values(
            by='metric', ascending=self.metric_less_is_better
        ).reset_index(drop=True)

        self.leader_board_.columns.name = 'Leaderboard'

        # Log formatted leaderboard
        self.logger.info(f"\n{self._format_leaderboard_table(self.leader_board_)}")

        self.best_model_ = self.get_model(self.leader_board_.iloc[0, :]['model'])
        self.logger.info(
            f"Best model: {self.leader_board_.iloc[0]['model']} "
            f"(metric={self.leader_board_.iloc[0]['metric']:.6f})"
        )

        self._training_data = data if hasattr(data, 'copy') else None

        del valid_data, df, valid_df, res
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
        >>> specific_model = pipeline.get_model('torch_boosting_forest_0')

        Notes
        -----
        - If model_name is not provided, the function returns the best-performing model based on the leaderboard.
        - The function allows retrieving a specific trained model by providing its unique name (e.g., 'torch_boosting_forest_0').
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
        >>> specific_model_configs = pipeline.get_model_all_configs('torch_boosting_forest_0')

        Notes
        -----
        - If model_name is not provided, the function returns the configuration details of the best-performing model.
        - The function allows retrieving configuration details for a specific trained model by providing its unique name (e.g., 'torch_boosting_forest_0').
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
    def predict(self, n, data=None, model_name=None, future_covariates=None):
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
        future_covariates : pd.DataFrame or None, optional, default: None
            Future known covariate values for the forecast horizon.
            Must have at least n rows and columns matching known_covariates.
            For multi-series, include the id_col to provide per-series covariates.

        Returns
        -------
        predictions : pd.DataFrame
            DataFrame containing the predicted values for the specified model or the best model.
        """
        df = None
        if data is not None:
            df = data.copy()
            if self.scaler is not None:
                if self.id_col is not None and self.id_col in df.columns and self._panel_scalers:
                    for sid, idx in df.groupby(self.id_col).groups.items():
                        if sid in self._panel_scalers:
                            df.loc[idx, self.target_col] = self._panel_scalers[sid].transform(
                                df.loc[idx, self.target_col].values.reshape(-1, 1)
                            ).squeeze()
                else:
                    df[self.target_col] = self.scaler.transform(
                        df[self.target_col].values.reshape(-1, 1)).squeeze()

        target_model = self.get_model(model_name) if model_name is not None else self.best_model_

        # Build predict kwargs based on model capabilities
        predict_kwargs = {'n': n}
        if check_has_param(target_model.predict, 'data'):
            predict_kwargs['data'] = df
        if future_covariates is not None and check_has_param(target_model.predict, 'future_covariates'):
            predict_kwargs['future_covariates'] = future_covariates

        res = target_model.predict(**predict_kwargs)

        for i in res.columns:
            if i.startswith(self.target_col):
                res = self._inverse_data(res, columns=i)

        return res

    def predict_quantiles(self, n, levels=None, data=None, model_name=None,
                          future_covariates=None):
        """Produce multi-quantile forecasts.

        Uses stored conformal residuals from the calibration phase to compute
        prediction intervals at arbitrary coverage levels.  Residuals are
        applied in **scaled space** (where calibration happened) and the
        resulting bounds are inverse-transformed to the original scale.

        Parameters
        ----------
        n : int
            Number of future steps to predict.
        levels : list of float or None
            Coverage levels, e.g. ``[0.5, 0.8, 0.9]``.
            Defaults to ``[0.5, 0.8, 0.9, 0.95]``.
        data : pd.DataFrame or None
            Optional input data for prediction.
        model_name : str or None
            Model to use. None = best model.
        future_covariates : pd.DataFrame or None
            Future known covariate values.

        Returns
        -------
        pd.DataFrame
            DataFrame with time_col, target_col (point prediction), and
            ``{target}_q{level}_lower`` / ``{target}_q{level}_upper`` for
            each requested level.
        """
        if levels is None:
            levels = [0.5, 0.8, 0.9, 0.95]

        # Get the inverse-transformed point predictions
        point_df = self.predict(n=n, data=data, model_name=model_name,
                                future_covariates=future_covariates)

        target_model = (self.get_model(model_name)
                        if model_name is not None else self.best_model_)

        if not hasattr(target_model, 'predict_quantiles'):
            return point_df

        # Conformal residuals were collected in *scaled* space, so we need
        # to compute bounds in scaled space then inverse-transform them.
        # Re-scale the point predictions back to scaled space for offset math.
        point_vals = point_df[self.target_col].values.copy()
        if self.scaler is not None:
            scaled_pts = self.scaler.transform(
                point_vals.reshape(-1, 1)).squeeze()
        else:
            scaled_pts = point_vals

        # Compute quantile bounds in scaled space
        q_result = target_model.predict_quantiles(scaled_pts, levels)

        # Build output DataFrame
        result = point_df[[self.time_col, self.target_col]].copy()
        if self.id_col is not None and self.id_col in point_df.columns:
            result[self.id_col] = point_df[self.id_col].values

        for lv in sorted(levels):
            lo, hi = q_result[lv]
            # Inverse-transform bounds from scaled space to original
            if self.scaler is not None:
                lo = self.scaler.inverse_transform(
                    np.asarray(lo).reshape(-1, 1)).squeeze()
                hi = self.scaler.inverse_transform(
                    np.asarray(hi).reshape(-1, 1)).squeeze()

            lv_str = f"{lv:.2f}".rstrip('0').rstrip('.')
            result[f"{self.target_col}_q{lv_str}_lower"] = lo
            result[f"{self.target_col}_q{lv_str}_upper"] = hi

        return result

    def update(self, new_data, update_epochs=50, refit_all=False):
        """Incrementally update fitted models with new data.

        Concatenates *new_data* with the stored training data and refits
        each model.  Neural-network models warm-start from their current
        weights with *update_epochs* epochs; tree / statistical models
        are retrained from scratch on the combined data (fast).

        Parameters
        ----------
        new_data : pd.DataFrame
            New observations.  Must have the same columns as the
            original training data.
        update_epochs : int, default 50
            Number of training epochs for NN warm-start updates.
        refit_all : bool, default False
            If True, refit every model in ``models_``.
            If False (default), refit only the best model.

        Returns
        -------
        self
        """
        if self._training_data is None:
            raise ValueError("No training data stored. Call fit() first.")
        if self.best_model_ is None:
            raise ValueError("Pipeline has not been fitted yet.")

        check_time_col_is_timestamp(new_data, self.time_col)

        # Combine old + new data
        combined = pd.concat(
            [self._training_data, new_data], ignore_index=True
        ).sort_values(self.time_col).reset_index(drop=True)

        # Re-scale combined data
        if self.scaler is not None:
            if self.id_col is not None and self.id_col in combined.columns:
                for sid, idx in combined.groupby(self.id_col).groups.items():
                    s = MinMaxScaler()
                    combined.loc[idx, self.target_col] = s.fit_transform(
                        combined.loc[idx, self.target_col].values.reshape(-1, 1)
                    ).squeeze()
                    self._panel_scalers[sid] = s
            else:
                self.scaler.fit(combined[self.target_col].values.reshape(-1, 1))
                combined[self.target_col] = self.scaler.transform(
                    combined[self.target_col].values.reshape(-1, 1)
                ).squeeze()

        # Determine which models to refit
        if refit_all:
            targets = list(self.models_)
        else:
            best_name = self.leader_board_.iloc[0]['model']
            targets = [(n, m) for n, m in self.models_ if n == best_name]

        n_updated = 0
        for model_name, model in targets:
            try:
                # Build fit_kwargs — NN models get reduced epochs
                fit_kwargs = {}
                if hasattr(model, 'model') and hasattr(model.model, 'fit'):
                    inner = model.model
                    if hasattr(inner, 'training_logs'):
                        # TorchModelMixin — warm-start with fewer epochs
                        fit_kwargs['epochs'] = update_epochs
                        fit_kwargs['verbose'] = False

                # Refit the model on combined data
                if check_has_param(model.fit, 'fit_kwargs'):
                    model.fit(combined, cv=min(self.cv, 3),
                              fit_kwargs=fit_kwargs)
                elif check_has_param(model.fit, 'cv'):
                    model.fit(combined, cv=min(self.cv, 3))
                else:
                    model.fit(combined)

                n_updated += 1
                self.logger.info(f"  Updated: {model_name}")
            except Exception as e:
                self.logger.warning(
                    f"  Failed to update {model_name}: {type(e).__name__}: {e}"
                )

        # Store updated training data (in original scale)
        self._training_data = pd.concat(
            [self._training_data, new_data], ignore_index=True
        ).sort_values(self.time_col).reset_index(drop=True)

        self.logger.info(
            f"Incremental update complete: {n_updated}/{len(targets)} models updated, "
            f"total training rows: {len(self._training_data)}"
        )

        return self

    @property
    def failed_models(self):
        """Return list of failed model details."""
        return list(self._failed_models)

    @property
    def skipped_models(self):
        """Return list of skipped model names and reasons."""
        return list(self._skipped_models)

    def plot(self, n=None, data=None, model_name=None, history_tail=None,
             lang='zh', figsize=(14, 5), show=True):
        """Plot forecast from the best (or specified) model against history.

        Parameters
        ----------
        n : int or None
            Forecast horizon. Defaults to ``lags``.
        data : pd.DataFrame or None
            Custom data for prediction. None uses last training data.
        model_name : str or None
            Model to use. None uses the best model.
        history_tail : int or None
            Show only last N history points for clarity.
        lang : 'zh' or 'en'
        figsize : tuple
        show : bool

        Returns
        -------
        fig : matplotlib Figure
        """
        from PipelineTS.plot.ts_plot import plot_forecast
        if n is None:
            n = self.lags
        pred = self.predict(n=n, data=data, model_name=model_name)
        train = self._training_data if self._training_data is not None else pd.DataFrame()
        return plot_forecast(
            train, pred, self.time_col, self.target_col,
            history_tail=history_tail, lang=lang, figsize=figsize, show=show,
        )

    def plot_leaderboard(self, lang='zh', figsize=(10, 5), show=True):
        """Plot the model leaderboard as a bar chart.

        Parameters
        ----------
        lang : 'zh' or 'en'
        figsize : tuple
        show : bool

        Returns
        -------
        fig : matplotlib Figure
        """
        from PipelineTS.plot.ts_plot import plot_leaderboard_detail
        if self.leader_board_ is None or self.leader_board_.empty:
            raise ValueError("No leaderboard available. Call fit() first.")
        return plot_leaderboard_detail(
            self.leader_board_, lang=lang, figsize=figsize, show=show,
        )

    def save(self, path):
        """Save this fitted pipeline to a zip file.

        Parameters
        ----------
        path : str
            File path ending with '.zip'.

        Returns
        -------
        str
            The path to the saved zip file.

        Examples
        --------
        >>> pipeline.save('my_pipeline.zip')
        >>> loaded = ModelPipeline.load('my_pipeline.zip')
        """
        from PipelineTS.io import save_model
        return save_model(path, self)

    @staticmethod
    def load(path):
        """Load a fitted pipeline from a zip file.

        Parameters
        ----------
        path : str
            File path ending with '.zip'.

        Returns
        -------
        ModelPipeline
            The loaded pipeline with all models restored.

        Examples
        --------
        >>> pipeline = ModelPipeline.load('my_pipeline.zip')
        >>> pipeline.predict(n=12)
        """
        from PipelineTS.io import load_model
        return load_model(path)
