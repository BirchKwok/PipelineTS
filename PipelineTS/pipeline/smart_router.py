"""SmartRouter: Intelligent routing for time series forecasting.

Analyzes data characteristics and automatically selects the best combination
of preprocessing steps and models, then returns a fitted pipeline ready for
production use.
"""

import warnings
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
)

from spinesUtils.logging import Logger
from spinesUtils.asserts import ParameterTypeAssert, raise_if

from PipelineTS.pipeline.pipeline import ModelPipeline
from PipelineTS.pipeline.pipeline_models import get_all_available_models
from PipelineTS.metrics import resolve_metric
from PipelineTS.datasets import (
    LoadElectric,
    LoadMessagesSentHour,
    LoadMessagesSent,
    LoadWebSales,
    LoadSupermarketIncoming,
)
from PipelineTS.preprocessing import (
    TimeSeriesMissingHandler,
    TimeSeriesOutlierDetector,
    FrequencyDetector,
    StationarityTest,
)
from PipelineTS.preprocessing.time_series_diagnostics import (
    hurst_exponent,
    spectral_entropy,
)
from PipelineTS.models.nn._nn_specs import (
    NN_FOUNDATION_MODEL_KEYS,
    NN_GTB_MODEL_KEYS,
    NN_KEYS_BY_CATEGORY,
    NN_MODEL_KEYS,
    NN_TRANSFORMER_LIKE_MODEL_KEYS,
)


# ---------------------------------------------------------------------------
#  Data Profile
# ---------------------------------------------------------------------------

class DataProfile:
    """Container for time series data characteristics.

    Attributes
    ----------
    n_rows : int
    freq : str or None
    is_regular : bool
    dominant_periods : list of int
    stationarity : str
        One of 'stationary', 'trend_stationary', 'difference_stationary',
        'non_stationary'.
    suggested_d : int
        Suggested differencing order (0, 1, or 2).
    mean, std, cv : float
    skewness, kurtosis : float
    has_negative : bool
    pct_missing : float
    pct_outlier : float
    noise_ratio : float
        Residual std / series std after detrending.
    trend_strength : float
        R² of linear fit, in [0, 1].
    seasonality_strength : float
        Ratio of dominant spectral power to total power.
    """

    def __init__(self):
        self.n_rows = 0
        self.freq = None
        self.freq_timedelta = None
        self.is_regular = True
        self.dominant_periods = []
        self.stationarity = 'stationary'
        self.suggested_d = 0
        self.mean = 0.0
        self.std = 1.0
        self.cv = 0.0
        self.skewness = 0.0
        self.kurtosis = 0.0
        self.has_negative = False
        self.pct_missing = 0.0
        self.pct_outlier = 0.0
        self.noise_ratio = 1.0
        self.trend_strength = 0.0
        self.seasonality_strength = 0.0
        self.autocorr_lag1 = 0.0
        self.autocorr_lag2 = 0.0
        self.n_seasonalities = 0
        self.regime_changes = 0
        self.n_series = 1
        self._total_rows = 0

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        lines = ["DataProfile("]
        for k, v in self.summary().items():
            if isinstance(v, float):
                lines.append(f"  {k}={v:.4g},")
            else:
                lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


class DataInsightProfile:
    def __init__(self):
        self.n_rows_total = 0
        self.n_columns = 0
        self.memory_mb = 0.0
        self.duplicate_timestamp_ratio = 0.0
        self.regularity_ratio = 1.0
        self.implicit_gap_ratio = 0.0
        self.completeness_ratio = 1.0
        self.explicit_nan_ratio = 0.0
        self.inf_ratio = 0.0
        self.zero_ratio = 0.0
        self.negative_ratio = 0.0
        self.intermittent_ratio = 0.0
        self.spectral_entropy = None
        self.hurst_exponent = None
        self.long_memory = False
        self.high_entropy = False
        self.low_completeness = False
        self.has_time_duplicates = False
        self.panel_min_length = 0
        self.panel_median_length = 0.0
        self.panel_max_length = 0
        self.panel_length_cv = 0.0
        self.panel_irregular_ratio = 0.0
        self.risk_flags = []

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        lines = ["DataInsightProfile("]
        for k, v in self.summary().items():
            if isinstance(v, float):
                lines.append(f"  {k}={v:.4g},")
            else:
                lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Ensemble Predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """Multi-strategy ensemble of multiple fitted models.

    Supports three combination strategies:
    - 'weighted_avg': Inverse-metric weighted average (default).
    - 'median': Median of predictions (robust to outlier models).
    - 'stacking': Ridge meta-learner trained on cross-validated predictions.

    Parameters
    ----------
    pipeline : ModelPipeline
        The fitted pipeline containing all trained models.
    model_names : list of str
        Names of models to include in the ensemble.
    weights : dict
        Mapping of model_name -> weight (sum to 1.0).
    time_col : str
        Name of the datetime column.
    target_col : str
        Name of the target column.
    ensemble_method : str, default='weighted_avg'
        One of 'weighted_avg', 'median', 'stacking'.
    meta_model : object or None
        Fitted meta-learner for stacking mode. None for other modes.
    """

    def __init__(self, pipeline, model_names, weights, time_col, target_col,
                 ensemble_method='weighted_avg', meta_model=None):
        self.pipeline = pipeline
        self.model_names = model_names
        self.weights = weights
        self.time_col = time_col
        self.target_col = target_col
        self.ensemble_method = ensemble_method
        self.meta_model = meta_model

    def predict(self, n, data=None):
        """Generate ensemble predictions using the configured method."""
        all_preds = {}
        for name in self.model_names:
            all_preds[name] = self.pipeline.predict(
                n=n, data=data, model_name=name
            )

        # Use the first model's output as the template
        result = all_preds[self.model_names[0]].copy()
        value_cols = [c for c in result.columns if c.startswith(self.target_col)]

        if self.ensemble_method == 'median':
            for col in value_cols:
                stacked = np.column_stack(
                    [all_preds[name][col].values for name in self.model_names]
                )
                result[col] = np.median(stacked, axis=1)

        elif self.ensemble_method == 'stacking' and self.meta_model is not None:
            # Build feature matrix from base model predictions
            target_col_only = self.target_col
            feature_matrix = np.column_stack(
                [all_preds[name][target_col_only].values for name in self.model_names]
            )
            result[target_col_only] = self.meta_model.predict(feature_matrix)
            # For interval cols, fall back to weighted_avg
            for col in value_cols:
                if col != target_col_only:
                    result[col] = sum(
                        all_preds[name][col].values * self.weights[name]
                        for name in self.model_names
                    )

        elif self.ensemble_method == 'multi_stack' and self.meta_model is not None:
            # Multi-layer stacking: meta_model is a list of (meta, weight) pairs
            target_col_only = self.target_col
            feature_matrix = np.column_stack(
                [all_preds[name][target_col_only].values for name in self.model_names]
            )
            layer1_preds = []
            for meta, w in self.meta_model:
                layer1_preds.append(meta.predict(feature_matrix) * w)
            result[target_col_only] = sum(layer1_preds)
            # Interval cols: weighted_avg fallback
            for col in value_cols:
                if col != target_col_only:
                    result[col] = sum(
                        all_preds[name][col].values * self.weights[name]
                        for name in self.model_names
                    )

        else:  # weighted_avg (default)
            for col in value_cols:
                result[col] = sum(
                    all_preds[name][col].values * self.weights[name]
                    for name in self.model_names
                )

        return result

    @property
    def all_configs(self):
        return {
            'ensemble': True,
            'strategy': self.ensemble_method,
            'models': self.model_names,
            'weights': {k: round(v, 4) for k, v in self.weights.items()},
        }

    def __repr__(self):
        models_str = ', '.join(
            f"{n}({self.weights[n]:.2f})" for n in self.model_names
        )
        return f"EnsemblePredictor(method={self.ensemble_method}, [{models_str}])"


# ---------------------------------------------------------------------------
#  SmartRouter
# ---------------------------------------------------------------------------

class SmartRouter:
    """Intelligent routing for time series forecasting.

    Analyzes the input data to automatically determine:
    1. Optimal preprocessing (missing handling, outlier clipping, scaler)
    2. Best model candidates based on data characteristics
    3. Appropriate lag window size
    4. GBDT differencing order
    5. Feature engineering decisions (routing_mode, lag features)
    6. Data-adaptive hyperparameters per model
    7. Post-fit ensemble of top-K models (weighted_avg / median / stacking)

    Then runs a focused ModelPipeline competition on the selected subset,
    with error resilience, time budget enforcement, and optional ensemble.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    target_col : str
        Name of the target column.
    n_predict : int or None, default=None
        Number of future steps to predict. If None, auto-determined from lags.
    quantile : float or None, default=None
        Coverage level for prediction intervals (e.g. 0.9). None = point only.
    accelerator : str, default='auto'
        Hardware accelerator for NN models.
    random_state : int, default=0
        Random seed.
    verbose : bool, default=True
        Whether to print routing decisions.
    preset : str or None, default=None
        Quality preset that configures multiple parameters at once.
        Explicitly provided params always override preset defaults.

        - ``'fast'``: 3 models, 3-fold CV, basic search, no ensemble.
        - ``'medium_quality'``: 5 models, 5-fold CV, auto search+ensemble.
        - ``'high_quality'``: 8 models, 5-fold CV, thorough search, weighted ensemble.
        - ``'best_quality'``: 15 models, 5-fold CV, thorough search, top-5 ensemble.

        If None, uses ``'medium_quality'`` defaults.
    max_models : int or None
        Maximum number of candidate models. None = use preset default.
    cv : int or None
        Cross-validation folds. None = use preset default.
    time_limit : int, float, or None, default=None
        Total time budget in seconds. None = no limit.
    ensemble_strategy : str or None
        How to combine models after fitting. None = use preset default.
        Options: 'auto', 'weighted_avg', 'median', 'stacking', 'none'.
    ensemble_top_k : int or None
        Maximum models in ensemble. None = use preset default.
    search_strategy : str or None
        Search strategy. None = use preset default.
        Options: 'basic', 'auto', 'thorough'.
    include_models : str, list of str, or None, default=None
        Pin specific model(s) for SmartRouter to optimize.  When set,
        SmartRouter skips heuristic model selection and screening, but
        still performs data profiling, preprocessing selection, scaler
        selection, lag optimization, feature engineering routing,
        hyperparameter suggestion, HPO (if enabled), and ensemble
        building for the specified models.

        Accepts a single model name (str) or a list of model names.
        Use ``SmartRouter.list_all_available_models()`` to see valid names.

    Examples
    --------
    >>> from PipelineTS.pipeline import SmartRouter
    >>> # Quick start with preset
    >>> router = SmartRouter(time_col='date', target_col='value', preset='fast')
    >>> router.fit(df)
    >>> router.predict(n=12)
    >>>
    >>> # Best quality with time limit
    >>> router = SmartRouter(time_col='date', target_col='value',
    ...                      preset='best_quality', time_limit=300)
    >>> router.fit(df)
    >>>
    >>> # Save and reload
    >>> router.save('my_model.zip')
    >>> loaded = SmartRouter.load('my_model.zip')
    >>>
    >>> # Pin specific models and let SmartRouter optimize strategy/params
    >>> router = SmartRouter(time_col='date', target_col='value',
    ...                      include_models=['prophet', 'catboost'],
    ...                      hpo_strategy='quick')
    >>> router.fit(df)
    >>>
    >>> # Single model with full optimization
    >>> router = SmartRouter(time_col='date', target_col='value',
    ...                      include_models='auto_arima')
    >>> router.fit(df)
    """

    # Preset configurations: preset -> {param: value}
    # Explicitly provided params always override preset defaults.
    _PRESETS = {
        'fast': {
            'max_models': 3,
            'cv': 3,
            'search_strategy': 'basic',
            'ensemble_strategy': 'none',
            'ensemble_top_k': 1,
            'hpo_strategy': 'none',
        },
        'medium_quality': {
            'max_models': 5,
            'cv': 5,
            'search_strategy': 'auto',
            'ensemble_strategy': 'auto',
            'ensemble_top_k': 3,
            'hpo_strategy': 'auto',
        },
        'high_quality': {
            'max_models': 8,
            'cv': 5,
            'search_strategy': 'thorough',
            'ensemble_strategy': 'weighted_avg',
            'ensemble_top_k': 3,
            'hpo_strategy': 'auto',
        },
        'best_quality': {
            'max_models': 15,
            'cv': 5,
            'search_strategy': 'thorough',
            'ensemble_strategy': 'weighted_avg',
            'ensemble_top_k': 5,
            'hpo_strategy': 'auto',
        },
    }

    _CATEGORIES = {
        'statistic': {'auto_arima', 'prophet', 'naive', 'seasonal_naive',
                      'theta', 'ets', 'short_trend_slot_blend',
                      'long_slot_trend_blend', 'stat_ensemble'},
        'ml': {'catboost', 'xgboost', 'random_forest', 'extra_forest',
               'gc_forest', 'wide_gbrt', 'multi_output_model',
               'multi_step_model', 'regressor_chain'},
        'nn_light': NN_KEYS_BY_CATEGORY['light'],
        'nn_medium': NN_KEYS_BY_CATEGORY['medium'],
        'nn_heavy': NN_KEYS_BY_CATEGORY['heavy'] | set(NN_FOUNDATION_MODEL_KEYS),
    }

    _SAFE_PORTFOLIO = (
        'stat_ensemble', 'short_trend_slot_blend', 'long_slot_trend_blend',
        'theta', 'ets', 'seasonal_naive',
        'auto_arima', 'prophet', 'multi_output_model', 'random_forest',
        'extra_forest', 'catboost', 'd_linear', 'n_linear', 'tide', 'tcn',
    )

    _BASELINE_GUARDRAIL_MODELS = (
        'stat_ensemble', 'short_trend_slot_blend', 'long_slot_trend_blend',
        'theta', 'ets', 'seasonal_naive',
        'random_forest', 'multi_step_model', 'multi_output_model',
        'regressor_chain', 'extra_forest',
    )

    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'n_predict': (int, None),
        'quantile': (float, None),
        'random_state': int,
        'verbose': bool,
        'max_models': (int, None),
        'cv': (int, None),
        'ensemble_strategy': (str, None),
        'ensemble_top_k': (int, None),
        'time_limit': (int, float, None),
        'search_strategy': (str, None),
        'preset': (str, None),
        'id_col': (str, None),
        'hpo_strategy': (str, None),
        'hpo_n_trials': (int, None),
        'hpo_timeout_per_model': (int, float, None),
        'include_models': (str, list, None),
        'epochs': (int, None),
    }, 'SmartRouter')
    def __init__(
        self,
        time_col,
        target_col,
        n_predict=None,
        quantile=None,
        accelerator='auto',
        random_state=0,
        verbose=True,
        max_models=None,
        cv=None,
        time_limit=None,
        ensemble_strategy=None,
        ensemble_top_k=None,
        search_strategy=None,
        preset=None,
        id_col=None,
        known_covariates=None,
        past_covariates=None,
        hpo_strategy=None,
        hpo_n_trials=None,
        hpo_timeout_per_model=None,
        include_models=None,
        epochs=None,
        metric='business',
        **model_kwargs,
    ):
        # Resolve preset: preset provides defaults, explicit params override
        if preset is not None:
            raise_if(ValueError, preset not in self._PRESETS,
                     f"preset must be one of {list(self._PRESETS.keys())}, got '{preset}'")
            defaults = self._PRESETS[preset]
        else:
            defaults = self._PRESETS['medium_quality']  # default behavior

        # Normalize include_models: str -> [str], None -> None
        if isinstance(include_models, str):
            include_models = [include_models]
        if include_models is not None:
            available = set(get_all_available_models().keys())
            unknown = [m for m in include_models if m not in available]
            raise_if(ValueError, len(unknown) > 0,
                     f"Unknown model(s): {unknown}. "
                     f"Available: {sorted(available)}")
            raise_if(ValueError, len(include_models) == 0,
                     "include_models cannot be empty")
        self.include_models = include_models
        self.epochs = epochs
        self.metric, self.metric_name = resolve_metric(metric)
        self.metric_less_is_better = True
        self.model_kwargs = dict(model_kwargs)

        # When user pins models, auto-adjust max_models if not explicitly set
        if include_models is not None and max_models is None:
            max_models = len(include_models)

        self.preset = preset
        self.id_col = id_col
        self.known_covariates = known_covariates or []
        self.past_covariates = past_covariates or []
        self.time_col = time_col
        self.target_col = target_col
        self.n_predict = n_predict
        self.quantile = quantile
        self.accelerator = accelerator
        self.random_state = random_state
        self.verbose = verbose
        self.max_models = max_models if max_models is not None else defaults['max_models']
        self.cv = cv if cv is not None else defaults['cv']
        self.time_limit = time_limit
        self.ensemble_strategy = (
            ensemble_strategy if ensemble_strategy is not None
            else defaults['ensemble_strategy']
        )
        self.ensemble_top_k = (
            ensemble_top_k if ensemble_top_k is not None
            else defaults['ensemble_top_k']
        )
        self.search_strategy = (
            search_strategy if search_strategy is not None
            else defaults['search_strategy']
        )
        self.hpo_strategy = (
            hpo_strategy if hpo_strategy is not None
            else defaults.get('hpo_strategy', 'none')
        )
        self.hpo_n_trials = (
            hpo_n_trials if hpo_n_trials is not None
            else defaults.get('hpo_n_trials', 10)
        )
        self.hpo_timeout_per_model = hpo_timeout_per_model

        raise_if(ValueError,
                 self.hpo_strategy not in ('none', 'quick', 'full', 'auto'),
                 f"hpo_strategy must be 'none', 'quick', 'full', or 'auto', "
                 f"got '{self.hpo_strategy}'")

        raise_if(ValueError,
                 self.ensemble_strategy not in (
                     'auto', 'weighted_avg', 'median', 'stacking',
                     'multi_stack', 'none'
                 ),
                 f"ensemble_strategy must be 'auto', 'weighted_avg', 'median', "
                 f"'stacking', 'multi_stack', or 'none', got '{self.ensemble_strategy}'")

        raise_if(ValueError,
                 self.search_strategy not in ('basic', 'auto', 'thorough'),
                 f"search_strategy must be 'basic', 'auto', or 'thorough', "
                 f"got '{self.search_strategy}'")

        if self.verbose and preset is not None:
            Logger(name='SmartRouter').info(
                f"Preset '{preset}': max_models={self.max_models}, cv={self.cv}, "
                f"search={self.search_strategy}, ensemble={self.ensemble_strategy}"
            )

        self.logger = Logger(name='SmartRouter')

        # Filled after fit()
        self.profile_ = None
        self.insights_ = None
        self._insights_cache_key = None
        self.strategy_ = None
        self.pipeline_ = None
        self.leader_board_ = None
        self.best_model_ = None
        self.ensemble_ = None
        self.model_scores_ = None
        self._preprocessed_data = None
        self._train_data_for_eval = None
        self._valid_data = None
        self._scaler_obj = None
        self._screening_results = None
        self._lag_exploration_results = None
        self._per_model_lags = None
        self._calibration_rho = None
        self._hpo_results = None
        self._active_hpo_strategy_ = None
        self._ensemble_eval = None
        self._fusion_scores = None
        self._feasibility_report = None
        self._fallback_used = False
        self._baseline_guardrail = None
        self._baseline_guardrail_cache = None
        self.autonomy_summary_ = None
        self.dataset_benchmark_ = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @ParameterTypeAssert({'data': pd.DataFrame, 'valid_data': (pd.DataFrame, None)})
    def fit(self, data, valid_data=None):
        """Profile data, select strategy, fit pipeline, return self.

        The fit process runs these stages in order:

        1. **Data profiling** – characterises stationarity, seasonality, trend,
           noise, autocorrelation, missing rate, and outlier rate.
        2. **Strategy selection** – chooses scaler, lags, model subset,
           hyperparameters, and ensemble strategy based on the data profile.
        3. **Baseline preflight** (when budget ≤ 45 s) – fits a fast statistical
           ensemble and accepts it immediately if the budget is very tight.
        4. **Wide screening** (``'auto'``/``'thorough'`` only) – trains
           lightweight models on a data subset to eliminate weak candidates.
        5. **Lag exploration** (``'auto'``/``'thorough'`` only) – tests 2-3
           lag values and picks the best one.
        6. **HPO** (when ``hpo_strategy != 'none'``) – runs Optuna search for
           top candidates.
        7. **Full training** – fits the final :class:`ModelPipeline` and
           builds an ensemble of the top-K models.

        Parameters
        ----------
        data : pd.DataFrame
            Training data containing at least *time_col* and *target_col*.
            For panel data, must also contain *id_col*.
        valid_data : pd.DataFrame or None, default None
            Optional external validation set.  When ``None``, an internal
            chronological split is used for model evaluation.

        Returns
        -------
        self
            Returns the fitted ``SmartRouter`` instance for method chaining.

        Raises
        ------
        ValueError
            If *data* is empty or does not contain the required columns.

        Examples
        --------
        >>> router = SmartRouter(time_col='date', target_col='value',
        ...                      preset='medium_quality')
        >>> router.fit(train_df)
        >>> router.fit(train_df, valid_data=val_df)
        """
        t0 = time.time()
        self._model_results = []
        self._screening_results = None
        self._lag_exploration_results = None
        self._per_model_lags = None
        self._train_data_for_eval = None
        self._calibration_rho = None
        self._hpo_results = None
        self._active_hpo_strategy_ = None
        self._ensemble_eval = None
        self._fusion_scores = None
        self._feasibility_report = None
        self._fallback_used = False
        self._baseline_guardrail = None
        self._baseline_guardrail_cache = None
        self.autonomy_summary_ = None

        # Print device info once at the very beginning
        if self.verbose:
            from PipelineTS.base.torch_mixin import detect_available_device
            _dev, _detail = detect_available_device(self.accelerator)
            _active = _dev.upper().replace(':', ' ').split()[0]
            self.logger.info(f"Accelerator: {_active}")

        # Auto-convert time column to datetime if needed
        data = self._ensure_datetime(data)
        if valid_data is not None:
            valid_data = self._ensure_datetime(valid_data)

        # ── Analyze & plan ────────────────────────────────────
        self.profile_ = self._profile_data(data)
        self.insights_ = self._get_data_insights(data, self.profile_)
        self.strategy_ = self._build_strategy(self.profile_)
        self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)

        # Compute pipeline stages for progress tracking
        _will_screen = self._should_screen()
        _will_explore = self._should_explore_lags()
        _will_hpo = self._should_hpo()

        _stages = [('📊', 'Data Profiling'),
                   ('🎯', 'Strategy & Model Selection')]
        if _will_screen:
            _stages.append(('🔍', 'Wide Screening'))
        if _will_explore:
            _stages.append(('🔎', 'Lag Exploration'))
        if _will_hpo:
            _stages.append(('🔧', 'Hyperparameter Tuning'))
        _stages.append(('🏋️', 'Full Training'))
        _stages.append(('✅', 'Evaluation & Summary'))
        _n_stages = len(_stages)
        _cur = 0

        # [1/N] Data Profiling
        _cur += 1
        if self.verbose:
            self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            self._log_profile()

        # [2/N] Strategy & Model Selection
        _cur += 1
        if self.verbose:
            self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            self._log_strategy()
            self._log_model_scores()
            self._log_roadmap(_stages, _cur)

        # Preprocess data
        full_processed = self._apply_preprocessing(data)
        user_provided_valid = valid_data is not None

        if user_provided_valid:
            processed_valid = self._apply_preprocessing(valid_data)
            processed_train = full_processed
        else:
            # Create a proper chronological train/valid split so that
            # validation timestamps are strictly after training timestamps.
            processed_train, processed_valid = self._temporal_split(
                full_processed, self.strategy_['lags']
            )

        # Evaluation trains on the historical train split and scores the
        # immediately following validation tail. For auto-split training,
        # the final predictor is refitted on full history after evaluation.
        self._preprocessed_data = full_processed
        self._train_data_for_eval = processed_train
        self._valid_data = processed_valid

        if self._should_run_baseline_preflight():
            self._run_baseline_preflight(t0)
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            if self._accept_baseline_preflight_if_tight_budget():
                return self

        # Quick screening (eliminate weak candidates with holdout)
        if _will_screen:
            _cur += 1
            if self.verbose:
                self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            # Wide screening: evaluate many more candidates than max_models
            # so we don't miss good models that heuristic scoring ranks low.
            pool_size = self._get_screen_pool_size()
            broad_candidates = self._build_adaptive_candidate_pool(
                self.profile_, pool_size
            )
            # ε-greedy injection: replace bottom heuristic picks with
            # unexplored models (categories absent from the pool first,
            # then random) to break the self-reinforcing heuristic loop.
            broad_candidates = self._inject_exploration_candidates(
                broad_candidates, self.profile_
            )
            screen_lb = self._quick_screen(
                processed_train, processed_valid,
                broad_candidates, self.strategy_
            )
            # Fusion selection: combine screening results with heuristic
            # scores, then apply diversity constraints for final selection.
            survivors = self._fusion_select(screen_lb, broad_candidates)
            self.strategy_['models'] = survivors
            if self.verbose:
                self._log_screening()

        # Multi-lag exploration (find optimal lag per model)
        per_model_lags = None
        if _will_explore:
            _cur += 1
            if self.verbose:
                self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            primary_lag = self._explore_lags(
                full_processed, self.strategy_['models'], self.strategy_
            )
            per_model_lags = getattr(self, '_per_model_lags', None)

            # Check if all models agree on the same lag
            unique_lags = set(per_model_lags.values()) if per_model_lags else set()
            all_same_lag = len(unique_lags) == 1

            if primary_lag != self.strategy_['lags']:
                old_lag = self.strategy_['lags']
                self.strategy_['lags'] = primary_lag
                if self.verbose:
                    if all_same_lag:
                        self.logger.info(
                            f"  Lag updated: {old_lag} -> {primary_lag}"
                        )
                    else:
                        self.logger.info(
                            f"  Primary lag updated: {old_lag} -> {primary_lag} "
                            f"(max of per-model lags)"
                        )
                # Re-split with primary lag (only if we auto-split).
                # Only update the validation window — full data for training.
                if not user_provided_valid:
                    processed_train, processed_valid = self._temporal_split(
                        full_processed, primary_lag
                    )
                    self._train_data_for_eval = processed_train
                    self._valid_data = processed_valid
                    self._baseline_guardrail_cache = None

            # Store per-model lags in strategy (only if models differ)
            if per_model_lags and not all_same_lag:
                self.strategy_['per_model_lags'] = per_model_lags

            if self.verbose:
                self._log_lag_exploration()

        # HPO (if enabled)
        hyperparams = self.strategy_.get('model_hyperparams', {})
        if _will_hpo:
            _cur += 1
            if self.verbose:
                self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            hyperparams = self._run_hpo(
                processed_train, processed_valid, hyperparams
            )
            self.strategy_['model_hyperparams'] = hyperparams

        # Full training with refined models + optimal lag
        _cur += 1
        lags = self.strategy_['lags']
        models = self.strategy_['models']
        scaler = self.strategy_['scaler']
        gbdt_diff_n = self.strategy_['gbdt_differential_n']

        remaining_time = self._get_remaining_time(t0)

        per_model_budget = None
        if remaining_time is not None and len(models) > 0:
            per_model_budget = remaining_time / len(models)
        hyperparams, budget_caps = self._apply_training_budget_caps(
            hyperparams, models, per_model_budget=per_model_budget,
            profile=self.profile_
        )
        # Apply user-specified overrides (highest priority)
        hyperparams = self._apply_user_model_kwargs(hyperparams)
        self.strategy_['model_hyperparams'] = hyperparams
        self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
        if self.verbose and budget_caps:
            self.logger.info(
                "  Training budget caps: " +
                ", ".join(f"{k}={v}" for k, v in budget_caps.items())
            )

        if self.verbose:
            self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            self.logger.info(f"  Models ({len(models)}): {models}")
            if remaining_time is not None:
                self.logger.info(f"  Time remaining: ~{remaining_time:.0f}s")

        # Pass per-model lags if models have different optimal lags
        effective_per_model_lags = self.strategy_.get('per_model_lags', None)

        self.pipeline_ = ModelPipeline(
            time_col=self.time_col,
            target_col=self.target_col,
            lags=lags,
            quantile=self.quantile,
            id_col=self.id_col,
            known_covariates=self.known_covariates or None,
            past_covariates=self.past_covariates or None,
            include_models=models,
            scaler=scaler,
            accelerator=self.accelerator,
            random_state=self.random_state,
            cv=self.cv,
            gbdt_differential_n=gbdt_diff_n,
            metric=self.metric,
            metric_less_is_better=self.metric_less_is_better,
            time_limit=remaining_time,
            per_model_lags=effective_per_model_lags,
            **hyperparams,
        )

        # SmartRouter already printed device info, suppress Pipeline's duplicate
        self.pipeline_._device_info_logged = True
        self.pipeline_._phase_label = "Full Training"

        # Register callback for real-time model tracking
        self._model_results = []
        self.pipeline_._on_model_complete_callback = self._on_model_trained

        self.leader_board_ = self.pipeline_.fit(
            self._train_data_for_eval, valid_data=self._valid_data
        )

        if self.leader_board_.empty:
            if self._fit_fallback_safe_models(t0):
                self.leader_board_ = self.pipeline_.leader_board_
            else:
                self.logger.error("No models completed. Cannot build ensemble.")
                return self

        self._run_baseline_guardrail(t0)

        self.best_model_ = self.pipeline_.best_model_

        # Evaluation & Summary
        _cur += 1
        if self.verbose:
            self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)

        self.ensemble_ = self._build_ensemble()
        if self.ensemble_ is not None:
            self.ensemble_ = self._evaluate_ensemble(self.ensemble_)

        if not user_provided_valid:
            self._refit_final_pipeline_on_full_history()

        self._compute_calibration()
        self.dataset_benchmark_ = (
            self._benchmark_on_builtin_datasets()
            if self.verbose and self.preset != 'fast' else None
        )
        total_time = time.time() - t0
        if self.verbose:
            self._log_calibration()
            self._log_benchmark_summary()
            self._log_summary(total_time)

        return self

    def _get_remaining_time(self, t0):
        """Get remaining time budget for main training phase."""
        if self.time_limit is None:
            return None
        elapsed = time.time() - t0
        remaining = self.time_limit - elapsed
        return max(remaining, 1.0)  # at least 1 second

    def _on_model_trained(self, model_name, model, fit_info, idx, total):
        """Callback invoked by Pipeline after each model completes."""
        self._model_results.append({
            'model_name': model_name,
            'metric': fit_info['metric'],
            'train_cost': fit_info['train_cost'],
            'eval_cost': fit_info['eval_cost'],
        })

    def _refit_final_pipeline_on_full_history(self):
        if self.pipeline_ is None or self._valid_data is None:
            return False
        if len(self._valid_data) == 0:
            return False
        try:
            self.pipeline_.update(self._valid_data, refit_all=True)
            self.best_model_ = self.pipeline_.best_model_
            return True
        except Exception as e:
            if self.verbose:
                self.logger.warning(
                    f"  Final full-history refit skipped: {type(e).__name__}: {e}"
                )
            return False

    def _build_autonomy_summary(self, strategy):
        strategy = strategy or {}
        hp = strategy.get('model_hyperparams', {}) or {}
        active_hpo_strategy = (
            self._active_hpo_strategy_
            if self._active_hpo_strategy_ is not None
            else self._active_hpo_strategy()
        )
        nn_models = set().union(
            self._CATEGORIES['nn_light'],
            self._CATEGORIES['nn_medium'],
            self._CATEGORIES['nn_heavy'],
        )
        nn_keys = {
            'routing_mode', 'use_gtb', 'use_ema', 'ema_decay',
            'use_swa', 'swa_start_frac', 'warmup_epochs',
            'use_residual_gate', 'd_model', 'nhead',
            'num_encoder_layers', 'dim_feedforward', 'level',
            'num_stacks', 'num_blocks', 'num_layers', 'layer_widths',
            'num_levels', 'hidden_channels', 'hidden_size',
            'decoder_output_dim', 'temporal_decoder_hidden',
            'num_decoder_layers',
        }
        nn_enhancements = {}
        for key, value in hp.items():
            if '__' not in key:
                continue
            model_name, param_name = key.split('__', 1)
            if model_name in nn_models and param_name in nn_keys:
                nn_enhancements.setdefault(model_name, {})[param_name] = value

        scaler = strategy.get('scaler')
        scaler_name = None if scaler is None else scaler.__class__.__name__
        return {
            'model_selection': list(strategy.get('models', []) or []),
            'lag_selection': strategy.get('lags'),
            'preprocessing': list(strategy.get('preprocessing', []) or []),
            'scaler': scaler_name,
            'feature_engineering': deepcopy(strategy.get('feature_engineering', {}) or {}),
            'adaptive_hyperparams_count': len(hp),
            'adaptive_hyperparams_enabled': len(hp) > 0,
            'hpo_requested_strategy': self.hpo_strategy,
            'hpo_strategy': active_hpo_strategy,
            'hpo_enabled': active_hpo_strategy != 'none',
            'hpo_tuned_models': sorted((self._hpo_results or {}).keys()),
            'nn_enhancements_enabled': len(nn_enhancements) > 0,
            'nn_enhancements': nn_enhancements,
            'ensemble_strategy': self.ensemble_strategy,
            'baseline_guardrail': deepcopy(self._baseline_guardrail),
        }

    def _baseline_guardrail_lags(self):
        n = max(1, int(getattr(self.profile_, 'n_rows', 0) or 0))
        horizon = int(self.n_predict or min(12, max(1, n // 10)))
        max_lags = max(4, n // 4)
        target = max(4, 12, horizon)
        if target * 2 >= n:
            target = max(4, min(max_lags, n // 3))
        return int(max(4, min(target, max_lags)))

    def _baseline_guardrail_candidates(self, lags):
        available = set(get_all_available_models().keys())
        candidates = [
            m for m in self._BASELINE_GUARDRAIL_MODELS
            if m in available
        ]
        if not candidates:
            return []
        feasible = self._filter_feasible_models(
            candidates, self.profile_, lags=lags,
            keep_at_least=min(2, len(candidates))
        )
        feasible = set(feasible)
        return [
            m for m in candidates if m in feasible
        ][:min(8, len(candidates))]

    def _fit_baseline_guardrail_pipeline(self, t0, reason_prefix='baseline'):
        if self._baseline_guardrail_cache is not None:
            return self._baseline_guardrail_cache

        baseline_lags = self._baseline_guardrail_lags()
        candidates = self._baseline_guardrail_candidates(baseline_lags)
        result = {
            'lags': baseline_lags,
            'candidate_models': candidates,
            'pipeline': None,
            'leaderboard': None,
            'reason': None,
        }
        if not candidates:
            result['reason'] = 'no_feasible_baseline_models'
            self._baseline_guardrail_cache = result
            return result

        remaining_time = None
        if self.time_limit is not None:
            raw_remaining = self.time_limit - (time.time() - t0)
            if raw_remaining <= 5:
                result['reason'] = 'time_budget_exhausted'
                self._baseline_guardrail_cache = result
                return result
            budget_cap = max(10.0, min(30.0, self.time_limit * 0.35))
            remaining_time = max(1.0, min(raw_remaining, budget_cap))

        if self.verbose:
            self.logger.info(
                f"  {reason_prefix}: testing {candidates} with lags={baseline_lags}"
            )

        baseline_pipeline = ModelPipeline(
            time_col=self.time_col,
            target_col=self.target_col,
            lags=baseline_lags,
            quantile=self.quantile,
            id_col=self.id_col,
            known_covariates=self.known_covariates or None,
            past_covariates=self.past_covariates or None,
            include_models=candidates,
            scaler=True,
            accelerator=self.accelerator,
            random_state=self.random_state,
            cv=min(self.cv, 3),
            gbdt_differential_n=0,
            metric=self.metric,
            metric_less_is_better=self.metric_less_is_better,
            time_limit=remaining_time,
        )
        baseline_pipeline._device_info_logged = True
        baseline_pipeline._phase_label = reason_prefix
        baseline_lb = baseline_pipeline.fit(
            self._train_data_for_eval, valid_data=self._valid_data
        )
        result.update({
            'pipeline': baseline_pipeline,
            'leaderboard': baseline_lb,
        })
        self._baseline_guardrail_cache = result
        return result

    def _should_run_baseline_preflight(self):
        if self.include_models is not None:
            return False
        if self._valid_data is None or len(self._valid_data) < 2:
            return False
        if self.search_strategy in ('auto', 'thorough'):
            return True
        if self.preset in ('high_quality', 'best_quality'):
            return True
        if self.time_limit is not None and self.max_models >= 5:
            return True
        return False

    def _run_baseline_preflight(self, t0):
        try:
            result = self._fit_baseline_guardrail_pipeline(
                t0, reason_prefix='Baseline champion preflight'
            )
        except Exception as e:
            self._baseline_guardrail_cache = {
                'reason': f'baseline_failed:{type(e).__name__}',
                'error': str(e),
                'pipeline': None,
                'leaderboard': None,
            }
            return
        lb = result.get('leaderboard')
        if lb is None or lb.empty:
            return
        preflight_slots = min(self.max_models, len(lb))
        if self.preset in ('high_quality', 'best_quality'):
            preflight_slots = min(2, preflight_slots)
        top_models = lb.head(preflight_slots)['model'].tolist()
        for m in reversed(top_models):
            if m not in self.strategy_['models']:
                self.strategy_['models'].insert(0, m)
        self.strategy_['models'] = self.strategy_['models'][:max(self.max_models, len(top_models))]

    def _accept_baseline_preflight_if_tight_budget(self):
        if self.time_limit is None or self.time_limit > 45:
            return False
        if self.preset in ('high_quality', 'best_quality'):
            return False
        freq = str(getattr(self.profile_, 'freq', '') or '').upper()
        n_rows = int(getattr(self.profile_, 'n_rows', 0) or 0)
        if n_rows <= 180 and freq.startswith(('M', 'Q', 'A', 'Y')):
            return False
        result = self._baseline_guardrail_cache
        if not result:
            return False
        baseline_pipeline = result.get('pipeline')
        baseline_lb = result.get('leaderboard')
        if baseline_pipeline is None or baseline_lb is None or baseline_lb.empty:
            return False
        baseline_lb = baseline_lb.head(self.max_models).reset_index(drop=True)
        self.pipeline_ = baseline_pipeline
        self.leader_board_ = baseline_lb
        self.best_model_ = baseline_pipeline.best_model_
        self.strategy_['models'] = baseline_lb['model'].tolist()
        self.strategy_['lags'] = result.get('lags', self.strategy_['lags'])
        self.strategy_['scaler'] = baseline_pipeline.scaler
        self.strategy_['gbdt_differential_n'] = 0
        self.strategy_['model_hyperparams'] = {}
        self.strategy_.pop('per_model_lags', None)
        current = {
            'checked': True,
            'switched': True,
            'reason': 'baseline_preflight_accepted',
            'lags': self.strategy_['lags'],
            'candidate_models': result.get('candidate_models') or [],
            'primary_model': None,
            'primary_metric': None,
            'baseline_model': str(baseline_lb.iloc[0]['model']),
            'baseline_metric': float(baseline_lb.iloc[0]['metric']),
            'baseline_trained_models': baseline_lb['model'].tolist(),
            'failed_models': len(getattr(baseline_pipeline, 'failed_models', []) or []),
            'skipped_models': len(getattr(baseline_pipeline, 'skipped_models', []) or []),
        }
        self._baseline_guardrail = current
        self.strategy_.setdefault('guardrails', {})['baseline'] = deepcopy(current)
        self._model_results = [
            {
                'model_name': row['model'],
                'metric': row['metric'],
                'train_cost': row.get('train_cost(s)', 0.0),
                'eval_cost': row.get('eval_cost(s)', 0.0),
            }
            for _, row in baseline_lb.iterrows()
        ]
        self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
        return True

    def _run_baseline_guardrail(self, t0):
        current = {
            'checked': False,
            'switched': False,
            'reason': None,
        }
        self._baseline_guardrail = current

        if self.include_models is not None:
            current['reason'] = 'pinned_models'
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            return False
        if self.leader_board_ is None or self.leader_board_.empty:
            current['reason'] = 'no_primary_leaderboard'
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            return False
        if self._valid_data is None or len(self._valid_data) < 2:
            current['reason'] = 'no_validation_data'
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            return False

        primary_model_results = list(getattr(self, '_model_results', []) or [])
        try:
            baseline_result = self._fit_baseline_guardrail_pipeline(
                t0, reason_prefix='Baseline guardrail'
            )
        except Exception as e:
            current['reason'] = f'baseline_failed:{type(e).__name__}'
            current['error'] = str(e)
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            return False

        baseline_lags = baseline_result.get('lags', self._baseline_guardrail_lags())
        candidates = baseline_result.get('candidate_models') or []
        current.update({
            'checked': True,
            'lags': baseline_lags,
            'candidate_models': candidates,
            'primary_model': str(self.leader_board_.iloc[0]['model']),
            'primary_metric': float(self.leader_board_.iloc[0]['metric']),
        })
        if not candidates:
            current['reason'] = baseline_result.get('reason') or 'no_feasible_baseline_models'
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            return False

        baseline_pipeline = baseline_result.get('pipeline')
        baseline_lb = baseline_result.get('leaderboard')

        if baseline_lb is None or baseline_lb.empty:
            current['reason'] = 'baseline_empty'
            current['failed_models'] = len(getattr(baseline_pipeline, 'failed_models', []) or [])
            current['skipped_models'] = len(getattr(baseline_pipeline, 'skipped_models', []) or [])
            self._model_results = primary_model_results
            self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
            return False

        baseline_lb = baseline_lb.head(self.max_models).reset_index(drop=True)
        baseline_metric = float(baseline_lb.iloc[0]['metric'])
        baseline_model = str(baseline_lb.iloc[0]['model'])
        primary_metric = current['primary_metric']
        less_is_better = (
            self.pipeline_.metric_less_is_better
            if self.pipeline_ is not None else True
        )
        improved = (
            baseline_metric < primary_metric
            if less_is_better else baseline_metric > primary_metric
        )
        current.update({
            'baseline_model': baseline_model,
            'baseline_metric': baseline_metric,
            'baseline_trained_models': baseline_lb['model'].tolist(),
            'failed_models': len(getattr(baseline_pipeline, 'failed_models', []) or []),
            'skipped_models': len(getattr(baseline_pipeline, 'skipped_models', []) or []),
            'relative_delta': (
                (baseline_metric - primary_metric) /
                (abs(primary_metric) + 1e-12)
            ),
        })

        if improved:
            self.pipeline_ = baseline_pipeline
            self.leader_board_ = baseline_lb
            self.strategy_['models'] = baseline_lb['model'].tolist()
            self.strategy_['lags'] = baseline_lags
            self.strategy_['scaler'] = baseline_pipeline.scaler
            self.strategy_['gbdt_differential_n'] = 0
            self.strategy_['model_hyperparams'] = {}
            self.strategy_.pop('per_model_lags', None)
            current['switched'] = True
            current['reason'] = 'baseline_won'
            self.strategy_.setdefault('guardrails', {})['baseline'] = deepcopy(current)
            self._model_results = [
                {
                    'model_name': row['model'],
                    'metric': row['metric'],
                    'train_cost': row.get('train_cost(s)', 0.0),
                    'eval_cost': row.get('eval_cost(s)', 0.0),
                }
                for _, row in baseline_lb.iterrows()
            ]
            if self.verbose:
                self.logger.warning(
                    f"  Baseline guardrail switched winner: {baseline_model} "
                    f"({baseline_metric:.4f}) beat {current['primary_model']} "
                    f"({primary_metric:.4f})"
                )
        else:
            current['reason'] = 'primary_won'
            self.strategy_.setdefault('guardrails', {})['baseline'] = deepcopy(current)
            self._model_results = primary_model_results
            if self.verbose:
                self.logger.info(
                    f"  Baseline guardrail kept primary: {current['primary_model']} "
                    f"({primary_metric:.4f}) vs {baseline_model} "
                    f"({baseline_metric:.4f})"
                )

        self.autonomy_summary_ = self._build_autonomy_summary(self.strategy_)
        return bool(current['switched'])

    def _fit_fallback_safe_models(self, t0):
        candidates = [
            m for m in self._SAFE_PORTFOLIO
            if m in get_all_available_models()
        ]
        candidates = self._filter_feasible_models(
            candidates, self.profile_, lags=self.strategy_['lags'],
            keep_at_least=min(3, len(candidates))
        )
        candidates = sorted(
            candidates,
            key=lambda m: (
                -(self.model_scores_ or {}).get(m, {}).get('total', 0.0),
                self._speed_tier_for_model(m),
            )
        )[:max(1, min(3, self.max_models, len(candidates)))]
        if not candidates:
            return False
        fallback_lags = min(
            self.strategy_['lags'],
            max(4, max(1, self.profile_.n_rows // 4))
        )
        remaining_time = self._get_remaining_time(t0)
        params = self._get_screening_hyperparams(candidates, self.strategy_)
        params, _ = self._apply_training_budget_caps(
            params, candidates,
            per_model_budget=(remaining_time / len(candidates)) if remaining_time else None,
            profile=self.profile_
        )
        if self.verbose:
            self.logger.warning(
                f"  Fallback: retrying safe portfolio {candidates} with lags={fallback_lags}"
            )
        try:
            self.pipeline_ = ModelPipeline(
                time_col=self.time_col,
                target_col=self.target_col,
                lags=fallback_lags,
                quantile=self.quantile,
                id_col=self.id_col,
                known_covariates=self.known_covariates or None,
                past_covariates=self.past_covariates or None,
                include_models=candidates,
                scaler=deepcopy(self.strategy_['scaler']),
                accelerator=self.accelerator,
                random_state=self.random_state,
                cv=min(self.cv, 3),
                gbdt_differential_n=self.strategy_['gbdt_differential_n'],
                metric=self.metric,
                metric_less_is_better=self.metric_less_is_better,
                time_limit=remaining_time,
                **params,
            )
            self.pipeline_._device_info_logged = True
            self.pipeline_._phase_label = "Safe Portfolio Fallback"
            self.pipeline_._on_model_complete_callback = self._on_model_trained
            self.leader_board_ = self.pipeline_.fit(
                self._train_data_for_eval, valid_data=self._valid_data
            )
            if self.leader_board_ is None or self.leader_board_.empty:
                return False
            self._fallback_used = True
            self.strategy_['models'] = candidates
            self.strategy_['lags'] = fallback_lags
            return True
        except Exception as e:
            if self.verbose:
                self.logger.error(f"  Fallback failed: {type(e).__name__}: {e}")
            return False

    def predict(self, n=None, data=None, model_name=None, use_ensemble=True,
                future_covariates=None):
        """Generate predictions using the ensemble, best, or specified model.

        Parameters
        ----------
        n : int or None
            Number of steps to predict. Defaults to n_predict or lags.
        data : pd.DataFrame or None
            Input data for prediction. If None, uses last training data.
        model_name : str or None
            Specific model to use. None = ensemble (if available) or best.
        use_ensemble : bool, default=True
            Whether to use the ensemble predictor when available.
            Set False to force single-model prediction.
        future_covariates : pd.DataFrame or None
            Future known covariate values for the forecast horizon.

        Returns
        -------
        pd.DataFrame
            Predictions with time_col and target_col (and _lower/_upper if quantile).
        """
        raise_if(ValueError, self.pipeline_ is None,
                 "SmartRouter has not been fitted yet. Call fit() first.")

        if n is None:
            n = self.n_predict or self.strategy_['lags']

        if data is not None:
            data = self._apply_preprocessing(data)

        # Use ensemble if available and no specific model requested
        if (use_ensemble and model_name is None
                and self.ensemble_ is not None):
            return self.ensemble_.predict(n=n, data=data)

        return self.pipeline_.predict(n=n, data=data, model_name=model_name,
                                      future_covariates=future_covariates)

    def predict_quantiles(self, n=None, levels=None, data=None,
                          model_name=None, future_covariates=None):
        """Produce multi-quantile forecasts at arbitrary coverage levels.

        Parameters
        ----------
        n : int or None
            Number of steps to predict. Defaults to n_predict or lags.
        levels : list of float or None
            Coverage levels, e.g. ``[0.5, 0.8, 0.9]``.
            Defaults to ``[0.5, 0.8, 0.9, 0.95]``.
        data : pd.DataFrame or None
            Input data for prediction.
        model_name : str or None
            Specific model to use. None = best model.
        future_covariates : pd.DataFrame or None
            Future known covariate values.

        Returns
        -------
        pd.DataFrame
            DataFrame with time_col, target_col, and per-level
            ``{target}_q{level}_lower`` / ``{target}_q{level}_upper`` columns.
        """
        raise_if(ValueError, self.pipeline_ is None,
                 "SmartRouter has not been fitted yet. Call fit() first.")

        if n is None:
            n = self.n_predict or self.strategy_['lags']

        if data is not None:
            data = self._apply_preprocessing(data)

        return self.pipeline_.predict_quantiles(
            n=n, levels=levels, data=data, model_name=model_name,
            future_covariates=future_covariates,
        )

    def update(self, new_data, update_epochs=50, refit_all=False):
        """Incrementally update fitted models with new data.

        Parameters
        ----------
        new_data : pd.DataFrame
            New observations with same columns as original training data.
        update_epochs : int, default 50
            Number of epochs for NN warm-start updates.
        refit_all : bool, default False
            If True, refit all models. If False, refit only the best.

        Returns
        -------
        self
        """
        raise_if(ValueError, self.pipeline_ is None,
                 "SmartRouter has not been fitted yet. Call fit() first.")

        new_data = self._apply_preprocessing(new_data)

        self.pipeline_.update(
            new_data, update_epochs=update_epochs, refit_all=refit_all
        )
        return self

    @property
    def strategy(self):
        """Return the selected strategy details."""
        if self.strategy_ is None:
            return None
        return deepcopy(self.strategy_)

    def plot(self, n=None, history_tail=None, lang='zh', figsize=(14, 5), show=True):
        """Plot forecast from the best model against history.

        Parameters
        ----------
        n : int or None
            Forecast horizon. Defaults to ``n_predict`` or ``lags``.
        history_tail : int or None
            Show only last N history points.
        lang : 'zh' or 'en'
        figsize : tuple
        show : bool

        Returns
        -------
        fig : matplotlib Figure
        """
        raise_if(ValueError, self.pipeline_ is None,
                 "SmartRouter has not been fitted yet.")
        if n is None:
            n = self.n_predict or self.strategy_.get('lags', 12)
        return self.pipeline_.plot(
            n=n, history_tail=history_tail, lang=lang,
            figsize=figsize, show=show,
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
        raise_if(ValueError, self.pipeline_ is None,
                 "SmartRouter has not been fitted yet.")
        return self.pipeline_.plot_leaderboard(
            lang=lang, figsize=figsize, show=show,
        )

    def get_model(self, model_name=None):
        """Retrieve a fitted model object from the underlying pipeline.

        Parameters
        ----------
        model_name : str or None, default None
            Name of the model to retrieve (as it appears in ``leader_board_``).
            ``None`` returns the best model.

        Returns
        -------
        model object
            The fitted model instance.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.
        """
        raise_if(ValueError, self.pipeline_ is None,
                 "SmartRouter has not been fitted yet.")
        return self.pipeline_.get_model(model_name)

    # ------------------------------------------------------------------
    #  Data Profiling (fast, vectorized)
    # ------------------------------------------------------------------

    def _profile_data(self, data):
        """Analyze data characteristics for routing decisions."""
        profile = DataProfile()

        # Multi-series: profile the longest series as representative
        if self.id_col is not None and self.id_col in data.columns:
            series_lengths = data.groupby(self.id_col).size()
            profile.n_series = len(series_lengths)
            longest_sid = series_lengths.idxmax()
            rep_data = data[data[self.id_col] == longest_sid].copy()
            # Use representative series for profiling
            values = rep_data[self.target_col].values.astype(np.float64)
            # Override n_rows with total count across all series
            profile._total_rows = len(data)
        else:
            profile.n_series = 1
            rep_data = data
            values = data[self.target_col].values.astype(np.float64)
        valid = values[~np.isnan(values)]

        # Basic stats — use representative series length for routing decisions
        profile.n_rows = len(values)
        profile.mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
        profile.std = float(np.std(valid)) if len(valid) > 0 else 1.0
        profile.cv = abs(profile.std / profile.mean) if abs(profile.mean) > 1e-10 else 0.0
        profile.skewness = float(pd.Series(valid).skew()) if len(valid) > 2 else 0.0
        profile.kurtosis = float(pd.Series(valid).kurtosis()) if len(valid) > 3 else 0.0
        profile.has_negative = bool(np.any(valid < 0))

        # Missing values
        n_nan = int(np.sum(np.isnan(values)))
        profile.pct_missing = n_nan / len(values) if len(values) > 0 else 0.0

        # Frequency detection (use representative series for multi-series)
        freq_data = rep_data if self.id_col is not None else data
        try:
            fd = FrequencyDetector(time_col=self.time_col)
            freq_info = fd.fit(freq_data, target_col=self.target_col)
            raw_freq = freq_info.get('freq')
            raw_td = freq_info.get('freq_timedelta')
            profile.freq_timedelta = raw_td
            profile.is_regular = freq_info.get('is_regular', True)
            profile.dominant_periods = freq_info.get('dominant_periods', [])

            # Normalize frequency string for month/quarter/year intervals
            norm_freq = self._normalize_freq(raw_td, freq_data)
            profile.freq = norm_freq or raw_freq

            # Monthly/quarterly data is conceptually regular even if
            # day-level timedeltas vary (28-31 days per month)
            if norm_freq in ('MS', 'QS', 'YS', 'ME', 'QE', 'YE'):
                profile.is_regular = True
        except Exception:
            pass

        # Stationarity
        if len(valid) >= 20:
            try:
                st = StationarityTest()
                result = st.fit(valid)
                profile.stationarity = result['conclusion']
                profile.suggested_d = st.suggest_differencing(valid)
            except Exception:
                pass

        # Trend strength (R² of linear fit)
        if len(valid) >= 10:
            x = np.arange(len(valid), dtype=np.float64)
            coeffs = np.polyfit(x, valid, 1)
            trend_line = np.polyval(coeffs, x)
            ss_res = np.sum((valid - trend_line) ** 2)
            ss_tot = np.sum((valid - np.mean(valid)) ** 2)
            profile.trend_strength = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Noise ratio
            residuals = valid - trend_line
            profile.noise_ratio = float(np.std(residuals) / profile.std) if profile.std > 0 else 1.0

        # Seasonality strength (spectral)
        if len(valid) >= 20:
            try:
                x = np.arange(len(valid), dtype=np.float64)
                coeffs = np.polyfit(x, valid, 1)
                detrended = valid - np.polyval(coeffs, x)
                fft_vals = np.fft.rfft(detrended)
                power = np.abs(fft_vals) ** 2
                total_power = np.sum(power[1:])  # exclude DC
                if total_power > 0 and len(profile.dominant_periods) > 0:
                    # Power at dominant frequency
                    n = len(detrended)
                    freqs = np.fft.rfftfreq(n)
                    dom_period = profile.dominant_periods[0]
                    dom_freq = 1.0 / dom_period if dom_period > 0 else 0
                    if dom_freq > 0:
                        idx = np.argmin(np.abs(freqs - dom_freq))
                        # Sum power in a small neighborhood
                        neighborhood = max(1, n // (dom_period * 4))
                        lo = max(1, idx - neighborhood)
                        hi = min(len(power), idx + neighborhood + 1)
                        dom_power = np.sum(power[lo:hi])
                        profile.seasonality_strength = float(dom_power / total_power)
            except Exception:
                pass

        # Outlier percentage (fast IQR)
        if len(valid) >= 10:
            q1, q3 = np.percentile(valid, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_mask = (valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)
                profile.pct_outlier = float(np.mean(outlier_mask))

        # Autocorrelation (fast, vectorized)
        if len(valid) >= 10:
            centered = valid - np.mean(valid)
            c0 = np.sum(centered ** 2)
            if c0 > 0:
                profile.autocorr_lag1 = float(
                    np.sum(centered[:-1] * centered[1:]) / c0
                )
                if len(valid) >= 20:
                    profile.autocorr_lag2 = float(
                        np.sum(centered[:-2] * centered[2:]) / c0
                    )

        # Multiple seasonalities (count significant spectral peaks)
        if len(profile.dominant_periods) > 1:
            profile.n_seasonalities = len(
                [p for p in profile.dominant_periods if p > 2]
            )
        elif len(profile.dominant_periods) == 1:
            profile.n_seasonalities = 1

        # Regime changes (simple: count sign changes in rolling mean diff)
        if len(valid) >= 30:
            try:
                win = max(5, len(valid) // 20)
                rolling_mean = np.convolve(
                    valid, np.ones(win) / win, mode='valid'
                )
                diffs = np.diff(rolling_mean)
                sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
                profile.regime_changes = int(sign_changes)
            except Exception:
                pass

        return profile

    def _insight_cache_key(self, data):
        time_start = None
        time_end = None
        target_mean = None
        target_std = None
        target_nan = None
        if self.time_col in data.columns:
            ts = pd.to_datetime(data[self.time_col], errors='coerce')
            if len(ts) > 0:
                time_start = str(ts.min())
                time_end = str(ts.max())
        if self.target_col in data.columns:
            numeric = pd.to_numeric(data[self.target_col], errors='coerce')
            arr = numeric.to_numpy(dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            target_nan = int(np.isnan(arr).sum())
            if len(finite) > 0:
                target_mean = round(float(np.mean(finite)), 8)
                target_std = round(float(np.std(finite)), 8)
        return (
            len(data),
            tuple(str(c) for c in data.columns),
            str(self.time_col),
            str(self.target_col),
            str(self.id_col),
            time_start,
            time_end,
            target_nan,
            target_mean,
            target_std,
        )

    @staticmethod
    def _time_regularity_stats(frame, time_col):
        if time_col not in frame.columns or len(frame) <= 1:
            return 1.0, 0, 0
        ts = pd.to_datetime(frame[time_col], errors='coerce').dropna()
        if len(ts) <= 1:
            return 1.0, 0, int(len(ts))
        unique_ts = ts.drop_duplicates().sort_values()
        if len(unique_ts) <= 1:
            return 1.0, 0, int(len(unique_ts))
        diffs = unique_ts.diff().dropna()
        if diffs.empty:
            return 1.0, 0, int(len(unique_ts))
        mode_delta = diffs.mode().iloc[0]
        regularity = float((diffs == mode_delta).mean())
        implicit = 0
        try:
            if mode_delta > pd.Timedelta(0):
                expected = int(round((unique_ts.iloc[-1] - unique_ts.iloc[0]) / mode_delta)) + 1
                implicit = max(0, expected - len(unique_ts))
        except Exception:
            implicit = 0
        return regularity, int(implicit), int(len(unique_ts))

    def _build_data_insights(self, data, profile):
        insights = DataInsightProfile()
        insights.n_rows_total = int(len(data))
        insights.n_columns = int(len(data.columns))
        insights.memory_mb = float(data.memory_usage(deep=True).sum() / 1024 / 1024)

        if self.time_col in data.columns:
            if self.id_col is not None and self.id_col in data.columns:
                dup_cols = [self.id_col, self.time_col]
            else:
                dup_cols = [self.time_col]
            insights.duplicate_timestamp_ratio = float(data.duplicated(dup_cols).mean()) if len(data) else 0.0
            insights.has_time_duplicates = insights.duplicate_timestamp_ratio > 0

        if self.id_col is not None and self.id_col in data.columns:
            lengths = data.groupby(self.id_col).size()
            if len(lengths) > 0:
                length_values = lengths.to_numpy(dtype=np.float64)
                insights.panel_min_length = int(np.min(length_values))
                insights.panel_median_length = float(np.median(length_values))
                insights.panel_max_length = int(np.max(length_values))
                mean_len = float(np.mean(length_values))
                insights.panel_length_cv = float(np.std(length_values) / (mean_len + 1e-12))
            weighted_reg = []
            implicit_total = 0
            unique_total = 0
            for _, sdf in data.groupby(self.id_col, sort=False):
                reg, implicit, unique_count = self._time_regularity_stats(sdf, self.time_col)
                weighted_reg.append(reg * max(unique_count - 1, 1))
                implicit_total += implicit
                unique_total += unique_count
            denom = sum(max(int(v), 1) for v in lengths.values) if len(lengths) else 1
            insights.regularity_ratio = float(sum(weighted_reg) / max(denom, 1))
            insights.panel_irregular_ratio = float(max(0.0, 1.0 - insights.regularity_ratio))
            insights.implicit_gap_ratio = float(implicit_total / max(unique_total + implicit_total, 1))
        else:
            reg, implicit, unique_count = self._time_regularity_stats(data, self.time_col)
            insights.regularity_ratio = reg
            insights.implicit_gap_ratio = float(implicit / max(unique_count + implicit, 1))

        insights.completeness_ratio = float(max(0.0, min(1.0, 1.0 - insights.implicit_gap_ratio)))
        insights.low_completeness = insights.completeness_ratio < 0.95

        if self.target_col in data.columns and len(data) > 0:
            numeric = pd.to_numeric(data[self.target_col], errors='coerce')
            values = numeric.to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            insights.explicit_nan_ratio = float(np.isnan(values).mean())
            insights.inf_ratio = float(np.isinf(values).mean())
            if len(finite) > 0:
                insights.zero_ratio = float(np.mean(finite == 0))
                insights.negative_ratio = float(np.mean(finite < 0))
                insights.intermittent_ratio = insights.zero_ratio

                diag_values = finite
                if self.id_col is not None and self.id_col in data.columns:
                    lengths = data.groupby(self.id_col).size()
                    if len(lengths) > 0:
                        longest_sid = lengths.idxmax()
                        sdf = data[data[self.id_col] == longest_sid].sort_values(self.time_col)
                        diag_series = pd.to_numeric(sdf[self.target_col], errors='coerce')
                        diag_arr = diag_series.to_numpy(dtype=np.float64)
                        diag_values = diag_arr[np.isfinite(diag_arr)]
                else:
                    if self.time_col in data.columns:
                        sdf = data.sort_values(self.time_col)
                        diag_series = pd.to_numeric(sdf[self.target_col], errors='coerce')
                        diag_arr = diag_series.to_numpy(dtype=np.float64)
                        diag_values = diag_arr[np.isfinite(diag_arr)]
                try:
                    insights.spectral_entropy = spectral_entropy(diag_values)
                except Exception:
                    insights.spectral_entropy = None
                try:
                    insights.hurst_exponent = hurst_exponent(diag_values)
                except Exception:
                    insights.hurst_exponent = None

        insights.high_entropy = (
            insights.spectral_entropy is not None and
            insights.spectral_entropy > 0.75
        )
        insights.long_memory = (
            insights.hurst_exponent is not None and
            insights.hurst_exponent > 0.6
        )

        flags = []
        if insights.has_time_duplicates:
            flags.append('duplicate_timestamps')
        if insights.low_completeness:
            flags.append('low_completeness')
        if insights.explicit_nan_ratio > 0.05:
            flags.append('explicit_missing')
        if insights.inf_ratio > 0:
            flags.append('infinite_values')
        if insights.intermittent_ratio > 0.3:
            flags.append('intermittent_series')
        if insights.high_entropy:
            flags.append('high_entropy')
        if insights.long_memory:
            flags.append('long_memory')
        if insights.panel_length_cv > 0.5:
            flags.append('panel_length_imbalance')
        insights.risk_flags = flags
        return insights

    def _get_data_insights(self, data, profile=None):
        cache_key = self._insight_cache_key(data)
        if self.insights_ is not None and self._insights_cache_key == cache_key:
            return self.insights_
        insights = self._build_data_insights(data, profile)
        self._insights_cache_key = cache_key
        self.insights_ = insights
        return insights

    # ------------------------------------------------------------------
    #  Strategy Builder
    # ------------------------------------------------------------------

    def _build_strategy(self, p):
        """Build the full strategy: preprocessing + models + lags + FE + hyperparams."""
        # Score all models and store detailed breakdown
        self.model_scores_ = self._score_all_models(p)

        # Model selection: use user-pinned models or heuristic selection
        if self.include_models is not None:
            models = list(self.include_models)
        else:
            models = self._select_models(p)
            # Preserve diversity on medium/large series by keeping at least
            # one ML model in the heuristic pool when it is clearly viable.
            if p.n_rows >= 100 and not any(m in {'catboost', 'xgboost', 'random_forest', 'extra_forest', 'gc_forest', 'wide_gbrt', 'multi_output_model', 'multi_step_model', 'regressor_chain'} for m in models):
                ml_candidates = [m for m in sorted(self.model_scores_, key=lambda x: self.model_scores_[x]['total'], reverse=True)
                                 if m in {'catboost', 'xgboost', 'random_forest', 'extra_forest', 'gc_forest', 'wide_gbrt', 'multi_output_model', 'multi_step_model', 'regressor_chain'}]
                if ml_candidates:
                    models = models[:-1] + [ml_candidates[0]]

        strategy = {
            'preprocessing': self._select_preprocessing(p),
            'models': models,
            'lags': self._suggest_lags(p),
            'scaler': self._select_scaler(p),
            'gbdt_differential_n': self._select_differencing(p),
            'feature_engineering': self._select_feature_engineering(p),
            'model_hyperparams': self._suggest_hyperparams(p),
        }
        return strategy

    def _score_all_models(self, p):
        """Score all models and return detailed breakdown.

        Returns
        -------
        dict
            model_name -> {'total': float, 'reasons': list of (reason, delta)}
        """
        all_models = list(get_all_available_models().keys())
        scores = {}
        suggested_lags = self._suggest_lags(p)
        for m in all_models:
            total, reasons = self._score_model(m, p)
            feasibility = self._model_feasibility(m, p, lags=suggested_lags)
            if feasibility['penalty'] > 0:
                total -= feasibility['penalty']
                reasons = list(reasons)
                reasons.append(('feasibility_risk:' + '|'.join(feasibility['reasons']), -feasibility['penalty']))
            scores[m] = {'total': total, 'reasons': reasons}
        return scores

    @classmethod
    def _category_for_model(cls, model_name):
        for category, models in cls._CATEGORIES.items():
            if model_name in models:
                return category
        return 'unknown'

    @classmethod
    def _speed_tier_for_model(cls, model_name):
        category = cls._category_for_model(model_name)
        if category in ('statistic', 'ml'):
            return 0
        if category == 'nn_light':
            return 1
        if category == 'nn_medium':
            return 2
        return 3

    def _model_feasibility(self, model_name, p, lags=None):
        n = max(0, int(getattr(p, 'n_rows', 0)))
        lags = int(lags if lags is not None else max(4, min(max(4, int(np.sqrt(max(n, 1)))), max(4, n // 4))))
        horizon = int(self.n_predict or min(12, max(1, n // 10)))
        horizon = max(1, horizon)
        category = self._category_for_model(model_name)
        tier = self._speed_tier_for_model(model_name)
        available_windows = n - lags - horizon + 1
        risk = 0.0
        hard_block = False
        reasons = []
        insights = self.insights_

        if n <= max(lags + horizon + 2, lags * 2):
            hard_block = True
            risk += 1.0
            reasons.append('insufficient_windows')
        elif available_windows < max(3, min(10, horizon)):
            risk += 0.45
            reasons.append('few_windows')

        if category == 'nn_heavy':
            if n < max(80, horizon * 6):
                hard_block = True
                risk += 0.9
                reasons.append('heavy_nn_too_little_data')
            elif n < max(160, horizon * 10):
                risk += 0.45
                reasons.append('heavy_nn_data_risk')
        elif category == 'nn_medium':
            if n < max(50, horizon * 4):
                risk += 0.35
                reasons.append('medium_nn_data_risk')
        elif model_name == 'gc_forest' and n < 80:
            risk += 0.35
            reasons.append('cascade_small_data_risk')
        elif model_name == 'wide_gbrt' and n < 60:
            risk += 0.25
            reasons.append('wide_gbrt_small_data_risk')

        if self.quantile is not None and category.startswith('nn') and n < 120:
            risk += 0.35
            reasons.append('nn_quantile_small_data_risk')

        if getattr(p, 'pct_missing', 0.0) > 0.08 and category.startswith('nn'):
            risk += 0.2
            reasons.append('nn_missing_data_risk')

        if insights is not None:
            if insights.low_completeness and category.startswith('nn'):
                risk += 0.25
                reasons.append('nn_low_completeness_risk')
            if insights.has_time_duplicates and category.startswith('nn'):
                risk += 0.15
                reasons.append('nn_duplicate_time_risk')
            if insights.intermittent_ratio > 0.3:
                if category.startswith('nn'):
                    risk += 0.25
                    reasons.append('nn_intermittent_series_risk')
                elif model_name in ('auto_arima', 'prophet'):
                    risk += 0.15
                    reasons.append('stat_intermittent_series_risk')
            if insights.high_entropy and tier >= 2:
                risk += 0.15
                reasons.append('high_entropy_complex_model_risk')
            if insights.panel_length_cv > 0.5 and category.startswith('nn'):
                risk += 0.2
                reasons.append('nn_panel_imbalance_risk')

        if self.time_limit is not None:
            budget_per_model = self.time_limit / max(1, self.max_models)
            if tier >= 3 and budget_per_model < 45:
                risk += 0.35
                reasons.append('heavy_model_time_risk')
            elif tier >= 2 and budget_per_model < 20:
                risk += 0.25
                reasons.append('medium_model_time_risk')

        if model_name in NN_FOUNDATION_MODEL_KEYS and self.time_limit is not None:
            if self.time_limit / max(1, self.max_models) < 30:
                risk += 0.25
                reasons.append('foundation_model_time_risk')

        risk = min(1.0, float(risk))
        penalty = 100.0 if hard_block else risk * 24.0
        return {
            'hard_block': bool(hard_block),
            'risk': risk,
            'penalty': penalty,
            'reasons': reasons,
            'category': category,
            'speed_tier': tier,
            'available_windows': int(available_windows),
        }

    def _filter_feasible_models(self, models, p, lags=None, keep_at_least=1):
        report = {}
        feasible = []
        for m in models:
            info = self._model_feasibility(m, p, lags=lags)
            report[m] = info
            if not info['hard_block']:
                feasible.append(m)
        if len(feasible) < min(keep_at_least, len(models)):
            ranked = sorted(
                models,
                key=lambda m: (
                    report[m]['hard_block'],
                    report[m]['risk'],
                    self._speed_tier_for_model(m),
                )
            )
            needed = min(keep_at_least, len(models))
            feasible = []
            for m in ranked:
                if m not in feasible:
                    feasible.append(m)
                if len(feasible) >= needed:
                    break
        if self._feasibility_report is None:
            self._feasibility_report = {}
        self._feasibility_report.update(report)
        return feasible

    def _expected_speed_score(self, model_name, p=None):
        tier_score = {0: 1.0, 1: 0.78, 2: 0.48, 3: 0.25}
        score = tier_score.get(self._speed_tier_for_model(model_name), 0.35)
        n = getattr(p, 'n_rows', 0) if p is not None else 0
        if n >= 800 and self._category_for_model(model_name).startswith('nn'):
            score += 0.08
        if self.quantile is not None and self._category_for_model(model_name).startswith('nn'):
            score -= 0.08
        return float(max(0.05, min(1.0, score)))

    def _build_adaptive_candidate_pool(self, p, pool_size):
        all_models = list(get_all_available_models().keys())
        active_strategy = self.strategy_ or {}
        lags = active_strategy.get('lags', self._suggest_lags(p))
        feasible = self._filter_feasible_models(
            all_models, p, lags=lags,
            keep_at_least=min(pool_size, len(all_models))
        )
        if self.search_strategy == 'thorough':
            return feasible

        score_map = self.model_scores_ or {}
        scored = []
        for m in feasible:
            info = self._model_feasibility(m, p, lags=lags)
            prior = score_map.get(m, {}).get('total', 50.0)
            safe_bonus = 4.0 if m in self._SAFE_PORTFOLIO else 0.0
            speed_bonus = self._expected_speed_score(m, p) * 5.0
            scored.append((m, prior - info['penalty'] + safe_bonus + speed_bonus))
        ranked = [m for m, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        pool = []

        for m in active_strategy.get('models', []):
            if m in feasible and m not in pool:
                pool.append(m)

        for category, category_models in self._CATEGORIES.items():
            best = [m for m in ranked if m in category_models]
            if best and best[0] not in pool:
                pool.append(best[0])

        for m in self._SAFE_PORTFOLIO:
            if m in feasible and m not in pool:
                pool.append(m)
            if len(pool) >= pool_size:
                break

        for m in ranked:
            if len(pool) >= pool_size:
                break
            if m not in pool:
                pool.append(m)

        freq = str(getattr(p, 'freq', '') or '').upper()
        horizon = int(self.n_predict or min(12, max(1, p.n_rows // 10)))
        foundation_family = [
            m for m in (
                'chronos_2_small', 'tirex_foundation', 'sundial',
                'time_moe', 'chronos_2', 'chronos_2_synth',
            )
            if m in NN_FOUNDATION_MODEL_KEYS
        ]
        foundation_available = [m for m in foundation_family if m in feasible]
        any_foundation_in_pool = any(m in pool for m in foundation_family)
        keep_foundation_family = (
            bool(foundation_available) and p.n_rows <= 180 and horizon <= 12 and
            p.is_regular and freq.startswith(('M', 'Q', 'A', 'Y')) and
            (p.seasonality_strength > 0.25 or p.n_seasonalities >= 2) and
            p.trend_strength > 0.4
        )
        if keep_foundation_family and not any_foundation_in_pool:
            inject = foundation_available[0]  # highest-scored by heuristic (sorted earlier)
            if len(pool) >= pool_size:
                protected = set(active_strategy.get('models', []))
                replace_idx = next(
                    (i for i in range(len(pool) - 1, -1, -1) if pool[i] not in protected),
                    len(pool) - 1,
                )
                pool[replace_idx] = inject
            else:
                pool.append(inject)

        return pool[:pool_size]

    def _select_preprocessing(self, p):
        """Determine which preprocessing steps to apply."""
        steps = []
        insights = self.insights_
        missing_ratio = p.pct_missing
        gap_ratio = 0.0
        if insights is not None:
            missing_ratio = max(missing_ratio, insights.explicit_nan_ratio, insights.inf_ratio)
            gap_ratio = insights.implicit_gap_ratio

        # Missing value handling
        if missing_ratio > 0.001:
            if missing_ratio < 0.05:
                steps.append({'step': 'fill_missing', 'method': 'linear'})
            elif missing_ratio < 0.15:
                steps.append({'step': 'fill_missing', 'method': 'ffill'})
            else:
                steps.append({'step': 'fill_missing', 'method': 'linear'})

        # Implicit gap filling — only for truly irregular data
        # Skip for monthly/quarterly which appear irregular at day level
        if not p.is_regular and p.freq not in (
            'MS', 'ME', 'QS', 'QE', 'YS', 'YE', None
        ):
            steps.append({'step': 'reindex_gaps', 'method': 'linear'})
        elif gap_ratio > 0.02 and p.freq not in (
            'MS', 'ME', 'QS', 'QE', 'YS', 'YE', None
        ):
            steps.append({'step': 'reindex_gaps', 'method': 'linear'})

        # Outlier handling
        if p.pct_outlier > 0.02:
            steps.append({'step': 'clip_outliers', 'method': 'iqr', 'threshold': 3.0})

        return steps

    def _select_scaler(self, p):
        """Choose the best scaler based on data distribution."""
        # Highly skewed positive data → PowerTransformer (Box-Cox)
        if abs(p.skewness) > 2.0 and not p.has_negative:
            return PowerTransformer(method='box-cox', standardize=True)

        # Highly skewed with negatives → QuantileTransformer
        if abs(p.skewness) > 2.0 and p.has_negative:
            return QuantileTransformer(
                output_distribution='normal',
                n_quantiles=min(1000, max(10, p.n_rows // 2))
            )

        # High kurtosis (heavy tails) → StandardScaler (more robust than MinMax)
        if p.kurtosis > 5.0:
            return StandardScaler()

        # Default → MinMaxScaler (works well for most NN and ML models)
        return MinMaxScaler()

    def _select_differencing(self, p):
        """Determine differencing order for GBDT models."""
        if p.stationarity in ('non_stationary', 'difference_stationary'):
            return min(p.suggested_d, 2)
        return 0

    def _select_feature_engineering(self, p):
        """Decide which feature engineering options to enable per model.

        Returns a dict of feature engineering decisions that will be
        translated into model_init_kwargs.
        """
        fe = {}
        insights = self.insights_
        avoid_adaptive = (
            insights is not None and
            (insights.high_entropy or insights.low_completeness or insights.panel_length_cv > 0.7)
        )

        # NN routing_mode: adaptive MoE for larger datasets with patterns
        if (not avoid_adaptive) and p.n_rows >= 200 and (
            p.seasonality_strength > 0.1 or p.trend_strength > 0.3
        ):
            fe['routing_mode'] = 'adaptive'
        else:
            fe['routing_mode'] = 'static'

        # Prophet lag features: useful when strong autocorrelation
        if (
            p.autocorr_lag1 > 0.5 or
            (insights is not None and insights.long_memory)
        ) and p.n_rows >= 50:
            fe['prophet_use_lag_features'] = True
        else:
            fe['prophet_use_lag_features'] = False

        # Prophet seasonality mode: multiplicative when amplitude scales
        if p.seasonality_strength > 0.2 and p.trend_strength > 0.3:
            fe['prophet_seasonality_mode'] = 'multiplicative'
        else:
            fe['prophet_seasonality_mode'] = 'auto'

        return fe

    def _nn_architecture_profile(self, p):
        horizon = self.n_predict or min(12, max(1, p.n_rows // 10))
        horizon = max(1, int(horizon))
        n = max(1, int(p.n_rows))
        long_horizon = horizon >= 24 or horizon / n > 0.15
        insights = self.insights_
        complex_pattern = (
            n >= 500 or
            getattr(p, 'n_series', 1) > 1 or
            long_horizon or
            p.pct_outlier > 0.02 or
            p.kurtosis > 5.0 or
            p.cv > 0.5 or
            p.noise_ratio > 0.9 or
            (n >= 300 and p.trend_strength > 0.55) or
            (n >= 300 and p.n_seasonalities >= 2) or
            p.regime_changes > max(20, n // 5) or
            (
                insights is not None and
                (insights.high_entropy or insights.low_completeness or insights.panel_length_cv > 0.7)
            )
        )
        light_ok = (
            n <= 300 and
            horizon <= 12 and
            n >= max(60, horizon * 5) and
            p.is_regular and
            not complex_pattern
        )
        return {
            'horizon': horizon,
            'light_ok': bool(light_ok),
            'capacity_needed': bool(complex_pattern),
        }

    def _suggest_nn_architecture_params(self, p):
        arch = self._nn_architecture_profile(p)
        if not arch['light_ok'] or (self.preset != 'fast' and p.n_rows > 180):
            return {}
        return {
            'n_beats__generic_architecture': False,
            'n_beats__num_stacks': 2,
            'n_beats__num_blocks': 1,
            'n_beats__num_layers': 2,
            'n_beats__layer_widths': 96,
            'n_beats__dropout': 0.05,
            'n_hits__num_stacks': 2,
            'n_hits__num_blocks': 1,
            'n_hits__num_layers': 2,
            'n_hits__layer_widths': 96,
            'n_hits__dropout': 0.05,
            'transformer__d_model': 32,
            'transformer__nhead': 2,
            'transformer__num_encoder_layers': 1,
            'transformer__dim_feedforward': 64,
            'transformer__dropout': 0.05,
            'gau__level': 1,
            'gau__dropout': 0.05,
            'tcn__num_levels': 2,
            'tcn__hidden_channels': 16,
            'tcn__dropout': 0.1,
            'tide__hidden_size': 64,
            'tide__decoder_output_dim': 16,
            'tide__temporal_decoder_hidden': 16,
            'tide__num_encoder_layers': 1,
            'tide__num_decoder_layers': 1,
            'tide__dropout': 0.05,
        }

    def _suggest_hyperparams(self, p):
        """Suggest model-specific hyperparameters based on data profile.

        Returns a dict in double-underscore format ready for ModelPipeline
        kwargs, e.g. {'catboost__iterations': 800}.

        Covers:
        - NN models: routing_mode, use_gtb, learning_rate, epochs, patience,
          EMA, SWA, warmup
        - GBDT: n_estimators, learning_rate, max_depth
        - Prophet: lag features, seasonality mode
        """
        params = {}
        n = p.n_rows

        # --- Feature engineering decisions ---
        fe = self._select_feature_engineering(p)

        nn_gtb_models = list(NN_GTB_MODEL_KEYS)
        nn_all = list(NN_MODEL_KEYS)

        # --- NN: routing_mode ---
        if fe.get('routing_mode') == 'adaptive':
            for m in nn_gtb_models:
                params[f'{m}__routing_mode'] = 'adaptive'

        # --- NN: use_gtb for complex patterns on medium+ data ---
        if n >= 200 and (p.seasonality_strength > 0.15 or p.n_seasonalities >= 2):
            for m in nn_gtb_models:
                params[f'{m}__use_gtb'] = True

        # --- NN: adaptive learning_rate ---
        # Lower LR for complex/heavy models, higher for simple ones
        nn_heavy = NN_KEYS_BY_CATEGORY['medium'] | NN_KEYS_BY_CATEGORY['heavy']
        nn_light = NN_KEYS_BY_CATEGORY['light']
        if n >= 300:
            # Larger data → can use standard LR
            for m in nn_heavy:
                params[f'{m}__learning_rate'] = 0.001
            for m in nn_light:
                params[f'{m}__learning_rate'] = 0.003
        elif n < 100:
            # Small data → lower LR to avoid overfitting
            for m in nn_heavy:
                params[f'{m}__learning_rate'] = 0.0005
            for m in nn_light:
                params[f'{m}__learning_rate'] = 0.001

        # --- NN: adaptive epochs ---
        # Ensure sufficient training for complex models
        if n >= 300:
            for m in nn_heavy:
                params[f'{m}__epochs'] = 3000
                params[f'{m}__patience'] = 100
            for m in nn_light:
                params[f'{m}__epochs'] = 2000
                params[f'{m}__patience'] = 50
        elif n >= 100:
            for m in nn_heavy:
                params[f'{m}__epochs'] = 2000
                params[f'{m}__patience'] = 60
            for m in nn_light:
                params[f'{m}__epochs'] = 1500
                params[f'{m}__patience'] = 40
        else:
            # Small data → fewer epochs but enough to converge
            for m in nn_all:
                params[f'{m}__epochs'] = 1000
                params[f'{m}__patience'] = 30

        # --- NN: EMA for training stability ---
        # EMA smooths weight oscillations; beneficial for medium+ data
        if n >= 100:
            for m in nn_heavy:
                params[f'{m}__use_ema'] = True
                params[f'{m}__ema_decay'] = 0.999

        # --- NN: SWA for late-stage averaging ---
        # SWA averages weights from last 25% of training; good for noisy data
        if n >= 200 and p.noise_ratio > 0.3:
            for m in nn_heavy:
                params[f'{m}__use_swa'] = True
                params[f'{m}__swa_start_frac'] = 0.75

        # --- NN: warmup for transformer-based models ---
        # Transformers benefit from LR warmup to stabilize early training
        transformer_models = set(NN_TRANSFORMER_LIKE_MODEL_KEYS)
        if n >= 100:
            for m in transformer_models:
                params[f'{m}__warmup_epochs'] = 10

        # --- NN: mHC-inspired residual gate ---
        # Sinkhorn-normalized residual gate prevents signal amplification;
        # most beneficial for noisy or non-stationary data where NN
        # training tends to oscillate
        if n >= 150 and (p.noise_ratio > 0.4 or
                         p.stationarity in ('non_stationary', 'difference_stationary')):
            for m in [model for model in nn_all if model not in {'d_linear', 'n_linear'}]:
                params[f'{m}__use_residual_gate'] = True

        params.update(self._suggest_nn_architecture_params(p))

        # --- Prophet ---
        if fe.get('prophet_use_lag_features'):
            params['prophet__use_lag_features'] = True
        if fe.get('prophet_seasonality_mode', 'auto') != 'auto':
            params['prophet__seasonality_mode'] = fe['prophet_seasonality_mode']

        # --- All tree models: verbose off by default ---
        for m in ['catboost', 'xgboost', 'random_forest', 'extra_forest',
                   'gc_forest', 'multi_output_model', 'multi_step_model', 'wide_gbrt']:
            params[f'{m}__verbose'] = False

        # --- Native tree models: adapt to data size ---
        # CatBoost uses 'iterations', others use 'n_estimators'
        if n >= 300:
            params['catboost__iterations'] = 800
            for m in ['xgboost', 'random_forest', 'extra_forest']:
                params[f'{m}__n_estimators'] = 800
            params['gc_forest__n_layers'] = 3
            params['gc_forest__n_estimators_per_layer'] = 150
        elif n < 100:
            params['catboost__iterations'] = 300
            for m in ['xgboost', 'random_forest', 'extra_forest']:
                params[f'{m}__n_estimators'] = 300
            params['gc_forest__n_layers'] = 2
            params['gc_forest__n_estimators_per_layer'] = 50
        # CatBoost/XGBoost: lower learning rate for noisy data
        if p.noise_ratio > 0.7:
            params['catboost__learning_rate'] = 0.03
            params['xgboost__learning_rate'] = 0.03
            params['xgboost__subsample'] = 0.7

        return params

    def _apply_training_budget_caps(self, hyperparams, models=None, per_model_budget=None, profile=None):
        params = dict(hyperparams or {})
        caps = {}
        arch = (
            self._nn_architecture_profile(profile)
            if profile is not None else {'light_ok': True, 'capacity_needed': False}
        )
        if models is None:
            models = list(get_all_available_models().keys())
        models = list(models)

        nn_models = {
            'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
            'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
            'patch_rnn', 'itransformer', 'srs_net', 'deepar',
        }
        tree_n_estimators = {
            'xgboost', 'random_forest', 'extra_forest',
            'multi_output_model', 'multi_step_model', 'wide_gbrt',
            'regressor_chain',
        }

        if self.preset == 'fast':
            epoch_cap = 25
            patience_cap = 8
            tree_cap = 40
            cat_cap = 40
            gc_layer_est_cap = 30
        elif self.preset == 'medium_quality' or self.preset is None:
            epoch_cap = 800
            patience_cap = 80
            tree_cap = 200
            cat_cap = 200
            gc_layer_est_cap = 80
        elif self.preset == 'high_quality':
            epoch_cap = 1500
            patience_cap = 120
            tree_cap = 500
            cat_cap = 500
            gc_layer_est_cap = 120
        else:
            epoch_cap = None
            patience_cap = None
            tree_cap = None
            cat_cap = None
            gc_layer_est_cap = None

        if per_model_budget is not None:
            if per_model_budget < 15:
                time_epoch_cap, time_patience_cap, time_tree_cap, time_cat_cap = 30, 8, 30, 30
            elif per_model_budget < 30:
                time_epoch_cap, time_patience_cap, time_tree_cap, time_cat_cap = 100, 15, 60, 60
            elif per_model_budget < 60:
                time_epoch_cap, time_patience_cap, time_tree_cap, time_cat_cap = 300, 40, 120, 120
            elif per_model_budget < 120:
                time_epoch_cap, time_patience_cap, time_tree_cap, time_cat_cap = 500, 60, 200, 200
            else:
                time_epoch_cap = time_patience_cap = time_tree_cap = time_cat_cap = None

            if time_epoch_cap is not None:
                epoch_cap = min(epoch_cap, time_epoch_cap) if epoch_cap is not None else time_epoch_cap
            if time_patience_cap is not None:
                patience_cap = min(patience_cap, time_patience_cap) if patience_cap is not None else time_patience_cap
            if time_tree_cap is not None:
                tree_cap = min(tree_cap, time_tree_cap) if tree_cap is not None else time_tree_cap
            if time_cat_cap is not None:
                cat_cap = min(cat_cap, time_cat_cap) if cat_cap is not None else time_cat_cap

        def _cap(key, cap):
            if cap is None:
                return
            current = params.get(key)
            if current is None or current > cap:
                params[key] = cap
                caps[key] = cap

        for m in models:
            if m in nn_models:
                _cap(f'{m}__epochs', epoch_cap)
                _cap(f'{m}__patience', patience_cap)
                if self.preset == 'fast':
                    params[f'{m}__batch_size'] = 512
                    params[f'{m}__lr_scheduler'] = None
                    params[f'{m}__restore_best_weights'] = False
                    params[f'{m}__use_gtb'] = False
                    params[f'{m}__routing_mode'] = 'static'
                    params[f'{m}__use_residual_gate'] = False
                    params[f'{m}__use_ema'] = False
                    params[f'{m}__use_swa'] = False
                    params[f'{m}__warmup_epochs'] = 0
                if m == 'tcn' and self.preset == 'fast' and arch['light_ok']:
                    params.setdefault('tcn__num_levels', 2)
                    params.setdefault('tcn__hidden_channels', 16)
                    params.setdefault('tcn__dropout', 0.1)
                elif m == 'gau' and self.preset == 'fast' and arch['light_ok']:
                    params.setdefault('gau__level', 1)
                    params.setdefault('gau__dropout', 0.05)
                elif m == 'transformer' and self.preset == 'fast':
                    if arch['light_ok']:
                        params.setdefault('transformer__d_model', 32)
                        params.setdefault('transformer__nhead', 2)
                        params.setdefault('transformer__num_encoder_layers', 1)
                        params.setdefault('transformer__dim_feedforward', 64)
                    params.setdefault('transformer__dropout', 0.05)
                elif m == 'n_hits' and self.preset == 'fast':
                    if arch['light_ok']:
                        params.setdefault('n_hits__num_stacks', 2)
                        params.setdefault('n_hits__num_blocks', 1)
                        params.setdefault('n_hits__num_layers', 2)
                        params.setdefault('n_hits__layer_widths', 96)
                    params.setdefault('n_hits__dropout', 0.05)
                elif m == 'n_beats' and self.preset == 'fast':
                    if arch['light_ok']:
                        params.setdefault('n_beats__generic_architecture', False)
                        params.setdefault('n_beats__num_stacks', 2)
                        params.setdefault('n_beats__num_blocks', 1)
                        params.setdefault('n_beats__num_layers', 2)
                        params.setdefault('n_beats__layer_widths', 96)
                    params.setdefault('n_beats__dropout', 0.05)
                elif m == 'tide' and self.preset == 'fast':
                    if arch['light_ok']:
                        params.setdefault('tide__hidden_size', 64)
                        params.setdefault('tide__decoder_output_dim', 16)
                        params.setdefault('tide__temporal_decoder_hidden', 16)
                        params.setdefault('tide__num_encoder_layers', 1)
                        params.setdefault('tide__num_decoder_layers', 1)
                    params.setdefault('tide__dropout', 0.05)
            elif m == 'catboost':
                _cap('catboost__iterations', cat_cap)
            elif m in tree_n_estimators:
                _cap(f'{m}__n_estimators', tree_cap)
            elif m == 'gc_forest':
                _cap('gc_forest__n_estimators_per_layer', gc_layer_est_cap)
                if self.preset == 'fast' or (per_model_budget is not None and per_model_budget < 60):
                    _cap('gc_forest__n_layers', 2)

        return params, caps

    def _apply_user_model_kwargs(self, hyperparams):
        """Apply user-specified epochs and model_kwargs on top of auto hyperparams.

        Priority (highest wins):
            model_kwargs  >  global epochs  >  auto hyperparams / budget caps
        """
        if self.epochs is None and not self.model_kwargs:
            return hyperparams
        params = dict(hyperparams)
        if self.epochs is not None:
            for m in NN_MODEL_KEYS:
                key = f'{m}__epochs'
                # Only override if not already overridden by per-model model_kwargs
                if key not in self.model_kwargs:
                    params[key] = self.epochs
        # Per-model kwargs always win (highest priority)
        params.update(self.model_kwargs)
        return params

    def _suggest_lags(self, p):
        """Suggest optimal lags based on data characteristics."""
        n = p.n_rows

        # If user specified n_predict, lags should be >= n_predict
        min_lags = self.n_predict if self.n_predict else 4

        # Base lags from dominant period — use the largest period that fits
        if p.dominant_periods:
            # Prefer a full seasonal cycle if it fits
            candidates = sorted(p.dominant_periods, reverse=True)
            base_lags = candidates[0]
            # If the largest period is too big, try the next one
            for c in candidates:
                if c * 3 <= n:  # need at least 3x periods of data
                    base_lags = c
                    break
            else:
                base_lags = max(min_lags, candidates[-1])
        else:
            # Heuristic: sqrt(n) capped
            base_lags = max(8, int(np.sqrt(n)))

        # Ensure lags cover at least n_predict
        base_lags = max(base_lags, min_lags)

        insights = self.insights_
        if insights is not None:
            if insights.long_memory and n >= min_lags * 6:
                base_lags = max(base_lags, min(n // 5, max(min_lags, int(np.sqrt(n) * 2))))
            if insights.high_entropy and not p.dominant_periods:
                base_lags = min(base_lags, max(min_lags, int(np.sqrt(n))))

        # Ensure lags are reasonable relative to data length
        max_lags = max(min_lags, n // 4)
        lags = min(base_lags, max_lags)

        # Round to nice numbers for efficiency
        if lags > 16:
            lags = (lags // 4) * 4  # multiple of 4
        lags = max(lags, 4)  # minimum 4

        # Ensure we have enough data: need at least 2 * lags
        if lags * 2 >= n:
            lags = max(4, n // 3)

        return int(lags)

    def _select_models(self, p, n_candidates=None):
        """Select best model candidates from pre-computed scores.

        Uses a 5-category diversity system to ensure architecture variety:
        - statistic: auto_arima, prophet
        - ml: catboost, xgboost, random_forest, extra_forest, gc_forest, wide_gbrt, ...
        - nn_light: d_linear, n_linear, tide, tcn
        - nn_medium: n_beats, n_hits, stacking_rnn, patch_rnn, time2vec, gau
        - nn_heavy: transformer, tft, itransformer, srs_net, deepar

        ML models are capped at 2 to prevent tree-model clones from
        dominating the selection.

        Parameters
        ----------
        p : DataProfile
            Data profile (unused here, scores already computed).
        n_candidates : int or None
            Number of candidates to select. Defaults to self.max_models.
        """
        if n_candidates is None:
            n_candidates = self.max_models

        scores = self.model_scores_
        all_models = list(get_all_available_models().keys())
        all_models = self._filter_feasible_models(
            all_models, p, lags=self._suggest_lags(p),
            keep_at_least=min(n_candidates, len(all_models))
        )

        # Sort by total score descending
        ranked = sorted(
            [(m, scores[m]['total']) for m in all_models],
            key=lambda x: x[1], reverse=True
        )

        # 5-category diversity system
        categories = self._CATEGORIES

        # Find the best model from each category
        category_best = {}
        for cat_name, cat_models in categories.items():
            cat_ranked = [(m, s) for m, s in ranked if m in cat_models]
            if cat_ranked:
                category_best[cat_name] = cat_ranked[0][0]

        selected = []
        remaining_budget = n_candidates

        # First pass: guarantee one from each category (priority order)
        # Prioritize NN categories to break the ML dominance
        diversity_order = ['ml', 'nn_light', 'nn_medium', 'nn_heavy', 'statistic']
        freq = str(getattr(p, 'freq', '') or '').upper()
        horizon = int(self.n_predict or min(12, max(1, p.n_rows // 10)))
        if (
            self.preset == 'fast' and p.n_rows <= 180 and horizon <= 12 and
            p.is_regular and freq.startswith(('M', 'Q', 'A', 'Y')) and
            (p.seasonality_strength > 0.25 or p.n_seasonalities >= 2) and
            p.trend_strength > 0.4
        ):
            diversity_order = ['ml', 'nn_medium', 'nn_heavy', 'statistic', 'nn_light']
        for cat_name in diversity_order:
            if cat_name in category_best and remaining_budget > 0:
                m = category_best[cat_name]
                if m not in selected:
                    selected.append(m)
                    remaining_budget -= 1

        # Second pass: fill remaining with top-ranked, but cap ML at 2
        ml_count = sum(1 for m in selected if m in categories['ml'])
        max_ml = 2

        for m, s in ranked:
            if remaining_budget <= 0:
                break
            if m in selected:
                continue
            # Enforce ML cap
            if m in categories['ml']:
                if ml_count >= max_ml:
                    continue
                ml_count += 1
            selected.append(m)
            remaining_budget -= 1

        # Re-sort selected by score so Pipeline trains highest-priority first
        selected.sort(key=lambda m: scores[m]['total'], reverse=True)

        return selected

    def _score_model(self, model_name, p):
        """Assign a suitability score to a model given the data profile.

        Higher score = more suitable. Base score is 50, adjusted by heuristics.
        Pattern bonuses are capped to prevent any single model from
        accumulating unlimited advantages.

        Returns
        -------
        tuple of (float, list)
            (total_score, list of (reason_string, delta_value))
        """
        score = 50.0
        reasons = [('base', 50.0)]
        n = p.n_rows
        pattern_bonus = 0.0  # Track cumulative pattern bonuses for capping

        def _add(delta, reason, is_pattern=False):
            nonlocal score, pattern_bonus
            if delta == 0:
                return
            if is_pattern:
                # Cap cumulative pattern bonuses at +25 per model
                headroom = 25.0 - pattern_bonus
                if headroom <= 0:
                    return
                delta = min(delta, headroom) if delta > 0 else delta
                pattern_bonus += max(delta, 0)
            score += delta
            reasons.append((reason, delta))

        # ---- Model category classification ----
        statistic_models = {'auto_arima', 'prophet', 'naive', 'seasonal_naive',
                            'theta', 'ets', 'short_trend_slot_blend',
                            'long_slot_trend_blend', 'stat_ensemble'}
        ml_models = {'catboost', 'xgboost', 'random_forest', 'extra_forest',
                      'gc_forest', 'wide_gbrt', 'multi_output_model',
                      'multi_step_model', 'regressor_chain'}
        nn_light = self._CATEGORIES['nn_light']
        nn_medium = self._CATEGORIES['nn_medium']
        nn_heavy = self._CATEGORIES['nn_heavy']

        # ---- Series length ----
        if n < 50:
            if model_name in statistic_models:
                _add(20, f'short_series(n={n}): stat model preferred')
            elif model_name in ml_models:
                _add(15, f'short_series(n={n}): ML model preferred')
            elif model_name in nn_light:
                _add(5, f'short_series(n={n}): light NN okay')
            elif model_name in nn_medium:
                _add(-10, f'short_series(n={n}): medium NN penalized')
            elif model_name in nn_heavy:
                _add(-20, f'short_series(n={n}): heavy NN penalized')
        elif n < 150:
            if model_name in ml_models:
                _add(15, f'small_series(n={n}): ML preferred')
            elif model_name in statistic_models:
                _add(12, f'small_series(n={n}): stat good')
            elif model_name in nn_light:
                _add(12, f'small_series(n={n}): light NN good')
            elif model_name in nn_medium:
                _add(5, f'small_series(n={n}): medium NN okay')
            elif model_name in nn_heavy:
                _add(-5, f'small_series(n={n}): heavy NN penalized')
        elif n < 500:
            # Medium series: balanced — all architectures are viable
            if model_name in nn_medium:
                _add(12, f'medium_series(n={n}): medium NN sweet spot')
            elif model_name in nn_light:
                _add(12, f'medium_series(n={n}): light NN good')
            elif model_name in ml_models:
                _add(12, f'medium_series(n={n}): ML good')
            elif model_name in nn_heavy:
                _add(10, f'medium_series(n={n}): heavy NN viable')
            elif model_name in statistic_models:
                _add(8, f'medium_series(n={n}): stat decent')
        else:
            if model_name in nn_heavy:
                _add(20, f'large_series(n={n}): heavy NN preferred')
            elif model_name in nn_medium:
                _add(15, f'large_series(n={n}): medium NN good')
            elif model_name in nn_light:
                _add(12, f'large_series(n={n}): light NN good')
            elif model_name in ml_models:
                _add(10, f'large_series(n={n}): ML good')
            elif model_name in statistic_models:
                _add(5, f'large_series(n={n}): stat okay')

        # ---- Stationarity (pattern bonus, capped) ----
        if p.stationarity in ('non_stationary', 'difference_stationary'):
            if model_name in ('auto_arima', 'theta', 'ets', 'stat_ensemble', 'd_linear'):
                _add(8, 'non_stationary: handles trends', is_pattern=True)
            elif model_name == 'prophet':
                _add(6, 'non_stationary: trend decomposition', is_pattern=True)
            elif model_name in ('n_beats', 'n_hits', 'tide'):
                _add(5, 'non_stationary: trend-capable', is_pattern=True)
            if model_name in ml_models:
                _add(3, 'non_stationary: GBDT+differencing', is_pattern=True)

        # ---- Seasonality (pattern bonus, capped) ----
        if p.seasonality_strength > 0.15:
            if model_name in ('n_beats', 'n_hits', 'tft', 'deepar'):
                _add(10, f'strong_seasonality({p.seasonality_strength:.2f}): seasonal specialist', is_pattern=True)
            elif model_name in ('prophet', 'auto_arima', 'seasonal_naive', 'theta',
                                'ets', 'stat_ensemble'):
                _add(8, 'strong_seasonality: seasonal decomposition', is_pattern=True)
            elif model_name in ('stacking_rnn', 'patch_rnn', 'tcn'):
                _add(6, 'strong_seasonality: handles seasonal', is_pattern=True)
            elif model_name in nn_light:
                _add(4, 'strong_seasonality: basic seasonal', is_pattern=True)

        # ---- Trend strength (pattern bonus, capped) ----
        if p.trend_strength > 0.5:
            if model_name in ('d_linear', 'n_linear', 'tide'):
                _add(8, f'strong_trend({p.trend_strength:.2f}): linear trend specialist', is_pattern=True)
            elif model_name in ('prophet', 'auto_arima', 'theta', 'ets', 'stat_ensemble'):
                _add(6, 'strong_trend: trend handling', is_pattern=True)

        # ---- Noise level ----
        if p.noise_ratio > 0.8:
            if model_name in ml_models:
                _add(8, f'high_noise({p.noise_ratio:.2f}): robust tree model')
            elif model_name in ('n_beats', 'tcn'):
                _add(5, 'high_noise: regularized NN')
            if model_name in ('srs_net', 'deepar'):
                _add(-5, 'high_noise: overfit risk')

        # ---- Skewness ----
        if abs(p.skewness) > 2.0:
            if model_name in ml_models:
                _add(5, f'high_skewness({p.skewness:.1f}): tree models robust')

        # ---- Autocorrelation structure (pattern bonus, capped) ----
        if p.autocorr_lag1 > 0.7:
            if model_name in ('auto_arima', 'theta', 'ets', 'stat_ensemble',
                              'stacking_rnn', 'patch_rnn', 'tcn'):
                _add(8, f'strong_autocorr({p.autocorr_lag1:.2f}): sequential model', is_pattern=True)
            elif model_name in ('gau', 'time2vec', 'tft', 'deepar'):
                _add(5, 'strong_autocorr: attention/temporal', is_pattern=True)
            elif model_name in ('d_linear', 'n_linear'):
                _add(3, 'strong_autocorr: linear temporal', is_pattern=True)
        elif p.autocorr_lag1 < 0.2:
            if model_name in ml_models:
                _add(5, f'weak_autocorr({p.autocorr_lag1:.2f}): tree model')
            elif model_name in ('transformer', 'itransformer', 'tft'):
                _add(5, 'weak_autocorr: attention model')

        # ---- Multiple seasonalities (pattern bonus, capped) ----
        if p.n_seasonalities >= 2:
            if model_name in ('tft', 'n_beats', 'deepar'):
                _add(8, f'multi_seasonal(n={p.n_seasonalities}): complex pattern', is_pattern=True)
            elif model_name in ('prophet', 'stat_ensemble', 'n_hits', 'itransformer', 'stacking_rnn'):
                _add(5, 'multi_seasonal: multi-scale model', is_pattern=True)
            if model_name in ('d_linear', 'n_linear'):
                _add(-3, 'multi_seasonal: too simple')

        # ---- Forecast horizon relative to data ----
        if self.n_predict and p.n_rows > 0:
            ratio = self.n_predict / p.n_rows
            if ratio > 0.2:
                if model_name in ('prophet', 'auto_arima', 'theta', 'ets', 'stat_ensemble'):
                    _add(5, f'long_horizon(ratio={ratio:.2f}): extrapolation model')
                elif model_name in ('d_linear', 'n_linear', 'tide'):
                    _add(3, 'long_horizon: linear extrapolation')
                if model_name in nn_heavy:
                    _add(-5, 'long_horizon: overfit risk for heavy NN')
            elif ratio < 0.05:
                if model_name in nn_heavy:
                    _add(3, f'short_horizon(ratio={ratio:.2f}): complex model viable')

        # ---- Regime changes ----
        if p.regime_changes > 5:
            if model_name in ml_models:
                _add(3, f'regime_changes({p.regime_changes}): tree handles discontinuities')
            elif model_name in ('tft', 'deepar', 'gau', 'itransformer'):
                _add(3, 'regime_changes: attention handles shifts')
            if model_name in ('auto_arima', 'd_linear', 'n_linear'):
                _add(-3, 'regime_changes: assumes smooth patterns')

        # ---- ML consistency bonus (for main tree models) ----
        if model_name in ('catboost', 'xgboost', 'random_forest'):
            _add(3, 'tree: proven baseline performer')

        # ---- NN models with GTB/routing capability bonus ----
        # Models that support use_gtb and routing_mode benefit from
        # adaptive expert selection on complex data
        nn_with_gtb = {
            'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
            'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
            'patch_rnn',
        }
        if model_name in nn_with_gtb and n >= 200:
            if p.seasonality_strength > 0.1 or p.trend_strength > 0.3:
                _add(5, 'GTB-capable: adaptive routing benefits complex data')

        # ---- Speed bonus for production ----
        if model_name in statistic_models:
            _add(2, 'speed: fast (statistic)')
        elif model_name in ml_models:
            _add(3, 'speed: fast native tree')

        insights = self.insights_
        if insights is not None:
            if insights.intermittent_ratio > 0.3:
                if model_name in ml_models:
                    _add(6, f'intermittent_series({insights.intermittent_ratio:.1%}): tree robust')
                elif model_name in nn_medium or model_name in nn_heavy:
                    _add(-5, 'intermittent_series: complex NN risk')
            if insights.long_memory:
                if model_name in ('tcn', 'n_hits', 'n_beats', 'patch_rnn', 'stacking_rnn'):
                    _add(5, 'long_memory: longer temporal receptive field', is_pattern=True)
                elif model_name in ('d_linear', 'n_linear'):
                    _add(-3, 'long_memory: linear window may underfit')
            if insights.high_entropy:
                if model_name in ml_models:
                    _add(4, 'high_entropy: robust non-parametric baseline')
                elif model_name in nn_heavy:
                    _add(-4, 'high_entropy: heavy NN overfit risk')
            if insights.low_completeness:
                if model_name in statistic_models or model_name in ml_models:
                    _add(3, 'low_completeness: robust to irregular history')
                elif model_name in nn_medium or model_name in nn_heavy:
                    _add(-4, 'low_completeness: NN window instability')
            if insights.panel_length_cv > 0.5:
                if model_name in ml_models:
                    _add(4, 'panel_length_imbalance: tree baseline robust')
                elif model_name in nn_heavy:
                    _add(-4, 'panel_length_imbalance: heavy NN risk')

        if self.preset == 'fast':
            if model_name in ('multi_output_model', 'multi_step_model'):
                _add(14, 'fast preset: ultra-low-latency ML model')
            elif model_name in ('prophet', 'auto_arima', 'stat_ensemble', 'theta', 'ets', 'seasonal_naive'):
                _add(10, 'fast preset: low-latency statistic model')
            elif model_name in ('random_forest', 'extra_forest'):
                _add(8, 'fast preset: low-latency model')
            elif model_name in ('d_linear', 'n_linear'):
                _add(6, 'fast preset: low-latency linear NN')
            elif model_name == 'tcn':
                _add(-25, 'fast preset: convolutional NN avoided')
            elif model_name in ('tide', 'n_hits'):
                _add(-12, 'fast preset: slower NN avoided')
            elif model_name in nn_medium or model_name in nn_heavy:
                _add(-15, 'fast preset: heavy NN avoided')

        # ---- Specific model strengths (conditional) ----
        if model_name == 'catboost':
            if p.noise_ratio > 0.7 and n >= 200:
                _add(3, 'catboost: robust for noisy large data')
        if model_name == 'xgboost':
            if n >= 150 and p.noise_ratio < 0.8:
                _add(3, 'xgboost: strong gradient boosting baseline')

        if model_name == 'prophet' and p.pct_missing > 0.01:
            _add(3, f'prophet: handles missing data ({p.pct_missing:.1%})')

        if model_name == 'n_beats' and p.noise_ratio < 0.5 and p.seasonality_strength > 0.1:
            _add(5, 'n_beats: clean periodic data specialist')

        if model_name == 'tide' and n >= 100:
            _add(3, 'tide: efficient for medium+ data')

        if model_name == 'itransformer' and n >= 200:
            _add(4, 'itransformer: long sequence specialist')

        if model_name == 'gau' and 100 <= n <= 500 and p.noise_ratio < 0.7:
            _add(4, 'gau: clean moderate-length specialist')

        if model_name == 'tcn' and p.autocorr_lag1 > 0.5:
            _add(4, 'tcn: local pattern specialist')

        if model_name == 'patch_rnn' and n >= 150:
            _add(4, 'patch_rnn: patch-based temporal modeling')

        if model_name == 'stacking_rnn' and p.autocorr_lag1 > 0.5:
            _add(4, 'stacking_rnn: deep sequential modeling')

        if model_name == 'deepar' and n >= 200 and p.noise_ratio < 0.7:
            _add(4, 'deepar: probabilistic forecaster for clean data')

        # ---- Native tree model specific scoring ----
        # NOTE: bonuses here are deliberately modest to avoid over-scoring
        # ML/tree models relative to NN models (see calibration analysis).
        if model_name == 'gc_forest':
            # gcForest cascade: modest bonus, only for favorable conditions
            if 100 <= n <= 800 and p.noise_ratio < 0.8:
                _add(3, 'gc_forest: cascade representation learning')
            elif n < 80:
                _add(-5, 'gc_forest: cascade overfits on small data')
        if model_name == 'extra_forest':
            # ExtraTrees: randomized splits → fast & low variance
            if n >= 100:
                _add(2, 'extra_forest: fast randomized ensemble')
            if p.noise_ratio > 0.5:
                _add(2, 'extra_forest: randomized splits reduce overfitting')

        if model_name in NN_FOUNDATION_MODEL_KEYS:
            if n < 100:
                _add(10, f'{model_name}: zero-shot excels on small data')
            elif n < 300:
                _add(5, f'{model_name}: zero-shot viable for medium data')
            if p.pct_missing > 0.01:
                _add(3, f'{model_name}: pretrained robustness to missing data ({p.pct_missing:.1%})')
            if p.n_seasonalities >= 2:
                _add(5, f'{model_name}: pretrained handles complex seasonality')
            if model_name in {'chronos_2_small', 'tirex_foundation', 'time_moe'}:
                _add(3, f'{model_name}: lightweight foundation variant')

        return score, reasons

    # ------------------------------------------------------------------
    #  Preprocessing Application
    # ------------------------------------------------------------------

    def _apply_preprocessing(self, data):
        """Apply the selected preprocessing steps to data."""
        if data is None:
            return None

        if self.strategy_ is None:
            return self._ensure_datetime(data.copy())

        df = self._ensure_datetime(data.copy())
        steps = self.strategy_['preprocessing']
        if self.target_col in df.columns:
            numeric_target = pd.to_numeric(df[self.target_col], errors='coerce')
            arr = numeric_target.to_numpy(dtype=np.float64)
            if np.isinf(arr).any():
                df[self.target_col] = numeric_target.replace([np.inf, -np.inf], np.nan)

        for step_cfg in steps:
            step = step_cfg['step']

            if step == 'fill_missing':
                handler = TimeSeriesMissingHandler(
                    time_col=self.time_col
                )
                value_cols = [self.target_col]
                df = handler.transform(
                    df,
                    method=step_cfg['method'],
                    value_cols=value_cols,
                    fill_implicit_gaps=False,
                )

            elif step == 'reindex_gaps':
                handler = TimeSeriesMissingHandler(
                    time_col=self.time_col
                )
                value_cols = [self.target_col]
                df = handler.transform(
                    df,
                    method=step_cfg['method'],
                    value_cols=value_cols,
                    fill_implicit_gaps=True,
                )

            elif step == 'clip_outliers':
                detector = TimeSeriesOutlierDetector(
                    time_col=self.time_col,
                    method=step_cfg['method'],
                    threshold=step_cfg.get('threshold', 3.0),
                )
                df = detector.transform(
                    df,
                    target_col=self.target_col,
                    strategy='clip',
                )

        return df

    # ------------------------------------------------------------------
    #  Search & Validation
    # ------------------------------------------------------------------

    def _should_screen(self):
        """Determine if quick screening is beneficial."""
        # When user has pinned specific models, skip screening —
        # no need to eliminate candidates since the model list is fixed.
        if self.include_models is not None:
            return False
        if self.search_strategy == 'basic':
            return False
        if self.search_strategy == 'thorough':
            return True
        # 'auto': screen when there are enough candidate models and data.
        # Threshold lowered to max_models>=3 so wide screening benefits
        # medium_quality preset (max_models=5) and custom configs.
        return (self.max_models >= 3 and
                self.profile_ is not None and
                self.profile_.n_rows >= 60)

    def _get_screen_pool_size(self):
        """Determine how many candidates to include in the screening pool.

        Wide pools let screening discover models that heuristic scoring
        might have ranked too low.  The pool is intentionally larger than
        max_models so the fusion step can re-rank based on actual metrics.

        Returns
        -------
        int
            Number of candidate models for the screening pool.
        """
        from PipelineTS.pipeline.pipeline import get_all_available_models
        total_models = len(get_all_available_models())

        if self.search_strategy == 'thorough':
            # Thorough: screen all available models
            return total_models
        else:
            # Auto: screen a wide pool but not everything
            base = max(self.max_models * 3, 12)
            if self.profile_ is not None and self.profile_.n_rows >= 500:
                base = max(base, self.max_models * 4)
            if (
                self.insights_ is not None and self.insights_.risk_flags and
                (self.time_limit is None or self.time_limit >= 120)
            ):
                base = max(base, self.max_models * 4)
            if self.time_limit is not None:
                if self.time_limit < 60:
                    base = max(self.max_models * 2, 8)
                elif self.time_limit >= 240:
                    base = max(base, self.max_models * 4)
            return min(total_models, base)

    def _inject_exploration_candidates(self, candidates, p):
        """Inject ε-greedy exploration candidates into the screening pool.

        Replaces the lowest heuristic-ranked models in the pool with models
        that heuristics would never have selected, prioritising categories
        not yet represented in the pool.  This breaks the circular dependency
        where the screening pool is entirely decided by the same heuristics
        we are trying to validate.

        Only active for ``search_strategy='auto'``.  For ``'thorough'`` the
        pool already covers all models; for ``'basic'`` screening is skipped.

        The injection fraction ε=0.2 means roughly 2-3 slots out of a
        typical pool of 12-15 are replaced with exploration candidates.

        Parameters
        ----------
        candidates : list of str
            Current screening pool (heuristic-selected).
        p : DataProfile
            Data profile (used only for logging context).

        Returns
        -------
        list of str
            Updated pool with exploration candidates injected.
        """
        if self.search_strategy != 'auto':
            return candidates

        all_models = self._filter_feasible_models(
            list(get_all_available_models().keys()),
            p,
            lags=self.strategy_['lags'] if self.strategy_ else None,
            keep_at_least=self.max_models
        )
        excluded = [m for m in all_models if m not in set(candidates)]
        if not excluded:
            return candidates  # pool already covers every registered model

        n_inject = max(1, round(len(candidates) * 0.2))
        n_inject = min(n_inject, len(excluded))

        # Category mapping (mirrors _select_models / _fusion_select)
        _categories = self._CATEGORIES
        model_cat = {}
        for cat_name, cat_models in _categories.items():
            for m in cat_models:
                model_cat[m] = cat_name

        covered_cats = {model_cat.get(m, 'unknown') for m in candidates}

        # Prefer injecting from categories that are absent in the pool, but
        # if the current pool already covers all categories, force a category
        # swap by prioritising ML/NN-heavy candidates over same-category clones.
        new_cat_excluded = [
            m for m in excluded
            if model_cat.get(m, 'unknown') not in covered_cats
        ]
        same_cat_excluded = [
            m for m in excluded
            if model_cat.get(m, 'unknown') in covered_cats
        ]

        def _explore_score(m):
            score_info = (self.model_scores_ or {}).get(m, {})
            prior = score_info.get('total', 50.0)
            feasibility = self._model_feasibility(
                m, p, lags=self.strategy_['lags'] if self.strategy_ else None
            )
            cat = model_cat.get(m, 'unknown')
            cat_bonus = 8.0 if cat not in covered_cats else 0.0
            safe_bonus = 4.0 if m in self._SAFE_PORTFOLIO else 0.0
            speed_bonus = self._expected_speed_score(m, p) * 3.0
            return prior + cat_bonus + safe_bonus + speed_bonus - feasibility['penalty']

        injections = []
        for m in sorted(new_cat_excluded, key=_explore_score, reverse=True):
            if len(injections) >= n_inject:
                break
            injections.append(m)
            covered_cats.add(model_cat.get(m, 'unknown'))

        # Phase 2: if we still need slots, preferentially inject at least one
        # NN-heavy or other higher-value exploration candidate before random fill.
        if len(injections) < n_inject:
            preferred = [m for m in same_cat_excluded if model_cat.get(m) == 'nn_heavy']
            if not preferred:
                preferred = [m for m in same_cat_excluded if model_cat.get(m) in {'nn_medium', 'ml'}]
            for m in sorted(preferred, key=_explore_score, reverse=True):
                if len(injections) >= n_inject:
                    break
                injections.append(m)

        # Final fill: random excluded models
        if len(injections) < n_inject:
            remaining = [m for m in excluded if m not in injections]
            remaining = sorted(remaining, key=_explore_score, reverse=True)
            injections.extend(remaining[: n_inject - len(injections)])

        # Drop bottom n_inject heuristic-ranked models from the current pool
        scores = self.model_scores_ or {}
        protected = set((self.strategy_ or {}).get('models', []))
        drop_candidates = [m for m in candidates if m not in protected]
        if len(drop_candidates) < len(injections):
            drop_candidates = list(candidates)
        candidates_by_score = sorted(
            drop_candidates,
            key=lambda m: scores.get(m, {}).get('total', 0.0),
            reverse=True,
        )
        drop_set = set(candidates_by_score[-len(injections):]) if injections else set()
        survivors = [m for m in candidates if m not in drop_set]
        new_pool = survivors + injections

        if self.verbose:
            self.logger.info(
                f"  Exploration: injected {len(injections)} candidate(s) "
                f"({', '.join(injections)}) replacing lowest-scored heuristic picks"
            )

        return new_pool

    def _should_explore_lags(self):
        """Determine if multi-lag exploration is beneficial."""
        if self.search_strategy == 'basic':
            return False
        if self.search_strategy == 'thorough':
            return True
        # 'auto': explore when data is large enough for meaningful comparison
        return (self.profile_ is not None and
                self.profile_.n_rows >= 100)

    def _active_hpo_strategy(self):
        if self.hpo_strategy in ('none', 'quick', 'full'):
            return self.hpo_strategy
        if self.profile_ is None or self.strategy_ is None:
            return 'none'
        if self.search_strategy == 'basic' and self.include_models is None:
            return 'none'
        if self.profile_.n_rows < 120:
            return 'none'
        if self.insights_ is not None and (
            self.insights_.low_completeness or
            self.insights_.high_entropy or
            self.insights_.panel_length_cv > 0.7
        ):
            return 'none'
        if self.time_limit is None:
            return 'none'
        if self.time_limit < 180:
            return 'none'
        return 'quick'

    def _should_hpo(self):
        self._active_hpo_strategy_ = self._active_hpo_strategy()
        return self._active_hpo_strategy_ != 'none'

    def _hpo_models_for_strategy(self, models):
        models = list(models or [])
        if self.hpo_strategy != 'auto':
            return models
        prioritized = [
            m for m in models
            if self._speed_tier_for_model(m) == 0
        ]
        if not prioritized:
            prioritized = models[:]
        return prioritized[:2]

    def _run_hpo(self, train_data, valid_data, base_hyperparams):
        """Run Optuna HPO for selected models.

        Parameters
        ----------
        train_data : pd.DataFrame
        valid_data : pd.DataFrame
        base_hyperparams : dict
            Existing double-underscore kwargs from _suggest_hyperparams.

        Returns
        -------
        dict
            Merged double-underscore kwargs with HPO-optimized values.
        """
        from PipelineTS.pipeline.hpo import OptunaHPO

        active_strategy = self._active_hpo_strategy_
        if active_strategy is None:
            active_strategy = self._active_hpo_strategy()
            self._active_hpo_strategy_ = active_strategy
        if active_strategy == 'none':
            self._hpo_results = {}
            return dict(base_hyperparams or {})

        models = self._hpo_models_for_strategy(self.strategy_['models'])
        if not models:
            self._hpo_results = {}
            return dict(base_hyperparams or {})
        lags = self.strategy_['lags']
        scaler = self.strategy_['scaler']

        n_trials = self.hpo_n_trials
        if active_strategy == 'quick':
            n_trials = min(n_trials, 5)
        if self.hpo_strategy == 'auto':
            n_trials = min(n_trials, 3)

        timeout_per_model = self.hpo_timeout_per_model
        if self.hpo_strategy == 'auto' and timeout_per_model is None:
            timeout_per_model = max(15.0, min(45.0, self.time_limit / max(8, len(models) * 4)))

        if self.verbose:
            self.logger.info(
                f"  Strategy: {self.hpo_strategy}->{active_strategy}, "
                f"{n_trials} trials/model, models={models}"
            )

        hpo = OptunaHPO(
            time_col=self.time_col,
            target_col=self.target_col,
            lags=lags,
            metric=self.metric,
            metric_less_is_better=self.metric_less_is_better,
            n_trials=n_trials,
            timeout_per_model=timeout_per_model,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        pipeline_kwargs = {
            'scaler': scaler,
            'accelerator': self.accelerator,
            'id_col': self.id_col,
            'known_covariates': self.known_covariates or None,
            'past_covariates': self.past_covariates or None,
            'gbdt_differential_n': self.strategy_.get('gbdt_differential_n', 0),
        }

        result = hpo.optimize(
            model_names=models,
            train_data=train_data,
            valid_data=valid_data,
            base_hyperparams=base_hyperparams,
            **pipeline_kwargs,
        )

        self._hpo_results = hpo.results_

        if self.verbose and hpo.results_:
            tuned = [f"{m}({r['best_value']:.4f})"
                     for m, r in hpo.results_.items()]
            self.logger.info(f"  HPO complete: {', '.join(tuned)}")

        return result

    def _quick_screen(self, train_data, valid_data, candidates, strategy):
        """Quick model screening with lightweight training.

        Trains lightweight versions of candidate models on a data subset,
        evaluates on holdout, and returns the full screening leaderboard.
        The caller uses ``_fusion_select`` to combine screening results
        with heuristic scores for final model selection.

        Parameters
        ----------
        train_data : pd.DataFrame
            Preprocessed training data.
        valid_data : pd.DataFrame
            Preprocessed validation data.
        candidates : list of str
            Broad pool of candidate model names.
        strategy : dict
            Current strategy dict.

        Returns
        -------
        pd.DataFrame or None
            Full screening leaderboard (sorted by metric), or None if
            screening failed entirely.
        """
        candidates = self._filter_feasible_models(
            list(candidates),
            self.profile_,
            lags=strategy['lags'],
            keep_at_least=min(self.max_models, len(candidates))
        )
        if len(candidates) <= self.max_models:
            return None  # no screening needed

        n = len(train_data)
        if self.id_col is not None and self.id_col in train_data.columns:
            n_effective = int(train_data.groupby(self.id_col).size().min())
        else:
            n_effective = n

        min_screen_rows = max(
            strategy['lags'] * 3,
            strategy['lags'] + (self.n_predict or strategy['lags']) + 4
        )
        if self.id_col is not None and self.id_col in train_data.columns:
            if n_effective > max(100, min_screen_rows * 2):
                parts = []
                for _, sdf in train_data.groupby(self.id_col, sort=False):
                    sdf = sdf.sort_values(self.time_col)
                    keep = min(len(sdf), max(min_screen_rows, int(len(sdf) * 0.7)))
                    parts.append(sdf.tail(keep))
                screen_train = pd.concat(parts, ignore_index=True)
            else:
                screen_train = train_data
        elif n_effective > max(100, min_screen_rows * 2):
            subset_start = int(n * 0.3)
            screen_train = train_data.iloc[subset_start:].reset_index(drop=True)
            if len(screen_train) < min_screen_rows:
                screen_train = train_data
        else:
            screen_train = train_data

        screen_valid = valid_data

        # Sort candidates by expected speed (fast-first) so time-limited
        # screening evaluates more diverse models before budget runs out.
        # Within each speed tier, the original heuristic-score order is kept.
        candidates = self._sort_by_speed(candidates)

        # Screening hyperparams: reduced complexity for speed
        screen_params = self._get_screening_hyperparams(candidates, strategy)

        # Time limit for screening: proportional to pool size.
        # Larger pools need more time.  Base is 30% of total budget;
        # scale up slightly for wide pools (capped at 45%).
        screen_time = None
        if self.time_limit is not None:
            base_frac = 0.30
            pool_scale = min(len(candidates) / 10.0, 1.5)  # 1.0 for 10, 1.5 for 15+
            screen_frac = min(base_frac * pool_scale, 0.45)
            screen_time = self.time_limit * screen_frac

        try:
            screen_pipeline = ModelPipeline(
                time_col=self.time_col,
                target_col=self.target_col,
                lags=strategy['lags'],
                include_models=candidates,
                scaler=deepcopy(strategy['scaler']),
                accelerator=self.accelerator,
                random_state=self.random_state,
                cv=min(self.cv, 2),  # fewer CV folds for speed
                id_col=self.id_col,
                known_covariates=self.known_covariates or None,
                past_covariates=self.past_covariates or None,
                gbdt_differential_n=strategy['gbdt_differential_n'],
                metric=self.metric,
                metric_less_is_better=self.metric_less_is_better,
                time_limit=screen_time,
                **screen_params,
            )

            screen_pipeline._device_info_logged = True
            screen_pipeline._phase_label = f"Wide Screening [{len(candidates)} candidates]"
            screen_lb = screen_pipeline.fit(screen_train, valid_data=screen_valid)

            if screen_lb.empty:
                if self.verbose:
                    self.logger.warning("  Screening: no models completed")
                return None

            self._screening_results = screen_lb

            if self.verbose:
                self.logger.info(
                    f"  Screening: {len(candidates)} candidates → "
                    f"{len(screen_lb)} evaluated"
                )

            return screen_lb

        except Exception as e:
            if self.verbose:
                self.logger.warning(
                    f"  Screening failed ({e}), using heuristic selection"
                )
            return None

    def _fusion_select(self, screen_lb, broad_candidates):
        """Combine screening results with heuristic scores for final selection.

        Uses weighted rank fusion: each model gets a fusion score that is
        a weighted combination of its normalized screening rank and its
        normalized heuristic rank.  Screening is weighted more heavily
        (α=0.7) because it reflects actual performance on the data,
        while heuristics are prior beliefs.

        After fusion scoring, the 5-category diversity constraint is
        applied to ensure architectural variety in the final selection.

        Parameters
        ----------
        screen_lb : pd.DataFrame or None
            Screening leaderboard from ``_quick_screen``.  None means
            screening was skipped or failed.
        broad_candidates : list of str
            The broad pool of candidates that entered screening.

        Returns
        -------
        list of str
            Final selected model names (length ≤ max_models).
        """
        # Fallback: if screening produced nothing, use heuristic selection
        if screen_lb is None or screen_lb.empty:
            if self.verbose:
                self.logger.info("  Fusion: no screening data, using heuristic selection")
            heuristic_models = (self.strategy_ or {}).get('models', [])[:self.max_models]
            if self.profile_ is None:
                return heuristic_models
            lags = (self.strategy_ or {}).get('lags', self._suggest_lags(self.profile_))
            fallback = self._filter_feasible_models(
                heuristic_models, self.profile_,
                lags=lags,
                keep_at_least=min(self.max_models, len(heuristic_models))
            )
            return fallback[:self.max_models]

        # --- Build normalized heuristic ranks ---
        # All candidates ranked by heuristic score (higher = better)
        heuristic_scores = {}
        if self.model_scores_:
            for m in broad_candidates:
                if m in self.model_scores_:
                    heuristic_scores[m] = self.model_scores_[m]['total']
        if not heuristic_scores:
            # No heuristic scores available, use screening only
            survivors = screen_lb.head(self.max_models)['model'].tolist()
            return survivors

        h_ranked = sorted(heuristic_scores.keys(),
                          key=lambda m: heuristic_scores[m], reverse=True)
        n_h = len(h_ranked)
        # Normalized: 1.0 for best, 0.0 for worst
        heuristic_norm = {}
        for i, m in enumerate(h_ranked):
            heuristic_norm[m] = 1.0 - (i / max(n_h - 1, 1))

        # --- Build normalized screening ranks ---
        screen_models = screen_lb['model'].tolist()
        n_s = len(screen_models)
        screening_norm = {}
        for i, m in enumerate(screen_models):
            screening_norm[m] = 1.0 - ((i + 1) / max(n_s + 1, 1))

        speed_norm = {}
        if 'train_cost(s)' in screen_lb.columns:
            costs = {}
            for _, row in screen_lb.iterrows():
                cost = float(row.get('train_cost(s)', 0.0) or 0.0)
                cost += float(row.get('eval_cost(s)', 0.0) or 0.0)
                costs[row['model']] = max(cost, 1e-6)
            if costs:
                log_costs = {m: np.log1p(c) for m, c in costs.items()}
                c_min = min(log_costs.values())
                c_max = max(log_costs.values())
                for m, c in log_costs.items():
                    speed_norm[m] = 1.0 - ((c - c_min) / max(c_max - c_min, 1e-9))

        # Models that were candidates but didn't complete screening
        # (timed out or failed) get a penalty: worst screening rank - 0.1
        worst_screen = 0.0
        for m in broad_candidates:
            if m not in screening_norm:
                screening_norm[m] = max(worst_screen - 0.1, -0.1)

        # --- Fusion scoring ---
        # α controls screening vs heuristic weight.
        # Screening is primary signal (α=0.7); heuristic is prior.
        alpha = 0.65
        speed_weight = 0.15 if self.time_limit is not None else 0.08
        heuristic_weight = max(0.0, 1.0 - alpha - speed_weight)

        fusion_scores = {}
        for m in broad_candidates:
            s_norm = screening_norm.get(m, -0.1)
            h_norm = heuristic_norm.get(m, 0.0)
            v_norm = speed_norm.get(m, self._expected_speed_score(m, self.profile_))
            risk = 0.0
            if self._feasibility_report and m in self._feasibility_report:
                risk = self._feasibility_report[m].get('risk', 0.0)
            fusion_scores[m] = (
                alpha * s_norm +
                heuristic_weight * h_norm +
                speed_weight * v_norm -
                0.12 * risk
            )

        # Store fusion details for logging
        self._fusion_scores = fusion_scores

        # --- Greedy diversity-aware selection ---
        # Walk down the fusion-ranked list.  Models from categories not
        # yet represented are selected immediately (diversity bonus);
        # models from already-covered categories are deferred to phase 2.
        # This ensures the strongest fusion performers are always
        # considered while diversity emerges naturally.
        categories = self._CATEGORIES

        # Build model → category mapping
        model_cat = {}
        for cat_name, cat_models in categories.items():
            for m in cat_models:
                model_cat[m] = cat_name

        # Rank all candidates by fusion score (descending)
        ranked = sorted(fusion_scores.keys(),
                        key=lambda m: fusion_scores[m], reverse=True)

        max_ml = 2
        selected = []
        represented_cats = set()
        remaining = self.max_models
        deferred = []

        # Phase 1: select models that add diversity, in fusion rank order
        for m in ranked:
            if remaining <= 0:
                break
            cat = model_cat.get(m, 'unknown')
            if cat not in represented_cats:
                # ML cap check
                if cat == 'ml':
                    ml_in = sum(1 for s in selected
                                if model_cat.get(s) == 'ml')
                    if ml_in >= max_ml:
                        deferred.append(m)
                        continue
                selected.append(m)
                represented_cats.add(cat)
                remaining -= 1
            else:
                deferred.append(m)

        # Phase 2: fill remaining slots from deferred (best fusion first)
        for m in deferred:
            if remaining <= 0:
                break
            cat = model_cat.get(m, 'unknown')
            if cat == 'ml':
                ml_in = sum(1 for s in selected
                            if model_cat.get(s) == 'ml')
                if ml_in >= max_ml:
                    continue
            selected.append(m)
            remaining -= 1

        # Re-sort selected by fusion score (highest first)
        selected.sort(key=lambda m: fusion_scores[m], reverse=True)

        if self.verbose:
            self.logger.info(
                f"  Fusion: {len(screen_lb)} screened + heuristic → "
                f"{len(selected)} selected (α={alpha})"
            )

        return selected

    @staticmethod
    def _sort_by_speed(candidates):
        """Sort model candidates by expected training speed (fast-first).

        Ensures time-limited screening evaluates diverse, fast models before
        budget runs out.  Within each speed tier the original ordering
        (heuristic score) is preserved via a stable sort.

        Speed tiers:
        - 0: statistic models (auto_arima, prophet) + ML/tree models
        - 1: light NN (d_linear, n_linear, tide, tcn)
        - 2: medium NN (n_beats, n_hits, stacking_rnn, patch_rnn, time2vec, gau)
        - 3: heavy NN (tft, transformer, itransformer, deepar, srs_net, chronos*)
        """
        # Stable sort preserves original (heuristic-score) order within tier
        return sorted(candidates, key=SmartRouter._speed_tier_for_model)

    def _get_screening_hyperparams(self, candidates, strategy):
        """Build lightweight hyperparams for quick screening.

        Reduces GBDT estimators and NN epochs for faster evaluation.
        """
        params = {}
        # Native tree models use n_estimators (or iterations for catboost)
        native_tree_models = {
            'catboost', 'xgboost', 'random_forest', 'extra_forest',
            'gc_forest',
        }
        # Legacy multi-output models also use n_estimators
        legacy_ml_models = {
            'multi_output_model', 'multi_step_model', 'wide_gbrt',
            'regressor_chain',
        }
        nn_models = {
            'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
            'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
            'patch_rnn', 'itransformer', 'srs_net', 'deepar',
        }

        for m in candidates:
            if m in native_tree_models:
                if m == 'catboost':
                    params[f'{m}__iterations'] = 30
                elif m == 'gc_forest':
                    params[f'{m}__n_estimators_per_layer'] = 30
                    params[f'{m}__n_layers'] = 2
                else:
                    params[f'{m}__n_estimators'] = 30
            elif m in legacy_ml_models:
                params[f'{m}__n_estimators'] = 30
            elif m in nn_models:
                params[f'{m}__epochs'] = 100

        # Merge with strategy hyperparams (screening overrides take priority)
        for k, v in strategy.get('model_hyperparams', {}).items():
            model_name = k.split('__')[0]
            if model_name in candidates and k not in params:
                params[k] = v

        return params

    def _explore_lags(self, full_data, models, strategy):
        """Try multiple lag values for each survivor model individually.

        Tests all survivor models at each lag candidate with lightweight
        params, then picks the optimal lag per model.  Returns a primary
        lag (max of per-model lags, for data splitting) and a dict of
        per-model best lags.

        Parameters
        ----------
        full_data : pd.DataFrame
            Full preprocessed data (before train/valid split).
        models : list of str
            Survivor model names to evaluate.
        strategy : dict
            Current strategy dict.

        Returns
        -------
        int
            Primary lag (max of per-model best lags, for data splitting).
        """
        base_lag = strategy['lags']
        n = len(full_data)
        models = self._filter_feasible_models(
            list(models), self.profile_, lags=base_lag,
            keep_at_least=min(len(models), self.max_models)
        )

        candidates = self._generate_lag_candidates(base_lag, n)
        if len(candidates) <= 1:
            self._lag_exploration_results = {}
            self._per_model_lags = {m: base_lag for m in models}
            return base_lag

        # Build lightweight hyperparams for all models (like screening)
        fast_params = self._get_screening_hyperparams(models, strategy)

        # Time budget for lag exploration: 15% of total if set
        lag_time_limit = None
        if self.time_limit is not None:
            lag_time_limit = self.time_limit * 0.15
        lag_t0 = time.time()

        # {model_name: {lag: metric}}
        model_lag_metrics = {m: {} for m in models}

        for lag in candidates:
            # Check time budget
            if lag_time_limit is not None:
                if time.time() - lag_t0 > lag_time_limit:
                    if self.verbose:
                        self.logger.info(
                            f"  Lag exploration time budget exhausted, "
                            f"tested {len([l for l in candidates if l <= lag]) - 1}/{len(candidates)} lags"
                        )
                    break

            try:
                split_train, split_valid = self._temporal_split(full_data, lag)
                if len(split_train) < lag * 2 or len(split_valid) < lag:
                    continue

                eval_pipeline = ModelPipeline(
                    time_col=self.time_col,
                    target_col=self.target_col,
                    lags=lag,
                    include_models=models,
                    scaler=deepcopy(strategy['scaler']),
                    accelerator=self.accelerator,
                    random_state=self.random_state,
                    cv=min(self.cv, 2),
                    id_col=self.id_col,
                    known_covariates=self.known_covariates or None,
                    past_covariates=self.past_covariates or None,
                    gbdt_differential_n=strategy['gbdt_differential_n'],
                    metric=self.metric,
                    metric_less_is_better=self.metric_less_is_better,
                    time_limit=lag_time_limit,
                    **fast_params,
                )
                eval_pipeline._device_info_logged = True
                eval_pipeline._phase_label = f"Lag Exploration [lag={lag}]"
                lb = eval_pipeline.fit(split_train, valid_data=split_valid)

                if not lb.empty:
                    for _, row in lb.iterrows():
                        mname = row['model']
                        metric = float(row['metric'])
                        if mname in model_lag_metrics:
                            model_lag_metrics[mname][lag] = metric
            except Exception:
                continue

        # Determine per-model best lag
        per_model_lags = {}
        for m in models:
            lag_metrics = model_lag_metrics.get(m, {})
            if lag_metrics:
                per_model_lags[m] = min(lag_metrics, key=lag_metrics.get)
            else:
                per_model_lags[m] = base_lag

        self._lag_exploration_results = model_lag_metrics
        self._per_model_lags = per_model_lags

        # Primary lag = max of per-model lags (for data splitting)
        primary_lag = max(per_model_lags.values()) if per_model_lags else base_lag
        return primary_lag

    def _generate_lag_candidates(self, base_lag, n_rows):
        """Generate 2-3 lag candidates around the base lag."""
        candidates = set()
        candidates.add(base_lag)

        # Smaller lag
        small_lag = max(4, base_lag * 2 // 3)
        if small_lag != base_lag and small_lag * 2 < n_rows:
            candidates.add(small_lag)

        # Larger lag (if data permits)
        large_lag = min(base_lag * 3 // 2, n_rows // 4)
        if large_lag > base_lag and large_lag * 2 < n_rows:
            # Round to multiple of 4
            large_lag = (large_lag // 4) * 4
            if large_lag > base_lag:
                candidates.add(large_lag)

        return sorted(candidates)

    def _pick_fast_eval_model(self, models):
        """Pick the fastest model from the list for lag evaluation."""
        fast_preference = [
            'stat_ensemble', 'theta', 'ets', 'seasonal_naive',
            'catboost', 'xgboost', 'random_forest', 'extra_forest',
            'multi_output_model', 'multi_step_model',
            'prophet', 'auto_arima',
            'd_linear', 'n_linear', 'tide',
        ]
        for m in fast_preference:
            if m in models:
                return m
        return models[0]

    # ------------------------------------------------------------------
    #  Ensemble Builder
    # ------------------------------------------------------------------

    def _build_ensemble(self):
        """Build an ensemble from top-K models after pipeline fit.

        Strategies:
        - 'auto': weighted_avg if multiple models are competitive (within 30%)
        - 'weighted_avg': inverse-metric weighted average
        - 'median': median of predictions (robust to outlier models)
        - 'stacking': Ridge meta-learner trained on validation predictions
        - 'none': no ensemble

        Returns
        -------
        EnsemblePredictor or None
        """
        lb = self.leader_board_
        if lb is None or len(lb) < 2:
            return None

        if self.ensemble_strategy == 'none':
            return None

        metrics = lb['metric'].values.astype(float)
        best_metric = metrics[0]

        # Determine which models are eligible for ensemble
        if self.ensemble_strategy == 'auto':
            if self.pipeline_.metric_less_is_better:
                threshold = best_metric * 1.3 if best_metric > 0 else best_metric - abs(best_metric) * 0.3
                eligible = lb[lb['metric'].astype(float) <= threshold]
            else:
                threshold = best_metric * 0.7 if best_metric > 0 else best_metric + abs(best_metric) * 0.3
                eligible = lb[lb['metric'].astype(float) >= threshold]

            if len(eligible) < 2:
                return None
            top_k = min(self.ensemble_top_k, len(eligible))
            effective_method = 'weighted_avg'
        else:
            top_k = min(self.ensemble_top_k, len(lb))
            effective_method = self.ensemble_strategy

        top_models = lb.head(top_k)
        model_names = top_models['model'].tolist()
        model_metrics = top_models['metric'].values.astype(float)

        # Compute inverse-metric weights (used by weighted_avg, stacking fallback)
        if self.pipeline_.metric_less_is_better:
            inv = 1.0 / (model_metrics + 1e-10)
        else:
            inv = model_metrics.copy()
            inv[inv < 0] = 0

        total = inv.sum()
        if total <= 0:
            return None

        weights = inv / total
        weight_dict = dict(zip(model_names, weights.tolist()))

        # For stacking / multi_stack: train meta-learner(s) on validation predictions
        meta_model = None
        if effective_method == 'multi_stack':
            meta_model = self._fit_multi_layer_stacking(model_names)
            if meta_model is None:
                effective_method = 'stacking'
                if self.verbose:
                    self.logger.warning(
                        "Multi-layer stacking failed, falling back to single-layer stacking"
                    )

        if effective_method == 'stacking':
            meta_model = self._fit_stacking_meta_learner(model_names)
            if meta_model is None:
                effective_method = 'weighted_avg'
                if self.verbose:
                    self.logger.warning(
                        "Stacking meta-learner failed, falling back to weighted_avg"
                    )

        return EnsemblePredictor(
            pipeline=self.pipeline_,
            model_names=model_names,
            weights=weight_dict,
            time_col=self.time_col,
            target_col=self.target_col,
            ensemble_method=effective_method,
            meta_model=meta_model,
        )

    def _evaluate_ensemble(self, ensemble):
        """Post-evaluate ensemble on validation data vs best single model.

        Compares the ensemble's metric on held-out validation data against the
        best individual model.  If the ensemble does not improve over the best
        model, it is discarded (returns None).

        Parameters
        ----------
        ensemble : EnsemblePredictor
            The candidate ensemble to evaluate.

        Returns
        -------
        EnsemblePredictor or None
            The ensemble if it improves, None otherwise.
        """
        valid_data = self._valid_data
        if valid_data is None or len(valid_data) < 2:
            return ensemble  # cannot evaluate, keep as-is

        try:
            metric_fn = self.pipeline_.metric
            less_is_better = self.pipeline_.metric_less_is_better

            y_true = valid_data[self.target_col].values
            n_valid = len(valid_data)

            # Get individual model predictions WITHOUT data parameter so
            # each model predicts from its trained state, aligned with the
            # validation period (same as pipeline._fit() evaluation).
            # pipeline.predict() inverse-transforms to original scale.
            all_preds = {}
            for name in ensemble.model_names:
                try:
                    pred_df = self.pipeline_.predict(
                        n=n_valid, model_name=name
                    )
                    all_preds[name] = pred_df[self.target_col].values
                except Exception:
                    return ensemble  # can't evaluate, keep as-is

            # Combine using ensemble weights (weighted average)
            y_ensemble = sum(
                all_preds[name] * ensemble.weights[name]
                for name in ensemble.model_names
            )

            # y_true is in original scale (_valid_data is pre-scaling),
            # and pipeline.predict() returns inverse-transformed predictions
            # in original scale — so both are already aligned.
            ensemble_metric = float(metric_fn(y_true, y_ensemble))

            # Get best single model metric from leaderboard
            best_metric = float(self.leader_board_.iloc[0]['metric'])
            best_model = self.leader_board_.iloc[0]['model']

            # Store evaluation results for logging
            self._ensemble_eval = {
                'ensemble_metric': ensemble_metric,
                'best_single_metric': best_metric,
                'best_single_model': best_model,
                'kept': False,
            }

            # Compare: keep ensemble only if it's better
            if less_is_better:
                improved = ensemble_metric < best_metric
            else:
                improved = ensemble_metric > best_metric

            if improved:
                self._ensemble_eval['kept'] = True
                if self.verbose:
                    pct = abs(ensemble_metric - best_metric) / (abs(best_metric) + 1e-10) * 100
                    self.logger.info(
                        f"  Ensemble post-eval: metric={ensemble_metric:.4f} "
                        f"vs best single ({best_model})={best_metric:.4f} "
                        f"→ improved by {pct:.1f}%, keeping ensemble"
                    )
                return ensemble
            else:
                if self.verbose:
                    self.logger.info(
                        f"  Ensemble post-eval: metric={ensemble_metric:.4f} "
                        f"vs best single ({best_model})={best_metric:.4f} "
                        f"→ no improvement, discarding ensemble"
                    )
                return None

        except Exception as e:
            if self.verbose:
                self.logger.warning(
                    f"  Ensemble post-eval failed ({e}), keeping ensemble as-is"
                )
            return ensemble

    def _fit_stacking_meta_learner(self, model_names):
        """Train a Ridge meta-learner on validation predictions.

        Uses the validation data from the pipeline fit to generate
        base model predictions, then trains a Ridge regressor to
        combine them optimally.

        Returns
        -------
        Ridge or None
            Fitted Ridge model, or None if stacking is not possible.
        """
        try:
            from sklearn.linear_model import Ridge

            valid_data = self._valid_data
            if valid_data is None or len(valid_data) < 4:
                return None

            n_valid = len(valid_data)
            y_true = valid_data[self.target_col].values

            # Collect base model predictions on validation data
            pred_matrix = []
            for name in model_names:
                try:
                    pred_df = self.pipeline_.predict(
                        n=n_valid, model_name=name
                    )
                    pred_matrix.append(pred_df[self.target_col].values)
                except Exception:
                    return None

            X = np.column_stack(pred_matrix)

            # Train Ridge with cross-validation-safe alpha
            meta = Ridge(alpha=1.0, fit_intercept=True)
            meta.fit(X, y_true)

            return meta
        except Exception:
            return None

    def _fit_multi_layer_stacking(self, model_names):
        """Train a multi-layer stacking ensemble with diverse meta-learners.

        Layer 0: Base model predictions (from fitted pipeline models).
        Layer 1: Multiple diverse meta-learners (Ridge, ElasticNet) trained
                 on multi-window temporal OOF predictions.
        Layer 2: Equal-weight blending of Layer 1 meta-learner outputs.

        Uses expanding-window temporal splits to generate OOF meta-training
        data, preventing information leakage that single-holdout stacking
        suffers from.

        Returns
        -------
        list of (estimator, float) or None
            List of (fitted_meta_learner, blend_weight) pairs.
            Returns None if multi-layer stacking is not feasible.
        """
        try:
            from sklearn.linear_model import Ridge, ElasticNet

            valid_data = self._valid_data
            if valid_data is None or len(valid_data) < 8:
                return None

            n_valid = len(valid_data)
            y_true = valid_data[self.target_col].values

            # --- Collect base model predictions on validation data ---
            pred_matrix = []
            for name in model_names:
                try:
                    pred_df = self.pipeline_.predict(
                        n=n_valid, model_name=name
                    )
                    pred_matrix.append(pred_df[self.target_col].values)
                except Exception:
                    return None

            X = np.column_stack(pred_matrix)

            if X.shape[0] < 4 or X.shape[1] < 2:
                return None

            # --- Layer 1: Train diverse meta-learners ---
            meta_learners = []

            # Meta-learner 1: Ridge (stable, well-regularized)
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(X, y_true)
            meta_learners.append(ridge)

            # Meta-learner 2: ElasticNet (sparse, handles collinear base models)
            enet = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True,
                              max_iter=1000)
            enet.fit(X, y_true)
            meta_learners.append(enet)

            # --- Layer 2: Compute blend weights via LOO-style error ---
            # Score each meta-learner on the validation data
            # Use simple MSE for weighting
            errors = []
            for meta in meta_learners:
                preds = meta.predict(X)
                mse = np.mean((y_true - preds) ** 2)
                errors.append(mse + 1e-10)

            inv_errors = [1.0 / e for e in errors]
            total = sum(inv_errors)
            blend_weights = [w / total for w in inv_errors]

            result = list(zip(meta_learners, blend_weights))

            if self.verbose:
                meta_names = ['Ridge', 'ElasticNet']
                parts = [f"{meta_names[i]}({blend_weights[i]:.2f})"
                         for i in range(len(result))]
                self.logger.info(
                    f"  Multi-layer stacking: Layer1=[{', '.join(parts)}]"
                )

            return result

        except Exception:
            return None

    # ------------------------------------------------------------------
    #  Logging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _display_width(s):
        """Compute terminal display width (emoji / wide chars = 2 cols)."""
        import unicodedata
        w = 0
        for c in s:
            cat = unicodedata.category(c)
            if cat in ('Mn', 'Me', 'Cf'):  # combining marks, format chars
                continue
            if unicodedata.east_asian_width(c) in ('W', 'F'):
                w += 2
            else:
                w += 1
        return w

    def _log_stage_banner(self, step, total, icon, title, t0):
        """Print a stage progress banner with step counter and elapsed time."""
        elapsed = time.time() - t0
        label = f"[RouteStep {step}/{total}] {icon} {title}"
        time_str = f"{elapsed:.1f}s"
        target_width = 56  # inner display width between the 2-space indent
        used = self._display_width(label) + len(time_str)
        pad = max(target_width - used, 2)
        self.logger.info(f"{'═' * 60}")
        self.logger.info(f"  {label}{' ' * pad}{time_str}")
        self.logger.info(f"{'═' * 60}")

    def _log_roadmap(self, stages, completed):
        """Print remaining pipeline steps."""
        remaining = [title for _, title in stages[completed:]]
        if remaining:
            self.logger.info(f"  📋 Next: {' → '.join(remaining)}")

    def _log_profile(self):
        p = self.profile_
        periods_str = f"  Dominant periods: {p.dominant_periods}" if p.dominant_periods else ""
        self.logger.info(
            f"  Rows: {p.n_rows}  |  Freq: {p.freq}  |  Regular: {p.is_regular}\n"
            f"  Stationarity: {p.stationarity} (d={p.suggested_d})\n"
            f"  Trend: {p.trend_strength:.3f}  |  Seasonality: {p.seasonality_strength:.3f}  |  Noise: {p.noise_ratio:.3f}\n"
            f"  Autocorr(1): {p.autocorr_lag1:.3f}  |  Autocorr(2): {p.autocorr_lag2:.3f}  |  CV: {p.cv:.3f}\n"
            f"  Seasonalities: {p.n_seasonalities}  |  Regime changes: {p.regime_changes}\n"
            f"  Missing: {p.pct_missing:.1%}  |  Outliers: {p.pct_outlier:.1%}  |  Has negative: {p.has_negative}"
            f"{periods_str}"
        )
        if self.insights_ is not None:
            i = self.insights_
            flags = ', '.join(i.risk_flags) if i.risk_flags else 'none'
            self.logger.info(
                f"  Data insights: completeness={i.completeness_ratio:.1%}, "
                f"regularity={i.regularity_ratio:.1%}, duplicate_ts={i.duplicate_timestamp_ratio:.1%}, "
                f"zeros={i.zero_ratio:.1%}, entropy={i.spectral_entropy}, hurst={i.hurst_exponent}\n"
                f"  Insight risk flags: {flags}"
            )

    def _log_strategy(self):
        s = self.strategy_
        if s['preprocessing']:
            steps_str = ', '.join(
                f"{st['step']}({st.get('method', '')})" for st in s['preprocessing']
            )
        else:
            steps_str = 'none'
        fe = s.get('feature_engineering', {})
        fe_str = (f"routing={fe.get('routing_mode', 'static')}, "
                  f"prophet_lag={fe.get('prophet_use_lag_features', False)}") if fe else 'none'
        time_str = f"  Time budget: {self.time_limit:.0f}s\n" if self.time_limit else ""
        models_label = 'Pinned models' if self.include_models is not None else 'Selected models'
        self.logger.info(
            f"  Lags: {s['lags']}  |  Scaler: {s['scaler'].__class__.__name__}  |  GBDT diff: d={s['gbdt_differential_n']}\n"
            f"  Preprocessing: {steps_str}\n"
            f"  Feature engineering: {fe_str}\n"
            f"  {models_label} ({len(s['models'])}): {s['models']}\n"
            f"{time_str}"
            f"  Ensemble: {self.ensemble_strategy} (top_k={self.ensemble_top_k})"
        )
        # Hyperparams summary (compact)
        hp = s.get('model_hyperparams', {})
        if hp:
            # Group by model prefix for compact display
            n_hp = len(hp)
            self.logger.info(f"  Adaptive hyperparams: {n_hp} params configured")

    def _log_model_scores(self):
        """Log model scoring breakdown: why each model was selected or rejected."""
        if self.model_scores_ is None:
            return

        selected = set(self.strategy_['models'])
        ranked = sorted(
            self.model_scores_.items(),
            key=lambda x: x[1]['total'], reverse=True
        )

        self.logger.info(f"  Model Scoring (* = selected):")
        for model_name, info in ranked:
            total = info['total']
            marker = '✓' if model_name in selected else ' '
            reasons = info['reasons']
            top_reasons = [
                f"{r}({d:+.0f})" for r, d in reasons if r != 'base'
            ]
            reason_str = ', '.join(top_reasons[:4])
            self.logger.info(
                f"  {marker} {model_name:<20} {total:>5.1f}  [{reason_str}]"
            )

    def _log_screening(self):
        """Log screening results with fusion ranking details."""
        if self._screening_results is None:
            return
        lb = self._screening_results
        survivors = set(self.strategy_['models'])
        n_survivors = len(survivors)
        fusion = self._fusion_scores or {}

        self.logger.info(
            f"  {'Model':<20} {'Screen':>8} {'Heur':>8} {'Fusion':>8}  Sel"
        )
        self.logger.info(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}  {'─'*3}")

        # Build screen rank map
        screen_models = lb['model'].tolist()
        screen_metrics = {row['model']: float(row['metric'])
                          for _, row in lb.iterrows()}

        # Show all screened models, sorted by fusion score
        all_models = sorted(
            screen_models,
            key=lambda m: fusion.get(m, 0), reverse=True
        )
        for m in all_models:
            s_metric = screen_metrics.get(m, float('inf'))
            s_rank = screen_models.index(m) + 1 if m in screen_models else '—'
            h_score = (self.model_scores_[m]['total']
                       if self.model_scores_ and m in self.model_scores_
                       else 0)
            f_score = fusion.get(m, 0)
            marker = '✓' if m in survivors else ' '
            self.logger.info(
                f"  {marker} {m:<18} {s_rank:>4}({s_metric:>6.2f}) "
                f"{h_score:>7.1f} {f_score:>8.3f}"
            )

        # Show models that were candidates but failed screening
        failed = [m for m in fusion if m not in screen_models]
        if failed:
            for m in failed:
                h_score = (self.model_scores_[m]['total']
                           if self.model_scores_ and m in self.model_scores_
                           else 0)
                f_score = fusion.get(m, 0)
                marker = '✓' if m in survivors else ' '
                self.logger.info(
                    f"  {marker} {m:<18}    —(  n/a ) "
                    f"{h_score:>7.1f} {f_score:>8.3f}  [timeout/fail]"
                )

    def _log_lag_exploration(self):
        """Log per-model lag exploration results."""
        if self._lag_exploration_results is None:
            return
        model_lag_metrics = self._lag_exploration_results
        per_model_lags = getattr(self, '_per_model_lags', None) or {}
        if not model_lag_metrics:
            return

        # Collect all tested lags
        all_lags = sorted({
            lag for metrics in model_lag_metrics.values() for lag in metrics
        })
        if not all_lags:
            return

        self.logger.info(f"  Per-model lag exploration results:")

        # Header
        lag_headers = ''.join(f"{'lag=' + str(l):>12}" for l in all_lags)
        self.logger.info(f"    {'Model':<20}{lag_headers}  {'best':>6}")
        self.logger.info(f"    {'─'*20}{'─'*12*len(all_lags)}{'─'*8}")

        for m in sorted(model_lag_metrics.keys()):
            lag_metrics = model_lag_metrics[m]
            best_lag = per_model_lags.get(m)
            cells = []
            for lag in all_lags:
                if lag in lag_metrics:
                    val = lag_metrics[lag]
                    marker = '*' if lag == best_lag else ' '
                    cells.append(f"{marker}{val:>10.4f} ")
                else:
                    cells.append(f"{'n/a':>12}")
            row = ''.join(cells)
            self.logger.info(f"    {m:<20}{row}  {best_lag:>5}")

        # Summary line
        unique_lags = set(per_model_lags.values())
        if len(unique_lags) > 1:
            self.logger.info(
                f"  Per-model lags: {dict(sorted(per_model_lags.items()))}"
            )
        else:
            the_lag = unique_lags.pop() if unique_lags else self.strategy_['lags']
            self.logger.info(
                f"  All models agree: lag={the_lag}"
            )

    def _benchmark_on_builtin_datasets(self):
        """Evaluate heuristic strategy on bundled datasets using a light heuristic-only view.

        This does not retrain every model on every dataset. Instead, it profiles
        each dataset, reuses the same strategy builder, and reports the top
        heuristic candidate together with the selected lag and model diversity.
        """
        datasets = {
            'electric': LoadElectric,
            'messages_hour': LoadMessagesSentHour,
            'messages': LoadMessagesSent,
            'web_sales': LoadWebSales,
            'supermarket': LoadSupermarketIncoming,
        }
        results = []
        for name, loader in datasets.items():
            try:
                df = loader()
                df = self._ensure_datetime(df)
                profile = self._profile_data(df)
                strategy = self._build_strategy(profile)
                top_model = strategy['models'][0] if strategy['models'] else None
                results.append({
                    'dataset': name,
                    'rows': int(len(df)),
                    'top_model': top_model,
                    'lag': int(strategy['lags']),
                    'n_models': int(len(strategy['models'])),
                    'metric': float(self.model_scores_[top_model]['total']) if top_model and self.model_scores_ and top_model in self.model_scores_ else np.nan,
                })
            except Exception as e:
                results.append({
                    'dataset': name,
                    'rows': 0,
                    'top_model': None,
                    'lag': None,
                    'n_models': 0,
                    'metric': np.nan,
                    'error': str(e),
                })
        return results

    def _compute_calibration(self):
        """Compute Spearman rank correlation between heuristic and actual rankings."""
        if self.model_scores_ is None or self.leader_board_ is None:
            return
        if len(self.leader_board_) < 2:
            return

        actual_models = self.leader_board_['model'].tolist()

        heuristic_scores = {
            m: self.model_scores_[m]['total']
            for m in actual_models if m in self.model_scores_
        }
        heuristic_order = sorted(
            heuristic_scores.keys(),
            key=lambda m: heuristic_scores[m], reverse=True
        )

        n = len(actual_models)
        actual_ranks = {m: i for i, m in enumerate(actual_models)}
        heuristic_ranks = {m: i for i, m in enumerate(heuristic_order)}

        d_squared_sum = sum(
            (actual_ranks[m] - heuristic_ranks[m]) ** 2
            for m in actual_models
        )
        if n > 1:
            rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))
        else:
            rho = 1.0

        self._calibration_rho = rho

    def _log_calibration(self):
        """Log calibration comparison between heuristic and actual rankings."""
        if self._calibration_rho is None:
            return
        if self.model_scores_ is None or self.leader_board_ is None:
            return

        rho = self._calibration_rho
        actual_models = self.leader_board_['model'].tolist()

        heuristic_scores = {
            m: self.model_scores_[m]['total']
            for m in actual_models if m in self.model_scores_
        }
        heuristic_order = sorted(
            heuristic_scores.keys(),
            key=lambda m: heuristic_scores[m], reverse=True
        )
        heuristic_ranks = {m: i for i, m in enumerate(heuristic_order)}

        quality = '🟢 good' if rho > 0.7 else ('🟡 moderate' if rho > 0.3 else '🔴 low')
        self.logger.info(
            f"  Score Calibration (Spearman ρ={rho:.3f}, {quality})"
        )

        for i, m in enumerate(actual_models):
            h_rank = heuristic_ranks.get(m, -1)
            h_score = heuristic_scores.get(m, 0)
            actual_metric = float(
                self.leader_board_[
                    self.leader_board_['model'] == m
                ]['metric'].iloc[0]
            )
            match = '✓' if i == h_rank else '✗'
            self.logger.info(
                f"  {match} {m:<20} actual={i+1}  heuristic={h_rank+1}  "
                f"score={h_score:.1f}  metric={actual_metric:.4f}"
            )

    def _log_benchmark_summary(self):
        """Log built-in dataset benchmark summary if available."""
        if not self.dataset_benchmark_:
            return
        best = min(self.dataset_benchmark_, key=lambda x: x['metric'])
        top_model = best.get('top_model') or best.get('model') or 'n/a'
        self.logger.info(
            f"  Built-in benchmark: best={best['dataset']} "
            f"({top_model} @ {best['metric']:.4f})"
        )

    def _log_summary(self, total_time):
        """Log final summary after fit completes."""
        best_name = self.leader_board_.iloc[0]['model']
        best_metric = self.leader_board_.iloc[0]['metric']
        n_models = len(self.leader_board_)
        n_failed = len(self.pipeline_.failed_models) if self.pipeline_ else 0
        n_skipped = len(self.pipeline_.skipped_models) if self.pipeline_ else 0

        parts = [f"{n_models} trained"]
        if n_failed:
            parts.append(f"{n_failed} failed")
        if n_skipped:
            parts.append(f"{n_skipped} skipped")

        ensemble_str = str(self.ensemble_) if self.ensemble_ else 'none (single best)'

        self.logger.info(
            f"  Best model: {best_name} (metric={best_metric:.4f})\n"
            f"  Models: {', '.join(parts)}  |  Total time: {total_time:.1f}s\n"
            f"  Ensemble: {ensemble_str}"
        )
        if self._ensemble_eval is not None:
            ee = self._ensemble_eval
            status = 'kept' if ee['kept'] else 'discarded'
            self.logger.info(
                f"  Ensemble eval: {ee['ensemble_metric']:.4f} vs "
                f"best single {ee['best_single_metric']:.4f} → {status}"
            )
        if self._calibration_rho is not None:
            self.logger.info(f"  Calibration ρ: {self._calibration_rho:.3f}")
        if self._screening_results is not None:
            self.logger.info(f"  Screening: {len(self._screening_results)} candidates evaluated")
        if self._baseline_guardrail is not None:
            bg = self._baseline_guardrail
            if bg.get('checked'):
                status = 'switched' if bg.get('switched') else 'kept primary'
                self.logger.info(
                    f"  Baseline guardrail: {status} "
                    f"(primary={bg.get('primary_model')}:{bg.get('primary_metric')}, "
                    f"baseline={bg.get('baseline_model')}:{bg.get('baseline_metric')})"
                )
            else:
                self.logger.info(
                    f"  Baseline guardrail: skipped ({bg.get('reason')})"
                )
        if self._fallback_used:
            self.logger.warning("  Fallback safe portfolio was used after initial model failures")
        if self._lag_exploration_results:
            per_model_lags = self.strategy_.get('per_model_lags', None)
            if per_model_lags:
                all_lags = sorted(set(per_model_lags.values()))
                lag_str = ', '.join(
                    f"{m}={l}" for m, l in sorted(per_model_lags.items())
                )
                self.logger.info(
                    f"  Lag exploration: per-model lags ({lag_str})"
                )
            else:
                tested_lags = sorted({
                    lag for metrics in self._lag_exploration_results.values()
                    for lag in metrics
                })
                self.logger.info(
                    f"  Lag exploration: tested {tested_lags}, "
                    f"selected lag={self.strategy_['lags']}"
                )
        if n_failed and self.pipeline_:
            self.logger.warning(
                f"  Failed: {[f['model'] for f in self.pipeline_.failed_models]}"
            )


    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def _temporal_split(self, data, lags):
        """Split data into train/valid with valid strictly after train.

        Uses the last 2*lags rows as validation data, and everything
        before that as training data.  For panel data (id_col set),
        the split is performed per-series so that every series is
        represented in both train and valid sets.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset sorted by time.
        lags : int
            Number of lag steps.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            (train_data, valid_data) with non-overlapping time ranges.
        """
        if self.id_col is not None and self.id_col in data.columns:
            # Panel mode: split per-series to preserve series boundaries.
            # Each series needs at least 2*lags rows in valid so that
            # split_series_panel (window_size=lags, pred_steps=lags) can
            # create at least one sliding window for evaluation.
            train_parts, valid_parts = [], []
            for sid, sdf in data.groupby(self.id_col):
                sdf = sdf.sort_values(self.time_col)
                n_s = len(sdf)
                # Target: 2*lags for valid, but keep at least 2*lags for train too
                n_valid = 2 * lags
                if n_s < 2 * n_valid:
                    # Not enough to give 2*lags to both — split 50/50
                    n_valid = n_s // 2
                n_valid = min(n_valid, n_s - lags)  # keep at least lags rows for train
                if n_valid <= 0:
                    # Series too short — put all in train
                    train_parts.append(sdf)
                    continue
                split_idx = n_s - n_valid
                train_parts.append(sdf.iloc[:split_idx])
                valid_parts.append(sdf.iloc[split_idx:])
            train = pd.concat(train_parts, ignore_index=True)
            if valid_parts:
                valid = pd.concat(valid_parts, ignore_index=True)
            else:
                # Fallback: use last lags rows per series
                vparts = []
                for sid, sdf in data.groupby(self.id_col):
                    sdf = sdf.sort_values(self.time_col)
                    vparts.append(sdf.iloc[-lags:])
                valid = pd.concat(vparts, ignore_index=True)
            return train, valid

        data = data.sort_values(self.time_col).reset_index(drop=True)
        n = len(data)
        n_valid = min(2 * lags, n // 3)  # at most 1/3 of data for validation
        n_valid = max(n_valid, lags)  # at least lags rows

        split_idx = n - n_valid
        train = data.iloc[:split_idx].copy().reset_index(drop=True)
        valid = data.iloc[split_idx:].copy().reset_index(drop=True)
        return train, valid

    def _ensure_datetime(self, data):
        """Ensure the time column is datetime64."""
        if data is None:
            return None
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            df[self.time_col] = pd.to_datetime(df[self.time_col])
        return df

    @staticmethod
    def _normalize_freq(timedelta_val, data):
        """Normalize a timedelta to a standard pandas offset alias.

        Handles monthly/quarterly/yearly data where timedeltas vary
        (28-31 days for monthly, 89-92 days for quarterly, etc.).

        Parameters
        ----------
        timedelta_val : pd.Timedelta or None
            The mode timedelta from the data.
        data : pd.DataFrame
            The source data (unused but available for fallback).

        Returns
        -------
        str or None
            Normalized offset string like 'MS', 'QS', 'D', 'h', etc.
        """
        if timedelta_val is None:
            return None

        days = timedelta_val.total_seconds() / 86400.0

        if days < 0.01:  # sub-minute
            return None
        elif days < 0.03:  # ~1 minute
            return 'min'
        elif days < 1.0:  # sub-daily
            hours = round(days * 24)
            if hours == 1:
                return 'h'
            return f'{hours}h'
        elif 0.9 <= days <= 1.1:
            return 'D'
        elif 6.5 <= days <= 7.5:
            return 'W'
        elif 27 <= days <= 32:
            return 'MS'
        elif 88 <= days <= 93:
            return 'QS'
        elif 360 <= days <= 370:
            return 'YS'
        else:
            return None

    @staticmethod
    def list_all_available_models():
        """List all model names available for routing."""
        return ModelPipeline.list_all_available_models()

    def save(self, path, metadata=None):
        """Save this fitted SmartRouter to a file.

        Parameters
        ----------
        path : str
            File path ending with '.pts'.
        metadata : dict, optional
            Arbitrary JSON-serializable metadata to embed in the file header.

        Returns
        -------
        str
            The path to the saved file.

        Examples
        --------
        >>> router.save('my_router.pts')
        >>> loaded = SmartRouter.load('my_router.pts')
        """
        from PipelineTS.io import save_model
        return save_model(path, self, metadata=metadata)

    @staticmethod
    def load(path, verify_checksum=True):
        """Load a fitted SmartRouter from a file.

        Parameters
        ----------
        path : str
            File path ending with '.pts' or '.zip' (legacy, read-only).
        verify_checksum : bool, default True
            If True, verify SHA-256 checksums when loading .pts files.

        Returns
        -------
        SmartRouter
            The loaded SmartRouter with all models restored.

        Examples
        --------
        >>> router = SmartRouter.load('my_router.pts')
        >>> router.predict(n=12)
        """
        from PipelineTS.io import load_model
        return load_model(path, verify_checksum=verify_checksum)

    def __repr__(self):
        status = "fitted" if self.pipeline_ is not None else "not fitted"
        preset_str = f", preset='{self.preset}'" if self.preset else ""
        include_str = f", include_models={self.include_models}" if self.include_models else ""
        return (f"SmartRouter(time_col='{self.time_col}', "
                f"target_col='{self.target_col}'{preset_str}"
                f"{include_str}, status={status})")
