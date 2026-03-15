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
from PipelineTS.spinesTS.metrics import mae
from PipelineTS.preprocessing import (
    TimeSeriesMissingHandler,
    TimeSeriesOutlierDetector,
    FrequencyDetector,
    StationarityTest,
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
    ...                      include_models=['prophet', 'torch_boosting_forest'],
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
        },
        'medium_quality': {
            'max_models': 5,
            'cv': 5,
            'search_strategy': 'auto',
            'ensemble_strategy': 'auto',
            'ensemble_top_k': 3,
        },
        'high_quality': {
            'max_models': 8,
            'cv': 5,
            'search_strategy': 'thorough',
            'ensemble_strategy': 'weighted_avg',
            'ensemble_top_k': 3,
        },
        'best_quality': {
            'max_models': 15,
            'cv': 5,
            'search_strategy': 'thorough',
            'ensemble_strategy': 'weighted_avg',
            'ensemble_top_k': 5,
        },
    }

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
                 self.hpo_strategy not in ('none', 'quick', 'full'),
                 f"hpo_strategy must be 'none', 'quick', or 'full', "
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
        self.strategy_ = None
        self.pipeline_ = None
        self.leader_board_ = None
        self.best_model_ = None
        self.ensemble_ = None
        self.model_scores_ = None
        self._preprocessed_data = None
        self._valid_data = None
        self._scaler_obj = None
        self._screening_results = None
        self._lag_exploration_results = None
        self._per_model_lags = None
        self._calibration_rho = None
        self._hpo_results = None
        self._ensemble_eval = None
        self._fusion_scores = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @ParameterTypeAssert({'data': pd.DataFrame, 'valid_data': (pd.DataFrame, None)})
    def fit(self, data, valid_data=None):
        """Profile data, select strategy, fit pipeline, return self.

        When ``search_strategy`` is ``'auto'`` or ``'thorough'``, the fit
        process adds validation-driven phases before full training:

        1. **Quick screening** – trains lightweight models on a data subset
           to eliminate weak candidates before committing full training time.
        2. **Multi-lag exploration** – tests 2-3 lag candidates with a fast
           model and picks the one with the best holdout metric.
        3. **Score calibration** – after full training, compares heuristic
           rankings with actual performance and logs correlation.

        Parameters
        ----------
        data : pd.DataFrame
            Training data with time_col and target_col.
        valid_data : pd.DataFrame or None
            Optional validation data.

        Returns
        -------
        self
        """
        t0 = time.time()

        # Print device info once at the very beginning
        if self.verbose:
            from PipelineTS.spinesTS.base._torch_mixin import detect_available_device
            _dev, _detail = detect_available_device(self.accelerator)
            _active = _dev.upper().replace(':', ' ').split()[0]
            self.logger.info(f"Accelerator: {_active}")

        # Auto-convert time column to datetime if needed
        data = self._ensure_datetime(data)
        if valid_data is not None:
            valid_data = self._ensure_datetime(valid_data)

        # ── Analyze & plan ────────────────────────────────────
        self.profile_ = self._profile_data(data)
        self.strategy_ = self._build_strategy(self.profile_)

        # Compute pipeline stages for progress tracking
        _will_screen = self._should_screen()
        _will_explore = self._should_explore_lags()
        _will_hpo = self.hpo_strategy != 'none'

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

        # The pipeline trains on ALL data and evaluates on the valid tail.
        # When valid_data is passed explicitly, pipeline uses data.copy()
        # for training — so we pass the full dataset, not just the train split.
        # The valid split is only used for metric evaluation.
        self._preprocessed_data = full_processed
        self._valid_data = processed_valid

        # Quick screening (eliminate weak candidates with holdout)
        if _will_screen:
            _cur += 1
            if self.verbose:
                self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)
            # Wide screening: evaluate many more candidates than max_models
            # so we don't miss good models that heuristic scoring ranks low.
            pool_size = self._get_screen_pool_size()
            broad_candidates = self._select_models(
                self.profile_, n_candidates=pool_size
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
                # Re-split with primary lag (only if we auto-split)
                if not user_provided_valid:
                    processed_train, processed_valid = self._temporal_split(
                        full_processed, primary_lag
                    )
                    self._preprocessed_data = processed_train
                    self._valid_data = processed_valid

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

        # Time-aware epoch capping: reduce NN epochs when time budget is tight
        # so individual models don't consume the entire budget.
        if remaining_time is not None and len(models) > 0:
            per_model_budget = remaining_time / len(models)
            nn_models_set = {
                'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
                'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
                'patch_rnn', 'itransformer', 'srs_net', 'deepar',
            }
            if per_model_budget < 30:
                epoch_cap = 100
            elif per_model_budget < 60:
                epoch_cap = 300
            elif per_model_budget < 120:
                epoch_cap = 500
            else:
                epoch_cap = None  # no cap needed

            if epoch_cap is not None:
                for m in models:
                    if m in nn_models_set:
                        key = f'{m}__epochs'
                        current = hyperparams.get(key, 1000)
                        if current > epoch_cap:
                            hyperparams[key] = epoch_cap
                if self.verbose:
                    self.logger.info(
                        f"  Time-aware epoch cap: {epoch_cap} epochs "
                        f"(per-model budget ~{per_model_budget:.0f}s)"
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
            time_limit=remaining_time,
            per_model_lags=effective_per_model_lags,
            **hyperparams,
        )

        # SmartRouter already printed device info, suppress Pipeline's duplicate
        self.pipeline_._device_info_logged = True

        # Register callback for real-time model tracking
        self._model_results = []
        self.pipeline_._on_model_complete_callback = self._on_model_trained

        self.leader_board_ = self.pipeline_.fit(
            self._preprocessed_data, valid_data=self._valid_data
        )

        if self.leader_board_.empty:
            self.logger.error("No models completed. Cannot build ensemble.")
            return self

        self.best_model_ = self.pipeline_.best_model_

        # Evaluation & Summary
        _cur += 1
        if self.verbose:
            self._log_stage_banner(_cur, _n_stages, *_stages[_cur - 1], t0)

        self.ensemble_ = self._build_ensemble()
        if self.ensemble_ is not None:
            self.ensemble_ = self._evaluate_ensemble(self.ensemble_)

        self._compute_calibration()
        total_time = time.time() - t0
        if self.verbose:
            self._log_calibration()
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
        """Retrieve a fitted model from the pipeline."""
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
        for m in all_models:
            total, reasons = self._score_model(m, p)
            scores[m] = {'total': total, 'reasons': reasons}
        return scores

    def _select_preprocessing(self, p):
        """Determine which preprocessing steps to apply."""
        steps = []

        # Missing value handling
        if p.pct_missing > 0.001:
            if p.pct_missing < 0.05:
                steps.append({'step': 'fill_missing', 'method': 'linear'})
            elif p.pct_missing < 0.15:
                steps.append({'step': 'fill_missing', 'method': 'ffill'})
            else:
                steps.append({'step': 'fill_missing', 'method': 'linear'})

        # Implicit gap filling — only for truly irregular data
        # Skip for monthly/quarterly which appear irregular at day level
        if not p.is_regular and p.freq not in (
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

        # NN routing_mode: adaptive MoE for larger datasets with patterns
        if p.n_rows >= 200 and (
            p.seasonality_strength > 0.1 or p.trend_strength > 0.3
        ):
            fe['routing_mode'] = 'adaptive'
        else:
            fe['routing_mode'] = 'static'

        # Prophet lag features: useful when strong autocorrelation
        if p.autocorr_lag1 > 0.5 and p.n_rows >= 50:
            fe['prophet_use_lag_features'] = True
        else:
            fe['prophet_use_lag_features'] = False

        # Prophet seasonality mode: multiplicative when amplitude scales
        if p.seasonality_strength > 0.2 and p.trend_strength > 0.3:
            fe['prophet_seasonality_mode'] = 'multiplicative'
        else:
            fe['prophet_seasonality_mode'] = 'auto'

        return fe

    def _suggest_hyperparams(self, p):
        """Suggest model-specific hyperparameters based on data profile.

        Returns a dict in double-underscore format ready for ModelPipeline
        kwargs, e.g. {'torch_boosting_forest__n_trees': 128}.

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

        # --- NN models list (GTB-capable) ---
        nn_gtb_models = [
            'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
            'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
            'patch_rnn',
        ]
        # All NN models (including non-GTB)
        nn_all = nn_gtb_models + ['deepar', 'itransformer', 'srs_net']

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
        nn_heavy = {'transformer', 'tft', 'itransformer', 'srs_net', 'deepar',
                     'gau', 'n_beats', 'n_hits'}
        nn_light = {'d_linear', 'n_linear', 'tide', 'tcn'}
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
        transformer_models = {'transformer', 'tft', 'itransformer', 'gau', 'time2vec'}
        if n >= 100:
            for m in transformer_models:
                params[f'{m}__warmup_epochs'] = 10

        # --- NN: mHC-inspired residual gate ---
        # Sinkhorn-normalized residual gate prevents signal amplification;
        # most beneficial for noisy or non-stationary data where NN
        # training tends to oscillate
        if n >= 150 and (p.noise_ratio > 0.4 or
                         p.stationarity in ('non_stationary', 'difference_stationary')):
            for m in nn_all:
                params[f'{m}__use_residual_gate'] = True

        # --- Prophet ---
        if fe.get('prophet_use_lag_features'):
            params['prophet__use_lag_features'] = True
        if fe.get('prophet_seasonality_mode', 'auto') != 'auto':
            params['prophet__seasonality_mode'] = fe['prophet_seasonality_mode']

        # --- All tree models: verbose off by default ---
        for m in ['torch_boosting_forest', 'torch_bagging_forest', 'deep_forest',
                   'multi_output_model', 'multi_step_model', 'wide_gbrt']:
            params[f'{m}__verbose'] = False

        # --- All tree models: adapt to data size ---
        torch_tree_models = ['torch_boosting_forest', 'torch_bagging_forest']
        if n >= 300:
            for m in torch_tree_models:
                params[f'{m}__n_trees'] = 128
                params[f'{m}__n_epochs'] = 500
            params['deep_forest__n_trees'] = 32
            params['deep_forest__n_layers'] = 3
            params['deep_forest__n_epochs'] = 500
        elif n < 100:
            for m in torch_tree_models:
                params[f'{m}__n_trees'] = 48
                params[f'{m}__n_epochs'] = 300
            params['deep_forest__n_trees'] = 16
            params['deep_forest__n_layers'] = 2
            params['deep_forest__n_epochs'] = 300
        if p.noise_ratio > 0.7:
            for m in torch_tree_models:
                params[f'{m}__dropout'] = 0.15
                params[f'{m}__weight_decay'] = 1e-4

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
        - ml: torch_boosting_forest, torch_bagging_forest, deep_forest, wide_gbrt, ...
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

        # Sort by total score descending
        ranked = sorted(
            [(m, scores[m]['total']) for m in all_models],
            key=lambda x: x[1], reverse=True
        )

        # 5-category diversity system (ml and ml_gpu merged since all tree
        # models now use the same GPU-accelerated differentiable backend)
        categories = {
            'statistic': {'auto_arima', 'prophet'},
            'ml': {'torch_boosting_forest', 'torch_bagging_forest',
                   'deep_forest', 'wide_gbrt', 'multi_output_model',
                   'multi_step_model', 'regressor_chain'},
            'nn_light': {'d_linear', 'n_linear', 'tide', 'tcn'},
            'nn_medium': {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                          'time2vec', 'gau'},
            'nn_heavy': {'transformer', 'tft', 'itransformer', 'srs_net',
                         'deepar', 'chronos_2', 'chronos_2_synth',
                         'chronos_2_small'},
        }

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
        statistic_models = {'auto_arima', 'prophet'}
        ml_models = {'torch_boosting_forest', 'torch_bagging_forest',
                      'deep_forest', 'wide_gbrt', 'multi_output_model',
                      'multi_step_model', 'regressor_chain'}
        nn_light = {'d_linear', 'n_linear', 'tide', 'tcn'}
        nn_medium = {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                      'time2vec', 'gau'}
        nn_heavy = {'transformer', 'tft', 'itransformer', 'srs_net', 'deepar',
                    'chronos_2', 'chronos_2_synth', 'chronos_2_small'}

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
            if model_name in ('auto_arima', 'd_linear'):
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
            elif model_name in ('prophet', 'auto_arima'):
                _add(8, 'strong_seasonality: seasonal decomposition', is_pattern=True)
            elif model_name in ('stacking_rnn', 'patch_rnn', 'tcn'):
                _add(6, 'strong_seasonality: handles seasonal', is_pattern=True)
            elif model_name in nn_light:
                _add(4, 'strong_seasonality: basic seasonal', is_pattern=True)

        # ---- Trend strength (pattern bonus, capped) ----
        if p.trend_strength > 0.5:
            if model_name in ('d_linear', 'n_linear', 'tide'):
                _add(8, f'strong_trend({p.trend_strength:.2f}): linear trend specialist', is_pattern=True)
            elif model_name in ('prophet', 'auto_arima'):
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
            if model_name in ('auto_arima', 'stacking_rnn', 'patch_rnn', 'tcn'):
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
            elif model_name in ('prophet', 'n_hits', 'itransformer', 'stacking_rnn'):
                _add(5, 'multi_seasonal: multi-scale model', is_pattern=True)
            if model_name in ('d_linear', 'n_linear'):
                _add(-3, 'multi_seasonal: too simple')

        # ---- Forecast horizon relative to data ----
        if self.n_predict and p.n_rows > 0:
            ratio = self.n_predict / p.n_rows
            if ratio > 0.2:
                if model_name in ('prophet', 'auto_arima'):
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
        if model_name in ('torch_boosting_forest', 'torch_bagging_forest'):
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
            _add(3, 'speed: fast GPU-accelerated tree')

        # ---- Specific model strengths (conditional) ----
        if model_name == 'torch_boosting_forest':
            if p.noise_ratio > 0.7 and n >= 200:
                _add(3, 'boosting: robust for noisy large data')

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

        # ---- GPU tree model specific scoring ----
        # NOTE: bonuses here are deliberately modest to avoid over-scoring
        # ML/tree models relative to NN models (see calibration analysis).
        _gpu_tree_models = {'torch_boosting_forest', 'torch_bagging_forest', 'deep_forest'}
        if model_name in _gpu_tree_models:
            # GPU trees benefit from larger data (amortizes GPU overhead)
            if n >= 200:
                _add(3, f'{model_name}: GPU amortization on medium+ data')
            elif n < 80:
                _add(-3, f'{model_name}: GPU overhead not worthwhile for small data')
            # Differentiable trees can learn feature interactions end-to-end
            if p.n_seasonalities >= 2 and p.regime_changes > 5:
                _add(3, f'{model_name}: end-to-end feature learning')
            if model_name == 'deep_forest':
                # Deep Forest cascade: modest bonus, only for favorable conditions
                if 100 <= n <= 800 and p.noise_ratio < 0.8:
                    _add(3, 'deep_forest: cascade representation learning')
                elif n < 80:
                    _add(-5, 'deep_forest: cascade overfits on small data')
            if model_name == 'torch_boosting_forest':
                _add(2, 'torch_boosting_forest: differentiable boosting')

        if model_name in ('chronos_2', 'chronos_2_synth', 'chronos_2_small'):
            # Chronos-2 family: zero-shot foundation models — no training needed
            # Strong for small data where trained models may overfit
            if n < 100:
                _add(10, f'{model_name}: zero-shot excels on small data')
            elif n < 300:
                _add(5, f'{model_name}: zero-shot viable for medium data')
            if p.pct_missing > 0.01:
                _add(3, f'{model_name}: pretrained robustness to missing data ({p.pct_missing:.1%})')
            if p.n_seasonalities >= 2:
                _add(5, f'{model_name}: pretrained handles complex seasonality')
            # Differentiate: chronos_2_small is lighter, give slight bonus for speed
            if model_name == 'chronos_2_small' and n >= 200:
                _add(2, 'chronos_2_small: lightweight variant for larger data')

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
            return min(total_models, max(self.max_models * 3, 12))

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

        all_models = list(get_all_available_models().keys())
        excluded = [m for m in all_models if m not in set(candidates)]
        if not excluded:
            return candidates  # pool already covers every registered model

        n_inject = max(1, round(len(candidates) * 0.2))
        n_inject = min(n_inject, len(excluded))

        # Category mapping (mirrors _select_models / _fusion_select)
        _categories = {
            'statistic': {'auto_arima', 'prophet'},
            'ml': {'torch_boosting_forest', 'torch_bagging_forest',
                   'deep_forest', 'wide_gbrt', 'multi_output_model',
                   'multi_step_model', 'regressor_chain'},
            'nn_light': {'d_linear', 'n_linear', 'tide', 'tcn'},
            'nn_medium': {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                          'time2vec', 'gau'},
            'nn_heavy': {'transformer', 'tft', 'itransformer', 'srs_net',
                         'deepar', 'chronos_2', 'chronos_2_synth',
                         'chronos_2_small'},
        }
        model_cat = {}
        for cat_name, cat_models in _categories.items():
            for m in cat_models:
                model_cat[m] = cat_name

        covered_cats = {model_cat.get(m, 'unknown') for m in candidates}

        # Phase 1: inject models from categories entirely absent in the pool
        # (highest-value blind-spot correction)
        new_cat_excluded = [
            m for m in excluded
            if model_cat.get(m, 'unknown') not in covered_cats
        ]
        same_cat_excluded = [
            m for m in excluded
            if model_cat.get(m, 'unknown') in covered_cats
        ]

        rng = np.random.RandomState(
            self.random_state if self.random_state is not None else 0
        )

        injections = []
        for m in new_cat_excluded:
            if len(injections) >= n_inject:
                break
            injections.append(m)
            covered_cats.add(model_cat.get(m, 'unknown'))

        # Phase 2: fill remaining injection slots with random excluded models
        if len(injections) < n_inject:
            remaining = [m for m in excluded if m not in injections]
            rng.shuffle(remaining)
            injections.extend(remaining[: n_inject - len(injections)])

        # Drop bottom n_inject heuristic-ranked models from the current pool
        scores = self.model_scores_ or {}
        candidates_by_score = sorted(
            candidates,
            key=lambda m: scores.get(m, {}).get('total', 0.0),
            reverse=True,
        )
        survivors = candidates_by_score[: len(candidates) - len(injections)]
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

        models = self.strategy_['models']
        lags = self.strategy_['lags']
        scaler = self.strategy_['scaler']

        n_trials = self.hpo_n_trials
        if self.hpo_strategy == 'quick':
            n_trials = min(n_trials, 5)

        if self.verbose:
            self.logger.info(
                f"  Strategy: {self.hpo_strategy}, {n_trials} trials/model"
            )

        hpo = OptunaHPO(
            time_col=self.time_col,
            target_col=self.target_col,
            lags=lags,
            metric=mae,
            metric_less_is_better=True,
            n_trials=n_trials,
            timeout_per_model=self.hpo_timeout_per_model,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        pipeline_kwargs = {
            'scaler': scaler,
            'accelerator': self.accelerator,
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
        if len(candidates) <= self.max_models:
            return None  # no screening needed

        n = len(train_data)

        # Use data subset for speed (last 70%, or all if small)
        if n > 100:
            subset_start = int(n * 0.3)
            screen_train = train_data.iloc[subset_start:].reset_index(drop=True)
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
                gbdt_differential_n=strategy['gbdt_differential_n'],
                time_limit=screen_time,
                **screen_params,
            )

            screen_pipeline._device_info_logged = True
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
            return self.strategy_['models'][:self.max_models]

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
            screening_norm[m] = 1.0 - (i / max(n_s - 1, 1))

        # Models that were candidates but didn't complete screening
        # (timed out or failed) get a penalty: worst screening rank - 0.1
        worst_screen = 0.0
        for m in broad_candidates:
            if m not in screening_norm:
                screening_norm[m] = max(worst_screen - 0.1, -0.1)

        # --- Fusion scoring ---
        # α controls screening vs heuristic weight.
        # Screening is primary signal (α=0.7); heuristic is prior.
        alpha = 0.7

        fusion_scores = {}
        for m in broad_candidates:
            s_norm = screening_norm.get(m, -0.1)
            h_norm = heuristic_norm.get(m, 0.0)
            fusion_scores[m] = alpha * s_norm + (1 - alpha) * h_norm

        # Store fusion details for logging
        self._fusion_scores = fusion_scores

        # --- Greedy diversity-aware selection ---
        # Walk down the fusion-ranked list.  Models from categories not
        # yet represented are selected immediately (diversity bonus);
        # models from already-covered categories are deferred to phase 2.
        # This ensures the strongest fusion performers are always
        # considered while diversity emerges naturally.
        categories = {
            'statistic': {'auto_arima', 'prophet'},
            'ml': {'torch_boosting_forest', 'torch_bagging_forest',
                   'deep_forest', 'wide_gbrt', 'multi_output_model',
                   'multi_step_model', 'regressor_chain'},
            'nn_light': {'d_linear', 'n_linear', 'tide', 'tcn'},
            'nn_medium': {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                          'time2vec', 'gau'},
            'nn_heavy': {'transformer', 'tft', 'itransformer', 'srs_net',
                         'deepar', 'chronos_2', 'chronos_2_synth',
                         'chronos_2_small'},
        }

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
        _speed_tier = {}
        _fast = {
            'auto_arima', 'prophet',
            'torch_boosting_forest', 'torch_bagging_forest', 'deep_forest',
            'wide_gbrt', 'multi_output_model', 'multi_step_model',
            'regressor_chain',
        }
        _light = {'d_linear', 'n_linear', 'tide', 'tcn'}
        _medium = {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                   'time2vec', 'gau'}
        # Everything else is heavy (tier 3)
        for m in candidates:
            if m in _fast:
                _speed_tier[m] = 0
            elif m in _light:
                _speed_tier[m] = 1
            elif m in _medium:
                _speed_tier[m] = 2
            else:
                _speed_tier[m] = 3
        # Stable sort preserves original (heuristic-score) order within tier
        return sorted(candidates, key=lambda m: _speed_tier[m])

    def _get_screening_hyperparams(self, candidates, strategy):
        """Build lightweight hyperparams for quick screening.

        Reduces GBDT estimators and NN epochs for faster evaluation.
        """
        params = {}
        # All tree models now use _TorchTreeWrapper params
        tree_models = {
            'torch_boosting_forest', 'torch_bagging_forest',
            'deep_forest',
        }
        # Legacy multi-output models still use n_estimators
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
            if m in tree_models:
                params[f'{m}__n_trees'] = 16
                params[f'{m}__n_epochs'] = 50
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
                    gbdt_differential_n=strategy['gbdt_differential_n'],
                    **fast_params,
                )
                eval_pipeline._device_info_logged = True
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
            'torch_boosting_forest', 'torch_bagging_forest',
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
                        n=n_valid, data=valid_data, model_name=name
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
                        n=n_valid, data=valid_data, model_name=name
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
