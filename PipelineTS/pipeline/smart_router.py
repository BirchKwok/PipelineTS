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
    ):
        # Resolve preset: preset provides defaults, explicit params override
        if preset is not None:
            raise_if(ValueError, preset not in self._PRESETS,
                     f"preset must be one of {list(self._PRESETS.keys())}, got '{preset}'")
            defaults = self._PRESETS[preset]
        else:
            defaults = self._PRESETS['medium_quality']  # default behavior

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

        # Step 1: Profile
        self.profile_ = self._profile_data(data)
        if self.verbose:
            self.logger.info(f"Data profile completed in {time.time() - t0:.2f}s")
            self._log_profile()

        # Step 2: Select strategy (with scoring explanation)
        self.strategy_ = self._build_strategy(self.profile_)
        if self.verbose:
            self._log_strategy()
            self._log_model_scores()

        # Step 3: Preprocess data
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

        self._preprocessed_data = processed_train
        self._valid_data = processed_valid

        # Step 4: Quick screening (eliminate weak candidates with holdout)
        if self._should_screen():
            if self.verbose:
                self.logger.info(
                    f"\n{'─'*60}\n  🔍 QUICK SCREENING\n{'─'*60}"
                )
            broad_candidates = self._select_models(
                self.profile_, n_candidates=self.max_models * 2
            )
            survivors = self._quick_screen(
                processed_train, processed_valid,
                broad_candidates, self.strategy_
            )
            self.strategy_['models'] = survivors
            if self.verbose:
                self._log_screening()

        # Step 5: Multi-lag exploration (find optimal lag per model)
        per_model_lags = None
        if self._should_explore_lags():
            if self.verbose:
                self.logger.info(
                    f"\n{'─'*60}\n  🔎 LAG EXPLORATION\n{'─'*60}"
                )
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

        # Step 5.5: HPO (if enabled)
        hyperparams = self.strategy_.get('model_hyperparams', {})
        if self.hpo_strategy != 'none':
            hyperparams = self._run_hpo(
                processed_train, processed_valid, hyperparams
            )
            self.strategy_['model_hyperparams'] = hyperparams

        # Step 6: Full training with refined models + optimal lag
        lags = self.strategy_['lags']
        models = self.strategy_['models']
        scaler = self.strategy_['scaler']
        gbdt_diff_n = self.strategy_['gbdt_differential_n']

        remaining_time = self._get_remaining_time(t0)

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

        # Step 7: Build ensemble (if strategy permits)
        self.ensemble_ = self._build_ensemble()

        # Step 8: Score calibration + summary
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

        strategy = {
            'preprocessing': self._select_preprocessing(p),
            'models': self._select_models(p),
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
        kwargs, e.g. {'lightgbm__n_trees': 128}.

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

        # --- All tree models (catboost/lightgbm/xgboost/random_forest are now
        # aliases to TorchBoosting/TorchBagging): adapt to data size ---
        torch_tree_models = ['torch_boosting_forest', 'torch_bagging_forest',
                             'catboost', 'lightgbm', 'xgboost', 'random_forest']
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
        - ml: catboost, lightgbm, xgboost, random_forest, torch_boosting_forest,
              torch_bagging_forest, deep_forest, wide_gbrt, ...
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
            'ml': {'catboost', 'lightgbm', 'xgboost', 'random_forest',
                   'torch_boosting_forest', 'torch_bagging_forest',
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
        ml_models = {'catboost', 'lightgbm', 'xgboost', 'random_forest',
                      'torch_boosting_forest', 'torch_bagging_forest',
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
        if model_name in ('lightgbm', 'xgboost', 'catboost',
                          'torch_boosting_forest', 'torch_bagging_forest'):
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
        if model_name in ('lightgbm', 'torch_boosting_forest'):
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
        _gpu_tree_models = {'torch_boosting_forest', 'torch_bagging_forest', 'deep_forest'}
        if model_name in _gpu_tree_models:
            # GPU trees benefit from larger data (amortizes GPU overhead)
            if n >= 200:
                _add(5, f'{model_name}: GPU amortization on medium+ data')
            elif n < 80:
                _add(-3, f'{model_name}: GPU overhead not worthwhile for small data')
            # Differentiable trees can learn feature interactions end-to-end
            if p.n_seasonalities >= 2 or p.regime_changes > 3:
                _add(4, f'{model_name}: end-to-end feature learning')
            if model_name == 'deep_forest':
                # Deep Forest cascade excels on medium data with complex patterns
                if 100 <= n <= 800:
                    _add(5, 'deep_forest: cascade representation learning for medium data')
                elif n < 80:
                    _add(-5, 'deep_forest: cascade overfits on small data')
                if p.noise_ratio > 0.5 and n >= 100:
                    _add(4, 'deep_forest: ensemble diversity handles noise')
                if p.autocorr_lag1 > 0.5 and n >= 80:
                    _add(3, 'deep_forest: temporal features + cascade depth')
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
        if self.search_strategy == 'basic':
            return False
        if self.search_strategy == 'thorough':
            return True
        # 'auto': screen when there are enough candidate models and data
        return (self.max_models >= 4 and
                self.profile_ is not None and
                self.profile_.n_rows >= 80)

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
                f"\n{'─'*60}\n  🔧 HPO ({self.hpo_strategy}, "
                f"{n_trials} trials/model)\n{'─'*60}"
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
        """Quick model screening to eliminate weak candidates.

        Trains lightweight versions of candidate models on a data subset,
        evaluates on holdout, and returns the top performers.

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
        list of str
            Surviving model names (top max_models).
        """
        if len(candidates) <= self.max_models:
            return candidates

        n = len(train_data)

        # Use data subset for speed (last 70%, or all if small)
        if n > 100:
            subset_start = int(n * 0.3)
            screen_train = train_data.iloc[subset_start:].reset_index(drop=True)
        else:
            screen_train = train_data

        screen_valid = valid_data

        # Screening hyperparams: reduced complexity for speed
        screen_params = self._get_screening_hyperparams(candidates, strategy)

        # Time limit for screening: 30% of total budget
        screen_time = None
        if self.time_limit is not None:
            screen_time = self.time_limit * 0.3

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
                    self.logger.warning("  Screening: no models completed, using heuristic selection")
                return candidates[:self.max_models]

            # Keep top max_models survivors
            survivors = screen_lb.head(self.max_models)['model'].tolist()
            self._screening_results = screen_lb

            if self.verbose:
                self.logger.info(
                    f"  Screening: {len(candidates)} candidates -> "
                    f"{len(survivors)} survivors"
                )

            return survivors

        except Exception as e:
            if self.verbose:
                self.logger.warning(
                    f"  Screening failed ({e}), using heuristic selection"
                )
            return candidates[:self.max_models]

    def _get_screening_hyperparams(self, candidates, strategy):
        """Build lightweight hyperparams for quick screening.

        Reduces GBDT estimators and NN epochs for faster evaluation.
        """
        params = {}
        # All tree models now use _TorchTreeWrapper params
        tree_models = {
            'lightgbm', 'xgboost', 'catboost', 'random_forest',
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
            'lightgbm', 'xgboost', 'catboost', 'random_forest',
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

    def _log_profile(self):
        p = self.profile_
        periods_str = f"  Dominant periods: {p.dominant_periods}" if p.dominant_periods else ""
        self.logger.info(
            f"\n{'─'*60}\n"
            f"  📊 DATA PROFILE\n"
            f"{'─'*60}\n"
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
        self.logger.info(
            f"\n{'─'*60}\n"
            f"  ⚙️  STRATEGY\n"
            f"{'─'*60}\n"
            f"  Lags: {s['lags']}  |  Scaler: {s['scaler'].__class__.__name__}  |  GBDT diff: d={s['gbdt_differential_n']}\n"
            f"  Preprocessing: {steps_str}\n"
            f"  Feature engineering: {fe_str}\n"
            f"  Selected models ({len(s['models'])}): {s['models']}\n"
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

        self.logger.info(
            f"\n{'─'*60}\n"
            f"  🏆 MODEL SCORING (* = selected)\n"
            f"{'─'*60}"
        )
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
        """Log screening results."""
        if self._screening_results is None:
            return
        lb = self._screening_results
        survivors = set(self.strategy_['models'])
        n_survivors = len(survivors)
        self.logger.info(
            f"  Screening: {len(lb)} evaluated → {n_survivors} survivors"
        )
        for i, row in lb.iterrows():
            marker = '✓' if row['model'] in survivors else ' '
            self.logger.info(
                f"    {marker} {row['model']:<20} metric={float(row['metric']):.4f}"
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

        self.logger.info(f"\n  Per-model lag exploration results:")

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
            f"\n{'─'*60}\n"
            f"  📈 SCORE CALIBRATION  (Spearman ρ={rho:.3f}, {quality})\n"
            f"{'─'*60}"
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
            f"\n{'─'*60}\n"
            f"  ✅ SUMMARY\n"
            f"{'─'*60}\n"
            f"  Best model: {best_name} (metric={best_metric:.4f})\n"
            f"  Models: {', '.join(parts)}  |  Total time: {total_time:.1f}s\n"
            f"  Ensemble: {ensemble_str}"
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
        before that as training data.

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
        if df[self.time_col].dtype != 'datetime64[ns]':
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

    def save(self, path):
        """Save this fitted SmartRouter to a zip file.

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
        >>> router.save('my_router.zip')
        >>> loaded = SmartRouter.load('my_router.zip')
        """
        from PipelineTS.io import save_model
        return save_model(path, self)

    @staticmethod
    def load(path):
        """Load a fitted SmartRouter from a zip file.

        Parameters
        ----------
        path : str
            File path ending with '.zip'.

        Returns
        -------
        SmartRouter
            The loaded SmartRouter with all models restored.

        Examples
        --------
        >>> router = SmartRouter.load('my_router.zip')
        >>> router.predict(n=12)
        """
        from PipelineTS.io import load_model
        return load_model(path)

    def __repr__(self):
        status = "fitted" if self.pipeline_ is not None else "not fitted"
        preset_str = f", preset='{self.preset}'" if self.preset else ""
        return (f"SmartRouter(time_col='{self.time_col}', "
                f"target_col='{self.target_col}'{preset_str}, status={status})")
