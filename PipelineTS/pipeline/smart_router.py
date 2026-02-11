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
    """Weighted ensemble of multiple fitted models.

    Combines predictions from top-K models using inverse-metric weighting
    for both point forecasts and interval predictions.

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
    """

    def __init__(self, pipeline, model_names, weights, time_col, target_col):
        self.pipeline = pipeline
        self.model_names = model_names
        self.weights = weights
        self.time_col = time_col
        self.target_col = target_col

    def predict(self, n, data=None):
        """Generate weighted-average predictions from ensemble members."""
        all_preds = {}
        for name in self.model_names:
            all_preds[name] = self.pipeline.predict(
                n=n, data=data, model_name=name
            )

        # Use the first model's output as the template
        result = all_preds[self.model_names[0]].copy()
        value_cols = [c for c in result.columns if c.startswith(self.target_col)]

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
            'strategy': 'weighted_avg',
            'models': self.model_names,
            'weights': {k: round(v, 4) for k, v in self.weights.items()},
        }

    def __repr__(self):
        models_str = ', '.join(
            f"{n}({self.weights[n]:.2f})" for n in self.model_names
        )
        return f"EnsemblePredictor([{models_str}])"


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
    7. Post-fit ensemble of top-K models (weighted averaging)

    Then runs a focused ModelPipeline competition on the selected subset
    and optionally builds a weighted ensemble for production predictions.

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
    max_models : int, default=8
        Maximum number of candidate models to evaluate.
    cv : int, default=5
        Cross-validation folds.
    ensemble_strategy : str, default='auto'
        How to combine models after fitting.
        - 'auto': build ensemble when top models are within 30% of best.
        - 'weighted_avg': always build inverse-metric weighted ensemble.
        - 'none': no ensemble, use single best model.
    ensemble_top_k : int, default=3
        Maximum number of models to include in the ensemble.

    Examples
    --------
    >>> from PipelineTS.pipeline import SmartRouter
    >>> router = SmartRouter(time_col='date', target_col='value')
    >>> router.fit(df)
    >>> print(router.strategy)
    >>> predictions = router.predict(n=16)
    >>> print(router.leader_board_)
    """

    @ParameterTypeAssert({
        'time_col': str,
        'target_col': str,
        'n_predict': (int, None),
        'quantile': (float, None),
        'random_state': int,
        'verbose': bool,
        'max_models': int,
        'cv': int,
        'ensemble_strategy': str,
        'ensemble_top_k': int,
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
        max_models=8,
        cv=5,
        ensemble_strategy='auto',
        ensemble_top_k=3,
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.n_predict = n_predict
        self.quantile = quantile
        self.accelerator = accelerator
        self.random_state = random_state
        self.verbose = verbose
        self.max_models = max_models
        self.cv = cv
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_top_k = ensemble_top_k

        raise_if(ValueError,
                 ensemble_strategy not in ('auto', 'weighted_avg', 'none'),
                 f"ensemble_strategy must be 'auto', 'weighted_avg', or 'none', "
                 f"got '{ensemble_strategy}'")

        self.logger = Logger(name='SmartRouter')

        # Filled after fit()
        self.profile_ = None
        self.strategy_ = None
        self.pipeline_ = None
        self.leader_board_ = None
        self.best_model_ = None
        self.ensemble_ = None
        self._preprocessed_data = None
        self._scaler_obj = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @ParameterTypeAssert({'data': pd.DataFrame, 'valid_data': (pd.DataFrame, None)})
    def fit(self, data, valid_data=None):
        """Profile data, select strategy, fit pipeline, return self.

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

        # Auto-convert time column to datetime if needed
        data = self._ensure_datetime(data)
        if valid_data is not None:
            valid_data = self._ensure_datetime(valid_data)

        # Step 1: Profile
        self.profile_ = self._profile_data(data)
        if self.verbose:
            self.logger.info(f"Data profile completed in {time.time() - t0:.2f}s")
            self._log_profile()

        # Step 2: Select strategy
        self.strategy_ = self._build_strategy(self.profile_)
        if self.verbose:
            self._log_strategy()

        # Step 3: Preprocess data
        processed_data = self._apply_preprocessing(data)
        if valid_data is not None:
            processed_valid = self._apply_preprocessing(valid_data)
        else:
            # Create a proper chronological train/valid split so that
            # validation timestamps are strictly after training timestamps.
            # This avoids NN model validation ordering checks failing.
            processed_data, processed_valid = self._temporal_split(
                processed_data, self.strategy_['lags']
            )

        self._preprocessed_data = processed_data

        # Step 4: Build and fit pipeline
        lags = self.strategy_['lags']
        models = self.strategy_['models']
        scaler = self.strategy_['scaler']
        gbdt_diff_n = self.strategy_['gbdt_differential_n']
        hyperparams = self.strategy_.get('model_hyperparams', {})

        self.pipeline_ = ModelPipeline(
            time_col=self.time_col,
            target_col=self.target_col,
            lags=lags,
            quantile=self.quantile,
            include_models=models,
            scaler=scaler,
            accelerator=self.accelerator,
            random_state=self.random_state,
            cv=self.cv,
            gbdt_differential_n=gbdt_diff_n,
            **hyperparams,
        )

        self.leader_board_ = self.pipeline_.fit(processed_data, valid_data=processed_valid)
        self.best_model_ = self.pipeline_.best_model_

        # Step 5: Build ensemble (if strategy permits)
        self.ensemble_ = self._build_ensemble()

        total_time = time.time() - t0
        if self.verbose:
            self.logger.info(f"SmartRouter completed in {total_time:.1f}s")
            self.logger.info(f"Best model: {self.leader_board_.iloc[0]['model']}")
            if self.ensemble_ is not None:
                self.logger.info(f"Ensemble: {self.ensemble_}")
            else:
                self.logger.info("Ensemble: not built (single best model used)")

        return self

    def predict(self, n=None, data=None, model_name=None, use_ensemble=True):
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

        return self.pipeline_.predict(n=n, data=data, model_name=model_name)

    @property
    def strategy(self):
        """Return the selected strategy details."""
        if self.strategy_ is None:
            return None
        return deepcopy(self.strategy_)

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
        values = data[self.target_col].values.astype(np.float64)
        valid = values[~np.isnan(values)]

        # Basic stats
        profile.n_rows = len(data)
        profile.mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
        profile.std = float(np.std(valid)) if len(valid) > 0 else 1.0
        profile.cv = abs(profile.std / profile.mean) if abs(profile.mean) > 1e-10 else 0.0
        profile.skewness = float(pd.Series(valid).skew()) if len(valid) > 2 else 0.0
        profile.kurtosis = float(pd.Series(valid).kurtosis()) if len(valid) > 3 else 0.0
        profile.has_negative = bool(np.any(valid < 0))

        # Missing values
        n_nan = int(np.sum(np.isnan(values)))
        profile.pct_missing = n_nan / len(values) if len(values) > 0 else 0.0

        # Frequency detection
        try:
            fd = FrequencyDetector(time_col=self.time_col)
            freq_info = fd.fit(data, target_col=self.target_col)
            raw_freq = freq_info.get('freq')
            raw_td = freq_info.get('freq_timedelta')
            profile.freq_timedelta = raw_td
            profile.is_regular = freq_info.get('is_regular', True)
            profile.dominant_periods = freq_info.get('dominant_periods', [])

            # Normalize frequency string for month/quarter/year intervals
            norm_freq = self._normalize_freq(raw_td, data)
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
        kwargs, e.g. {'lightgbm__n_estimators': 200}.
        """
        params = {}

        # --- NN models: routing_mode ---
        fe = self._select_feature_engineering(p)
        nn_models = [
            'd_linear', 'n_linear', 'n_beats', 'n_hits', 'tcn', 'tft',
            'gau', 'stacking_rnn', 'time2vec', 'transformer', 'tide',
            'patch_rnn',
        ]
        if fe.get('routing_mode') == 'adaptive':
            for m in nn_models:
                params[f'{m}__routing_mode'] = 'adaptive'

        # --- Prophet ---
        if fe.get('prophet_use_lag_features'):
            params['prophet__use_lag_features'] = True
        if fe.get('prophet_seasonality_mode', 'auto') != 'auto':
            params['prophet__seasonality_mode'] = fe['prophet_seasonality_mode']

        # --- GBDT: adapt complexity to data ---
        if p.n_rows >= 300 and p.seasonality_strength > 0.1:
            params['lightgbm__n_estimators'] = 200
            params['xgboost__n_estimators'] = 200
            params['catboost__n_estimators'] = 200
        elif p.n_rows < 80:
            params['lightgbm__n_estimators'] = 50
            params['xgboost__n_estimators'] = 50
            params['catboost__n_estimators'] = 50

        # High noise → stronger regularization for GBDT
        if p.noise_ratio > 0.8:
            params['lightgbm__learning_rate'] = 0.05
            params['xgboost__learning_rate'] = 0.05

        # Strong autocorrelation → deeper GBDT trees
        if p.autocorr_lag1 > 0.7:
            params['lightgbm__max_depth'] = 8
            params['xgboost__max_depth'] = 8

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

    def _select_models(self, p):
        """Score and select the best model candidates."""
        all_models = list(get_all_available_models().keys())

        # Score each model based on data characteristics
        scores = {}
        for m in all_models:
            scores[m] = self._score_model(m, p)

        # Sort by score descending, pick top max_models
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Ensure diversity: include at least one model from each viable category
        selected = []
        categories = {
            'statistic': {'auto_arima', 'prophet'},
            'ml': {'catboost', 'lightgbm', 'xgboost', 'random_forest',
                   'wide_gbrt', 'multi_output_model', 'multi_step_model',
                   'regressor_chain'},
            'nn': set(all_models) - {'auto_arima', 'prophet', 'catboost',
                    'lightgbm', 'xgboost', 'random_forest', 'wide_gbrt',
                    'multi_output_model', 'multi_step_model', 'regressor_chain'},
        }

        # First pass: add best from each category (if budget allows)
        remaining_budget = self.max_models
        category_best = {}
        for cat_name, cat_models in categories.items():
            cat_ranked = [(m, s) for m, s in ranked if m in cat_models]
            if cat_ranked:
                category_best[cat_name] = cat_ranked[0][0]

        # Guarantee at least one from each category with positive score
        for cat_name in ('ml', 'nn', 'statistic'):
            if cat_name in category_best and remaining_budget > 0:
                m = category_best[cat_name]
                if m not in selected:
                    selected.append(m)
                    remaining_budget -= 1

        # Fill remaining slots with top-ranked models
        for m, s in ranked:
            if remaining_budget <= 0:
                break
            if m not in selected:
                selected.append(m)
                remaining_budget -= 1

        return selected

    def _score_model(self, model_name, p):
        """Assign a suitability score to a model given the data profile.

        Higher score = more suitable. Base score is 50, adjusted by heuristics.
        """
        score = 50.0
        n = p.n_rows

        # ---- Model category classification ----
        statistic_models = {'auto_arima', 'prophet'}
        ml_models = {'catboost', 'lightgbm', 'xgboost', 'random_forest',
                      'wide_gbrt', 'multi_output_model', 'multi_step_model',
                      'regressor_chain'}
        nn_light = {'d_linear', 'n_linear', 'tide', 'tcn'}
        nn_medium = {'n_beats', 'n_hits', 'stacking_rnn', 'patch_rnn',
                      'time2vec', 'gau'}
        nn_heavy = {'transformer', 'tft', 'itransformer', 'srs_net', 'deepar'}

        # ---- Series length ----
        if n < 50:
            # Very short series: strongly prefer statistical + ML
            if model_name in statistic_models:
                score += 30
            elif model_name in ml_models:
                score += 20
            elif model_name in nn_light:
                score += 5
            elif model_name in nn_medium:
                score -= 10
            elif model_name in nn_heavy:
                score -= 25
        elif n < 150:
            # Short series
            if model_name in statistic_models:
                score += 20
            elif model_name in ml_models:
                score += 20
            elif model_name in nn_light:
                score += 15
            elif model_name in nn_medium:
                score += 5
            elif model_name in nn_heavy:
                score -= 5
        elif n < 500:
            # Medium series: all viable, slight preference for ML + light NN
            if model_name in ml_models:
                score += 15
            elif model_name in nn_light:
                score += 15
            elif model_name in nn_medium:
                score += 10
            elif model_name in nn_heavy:
                score += 5
            elif model_name in statistic_models:
                score += 10
        else:
            # Large series: NN models shine
            if model_name in nn_heavy:
                score += 20
            elif model_name in nn_medium:
                score += 15
            elif model_name in nn_light:
                score += 15
            elif model_name in ml_models:
                score += 10
            elif model_name in statistic_models:
                score += 5

        # ---- Stationarity ----
        if p.stationarity in ('non_stationary', 'difference_stationary'):
            # Non-stationary data: prefer models that handle trends well
            if model_name in ('prophet', 'auto_arima', 'd_linear'):
                score += 10
            elif model_name in ('n_beats', 'n_hits', 'tide'):
                score += 5
            # GBDT with differencing handles this too
            if model_name in ml_models:
                score += 5

        # ---- Seasonality ----
        if p.seasonality_strength > 0.15:
            # Strong seasonality
            if model_name in ('prophet', 'n_beats', 'n_hits', 'tft', 'deepar'):
                score += 15
            elif model_name in ('auto_arima', 'stacking_rnn', 'patch_rnn'):
                score += 8
            elif model_name in nn_light:
                score += 5

        # ---- Trend strength ----
        if p.trend_strength > 0.5:
            if model_name in ('prophet', 'd_linear', 'n_linear', 'tide'):
                score += 10
            elif model_name == 'auto_arima':
                score += 8

        # ---- Noise level ----
        if p.noise_ratio > 0.8:
            # High noise: prefer robust / regularized models
            if model_name in ('lightgbm', 'xgboost', 'random_forest', 'catboost'):
                score += 8
            elif model_name in ('n_beats', 'tcn'):
                score += 5
            # Penalize overly flexible models
            if model_name in ('srs_net', 'deepar'):
                score -= 5

        # ---- Skewness ----
        if abs(p.skewness) > 2.0:
            # Highly skewed: tree models handle this naturally
            if model_name in ml_models:
                score += 5
            # NN models may struggle without proper scaling (handled by scaler selection)

        # ---- Autocorrelation structure ----
        if p.autocorr_lag1 > 0.7:
            # Strong AR structure: sequential models excel
            if model_name in ('auto_arima', 'stacking_rnn', 'patch_rnn', 'tcn'):
                score += 10
            elif model_name in ('gau', 'time2vec', 'tft'):
                score += 5
            elif model_name in ('d_linear', 'n_linear'):
                score += 3
        elif p.autocorr_lag1 < 0.2:
            # Weak autocorrelation: tree models and attention-based
            if model_name in ml_models:
                score += 5
            elif model_name in ('transformer', 'itransformer', 'tft'):
                score += 5

        # ---- Multiple seasonalities ----
        if p.n_seasonalities >= 2:
            # Complex multi-seasonal patterns
            if model_name in ('prophet', 'tft', 'n_beats', 'deepar'):
                score += 12
            elif model_name in ('n_hits', 'itransformer', 'stacking_rnn'):
                score += 6
            # Penalize simple models for complex patterns
            if model_name in ('d_linear', 'n_linear'):
                score -= 3

        # ---- Forecast horizon relative to data ----
        if self.n_predict and p.n_rows > 0:
            ratio = self.n_predict / p.n_rows
            if ratio > 0.2:
                # Long horizon relative to data: extrapolation-capable models
                if model_name in ('prophet', 'auto_arima'):
                    score += 8
                elif model_name in ('d_linear', 'n_linear', 'tide'):
                    score += 5
                # Heavy NN models overfit on short data + long horizon
                if model_name in nn_heavy:
                    score -= 5
            elif ratio < 0.05:
                # Short horizon: complex models can afford to be used
                if model_name in nn_heavy:
                    score += 5

        # ---- Regime changes ----
        if p.regime_changes > 5:
            # Data with structural breaks: tree models handle discontinuities
            if model_name in ml_models:
                score += 8
            elif model_name in ('tft', 'deepar'):
                score += 5
            # Penalize models that assume smooth patterns
            if model_name in ('auto_arima', 'd_linear', 'n_linear'):
                score -= 3

        # ---- Speed bonus for production ----
        if model_name in statistic_models:
            score += 3
        elif model_name in ml_models:
            score += 2
        elif model_name in nn_light:
            score += 1

        # ---- Specific model strengths (conditional) ----
        # LightGBM: strong when data has sufficient features/complexity
        if model_name == 'lightgbm':
            if p.noise_ratio > 0.5 or n >= 100:
                score += 5

        # Prophet: handles missing data and holidays naturally
        if model_name == 'prophet' and p.pct_missing > 0.01:
            score += 5

        # NBeats: strong on clean periodic data
        if model_name == 'n_beats' and p.noise_ratio < 0.5 and p.seasonality_strength > 0.1:
            score += 8

        # TiDE: efficient and strong on medium-to-large data
        if model_name == 'tide' and n >= 100:
            score += 5

        # ITransformer: good for longer sequences
        if model_name == 'itransformer' and n >= 200:
            score += 5

        # GAU: strong on clean moderate-length data
        if model_name == 'gau' and 100 <= n <= 500 and p.noise_ratio < 0.7:
            score += 5

        # TCN: good for high-frequency data with local patterns
        if model_name == 'tcn' and p.autocorr_lag1 > 0.5:
            score += 5

        return score

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
    #  Ensemble Builder
    # ------------------------------------------------------------------

    def _build_ensemble(self):
        """Build a weighted ensemble from top-K models after pipeline fit.

        Uses inverse-metric weighting: better models get higher weights.
        In 'auto' mode, only builds ensemble when top models are within
        30% of the best model's metric (suggesting diverse strengths).

        Returns
        -------
        EnsemblePredictor or None
            The ensemble predictor, or None if ensemble is not beneficial.
        """
        lb = self.leader_board_
        if lb is None or len(lb) < 2:
            return None

        if self.ensemble_strategy == 'none':
            return None

        metrics = lb['metric'].values.astype(float)
        best_metric = metrics[0]

        if self.ensemble_strategy == 'auto':
            # Auto-detect: ensemble if multiple models are competitive
            if self.pipeline_.metric_less_is_better:
                # Lower is better: threshold = best * 1.3
                threshold = best_metric * 1.3 if best_metric > 0 else best_metric - abs(best_metric) * 0.3
                eligible = lb[lb['metric'].astype(float) <= threshold]
            else:
                # Higher is better: threshold = best * 0.7
                threshold = best_metric * 0.7 if best_metric > 0 else best_metric + abs(best_metric) * 0.3
                eligible = lb[lb['metric'].astype(float) >= threshold]

            if len(eligible) < 2:
                return None
            top_k = min(self.ensemble_top_k, len(eligible))
        else:
            # 'weighted_avg': always build ensemble
            top_k = min(self.ensemble_top_k, len(lb))

        top_models = lb.head(top_k)
        model_names = top_models['model'].tolist()
        model_metrics = top_models['metric'].values.astype(float)

        # Inverse-metric weighting
        if self.pipeline_.metric_less_is_better:
            # Lower is better: invert so that lower metric → higher weight
            inv = 1.0 / (model_metrics + 1e-10)
        else:
            # Higher is better: use directly
            inv = model_metrics.copy()
            inv[inv < 0] = 0  # safety

        total = inv.sum()
        if total <= 0:
            return None

        weights = inv / total
        weight_dict = dict(zip(model_names, weights.tolist()))

        return EnsemblePredictor(
            pipeline=self.pipeline_,
            model_names=model_names,
            weights=weight_dict,
            time_col=self.time_col,
            target_col=self.target_col,
        )

    # ------------------------------------------------------------------
    #  Logging helpers
    # ------------------------------------------------------------------

    def _log_profile(self):
        p = self.profile_
        self.logger.info(
            f"Data profile: n={p.n_rows}, freq={p.freq}, "
            f"stationarity={p.stationarity}, "
            f"trend={p.trend_strength:.2f}, "
            f"seasonality={p.seasonality_strength:.2f}, "
            f"noise={p.noise_ratio:.2f}, "
            f"skew={p.skewness:.2f}, "
            f"missing={p.pct_missing:.1%}, "
            f"outliers={p.pct_outlier:.1%}"
        )
        self.logger.info(
            f"  autocorr_lag1={p.autocorr_lag1:.2f}, "
            f"autocorr_lag2={p.autocorr_lag2:.2f}, "
            f"n_seasonalities={p.n_seasonalities}, "
            f"regime_changes={p.regime_changes}"
        )
        if p.dominant_periods:
            self.logger.info(f"Dominant periods: {p.dominant_periods}")

    def _log_strategy(self):
        s = self.strategy_
        self.logger.info(f"Selected lags: {s['lags']}")
        self.logger.info(f"Selected scaler: {s['scaler'].__class__.__name__}")
        self.logger.info(f"GBDT differencing: d={s['gbdt_differential_n']}")
        if s['preprocessing']:
            steps_str = ', '.join(
                f"{st['step']}({st.get('method', '')})" for st in s['preprocessing']
            )
            self.logger.info(f"Preprocessing: {steps_str}")
        else:
            self.logger.info("Preprocessing: none needed")
        self.logger.info(f"Selected models ({len(s['models'])}): {s['models']}")
        # Feature engineering
        fe = s.get('feature_engineering', {})
        if fe:
            self.logger.info(
                f"Feature engineering: routing_mode={fe.get('routing_mode', 'static')}, "
                f"prophet_lag_features={fe.get('prophet_use_lag_features', False)}, "
                f"prophet_season_mode={fe.get('prophet_seasonality_mode', 'auto')}"
            )
        # Hyperparams
        hp = s.get('model_hyperparams', {})
        if hp:
            hp_str = ', '.join(f"{k}={v}" for k, v in hp.items())
            self.logger.info(f"Adaptive hyperparams: {hp_str}")
        # Ensemble strategy
        self.logger.info(f"Ensemble strategy: {self.ensemble_strategy} (top_k={self.ensemble_top_k})")


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

    def __repr__(self):
        status = "fitted" if self.pipeline_ is not None else "not fitted"
        return (f"SmartRouter(time_col='{self.time_col}', "
                f"target_col='{self.target_col}', status={status})")
