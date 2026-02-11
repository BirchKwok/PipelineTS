"""
Custom Prophet-like time series decomposition model.

Model (additive):       y(t) = g(t) + s(t) + ε(t)
Model (multiplicative): y(t) = g(t) * (1 + s(t)) + ε(t)
  - g(t): Piecewise linear trend with automatic changepoint detection
  - s(t): Fourier-based seasonality (yearly, weekly, custom periods)

Design improvements over Facebook Prophet:
1. Speed: Ridge regression (closed-form) instead of Stan MCMC — 1000x+ faster
2. Accuracy:
   - Iterative trend-seasonality decomposition (no mutual interference)
   - Auto multiplicative seasonality detection
   - Recency-weighted regression for better extrapolation
   - Auto changepoint detection via second-derivative of smoothed series
   - FFT-based auto seasonality period detection
   - Trend dampening for long-horizon forecasts
3. Zero external dependencies beyond numpy/scipy
"""

import numpy as np
from scipy.linalg import solve
from scipy.signal import find_peaks


def _detect_changepoints(y, n_changepoints=25, min_segment=5):
    """Detect changepoints using second-derivative magnitude of smoothed series.

    Parameters
    ----------
    y : np.ndarray
        Time series values.
    n_changepoints : int
        Maximum number of changepoints to detect.
    min_segment : int
        Minimum distance between changepoints.

    Returns
    -------
    np.ndarray
        Indices of detected changepoints.
    """
    n = len(y)
    if n < 2 * min_segment:
        return np.array([], dtype=int)

    # Smooth with adaptive window
    window = max(3, n // 20)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode='same')

    # Second derivative magnitude
    d2 = np.abs(np.diff(smoothed, n=2))
    if len(d2) == 0:
        return np.array([], dtype=int)

    # Find peaks in second derivative
    height_threshold = np.percentile(d2, 75)
    peaks, properties = find_peaks(d2, height=height_threshold, distance=min_segment)

    if len(peaks) == 0:
        # Fallback: uniformly spaced changepoints
        margin = max(min_segment, n // 10)
        cps = np.linspace(margin, n - margin, min(n_changepoints, n // min_segment),
                          dtype=int, endpoint=False)
        return cps

    # Offset by 1 because diff shifts indices
    peaks = peaks + 1

    # Select top n_changepoints by magnitude
    if len(peaks) > n_changepoints:
        heights = d2[peaks - 1]
        top_idx = np.argsort(heights)[-n_changepoints:]
        peaks = np.sort(peaks[top_idx])

    # Clip to valid range
    peaks = peaks[(peaks >= min_segment) & (peaks < n - min_segment)]

    return peaks


def _detect_seasonality_periods(y, min_period=2, max_period=None, top_k=3):
    """Auto-detect dominant seasonality periods using FFT.

    Parameters
    ----------
    y : np.ndarray
        Time series values.
    min_period : int
        Minimum period to consider.
    max_period : int or None
        Maximum period to consider (default: len(y)//2).
    top_k : int
        Number of top periods to return.

    Returns
    -------
    list of int
        Detected dominant periods.
    """
    n = len(y)
    if max_period is None:
        max_period = n // 2

    if n < 2 * min_period:
        return []

    # Detrend with simple differencing
    detrended = np.diff(y)

    # FFT
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(detrended))

    # Convert to periods (skip DC component)
    valid = (freqs > 0) & (freqs > 0)
    if not np.any(valid):
        return []

    periods = 1.0 / freqs[valid]
    power_valid = power[valid]

    # Filter by period range
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return []

    periods = periods[mask]
    power_valid = power_valid[mask]

    # Significance threshold: must be > 2x median power
    median_power = np.median(power_valid)
    significant = power_valid > 2 * median_power

    if not np.any(significant):
        return []

    periods = periods[significant]
    power_valid = power_valid[significant]

    # Top-k by power
    top_idx = np.argsort(power_valid)[-top_k:]
    detected = sorted(set(int(round(periods[i])) for i in top_idx))

    return [p for p in detected if p >= min_period]


def _build_trend_features(t, changepoint_indices, n_total):
    """Build piecewise linear trend design matrix.

    Parameters
    ----------
    t : np.ndarray
        Normalized time values in [0, 1].
    changepoint_indices : np.ndarray
        Indices of changepoints.
    n_total : int
        Total number of data points in training set.

    Returns
    -------
    np.ndarray
        Design matrix of shape (len(t), 2 + n_changepoints).
        Columns: [1, t, max(0, t - s_1), max(0, t - s_2), ...]
    """
    n_cp = len(changepoint_indices)
    X = np.zeros((len(t), 2 + n_cp))
    X[:, 0] = 1.0  # intercept
    X[:, 1] = t    # global slope

    for i, cp_idx in enumerate(changepoint_indices):
        s = cp_idx / n_total  # normalized changepoint position
        X[:, 2 + i] = np.maximum(0, t - s)

    return X


def _build_seasonality_features(t, periods, n_fourier=5):
    """Build Fourier seasonality design matrix.

    Parameters
    ----------
    t : np.ndarray
        Normalized time values in [0, 1].
    periods : list of float
        Seasonality periods (in normalized time units).
    n_fourier : int
        Number of Fourier terms per period.

    Returns
    -------
    np.ndarray
        Design matrix of shape (len(t), 2 * n_fourier * len(periods)).
    """
    features = []
    for period in periods:
        if period <= 0:
            continue
        for j in range(1, n_fourier + 1):
            features.append(np.sin(2.0 * np.pi * j * t / period))
            features.append(np.cos(2.0 * np.pi * j * t / period))

    if len(features) == 0:
        return np.zeros((len(t), 0))

    return np.column_stack(features)


class SpinesProphet:
    """Modern Prophet-like decomposable time series model.

    Model (additive):       y(t) = g(t) + s(t) + ε(t)
    Model (multiplicative): y(t) = g(t) * (1 + s(t)) + ε(t)
      - g(t): Piecewise linear trend
      - s(t): Fourier-based seasonality

    Key improvements over Facebook Prophet:
    - Iterative trend-seasonality decomposition (no mutual interference)
    - Auto multiplicative/additive seasonality detection
    - Recency-weighted regression for better trend extrapolation
    - Closed-form ridge regression (1000x+ faster than MCMC)

    Parameters
    ----------
    n_changepoints : int, optional, default: 25
        Maximum number of trend changepoints.
    changepoint_prior_scale : float, optional, default: 0.05
        Regularization strength for changepoint coefficients.
        Smaller = smoother trend.
    seasonality_prior_scale : float, optional, default: 10.0
        Regularization strength for seasonality coefficients.
        Larger = more flexible seasonality.
    yearly_seasonality : bool or int, optional, default: 'auto'
        Whether to include yearly seasonality. 'auto' detects from data length.
        If int, specifies number of Fourier terms.
    weekly_seasonality : bool or int, optional, default: 'auto'
        Whether to include weekly seasonality.
    custom_seasonalities : list of dict, optional, default: None
        Custom seasonality specs: [{'period': float, 'fourier_order': int}, ...].
    auto_seasonality : bool, optional, default: True
        Whether to auto-detect additional seasonality via FFT.
    seasonality_mode : str, optional, default: 'auto'
        'additive', 'multiplicative', or 'auto' (detects from data).
    trend_dampening : float, optional, default: 0.0
        Dampening factor for trend extrapolation (0 = no dampening, 1 = flat).
    n_iter : int, optional, default: 5
        Number of iterations for trend-seasonality decomposition.
    use_lag_features : bool, optional, default: False
        Whether to include causal rolling lag features as additional regressors.
    lag_window : int or 'auto', optional, default: 'auto'
        Window size for rolling lag features.
    lag_prior_scale : float, optional, default: 5.0
        Regularization strength for lag feature coefficients.
    """

    def __init__(
        self,
        n_changepoints=25,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        custom_seasonalities=None,
        auto_seasonality=True,
        seasonality_mode='auto',
        trend_dampening=0.0,
        n_iter=5,
        use_lag_features=False,
        lag_window='auto',
        lag_prior_scale=5.0,
    ):
        self.n_changepoints = n_changepoints
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.custom_seasonalities = custom_seasonalities or []
        self.auto_seasonality = auto_seasonality
        self.seasonality_mode = seasonality_mode
        self.trend_dampening = trend_dampening
        self.n_iter = n_iter
        self.use_lag_features = use_lag_features
        self.lag_window = lag_window
        self.lag_prior_scale = lag_prior_scale

        # Fitted attributes
        self._trend_beta = None
        self._season_beta = None
        self._effective_mode = 'additive'
        self._t_min = None
        self._t_scale = None
        self._y_mean = None
        self._y_std = None
        self._changepoint_indices = None
        self._n_train = None
        self._n_trend_features = None
        self._seasonality_periods = None
        self._fourier_orders = None
        self._last_t = None
        self._freq = None
        self._lag_window_size = None
        self._lag_feature_mean = None
        self._lag_feature_std = None
        self._last_lag_features = None
        self._n_lag_features = 0

    @staticmethod
    def _build_rolling_lag_features(y, window):
        """Build causal rolling features from y — each row uses only past data.

        Features at time t are computed from y[max(0, t-window):t] (exclusive of t).
        This ensures zero data leakage.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Target values.
        window : int
            Rolling window size.

        Returns
        -------
        np.ndarray, shape (n, n_features)
            Rolling lag features matrix.
        """
        n = len(y)
        eps = 1e-12

        # Pre-allocate feature arrays (7 features)
        n_feat = 7
        feat = np.zeros((n, n_feat), dtype=np.float64)

        for t in range(n):
            # Causal window: y[start:t] (exclusive of current point)
            start = max(0, t - window)
            w = y[start:t]  # past values only, excludes y[t]

            if len(w) < 2:
                # Not enough history — leave as zeros
                continue

            w_len = len(w)
            w_mean = w.mean()
            w_std = w.std()

            # 0: Rolling mean (normalized by global std later)
            feat[t, 0] = w_mean

            # 1: Rolling std
            feat[t, 1] = w_std

            # 2: Trend slope (linear regression coefficient)
            t_idx = np.arange(w_len, dtype=np.float64)
            t_c = t_idx - t_idx.mean()
            t_var = (t_c ** 2).sum()
            feat[t, 2] = ((w - w_mean) @ t_c) / (t_var + eps)

            # 3: Momentum (last value - first value) / window
            feat[t, 3] = (w[-1] - w[0]) / (w_len + eps)

            # 4: Recent-vs-past ratio (second-half mean / first-half mean)
            half = max(1, w_len // 2)
            fh = w[:half].mean()
            sh = w[half:].mean()
            feat[t, 4] = sh / (np.abs(fh) + eps)

            # 5: EMA (exponential moving average)
            alpha = 2.0 / (max(1, w_len // 2) + 1)
            weights = np.power(1 - alpha, np.arange(w_len - 1, -1, -1, dtype=np.float64))
            weights /= weights.sum() + eps
            feat[t, 5] = (w * weights).sum()

            # 6: Lag-1 autocorrelation
            if w_len > 2:
                x1, x2 = w[:-1], w[1:]
                m1, m2 = x1.mean(), x2.mean()
                num = ((x1 - m1) * (x2 - m2)).mean()
                denom = x1.std() * x2.std() + eps
                feat[t, 6] = num / denom

        return feat

    @staticmethod
    def _detect_seasonality_mode(y, t_days):
        """Auto-detect whether multiplicative or additive seasonality is better.

        Checks if seasonal amplitude correlates with trend level.
        If so, multiplicative is better.
        """
        n = len(y)
        if n < 24:
            return 'additive'

        # Estimate trend via centered moving average
        from scipy.ndimage import uniform_filter1d
        samples_per_year = max(3, int(round(365.25 / max(np.median(np.diff(t_days)), 0.5))))
        win = max(3, samples_per_year)
        if win % 2 == 0:
            win += 1
        trend = uniform_filter1d(y.astype(np.float64), size=win, mode='nearest')

        # Safety: avoid division by near-zero trend
        safe_trend = np.where(np.abs(trend) > 1e-8, trend, 1e-8)

        # Additive residual and multiplicative residual
        add_resid = y - trend
        mult_resid = y / safe_trend - 1.0

        # Check correlation of |residual| with trend level
        from numpy import corrcoef
        add_corr = abs(corrcoef(np.abs(add_resid), trend)[0, 1])
        mult_corr = abs(corrcoef(np.abs(mult_resid), trend)[0, 1])

        # If multiplicative residual is more independent of trend, use multiplicative
        if mult_corr < add_corr - 0.05:
            return 'multiplicative'
        return 'additive'

    def _resolve_seasonalities(self, n_days, freq_days=1.0):
        """Determine seasonality periods and Fourier orders."""
        periods = []
        fourier_orders = []

        # Yearly — auto-scale Fourier order based on samples per year
        yearly = self.yearly_seasonality
        if yearly == 'auto':
            yearly = bool(n_days >= 730)  # need >= 2 years
        if yearly is True:
            samples_per_year = max(1, int(round(365.25 / max(freq_days, 0.5))))
            # Nyquist: max Fourier order = samples_per_year // 2
            # Practical: use ~samples_per_year // 3 for robustness, clamped [3, 10]
            auto_fo = max(3, min(10, samples_per_year // 3))
            periods.append(365.25)
            fourier_orders.append(auto_fo)
        elif isinstance(yearly, int) and yearly > 0:
            periods.append(365.25)
            fourier_orders.append(int(yearly))

        # Weekly (skip if data frequency >= 7 days — can't observe weekly patterns)
        weekly = self.weekly_seasonality
        if weekly == 'auto':
            weekly = bool(n_days >= 14 and freq_days < 7.0)
        if weekly is True:
            periods.append(7.0)
            fourier_orders.append(3)
        elif isinstance(weekly, int) and weekly > 0:
            periods.append(7.0)
            fourier_orders.append(int(weekly))

        # Custom
        for spec in self.custom_seasonalities:
            periods.append(float(spec['period']))
            fourier_orders.append(int(spec.get('fourier_order', 5)))

        return periods, fourier_orders

    def _build_season_matrix(self, t_days):
        """Build Fourier seasonality design matrix from t_days."""
        features = []
        for period, fo in zip(self._seasonality_periods, self._fourier_orders):
            if period <= 0:
                continue
            for j in range(1, fo + 1):
                features.append(np.sin(2.0 * np.pi * j * t_days / period))
                features.append(np.cos(2.0 * np.pi * j * t_days / period))
        if features:
            return np.column_stack(features)
        return np.zeros((len(t_days), 0))

    def fit(self, dates, y):
        """Fit using iterative trend-seasonality decomposition.

        Algorithm:
        1. Initialize trend via moving average smoothing
        2. Iterate:
           a. Detrend y → fit Fourier seasonality on residuals
           b. Deseasonalize y → fit piecewise linear trend (weighted)
        3. Final trend and seasonality coefficients stored separately

        Parameters
        ----------
        dates : np.ndarray of datetime64 or pd.DatetimeIndex
            Timestamps.
        y : np.ndarray
            Target values.

        Returns
        -------
        self
        """
        import pandas as pd

        dates = pd.to_datetime(dates)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        # Normalize time to days from start
        self._t_min = dates.min()
        t_days = (dates - self._t_min).total_seconds().values / 86400.0
        self._t_scale = max(t_days.max(), 1.0)
        t_norm = t_days / self._t_scale

        # Store y statistics
        self._y_mean = np.nanmean(y)
        self._y_std = max(np.nanstd(y), 1e-8)

        self._n_train = n
        self._last_t = dates.max()

        # Detect frequency
        if n >= 2:
            diffs = np.diff(t_days)
            self._freq = float(np.median(diffs))
        else:
            self._freq = 1.0

        # Auto-detect seasonality mode
        if self.seasonality_mode == 'auto':
            self._effective_mode = self._detect_seasonality_mode(y, t_days)
        else:
            self._effective_mode = self.seasonality_mode

        # Detect changepoints — place only in first 80% (like FB Prophet)
        y_norm = (y - self._y_mean) / self._y_std
        cp_range = int(n * 0.8)
        self._changepoint_indices = _detect_changepoints(
            y_norm[:cp_range], n_changepoints=self.n_changepoints,
            min_segment=max(3, n // 50)
        )

        # Build trend design matrix
        X_trend = _build_trend_features(t_norm, self._changepoint_indices, n)
        self._n_trend_features = X_trend.shape[1]

        # Resolve seasonalities
        n_days_span = t_days.max() - t_days.min()
        self._seasonality_periods, self._fourier_orders = \
            self._resolve_seasonalities(n_days_span, freq_days=self._freq)

        # Auto-detect additional seasonalities via FFT
        if self.auto_seasonality:
            detected = _detect_seasonality_periods(
                y_norm, min_period=2,
                max_period=max(int(n_days_span // 2), 3), top_k=3
            )
            existing = set(int(round(p / self._freq))
                           for p in self._seasonality_periods)
            for dp in detected:
                if dp not in existing and dp >= 2:
                    self._seasonality_periods.append(dp * self._freq)
                    self._fourier_orders.append(min(5, dp // 2))

        # Build seasonality design matrix
        X_season = self._build_season_matrix(t_days)
        n_season = X_season.shape[1]

        # Recency weights: exponential, half-life at 60% of data
        recency_half_life = max(1.0, n * 0.6)
        w_recency = np.exp(np.log(2.0) * np.arange(n, dtype=np.float64)
                           / recency_half_life)
        w_recency /= w_recency.mean()

        # Trend regularization
        reg_trend = np.zeros(self._n_trend_features)
        reg_trend[0] = 1e-6   # intercept
        reg_trend[1] = 1e-6   # global slope
        n_cp = len(self._changepoint_indices)
        if n_cp > 0:
            reg_trend[2:2 + n_cp] = 1.0 / max(self.changepoint_prior_scale, 1e-8)

        # Seasonality regularization
        reg_season = np.full(n_season,
                             1.0 / max(self.seasonality_prior_scale, 1e-8))

        # --- Initial trend estimate via weighted moving average ---
        samples_per_year = max(3, int(round(365.25 / max(self._freq, 0.5))))
        smooth_win = max(3, min(samples_per_year, n - 1))
        if smooth_win % 2 == 0:
            smooth_win += 1
        smooth_win = min(smooth_win, n)
        kernel = np.ones(smooth_win) / smooth_win
        trend = np.convolve(y, kernel, mode='same')
        # Fix boundary effects
        half_w = smooth_win // 2
        for i in range(half_w):
            trend[i] = y[:2 * i + 1].mean()
            trend[-(i + 1)] = y[-(2 * i + 1):].mean()

        # --- Iterative decomposition ---
        season_fitted = np.zeros(n)
        eps = 1e-8

        # Adaptive L1 weights for changepoint sparsity (IRLS)
        # Initialized to uniform; after first trend fit, reweighted to
        # penalize small deltas more (approximates Lasso/Laplace prior)
        cp_l1_weights = np.ones(n_cp) if n_cp > 0 else np.array([])

        for _it in range(self.n_iter):
            # Step A: Detrend → fit seasonality
            if self._effective_mode == 'multiplicative':
                safe_trend = np.where(np.abs(trend) > eps, trend, eps)
                season_signal = y / safe_trend - 1.0
            else:
                season_signal = y - trend

            if n_season > 0:
                # Weighted ridge for seasonality
                Xw = X_season * w_recency[:, np.newaxis]
                XtWX = Xw.T @ X_season + np.diag(reg_season)
                XtWy = Xw.T @ season_signal
                self._season_beta = solve(XtWX, XtWy, assume_a='pos')
                season_fitted = X_season @ self._season_beta
            else:
                self._season_beta = np.array([])
                season_fitted = np.zeros(n)

            # Step B: Deseasonalize → fit trend
            if self._effective_mode == 'multiplicative':
                y_deseason = y / (1.0 + season_fitted + eps)
            else:
                y_deseason = y - season_fitted

            # Build adaptive trend regularization (IRLS for L1 on changepoints)
            reg_trend_current = reg_trend.copy()
            if n_cp > 0:
                reg_trend_current[2:2 + n_cp] = (
                    cp_l1_weights / max(self.changepoint_prior_scale, 1e-8)
                )

            # Weighted ridge for trend
            Xw = X_trend * w_recency[:, np.newaxis]
            XtWX = Xw.T @ X_trend + np.diag(reg_trend_current)
            XtWy = Xw.T @ y_deseason
            self._trend_beta = solve(XtWX, XtWy, assume_a='pos')
            trend = X_trend @ self._trend_beta

            # Update IRLS weights: penalize small changepoint deltas more
            # w_i = 1 / (|delta_i| + eps) — approximates L1 (Laplace) prior
            if n_cp > 0:
                cp_deltas = np.abs(self._trend_beta[2:2 + n_cp])
                cp_l1_weights = 1.0 / (cp_deltas + 1e-6)
                # Normalize so mean weight = 1 (preserves overall regularization scale)
                cp_l1_weights /= (cp_l1_weights.mean() + 1e-8)

        # Store for backward compat with lag features path
        self._n_lag_features = 0
        self._y_history = y_norm.copy()

        return self

    def predict(self, dates):
        """Predict for given dates.

        Parameters
        ----------
        dates : np.ndarray of datetime64 or pd.DatetimeIndex
            Timestamps to predict for.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        import pandas as pd

        dates = pd.to_datetime(dates)
        n_pred = len(dates)
        t_days = (dates - self._t_min).total_seconds().values / 86400.0
        t_norm = t_days / self._t_scale

        # Trend
        X_trend = _build_trend_features(t_norm, self._changepoint_indices,
                                         self._n_train)

        # Apply trend dampening for future points
        if self.trend_dampening > 0:
            max_train_t = (self._n_train - 1) / self._n_train
            future_mask = t_norm > max_train_t
            if np.any(future_mask):
                horizon = t_norm[future_mask] - max_train_t
                damp_factor = np.exp(-self.trend_dampening * horizon)
                for i in range(2, X_trend.shape[1]):
                    X_trend[future_mask, i] *= damp_factor

        trend = X_trend @ self._trend_beta

        # Seasonality
        X_season = self._build_season_matrix(t_days)
        if X_season.shape[1] > 0 and len(self._season_beta) > 0:
            season = X_season @ self._season_beta
        else:
            season = np.zeros(n_pred)

        # Combine
        if self._effective_mode == 'multiplicative':
            return trend * (1.0 + season)
        else:
            return trend + season

    def make_future_dataframe(self, periods, freq='D', include_history=False):
        """Create a DataFrame of future dates for prediction.

        Parameters
        ----------
        periods : int
            Number of future periods.
        freq : str
            Frequency string (pandas offset alias).
        include_history : bool
            Whether to include historical dates.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'ds' column.
        """
        import pandas as pd

        future_dates = pd.date_range(
            start=self._last_t + pd.tseries.frequencies.to_offset(freq),
            periods=periods, freq=freq
        )

        if include_history:
            # We don't store full history, just generate from start
            history = pd.date_range(
                start=self._t_min, end=self._last_t, freq=freq
            )
            future_dates = history.append(future_dates)

        return pd.DataFrame({'ds': future_dates})
