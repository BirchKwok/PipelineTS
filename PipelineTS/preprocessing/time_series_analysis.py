"""Stationarity tests, frequency auto-detection, and time series splitting.

All statistical tests delegate to statsmodels for correctness.
Splitting utilities are pure numpy/pandas for performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, Tuple, Generator


# ---------------------------------------------------------------------------
#  Stationarity Tests
# ---------------------------------------------------------------------------

class StationarityTest:
    """Stationarity testing for time series data.

    Wraps ADF and KPSS tests with a unified interface and actionable output.

    Parameters
    ----------
    significance_level : float, default=0.05
        p-value threshold for stationarity decisions.

    Examples
    --------
    >>> tester = StationarityTest()
    >>> result = tester.fit(df['value'].values)
    >>> print(result['conclusion'])
    >>> tester.suggest_differencing(df['value'].values)
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def adf_test(self, series: np.ndarray) -> dict:
        """Augmented Dickey-Fuller test for unit root.

        Parameters
        ----------
        series : array-like
            Time series values.

        Returns
        -------
        dict
            Keys: 'statistic', 'p_value', 'used_lag', 'n_obs',
            'critical_values', 'is_stationary'.
        """
        from statsmodels.tsa.stattools import adfuller

        series = np.asarray(series, dtype=np.float64)
        series = series[~np.isnan(series)]
        result = adfuller(series, autolag='AIC')

        return {
            'test': 'ADF',
            'statistic': float(result[0]),
            'p_value': float(result[1]),
            'used_lag': int(result[2]),
            'n_obs': int(result[3]),
            'critical_values': {k: float(v) for k, v in result[4].items()},
            'is_stationary': result[1] < self.significance_level,
        }

    def kpss_test(self, series: np.ndarray, regression: str = 'c') -> dict:
        """KPSS test for stationarity.

        Parameters
        ----------
        series : array-like
            Time series values.
        regression : {'c', 'ct'}
            'c' for level stationarity, 'ct' for trend stationarity.

        Returns
        -------
        dict
            Keys: 'statistic', 'p_value', 'used_lag',
            'critical_values', 'is_stationary'.
        """
        from statsmodels.tsa.stattools import kpss
        import warnings

        series = np.asarray(series, dtype=np.float64)
        series = series[~np.isnan(series)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value, lags, crit = kpss(series, regression=regression, nlags='auto')

        return {
            'test': 'KPSS',
            'statistic': float(stat),
            'p_value': float(p_value),
            'used_lag': int(lags),
            'critical_values': {k: float(v) for k, v in crit.items()},
            'is_stationary': p_value >= self.significance_level,
        }

    def fit(self, series: np.ndarray) -> dict:
        """Run both ADF and KPSS tests and provide a combined conclusion.

        Parameters
        ----------
        series : array-like
            Time series values.

        Returns
        -------
        dict
            Keys: 'adf', 'kpss', 'conclusion', 'suggested_action'.
        """
        adf = self.adf_test(series)
        kpss_result = self.kpss_test(series)

        if adf['is_stationary'] and kpss_result['is_stationary']:
            conclusion = 'stationary'
            action = 'No differencing needed.'
        elif adf['is_stationary'] and not kpss_result['is_stationary']:
            conclusion = 'trend_stationary'
            action = 'Consider detrending or first differencing.'
        elif not adf['is_stationary'] and kpss_result['is_stationary']:
            conclusion = 'difference_stationary'
            action = 'Apply first differencing (d=1).'
        else:
            conclusion = 'non_stationary'
            action = 'Apply differencing. Consider d=1 or d=2.'

        return {
            'adf': adf,
            'kpss': kpss_result,
            'conclusion': conclusion,
            'suggested_action': action,
        }

    def suggest_differencing(self, series: np.ndarray, max_d: int = 2) -> int:
        """Suggest the minimum differencing order to achieve stationarity.

        Parameters
        ----------
        series : array-like
            Time series values.
        max_d : int, default=2
            Maximum differencing order to try.

        Returns
        -------
        int
            Suggested differencing order (0, 1, or 2).
        """
        s = np.asarray(series, dtype=np.float64)
        s = s[~np.isnan(s)]
        for d in range(max_d + 1):
            adf = self.adf_test(s)
            if adf['is_stationary']:
                return d
            s = np.diff(s)
            if len(s) < 10:
                break
        return max_d

    # Backward-compatible alias
    test = fit


# ---------------------------------------------------------------------------
#  Frequency / Period Detection
# ---------------------------------------------------------------------------

class FrequencyDetector:
    """Auto-detect sampling frequency and dominant seasonal periods.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.

    Examples
    --------
    >>> detector = FrequencyDetector(time_col='date')
    >>> info = detector.fit(df)
    >>> print(info['freq'], info['dominant_periods'])
    """

    def __init__(self, time_col: str):
        self.time_col = time_col

    def fit(self, data: pd.DataFrame, target_col: Optional[str] = None) -> dict:
        """Detect frequency and dominant periods.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with a datetime column.
        target_col : str or None
            If provided, also runs spectral analysis to find dominant periods.

        Returns
        -------
        dict
            Keys: 'freq' (pd offset string), 'freq_timedelta', 'is_regular',
            'dominant_periods' (list of int, if target_col given).
        """
        ts = data[self.time_col].sort_values()
        diffs = ts.diff().dropna()

        mode_diff = diffs.mode().iloc[0]
        try:
            freq = pd.tseries.frequencies.to_offset(mode_diff)
            freq_str = str(freq)
        except Exception:
            freq_str = str(mode_diff)

        result = {
            'freq': freq_str,
            'freq_timedelta': mode_diff,
            'is_regular': bool((diffs == mode_diff).all()),
        }

        if target_col is not None and target_col in data.columns:
            result['dominant_periods'] = self._find_dominant_periods(
                data[target_col].values
            )

        return result

    @staticmethod
    def _find_dominant_periods(values: np.ndarray, top_k: int = 3) -> list:
        """Find dominant periods using FFT spectral analysis.

        Parameters
        ----------
        values : np.ndarray
            Time series values.
        top_k : int, default=3
            Number of top periods to return.

        Returns
        -------
        list of int
            Dominant periods sorted by spectral power (descending).
        """
        values = np.asarray(values, dtype=np.float64)
        values = values[~np.isnan(values)]
        n = len(values)
        if n < 6:
            return []

        # Detrend with linear fit for cleaner spectrum
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, values, 1)
        detrended = values - np.polyval(coeffs, x)

        fft_vals = np.fft.rfft(detrended)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # Skip DC component (index 0) and very low frequencies
        min_period = 2
        max_period = n // 2
        valid = (freqs > 0) & (1.0 / np.where(freqs > 0, freqs, 1) <= max_period)
        valid &= (1.0 / np.where(freqs > 0, freqs, 1) >= min_period)

        if not valid.any():
            return []

        valid_indices = np.where(valid)[0]
        sorted_by_power = valid_indices[np.argsort(power[valid_indices])[::-1]]
        top_indices = sorted_by_power[:top_k]

        periods = []
        for idx in top_indices:
            if freqs[idx] > 0:
                p = int(round(1.0 / freqs[idx]))
                if p >= min_period and p not in periods:
                    periods.append(p)

        return periods

    # Backward-compatible alias
    detect = fit


# ---------------------------------------------------------------------------
#  Time Series Train/Test Split
# ---------------------------------------------------------------------------

class TimeSeriesSplit:
    """Time-aware train/test splitting for time series data.

    Unlike sklearn's random split, this preserves temporal ordering.

    Examples
    --------
    >>> splitter = TimeSeriesSplit()
    >>> train, test = splitter.split(df, time_col='date', test_size=0.2)
    >>> for train_df, test_df in splitter.expanding_window(df, ...):
    ...     model.fit(train_df)
    """

    @staticmethod
    def split(
        data: pd.DataFrame,
        time_col: str,
        test_size: Union[int, float] = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simple temporal train/test split.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        time_col : str
            Datetime column name.
        test_size : int or float
            If float, fraction of data for test set.
            If int, number of rows for test set.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            (train, test) with original index reset.
        """
        df = data.sort_values(time_col).reset_index(drop=True)
        n = len(df)

        if isinstance(test_size, float):
            n_test = max(1, int(n * test_size))
        else:
            n_test = min(test_size, n - 1)

        split_idx = n - n_test
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    @staticmethod
    def expanding_window(
        data: pd.DataFrame,
        time_col: str,
        min_train_size: int,
        test_size: int,
        step: int = 1,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Expanding (anchored) window cross-validation.

        The training set always starts from the beginning and grows.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        time_col : str
            Datetime column name.
        min_train_size : int
            Minimum number of training observations.
        test_size : int
            Number of test observations per fold.
        step : int, default=1
            Number of observations to advance between folds.

        Yields
        ------
        tuple of (pd.DataFrame, pd.DataFrame)
            (train, test) for each fold.
        """
        df = data.sort_values(time_col).reset_index(drop=True)
        n = len(df)

        start = min_train_size
        while start + test_size <= n:
            train = df.iloc[:start].copy()
            test = df.iloc[start:start + test_size].copy()
            yield train, test
            start += step

    @staticmethod
    def sliding_window(
        data: pd.DataFrame,
        time_col: str,
        train_size: int,
        test_size: int,
        step: int = 1,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Sliding (rolling) window cross-validation.

        The training window has a fixed size and slides forward.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        time_col : str
            Datetime column name.
        train_size : int
            Fixed number of training observations.
        test_size : int
            Number of test observations per fold.
        step : int, default=1
            Number of observations to advance between folds.

        Yields
        ------
        tuple of (pd.DataFrame, pd.DataFrame)
            (train, test) for each fold.
        """
        df = data.sort_values(time_col).reset_index(drop=True)
        n = len(df)

        start = 0
        while start + train_size + test_size <= n:
            train = df.iloc[start:start + train_size].copy()
            test = df.iloc[start + train_size:start + train_size + test_size].copy()
            yield train, test
            start += step
