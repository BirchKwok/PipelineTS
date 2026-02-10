"""Outlier detection and handling for time series data.

All operations are vectorized via numpy/pandas for performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, Union


class TimeSeriesOutlierDetector:
    """Detect and handle outliers in time series data.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    method : {'iqr', 'zscore', 'rolling_zscore', 'grubbs'}
        Detection method.
        - 'iqr': Inter-Quartile Range (robust to skewness).
        - 'zscore': Standard Z-score (assumes roughly normal).
        - 'rolling_zscore': Z-score within a rolling window (captures local anomalies).
        - 'grubbs': Grubbs' test for single outlier (iterative).
    threshold : float, default=1.5
        Sensitivity threshold.
        - For 'iqr': IQR multiplier (default 1.5, use 3.0 for extreme only).
        - For 'zscore' / 'rolling_zscore': Z-score cutoff (default 3.0).
        - For 'grubbs': significance level alpha (default 0.05).
    window : int or None, default=None
        Rolling window size for 'rolling_zscore'. If None, defaults to
        max(10, len(data) // 20).

    Examples
    --------
    >>> detector = TimeSeriesOutlierDetector(time_col='date', method='iqr')
    >>> mask = detector.fit(df, target_col='value')
    >>> df_clean = detector.transform(df, target_col='value', strategy='clip')
    """

    _DEFAULT_THRESHOLDS = {
        'iqr': 1.5,
        'zscore': 3.0,
        'rolling_zscore': 3.0,
        'grubbs': 0.05,
    }

    def __init__(
        self,
        time_col: str,
        method: Literal['iqr', 'zscore', 'rolling_zscore', 'grubbs'] = 'iqr',
        threshold: Optional[float] = None,
        window: Optional[int] = None,
    ):
        self.time_col = time_col
        self.method = method
        self.threshold = threshold if threshold is not None else self._DEFAULT_THRESHOLDS[method]
        self.window = window

    def fit(
        self,
        data: pd.DataFrame,
        target_col: Union[str, list],
    ) -> pd.DataFrame:
        """Detect outliers and return a boolean mask DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        target_col : str or list of str
            Column(s) to check for outliers.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame with same index as data. True = outlier.
        """
        if isinstance(target_col, str):
            target_col = [target_col]

        result = pd.DataFrame(index=data.index)
        for col in target_col:
            values = data[col].values.astype(np.float64)
            if self.method == 'iqr':
                result[col] = self._detect_iqr(values)
            elif self.method == 'zscore':
                result[col] = self._detect_zscore(values)
            elif self.method == 'rolling_zscore':
                w = self.window or max(10, len(values) // 20)
                result[col] = self._detect_rolling_zscore(values, w)
            elif self.method == 'grubbs':
                result[col] = self._detect_grubbs(values, alpha=self.threshold)
            else:
                raise ValueError(f"Unknown method '{self.method}'.")

        return result

    def transform(
        self,
        data: pd.DataFrame,
        target_col: Union[str, list],
        strategy: Literal['clip', 'nan', 'median', 'linear'] = 'clip',
    ) -> pd.DataFrame:
        """Replace outliers using the specified strategy.

        Parameters
        ----------
        data : pd.DataFrame
            Input data. Not modified in place.
        target_col : str or list of str
            Column(s) to handle.
        strategy : {'clip', 'nan', 'median', 'linear'}
            - 'clip': Clip to IQR/zscore bounds.
            - 'nan': Replace outliers with NaN (for later interpolation).
            - 'median': Replace outliers with rolling median.
            - 'linear': Replace outliers with linear interpolation.

        Returns
        -------
        pd.DataFrame
            Cleaned data.
        """
        if isinstance(target_col, str):
            target_col = [target_col]

        df = data.copy()
        mask = self.fit(data, target_col)

        for col in target_col:
            outlier_mask = mask[col].values
            if not outlier_mask.any():
                continue

            values = df[col].values.astype(np.float64)

            if strategy == 'clip':
                lower, upper = self._get_bounds(values)
                df[col] = np.clip(values, lower, upper)
            elif strategy == 'nan':
                values[outlier_mask] = np.nan
                df[col] = values
            elif strategy == 'median':
                w = self.window or max(10, len(values) // 20)
                rolling_med = pd.Series(values).rolling(
                    window=w, center=True, min_periods=1
                ).median().values
                values[outlier_mask] = rolling_med[outlier_mask]
                df[col] = values
            elif strategy == 'linear':
                values[outlier_mask] = np.nan
                df[col] = pd.Series(values).interpolate(method='linear').values
            else:
                raise ValueError(f"Unknown strategy '{strategy}'.")

        return df

    # ---- vectorized detection methods ----

    def _detect_iqr(self, values: np.ndarray) -> np.ndarray:
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        return (values < lower) | (values > upper)

    def _detect_zscore(self, values: np.ndarray) -> np.ndarray:
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            return np.zeros(len(values), dtype=bool)
        z = np.abs((values - mean) / std)
        return z > self.threshold

    def _detect_rolling_zscore(self, values: np.ndarray, window: int) -> np.ndarray:
        s = pd.Series(values)
        rolling_mean = s.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = s.rolling(window=window, center=True, min_periods=1).std()
        rolling_std = rolling_std.replace(0, np.nan).ffill().bfill().fillna(1.0)
        z = np.abs((s - rolling_mean) / rolling_std)
        return (z > self.threshold).values

    def _detect_grubbs(self, values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Iterative Grubbs test. Marks at most ~5% of points to avoid runaway."""
        from scipy import stats

        mask = np.zeros(len(values), dtype=bool)
        working = values.copy()
        max_iters = max(1, len(values) // 20)

        for _ in range(max_iters):
            n = np.sum(~np.isnan(working))
            if n < 3:
                break
            valid = working[~np.isnan(working)]
            mean = np.mean(valid)
            std = np.std(valid, ddof=1)
            if std == 0:
                break

            residuals = np.abs(working - mean)
            idx = np.nanargmax(residuals)
            g = residuals[idx] / std

            # Critical value from t-distribution
            t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

            if g > g_crit:
                mask[idx] = True
                working[idx] = np.nan
            else:
                break

        return mask

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_col: Union[str, list],
        strategy: Literal['clip', 'nan', 'median', 'linear'] = 'clip',
    ) -> pd.DataFrame:
        """Fit (detect) and transform (handle) in one call."""
        self.fit(data, target_col)
        return self.transform(data, target_col, strategy=strategy)

    # Backward-compatible aliases
    detect = fit
    handle = transform

    def _get_bounds(self, values: np.ndarray):
        """Get lower/upper bounds based on current method."""
        if self.method == 'iqr':
            q1, q3 = np.nanpercentile(values, [25, 75])
            iqr = q3 - q1
            return q1 - self.threshold * iqr, q3 + self.threshold * iqr
        else:
            mean = np.nanmean(values)
            std = np.nanstd(values)
            return mean - self.threshold * std, mean + self.threshold * std
