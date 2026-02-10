"""Missing value detection and interpolation for time series data.

All operations are vectorized via numpy/pandas for performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal


class TimeSeriesMissingHandler:
    """Detect and fill missing values in time series data.

    Supports both explicit NaN gaps and implicit missing timestamps.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    freq : str or None, default=None
        Expected frequency (e.g. 'D', 'H', 'MS'). If None, auto-detected
        from the data using the most common time delta.

    Examples
    --------
    >>> handler = TimeSeriesMissingHandler(time_col='date')
    >>> report = handler.fit(df)
    >>> df_clean = handler.transform(df, method='linear')
    """

    def __init__(self, time_col: str, freq: Optional[str] = None):
        self.time_col = time_col
        self.freq = freq

    def _infer_freq(self, data: pd.DataFrame) -> str:
        """Infer frequency from the most common time delta."""
        if self.freq is not None:
            return self.freq
        diffs = data[self.time_col].diff().dropna()
        if len(diffs) == 0:
            raise ValueError("Cannot infer frequency from fewer than 2 data points.")
        return pd.tseries.frequencies.to_offset(diffs.mode().iloc[0])

    def fit(self, data: pd.DataFrame, value_cols: Optional[list] = None) -> dict:
        """Detect missing values (both explicit NaNs and implicit gaps).

        Parameters
        ----------
        data : pd.DataFrame
            Input time series data.
        value_cols : list or None
            Columns to check for NaN. If None, checks all non-time columns.

        Returns
        -------
        dict
            Report with keys:
            - 'n_explicit_nan': dict of {col: count} for NaN values
            - 'n_implicit_gaps': int, number of missing timestamps
            - 'missing_timestamps': pd.DatetimeIndex of missing timestamps
            - 'total_expected': int, total expected rows
            - 'completeness_ratio': float, actual/expected
        """
        if value_cols is None:
            value_cols = [c for c in data.columns if c != self.time_col]

        freq = self._infer_freq(data)
        ts = data[self.time_col]
        full_range = pd.date_range(start=ts.min(), end=ts.max(), freq=freq)
        existing = pd.DatetimeIndex(ts.values)
        missing_ts = full_range.difference(existing)

        nan_counts = {}
        for c in value_cols:
            n = int(data[c].isna().sum())
            if n > 0:
                nan_counts[c] = n

        total_expected = len(full_range)
        actual = len(data)

        return {
            'n_explicit_nan': nan_counts,
            'n_implicit_gaps': len(missing_ts),
            'missing_timestamps': missing_ts,
            'total_expected': total_expected,
            'completeness_ratio': actual / total_expected if total_expected > 0 else 1.0,
        }

    def transform(
        self,
        data: pd.DataFrame,
        method: Literal['linear', 'ffill', 'bfill', 'spline', 'zero'] = 'linear',
        value_cols: Optional[list] = None,
        spline_order: int = 3,
        fill_implicit_gaps: bool = True,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fill missing values in time series data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data. Not modified in place.
        method : {'linear', 'ffill', 'bfill', 'spline', 'zero'}
            Interpolation method.
        value_cols : list or None
            Columns to fill. If None, fills all non-time columns.
        spline_order : int, default=3
            Order of spline interpolation (only used when method='spline').
        fill_implicit_gaps : bool, default=True
            Whether to reindex to fill implicit timestamp gaps first.
        limit : int or None
            Maximum number of consecutive NaNs to fill.

        Returns
        -------
        pd.DataFrame
            Data with missing values filled.
        """
        df = data.copy()

        if value_cols is None:
            value_cols = [c for c in df.columns if c != self.time_col]

        if fill_implicit_gaps:
            freq = self._infer_freq(df)
            full_range = pd.date_range(
                start=df[self.time_col].min(),
                end=df[self.time_col].max(),
                freq=freq,
            )
            df = df.set_index(self.time_col).reindex(full_range)
            df.index.name = self.time_col
            df = df.reset_index()

        for c in value_cols:
            if c not in df.columns:
                continue
            if method == 'linear':
                df[c] = df[c].interpolate(method='linear', limit=limit)
            elif method == 'ffill':
                df[c] = df[c].ffill(limit=limit)
            elif method == 'bfill':
                df[c] = df[c].bfill(limit=limit)
            elif method == 'spline':
                df[c] = df[c].interpolate(method='spline', order=spline_order, limit=limit)
            elif method == 'zero':
                df[c] = df[c].fillna(0)
            else:
                raise ValueError(f"Unknown method '{method}'. "
                                 f"Choose from: 'linear', 'ffill', 'bfill', 'spline', 'zero'.")

        return df

    def fit_transform(
        self,
        data: pd.DataFrame,
        method: Literal['linear', 'ffill', 'bfill', 'spline', 'zero'] = 'linear',
        value_cols: Optional[list] = None,
        spline_order: int = 3,
        fill_implicit_gaps: bool = True,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fit (detect) and transform (fill) in one call."""
        self.fit(data, value_cols=value_cols)
        return self.transform(data, method=method, value_cols=value_cols,
                              spline_order=spline_order,
                              fill_implicit_gaps=fill_implicit_gaps, limit=limit)

    # Backward-compatible aliases
    detect = fit
    fill = transform
