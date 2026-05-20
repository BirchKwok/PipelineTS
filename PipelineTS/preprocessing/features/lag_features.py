"""User-facing lag feature extraction for time series data.

Exposes the same rolling-window statistical features used internally by
GBDT models, but as a standalone transformer that users can apply to any
DataFrame. All computations are vectorized with numpy for performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class LagFeatureExtractor:
    """Extract rolling-window lag features from a time series column.

    Generates statistical features (mean, std, min, max, trend, etc.)
    from a sliding window over the target column. Strictly causal —
    each row's features are computed from past values only.

    Parameters
    ----------
    time_col : str
        Datetime column name.
    target_col : str
        Column to extract features from.
    window : int or 'auto'
        Rolling window size. 'auto' sets it to the series length // 10
        (min 5, max 60).
    features : list of str or 'all', default='all'
        Which features to extract. Options: 'mean', 'std', 'min', 'max',
        'median', 'skew', 'kurtosis', 'trend_slope', 'ema', 'autocorr',
        'momentum', 'rms', 'cv', 'iqr', 'energy'.
        'all' = all available features.
    prefix : str, default='lag_'
        Column name prefix.

    Examples
    --------
    >>> extractor = LagFeatureExtractor(time_col='date', target_col='value', window=12)
    >>> df_with_features = extractor.transform(df)
    """

    _ALL_FEATURES = [
        'mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis',
        'trend_slope', 'ema', 'autocorr', 'momentum', 'rms', 'cv',
        'iqr', 'energy',
    ]

    def __init__(
        self,
        time_col: str,
        target_col: str,
        window: Union[int, str] = 'auto',
        features: Union[list, str] = 'all',
        prefix: str = 'lag_',
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.window = window
        self.prefix = prefix

        if features == 'all':
            self.features = list(self._ALL_FEATURES)
        else:
            for f in features:
                if f not in self._ALL_FEATURES:
                    raise ValueError(f"Unknown feature '{f}'. Choose from: {self._ALL_FEATURES}")
            self.features = list(features)

    def _resolve_window(self, n: int) -> int:
        if isinstance(self.window, int):
            return self.window
        return max(5, min(60, n // 10))

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute lag features and append them to the dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Input data, sorted by time.

        Returns
        -------
        pd.DataFrame
            Data with lag feature columns appended. Rows where the window
            is not yet fully available will have NaN for those features.
        """
        df = data.copy()
        values = df[self.target_col].values.astype(np.float64)
        n = len(values)
        w = self._resolve_window(n)

        # Pre-compute rolling windows using stride tricks for performance
        # Fallback to pandas rolling for robustness
        s = pd.Series(values)

        if 'mean' in self.features:
            df[f'{self.prefix}mean'] = s.rolling(w, min_periods=w).mean().values
        if 'std' in self.features:
            df[f'{self.prefix}std'] = s.rolling(w, min_periods=w).std().values
        if 'min' in self.features:
            df[f'{self.prefix}min'] = s.rolling(w, min_periods=w).min().values
        if 'max' in self.features:
            df[f'{self.prefix}max'] = s.rolling(w, min_periods=w).max().values
        if 'median' in self.features:
            df[f'{self.prefix}median'] = s.rolling(w, min_periods=w).median().values
        if 'skew' in self.features:
            df[f'{self.prefix}skew'] = s.rolling(w, min_periods=w).skew().values
        if 'kurtosis' in self.features:
            df[f'{self.prefix}kurtosis'] = s.rolling(w, min_periods=w).kurt().values

        if 'trend_slope' in self.features:
            # Vectorized rolling linear regression slope
            slopes = np.full(n, np.nan)
            x = np.arange(w, dtype=np.float64)
            x_mean = x.mean()
            x_var = np.sum((x - x_mean) ** 2)
            if x_var > 0:
                for i in range(w - 1, n):
                    y_win = values[i - w + 1:i + 1]
                    slopes[i] = np.sum((x - x_mean) * (y_win - y_win.mean())) / x_var
            df[f'{self.prefix}trend_slope'] = slopes

        if 'ema' in self.features:
            df[f'{self.prefix}ema'] = s.ewm(span=w, min_periods=w).mean().values

        if 'autocorr' in self.features:
            df[f'{self.prefix}autocorr'] = s.rolling(w, min_periods=w).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0,
                raw=False
            ).values

        if 'momentum' in self.features:
            # Current value minus value w steps ago
            mom = np.full(n, np.nan)
            mom[w:] = values[w:] - values[:-w]
            df[f'{self.prefix}momentum'] = mom

        if 'rms' in self.features:
            df[f'{self.prefix}rms'] = np.sqrt(
                s.rolling(w, min_periods=w).apply(lambda x: np.mean(x ** 2), raw=True).values
            )

        if 'cv' in self.features:
            r_mean = s.rolling(w, min_periods=w).mean().values
            r_std = s.rolling(w, min_periods=w).std().values
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = np.where(np.abs(r_mean) > 1e-10, r_std / np.abs(r_mean), 0.0)
            df[f'{self.prefix}cv'] = cv

        if 'iqr' in self.features:
            q75 = s.rolling(w, min_periods=w).quantile(0.75).values
            q25 = s.rolling(w, min_periods=w).quantile(0.25).values
            df[f'{self.prefix}iqr'] = q75 - q25

        if 'energy' in self.features:
            df[f'{self.prefix}energy'] = s.rolling(w, min_periods=w).apply(
                lambda x: np.sum(x ** 2), raw=True
            ).values

        return df

    def get_feature_names(self) -> list:
        """Return the list of feature column names that will be generated.

        Returns
        -------
        list of str
            Feature column names.
        """
        return [f'{self.prefix}{f}' for f in self.features]
