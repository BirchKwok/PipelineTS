"""Fourier (sin/cos) periodic features for time series data.

Generates deterministic periodic basis functions from a datetime column.
All operations are vectorized with numpy for performance.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class FourierFeatures:
    """Generate Fourier basis features from a datetime column.

    Creates sin/cos pairs for specified periods, capturing cyclical patterns
    without the overhead of one-hot encoding.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    periods : list of int or dict
        If list: periods in number of time steps (e.g. [7, 365] for weekly and yearly).
        If dict: {period_name: period_length} for readable column names.
    n_harmonics : int, default=1
        Number of sin/cos pairs per period. Higher = more flexibility.
    prefix : str, default='fourier_'
        Column name prefix.

    Examples
    --------
    >>> ff = FourierFeatures(time_col='date', periods=[7, 365], n_harmonics=2)
    >>> df_with_features = ff.transform(df)
    """

    def __init__(
        self,
        time_col: str,
        periods: Union[list, dict],
        n_harmonics: int = 1,
        prefix: str = 'fourier_',
    ):
        self.time_col = time_col
        self.n_harmonics = n_harmonics
        self.prefix = prefix

        if isinstance(periods, dict):
            self._periods = periods
        else:
            self._periods = {str(p): p for p in periods}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier features to the dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with a datetime column.

        Returns
        -------
        pd.DataFrame
            Data with added Fourier feature columns.
        """
        df = data.copy()
        ts = pd.to_datetime(df[self.time_col])

        # Convert timestamps to a numeric index (seconds since epoch for precision)
        t_numeric = (ts - ts.min()).dt.total_seconds().values.astype(np.float64)

        # Infer base period in seconds from the most common time delta
        diffs = np.diff(np.sort(t_numeric))
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            base_step = np.median(diffs)
        else:
            base_step = 1.0

        # Convert t_numeric to step units
        t_steps = t_numeric / base_step

        for pname, period in self._periods.items():
            for k in range(1, self.n_harmonics + 1):
                angle = 2.0 * np.pi * k * t_steps / period
                df[f'{self.prefix}{pname}_sin_{k}'] = np.sin(angle)
                df[f'{self.prefix}{pname}_cos_{k}'] = np.cos(angle)

        return df

    def get_feature_names(self) -> list:
        """Return the list of feature column names that will be generated.

        Returns
        -------
        list of str
            Feature column names.
        """
        names = []
        for pname in self._periods:
            for k in range(1, self.n_harmonics + 1):
                names.append(f'{self.prefix}{pname}_sin_{k}')
                names.append(f'{self.prefix}{pname}_cos_{k}')
        return names
