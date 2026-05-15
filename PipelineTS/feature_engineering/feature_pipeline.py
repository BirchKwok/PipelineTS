"""Unified feature engineering pipeline for time series data.

Composes calendar features, Fourier features, holiday features, and lag
features into a single configurable transformer. All sub-components are
optional and lazily instantiated.
"""

import pandas as pd
from typing import Optional, Union

from PipelineTS.feature_engineering.calendar_features import DateExtendFeatures
from .fourier_features import FourierFeatures
from .holiday_features import HolidayFeatures
from .lag_features import LagFeatureExtractor


class TimeSeriesFeatureEngineer:
    """One-stop feature engineering for time series data.

    Combines multiple feature extractors into a single ``transform()`` call.
    Each component can be enabled/disabled independently.

    Parameters
    ----------
    time_col : str
        Datetime column name.
    target_col : str or None, default=None
        Target column name. Required for lag features.
    use_calendar : bool, default=True
        Add calendar/date features (hour, weekday, month, season, etc.).
    use_fourier : bool, default=False
        Add Fourier periodic features.
    fourier_periods : list or dict or None, default=None
        Periods for Fourier features. Required if ``use_fourier=True``.
    fourier_harmonics : int, default=1
        Number of harmonics per Fourier period.
    use_holidays : bool, default=False
        Add holiday features.
    holiday_country : str or None, default=None
        Country code for holiday detection.
    custom_holidays : list or None, default=None
        Custom holiday dates.
    use_lags : bool, default=False
        Add rolling lag features.
    lag_window : int or 'auto', default='auto'
        Window size for lag features.
    lag_features : list or 'all', default='all'
        Which lag features to extract.
    drop_time_col : bool, default=False
        Whether to drop the time column from the output.

    Examples
    --------
    >>> engineer = TimeSeriesFeatureEngineer(
    ...     time_col='date',
    ...     target_col='value',
    ...     use_calendar=True,
    ...     use_fourier=True,
    ...     fourier_periods=[7, 365],
    ...     use_holidays=True,
    ...     holiday_country='US',
    ...     use_lags=True,
    ...     lag_window=12,
    ... )
    >>> df_enriched = engineer.transform(df)
    """

    def __init__(
        self,
        time_col: str,
        target_col: Optional[str] = None,
        use_calendar: bool = True,
        use_fourier: bool = False,
        fourier_periods: Optional[Union[list, dict]] = None,
        fourier_harmonics: int = 1,
        use_holidays: bool = False,
        holiday_country: Optional[str] = None,
        custom_holidays: Optional[list] = None,
        use_lags: bool = False,
        lag_window: Union[int, str] = 'auto',
        lag_features: Union[list, str] = 'all',
        drop_time_col: bool = False,
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.drop_time_col = drop_time_col

        self._components = []

        if use_calendar:
            self._calendar = DateExtendFeatures(
                date_col=time_col,
                drop_date_col=False,
                use_scale=False,
            )
            self._components.append('calendar')
        else:
            self._calendar = None

        if use_fourier:
            if fourier_periods is None:
                raise ValueError("fourier_periods must be provided when use_fourier=True.")
            self._fourier = FourierFeatures(
                time_col=time_col,
                periods=fourier_periods,
                n_harmonics=fourier_harmonics,
            )
            self._components.append('fourier')
        else:
            self._fourier = None

        if use_holidays:
            self._holidays = HolidayFeatures(
                time_col=time_col,
                country=holiday_country,
                custom_holidays=custom_holidays,
            )
            self._components.append('holidays')
        else:
            self._holidays = None

        if use_lags:
            if target_col is None:
                raise ValueError("target_col must be provided when use_lags=True.")
            self._lags = LagFeatureExtractor(
                time_col=time_col,
                target_col=target_col,
                window=lag_window,
                features=lag_features,
            )
            self._components.append('lags')
        else:
            self._lags = None

        self._is_fitted = False

    def fit(self, data: pd.DataFrame) -> 'TimeSeriesFeatureEngineer':
        """Fit the feature engineer (only needed for calendar scaler).

        Parameters
        ----------
        data : pd.DataFrame
            Training data.

        Returns
        -------
        self
        """
        if self._calendar is not None:
            self._calendar.fit(data)
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled feature extractors.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with at least the time column.

        Returns
        -------
        pd.DataFrame
            Data with all generated features appended.
        """
        if not self._is_fitted:
            self.fit(data)

        df = data.copy()

        if self._calendar is not None:
            df = self._calendar.transform(df)

        if self._fourier is not None:
            df = self._fourier.transform(df)

        if self._holidays is not None:
            df = self._holidays.transform(df)

        if self._lags is not None:
            df = self._lags.transform(df)

        if self.drop_time_col and self.time_col in df.columns:
            df = df.drop(columns=[self.time_col])

        return df

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            Transformed data.
        """
        return self.fit(data).transform(data)

    def get_feature_names(self) -> list:
        """Return all feature column names that will be generated.

        Returns
        -------
        list of str
        """
        names = []
        if self._fourier is not None:
            names.extend(self._fourier.get_feature_names())
        if self._holidays is not None:
            names.extend(self._holidays.get_feature_names())
        if self._lags is not None:
            names.extend(self._lags.get_feature_names())
        return names

    def __repr__(self) -> str:
        return (f"TimeSeriesFeatureEngineer(time_col='{self.time_col}', "
                f"components={self._components})")
