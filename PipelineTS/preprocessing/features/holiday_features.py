"""Holiday features for time series data.

Generates binary holiday indicators and distance-to-holiday features.
Uses a built-in lightweight calendar (no external dependency required),
with optional integration of the ``holidays`` library for country-specific holidays.

For **China (country='CN')**, the ``chinese-calendar`` package
(``pip install chinesecalendar``) is preferred as the authoritative source,
providing official government holiday schedules, make-up workdays (调休),
and holiday names. Falls back to the ``holidays`` library if not installed.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class HolidayFeatures:
    """Generate holiday-related features from a datetime column.

    Parameters
    ----------
    time_col : str
        Name of the datetime column.
    country : str or None, default=None
        ISO country code (e.g. 'US', 'CN', 'DE') for country-specific holidays.
        Requires the ``holidays`` package. If None, uses built-in generic holidays
        (New Year, Christmas, etc.).
    custom_holidays : list of str or pd.DatetimeIndex or None, default=None
        Additional custom holiday dates in 'YYYY-MM-DD' format.
    window : int, default=3
        Number of days before/after a holiday to compute distance features.
    prefix : str, default='holiday_'
        Column name prefix.

    Examples
    --------
    >>> hf = HolidayFeatures(time_col='date', country='US')
    >>> df_with_features = hf.transform(df)
    >>> # China with chinese-calendar (recommended):
    >>> hf = HolidayFeatures(time_col='date', country='CN')
    >>> df_with_features = hf.transform(df)  # includes is_workday, is_in_lieu
    >>> # Without external dependency (generic holidays only):
    >>> hf = HolidayFeatures(time_col='date')
    >>> df_with_features = hf.transform(df)
    """

    # Built-in generic holidays (month, day)
    _GENERIC_HOLIDAYS = [
        (1, 1),   # New Year's Day
        (2, 14),  # Valentine's Day
        (5, 1),   # Labour Day
        (10, 1),  # National Day (CN) / Unification Day
        (12, 25), # Christmas
        (12, 31), # New Year's Eve
    ]

    def __init__(
        self,
        time_col: str,
        country: Optional[str] = None,
        custom_holidays: Optional[Union[list, pd.DatetimeIndex]] = None,
        window: int = 3,
        prefix: str = 'holiday_',
    ):
        self.time_col = time_col
        self.country = country
        self.custom_holidays = custom_holidays
        self.window = window
        self.prefix = prefix

    def _get_chinese_calendar(self):
        """Try to import chinese_calendar. Returns module or None."""
        try:
            import chinese_calendar
            return chinese_calendar
        except ImportError:
            return None

    def _use_chinese_calendar(self) -> bool:
        """Whether to use chinese-calendar for CN holidays."""
        return (self.country is not None
                and self.country.upper() == 'CN'
                and self._get_chinese_calendar() is not None)

    def _get_holiday_dates(self, years: list) -> set:
        """Build the set of holiday dates for the given years."""
        holiday_dates = set()

        # China: prefer chinese-calendar for authoritative data
        if self.country is not None and self.country.upper() == 'CN':
            cc = self._get_chinese_calendar()
            if cc is not None:
                import datetime
                for year in years:
                    for month in range(1, 13):
                        for day in range(1, 32):
                            try:
                                dt = datetime.date(year, month, day)
                                if cc.is_holiday(dt):
                                    holiday_dates.add(pd.Timestamp(dt))
                            except (ValueError, NotImplementedError):
                                continue
                # Add custom holidays and return early
                if self.custom_holidays is not None:
                    for h in self.custom_holidays:
                        holiday_dates.add(pd.Timestamp(h))
                return holiday_dates
            else:
                import warnings
                warnings.warn(
                    "Package 'chinese-calendar' not installed. "
                    "Install with `pip install chinesecalendar` for accurate CN holidays. "
                    "Falling back to 'holidays' library or generic holidays.",
                    UserWarning,
                    stacklevel=2,
                )

        # Country-specific holidays via `holidays` library
        if self.country is not None:
            try:
                import holidays as holidays_lib
                country_holidays = holidays_lib.country_holidays(self.country, years=years)
                for dt in country_holidays.keys():
                    holiday_dates.add(pd.Timestamp(dt))
            except ImportError:
                import warnings
                warnings.warn(
                    f"Package 'holidays' not installed. "
                    f"Install with `pip install holidays` for country='{self.country}' support. "
                    f"Falling back to generic holidays.",
                    UserWarning,
                    stacklevel=2,
                )
                for year in years:
                    for m, d in self._GENERIC_HOLIDAYS:
                        try:
                            holiday_dates.add(pd.Timestamp(year=year, month=m, day=d))
                        except ValueError:
                            pass
        else:
            for year in years:
                for m, d in self._GENERIC_HOLIDAYS:
                    try:
                        holiday_dates.add(pd.Timestamp(year=year, month=m, day=d))
                    except ValueError:
                        pass

        # Custom holidays
        if self.custom_holidays is not None:
            for h in self.custom_holidays:
                holiday_dates.add(pd.Timestamp(h))

        return holiday_dates

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add holiday features to the dataframe.

        Generated features:
        - ``{prefix}is_holiday``: binary indicator (1 = holiday)
        - ``{prefix}days_to_nearest``: signed distance to nearest holiday
          (negative = days until next, positive = days since last)
        - ``{prefix}near_holiday``: binary indicator (within ±window days)

        Parameters
        ----------
        data : pd.DataFrame
            Input data with a datetime column.

        Returns
        -------
        pd.DataFrame
            Data with added holiday feature columns.
        """
        df = data.copy()
        ts = pd.to_datetime(df[self.time_col])

        years = sorted(ts.dt.year.unique().tolist())
        # Extend by 1 year each side to handle boundary effects
        years_extended = list(range(min(years) - 1, max(years) + 2))
        holiday_dates = self._get_holiday_dates(years_extended)

        if not holiday_dates:
            df[f'{self.prefix}is_holiday'] = 0
            df[f'{self.prefix}days_to_nearest'] = 0
            df[f'{self.prefix}near_holiday'] = 0
            return df

        # Convert to sorted numpy array of timestamps for vectorized lookup
        holiday_arr = np.array(sorted(holiday_dates), dtype='datetime64[ns]')
        ts_arr = ts.values.astype('datetime64[ns]')

        # Vectorized: is_holiday
        # Normalize both to date-level (remove time component)
        ts_dates = ts_arr.astype('datetime64[D]')
        holiday_dates_d = holiday_arr.astype('datetime64[D]')
        is_holiday = np.isin(ts_dates, holiday_dates_d).astype(np.int8)

        # Vectorized: distance to nearest holiday
        # Use searchsorted for O(n log m) instead of O(n * m)
        holiday_int = holiday_dates_d.astype(np.int64)
        ts_int = ts_dates.astype(np.int64)

        idx = np.searchsorted(holiday_int, ts_int)
        idx = np.clip(idx, 0, len(holiday_int) - 1)

        # Distance to nearest (left and right)
        dist_right = holiday_int[np.minimum(idx, len(holiday_int) - 1)] - ts_int
        dist_left = ts_int - holiday_int[np.maximum(idx - 1, 0)]

        # Pick the closer one
        days_to_nearest = np.where(
            np.abs(dist_right) <= np.abs(dist_left),
            dist_right,
            -dist_left,
        )
        # Convert from nanoseconds-int difference to days
        # datetime64[D] .astype(int64) gives days since epoch
        days_to_nearest = days_to_nearest.astype(np.float64)

        near_holiday = (np.abs(days_to_nearest) <= self.window).astype(np.int8)

        df[f'{self.prefix}is_holiday'] = is_holiday
        df[f'{self.prefix}days_to_nearest'] = days_to_nearest.astype(np.int32)
        df[f'{self.prefix}near_holiday'] = near_holiday

        # China-specific features via chinese-calendar
        if self._use_chinese_calendar():
            df = self._add_chinese_features(df, ts)

        return df

    def _add_chinese_features(self, df: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
        """Add China-specific features using chinese-calendar.

        Extra columns:
        - ``{prefix}is_workday``: 1 if official workday (including make-up days)
        - ``{prefix}is_in_lieu``: 1 if make-up workday (调休/补班)
        - ``{prefix}holiday_name``: holiday name string or '' if not a holiday
        """
        import datetime
        cc = self._get_chinese_calendar()

        dates = ts.dt.date.values
        n = len(dates)
        is_workday = np.empty(n, dtype=np.int8)
        is_in_lieu = np.empty(n, dtype=np.int8)
        holiday_names = []

        for i, d in enumerate(dates):
            dt = d if isinstance(d, datetime.date) else pd.Timestamp(d).date()
            try:
                is_workday[i] = 1 if cc.is_workday(dt) else 0
            except (ValueError, NotImplementedError):
                is_workday[i] = 1 if dt.weekday() < 5 else 0
            try:
                is_in_lieu[i] = 1 if cc.is_in_lieu(dt) else 0
            except (ValueError, NotImplementedError):
                is_in_lieu[i] = 0
            try:
                on_holiday, name = cc.get_holiday_detail(dt)
                if on_holiday and name is not None:
                    # Handle both enum (.value) and plain string returns
                    holiday_names.append(name.value if hasattr(name, 'value') else str(name))
                else:
                    holiday_names.append('')
            except (ValueError, NotImplementedError, AttributeError):
                holiday_names.append('')

        df[f'{self.prefix}is_workday'] = is_workday
        df[f'{self.prefix}is_in_lieu'] = is_in_lieu
        df[f'{self.prefix}holiday_name'] = holiday_names

        return df

    def get_feature_names(self) -> list:
        """Return the list of feature column names that will be generated.

        Returns
        -------
        list of str
            Feature column names.
        """
        names = [
            f'{self.prefix}is_holiday',
            f'{self.prefix}days_to_nearest',
            f'{self.prefix}near_holiday',
        ]
        if self._use_chinese_calendar():
            names.extend([
                f'{self.prefix}is_workday',
                f'{self.prefix}is_in_lieu',
                f'{self.prefix}holiday_name',
            ])
        return names
