from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _as_columns(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def _target_columns(target_col=None, columns: Sequence[str] | str | None = None) -> list[str]:
    return _as_columns(columns) or _as_columns(target_col)


def _numeric_columns(data: pd.DataFrame, target_col=None, columns: Sequence[str] | str | None = None) -> list[str]:
    candidates = _target_columns(target_col, columns)
    return [
        c for c in candidates
        if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
    ]


def _aggregate_duplicate_times(data: pd.DataFrame, keys: list[str], agg: str) -> pd.DataFrame:
    agg = agg or "mean"
    agg_map = {}
    for col in data.columns:
        if col in keys:
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            agg_map[col] = agg if agg in {"mean", "sum", "median", "min", "max", "first", "last"} else "mean"
        else:
            agg_map[col] = "first"
    return data.groupby(keys, as_index=False).agg(agg_map)


def _fill_numeric_columns(data: pd.DataFrame, columns: list[str], method: str) -> pd.DataFrame:
    result = data.copy()
    for col in columns:
        if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
            continue
        if method == "linear":
            result[col] = result[col].interpolate(method="linear").ffill().bfill()
        elif method == "ffill":
            result[col] = result[col].ffill().bfill()
        elif method == "bfill":
            result[col] = result[col].bfill().ffill()
        elif method == "zero":
            result[col] = result[col].fillna(0)
        elif method in {"none", None}:
            continue
    return result


def sort_and_deduplicate(
    data: pd.DataFrame,
    time_col: str,
    id_col: str | None = None,
    duplicate_strategy: str = "mean",
) -> pd.DataFrame:
    result = data.copy()
    if time_col not in result.columns:
        raise ValueError(f"time_col '{time_col}' not found in data")
    result[time_col] = pd.to_datetime(result[time_col], errors="coerce")
    result = result.dropna(subset=[time_col])
    keys = [time_col]
    if id_col and id_col in result.columns:
        keys = [id_col, time_col]
    if result.duplicated(keys).any():
        result = _aggregate_duplicate_times(result, keys, duplicate_strategy)
    return result.sort_values(keys).reset_index(drop=True)


def resample_time_series(
    data: pd.DataFrame,
    time_col: str,
    freq: str | None = None,
    id_col: str | None = None,
    agg: str = "mean",
    fill_method: str = "linear",
) -> pd.DataFrame:
    result = data.copy()
    if time_col not in result.columns:
        raise ValueError(f"time_col '{time_col}' not found in data")
    result[time_col] = pd.to_datetime(result[time_col], errors="coerce")
    result = result.dropna(subset=[time_col])
    if freq is None:
        try:
            freq = pd.infer_freq(result.sort_values(time_col)[time_col].drop_duplicates())
        except Exception:
            freq = None
        if not freq:
            diffs = result.sort_values(time_col)[time_col].diff().dropna()
            if len(diffs) == 0:
                raise ValueError("Cannot infer frequency from fewer than 2 timestamps")
            freq = pd.tseries.frequencies.to_offset(diffs.mode().iloc[0])

    numeric_cols = [c for c in result.select_dtypes(include=[np.number]).columns if c != id_col]
    non_numeric_cols = [
        c for c in result.columns
        if c not in numeric_cols and c not in {time_col, id_col}
    ]
    group_iter = [(None, result)]
    if id_col and id_col in result.columns:
        group_iter = list(result.groupby(id_col, sort=False))

    frames = []
    for group_id, group in group_iter:
        g = group.sort_values(time_col).set_index(time_col)
        agg_map = {c: agg for c in numeric_cols}
        for col in non_numeric_cols:
            agg_map[col] = "first"
        resampled = g.resample(freq).agg(agg_map)
        if group_id is not None:
            resampled[id_col] = group_id
        resampled = resampled.reset_index()
        resampled = _fill_numeric_columns(resampled, numeric_cols, fill_method)
        for col in non_numeric_cols:
            if col in resampled.columns:
                resampled[col] = resampled[col].ffill().bfill()
        frames.append(resampled)

    if not frames:
        return result.reset_index(drop=True)
    result = pd.concat(frames, ignore_index=True)
    sort_keys = [time_col]
    if id_col and id_col in result.columns:
        sort_keys = [id_col, time_col]
    return result.sort_values(sort_keys).reset_index(drop=True)


def transform_target(
    data: pd.DataFrame,
    target_col: Sequence[str] | str | None,
    method: str,
    columns: Sequence[str] | str | None = None,
    suffix: str | None = None,
    replace: bool = False,
) -> pd.DataFrame:
    cols = _numeric_columns(data, target_col, columns)
    if not cols:
        raise ValueError("No numeric target columns found")
    method = (method or "").lower()
    result = data.copy()
    for col in cols:
        values = pd.to_numeric(result[col], errors="coerce").astype(float)
        if method == "log1p":
            if values.min(skipna=True) < -1:
                raise ValueError(f"Cannot apply log1p to '{col}': values below -1 exist")
            transformed = np.log1p(values)
        elif method == "sqrt":
            if values.min(skipna=True) < 0:
                raise ValueError(f"Cannot apply sqrt to '{col}': negative values exist")
            transformed = np.sqrt(values)
        elif method == "boxcox":
            from scipy import stats

            min_val = values.min(skipna=True)
            shifted = values - min_val + 1e-6 if min_val <= 0 else values
            valid = shifted.dropna()
            if len(valid) < 3:
                raise ValueError(f"Cannot apply Box-Cox to '{col}': not enough valid positive values")
            transformed_values, lam = stats.boxcox(valid)
            transformed = pd.Series(index=values.index, dtype=float)
            transformed.loc[valid.index] = transformed_values
            result.attrs[f"{col}_boxcox_lambda"] = float(lam)
        elif method == "yeojohnson":
            from scipy import stats

            valid = values.dropna()
            if len(valid) < 3:
                raise ValueError(f"Cannot apply Yeo-Johnson to '{col}': not enough valid values")
            transformed_values, lam = stats.yeojohnson(valid)
            transformed = pd.Series(index=values.index, dtype=float)
            transformed.loc[valid.index] = transformed_values
            result.attrs[f"{col}_yeojohnson_lambda"] = float(lam)
        elif method == "standardize":
            transformed = (values - values.mean(skipna=True)) / (values.std(skipna=True) + 1e-12)
        elif method == "minmax":
            transformed = (values - values.min(skipna=True)) / (values.max(skipna=True) - values.min(skipna=True) + 1e-12)
        else:
            raise ValueError("Unknown transform method. Choose from: log1p, sqrt, boxcox, yeojohnson, standardize, minmax")
        out_col = col if replace else f"{col}_{suffix or method}"
        result[out_col] = transformed
    return result


def difference_series(
    data: pd.DataFrame,
    target_col: Sequence[str] | str | None,
    order: int = 1,
    seasonal_period: int | None = None,
    columns: Sequence[str] | str | None = None,
    suffix: str | None = None,
    drop_na: bool = False,
) -> pd.DataFrame:
    cols = _numeric_columns(data, target_col, columns)
    if not cols:
        raise ValueError("No numeric target columns found")
    result = data.copy()
    order = max(1, int(order or 1))
    created = []
    for col in cols:
        out = pd.to_numeric(result[col], errors="coerce")
        for _ in range(order):
            out = out.diff()
        if seasonal_period and seasonal_period > 1:
            out = out.diff(int(seasonal_period))
        name_parts = [col, suffix or f"diff{order}"]
        if seasonal_period and seasonal_period > 1:
            name_parts.append(f"s{seasonal_period}")
        out_col = "_".join(name_parts)
        result[out_col] = out
        created.append(out_col)
    if drop_na:
        result = result.dropna(subset=created).reset_index(drop=True)
    return result


def smooth_series(
    data: pd.DataFrame,
    target_col: Sequence[str] | str | None,
    method: str = "rolling_mean",
    window: int = 7,
    columns: Sequence[str] | str | None = None,
    suffix: str | None = None,
    replace: bool = False,
) -> pd.DataFrame:
    cols = _numeric_columns(data, target_col, columns)
    if not cols:
        raise ValueError("No numeric target columns found")
    result = data.copy()
    window = max(2, int(window or 7))
    method = method or "rolling_mean"
    for col in cols:
        s = pd.to_numeric(result[col], errors="coerce")
        if method == "rolling_mean":
            out = s.rolling(window, min_periods=1).mean()
        elif method == "rolling_median":
            out = s.rolling(window, min_periods=1).median()
        elif method == "ewm":
            out = s.ewm(span=window, adjust=False, min_periods=1).mean()
        else:
            raise ValueError("Unknown smoothing method. Choose from: rolling_mean, rolling_median, ewm")
        out_col = col if replace else f"{col}_{suffix or method}_{window}"
        result[out_col] = out
    return result


def clip_or_winsorize(
    data: pd.DataFrame,
    target_col: Sequence[str] | str | None = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    columns: Sequence[str] | str | None = None,
    replace: bool = True,
    suffix: str = "winsor",
) -> pd.DataFrame:
    cols = _numeric_columns(data, target_col, columns)
    if not cols:
        raise ValueError("No numeric target columns found")
    lower_q = max(0.0, min(float(lower_q), 1.0))
    upper_q = max(0.0, min(float(upper_q), 1.0))
    if upper_q <= lower_q:
        raise ValueError("upper_q must be greater than lower_q")
    result = data.copy()
    for col in cols:
        s = pd.to_numeric(result[col], errors="coerce")
        lo, hi = s.quantile([lower_q, upper_q])
        out_col = col if replace else f"{col}_{suffix}"
        result[out_col] = s.clip(lo, hi)
    return result


class TimeSeriesPreprocessor:
    sort_and_deduplicate = staticmethod(sort_and_deduplicate)
    resample_time_series = staticmethod(resample_time_series)
    transform_target = staticmethod(transform_target)
    difference_series = staticmethod(difference_series)
    smooth_series = staticmethod(smooth_series)
    clip_or_winsorize = staticmethod(clip_or_winsorize)
