from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from PipelineTS.preprocessing.time_series_preprocessing import (
    clip_or_winsorize,
    resample_time_series,
    sort_and_deduplicate,
)

_TIME_NAME_CANDIDATES = {
    "ds", "date", "datetime", "timestamp", "time", "dt", "day", "month",
    "week", "year", "period",
}
_TARGET_NAME_CANDIDATES = {
    "y", "value", "target", "sales", "demand", "count", "amount", "qty",
    "quantity", "volume", "traffic", "load", "revenue", "price",
}
_ID_NAME_CANDIDATES = {
    "id", "series_id", "item_id", "unique_id", "store_id", "sku", "sku_id",
    "group", "group_id", "entity", "entity_id",
}


def load_data(data: pd.DataFrame | str | Path, **read_kwargs) -> pd.DataFrame:
    """Load time series data from a file path or return a DataFrame as-is.

    Supports CSV (.csv), TSV (.tsv), Excel (.xlsx/.xls), Parquet (.parquet),
    and JSON (.json) files. Files without an extension are treated as CSV.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        A pandas DataFrame, or a path to a local data file.
    **read_kwargs
        Extra keyword arguments forwarded to the underlying pandas reader
        (e.g. ``sep``, ``encoding``, ``sheet_name``).

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Raises
    ------
    TypeError
        If *data* is not a DataFrame or a string/Path.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.

    Examples
    --------
    >>> df = load_data("sales.csv")
    >>> df = load_data(Path("/data/electric.parquet"))
    >>> df = load_data(existing_df)   # returns as-is
    """
    if isinstance(data, pd.DataFrame):
        return data
    if not isinstance(data, (str, Path)):
        raise TypeError("data must be a pandas DataFrame or a local data file path")

    path = Path(data).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Data path must be a file: {path}")

    suffix = path.suffix.lower()
    if suffix in {"", ".csv"}:
        return pd.read_csv(path, **read_kwargs)
    if suffix == ".tsv":
        kwargs = {"sep": "\t"}
        kwargs.update(read_kwargs)
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **read_kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **read_kwargs)
    if suffix == ".json":
        return pd.read_json(path, **read_kwargs)
    raise ValueError("Unsupported data file type. Use csv, tsv, xlsx, xls, parquet, or json.")


def _normalise_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _validate_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    data = load_data(data)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must not be empty")
    return data


def infer_time_col(data: pd.DataFrame, time_col: str | None = None) -> str:
    """Infer or validate the time column in a DataFrame.

    If *time_col* is given it is validated to exist in the data and returned.
    Otherwise the column is discovered automatically by checking:

    1. Column names that match known time-like names (``date``, ``ds``,
       ``timestamp``, ``datetime``, etc.).
    2. Columns with a ``datetime64`` dtype.
    3. Columns whose values can be parsed as dates with ≥ 80 % success rate.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    time_col : str or None, default None
        Name of the time column.  ``None`` triggers auto-inference.

    Returns
    -------
    str
        Name of the resolved time column.

    Raises
    ------
    ValueError
        If *time_col* is given but not found in the data, or if no time
        column can be inferred automatically.

    Examples
    --------
    >>> infer_time_col(df)                    # auto-detect
    'date'
    >>> infer_time_col(df, time_col='ts')     # explicit
    'ts'
    """
    data = _validate_dataframe(data)
    if time_col is not None:
        if time_col not in data.columns:
            raise ValueError(f"time_col '{time_col}' not found in data")
        return time_col

    exact = [c for c in data.columns if _normalise_name(c) in _TIME_NAME_CANDIDATES]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        warnings.warn(
            f"Multiple time-like columns found {exact}; using '{exact[0]}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return exact[0]

    datetime_cols = [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c])]
    if len(datetime_cols) == 1:
        return datetime_cols[0]
    if len(datetime_cols) > 1:
        warnings.warn(
            f"Multiple datetime columns found {datetime_cols}; using '{datetime_cols[0]}'.",
            RuntimeWarning,
            stacklevel=2,
        )
        return datetime_cols[0]

    parseable = []
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            continue
        sample = data[col].dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        score = float(parsed.notna().mean())
        if score >= 0.8:
            parseable.append((col, score))

    if parseable:
        parseable.sort(key=lambda x: x[1], reverse=True)
        return parseable[0][0]

    raise ValueError("Could not infer time_col. Pass time_col explicitly.")


def infer_id_col(data: pd.DataFrame, id_col: str | None = None) -> str | None:
    """Infer or validate the series-identifier column in a DataFrame.

    Used for panel (multi-series) data. If *id_col* is ``None``, returns
    ``None`` (single-series mode). If *id_col* is ``'auto'``, the function
    searches for a column whose name matches known ID-like names (``id``,
    ``series_id``, ``store_id``, ``sku``, etc.) and whose cardinality is
    between 2 and the total number of rows.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    id_col : str or None, default None
        Name of the series-ID column, ``'auto'`` for automatic discovery,
        or ``None`` to disable multi-series mode.

    Returns
    -------
    str or None
        Resolved column name, or ``None`` if not applicable.

    Raises
    ------
    ValueError
        If an explicit *id_col* is not found in the data.

    Examples
    --------
    >>> infer_id_col(df, id_col='store_id')   # explicit
    'store_id'
    >>> infer_id_col(df, id_col='auto')       # auto-detect
    'series_id'  # or None if nothing found
    """
    data = _validate_dataframe(data)
    if id_col is None:
        return None
    if id_col != "auto":
        if id_col not in data.columns:
            raise ValueError(f"id_col '{id_col}' not found in data")
        return id_col

    for col in data.columns:
        if _normalise_name(col) not in _ID_NAME_CANDIDATES:
            continue
        n_unique = data[col].nunique(dropna=True)
        if 1 < n_unique < len(data):
            return col
    return None


def infer_target_col(
    data: pd.DataFrame,
    target_col: str | None = None,
    time_col: str | None = None,
    id_col: str | None = None,
    exclude: Sequence[str] | str | None = None,
) -> str:
    """Infer or validate the forecast target column in a DataFrame.

    If *target_col* is given it is validated and returned. Otherwise the
    column is discovered automatically by:

    1. Selecting numeric columns that are not *time_col*, *id_col*, or any
       column listed in *exclude*.
    2. Preferring columns whose names match known target-like names (``y``,
       ``value``, ``sales``, ``demand``, ``revenue``, etc.).
    3. Falling back to the last numeric column when multiple candidates exist.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    target_col : str or None, default None
        Name of the target column. ``None`` triggers auto-inference.
    time_col : str or None, default None
        Name of the time column to exclude from candidate search.
    id_col : str or None, default None
        Name of the series-ID column to exclude from candidate search.
    exclude : sequence of str or str or None, default None
        Additional column names to exclude (e.g. covariate columns).

    Returns
    -------
    str
        Name of the resolved target column.

    Raises
    ------
    ValueError
        If *target_col* is given but not found, or if no numeric column
        can be identified as the target.

    Examples
    --------
    >>> infer_target_col(df, time_col='date')
    'value'
    >>> infer_target_col(df, target_col='sales', time_col='date')
    'sales'
    """
    data = _validate_dataframe(data)
    if target_col is not None:
        if target_col not in data.columns:
            raise ValueError(f"target_col '{target_col}' not found in data")
        return target_col

    excluded = set(_as_list(exclude))
    if time_col is not None:
        excluded.add(time_col)
    if id_col is not None:
        excluded.add(id_col)

    numeric_cols = [
        c for c in data.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(data[c])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    preferred = [c for c in numeric_cols if _normalise_name(c) in _TARGET_NAME_CANDIDATES]
    if preferred:
        if len(preferred) > 1:
            warnings.warn(
                f"Multiple target-like columns found {preferred}; using '{preferred[0]}'.",
                RuntimeWarning,
                stacklevel=2,
            )
        return preferred[0]

    if numeric_cols:
        warnings.warn(
            f"Multiple numeric columns found {numeric_cols}; using '{numeric_cols[-1]}' as target_col. "
            "Pass target_col explicitly to override.",
            RuntimeWarning,
            stacklevel=2,
        )
        return numeric_cols[-1]

    raise ValueError("Could not infer target_col. Pass target_col explicitly.")


def _fill_numeric_missing(
    data: pd.DataFrame,
    target_col: str,
    id_col: str | None,
    method: str,
) -> pd.DataFrame:
    if method in {None, "none"}:
        return data
    result = data.copy()
    numeric_cols = [
        c for c in result.select_dtypes(include=[np.number]).columns
        if c != id_col
    ]
    if target_col not in numeric_cols and target_col in result.columns:
        numeric_cols.append(target_col)

    groups = [(None, result.index)]
    if id_col is not None and id_col in result.columns:
        groups = list(result.groupby(id_col, sort=False).groups.items())

    for _, idx in groups:
        for col in numeric_cols:
            if col not in result.columns:
                continue
            values = pd.to_numeric(result.loc[idx, col], errors="coerce")
            if method == "linear":
                values = values.interpolate(method="linear").ffill().bfill()
            elif method == "ffill":
                values = values.ffill().bfill()
            elif method == "bfill":
                values = values.bfill().ffill()
            elif method == "zero":
                values = values.fillna(0)
            else:
                raise ValueError("fill_method must be one of: linear, ffill, bfill, zero, none")
            result.loc[idx, col] = values
    return result


def _should_clip_target(data: pd.DataFrame, target_col: str) -> bool:
    if target_col not in data.columns or len(data) < 20:
        return False
    values = pd.to_numeric(data[target_col], errors="coerce").dropna()
    if len(values) < 20:
        return False
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return False
    lo = q1 - 3.0 * iqr
    hi = q3 + 3.0 * iqr
    return bool(((values < lo) | (values > hi)).mean() > 0)


def _infer_freq(data: pd.DataFrame, time_col: str, id_col: str | None = None) -> str | None:
    if time_col not in data.columns:
        return None
    frame = data
    if id_col is not None and id_col in data.columns:
        sizes = data.groupby(id_col).size()
        if len(sizes):
            frame = data[data[id_col] == sizes.sort_values(ascending=False).index[0]]
    ts = pd.to_datetime(frame[time_col], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(ts) < 3:
        return None
    try:
        return pd.infer_freq(ts)
    except Exception:
        return None


def _status_from_report(report: str) -> str:
    for line in str(report).splitlines():
        text = line.strip()
        if text.startswith("Status:"):
            return text.split(":", 1)[1].strip()
    return "UNKNOWN"


def _safe_report(func: Callable, *args, **kwargs) -> str:
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        return f"{getattr(func, '__name__', 'report')} failed: {exc}"


def _resolve_metric(metric: str | Callable) -> tuple[Callable, str]:
    import PipelineTS.metrics as _metrics
    return _metrics.resolve_metric(metric)


def preprocess(
    data: pd.DataFrame | str | Path,
    time_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    freq: str | None | bool = None,
    fill_missing: bool = True,
    fill_method: str = "linear",
    deduplicate: bool = True,
    clip_outliers: bool | str = "auto",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and standardize a time series DataFrame for forecasting.

    Applies a sequence of safe, non-destructive preprocessing steps:

    1. **Column inference** — resolves *time_col*, *target_col*, *id_col*.
    2. **Sort & deduplicate** — sorts by time (and series ID) and removes
       duplicate timestamps per series.
    3. **Frequency resampling** (optional) — resamples to a regular grid at
       *freq*, filling gaps in the process.
    4. **Missing value filling** — fills NaN values in numeric columns using
       *fill_method* (when no resampling is performed).
    5. **Outlier clipping** — clips extreme values in *target_col* to the
       [*lower_q*, *upper_q*] quantile range when strong outliers are detected.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Input data or path to a CSV/Excel/Parquet/JSON file.
    time_col : str or None, default None
        Time column name. Auto-inferred when ``None``.
    target_col : str or None, default None
        Target column name. Auto-inferred when ``None``.
    id_col : str or None, default None
        Series-ID column for panel data. ``None`` = single-series mode.
    freq : str, bool, or None, default None
        Target frequency for resampling. Pass a pandas offset string (e.g.
        ``'D'``, ``'MS'``, ``'h'``), ``True`` / ``'auto'`` for auto-detection,
        or ``None`` / ``False`` to skip resampling.
    fill_missing : bool, default True
        Whether to fill NaN values in numeric columns.
    fill_method : {'linear', 'ffill', 'bfill', 'zero'}, default 'linear'
        Interpolation method for filling missing values.
    deduplicate : bool, default True
        Whether to sort and remove duplicate timestamps.
    clip_outliers : bool or 'auto', default 'auto'
        ``True`` always clips; ``False`` never clips; ``'auto'`` clips only
        when IQR-based outliers are detected in the target column.
    lower_q : float, default 0.01
        Lower quantile for outlier clipping.
    upper_q : float, default 0.99
        Upper quantile for outlier clipping.
    return_info : bool, default False
        When ``True``, return a ``(cleaned_df, info_dict)`` tuple instead of
        just the DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. Or ``(pd.DataFrame, dict)`` when *return_info* is
        ``True``.  The *info* dict contains keys: ``time_col``,
        ``target_col``, ``id_col``, ``rows``, ``freq``,
        ``filled_missing``, ``clipped_outliers``.

    Examples
    --------
    >>> clean = preprocess("sales.csv")
    >>> clean, info = preprocess(df, time_col='date', return_info=True)
    >>> clean = preprocess(df, freq='D', clip_outliers=True)
    """
    data = _validate_dataframe(data)
    resolved_time_col = infer_time_col(data, time_col)
    resolved_id_col = infer_id_col(data, id_col)
    resolved_target_col = infer_target_col(
        data, target_col, time_col=resolved_time_col, id_col=resolved_id_col
    )

    result = data.copy()
    if deduplicate:
        result = sort_and_deduplicate(
            result, time_col=resolved_time_col, id_col=resolved_id_col
        )
    else:
        result[resolved_time_col] = pd.to_datetime(result[resolved_time_col], errors="coerce")
        result = result.dropna(subset=[resolved_time_col])
        sort_keys = [resolved_time_col]
        if resolved_id_col is not None and resolved_id_col in result.columns:
            sort_keys = [resolved_id_col, resolved_time_col]
        result = result.sort_values(sort_keys).reset_index(drop=True)

    if freq not in {None, False}:
        result = resample_time_series(
            result,
            time_col=resolved_time_col,
            freq=None if freq in {True, "auto"} else str(freq),
            id_col=resolved_id_col,
            fill_method=fill_method if fill_missing else "none",
        )
    elif fill_missing:
        result = _fill_numeric_missing(
            result,
            target_col=resolved_target_col,
            id_col=resolved_id_col,
            method=fill_method,
        )

    should_clip = clip_outliers is True or (
        clip_outliers == "auto" and _should_clip_target(result, resolved_target_col)
    )
    if should_clip:
        result = clip_or_winsorize(
            result,
            target_col=resolved_target_col,
            lower_q=lower_q,
            upper_q=upper_q,
            replace=True,
        )

    info = {
        "time_col": resolved_time_col,
        "target_col": resolved_target_col,
        "id_col": resolved_id_col,
        "rows": len(result),
        "freq": None if freq in {None, False} else ("auto" if freq is True else freq),
        "filled_missing": bool(fill_missing),
        "clipped_outliers": bool(should_clip),
    }
    if return_info:
        return result, info
    return result


def diagnose(
    data: pd.DataFrame | str | Path,
    time_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    horizon: int | None = None,
    known_covariates: Sequence[str] | str | None = None,
    past_covariates: Sequence[str] | str | None = None,
    full: bool = False,
    **read_kwargs,
) -> dict[str, Any]:
    """Run a comprehensive readiness diagnostic on a time series dataset.

    Automatically infers column names, applies safe preprocessing, and
    generates several diagnostic reports to answer: *Is this data ready for
    forecasting?*  Use the returned ``status`` and ``next_step`` fields to
    guide the next action.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Input data or path to a file (CSV, Excel, Parquet, JSON).
    time_col : str or None, default None
        Time column name. Auto-inferred when ``None``.
    target_col : str or None, default None
        Target column name. Auto-inferred when ``None``.
    id_col : str or None, default None
        Series-ID column for panel data.
    horizon : int or None, default None
        Forecast horizon used in forecastability and baseline reports.
        Auto-determined as ``min(24, len(data) // 10)`` when ``None``.
    known_covariates : sequence of str or str or None, default None
        Covariate columns whose future values are known at prediction time.
        Excluded from target-column inference.
    past_covariates : sequence of str or str or None, default None
        Covariate columns that are only available historically.
    full : bool, default False
        When ``True``, include extended reports: ``time_index``,
        ``series_profile``, ``seasonality``, ``trend``, ``leakage_risk``.
    **read_kwargs
        Extra keyword arguments forwarded to the file reader.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``status`` : ``'READY'`` / ``'WARNING'`` / ``'NOT_READY'``
        - ``rows`` : int — raw row count.
        - ``clean_rows`` : int — row count after preprocessing.
        - ``time_col`` : str — resolved time column.
        - ``target_col`` : str — resolved target column.
        - ``id_col`` : str or None — resolved ID column.
        - ``candidate_id_col`` : str or None — auto-detected ID column.
        - ``freq`` : str or None — detected frequency (e.g. ``'MS'``).
        - ``horizon`` : int — effective forecast horizon used.
        - ``preprocess`` : dict — summary of preprocessing actions.
        - ``reports`` : dict — sub-reports (``readiness``,
          ``forecastability``, ``baseline``, ``recommendations``,
          optionally ``panel`` and full-mode reports).
        - ``next_step`` : str — copy-pasteable ``forecast()`` call.

    Examples
    --------
    >>> result = diagnose("sales.csv", horizon=12)
    >>> print(result["status"])        # 'READY'
    >>> print(result["next_step"])     # forecast(data, n=12, ...)
    >>> print(result["reports"]["forecastability"])
    """
    data = _validate_dataframe(load_data(data, **read_kwargs))
    resolved_time_col = infer_time_col(data, time_col)
    resolved_id_col = infer_id_col(data, id_col)
    candidate_id_col = infer_id_col(data, "auto")
    resolved_target_col = infer_target_col(
        data,
        target_col,
        time_col=resolved_time_col,
        id_col=resolved_id_col,
        exclude=_as_list(known_covariates) + _as_list(past_covariates),
    )
    cleaned, prep_info = preprocess(
        data,
        time_col=resolved_time_col,
        target_col=resolved_target_col,
        id_col=resolved_id_col,
        return_info=True,
    )

    from PipelineTS.preprocessing import time_series_diagnostics as tsdiag

    horizon_i = int(horizon or max(1, min(24, max(1, len(cleaned) // 10))))
    reports = {
        "readiness": _safe_report(
            tsdiag.modeling_readiness_report,
            data,
            resolved_time_col,
            resolved_target_col,
            id_col=resolved_id_col,
            horizon=horizon_i,
            known_covariates=_as_list(known_covariates),
            past_covariates=_as_list(past_covariates),
        ),
        "forecastability": _safe_report(
            tsdiag.forecastability_report,
            cleaned,
            resolved_target_col,
            horizon=horizon_i,
        ),
        "baseline": _safe_report(
            tsdiag.baseline_forecast_report,
            cleaned,
            resolved_time_col,
            resolved_target_col,
            horizon=horizon_i,
        ),
        "recommendations": _safe_report(
            tsdiag.recommendation_report,
            cleaned,
            resolved_time_col,
            resolved_target_col,
            id_col=resolved_id_col,
        ),
    }
    if resolved_id_col is not None:
        reports["panel"] = _safe_report(
            tsdiag.panel_structure_report,
            cleaned,
            resolved_time_col,
            resolved_target_col,
            id_col=resolved_id_col,
        )
    if full:
        reports.update({
            "time_index": _safe_report(tsdiag.time_index_report, cleaned, resolved_time_col),
            "series_profile": _safe_report(tsdiag.series_profile, cleaned, resolved_target_col),
            "seasonality": _safe_report(tsdiag.seasonality_report, cleaned, resolved_target_col),
            "trend": _safe_report(tsdiag.trend_report, cleaned, resolved_time_col, resolved_target_col),
            "leakage_risk": _safe_report(
                tsdiag.leakage_risk_report,
                cleaned,
                resolved_time_col,
                resolved_target_col,
                id_col=resolved_id_col,
            ),
        })

    return {
        "status": _status_from_report(reports["readiness"]),
        "rows": int(len(data)),
        "clean_rows": int(len(cleaned)),
        "time_col": resolved_time_col,
        "target_col": resolved_target_col,
        "id_col": resolved_id_col,
        "candidate_id_col": candidate_id_col,
        "freq": _infer_freq(cleaned, resolved_time_col, resolved_id_col),
        "horizon": horizon_i,
        "preprocess": prep_info,
        "reports": reports,
        "next_step": f"forecast(data, n={horizon_i}, time_col='{resolved_time_col}', target_col='{resolved_target_col}')",
    }


class AutoForecast:
    """Scikit-learn–style AutoML forecaster with progressive professional control.

    Wraps :class:`~PipelineTS.pipeline.SmartRouter` with automatic column
    inference and safe preprocessing so you get a production-ready forecaster
    in as few as three lines::

        model = AutoForecast(horizon=12)
        model.fit(data)
        pred = model.predict()

    The class exposes the full SmartRouter API through attribute delegation, so
    ``model.leader_board_``, ``model.strategy_``, ``model.profile_`` etc.
    all work as expected after :meth:`fit`.

    Parameters
    ----------
    time_col : str or None, default None
        Time column name. Auto-inferred when ``None``.
    target_col : str or None, default None
        Target column name. Auto-inferred when ``None``.
    horizon : int or None, default None
        Number of future steps to forecast.  Alias: *n_predict*.
    n_predict : int or None, default None
        Alias for *horizon*. Takes precedence if both are supplied.
    quantile : float or None, default None
        Coverage level for prediction intervals (e.g. ``0.9`` → 90 %).
        ``None`` produces point-only forecasts.
    preset : {'fast', 'medium_quality', 'high_quality', 'best_quality'}, default 'fast'
        Quality/speed preset forwarded to SmartRouter.

        - ``'fast'``: 3 models, basic search, no ensemble (~seconds).
        - ``'medium_quality'``: 5 models, auto search, auto ensemble.
        - ``'high_quality'``: 8 models, thorough search, weighted ensemble.
        - ``'best_quality'``: 15 models, thorough search, top-5 ensemble.

    time_limit : int, float, or None, default None
        Total training time budget in seconds. ``None`` = unlimited.
    id_col : str or None, default None
        Series-ID column for panel (multi-series) data.
    known_covariates : sequence of str or str or None, default None
        Covariate columns whose future values will be known at prediction time.
    past_covariates : sequence of str or str or None, default None
        Covariate columns that are only available historically.
    preprocess_data : bool or dict, default True
        When ``True``, runs :func:`preprocess` with default settings before
        fitting.  Pass a ``dict`` to override specific :func:`preprocess`
        keyword arguments.  ``False`` skips preprocessing.
    verbose : bool, default False
        Whether to print SmartRouter routing decisions during fit.
    **router_kwargs
        Additional keyword arguments forwarded directly to
        :class:`~PipelineTS.pipeline.SmartRouter`.

    Attributes
    ----------
    router_ : SmartRouter or None
        The fitted SmartRouter instance. Available after :meth:`fit`.
    inferred_columns_ : dict or None
        Resolved ``time_col``, ``target_col``, ``id_col`` after fit.
    training_data_ : pd.DataFrame or None
        The preprocessed training DataFrame used in the last :meth:`fit` call.
    leader_board_ : pd.DataFrame or None
        Model ranking table (delegated from ``router_``).
    strategy_ : dict or None
        SmartRouter routing strategy summary (delegated from ``router_``).
    best_model_ : object or None
        The best fitted model object (delegated from ``router_``).

    Examples
    --------
    >>> from PipelineTS import AutoForecast
    >>> model = AutoForecast(horizon=12, preset='medium_quality', quantile=0.9)
    >>> model.fit(data)
    >>> pred = model.predict()
    >>> model.save('forecaster.pts')
    >>> loaded = AutoForecast.load('forecaster.pts')
    """

    def __init__(
        self,
        time_col: str | None = None,
        target_col: str | None = None,
        horizon: int | None = None,
        n_predict: int | None = None,
        quantile: float | None = None,
        preset: str = "fast",
        time_limit: int | float | None = None,
        id_col: str | None = None,
        known_covariates: Sequence[str] | str | None = None,
        past_covariates: Sequence[str] | str | None = None,
        preprocess_data: bool | dict[str, Any] = True,
        verbose: bool = False,
        metric: str | Callable = "business",
        **router_kwargs,
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.horizon = n_predict if n_predict is not None else horizon
        self.quantile = quantile
        self.preset = preset
        self.time_limit = time_limit
        self.id_col = id_col
        self.known_covariates = _as_list(known_covariates)
        self.past_covariates = _as_list(past_covariates)
        self.preprocess_data = preprocess_data
        self.verbose = verbose
        self.metric = metric
        self.router_kwargs = dict(router_kwargs)
        self.router_ = None
        self.inferred_columns_ = None
        self.training_data_ = None

    def _resolve_columns(self, data: pd.DataFrame) -> tuple[str, str, str | None]:
        resolved_time_col = infer_time_col(data, self.time_col)
        resolved_id_col = infer_id_col(data, self.id_col)
        resolved_target_col = infer_target_col(
            data,
            self.target_col,
            time_col=resolved_time_col,
            id_col=resolved_id_col,
            exclude=self.known_covariates + self.past_covariates,
        )
        return resolved_time_col, resolved_target_col, resolved_id_col

    def _prepare(self, data: pd.DataFrame, time_col: str, target_col: str, id_col: str | None) -> pd.DataFrame:
        if not self.preprocess_data:
            result = data.copy()
            result[time_col] = pd.to_datetime(result[time_col], errors="coerce")
            return result.dropna(subset=[time_col]).reset_index(drop=True)

        kwargs = {}
        if isinstance(self.preprocess_data, dict):
            kwargs.update(self.preprocess_data)
        return preprocess(
            data,
            time_col=time_col,
            target_col=target_col,
            id_col=id_col,
            **kwargs,
        )

    def fit(self, data: pd.DataFrame | str | Path, valid_data: pd.DataFrame | None = None):
        """Fit the AutoML forecaster on training data.

        Resolves column names, applies preprocessing, then trains a
        :class:`~PipelineTS.pipeline.SmartRouter` on the cleaned data.

        Parameters
        ----------
        data : pd.DataFrame or str or Path
            Training data or path to a supported file (CSV, Excel, Parquet,
            JSON).
        valid_data : pd.DataFrame or None, default None
            Optional held-out validation set.  When provided, model evaluation
            uses this instead of an auto-split from the training data.

        Returns
        -------
        self
            Returns the fitted ``AutoForecast`` instance for method chaining.

        Examples
        --------
        >>> model = AutoForecast(horizon=12)
        >>> model.fit("sales.csv")
        >>> model.fit(train_df, valid_data=valid_df)
        """
        from PipelineTS.pipeline.smart_router import SmartRouter

        data = _validate_dataframe(data)
        time_col, target_col, id_col = self._resolve_columns(data)
        train_data = self._prepare(data, time_col, target_col, id_col)
        prepared_valid = None
        if valid_data is not None:
            prepared_valid = self._prepare(valid_data, time_col, target_col, id_col)

        router = SmartRouter(
            time_col=time_col,
            target_col=target_col,
            n_predict=self.horizon,
            quantile=self.quantile,
            preset=self.preset,
            time_limit=self.time_limit,
            id_col=id_col,
            known_covariates=self.known_covariates,
            past_covariates=self.past_covariates,
            verbose=self.verbose,
            metric=self.metric,
            **self.router_kwargs,
        )
        router.fit(train_data, valid_data=prepared_valid)
        self.router_ = router
        self.inferred_columns_ = {
            "time_col": time_col,
            "target_col": target_col,
            "id_col": id_col,
        }
        self.training_data_ = train_data
        return self

    def predict(
        self,
        n: int | None = None,
        data: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **predict_kwargs,
    ) -> pd.DataFrame:
        """Generate forecasts from the fitted model.

        Parameters
        ----------
        n : int or None, default None
            Number of future steps to forecast.  Falls back to the *horizon*
            set at construction when ``None``.
        data : pd.DataFrame or None, default None
            Optional new context data (must share the same columns as the
            training data).  When provided it replaces the stored training
            data as the autoregressive context window.
        future_covariates : pd.DataFrame or None, default None
            Future known covariate values for the forecast horizon.  Must have
            at least *n* rows and columns matching *known_covariates*.
        **predict_kwargs
            Extra keyword arguments forwarded to
            :meth:`~PipelineTS.pipeline.SmartRouter.predict`.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with columns ``[time_col, target_col]`` and,
            when quantile prediction was enabled, also
            ``[target_col_lower, target_col_upper]``.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.

        Examples
        --------
        >>> pred = model.predict()               # next *horizon* steps
        >>> pred = model.predict(n=24)           # override horizon
        >>> pred = model.predict(data=new_ctx)   # fresh context window
        """
        if self.router_ is None:
            raise ValueError("AutoForecast has not been fitted yet. Call fit() first.")
        prepared_data = data
        if data is not None and self.preprocess_data:
            cols = self.inferred_columns_ or {}
            prepared_data = self._prepare(
                data,
                cols.get("time_col", self.time_col),
                cols.get("target_col", self.target_col),
                cols.get("id_col", self.id_col),
            )
        horizon = n if n is not None else self.horizon
        return self.router_.predict(
            n=horizon,
            data=prepared_data,
            future_covariates=future_covariates,
            **predict_kwargs,
        )

    def fit_predict(
        self,
        data: pd.DataFrame | str | Path,
        n: int | None = None,
        valid_data: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        **predict_kwargs,
    ) -> pd.DataFrame:
        """Fit and immediately generate forecasts in a single call.

        Equivalent to calling :meth:`fit` followed by :meth:`predict`.

        Parameters
        ----------
        data : pd.DataFrame or str or Path
            Training data or path to a supported file.
        n : int or None, default None
            Number of future steps to forecast. Falls back to *horizon*.
        valid_data : pd.DataFrame or None, default None
            Optional held-out validation data.
        future_covariates : pd.DataFrame or None, default None
            Future known covariate values for the forecast horizon.
        **predict_kwargs
            Extra keyword arguments forwarded to :meth:`predict`.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame (same schema as :meth:`predict`).

        Examples
        --------
        >>> pred = AutoForecast(horizon=12).fit_predict("sales.csv")
        """
        self.fit(data, valid_data=valid_data)
        return self.predict(n=n, future_covariates=future_covariates, **predict_kwargs)

    @property
    def leader_board_(self):
        return None if self.router_ is None else self.router_.leader_board_

    @property
    def strategy_(self):
        return None if self.router_ is None else self.router_.strategy_

    @property
    def best_model_(self):
        return None if self.router_ is None else self.router_.best_model_

    def save(self, path: str, metadata: dict[str, Any] | None = None):
        """Save the fitted forecaster to a ``.pts`` binary file.

        The underlying SmartRouter is serialised using PipelineTS' custom
        binary format with SHA-256 integrity checksums.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'model.pts'``).
        metadata : dict or None, default None
            Optional user metadata stored in the file header.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been called yet.

        Examples
        --------
        >>> model.save('forecaster.pts', metadata={'version': '1.0'})
        """
        if self.router_ is None:
            raise ValueError("AutoForecast has not been fitted yet. Call fit() first.")
        return self.router_.save(path, metadata=metadata)

    @classmethod
    def load(cls, path: str | Path, verify_checksum: bool = True):
        """Load a previously saved AutoForecast from a ``.pts`` file.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pts`` file created by :meth:`save`.
        verify_checksum : bool, default True
            Whether to verify the SHA-256 integrity checksum before loading.

        Returns
        -------
        AutoForecast
            A restored ``AutoForecast`` instance ready for :meth:`predict`.

        Examples
        --------
        >>> loaded = AutoForecast.load('forecaster.pts')
        >>> pred = loaded.predict(n=12)
        """
        from PipelineTS.pipeline.smart_router import SmartRouter

        router = SmartRouter.load(str(path), verify_checksum=verify_checksum)
        wrapper = cls(
            time_col=getattr(router, "time_col", None),
            target_col=getattr(router, "target_col", None),
            horizon=getattr(router, "n_predict", None),
            quantile=getattr(router, "quantile", None),
            preset=getattr(router, "preset", "fast"),
            time_limit=getattr(router, "time_limit", None),
            id_col=getattr(router, "id_col", None),
            known_covariates=getattr(router, "known_covariates", None),
            past_covariates=getattr(router, "past_covariates", None),
            preprocess_data=False,
            verbose=getattr(router, "verbose", False),
        )
        wrapper.router_ = router
        wrapper.inferred_columns_ = {
            "time_col": getattr(router, "time_col", None),
            "target_col": getattr(router, "target_col", None),
            "id_col": getattr(router, "id_col", None),
        }
        return wrapper

    def __getattr__(self, name: str):
        router = self.__dict__.get("router_")
        if router is not None and hasattr(router, name):
            return getattr(router, name)
        raise AttributeError(name)


def forecast(
    data: pd.DataFrame | str | Path,
    n: int | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
    horizon: int | None = None,
    quantile: float | None = None,
    preset: str = "fast",
    time_limit: int | float | None = None,
    id_col: str | None = None,
    known_covariates: Sequence[str] | str | None = None,
    past_covariates: Sequence[str] | str | None = None,
    future_covariates: pd.DataFrame | None = None,
    preprocess_data: bool | dict[str, Any] = True,
    return_model: bool = False,
    verbose: bool = False,
    valid_data: pd.DataFrame | None = None,
    epochs: int | None = None,
    metric: str | Callable = "business",
    **router_kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, AutoForecast]:
    """One-line AutoML time series forecast — the simplest entry point.

    Automatically infers time/target columns, runs safe preprocessing,
    trains an :class:`AutoForecast` (backed by SmartRouter), and returns
    the next *n* predictions.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Historical time series data, or a path to a CSV/Excel/Parquet/JSON
        file.
    n : int or None, default None
        Number of future steps to forecast. Alias: *horizon*.
    time_col : str or None, default None
        Time column name. Auto-inferred when ``None``.
    target_col : str or None, default None
        Target column name. Auto-inferred when ``None``.
    horizon : int or None, default None
        Alias for *n*.  *n* takes precedence when both are given.
    quantile : float or None, default None
        Coverage level for prediction intervals (e.g. ``0.9``).  ``None``
        returns point forecasts only.
    preset : {'fast', 'medium_quality', 'high_quality', 'best_quality'}, default 'fast'
        Quality/speed trade-off preset.
    time_limit : int, float, or None, default None
        Total training time budget in seconds.
    id_col : str or None, default None
        Series-ID column for panel (multi-series) data.
    known_covariates : sequence of str or str or None, default None
        Columns whose future values are known at prediction time.
    past_covariates : sequence of str or str or None, default None
        Columns that are only available historically.
    future_covariates : pd.DataFrame or None, default None
        Future known covariate values for the forecast horizon (at least *n*
        rows, columns matching *known_covariates*).
    preprocess_data : bool or dict, default True
        Apply safe preprocessing before training.  Pass a dict to override
        specific :func:`preprocess` keyword arguments.
    return_model : bool, default False
        When ``True``, return ``(predictions, AutoForecast)`` instead of
        just the prediction DataFrame.
    verbose : bool, default False
        Print routing decisions during training.
    valid_data : pd.DataFrame or None, default None
        Optional held-out validation data.
    epochs : int or None, default None
        Maximum training epochs for all NN models. When ``None``, each model
        uses its preset-controlled default. Useful for quick experiments or
        debugging (e.g. ``epochs=50`` for a fast smoke test).
    **router_kwargs
        Additional keyword arguments forwarded to
        :class:`~PipelineTS.pipeline.SmartRouter`.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns ``[time_col, target_col]`` and
        optionally interval columns. Or ``(pd.DataFrame, AutoForecast)``
        when *return_model* is ``True``.

    Examples
    --------
    >>> from PipelineTS import forecast
    >>> pred = forecast("sales.csv", n=12)
    >>> pred = forecast(df, n=12, quantile=0.9, preset='high_quality')
    >>> pred, model = forecast(df, n=12, return_model=True)
    >>> model.save('forecaster.pts')
    """
    resolved_horizon = n if n is not None else horizon
    if epochs is not None:
        router_kwargs['epochs'] = epochs
    model = AutoForecast(
        time_col=time_col,
        target_col=target_col,
        horizon=resolved_horizon,
        quantile=quantile,
        preset=preset,
        time_limit=time_limit,
        id_col=id_col,
        known_covariates=known_covariates,
        past_covariates=past_covariates,
        preprocess_data=preprocess_data,
        verbose=verbose,
        metric=metric,
        **router_kwargs,
    )
    result = model.fit_predict(
        data,
        n=resolved_horizon,
        valid_data=valid_data,
        future_covariates=future_covariates,
    )
    if return_model:
        return result, model
    return result


def backtest(
    data: pd.DataFrame | str | Path,
    n: int | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
    id_col: str | None = None,
    n_splits: int = 3,
    test_size: int | None = None,
    metric: str | Callable = "mae",
    mode: str = "expanding",
    train_size: int | None = None,
    preset: str = "fast",
    time_limit: int | float | None = None,
    quantile: float | None = None,
    known_covariates: Sequence[str] | str | None = None,
    past_covariates: Sequence[str] | str | None = None,
    preprocess_data: bool | dict[str, Any] = True,
    verbose: bool = False,
    return_backtester: bool = False,
    **router_kwargs,
) -> dict[str, Any]:
    """Walk-forward backtesting with AutoML forecasting.

    Trains an :class:`AutoForecast` model on each expanding or sliding
    training window and evaluates it against a held-out test window,
    simulating how the model would have performed in production.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Historical time series data or path to a file.
    n : int or None, default None
        Test window size (forecast horizon per fold).  Auto-determined as
        ``min(12, len(data) // 10)`` when ``None``.
    time_col : str or None, default None
        Time column name. Auto-inferred when ``None``.
    target_col : str or None, default None
        Target column name. Auto-inferred when ``None``.
    id_col : str or None, default None
        Series-ID column for panel data.
    n_splits : int, default 3
        Number of walk-forward evaluation folds.
    test_size : int or None, default None
        Alias for *n*.  Used as test window size if *n* is ``None``.
    metric : str or callable, default 'mae'
        Evaluation metric.  Accepts a string shorthand (``'mae'``, ``'mse'``,
        ``'rmse'``, ``'mape'``, ``'smape'``, ``'wmape'``, ``'medae'``) or a
        callable with signature ``f(y_true, y_pred) -> float``.
    mode : {'expanding', 'sliding'}, default 'expanding'
        Walk-forward mode.

        - ``'expanding'``: training window grows with each fold.
        - ``'sliding'``: training window is fixed at *train_size*.

    train_size : int or None, default None
        Fixed training window size for ``'sliding'`` mode.  Ignored for
        ``'expanding'`` mode.
    preset : {'fast', 'medium_quality', 'high_quality', 'best_quality'}, default 'fast'
        Quality/speed preset for the :class:`AutoForecast` model.
    time_limit : int, float, or None, default None
        Time budget in seconds per fold.
    quantile : float or None, default None
        Coverage level for interval predictions.
    known_covariates : sequence of str or str or None, default None
        Columns with known future values.
    past_covariates : sequence of str or str or None, default None
        Columns that are only available historically.
    preprocess_data : bool or dict, default True
        Apply preprocessing before each fold.
    verbose : bool, default False
        Print progress during backtesting.
    return_backtester : bool, default False
        When ``True``, include the :class:`~PipelineTS.evaluation.Backtester`
        instance in the returned dict under key ``'backtester'``.
    **router_kwargs
        Additional keyword arguments forwarded to SmartRouter.

    Returns
    -------
    dict
        Result dictionary with keys:

        - ``results`` : list of per-fold metric values.
        - ``summary`` : dict with ``mean``, ``std``, ``min``, ``max``.
        - ``metric`` : str — name of the metric used.
        - ``time_col``, ``target_col``, ``id_col`` : resolved column names.
        - ``horizon`` : int — effective test window size.
        - ``n_splits`` : int — number of folds evaluated.
        - ``backtester`` : :class:`~PipelineTS.evaluation.Backtester`
          instance (only when *return_backtester* is ``True``).

    Examples
    --------
    >>> from PipelineTS import backtest
    >>> result = backtest("sales.csv", n=12, n_splits=5)
    >>> print(result["summary"])     # {'mean': ..., 'std': ..., ...}
    >>> result = backtest(df, n=12, metric='smape', mode='sliding',
    ...                   train_size=200)
    """
    from PipelineTS.evaluation import Backtester

    data = _validate_dataframe(data)
    resolved_time_col = infer_time_col(data, time_col)
    resolved_id_col = infer_id_col(data, id_col)
    resolved_target_col = infer_target_col(
        data,
        target_col,
        time_col=resolved_time_col,
        id_col=resolved_id_col,
        exclude=_as_list(known_covariates) + _as_list(past_covariates),
    )
    horizon = int(test_size or n or max(1, min(12, len(data) // 10)))
    clean_data = preprocess(
        data,
        time_col=resolved_time_col,
        target_col=resolved_target_col,
        id_col=resolved_id_col,
    )
    metric_func, metric_name = _resolve_metric(metric)
    model = AutoForecast(
        time_col=resolved_time_col,
        target_col=resolved_target_col,
        horizon=horizon,
        quantile=quantile,
        preset=preset,
        time_limit=time_limit,
        id_col=resolved_id_col,
        known_covariates=known_covariates,
        past_covariates=past_covariates,
        preprocess_data=preprocess_data,
        verbose=verbose,
        **router_kwargs,
    )
    tester = Backtester(
        model,
        time_col=resolved_time_col,
        target_col=resolved_target_col,
        metric=metric_func,
        metric_name=metric_name,
        id_col=resolved_id_col,
    )
    results = tester.fit(
        clean_data,
        n_splits=n_splits,
        test_size=horizon,
        mode=mode,
        train_size=train_size,
        verbose=verbose,
    )
    output = {
        "results": results,
        "summary": tester.summary(),
        "metric": metric_name,
        "time_col": resolved_time_col,
        "target_col": resolved_target_col,
        "id_col": resolved_id_col,
        "horizon": horizon,
        "n_splits": n_splits,
    }
    if return_backtester:
        output["backtester"] = tester
    return output
