from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from PipelineTS.utils.native_stats import ljung_box, pacf as native_pacf
from PipelineTS.utils.native_stats import seasonal_strength


def as_columns(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def primary_column(value) -> str:
    cols = as_columns(value)
    return cols[0] if cols else ""


def fmt(value, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return str(value)
        return f"{float(value):.{digits}g}"
    return str(value)


def finite_values(data: pd.DataFrame, col: str) -> np.ndarray:
    values = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy(dtype=np.float64)


def sorted_time_frame(data: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)


def acf_values(values: np.ndarray, max_lags: int) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 3:
        return np.array([])
    max_lags = int(min(max_lags, n - 2))
    y = y - np.mean(y)
    denom = float(np.dot(y, y))
    if denom <= 0:
        return np.zeros(max_lags + 1)
    out = [1.0]
    for lag in range(1, max_lags + 1):
        out.append(float(np.dot(y[:-lag], y[lag:]) / denom))
    return np.array(out)


def fft_periods(values: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return []
    x = np.arange(n, dtype=np.float64)
    try:
        coeffs = np.polyfit(x, y, 1)
        y = y - np.polyval(coeffs, x)
    except Exception:
        y = y - np.mean(y)
    y = y - np.mean(y)
    if np.nanstd(y) == 0:
        return []
    window = np.hanning(n)
    power = np.abs(np.fft.rfft(y * window)) ** 2
    freqs = np.fft.rfftfreq(n)
    periods = []
    for idx in np.argsort(power)[::-1]:
        if idx == 0 or freqs[idx] <= 0:
            continue
        period = int(round(1.0 / freqs[idx]))
        if 2 <= period <= max(2, n // 2) and all(abs(period - p) > max(1, p * 0.08) for p, _ in periods):
            periods.append((period, float(power[idx] / (power.sum() + 1e-12))))
        if len(periods) >= top_k:
            break
    return periods


def stl_strength(values: np.ndarray, period: int | None) -> dict:
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    if period is None or period < 2 or len(y) < period * 2:
        return {}
    try:
        return seasonal_strength(y, int(period))
    except Exception as exc:
        return {"error": str(exc)}


def hurst_exponent(values: np.ndarray) -> float | None:
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 32 or np.std(y) == 0:
        return None
    lags = np.unique(np.floor(np.logspace(np.log10(2), np.log10(max(3, n // 4)), 12)).astype(int))
    tau = []
    used = []
    for lag in lags:
        if lag < 2 or lag >= n:
            continue
        diff = y[lag:] - y[:-lag]
        s = np.sqrt(np.std(diff))
        if np.isfinite(s) and s > 0:
            tau.append(s)
            used.append(lag)
    if len(tau) < 3:
        return None
    slope = np.polyfit(np.log(used), np.log(tau), 1)[0]
    return float(slope * 2.0)


def spectral_entropy(values: np.ndarray) -> float | None:
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) < 8 or np.std(y) == 0:
        return None
    power = np.abs(np.fft.rfft(y - np.mean(y))) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 0:
        return None
    p = power / total
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def time_index_report(data: pd.DataFrame, time_col: str, max_examples: int = 8) -> str:
    if time_col not in data.columns:
        return f"Time column '{time_col}' not found."
    df = sorted_time_frame(data, time_col)
    if df.empty:
        return "No valid timestamps found."
    ts = df[time_col]
    diffs = ts.diff().dropna()
    lines = ["Time Index Analysis:"]
    lines.append(f"  Rows with valid timestamps: {len(df)} / {len(data)}")
    lines.append(f"  Range: {ts.min()} → {ts.max()}")
    lines.append(f"  Monotonic increasing: {pd.to_datetime(data[time_col], errors='coerce').is_monotonic_increasing}")
    lines.append(f"  Duplicate timestamps: {int(ts.duplicated().sum())}")
    if len(diffs) == 0:
        return "\n".join(lines)
    mode_delta = diffs.mode().iloc[0]
    inferred = pd.infer_freq(ts.drop_duplicates())
    regular_ratio = float((diffs == mode_delta).mean())
    lines.append(f"  pandas.infer_freq: {inferred}")
    lines.append(f"  Most common interval: {mode_delta}")
    lines.append(f"  Regularity ratio: {regular_ratio:.2%}")
    lines.append(f"  Min / median / max interval: {diffs.min()} / {diffs.median()} / {diffs.max()}")
    counts = diffs.value_counts().head(5)
    lines.append("  Top intervals: " + "; ".join(f"{idx}×{int(cnt)}" for idx, cnt in counts.items()))
    large_gaps = diffs[diffs > mode_delta]
    if len(large_gaps):
        examples = []
        for idx, delta in large_gaps.sort_values(ascending=False).head(max_examples).items():
            examples.append(f"{ts.iloc[idx - 1]}→{ts.iloc[idx]} ({delta})")
        lines.append(f"  Large gap examples: {'; '.join(examples)}")
    return "\n".join(lines)


def series_profile(data: pd.DataFrame, target_col) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    if not cols:
        return "No valid target columns found."
    lines = ["Series Characteristics:"]
    for col in cols:
        raw = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = raw.dropna().to_numpy(dtype=np.float64)
        lines.append(f"[{col}]")
        if len(y) == 0:
            lines.append("  No numeric values.")
            continue
        qs = np.nanpercentile(y, [1, 5, 25, 50, 75, 95, 99])
        mean = float(np.mean(y))
        std = float(np.std(y))
        cv = std / (abs(mean) + 1e-12)
        h = hurst_exponent(y)
        ent = spectral_entropy(y)
        lines.append(f"  Valid / missing: {len(y)} / {int(raw.isna().sum())}")
        lines.append(f"  Mean={fmt(mean)}, std={fmt(std)}, CV={fmt(cv)}, min={fmt(np.min(y))}, max={fmt(np.max(y))}")
        lines.append(f"  Quantiles p1/p5/p25/p50/p75/p95/p99: {', '.join(fmt(v) for v in qs)}")
        lines.append(f"  Skew={fmt(pd.Series(y).skew())}, kurtosis={fmt(pd.Series(y).kurtosis())}")
        lines.append(f"  Zeros={int(np.sum(y == 0))} ({np.mean(y == 0):.2%}), negatives={int(np.sum(y < 0))}")
        lines.append(f"  Hurst≈{fmt(h)}, spectral entropy≈{fmt(ent)}")
        suggestions = []
        if abs(pd.Series(y).skew()) > 1.5 and np.min(y) >= 0:
            suggestions.append("log1p/Box-Cox transform may reduce skewness")
        if cv > 1.0:
            suggestions.append("variance-stabilizing transform or robust loss may help")
        if ent is not None and ent < 0.45:
            suggestions.append("strong periodic structure likely present")
        if h is not None and h > 0.6:
            suggestions.append("persistent long-memory behavior; include longer lags")
        if suggestions:
            lines.append("  Hints: " + "; ".join(suggestions))
    return "\n".join(lines)


def autocorrelation_report(data: pd.DataFrame, target_col, max_lags: int = 40) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    lines = ["Autocorrelation Analysis:"]
    for col in cols:
        y = finite_values(data, col)
        max_lags_i = min(int(max_lags), max(1, len(y) - 2))
        acf = acf_values(y, max_lags_i)
        lines.append(f"[{col}]")
        if len(acf) <= 1:
            lines.append("  Not enough data.")
            continue
        conf = 1.96 / np.sqrt(len(y))
        significant = [(i, acf[i]) for i in range(1, len(acf)) if abs(acf[i]) > conf]
        top = sorted(significant, key=lambda x: abs(x[1]), reverse=True)[:10]
        first_zero = next((i for i in range(1, len(acf)) if acf[i] <= 0), None)
        lines.append(f"  Significant threshold: ±{conf:.3f}")
        lines.append(f"  ACF lag1={fmt(acf[1])}, lag2={fmt(acf[2] if len(acf) > 2 else None)}, first non-positive lag={first_zero}")
        lines.append("  Top significant lags: " + (", ".join(f"lag {i}: {v:.3f}" for i, v in top) if top else "None"))
        lb = ljung_box(y, lags=min(10, max_lags_i))
        p = float(lb["p_value"])
        lines.append(f"  Ljung-Box p-value@{min(10, max_lags_i)}: {p:.4g} ({'autocorrelated' if p < 0.05 else 'no strong evidence'})")
        pacf_vals = native_pacf(y, nlags=min(max_lags_i, len(y) // 2 - 1))
        top_pacf = sorted([(i, pacf_vals[i]) for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > conf], key=lambda x: abs(x[1]), reverse=True)[:8]
        lines.append("  Top PACF lags: " + (", ".join(f"lag {i}: {v:.3f}" for i, v in top_pacf) if top_pacf else "None"))
    return "\n".join(lines)


def seasonality_report(data: pd.DataFrame, target_col, period: int | None = None, top_k: int = 5) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    lines = ["Seasonality Detection:"]
    for col in cols:
        y = finite_values(data, col)
        lines.append(f"[{col}]")
        if len(y) < 8:
            lines.append("  Not enough data.")
            continue
        fft = fft_periods(y, top_k=top_k)
        acf = acf_values(y, min(len(y) // 2, max(10, int(max([p for p, _ in fft], default=2) * 2))))
        acf_peaks = []
        if len(acf) > 3:
            conf = 1.96 / np.sqrt(len(y))
            for lag in range(2, len(acf) - 1):
                if acf[lag] > conf and acf[lag] >= acf[lag - 1] and acf[lag] >= acf[lag + 1]:
                    acf_peaks.append((lag, float(acf[lag])))
        acf_peaks = sorted(acf_peaks, key=lambda x: x[1], reverse=True)[:top_k]
        candidate = period or (fft[0][0] if fft else (acf_peaks[0][0] if acf_peaks else None))
        strength = stl_strength(y, candidate)
        lines.append("  FFT candidate periods: " + (", ".join(f"{p} (power {s:.2%})" for p, s in fft) if fft else "None"))
        lines.append("  ACF peak periods: " + (", ".join(f"{p} (acf {s:.3f})" for p, s in acf_peaks) if acf_peaks else "None"))
        if candidate:
            lines.append(f"  Selected period for strength test: {candidate}")
        if strength:
            if "error" in strength:
                lines.append(f"  STL strength unavailable: {strength['error']}")
            else:
                lines.append(f"  STL seasonal strength={strength['seasonal_strength']:.3f}, trend strength={strength['trend_strength']:.3f}, residual std={fmt(strength['residual_std'])}")
        if candidate and strength.get("seasonal_strength", 0) > 0.3:
            lines.append("  Conclusion: meaningful seasonality detected.")
        elif fft or acf_peaks:
            lines.append("  Conclusion: possible weak/moderate seasonality; verify with plots or decomposition.")
        else:
            lines.append("  Conclusion: no clear seasonality detected.")
    return "\n".join(lines)


def trend_report(data: pd.DataFrame, time_col: str, target_col, window: int | None = None) -> str:
    from scipy import stats

    df = sorted_time_frame(data, time_col) if time_col in data.columns else data.copy()
    cols = [c for c in as_columns(target_col) if c in df.columns]
    lines = ["Trend Analysis:"]
    for col in cols:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(y)
        y = y[mask]
        x = np.arange(len(y), dtype=np.float64)
        lines.append(f"[{col}]")
        if len(y) < 3 or np.std(y) == 0:
            lines.append("  Not enough variation for trend analysis.")
            continue
        lr = stats.linregress(x, y)
        tau, tau_p = stats.kendalltau(x, y)
        rel_change = (lr.slope * (len(y) - 1)) / (abs(y[0]) + 1e-12)
        direction = "upward" if lr.slope > 0 else "downward" if lr.slope < 0 else "flat"
        lines.append(f"  Direction: {direction}; slope per step={fmt(lr.slope)}, total fitted change={rel_change:.2%}")
        lines.append(f"  Linear R²={lr.rvalue ** 2:.3f}, p-value={lr.pvalue:.4g}, Kendall tau={fmt(tau)} (p={fmt(tau_p)})")
        w = int(window or max(8, min(len(y) // 4, 60)))
        if len(y) >= w * 2:
            slopes = []
            starts = list(range(0, len(y) - w + 1, max(1, w // 2)))
            for start in starts:
                yy = y[start:start + w]
                slopes.append(stats.linregress(np.arange(len(yy)), yy).slope)
            slopes = np.asarray(slopes)
            reversals = int(np.sum(np.sign(slopes[1:]) != np.sign(slopes[:-1]))) if len(slopes) > 1 else 0
            lines.append(f"  Rolling slope median={fmt(np.median(slopes))}, min={fmt(np.min(slopes))}, max={fmt(np.max(slopes))}, sign reversals={reversals}")
    return "\n".join(lines)


def changepoint_report(data: pd.DataFrame, time_col: str, target_col, method: str = "auto", window: int | None = None, top_k: int = 5) -> str:
    df = sorted_time_frame(data, time_col) if time_col in data.columns else data.reset_index(drop=True)
    cols = [c for c in as_columns(target_col) if c in df.columns]
    lines = ["Changepoint Detection:"]
    for col in cols:
        y = pd.to_numeric(df[col], errors="coerce").interpolate().ffill().bfill().to_numpy(dtype=np.float64)
        lines.append(f"[{col}]")
        if len(y) < 20 or np.nanstd(y) == 0:
            lines.append("  Not enough data or variation.")
            continue
        n = len(y)
        w = int(window or max(8, min(n // 8, 60)))
        if n < w * 3:
            w = max(4, n // 4)
        scores = np.zeros(n)
        global_std = np.nanstd(y) + 1e-12
        for i in range(w, n - w):
            left = y[i - w:i]
            right = y[i:i + w]
            mean_score = abs(np.nanmean(right) - np.nanmean(left)) / global_std
            var_score = abs(np.nanstd(right) - np.nanstd(left)) / global_std
            if method == "mean":
                scores[i] = mean_score
            elif method == "variance":
                scores[i] = var_score
            elif method == "cusum":
                scores[i] = abs(np.nansum(right - np.nanmean(y)) - np.nansum(left - np.nanmean(y))) / (w * global_std)
            else:
                scores[i] = mean_score + 0.7 * var_score
        order = np.argsort(scores)[::-1]
        chosen = []
        for idx in order:
            if scores[idx] <= 0:
                break
            if all(abs(idx - j) >= w for j in chosen):
                chosen.append(int(idx))
            if len(chosen) >= top_k:
                break
        if not chosen:
            lines.append("  No obvious changepoints found.")
            continue
        for idx in sorted(chosen):
            left = y[max(0, idx - w):idx]
            right = y[idx:min(n, idx + w)]
            ts = df[time_col].iloc[idx] if time_col in df.columns else idx
            lines.append(f"  {ts}: score={scores[idx]:.3f}, mean {fmt(np.mean(left))} → {fmt(np.mean(right))}, std {fmt(np.std(left))} → {fmt(np.std(right))}")
    return "\n".join(lines)


def distribution_shift_report(data: pd.DataFrame, target_col, segments: int = 3) -> str:
    from scipy import stats

    cols = [c for c in as_columns(target_col) if c in data.columns]
    segments = max(2, min(int(segments), 8))
    lines = ["Distribution Shift Analysis:"]
    for col in cols:
        y = finite_values(data, col)
        lines.append(f"[{col}]")
        if len(y) < segments * 5:
            lines.append("  Not enough data for segment comparison.")
            continue
        parts = np.array_split(y, segments)
        base = parts[0]
        for i, part in enumerate(parts):
            lines.append(f"  Segment {i + 1}: n={len(part)}, mean={fmt(np.mean(part))}, median={fmt(np.median(part))}, std={fmt(np.std(part))}")
        for i in range(1, segments):
            ks = stats.ks_2samp(base, parts[i])
            mean_delta = (np.mean(parts[i]) - np.mean(base)) / (abs(np.mean(base)) + 1e-12)
            lines.append(f"  Segment 1 vs {i + 1}: KS p={ks.pvalue:.4g}, mean delta={mean_delta:.2%}")
        recent = parts[-1]
        ks_recent = stats.ks_2samp(np.concatenate(parts[:-1]), recent)
        lines.append(f"  Historical vs recent: KS p={ks_recent.pvalue:.4g} ({'shift likely' if ks_recent.pvalue < 0.05 else 'no strong shift evidence'})")
    return "\n".join(lines)


def volatility_report(data: pd.DataFrame, target_col, window: int | None = None) -> str:
    from scipy import stats

    cols = [c for c in as_columns(target_col) if c in data.columns]
    lines = ["Volatility Analysis:"]
    for col in cols:
        y = finite_values(data, col)
        lines.append(f"[{col}]")
        if len(y) < 10:
            lines.append("  Not enough data.")
            continue
        w = int(window or max(5, min(len(y) // 10, 60)))
        s = pd.Series(y)
        roll_std = s.rolling(w, min_periods=max(3, w // 3)).std().dropna().to_numpy()
        roll_mean = s.rolling(w, min_periods=max(3, w // 3)).mean().dropna().to_numpy()
        cv = roll_std / (np.abs(roll_mean) + 1e-12)
        lines.append(f"  Rolling window: {w}")
        lines.append(f"  Rolling std median={fmt(np.median(roll_std))}, p95={fmt(np.percentile(roll_std, 95))}, max={fmt(np.max(roll_std))}")
        lines.append(f"  Rolling CV median={fmt(np.median(cv))}, p95={fmt(np.percentile(cv, 95))}")
        if len(roll_std) > 3:
            lr = stats.linregress(np.arange(len(roll_std)), roll_std)
            lines.append(f"  Volatility trend slope={fmt(lr.slope)}, R²={lr.rvalue ** 2:.3f}, p={lr.pvalue:.4g}")
        high = np.where(roll_std >= np.percentile(roll_std, 95))[0][:8]
        if len(high):
            lines.append("  High-volatility window endpoints: " + ", ".join(str(int(i + w - 1)) for i in high))
        diffs = np.diff(y)
        lb = ljung_box(diffs ** 2, lags=min(10, len(diffs) - 1))
        p = float(lb["p_value"])
        lines.append(f"  Squared-difference Ljung-Box p={p:.4g} ({'volatility clustering likely' if p < 0.05 else 'no strong clustering evidence'})")
    return "\n".join(lines)


def lag_feature_report(data: pd.DataFrame, target_col, max_lags: int = 60, top_k: int = 10) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    lines = ["Lag Feature Diagnostics:"]
    for col in cols:
        y = finite_values(data, col)
        max_lags_i = min(int(max_lags), max(1, len(y) // 2))
        acf = acf_values(y, max_lags_i)
        lines.append(f"[{col}]")
        if len(acf) <= 1:
            lines.append("  Not enough data.")
            continue
        conf = 1.96 / np.sqrt(len(y))
        ranked = sorted([(i, acf[i]) for i in range(1, len(acf))], key=lambda x: abs(x[1]), reverse=True)[:top_k]
        sig_lags = [i for i in range(1, len(acf)) if abs(acf[i]) > conf]
        suggested_window = min(max_lags_i, max(sig_lags) if sig_lags else max(3, min(12, len(y) // 5)))
        lines.append(f"  Suggested lag window: {suggested_window}")
        lines.append("  Top lag correlations: " + ", ".join(f"lag {i}: {v:.3f}" for i, v in ranked))
        lines.append(f"  Significant lag count: {len(sig_lags)} using ±{conf:.3f}")
    return "\n".join(lines)


def calendar_effect_report(data: pd.DataFrame, time_col: str, target_col, granularity: str = "auto", top_k: int = 10) -> str:
    df = sorted_time_frame(data, time_col)
    target = primary_column(target_col)
    if target not in df.columns:
        return f"Target column '{target}' not found."
    y = pd.to_numeric(df[target], errors="coerce")
    candidates = {
        "hour": df[time_col].dt.hour,
        "weekday": df[time_col].dt.dayofweek,
        "dayofmonth": df[time_col].dt.day,
        "month": df[time_col].dt.month,
        "quarter": df[time_col].dt.quarter,
    }
    if granularity != "auto":
        candidates = {granularity: candidates[granularity]} if granularity in candidates else {}
    lines = ["Calendar Effect Detection:"]
    total_var = float(np.nanvar(y))
    if total_var <= 0:
        return "Calendar Effect Detection:\n  Target has no variance."
    results = []
    for name, labels in candidates.items():
        frame = pd.DataFrame({"label": labels, "y": y}).dropna()
        grouped = frame.groupby("label")["y"]
        if grouped.ngroups < 2 or grouped.size().min() < 2:
            continue
        means = grouped.mean()
        counts = grouped.size()
        weighted_between = float(((means - frame["y"].mean()) ** 2 * counts).sum() / len(frame))
        strength = weighted_between / (total_var + 1e-12)
        results.append((name, strength, means.sort_values(ascending=False), counts))
    if not results:
        lines.append("  No reliable calendar grouping available.")
        return "\n".join(lines)
    for name, strength, means, counts in sorted(results, key=lambda x: x[1], reverse=True):
        lines.append(f"  {name}: effect strength={strength:.3f}, groups={len(means)}")
        top = means.head(min(top_k, len(means)))
        lines.append("    Highest groups: " + "; ".join(f"{idx}: mean={fmt(val)}, n={int(counts.loc[idx])}" for idx, val in top.items()))
    return "\n".join(lines)


def covariate_relationship_report(data: pd.DataFrame, target_col, time_col: str | None = None, id_col: str | None = None, covariates: Sequence[str] | None = None, max_lag: int = 12, top_k: int = 10) -> str:
    from scipy import stats

    target = primary_column(target_col)
    if target not in data.columns:
        return f"Target column '{target}' not found."
    exclude = set(as_columns(target_col) + [c for c in [time_col, id_col] if c])
    if covariates:
        candidates = [c for c in covariates if c in data.columns and c != target]
    else:
        candidates = [c for c in data.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not candidates:
        return "Covariate Relationship Analysis:\n  No numeric covariates found."
    df = data.sort_values(time_col).reset_index(drop=True) if time_col and time_col in data.columns else data.reset_index(drop=True)
    y = pd.to_numeric(df[target], errors="coerce")
    rows = []
    lag_rows = []
    for col in candidates:
        x = pd.to_numeric(df[col], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pair) < 5 or pair["x"].std() == 0 or pair["y"].std() == 0:
            continue
        pear = pair["x"].corr(pair["y"], method="pearson")
        spear = pair["x"].corr(pair["y"], method="spearman")
        _, p = stats.pearsonr(pair["x"], pair["y"])
        rows.append((col, float(pear), float(spear), float(p), len(pair)))
        best_lag = None
        best_corr = 0.0
        for lag in range(-int(max_lag), int(max_lag) + 1):
            if lag == 0:
                shifted = x
            elif lag > 0:
                shifted = x.shift(lag)
            else:
                shifted = x.shift(lag)
            pair_lag = pd.DataFrame({"x": shifted, "y": y}).dropna()
            if len(pair_lag) < 5 or pair_lag["x"].std() == 0 or pair_lag["y"].std() == 0:
                continue
            corr = pair_lag["x"].corr(pair_lag["y"])
            if abs(corr) > abs(best_corr):
                best_corr = float(corr)
                best_lag = lag
        lag_rows.append((col, best_lag, best_corr))
    lines = ["Covariate Relationship Analysis:"]
    if rows:
        lines.append("  Same-time correlations:")
        for col, pear, spear, p, n in sorted(rows, key=lambda r: abs(r[1]), reverse=True)[:top_k]:
            lines.append(f"    {col}: Pearson={pear:.3f}, Spearman={spear:.3f}, p={p:.4g}, n={n}")
    if lag_rows:
        lines.append("  Best lead/lag correlations (positive lag means covariate is shifted later):")
        for col, lag, corr in sorted(lag_rows, key=lambda r: abs(r[2]), reverse=True)[:top_k]:
            lines.append(f"    {col}: lag={lag}, corr={corr:.3f}")
    return "\n".join(lines)


def intermittency_report(data: pd.DataFrame, target_col) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    lines = ["Intermittency / Demand Pattern Analysis:"]
    for col in cols:
        y = finite_values(data, col)
        lines.append(f"[{col}]")
        if len(y) == 0:
            lines.append("  No numeric values.")
            continue
        nonzero = y[y != 0]
        zero_ratio = float(np.mean(y == 0))
        if len(nonzero) == 0:
            lines.append("  All observations are zero.")
            continue
        idx = np.flatnonzero(y != 0)
        adi = float(len(y) / len(nonzero))
        cv2 = float((np.std(nonzero) / (np.mean(nonzero) + 1e-12)) ** 2)
        if adi < 1.32 and cv2 < 0.49:
            cls = "smooth"
        elif adi >= 1.32 and cv2 < 0.49:
            cls = "intermittent"
        elif adi < 1.32 and cv2 >= 0.49:
            cls = "erratic"
        else:
            cls = "lumpy"
        gaps = np.diff(idx) if len(idx) > 1 else np.array([])
        lines.append(f"  Zero ratio={zero_ratio:.2%}, nonzero count={len(nonzero)}")
        lines.append(f"  ADI={adi:.3f}, CV²={cv2:.3f}, classification={cls}")
        if len(gaps):
            lines.append(f"  Nonzero gap median={fmt(np.median(gaps))}, p95={fmt(np.percentile(gaps, 95))}, max={fmt(np.max(gaps))}")
    return "\n".join(lines)


def decomposition_report(data: pd.DataFrame, target_col, period: int | None = None) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    lines = ["Component Decomposition Summary:"]
    for col in cols:
        y = finite_values(data, col)
        if period is None:
            periods = fft_periods(y, top_k=1)
            period_i = periods[0][0] if periods else None
        else:
            period_i = period
        lines.append(f"[{col}]")
        if not period_i:
            lines.append("  No candidate period found.")
            continue
        strength = stl_strength(y, period_i)
        if not strength or "error" in strength:
            lines.append(f"  Decomposition unavailable: {strength.get('error', 'unknown error') if strength else 'not enough data'}")
            continue
        lines.append(f"  Period={period_i}, seasonal strength={strength['seasonal_strength']:.3f}, trend strength={strength['trend_strength']:.3f}, residual std={fmt(strength['residual_std'])}")
        if strength["seasonal_strength"] > 0.6:
            lines.append("  Interpretation: strong seasonality; seasonal models/features are recommended.")
        if strength["trend_strength"] > 0.6:
            lines.append("  Interpretation: strong trend; detrending/differencing or trend-aware models may help.")
    return "\n".join(lines)


def recommendation_report(data: pd.DataFrame, time_col: str, target_col, id_col: str | None = None) -> str:
    target = primary_column(target_col)
    lines = ["Time Series Action Recommendations:"]
    if time_col in data.columns:
        df = sorted_time_frame(data, time_col)
        diffs = df[time_col].diff().dropna()
        if len(diffs):
            mode_delta = diffs.mode().iloc[0]
            regularity = float((diffs == mode_delta).mean())
            if regularity < 0.98:
                lines.append(f"  Preprocessing: regularize/resample the time index; regularity is only {regularity:.2%}.")
            if int(df[time_col].duplicated().sum()) > 0:
                lines.append("  Preprocessing: aggregate or remove duplicate timestamps before modeling.")
    if target in data.columns:
        y = finite_values(data, target)
        if len(y):
            skew = float(pd.Series(y).skew())
            if abs(skew) > 1.5 and np.nanmin(y) >= 0:
                lines.append(f"  Transform: target is skewed (skew={skew:.2f}); try log1p or Box-Cox.")
            if np.mean(y == 0) > 0.2:
                lines.append(f"  Modeling: zero ratio is {np.mean(y == 0):.1%}; consider intermittent-demand diagnostics and robust losses.")
            acf = acf_values(y, min(60, max(2, len(y) // 2)))
            if len(acf) > 1 and acf[1] > 0.5:
                sig = [i for i in range(1, len(acf)) if abs(acf[i]) > 1.96 / np.sqrt(len(y))]
                if sig:
                    lines.append(f"  Features: strong autocorrelation; include lags up to about {max(sig)} and rolling statistics.")
            periods = fft_periods(y, top_k=3)
            if periods:
                lines.append("  Seasonality: candidate periods " + ", ".join(str(p) for p, _ in periods) + "; add Fourier/calendar features or seasonal models.")
            h = hurst_exponent(y)
            if h is not None and h > 0.6:
                lines.append(f"  Features: Hurst≈{h:.2f}; longer lookback windows may help.")
            ent = spectral_entropy(y)
            if ent is not None and ent > 0.85:
                lines.append(f"  Modeling: high spectral entropy ({ent:.2f}); prefer robust tree/ensemble models and wider validation.")
    if id_col and id_col in data.columns:
        sizes = data.groupby(id_col).size()
        lines.append(f"  Panel data: {len(sizes)} series; min/median/max length={int(sizes.min())}/{fmt(sizes.median())}/{int(sizes.max())}.")
        if sizes.min() < 20:
            lines.append("  Panel data: some series are short; global models or pooled feature engineering are preferred.")
    if len(lines) == 1:
        lines.append("  No specific issues detected from quick diagnostics. Proceed with visual inspection, decomposition, and SmartRouter.")
    return "\n".join(lines)


def _clean_numeric_sequence(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.array([])
    values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(values.notna().sum()) == 0:
        return np.array([])
    values = values.interpolate(method="linear").ffill().bfill().dropna()
    return values.to_numpy(dtype=np.float64)


def _safe_corr(a, b) -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n = min(len(x), len(y))
    if n < 4:
        return None
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return None
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y), len(p))
    if n == 0:
        return {}
    y = y[:n]
    p = p[:n]
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) == 0:
        return {}
    y = y[mask]
    p = p[mask]
    err = y - p
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wmape": float(np.sum(np.abs(err)) / (np.sum(np.abs(y)) + 1e-12)),
        "smape": float(np.mean(2.0 * np.abs(err) / (np.abs(y) + np.abs(p) + 1e-12))),
    }


def _evaluate_baselines(
    values: np.ndarray,
    horizon: int | None = None,
    seasonal_period: int | None = None,
    test_size: int | None = None,
) -> dict:
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return {}
    requested = test_size if test_size is not None else horizon
    holdout = int(requested) if requested else max(3, min(24, n // 5))
    holdout = max(1, min(holdout, max(1, n // 3)))
    if n - holdout < 4:
        return {}
    train = y[:-holdout]
    test = y[-holdout:]
    baselines = {
        "naive_last": np.repeat(train[-1], holdout),
        "mean": np.repeat(np.mean(train), holdout),
        "median": np.repeat(np.median(train), holdout),
    }
    if len(train) >= 2:
        step = (train[-1] - train[0]) / max(1, len(train) - 1)
        baselines["drift"] = train[-1] + step * np.arange(1, holdout + 1)
    period_i = seasonal_period
    if period_i is None:
        periods = fft_periods(train, top_k=1)
        period_i = periods[0][0] if periods else None
    if period_i and period_i >= 2 and len(train) >= period_i:
        last_season = train[-int(period_i):]
        baselines[f"seasonal_naive_p{int(period_i)}"] = np.asarray(
            [last_season[i % len(last_season)] for i in range(holdout)],
            dtype=np.float64,
        )
    metrics = {name: _metric_summary(test, pred) for name, pred in baselines.items()}
    metrics = {name: val for name, val in metrics.items() if val}
    if not metrics:
        return {}
    best = min(metrics.items(), key=lambda item: item[1]["mae"])
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seasonal_period": int(period_i) if period_i else None,
        "metrics": metrics,
        "best_name": best[0],
        "best_metric": best[1],
    }


def baseline_forecast_report(
    data: pd.DataFrame,
    time_col: str,
    target_col,
    id_col: str | None = None,
    horizon: int | None = None,
    seasonal_period: int | None = None,
    test_size: int | None = None,
) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    if not cols:
        return "Baseline Forecast Benchmark:\n  No valid target columns found."
    df = sorted_time_frame(data, time_col) if time_col in data.columns else data.reset_index(drop=True)
    lines = ["Baseline Forecast Benchmark:"]
    for col in cols:
        lines.append(f"[{col}]")
        results = []
        if id_col and id_col in df.columns:
            for group_id, group in df.groupby(id_col, sort=False):
                y = _clean_numeric_sequence(group.sort_values(time_col) if time_col in group.columns else group, col)
                result = _evaluate_baselines(y, horizon=horizon, seasonal_period=seasonal_period, test_size=test_size)
                if result:
                    result["series_id"] = group_id
                    results.append(result)
        else:
            y = _clean_numeric_sequence(df, col)
            result = _evaluate_baselines(y, horizon=horizon, seasonal_period=seasonal_period, test_size=test_size)
            if result:
                results.append(result)
        if not results:
            lines.append("  Not enough usable history to benchmark naive baselines.")
            continue
        weighted = {}
        total_points = 0
        for result in results:
            n_test = result["n_test"]
            total_points += n_test
            for name, metric in result["metrics"].items():
                if name not in weighted:
                    weighted[name] = {"weight": 0, "mae": 0.0, "rmse": 0.0, "wmape": 0.0, "smape": 0.0}
                weighted[name]["weight"] += n_test
                for key in ("mae", "rmse", "wmape", "smape"):
                    weighted[name][key] += metric[key] * n_test
        for name, metric in weighted.items():
            weight = max(1, metric.pop("weight"))
            for key in ("mae", "rmse", "wmape", "smape"):
                metric[key] /= weight
        best_name, best_metric = min(weighted.items(), key=lambda item: item[1]["mae"])
        lines.append(f"  Evaluated series: {len(results)}, holdout points: {total_points}")
        periods = sorted({r["seasonal_period"] for r in results if r.get("seasonal_period")})
        if periods:
            lines.append("  Seasonal periods used: " + ", ".join(str(p) for p in periods[:5]))
        lines.append(f"  Best baseline by MAE: {best_name} (MAE={fmt(best_metric['mae'])}, WMAPE={best_metric['wmape']:.2%})")
        for name, metric in sorted(weighted.items(), key=lambda item: item[1]["mae"]):
            lines.append(
                f"    {name}: MAE={fmt(metric['mae'])}, RMSE={fmt(metric['rmse'])}, "
                f"WMAPE={metric['wmape']:.2%}, sMAPE={metric['smape']:.2%}"
            )
        lines.append("  Modeling target: trained models should materially beat the best naive baseline on the same validation horizon.")
    return "\n".join(lines)


def forecastability_report(
    data: pd.DataFrame,
    target_col,
    horizon: int | None = None,
    seasonal_period: int | None = None,
) -> str:
    cols = [c for c in as_columns(target_col) if c in data.columns]
    if not cols:
        return "Forecastability Assessment:\n  No valid target columns found."
    lines = ["Forecastability Assessment:"]
    for col in cols:
        y = finite_values(data, col)
        lines.append(f"[{col}]")
        if len(y) < 8:
            lines.append("  Not enough numeric observations.")
            continue
        acf = acf_values(y, min(40, max(2, len(y) // 2)))
        acf1 = float(acf[1]) if len(acf) > 1 else 0.0
        entropy = spectral_entropy(y)
        hurst = hurst_exponent(y)
        period_i = seasonal_period
        if period_i is None:
            periods = fft_periods(y, top_k=1)
            period_i = periods[0][0] if periods else None
        strength = stl_strength(y, period_i)
        seasonal_strength = strength.get("seasonal_strength") if strength and "error" not in strength else None
        trend_strength = strength.get("trend_strength") if strength and "error" not in strength else None
        diff_ratio = float(np.nanstd(np.diff(y)) / (np.nanstd(y) + 1e-12)) if len(y) > 2 else None
        unique_ratio = float(len(np.unique(y)) / len(y))
        score = 50.0
        score += 18.0 * min(1.0, abs(acf1))
        if seasonal_strength is not None:
            score += 20.0 * seasonal_strength
        if trend_strength is not None:
            score += 8.0 * trend_strength
        if entropy is not None:
            score += (0.55 - entropy) * 25.0
        if diff_ratio is not None and diff_ratio > 1.2:
            score -= min(18.0, (diff_ratio - 1.2) * 12.0)
        if len(y) < max(30, int(horizon or 1) * 4):
            score -= 12.0
        if unique_ratio < 0.05:
            score -= 10.0
        score = float(np.clip(score, 0.0, 100.0))
        level = "high" if score >= 70 else "moderate" if score >= 45 else "low"
        lines.append(f"  Forecastability score: {score:.1f}/100 ({level})")
        lines.append(f"  ACF lag1={fmt(acf1)}, spectral entropy={fmt(entropy)}, Hurst≈{fmt(hurst)}, diff/std ratio={fmt(diff_ratio)}")
        if period_i:
            lines.append(f"  Candidate seasonal period: {period_i}")
        if seasonal_strength is not None or trend_strength is not None:
            lines.append(f"  STL strengths: seasonal={fmt(seasonal_strength)}, trend={fmt(trend_strength)}")
        hints = []
        if abs(acf1) < 0.15 and (entropy is None or entropy > 0.8):
            hints.append("weak memory/high entropy; expect limited gains over robust baselines")
        if seasonal_strength is not None and seasonal_strength > 0.4:
            hints.append("seasonal models, Fourier/calendar features, or seasonal naive baselines are important")
        if trend_strength is not None and trend_strength > 0.5:
            hints.append("trend-aware models or differencing/detrending should be considered")
        if len(y) < 50:
            hints.append("short history; prefer simple/statistical models and conservative validation")
        if hints:
            lines.append("  Hints: " + "; ".join(hints))
    return "\n".join(lines)


def panel_structure_report(
    data: pd.DataFrame,
    time_col: str,
    target_col,
    id_col: str | None,
) -> str:
    lines = ["Panel / Multi-Series Structure Analysis:"]
    if not id_col:
        lines.append("  No id_col configured; data is treated as a single series.")
        return "\n".join(lines)
    if id_col not in data.columns:
        lines.append(f"  id_col '{id_col}' not found.")
        return "\n".join(lines)
    if time_col not in data.columns:
        lines.append(f"  time_col '{time_col}' not found.")
        return "\n".join(lines)
    df = sorted_time_frame(data, time_col)
    grouped = df.groupby(id_col, sort=False)
    sizes = grouped.size()
    lines.append(f"  Series count: {len(sizes)}")
    lines.append(f"  Length min/median/mean/max: {int(sizes.min())}/{fmt(sizes.median())}/{fmt(sizes.mean())}/{int(sizes.max())}")
    duplicates = int(df.duplicated([id_col, time_col]).sum())
    lines.append(f"  Duplicate (id, time) rows: {duplicates}")
    regularities = []
    inferred_freqs = []
    ranges = []
    for _, group in grouped:
        ts = pd.to_datetime(group[time_col], errors="coerce").dropna().sort_values()
        if len(ts) == 0:
            continue
        ranges.append((ts.min(), ts.max()))
        if len(ts) > 2:
            diffs = ts.diff().dropna()
            if len(diffs):
                mode_delta = diffs.mode().iloc[0]
                regularities.append(float((diffs == mode_delta).mean()))
            try:
                inferred = pd.infer_freq(ts.drop_duplicates())
            except Exception:
                inferred = None
            if inferred:
                inferred_freqs.append(inferred)
    if regularities:
        lines.append(f"  Regularity ratio min/median: {min(regularities):.2%}/{np.median(regularities):.2%}")
    if inferred_freqs:
        counts = pd.Series(inferred_freqs).value_counts().head(5)
        lines.append("  Top inferred frequencies: " + "; ".join(f"{idx}×{int(cnt)}" for idx, cnt in counts.items()))
    if ranges:
        starts = [r[0] for r in ranges]
        ends = [r[1] for r in ranges]
        lines.append(f"  Global time coverage: {min(starts)} → {max(ends)}")
        lines.append(f"  Series start range: {min(starts)} → {max(starts)}")
        lines.append(f"  Series end range: {min(ends)} → {max(ends)}")
    target = primary_column(target_col)
    if target in df.columns:
        missing_by_series = grouped[target].apply(lambda s: pd.to_numeric(s, errors="coerce").isna().mean())
        means = grouped[target].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        stds = grouped[target].apply(lambda s: pd.to_numeric(s, errors="coerce").std())
        lines.append(f"  Target missing ratio median/max: {missing_by_series.median():.2%}/{missing_by_series.max():.2%}")
        if means.notna().sum() > 1:
            mean_cv = float(means.std() / (abs(means.mean()) + 1e-12))
            lines.append(f"  Cross-series mean CV: {fmt(mean_cv)}")
        if stds.notna().sum() > 1:
            std_cv = float(stds.std() / (abs(stds.mean()) + 1e-12))
            lines.append(f"  Cross-series volatility CV: {fmt(std_cv)}")
    short = int((sizes < 20).sum())
    if short:
        lines.append(f"  Warning: {short} series have fewer than 20 observations; prefer pooled/global models or exclude very short series.")
    if duplicates:
        lines.append("  Action: sort and aggregate duplicate (id, time) rows before modeling.")
    if regularities and min(regularities) < 0.98:
        lines.append("  Action: resample or regularize irregular series before models that assume fixed frequency.")
    return "\n".join(lines)


def _lead_corr_for_column(
    data: pd.DataFrame,
    time_col: str | None,
    target: str,
    feature: str,
    lag: int,
    id_col: str | None = None,
) -> float | None:
    frames = []
    if id_col and id_col in data.columns:
        iterator = data.groupby(id_col, sort=False)
    else:
        iterator = [(None, data)]
    for _, group in iterator:
        g = group.sort_values(time_col) if time_col and time_col in group.columns else group.reset_index(drop=True)
        x = pd.to_numeric(g[feature], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(g[target], errors="coerce").to_numpy(dtype=np.float64)
        if len(x) <= lag + 3:
            continue
        frames.append((x[:-lag], y[lag:]))
    if not frames:
        return None
    x_all = np.concatenate([item[0] for item in frames])
    y_all = np.concatenate([item[1] for item in frames])
    return _safe_corr(x_all, y_all)


def leakage_risk_report(
    data: pd.DataFrame,
    time_col: str,
    target_col,
    id_col: str | None = None,
    known_covariates: Sequence[str] | None = None,
    past_covariates: Sequence[str] | None = None,
    feature_cols: Sequence[str] | None = None,
    horizon: int | None = None,
    corr_threshold: float = 0.98,
) -> str:
    target = primary_column(target_col)
    lines = ["Leakage Risk Assessment:"]
    if target not in data.columns:
        lines.append(f"  Target column '{target}' not found.")
        return "\n".join(lines)
    known = set(known_covariates or [])
    past = set(past_covariates or [])
    configured = list(dict.fromkeys(list(known) + list(past) + list(feature_cols or [])))
    exclude = set(as_columns(target_col) + [c for c in [time_col, id_col] if c])
    if configured:
        candidates = [c for c in configured if c in data.columns and c not in exclude]
    else:
        candidates = [c for c in data.columns if c not in exclude]
    if not candidates:
        lines.append("  No feature/covariate columns to inspect.")
        return "\n".join(lines)
    high = []
    medium = []
    target_values = pd.to_numeric(data[target], errors="coerce")
    max_lag = max(1, min(int(horizon or 12), 24))
    target_name = target.lower()
    for col in candidates:
        lower = col.lower()
        if col in known:
            medium.append(f"{col}: configured as known_covariate; ensure future values are available for the forecast horizon.")
        if any(token in lower for token in ["lead", "future", "ahead", "next", "t+", "forecast"]):
            high.append(f"{col}: column name suggests future information.")
        if target_name and target_name in lower and col not in past:
            medium.append(f"{col}: column name references the target; verify it is causal and not target-derived at the same timestamp.")
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
        feature_values = pd.to_numeric(data[col], errors="coerce")
        same_corr = _safe_corr(feature_values.to_numpy(dtype=np.float64), target_values.to_numpy(dtype=np.float64))
        if same_corr is not None and abs(same_corr) >= 0.999:
            high.append(f"{col}: nearly identical to target at the same timestamp (corr={same_corr:.4f}).")
        elif same_corr is not None and abs(same_corr) >= corr_threshold:
            medium.append(f"{col}: very high same-time target correlation (corr={same_corr:.4f}); review availability and causality.")
        best_lag = None
        best_corr = 0.0
        for lag in range(1, max_lag + 1):
            corr = _lead_corr_for_column(data, time_col, target, col, lag, id_col=id_col)
            if corr is not None and abs(corr) > abs(best_corr):
                best_lag = lag
                best_corr = corr
        if best_lag is not None and abs(best_corr) >= 0.999:
            high.append(f"{col}: appears to contain target {best_lag} step(s) ahead (lead corr={best_corr:.4f}).")
        elif best_lag is not None and abs(best_corr) >= corr_threshold and col not in known:
            medium.append(f"{col}: high correlation with future target at lead {best_lag} (corr={best_corr:.4f}); verify it is observable at prediction time.")
    if high:
        lines.append("  High-risk findings:")
        for item in dict.fromkeys(high):
            lines.append(f"    - {item}")
    if medium:
        lines.append("  Review findings:")
        for item in dict.fromkeys(medium):
            lines.append(f"    - {item}")
    if not high and not medium:
        lines.append("  No obvious leakage risks detected from names, same-time correlations, or lead correlations.")
    lines.append("  Principle: features must be known at prediction time; target-derived or future-shifted columns should not be used as covariates.")
    return "\n".join(lines)


def modeling_readiness_report(
    data: pd.DataFrame,
    time_col: str,
    target_col,
    id_col: str | None = None,
    horizon: int | None = None,
    known_covariates: Sequence[str] | None = None,
    past_covariates: Sequence[str] | None = None,
    feature_cols: Sequence[str] | None = None,
) -> str:
    lines = ["Modeling Readiness Assessment:"]
    critical = []
    warnings_out = []
    suggestions = []
    if time_col not in data.columns:
        critical.append(f"time_col '{time_col}' not found")
    cols = [c for c in as_columns(target_col) if c in data.columns]
    if not cols:
        critical.append("no valid target columns found")
    df = sorted_time_frame(data, time_col) if time_col in data.columns else data.copy()
    horizon_i = int(horizon or max(1, min(24, max(1, len(df) // 10))))
    if time_col in data.columns:
        valid_ts = pd.to_datetime(data[time_col], errors="coerce").notna().mean()
        if valid_ts < 1.0:
            warnings_out.append(f"time column has {(1 - valid_ts):.2%} invalid timestamps")
        keys = [time_col]
        if id_col and id_col in data.columns:
            keys = [id_col, time_col]
        duplicates = int(data.dropna(subset=[time_col]).duplicated(keys).sum())
        if duplicates:
            warnings_out.append(f"{duplicates} duplicate time keys need aggregation")
        diffs = df[time_col].diff().dropna() if time_col in df.columns else pd.Series(dtype="timedelta64[ns]")
        if len(diffs):
            mode_delta = diffs.mode().iloc[0]
            regularity = float((diffs == mode_delta).mean())
            if regularity < 0.98 and not (id_col and id_col in data.columns):
                warnings_out.append(f"time index regularity is {regularity:.2%}; resampling may be needed")
    if id_col:
        if id_col not in data.columns:
            warnings_out.append(f"id_col '{id_col}' not found; panel modeling will not be used")
        else:
            sizes = data.groupby(id_col).size()
            lines.append(f"  Panel data: {len(sizes)} series, min/median/max length={int(sizes.min())}/{fmt(sizes.median())}/{int(sizes.max())}")
            if int(sizes.min()) < max(8, horizon_i * 2):
                warnings_out.append("some series are too short for the requested horizon")
            if len(sizes) > 1 and int(sizes.median()) < 50:
                suggestions.append("prefer global/panel-capable models over per-series local models")
    for col in cols:
        raw = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = raw.dropna().to_numpy(dtype=np.float64)
        lines.append(f"  Target '{col}': usable={len(y)}/{len(data)}, missing={raw.isna().mean():.2%}")
        if len(y) < max(30, horizon_i * 4):
            warnings_out.append(f"target '{col}' has limited usable history for horizon={horizon_i}")
        if len(y) and np.nanstd(y) <= 0:
            critical.append(f"target '{col}' is constant")
        if raw.isna().mean() > 0.1:
            warnings_out.append(f"target '{col}' has more than 10% missing values")
        if len(y) >= 8:
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_ratio = float(np.mean((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)))
                if outlier_ratio > 0.05:
                    warnings_out.append(f"target '{col}' has {outlier_ratio:.2%} IQR outliers")
            if np.mean(y == 0) > 0.2:
                suggestions.append(f"target '{col}' is zero-heavy; check intermittency and robust losses")
            periods = fft_periods(y, top_k=2)
            if periods:
                suggestions.append(f"candidate seasonal periods for '{col}': " + ", ".join(str(p) for p, _ in periods))
            acf = acf_values(y, min(40, max(2, len(y) // 2)))
            sig = [i for i in range(1, len(acf)) if abs(acf[i]) > 1.96 / np.sqrt(len(y))]
            if sig:
                suggestions.append(f"use lag window up to about {max(sig)} for '{col}'")
    configured = list(dict.fromkeys(list(known_covariates or []) + list(past_covariates or []) + list(feature_cols or [])))
    missing_features = [c for c in configured if c not in data.columns]
    if missing_features:
        warnings_out.append(f"configured feature/covariate columns not found: {missing_features}")
    if known_covariates:
        suggestions.append("provide future_covariates at prediction time for all known_covariates")
    if len(df) <= horizon_i * 3:
        warnings_out.append("history is short relative to forecast horizon; validation estimates may be unstable")
    suggestions.append(f"recommended validation holdout/test_size: at least {horizon_i}")
    suggestions.append("run baseline_forecast_report before expensive training and require models to beat the best naive baseline")
    status = "READY"
    if warnings_out:
        status = "READY_WITH_WARNINGS"
    if critical:
        status = "NOT_READY"
    lines.insert(1, f"  Status: {status}")
    if critical:
        lines.append("  Blocking issues:")
        for item in dict.fromkeys(critical):
            lines.append(f"    - {item}")
    if warnings_out:
        lines.append("  Warnings:")
        for item in dict.fromkeys(warnings_out):
            lines.append(f"    - {item}")
    if suggestions:
        lines.append("  Modeling guidance:")
        for item in dict.fromkeys(suggestions):
            lines.append(f"    - {item}")
    return "\n".join(lines)
