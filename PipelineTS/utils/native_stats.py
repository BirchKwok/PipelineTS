from collections import namedtuple

import numpy as np
from scipy import stats


DecompositionResult = namedtuple("DecompositionResult", ["trend", "seasonal", "resid"])


def finite_array(values):
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    return y[np.isfinite(y)]


def acf(values, nlags=40, fft=True):
    y = finite_array(values)
    n = len(y)
    if n == 0:
        return np.array([], dtype=np.float64)
    nlags = int(max(0, min(nlags, n - 1)))
    if n == 1:
        return np.ones(1, dtype=np.float64)
    y = y - np.mean(y)
    denom = float(np.dot(y, y))
    if denom <= 1e-15:
        out = np.zeros(nlags + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    if fft:
        nfft = 1 << int(np.ceil(np.log2(max(1, 2 * n - 1))))
        fy = np.fft.rfft(y, n=nfft)
        cov = np.fft.irfft(fy * np.conj(fy), n=nfft)[:nlags + 1]
    else:
        cov = np.array([np.dot(y[:n - lag], y[lag:]) for lag in range(nlags + 1)], dtype=np.float64)
    out = cov / denom
    out[0] = 1.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def pacf(values, nlags=40):
    y = finite_array(values)
    n = len(y)
    if n == 0:
        return np.array([], dtype=np.float64)
    nlags = int(max(0, min(nlags, n // 2 if n > 2 else n - 1)))
    r = acf(y, nlags=nlags, fft=True)
    out = np.zeros(nlags + 1, dtype=np.float64)
    out[0] = 1.0
    sigma = 1.0
    phi = np.array([], dtype=np.float64)
    for k in range(1, nlags + 1):
        if sigma <= 1e-12:
            out[k:] = 0.0
            break
        if k == 1:
            alpha = r[1]
        else:
            alpha = (r[k] - np.dot(phi, r[1:k][::-1])) / sigma
        alpha = float(np.clip(np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0), -0.999999, 0.999999))
        if k == 1:
            phi = np.array([alpha], dtype=np.float64)
        else:
            phi = np.concatenate([phi - alpha * phi[::-1], np.array([alpha], dtype=np.float64)])
        sigma *= max(1.0 - alpha * alpha, 1e-12)
        out[k] = alpha
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def ljung_box(values, lags=10):
    y = finite_array(values)
    n = len(y)
    lag = int(max(1, min(lags, n - 1))) if n > 1 else 1
    if n <= lag + 1:
        return {"statistic": np.nan, "p_value": np.nan, "has_autocorrelation": None}
    r = acf(y, nlags=lag, fft=True)[1:lag + 1]
    denom = n - np.arange(1, lag + 1, dtype=np.float64)
    statistic = float(n * (n + 2) * np.sum((r ** 2) / np.maximum(denom, 1.0)))
    p_value = float(stats.chi2.sf(statistic, df=lag))
    return {"statistic": statistic, "p_value": p_value, "has_autocorrelation": p_value < 0.05}


def _edge_fill(values):
    arr = np.asarray(values, dtype=np.float64).copy()
    if len(arr) == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def seasonal_decompose(values, period, model="additive"):
    y = _edge_fill(np.asarray(values, dtype=np.float64).reshape(-1))
    n = len(y)
    if n == 0:
        return DecompositionResult(np.array([]), np.array([]), np.array([]))
    period = int(period or max(2, n // 10))
    period = max(2, min(period, max(2, n)))
    kernel = np.ones(period, dtype=np.float64) / period
    trend = np.convolve(y, kernel, mode="same")
    half = period // 2
    if n > period and half > 0:
        trend[:half] = trend[half]
        trend[-half:] = trend[-half - 1]
    trend = _edge_fill(trend)
    eps = 1e-12
    if model == "multiplicative" and np.all(np.abs(trend) > eps):
        detrended = y / np.where(np.abs(trend) > eps, trend, eps)
        seasonal_pattern = np.array([np.nanmean(detrended[i::period]) for i in range(period)], dtype=np.float64)
        center = np.nanmean(seasonal_pattern)
        if np.isfinite(center) and abs(center) > eps:
            seasonal_pattern = seasonal_pattern / center
        seasonal = np.tile(seasonal_pattern, n // period + 1)[:n]
        seasonal = _edge_fill(seasonal)
        resid = y / np.where(np.abs(trend * seasonal) > eps, trend * seasonal, eps)
    else:
        detrended = y - trend
        seasonal_pattern = np.array([np.nanmean(detrended[i::period]) for i in range(period)], dtype=np.float64)
        seasonal_pattern = seasonal_pattern - np.nanmean(seasonal_pattern)
        seasonal = np.tile(seasonal_pattern, n // period + 1)[:n]
        seasonal = _edge_fill(seasonal)
        resid = y - trend - seasonal
    return DecompositionResult(trend, seasonal, resid)


def seasonal_strength(values, period):
    y = finite_array(values)
    if period is None or period < 2 or len(y) < period * 2:
        return {}
    result = seasonal_decompose(y, period=int(period), model="additive")
    resid = np.asarray(result.resid, dtype=np.float64)
    trend = np.asarray(result.trend, dtype=np.float64)
    seasonal = np.asarray(result.seasonal, dtype=np.float64)
    var_resid = np.nanvar(resid)
    season = max(0.0, min(1.0, 1.0 - var_resid / (np.nanvar(seasonal + resid) + 1e-12)))
    trend_s = max(0.0, min(1.0, 1.0 - var_resid / (np.nanvar(trend + resid) + 1e-12)))
    return {
        "seasonal_strength": float(season),
        "trend_strength": float(trend_s),
        "residual_std": float(np.nanstd(resid)),
    }


def _ols(y, x):
    xtx = x.T @ x
    xty = x.T @ y
    try:
        coef = np.linalg.solve(xtx, xty)
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
        coef = xtx_inv @ xty
    resid = y - x @ coef
    rss = float(np.dot(resid, resid))
    n, k = x.shape
    sigma2 = rss / max(n - k, 1)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    return coef, se, resid, rss


def _adf_critical_values(regression):
    if regression == "ct":
        return {"1%": -3.96, "5%": -3.41, "10%": -3.12}
    return {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def _adf_pvalue(statistic, critical_values):
    c1 = critical_values["1%"]
    c5 = critical_values["5%"]
    c10 = critical_values["10%"]
    if statistic <= c1:
        return 0.005
    if statistic <= c5:
        return 0.01 + 0.04 * (statistic - c1) / (c5 - c1)
    if statistic <= c10:
        return 0.05 + 0.05 * (statistic - c5) / (c10 - c5)
    return float(min(0.99, 0.10 + 0.20 * (statistic - c10)))


def adf_test(values, significance_level=0.05, max_lag=None, regression="c"):
    y = finite_array(values)
    n = len(y)
    critical_values = _adf_critical_values(regression)
    if n < 8 or np.nanstd(y) <= 1e-15:
        return {
            "test": "ADF",
            "statistic": 0.0,
            "p_value": 1.0,
            "used_lag": 0,
            "n_obs": int(n),
            "critical_values": critical_values,
            "is_stationary": False,
        }
    dy = np.diff(y)
    if max_lag is None:
        max_lag = int(np.floor(12 * (n / 100.0) ** 0.25))
    max_lag = int(max(0, min(max_lag, max(0, n // 3 - 1), len(dy) - 2)))
    best = None
    for lag in range(max_lag + 1):
        obs = len(dy) - lag
        if obs <= lag + 3:
            continue
        endog = dy[lag:]
        cols = []
        if regression in ("c", "ct"):
            cols.append(np.ones(obs, dtype=np.float64))
        if regression == "ct":
            cols.append(np.arange(lag + 1, n, dtype=np.float64))
        level_idx = len(cols)
        cols.append(y[lag:-1])
        for j in range(1, lag + 1):
            cols.append(dy[lag - j:-j])
        x = np.column_stack(cols)
        try:
            coef, se, _, rss = _ols(endog, x)
        except np.linalg.LinAlgError:
            continue
        k = x.shape[1]
        aic = obs * np.log(max(rss / obs, 1e-15)) + 2 * k
        stat = float(coef[level_idx] / max(se[level_idx], 1e-12))
        if best is None or aic < best[0]:
            best = (aic, stat, lag, obs)
    if best is None:
        statistic, used_lag, n_obs = 0.0, 0, n
    else:
        _, statistic, used_lag, n_obs = best
    p_value = float(_adf_pvalue(statistic, critical_values))
    return {
        "test": "ADF",
        "statistic": float(statistic),
        "p_value": p_value,
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": critical_values,
        "is_stationary": p_value < significance_level,
    }


def _kpss_critical_values(regression):
    if regression == "ct":
        return {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}
    return {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}


def _kpss_pvalue(statistic, critical_values):
    c10 = critical_values["10%"]
    c5 = critical_values["5%"]
    c25 = critical_values["2.5%"]
    c1 = critical_values["1%"]
    if statistic <= c10:
        return 0.10
    if statistic <= c5:
        return 0.10 - 0.05 * (statistic - c10) / (c5 - c10)
    if statistic <= c25:
        return 0.05 - 0.025 * (statistic - c5) / (c25 - c5)
    if statistic <= c1:
        return 0.025 - 0.015 * (statistic - c25) / (c1 - c25)
    return 0.01


def kpss_test(values, significance_level=0.05, regression="c", nlags="auto"):
    y = finite_array(values)
    n = len(y)
    critical_values = _kpss_critical_values(regression)
    if n < 8 or np.nanstd(y) <= 1e-15:
        return {
            "test": "KPSS",
            "statistic": 0.0,
            "p_value": 0.10,
            "used_lag": 0,
            "critical_values": critical_values,
            "is_stationary": True,
        }
    cols = [np.ones(n, dtype=np.float64)]
    if regression == "ct":
        cols.append(np.arange(n, dtype=np.float64))
    x = np.column_stack(cols)
    coef, _, resid, _ = _ols(y, x)
    if nlags == "auto":
        lag = int(np.floor(12 * (n / 100.0) ** 0.25))
    else:
        lag = int(nlags)
    lag = int(max(0, min(lag, n - 1)))
    s = np.cumsum(resid)
    eta = float(np.sum(s ** 2) / (n ** 2))
    lrv = float(np.dot(resid, resid) / n)
    for j in range(1, lag + 1):
        gamma = float(np.dot(resid[j:], resid[:-j]) / n)
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    statistic = float(eta / max(lrv, 1e-15))
    p_value = float(_kpss_pvalue(statistic, critical_values))
    return {
        "test": "KPSS",
        "statistic": statistic,
        "p_value": p_value,
        "used_lag": int(lag),
        "critical_values": critical_values,
        "is_stationary": p_value >= significance_level,
    }
