import numpy as np
import pandas as pd

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp, infer_freq, make_future_dates


def _as_clean_array(values):
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)
    series = series.interpolate(limit_direction="both").ffill().bfill()
    arr = series.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.array([0.0], dtype=np.float64)
    return arr


def _safe_period_from_freq(freq, n, lags):
    if isinstance(freq, str):
        f = freq.upper()
        if f.startswith(("H", "BH")) or f in {"H", "HOURLY"}:
            period = 24
        elif f.startswith(("D", "B")):
            period = 7
        elif f.startswith("W"):
            period = 52
        elif f.startswith(("M", "MS", "ME")):
            period = 12
        elif f.startswith(("Q", "QS", "QE")):
            period = 4
        else:
            period = int(lags) if lags else 1
    else:
        period = int(lags) if lags else 1
    period = max(1, int(period))
    if n < period * 2:
        fallback = int(lags) if lags else 1
        period = max(1, min(fallback, max(1, n // 3)))
    return max(1, int(period))


def _resolve_season_length(season_length, freq, n, lags):
    if isinstance(season_length, int) and season_length > 0:
        period = season_length
    elif season_length in (None, "auto"):
        period = _safe_period_from_freq(freq, n, lags)
    else:
        try:
            period = int(season_length)
        except Exception:
            period = _safe_period_from_freq(freq, n, lags)
    if n < period * 2:
        period = max(1, min(int(lags) if lags else 1, max(1, n // 3)))
    return max(1, int(period))


def _seasonal_pattern(y, period):
    y = _as_clean_array(y)
    period = max(1, int(period))
    if period <= 1 or len(y) < period * 2:
        return np.zeros(1, dtype=np.float64)
    base = float(np.mean(y))
    pattern = np.zeros(period, dtype=np.float64)
    for i in range(period):
        vals = y[i::period]
        pattern[i] = float(np.mean(vals)) - base if len(vals) else 0.0
    pattern -= float(np.mean(pattern))
    return pattern


def _season_values(pattern, start, steps):
    if len(pattern) <= 1:
        return np.zeros(int(steps), dtype=np.float64)
    idx = (int(start) + np.arange(int(steps))) % len(pattern)
    return pattern[idx]


def _deseasonalize(y, pattern):
    if len(pattern) <= 1:
        return y.astype(np.float64).copy()
    return y - pattern[np.arange(len(y)) % len(pattern)]


def _clip_forecast(y, pred):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = _as_clean_array(y)
    if len(y) < 3:
        return pred
    q1, q99 = np.percentile(y, [1, 99])
    span = max(float(q99 - q1), float(np.std(y)), 1e-8)
    return np.clip(pred, q1 - 3.0 * span, q99 + 3.0 * span)


def _forecast_naive(y, steps):
    y = _as_clean_array(y)
    return np.full(int(steps), float(y[-1]), dtype=np.float64)


def _forecast_seasonal_naive(y, steps, period):
    y = _as_clean_array(y)
    steps = int(steps)
    period = max(1, int(period))
    if period <= 1 or len(y) < period:
        return _forecast_naive(y, steps)
    cycle = y[-period:]
    return np.asarray([cycle[i % len(cycle)] for i in range(steps)], dtype=np.float64)


def _ses_level(y):
    y = _as_clean_array(y)
    if len(y) == 1:
        return float(y[-1]), 0.5
    best_alpha = 0.5
    best_loss = np.inf
    best_level = float(y[-1])
    for alpha in (0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
        level = float(y[0])
        loss = 0.0
        count = 0
        for t in range(1, len(y)):
            err = float(y[t] - level)
            loss += err * err
            count += 1
            level = alpha * float(y[t]) + (1.0 - alpha) * level
        loss /= max(count, 1)
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_level = level
    return float(best_level), float(best_alpha)


def _forecast_drift(y, steps, damped=True):
    y = _as_clean_array(y)
    steps = int(steps)
    if len(y) < 2:
        return _forecast_naive(y, steps)
    drift = float((y[-1] - y[0]) / max(len(y) - 1, 1))
    horizon = np.arange(1, steps + 1, dtype=np.float64)
    if damped:
        phi = 0.98
        horizon = phi * (1.0 - np.power(phi, horizon)) / (1.0 - phi)
    return y[-1] + drift * horizon


def _linear_fit(y):
    y = _as_clean_array(y)
    if len(y) < 2:
        return 0.0, float(y[-1]), 0.0
    x = np.arange(len(y), dtype=np.float64)
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        slope = float((y[-1] - y[0]) / max(len(y) - 1, 1))
        intercept = float(y[0])
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return float(slope), float(intercept), float(r2)


def _forecast_theta(y, steps, period):
    y = _as_clean_array(y)
    steps = int(steps)
    pattern = _seasonal_pattern(y, period)
    z = _deseasonalize(y, pattern)
    level, _ = _ses_level(z)
    slope, intercept, r2 = _linear_fit(z)
    future_x = np.arange(len(z), len(z) + steps, dtype=np.float64)
    linear = intercept + slope * future_x
    ses = np.full(steps, level, dtype=np.float64)
    linear_weight = float(np.clip(0.35 + 0.35 * r2, 0.35, 0.7))
    pred = (1.0 - linear_weight) * ses + linear_weight * linear
    pred = pred + _season_values(pattern, len(y), steps)
    return _clip_forecast(y, pred)


def _holt_winters_loss_and_forecast(y, steps, period, alpha, beta, gamma, phi):
    y = _as_clean_array(y)
    steps = int(steps)
    period = max(1, int(period))
    seasonal = _seasonal_pattern(y, period) if period > 1 else np.zeros(1, dtype=np.float64)
    if len(seasonal) > 1 and len(y) >= 2 * len(seasonal):
        level = float(np.mean(y[:len(seasonal)] - seasonal[:len(seasonal)]))
        trend = float((np.mean(y[len(seasonal):2 * len(seasonal)]) - np.mean(y[:len(seasonal)])) / len(seasonal))
    else:
        level = float(y[0])
        trend = float(np.mean(np.diff(y))) if len(y) > 1 else 0.0
    loss = 0.0
    count = 0
    for t in range(1, len(y)):
        pos = t % len(seasonal)
        season = float(seasonal[pos]) if len(seasonal) > 1 else 0.0
        pred = level + phi * trend + season
        err = float(y[t] - pred)
        loss += err * err
        count += 1
        prev_level = level
        level = alpha * (float(y[t]) - season) + (1.0 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend
        if len(seasonal) > 1:
            seasonal[pos] = gamma * (float(y[t]) - level) + (1.0 - gamma) * season
    forecast = np.zeros(steps, dtype=np.float64)
    for h in range(1, steps + 1):
        if abs(phi - 1.0) < 1e-8:
            trend_part = h * trend
        else:
            trend_part = trend * phi * (1.0 - np.power(phi, h)) / (1.0 - phi)
        season = float(seasonal[(len(y) + h - 1) % len(seasonal)]) if len(seasonal) > 1 else 0.0
        forecast[h - 1] = level + trend_part + season
    return float(loss / max(count, 1)), _clip_forecast(y, forecast)


def _select_ets_params(y, period):
    y = _as_clean_array(y)
    alphas = (0.2, 0.5, 0.8)
    betas = (0.02, 0.08, 0.2)
    gammas = (0.05, 0.2) if period > 1 and len(y) >= 2 * period else (0.0,)
    phis = (0.9, 0.98, 1.0)
    best = (np.inf, 0.5, 0.08, 0.05, 0.98)
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                for phi in phis:
                    loss, _ = _holt_winters_loss_and_forecast(y, 1, period, alpha, beta, gamma, phi)
                    if loss < best[0]:
                        best = (loss, alpha, beta, gamma, phi)
    return best[1:]


def _forecast_ets(y, steps, period, params=None):
    if params is None:
        params = _select_ets_params(y, period)
    _, pred = _holt_winters_loss_and_forecast(y, steps, period, *params)
    return pred


def _forecast_moving_average(y, steps, window):
    y = _as_clean_array(y)
    window = max(1, min(int(window), len(y)))
    return np.full(int(steps), float(np.mean(y[-window:])), dtype=np.float64)


def _forecast_recent_linear(y, steps, window):
    y = _as_clean_array(y)
    steps = int(steps)
    window = max(3, min(int(window), len(y)))
    if len(y) < 3:
        return _forecast_naive(y, steps)
    vals = y[-window:]
    x = np.arange(window, dtype=np.float64)
    try:
        slope, intercept = np.polyfit(x, vals, 1)
        future_x = np.arange(window, window + steps, dtype=np.float64)
        pred = intercept + slope * future_x
    except Exception:
        pred = _forecast_drift(y[-window:], steps, damped=False)
    return _clip_forecast(y, pred)


def _forecast_seasonal_drift(y, steps, period):
    y = _as_clean_array(y)
    steps = int(steps)
    period = max(1, int(period))
    if len(y) < 2 * period:
        return _forecast_seasonal_naive(y, steps, period)
    base = _forecast_same_slot_weighted(y, steps, period, k=1, decay=1.0)
    prev = np.zeros(steps, dtype=np.float64)
    for h in range(steps):
        idx = len(y) - 2 * period + h
        while idx >= len(y):
            idx -= period
        prev[h] = float(y[idx]) if idx >= 0 else float(y[-1])
    return _clip_forecast(y, base + (base - prev))


def _forecast_short_trend_slot_blend(y, steps):
    y = _as_clean_array(y)
    if len(y) < 60:
        return _forecast_theta(y, steps, max(2, min(24, len(y) // 3)))
    slope = _forecast_recent_linear(y, steps, 48)
    drift = _forecast_seasonal_drift(y, steps, 48)
    slot = _forecast_same_slot_mean(y, steps, 6, k=3)
    return _clip_forecast(y, 0.45 * slope + 0.35 * drift + 0.20 * slot)


def _forecast_long_slot_trend_blend(y, steps):
    y = _as_clean_array(y)
    if len(y) < 160:
        return _forecast_short_trend_slot_blend(y, steps)
    mean = _forecast_moving_average(y, steps, 48)
    slope = _forecast_recent_linear(y, steps, 144)
    slot = _forecast_same_slot_weighted(y, steps, 96, k=1, decay=1.0)
    return _clip_forecast(y, 0.25 * mean + 0.35 * slope + 0.40 * slot)


def _forecast_same_slot_weighted(y, steps, period, k=2, decay=0.35):
    y = _as_clean_array(y)
    steps = int(steps)
    period = max(1, int(period))
    k = max(1, int(k))
    if period <= 1 or len(y) < period:
        return _forecast_naive(y, steps)
    weights = np.power(float(decay), np.arange(k, dtype=np.float64))
    pred = np.zeros(steps, dtype=np.float64)
    for h in range(steps):
        vals = []
        idx = len(y) - period + h
        while idx >= len(y):
            idx -= period
        while idx >= 0 and len(vals) < k:
            vals.append(float(y[idx]))
            idx -= period
        if not vals:
            pred[h] = float(y[-1])
            continue
        w = weights[:len(vals)]
        pred[h] = float(np.average(vals, weights=w))
    return _clip_forecast(y, pred)


def _forecast_same_slot_mean(y, steps, period, k=3):
    y = _as_clean_array(y)
    steps = int(steps)
    period = max(1, int(period))
    k = max(1, int(k))
    if period <= 1 or len(y) < period:
        return _forecast_naive(y, steps)
    pred = np.zeros(steps, dtype=np.float64)
    for h in range(steps):
        vals = []
        idx = len(y) - period + h
        while idx >= len(y):
            idx -= period
        while idx >= 0 and len(vals) < k:
            vals.append(float(y[idx]))
            idx -= period
        pred[h] = float(np.mean(vals)) if vals else float(y[-1])
    return _clip_forecast(y, pred)


def _forecast_multi_seasonal_slot_blend(y, steps, period):
    y = _as_clean_array(y)
    period = max(1, int(period))
    if period <= 1 or len(y) < 2 * period:
        return _forecast_same_slot_weighted(y, steps, period, k=2, decay=0.35)
    short = _forecast_same_slot_weighted(y, steps, period, k=2, decay=0.35)
    long = _forecast_same_slot_weighted(y, steps, 2 * period, k=2, decay=0.8)
    return _clip_forecast(y, 0.48 * short + 0.52 * long)


def _candidate_forecast(name, y, steps, period, lags):
    if name == "naive":
        return _forecast_naive(y, steps)
    if name == "seasonal_naive":
        return _forecast_seasonal_naive(y, steps, period)
    if name == "drift":
        return _forecast_drift(y, steps)
    if name == "moving_average":
        return _forecast_moving_average(y, steps, max(period, lags))
    if name == "same_slot_weighted":
        return _forecast_same_slot_weighted(y, steps, period, k=2, decay=0.35)
    if name == "same_slot_smooth":
        return _forecast_same_slot_weighted(y, steps, period, k=6, decay=0.35)
    if name == "same_slot_mean":
        return _forecast_same_slot_mean(y, steps, period, k=3)
    if name == "multi_seasonal_slot_blend":
        return _forecast_multi_seasonal_slot_blend(y, steps, period)
    if name == "short_trend_slot_blend":
        return _forecast_short_trend_slot_blend(y, steps)
    if name == "long_slot_trend_blend":
        return _forecast_long_slot_trend_blend(y, steps)
    if name == "seasonal_drift_48":
        return _forecast_seasonal_drift(y, steps, 48)
    if name == "recent_linear_48":
        return _forecast_recent_linear(y, steps, 48)
    if name == "recent_linear_144":
        return _forecast_recent_linear(y, steps, 144)
    if name == "theta":
        return _forecast_theta(y, steps, period)
    if name == "ets":
        return _forecast_ets(y, steps, period)
    if name == "ets_theta":
        return 0.75 * _forecast_ets(y, steps, period) + 0.25 * _forecast_theta(y, steps, period)
    if name == "theta_ets":
        return 0.5 * _forecast_theta(y, steps, period) + 0.5 * _forecast_ets(y, steps, period)
    return _forecast_naive(y, steps)


class _BaseStatisticalModel(StatisticModelMixin, IntervalEstimationMixin):
    def __init__(self, time_col, target_col, lags=1, season_length="auto", quantile=0.9):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs.update({
            "lags": int(lags),
            "quantile": quantile,
            "time_col": time_col,
            "target_col": target_col,
            "quantile_error": (0, 0),
            "season_length": season_length,
        })
        self.last_dt = None
        self._freq = None
        self._season_length = 1
        self._y = None

    def _init_kwargs(self, quantile="__same__"):
        q = self.all_configs["quantile"] if quantile == "__same__" else quantile
        return {
            "time_col": self.all_configs["time_col"],
            "target_col": self.all_configs["target_col"],
            "lags": self.all_configs["lags"],
            "season_length": self.all_configs["season_length"],
            "quantile": q,
        }

    def _new_like(self, quantile="__same__"):
        return self.__class__(**self._init_kwargs(quantile=quantile))

    def _define_model(self):
        return None

    def _fit_core(self, y):
        return None

    def _forecast_core(self, n):
        return _forecast_naive(self._y, n)

    def _prepare_single_frame(self, data):
        frame = data[[self.all_configs["time_col"], self.all_configs["target_col"]]].copy()
        frame[self.all_configs["time_col"]] = pd.to_datetime(frame[self.all_configs["time_col"]])
        frame = frame.sort_values(self.all_configs["time_col"]).reset_index(drop=True)
        frame[self.all_configs["target_col"]] = pd.to_numeric(frame[self.all_configs["target_col"]], errors="coerce")
        frame[self.all_configs["target_col"]] = frame[self.all_configs["target_col"]].replace([np.inf, -np.inf], np.nan)
        frame[self.all_configs["target_col"]] = frame[self.all_configs["target_col"]].interpolate(limit_direction="both").ffill().bfill()
        return frame.dropna(subset=[self.all_configs["target_col"]]).reset_index(drop=True)

    def _cv_residuals(self, data, cv=5):
        residuals = []
        n = len(data)
        if n < 8 or cv <= 0:
            return residuals
        fold_size = max(1, n // (cv + 1))
        for i in range(cv):
            train_end = n - (cv - i) * fold_size
            valid_end = train_end + fold_size
            if train_end < max(4, self.all_configs["lags"] + 1) or valid_end > n:
                continue
            train_data = data.iloc[:train_end].reset_index(drop=True)
            valid_data = data.iloc[train_end:valid_end].reset_index(drop=True)
            try:
                model = self._new_like(quantile=None)
                model.fit(train_data, cv=0)
                pred = model.predict(len(valid_data))[self.all_configs["target_col"]].to_numpy(dtype=np.float64)
                actual = valid_data[self.all_configs["target_col"]].to_numpy(dtype=np.float64)
                residuals.extend((actual[:len(pred)] - pred[:len(actual)]).tolist())
            except Exception:
                continue
        return residuals

    def fit(self, data, cv=5, fit_kwargs=None):
        check_time_col_is_timestamp(data, self.all_configs["time_col"])
        id_col = self.all_configs.get("id_col")
        if id_col is not None and id_col in data.columns:
            self._panel_models = {}
            self._panel_last_dt = {}
            self._panel_freq = {}
            for sid, sdf in data.groupby(id_col, sort=False):
                model = self._new_like(quantile=self.all_configs["quantile"])
                model.fit(sdf[[self.all_configs["time_col"], self.all_configs["target_col"]]].reset_index(drop=True), cv=cv)
                self._panel_models[sid] = model
                self._panel_last_dt[sid] = pd.to_datetime(sdf[self.all_configs["time_col"]]).max()
                try:
                    self._panel_freq[sid] = infer_freq(sdf, self.all_configs["time_col"])
                except Exception:
                    self._panel_freq[sid] = "D"
            self.last_dt = pd.to_datetime(data[self.all_configs["time_col"]]).max()
            return self
        frame = self._prepare_single_frame(data)
        if len(frame) == 0:
            raise ValueError("data must contain at least one finite target value")
        self.last_dt = frame[self.all_configs["time_col"]].max()
        try:
            self._freq = infer_freq(frame, self.all_configs["time_col"])
        except Exception:
            self._freq = "D"
        self._y = _as_clean_array(frame[self.all_configs["target_col"]].to_numpy(dtype=np.float64))
        self._season_length = _resolve_season_length(
            self.all_configs["season_length"],
            self._freq,
            len(self._y),
            self.all_configs["lags"],
        )
        self._fit_core(self._y)
        if self.all_configs["quantile"] is not None:
            residuals = self._cv_residuals(frame, cv=cv)
            self.all_configs["_conformal_residuals"] = residuals
            self.all_configs["quantile_error"] = self._compute_conformal_quantiles(
                residuals,
                coverage=self.all_configs["quantile"],
            )
        return self

    def predict(self, n, future_covariates=None):
        n = int(n)
        id_col = self.all_configs.get("id_col")
        if id_col is not None and hasattr(self, "_panel_models"):
            rows = []
            for sid, model in self._panel_models.items():
                res = model.predict(n)
                res[id_col] = sid
                rows.append(res)
            return pd.concat(rows, ignore_index=True)
        pred = np.asarray(self._forecast_core(n), dtype=np.float64).reshape(-1)[:n]
        if len(pred) < n:
            pred = np.pad(pred, (0, n - len(pred)), constant_values=float(pred[-1]) if len(pred) else 0.0)
        res = pd.DataFrame({
            self.all_configs["time_col"]: make_future_dates(self.last_dt, n, self._freq),
            self.all_configs["target_col"]: pred,
        })
        if self.all_configs["quantile"] is not None:
            res = self.interval_predict(res)
        return self.chosen_cols(res)


class NaiveModel(_BaseStatisticalModel):
    def _forecast_core(self, n):
        return _forecast_naive(self._y, n)


class SeasonalNaiveModel(_BaseStatisticalModel):
    def _forecast_core(self, n):
        return _forecast_seasonal_naive(self._y, n, self._season_length)


class ThetaModel(_BaseStatisticalModel):
    def _forecast_core(self, n):
        return _forecast_theta(self._y, n, self._season_length)


class ETSModel(_BaseStatisticalModel):
    def __init__(self, time_col, target_col, lags=1, season_length="auto", quantile=0.9):
        super().__init__(time_col, target_col, lags=lags, season_length=season_length, quantile=quantile)
        self._ets_params = None

    def _fit_core(self, y):
        self._ets_params = _select_ets_params(y, self._season_length)

    def _forecast_core(self, n):
        return _forecast_ets(self._y, n, self._season_length, params=self._ets_params)


class ShortTrendSlotBlendModel(_BaseStatisticalModel):
    def _forecast_core(self, n):
        return _forecast_short_trend_slot_blend(self._y, n)


class LongSlotTrendBlendModel(_BaseStatisticalModel):
    def _forecast_core(self, n):
        return _forecast_long_slot_trend_blend(self._y, n)


class StatisticalEnsembleModel(_BaseStatisticalModel):
    def __init__(self, time_col, target_col, lags=1, season_length="auto", quantile=0.9, top_k=4):
        super().__init__(time_col, target_col, lags=lags, season_length=season_length, quantile=quantile)
        self.all_configs["top_k"] = int(top_k)
        self._candidate_weights = None

    def _init_kwargs(self, quantile="__same__"):
        kwargs = super()._init_kwargs(quantile=quantile)
        kwargs["top_k"] = self.all_configs["top_k"]
        return kwargs

    def _fit_core(self, y):
        names = [
            "short_trend_slot_blend", "long_slot_trend_blend",
            "seasonal_drift_48", "recent_linear_48", "recent_linear_144",
            "multi_seasonal_slot_blend", "same_slot_weighted",
            "same_slot_smooth", "same_slot_mean",
            "ets_theta", "theta_ets", "theta", "ets",
            "seasonal_naive", "drift", "moving_average", "naive",
        ]
        n = len(y)
        val_len = min(max(3, self.all_configs["lags"]), max(3, n // 4))
        if n <= val_len + 4:
            self._candidate_weights = {"theta": 0.35, "ets": 0.35, "seasonal_naive": 0.2, "naive": 0.1}
            return
        train_y = y[:-val_len]
        valid_y = y[-val_len:]
        errors = []
        for name in names:
            try:
                pred = _candidate_forecast(name, train_y, val_len, self._season_length, self.all_configs["lags"])
                err = float(np.mean(np.abs(valid_y[:len(pred)] - pred[:len(valid_y)])))
                if np.isfinite(err):
                    errors.append((name, max(err, 1e-10)))
            except Exception:
                continue
        if not errors:
            self._candidate_weights = {"theta": 0.5, "ets": 0.5}
            return
        errors = sorted(errors, key=lambda x: x[1])[:max(1, self.all_configs["top_k"])]
        inv = np.asarray([1.0 / err for _, err in errors], dtype=np.float64)
        inv = inv / max(float(np.sum(inv)), 1e-12)
        self._candidate_weights = {name: float(w) for (name, _), w in zip(errors, inv)}

    def _forecast_core(self, n):
        weights = self._candidate_weights or {"theta": 0.5, "ets": 0.5}
        preds = []
        ws = []
        for name, weight in weights.items():
            try:
                pred = _candidate_forecast(name, self._y, n, self._season_length, self.all_configs["lags"])
                if len(pred) == int(n) and np.all(np.isfinite(pred)):
                    preds.append(pred)
                    ws.append(float(weight))
            except Exception:
                continue
        if not preds:
            return _forecast_naive(self._y, n)
        weights_arr = np.asarray(ws, dtype=np.float64)
        weights_arr = weights_arr / max(float(np.sum(weights_arr)), 1e-12)
        stacked = np.vstack(preds)
        return _clip_forecast(self._y, weights_arr @ stacked)
