import warnings
import itertools

import numpy as np
import pandas as pd
from spinesUtils.preprocessing import gc_collector

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp, infer_freq, make_future_dates
from PipelineTS.utils.native_stats import adf_test


def _determine_d(y, max_d=2):
    """Determine differencing order using ADF test."""
    for d in range(max_d + 1):
        series = y.copy()
        for _ in range(d):
            series = np.diff(series)
        if len(series) < 3:
            return d
        try:
            if adf_test(series)['is_stationary']:
                return d
        except Exception:
            return d
    return max_d


def _difference_values(y, d=0):
    z = np.asarray(y, dtype=np.float64)
    for _ in range(int(d)):
        z = np.diff(z)
    return z


def _inverse_difference(history, diffs, d=0):
    history = np.asarray(history, dtype=np.float64)
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    if d <= 0:
        return diffs
    if len(diffs) == 0:
        return diffs
    if d == 1:
        return float(history[-1]) + np.cumsum(diffs)
    last_diff = float(history[-1] - history[-2]) if len(history) >= 2 else 0.0
    first_diff = last_diff + np.cumsum(diffs)
    return float(history[-1]) + np.cumsum(first_diff)


class _NativeARIMAResult:
    def __init__(
        self,
        y,
        order,
        seasonal_order,
        coef,
        feature_spec,
        residual_tail,
        aic,
        exog_mean=None,
    ):
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.order = order
        self.seasonal_order = seasonal_order
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.feature_spec = feature_spec
        self.residual_tail = list(np.asarray(residual_tail, dtype=np.float64).reshape(-1))
        self.aic = float(aic)
        self.exog_mean = None if exog_mean is None else np.asarray(exog_mean, dtype=np.float64).reshape(-1)

    def _row(self, z_hist, residual_hist, exog_row):
        row = [1.0]
        for lag in self.feature_spec['ar_lags']:
            row.append(float(z_hist[-lag]) if len(z_hist) >= lag else 0.0)
        for lag in self.feature_spec['ma_lags']:
            row.append(float(residual_hist[-lag]) if len(residual_hist) >= lag else 0.0)
        if exog_row is not None:
            row.extend(np.asarray(exog_row, dtype=np.float64).reshape(-1).tolist())
        return np.asarray(row, dtype=np.float64)

    def forecast(self, steps, exog=None):
        steps = int(steps)
        p, d, q = self.order
        z_hist = _difference_values(self.y, d=d).astype(np.float64).tolist()
        residual_hist = list(self.residual_tail)
        if exog is not None:
            exog_future = np.asarray(exog, dtype=np.float64)
            if exog_future.ndim == 1:
                exog_future = exog_future.reshape(-1, 1)
        elif self.exog_mean is not None:
            exog_future = np.tile(self.exog_mean.reshape(1, -1), (steps, 1))
        else:
            exog_future = None
        preds_z = []
        for i in range(steps):
            exog_row = None
            if exog_future is not None:
                if i < len(exog_future):
                    exog_row = exog_future[i]
                else:
                    exog_row = self.exog_mean if self.exog_mean is not None else np.zeros(exog_future.shape[1])
            row = self._row(z_hist, residual_hist, exog_row)
            pred = float(row @ self.coef)
            preds_z.append(pred)
            z_hist.append(pred)
            residual_hist.append(0.0)
        return _inverse_difference(self.y, np.asarray(preds_z, dtype=np.float64), d=d)


def _build_lags(p, q, seasonal_order):
    P, D, Q, m = seasonal_order if seasonal_order else (0, 0, 0, 0)
    ar_lags = list(range(1, int(p) + 1))
    ma_lags = list(range(1, int(q) + 1))
    if m and m > 1:
        ar_lags.extend([int(m) * i for i in range(1, int(P) + 1)])
        ma_lags.extend([int(m) * i for i in range(1, int(Q) + 1)])
    ar_lags = sorted(set(lag for lag in ar_lags if lag > 0))
    ma_lags = sorted(set(lag for lag in ma_lags if lag > 0))
    return ar_lags, ma_lags


def _fit_native_arima(y, order, seasonal_order=None, exog=None, ridge=1e-8):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p, d, q = order
    seasonal_order = seasonal_order if seasonal_order else (0, 0, 0, 0)
    z = _difference_values(y, d=d)
    if len(z) < 4:
        coef = np.array([float(np.mean(z)) if len(z) else 0.0])
        return _NativeARIMAResult(y, order, seasonal_order, coef, {'ar_lags': [], 'ma_lags': []}, [], np.inf)
    exog_arr = None
    exog_mean = None
    if exog is not None:
        exog_arr = np.asarray(exog, dtype=np.float64)
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        exog_arr = exog_arr[-len(z):]
        exog_mean = np.nanmean(exog_arr, axis=0)
        exog_arr = np.nan_to_num(exog_arr, nan=0.0, posinf=0.0, neginf=0.0)
    ar_lags, ma_lags = _build_lags(p, q, seasonal_order)
    max_lag = max(ar_lags + ma_lags + [0])
    if len(z) <= max_lag + 2:
        return None

    idx = np.arange(max_lag, len(z), dtype=np.int64)

    def design(residuals=None):
        columns = [np.ones(len(idx), dtype=np.float64)]
        columns.extend(z[idx - lag] for lag in ar_lags)
        if residuals is None:
            columns.extend(np.zeros(len(idx), dtype=np.float64) for _ in ma_lags)
        else:
            columns.extend(residuals[idx - lag] for lag in ma_lags)
        x = np.column_stack(columns)
        if exog_arr is not None:
            x = np.column_stack((x, exog_arr[idx]))
        return x, z[idx]

    x, target = design(None)
    try:
        xtx = x.T @ x + np.eye(x.shape[1]) * ridge
        coef = np.linalg.solve(xtx, x.T @ target)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(x) @ target
    residuals = np.zeros(len(z), dtype=np.float64)
    residuals[max_lag:] = target - x @ coef
    if ma_lags:
        x, target = design(residuals)
        try:
            xtx = x.T @ x + np.eye(x.shape[1]) * ridge
            coef = np.linalg.solve(xtx, x.T @ target)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(x) @ target
        residuals[max_lag:] = target - x @ coef
    rss = float(np.dot(residuals[max_lag:], residuals[max_lag:]))
    n_obs = max(len(target), 1)
    k = len(coef)
    aic = float(n_obs * np.log(max(rss / n_obs, 1e-12)) + 2 * k)
    feature_spec = {'ar_lags': ar_lags, 'ma_lags': ma_lags}
    residual_tail = residuals[-max(max_lag, 1):] if max_lag > 0 else []
    return _NativeARIMAResult(y, order, seasonal_order, coef, feature_spec, residual_tail, aic, exog_mean=exog_mean)


def _fit_arima(y, order, seasonal_order=None, suppress_warnings=True, exog=None):
    """Fit a native ARIMA model and return (model_result, aic) or None on failure."""
    try:
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")
            result = _fit_native_arima(y, order, seasonal_order, exog=exog)
            if result is None:
                return None
            return result, result.aic
    except Exception:
        return None


def _auto_arima_search(y, start_p=0, max_p=5, start_q=0, max_q=5,
                       max_d=2, seasonal=False, m=1,
                       max_P=2, max_Q=2, max_D=1, information_criterion='aic',
                       exog=None):
    """Grid search for best ARIMA/SARIMA order by AIC."""
    d = _determine_d(y, max_d=max_d)

    p_range = range(start_p, max_p + 1)
    q_range = range(start_q, max_q + 1)

    best_result = None
    best_aic = np.inf
    best_order = (0, d, 0)
    best_seasonal = (0, 0, 0, 0)

    if seasonal and m > 1:
        D = 0
        seasonal_combos = list(itertools.product(range(max_P + 1), range(max_Q + 1)))
    else:
        seasonal_combos = [(0, 0)]
        D = 0
        m = 0

    for p, q in itertools.product(p_range, q_range):
        for P, Q in seasonal_combos:
            order = (p, d, q)
            seasonal_order = (P, D, Q, m) if seasonal and m > 1 else (0, 0, 0, 0)

            out = _fit_arima(y, order, seasonal_order, exog=exog)
            if out is not None:
                result, aic = out
                if aic < best_aic:
                    best_aic = aic
                    best_result = result
                    best_order = order
                    best_seasonal = seasonal_order

    # Fallback: if grid search failed entirely, try (1, d, 0)
    if best_result is None:
        out = _fit_arima(y, (1, d, 0), exog=exog)
        if out is not None:
            best_result, best_aic = out
            best_order = (1, d, 0)

    return best_result, best_order, best_seasonal


class AutoARIMAModel(StatisticModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            start_p=0,
            max_p=5,
            start_q=0,
            max_q=5,
            max_d=2,
            seasonal=False,
            quantile=0.9,
            m=12,
            **arima_configs
    ):
        """
        AutoARIMAModel: Auto ARIMA using native conditional least squares with AIC-based grid search.
        No dependency on external ARIMA libraries.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values (kept for API compatibility).
        start_p : int, optional, default: 0
            Starting value for AR order search.
        max_p : int, optional, default: 5
            Maximum AR order to search.
        start_q : int, optional, default: 0
            Starting value for MA order search.
        max_q : int, optional, default: 5
            Maximum MA order to search.
        max_d : int, optional, default: 2
            Maximum differencing order (auto-determined via ADF test).
        seasonal : bool, optional, default: False
            Whether the time series exhibits seasonality.
        m : int, optional, default: 12
            The number of observations per seasonal cycle.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        **arima_configs
            Reserved for future extensions.

        Attributes
        ----------
        model : native ARIMA result
            The fitted native ARIMA model.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': (0, 0),
                'start_p': start_p,
                'max_p': max_p,
                'start_q': start_q,
                'max_q': max_q,
                'max_d': max_d,
                'seasonal': seasonal,
                'm': m,
            }
        )

        self.model = None
        self.last_dt = None
        self._order = None
        self._seasonal_order = None

    def _define_model(self):
        """Not used - model is created during fit."""
        return None

    def _cv_split(self, data, cv=5):
        """Expanding-window CV splits for time series."""
        n = len(data)
        fold_size = max(1, n // (cv + 1))
        for i in range(cv):
            train_end = n - (cv - i) * fold_size
            valid_end = train_end + fold_size
            if train_end < 3 or valid_end > n:
                continue
            yield data.iloc[:train_end], data.iloc[train_end:valid_end]

    @gc_collector()
    def _calculate_confidence_interval(self, data, cv=5):
        """Calculate conformal prediction intervals via expanding-window CV.

        Collects per-point signed residuals (y_true - y_pred) across CV folds,
        then computes asymmetric conformal quantiles with finite-sample correction.
        Reuses the order found during the main fit to avoid redundant grid searches.
        """
        from PipelineTS.base.base import IntervalEstimationMixin

        signed_residuals = []
        target_col = self.all_configs['target_col']

        known_cols = getattr(self, '_known_cov_cols', [])

        for train_data, valid_data in self._cv_split(data, cv=cv):
            valid_y = valid_data[target_col].values
            train_y = train_data[target_col].values.astype(np.float64)
            train_exog = train_data[known_cols].values.astype(np.float64) if known_cols else None
            valid_exog = valid_data[known_cols].values.astype(np.float64) if known_cols else None

            try:
                # Reuse the order from the main fit instead of re-running grid search
                result = _fit_arima(train_y, self._order, self._seasonal_order, exog=train_exog)
                if result is not None:
                    fitted_model, _ = result
                    preds = fitted_model.forecast(steps=len(valid_y), exog=valid_exog)
                    per_point = valid_y.flatten() - preds.flatten()
                    signed_residuals.extend(per_point.tolist())
            except Exception:
                continue

        return IntervalEstimationMixin._compute_conformal_quantiles(
            signed_residuals, coverage=self.all_configs['quantile']
        )

    def fit(self, data, cv=5, **kwargs):
        """
        Fit the AutoARIMA model on the input data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        cv : int, optional, default: 5
            The number of cross-validation folds.

        Returns
        -------
        self
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])
        id_col = self.all_configs.get('id_col')
        known_covs = self.all_configs.get('known_covariates') or []
        self._known_cov_cols = [c for c in known_covs if c in data.columns]

        self._freq = infer_freq(data, self.all_configs['time_col'])

        if id_col is not None and id_col in data.columns:
            # Multi-series: train per-series local ARIMA models
            self._panel_models = {}
            self._panel_last_dt = {}
            self._panel_orders = {}
            for sid, sdf in data.groupby(id_col):
                sdf = sdf.copy()
                sdf[self.all_configs['time_col']] = pd.to_datetime(sdf[self.all_configs['time_col']])
                train_y = sdf[self.all_configs['target_col']].values.astype(np.float64)
                exog = sdf[self._known_cov_cols].values.astype(np.float64) if self._known_cov_cols else None
                model, order, seasonal_order = _auto_arima_search(
                    train_y,
                    start_p=self.all_configs['start_p'],
                    max_p=self.all_configs['max_p'],
                    start_q=self.all_configs['start_q'],
                    max_q=self.all_configs['max_q'],
                    max_d=self.all_configs['max_d'],
                    seasonal=self.all_configs['seasonal'],
                    m=self.all_configs['m'],
                    exog=exog,
                )
                self._panel_models[sid] = model
                self._panel_last_dt[sid] = sdf[self.all_configs['time_col']].max()
                self._panel_orders[sid] = (order, seasonal_order)
            self.last_dt = data[self.all_configs['time_col']].max()
            return self

        keep_cols = [self.all_configs['time_col'], self.all_configs['target_col']] + self._known_cov_cols
        data = data[keep_cols].copy()
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        self.last_dt = data[self.all_configs['time_col']].max()

        train_y = data[self.all_configs['target_col']].values.astype(np.float64)
        exog = data[self._known_cov_cols].values.astype(np.float64) if self._known_cov_cols else None

        self.model, self._order, self._seasonal_order = _auto_arima_search(
            train_y,
            start_p=self.all_configs['start_p'],
            max_p=self.all_configs['max_p'],
            start_q=self.all_configs['start_q'],
            max_q=self.all_configs['max_q'],
            max_d=self.all_configs['max_d'],
            seasonal=self.all_configs['seasonal'],
            m=self.all_configs['m'],
            exog=exog,
        )

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self._calculate_confidence_interval(data, cv=cv)

        return self

    def predict(self, n, future_covariates=None, **kwargs):
        """
        Make predictions using the fitted AutoARIMA model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.
        future_covariates : pd.DataFrame or None
            Future known covariate values for the forecast horizon.

        Returns
        -------
        pd.DataFrame
        """
        id_col = self.all_configs.get('id_col')
        known_cols = getattr(self, '_known_cov_cols', [])

        # Multi-series panel prediction
        if id_col is not None and hasattr(self, '_panel_models') and self._panel_models:
            all_results = []
            for sid, model in self._panel_models.items():
                last_dt = self._panel_last_dt[sid]
                exog_future = None
                if known_cols and future_covariates is not None:
                    if id_col in future_covariates.columns:
                        sid_fc = future_covariates[future_covariates[id_col] == sid]
                    else:
                        sid_fc = future_covariates
                    exog_future = sid_fc[known_cols].values[:n].astype(np.float64)
                    if len(exog_future) < n:
                        pad = np.zeros((n - len(exog_future), len(known_cols)))
                        exog_future = np.vstack([exog_future, pad])
                elif known_cols:
                    exog_future = np.zeros((n, len(known_cols)))
                preds = model.forecast(steps=n, exog=exog_future)
                res = pd.DataFrame({
                    self.all_configs['target_col']: preds
                })
                res[self.all_configs['time_col']] = make_future_dates(last_dt, n, self._freq)
                if self.all_configs['quantile'] is not None:
                    res = self.interval_predict(res)
                res = self.chosen_cols(res)
                res[id_col] = sid
                all_results.append(res)
            return pd.concat(all_results, ignore_index=True)

        # Single-series prediction
        exog_future = None
        if known_cols and future_covariates is not None:
            exog_future = future_covariates[known_cols].values[:n].astype(np.float64)
            if len(exog_future) < n:
                pad = np.zeros((n - len(exog_future), len(known_cols)))
                exog_future = np.vstack([exog_future, pad])
        elif known_cols:
            exog_future = np.zeros((n, len(known_cols)))
        preds = self.model.forecast(steps=n, exog=exog_future)

        res = pd.DataFrame({
            self.all_configs['target_col']: preds
        })
        res[self.all_configs['time_col']] = make_future_dates(self.last_dt, n, self._freq)

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
