import warnings
import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from spinesUtils.preprocessing import gc_collector

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp


def _determine_d(y, max_d=2):
    """Determine differencing order using ADF test."""
    for d in range(max_d + 1):
        series = y.copy()
        for _ in range(d):
            series = np.diff(series)
        if len(series) < 3:
            return d
        try:
            p_value = adfuller(series, autolag='AIC')[1]
            if p_value < 0.05:
                return d
        except Exception:
            return d
    return max_d


def _fit_arima(y, order, seasonal_order=None, suppress_warnings=True):
    """Fit a SARIMAX model and return (model_result, aic) or None on failure."""
    try:
        with warnings.catch_warnings():
            if suppress_warnings:
                warnings.simplefilter("ignore")
            model = SARIMAX(
                y, order=order,
                seasonal_order=seasonal_order if seasonal_order else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False, maxiter=200)
            return result, result.aic
    except Exception:
        return None


def _auto_arima_search(y, start_p=0, max_p=5, start_q=0, max_q=5,
                       max_d=2, seasonal=False, m=1,
                       max_P=2, max_Q=2, max_D=1, information_criterion='aic'):
    """Grid search for best ARIMA/SARIMA order by AIC."""
    d = _determine_d(y, max_d=max_d)

    p_range = range(start_p, max_p + 1)
    q_range = range(start_q, max_q + 1)

    best_result = None
    best_aic = np.inf
    best_order = (0, d, 0)
    best_seasonal = (0, 0, 0, 0)

    if seasonal and m > 1:
        D = min(1, max_D)
        seasonal_combos = list(itertools.product(range(max_P + 1), range(max_Q + 1)))
    else:
        seasonal_combos = [(0, 0)]
        D = 0
        m = 0

    for p, q in itertools.product(p_range, q_range):
        for P, Q in seasonal_combos:
            order = (p, d, q)
            seasonal_order = (P, D, Q, m) if seasonal and m > 1 else (0, 0, 0, 0)

            out = _fit_arima(y, order, seasonal_order)
            if out is not None:
                result, aic = out
                if aic < best_aic:
                    best_aic = aic
                    best_result = result
                    best_order = order
                    best_seasonal = seasonal_order

    # Fallback: if grid search failed entirely, try (1, d, 0)
    if best_result is None:
        out = _fit_arima(y, (1, d, 0))
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
        AutoARIMAModel: Auto ARIMA using statsmodels SARIMAX with AIC-based grid search.
        No dependency on pmdarima or darts.

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
        model : statsmodels SARIMAX result
            The fitted SARIMAX model.
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
        """
        from PipelineTS.base.base import IntervalEstimationMixin

        signed_residuals = []
        target_col = self.all_configs['target_col']

        for train_data, valid_data in self._cv_split(data, cv=cv):
            valid_y = valid_data[target_col].values
            train_y = train_data[target_col].values.astype(np.float64)

            try:
                result, _, _ = _auto_arima_search(
                    train_y,
                    start_p=self.all_configs['start_p'],
                    max_p=self.all_configs['max_p'],
                    start_q=self.all_configs['start_q'],
                    max_q=self.all_configs['max_q'],
                    max_d=self.all_configs['max_d'],
                    seasonal=self.all_configs['seasonal'],
                    m=self.all_configs['m'],
                )
                if result is not None:
                    preds = result.forecast(steps=len(valid_y))
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

        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]].copy()
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        self.last_dt = data[self.all_configs['time_col']].max()

        train_y = data[self.all_configs['target_col']].values.astype(np.float64)

        self.model, self._order, self._seasonal_order = _auto_arima_search(
            train_y,
            start_p=self.all_configs['start_p'],
            max_p=self.all_configs['max_p'],
            start_q=self.all_configs['start_q'],
            max_q=self.all_configs['max_q'],
            max_d=self.all_configs['max_d'],
            seasonal=self.all_configs['seasonal'],
            m=self.all_configs['m'],
        )

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self._calculate_confidence_interval(data, cv=cv)

        return self

    def predict(self, n, **kwargs):
        """
        Make predictions using the fitted AutoARIMA model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.

        Returns
        -------
        pd.DataFrame
        """
        preds = self.model.forecast(steps=n)

        res = pd.DataFrame({
            self.all_configs['target_col']: preds
        })
        res[self.all_configs['time_col']] = \
            self.last_dt + pd.to_timedelta(range(n + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
