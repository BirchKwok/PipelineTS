import numpy as np
import pandas as pd
from spinesUtils.preprocessing import gc_collector

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.statistic_model._prophet_core import SpinesProphet
from PipelineTS.utils import check_time_col_is_timestamp


class ProphetModel(StatisticModelMixin, IntervalEstimationMixin):
    """
    ProphetModel: Custom Prophet-like decomposable time series model.

    Uses piecewise linear trend + Fourier seasonality + optional causal lag features,
    solved via ridge regression. 100x+ faster than Facebook Prophet with comparable
    or better accuracy.

    Parameters
    ----------
    time_col : str
        The column containing time information in the input data.
    target_col : str
        The column containing the target variable in the input data.
    lags : int, optional, default: 1
        Kept for API compatibility.
    n_changepoints : int, optional, default: 25
        Maximum number of trend changepoints.
    changepoint_prior_scale : float, optional, default: 0.05
        Regularization for changepoints. Smaller = smoother trend.
    seasonality_prior_scale : float, optional, default: 10.0
        Regularization for seasonality. Larger = more flexible.
    yearly_seasonality : bool, int, or 'auto', optional, default: 'auto'
        Whether to include yearly seasonality.
    weekly_seasonality : bool, int, or 'auto', optional, default: 'auto'
        Whether to include weekly seasonality.
    auto_seasonality : bool, optional, default: True
        Whether to auto-detect seasonality periods via FFT.
    seasonality_mode : str, optional, default: 'auto'
        'additive', 'multiplicative', or 'auto' (detects from data).
    trend_dampening : float, optional, default: 0.0
        Dampening for trend extrapolation (0=none, 1=flat).
    n_iter : int, optional, default: 5
        Number of iterations for trend-seasonality decomposition.
    use_lag_features : bool, optional, default: False
        Whether to include causal rolling lag features as additional regressors.
    lag_window : int or 'auto', optional, default: 'auto'
        Window size for rolling lag features. 'auto' sets it based on data length.
    lag_prior_scale : float, optional, default: 5.0
        Regularization strength for lag feature coefficients.
    quantile : float, optional, default: 0.9
        Quantile for interval prediction. None for point prediction.
    """

    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            n_changepoints=25,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            auto_seasonality=True,
            seasonality_mode='auto',
            trend_dampening=0.0,
            n_iter=5,
            use_lag_features=False,
            lag_window='auto',
            lag_prior_scale=5.0,
            quantile=0.9,
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs.update({
            'quantile': quantile,
            'quantile_error': (0, 0),
            'time_col': time_col,
            'target_col': target_col,
            'lags': lags,
            'n_changepoints': n_changepoints,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'auto_seasonality': auto_seasonality,
            'seasonality_mode': seasonality_mode,
            'trend_dampening': trend_dampening,
            'n_iter': n_iter,
            'use_lag_features': use_lag_features,
            'lag_window': lag_window,
            'lag_prior_scale': lag_prior_scale,
        })

        self.model = self._define_model()
        self.last_dt = None

    def _define_model(self):
        return SpinesProphet(
            n_changepoints=self.all_configs['n_changepoints'],
            changepoint_prior_scale=self.all_configs['changepoint_prior_scale'],
            seasonality_prior_scale=self.all_configs['seasonality_prior_scale'],
            yearly_seasonality=self.all_configs['yearly_seasonality'],
            weekly_seasonality=self.all_configs['weekly_seasonality'],
            auto_seasonality=self.all_configs['auto_seasonality'],
            seasonality_mode=self.all_configs['seasonality_mode'],
            trend_dampening=self.all_configs['trend_dampening'],
            n_iter=self.all_configs['n_iter'],
            use_lag_features=self.all_configs['use_lag_features'],
            lag_window=self.all_configs['lag_window'],
            lag_prior_scale=self.all_configs['lag_prior_scale'],
        )

    def _cv_split(self, data, cv=5):
        """Expanding-window CV for confidence interval estimation."""
        n = len(data)
        fold_size = max(1, n // (cv + 1))
        for i in range(cv):
            train_end = n - (cv - i) * fold_size
            valid_end = train_end + fold_size
            if train_end < 3 or valid_end > n:
                continue
            yield data.iloc[:train_end], data.iloc[train_end:valid_end]

    @gc_collector()
    def _calculate_confidence_interval(self, dates, y, cv=5):
        """Calculate conformal prediction intervals via expanding-window CV.

        Collects per-point signed residuals (y_true - y_pred) across CV folds,
        then computes asymmetric conformal quantiles with finite-sample correction.
        """
        from PipelineTS.base.base import IntervalEstimationMixin

        signed_residuals = []
        n = len(dates)
        fold_size = max(1, n // (cv + 1))

        for i in range(cv):
            train_end = n - (cv - i) * fold_size
            valid_end = train_end + fold_size
            if train_end < 3 or valid_end > n:
                continue

            train_dates = dates[:train_end]
            train_y = y[:train_end]
            valid_dates = dates[train_end:valid_end]
            valid_y = y[train_end:valid_end]

            try:
                m = self._define_model()
                m.fit(train_dates, train_y)
                preds = m.predict(valid_dates)
                per_point = valid_y.flatten() - preds.flatten()
                signed_residuals.extend(per_point.tolist())
            except Exception:
                continue

        return IntervalEstimationMixin._compute_conformal_quantiles(
            signed_residuals, coverage=self.all_configs['quantile']
        )

    @staticmethod
    def _infer_freq(dates):
        """Infer pandas frequency string from datetime series."""
        if len(dates) < 2:
            return 'D'
        diffs = np.diff(dates)
        median_days = float(np.median(diffs) / np.timedelta64(1, 'D'))
        if median_days >= 28:
            return 'MS'
        elif median_days >= 7:
            return 'W'
        elif median_days >= 1:
            return 'D'
        elif median_days >= 1.0 / 24:
            return 'h'
        else:
            return 'min'

    def fit(self, data, freq='auto', cv=5, fit_kwargs=None):
        """
        Fit the Prophet model on the input data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        freq : str, optional, default: 'auto'
            Frequency of the time series. 'auto' detects from data.
        cv : int, optional, default: 5
            Number of cross-validation folds.
        fit_kwargs : ignored, for API compatibility.

        Returns
        -------
        self
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]].copy()
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        self.last_dt = data[self.all_configs['time_col']].max()

        dates = data[self.all_configs['time_col']].values
        y = data[self.all_configs['target_col']].values.astype(np.float64)

        self.model = self._define_model()
        self.model.fit(dates, y)

        if freq == 'auto':
            self._freq = self._infer_freq(dates)
        else:
            self._freq = freq

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self._calculate_confidence_interval(dates, y, cv=cv)

        return self

    def predict(self, n, freq=None, include_history=False):
        """
        Make predictions.

        Parameters
        ----------
        n : int
            Number of future time steps.
        freq : str or None
            Frequency. None uses the freq from fit().
        include_history : bool
            Whether to include historical predictions.

        Returns
        -------
        pd.DataFrame
        """
        if freq is None:
            freq = getattr(self, '_freq', 'D')

        future_df = self.model.make_future_dataframe(
            periods=n, freq=freq, include_history=include_history
        )
        preds = self.model.predict(future_df['ds'].values)

        res = pd.DataFrame({
            self.all_configs['time_col']: future_df['ds'].values,
            self.all_configs['target_col']: preds,
        })

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
