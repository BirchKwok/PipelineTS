import numpy as np
import pandas as pd
from spinesUtils.asserts import ParameterTypeAssert
from spinesUtils.asserts import raise_if_not
from spinesUtils.preprocessing import gc_collector

from PipelineTS.spinesTS.preprocessing import split_series, lag_splits, split_series_panel, lag_splits_panel
from PipelineTS.base.base import GBDTModelMixin, IntervalEstimationMixin
from PipelineTS.base.spines_base import SpinesMLModelMixin
from PipelineTS.utils import check_time_col_is_timestamp


class _DirectGBDTMixin(GBDTModelMixin, IntervalEstimationMixin, SpinesMLModelMixin):
    """Base mixin for direct GBDT forecasting using lag features.

    Uses native ML libraries directly with RegressorChain for multi-step output.
    """

    @staticmethod
    def _row_autocorr(x, lag):
        """Vectorized lag-k autocorrelation for each row."""
        eps = 1e-12
        n = x.shape[1]
        if lag >= n:
            return np.zeros((x.shape[0], 1))
        x1 = x[:, :n - lag]
        x2 = x[:, lag:]
        m1 = x1.mean(axis=1, keepdims=True)
        m2 = x2.mean(axis=1, keepdims=True)
        num = ((x1 - m1) * (x2 - m2)).mean(axis=1, keepdims=True)
        denom = x1.std(axis=1, keepdims=True) * x2.std(axis=1, keepdims=True) + eps
        return num / denom

    @staticmethod
    def _build_lag_features(x):
        """Build statistical features from lag windows to enrich the feature set.

        All features are computed strictly per-row within each lag window,
        ensuring zero data leakage across samples.

        Parameters
        ----------
        x : np.ndarray, shape (N, lags)
            Raw lag windows.

        Returns
        -------
        np.ndarray, shape (N, lags + n_features)
            Concatenation of raw lags and computed statistical features.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        eps = 1e-12
        n_cols = x.shape[1]
        half = max(1, n_cols // 2)

        # Basic statistics
        mean_v = x.mean(axis=1, keepdims=True)
        std_v = x.std(axis=1, keepdims=True)
        min_v = x.min(axis=1, keepdims=True)
        max_v = x.max(axis=1, keepdims=True)
        p25 = np.percentile(x, 25, axis=1, keepdims=True)
        p75 = np.percentile(x, 75, axis=1, keepdims=True)

        # Distribution shape (vectorized numpy, ~10x faster than scipy)
        x_centered = x - mean_v
        m2 = (x_centered ** 2).mean(axis=1, keepdims=True)
        m3 = (x_centered ** 3).mean(axis=1, keepdims=True)
        m4 = (x_centered ** 4).mean(axis=1, keepdims=True)
        skewness = m3 / (np.power(m2, 1.5) + eps)
        kurt = m4 / (m2 ** 2 + eps) - 3.0
        cv = std_v / (np.abs(mean_v) + eps)

        # Range / spread
        iqr = p75 - p25
        full_range = max_v - min_v

        # Diff features
        diffs = np.diff(x, n=1, axis=1)
        avg_diff = diffs.mean(axis=1, keepdims=True)
        std_diff = diffs.std(axis=1, keepdims=True)

        # Trend slope (linear regression coefficient)
        t = np.arange(n_cols, dtype=np.float64)
        t_centered = t - t.mean()
        t_var = (t_centered ** 2).sum()
        x_centered = x - mean_v
        trend_slope = (x_centered @ t_centered).reshape(-1, 1) / (t_var + eps)

        # Autocorrelation lag-1 and lag-2
        autocorr1 = _DirectGBDTMixin._row_autocorr(x, 1)
        autocorr2 = _DirectGBDTMixin._row_autocorr(x, 2)

        # Ratio features
        last_to_mean = x[:, -1:] / (np.abs(mean_v) + eps)
        last_to_first = x[:, -1:] / (np.abs(x[:, :1]) + eps)
        energy = (x ** 2).mean(axis=1, keepdims=True)
        rms = np.sqrt(energy)

        # Sub-window comparison (second-half vs first-half) — captures regime change
        first_half_mean = x[:, :half].mean(axis=1, keepdims=True)
        second_half_mean = x[:, half:].mean(axis=1, keepdims=True)
        half_ratio = second_half_mean / (np.abs(first_half_mean) + eps)

        # EMA (exponential moving average with span ~n_cols/2)
        alpha = 2.0 / (max(1, n_cols // 2) + 1)
        weights = np.power(1 - alpha, np.arange(n_cols - 1, -1, -1, dtype=np.float64))
        weights /= weights.sum() + eps
        ema = (x * weights[np.newaxis, :]).sum(axis=1, keepdims=True)

        # Position features (argmax / argmin within window, normalized)
        argmax_pos = np.argmax(x, axis=1).reshape(-1, 1).astype(np.float64) / max(1, n_cols - 1)
        argmin_pos = np.argmin(x, axis=1).reshape(-1, 1).astype(np.float64) / max(1, n_cols - 1)

        # Sign-change count
        if diffs.shape[1] > 1:
            sign_changes = (np.diff(np.sign(diffs), axis=1) != 0).sum(
                axis=1, keepdims=True).astype(np.float64)
        else:
            sign_changes = np.zeros((x.shape[0], 1))

        # Mean-crossing count
        mean_crossing = (np.diff(np.sign(x - mean_v), axis=1) != 0).sum(
            axis=1, keepdims=True).astype(np.float64)

        feat = np.concatenate(
            (mean_v, std_v, min_v, max_v, p25, p75,
             skewness, kurt, cv, iqr, full_range,
             avg_diff, std_diff, trend_slope, autocorr1, autocorr2,
             last_to_mean, last_to_first, energy, rms,
             half_ratio, ema, argmax_pos, argmin_pos,
             sign_changes, mean_crossing), axis=1)

        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.concatenate((x, feat), axis=1)
        np.clip(result, -3.4e38, 3.4e38, out=result)
        return result.astype(np.float32)

    def _build_covariate_features(self, data, x_target, y_target, mode='train',
                                    id_col=None, group_ids=None):
        """Build covariate features aligned with target sliding windows.

        Parameters
        ----------
        data : pd.DataFrame
            Full data with covariate columns.
        x_target : np.ndarray, shape (N, lags)
            Target lag windows (used only for shape reference).
        y_target : np.ndarray or None
            Target prediction windows (used only for shape reference).
        mode : str
            'train' or 'predict'.
        id_col : str or None
            Series identifier column.
        group_ids : np.ndarray or None
            Group IDs array for panel mode.

        Returns
        -------
        np.ndarray or None
            Covariate features to concatenate, or None if no covariates.
        """
        known_covs = self.all_configs.get('known_covariates') or []
        past_covs = self.all_configs.get('past_covariates') or []

        if not known_covs and not past_covs:
            return None

        lags = self.all_configs['lags']
        parts = []

        if mode == 'train':
            for cov_col in known_covs:
                if cov_col not in data.columns:
                    continue
                if id_col is not None and group_ids is not None:
                    # Panel: extract horizon values per series
                    _, cov_horizon, _ = split_series_panel(
                        data[self.all_configs['target_col']].values,
                        data[cov_col].values,
                        group_ids,
                        window_size=lags, pred_steps=lags
                    )
                else:
                    _, cov_horizon = split_series(
                        data[self.all_configs['target_col']].values,
                        data[cov_col].values,
                        window_size=lags, pred_steps=lags
                    )
                parts.append(cov_horizon)

            for cov_col in past_covs:
                if cov_col not in data.columns:
                    continue
                if id_col is not None and group_ids is not None:
                    cov_lags, _, _ = split_series_panel(
                        data[cov_col].values,
                        data[self.all_configs['target_col']].values,
                        group_ids,
                        window_size=lags, pred_steps=lags
                    )
                else:
                    cov_lags, _ = split_series(
                        data[cov_col].values,
                        data[self.all_configs['target_col']].values,
                        window_size=lags, pred_steps=lags
                    )
                parts.append(self._build_lag_features(cov_lags))

        if not parts:
            return None
        return np.concatenate(parts, axis=1)

    def _data_preprocess(self, data, mode='train'):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        id_col = self.all_configs.get('id_col')

        if id_col is not None and id_col in data.columns:
            # Multi-series: split per-series to avoid cross-series leakage
            if mode == 'train':
                x, y, _ = split_series_panel(
                    data[self.all_configs['target_col']],
                    data[self.all_configs['target_col']],
                    data[id_col],
                    window_size=self.all_configs['lags'],
                    pred_steps=self.all_configs['lags']
                )
                x_feat = self._build_lag_features(x)
                cov_feat = self._build_covariate_features(
                    data, x, y, mode='train',
                    id_col=id_col, group_ids=data[id_col].values
                )
                if cov_feat is not None:
                    x_feat = np.concatenate([x_feat, cov_feat], axis=1)
                return x_feat, y
            else:
                # Return per-series last windows
                return lag_splits_panel(
                    data[self.all_configs['target_col']],
                    data[id_col],
                    window_size=self.all_configs['lags']
                )
        else:
            if mode == 'train':
                x, y = split_series(
                    data[self.all_configs['target_col']],
                    data[self.all_configs['target_col']],
                    window_size=self.all_configs['lags'],
                    pred_steps=self.all_configs['lags']
                )
                x_feat = self._build_lag_features(x)
                cov_feat = self._build_covariate_features(
                    data, x, y, mode='train'
                )
                if cov_feat is not None:
                    x_feat = np.concatenate([x_feat, cov_feat], axis=1)
                return x_feat, y
            else:
                raw = lag_splits(data[self.all_configs['target_col']],
                                 window_size=self.all_configs['lags'])
                return self._build_lag_features(raw)

    @gc_collector()
    def fit(self, data, cv=5, fit_kwargs=None, valid_data=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.
        cv : int, optional
            The number of cross-validation folds. Default is 5.
        fit_kwargs : dict or None, optional
            Additional keyword arguments for fitting the model.
        valid_data : ignored, for API compatibility.

        Returns
        -------
        self
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])
        id_col = self.all_configs.get('id_col')
        known_covs = self.all_configs.get('known_covariates') or []
        past_covs = self.all_configs.get('past_covariates') or []

        keep_cols = [self.all_configs['time_col'], self.all_configs['target_col']]
        if id_col is not None and id_col in data.columns:
            keep_cols.append(id_col)
        for c in known_covs + past_covs:
            if c in data.columns and c not in keep_cols:
                keep_cols.append(c)
        data = data[keep_cols]

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, mode='train')

        # Store last covariate windows for prediction
        lags = self.all_configs['lags']
        self._known_cov_cols = [c for c in known_covs if c in data.columns]
        self._past_cov_cols = [c for c in past_covs if c in data.columns]
        self._panel_past_cov_lags = {}  # {sid: {col: array(1, lags)}}
        self._past_cov_lags = {}  # {col: array(1, lags)} for single-series

        if id_col is not None and id_col in data.columns:
            # Multi-series: store per-series last windows and last datetimes
            self._panel_raw_lags = {}
            self._panel_last_dt = {}
            for sid, sdf in data.groupby(id_col):
                sdf = sdf.sort_values(self.all_configs['time_col'])
                vals = sdf[self.all_configs['target_col']].values
                if len(vals) >= lags:
                    self._panel_raw_lags[sid] = vals[-lags:].reshape(1, -1)
                    self._panel_last_dt[sid] = sdf[self.all_configs['time_col']].max()
                    # Store past covariate lags per series
                    pcl = {}
                    for c in self._past_cov_cols:
                        pcl[c] = sdf[c].values[-lags:].reshape(1, -1)
                    self._panel_past_cov_lags[sid] = pcl
            # Default single-series fallback for compatibility
            self.last_dt = data[self.all_configs['time_col']].max()
            first_sid = list(self._panel_raw_lags.keys())[0]
            self._raw_lags = self._panel_raw_lags[first_sid]
            self.x = self._build_lag_features(self._raw_lags)
        else:
            self.last_dt = data[self.all_configs['time_col']].max()
            # Store raw lags (without features) for iterative prediction
            raw_lags = lag_splits(data[self.all_configs['target_col']],
                                  window_size=lags)
            if raw_lags.ndim == 1:
                self._raw_lags = raw_lags.reshape(1, -1)
            else:
                self._raw_lags = raw_lags[-1:, :]
            self.x = self._build_lag_features(self._raw_lags)
            # Store past covariate lags for single-series
            for c in self._past_cov_cols:
                vals = data[c].values
                self._past_cov_lags[c] = vals[-lags:].reshape(1, -1)

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_gbrt(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n, raw_lags=None,
                         known_cov_future=None, past_cov_features=None):
        """Autoregressive multi-step prediction with optional covariate features.

        Parameters
        ----------
        x : np.ndarray, shape (1, n_features)
            Initial feature vector.
        n : int
            Number of steps to predict.
        raw_lags : np.ndarray or None
            Raw target lag window for feature rebuilding.
        known_cov_future : dict or None
            {col_name: np.ndarray of shape (n,)} with future known covariate values.
        past_cov_features : np.ndarray or None
            Precomputed past covariate features (constant during prediction).
        """
        raise_if_not(TypeError, isinstance(n, int), 'n must be int.')
        raise_if_not(ValueError, x.ndim == 2, 'x must be 2D.')

        lags = self.all_configs['lags']

        # Keep track of raw lags separately for feature rebuilding
        if raw_lags is None:
            raw_lags = x[:, :lags].copy()

        current_res = self.model.predict(x)
        if current_res.ndim == 1:
            current_res = current_res.reshape(1, -1)

        if n <= current_res.shape[1]:
            return current_res.squeeze().tolist()[:n]
        else:
            res = current_res.squeeze().tolist()
            for i in range(n - lags):
                # Shift raw lags and append newest prediction
                raw_lags = np.concatenate((raw_lags[:, 1:], current_res[:, 0:1]), axis=1)
                # Rebuild features from updated raw lags
                x = self._build_lag_features(raw_lags)
                # Append covariate features for this step
                cov_parts = []
                known_cols = getattr(self, '_known_cov_cols', [])
                past_cols = getattr(self, '_past_cov_cols', [])
                if known_cols:
                    step = lags + i  # current prediction position
                    for col in known_cols:
                        if known_cov_future is not None and col in known_cov_future:
                            vals = known_cov_future[col]
                            start = min(step, len(vals) - 1)
                            end = min(step + lags, len(vals))
                            window = vals[start:end]
                            if len(window) < lags:
                                window = np.pad(window, (0, lags - len(window)),
                                                mode='edge')
                            cov_parts.append(window.reshape(1, -1))
                        else:
                            cov_parts.append(np.zeros((1, lags)))
                if past_cols:
                    if past_cov_features is not None:
                        cov_parts.append(past_cov_features)
                    else:
                        cov_parts.append(np.zeros((1, len(past_cols) * (lags + 26))))
                if cov_parts:
                    x = np.concatenate([x] + cov_parts, axis=1)
                current_res = self.model.predict(x)
                if current_res.ndim == 1:
                    current_res = current_res.reshape(1, -1)
                res.append(current_res.squeeze().tolist()[-1])
            return res

    def _build_initial_cov_features(self, known_cov_future=None, past_cov_lags_dict=None):
        """Build covariate features for the initial prediction step.

        Always produces features matching training dimensions when covariates
        were used during fit. Uses zeros as placeholder when future values
        are not provided.

        Returns (initial_cov_features, past_cov_features) where:
        - initial_cov_features: features to append to x for the first prediction
        - past_cov_features: constant past covariate features for all steps
        """
        lags = self.all_configs['lags']
        known_cols = getattr(self, '_known_cov_cols', [])
        past_cols = getattr(self, '_past_cov_cols', [])

        if not known_cols and not past_cols:
            return None, None

        parts = []

        # Known covariates: horizon values (lags per column)
        for col in known_cols:
            if known_cov_future is not None and col in known_cov_future:
                vals = known_cov_future[col][:lags]
                if len(vals) < lags:
                    vals = np.pad(vals, (0, lags - len(vals)), mode='edge')
                parts.append(vals.reshape(1, -1).astype(np.float64))
            else:
                # Placeholder zeros to maintain feature count
                parts.append(np.zeros((1, lags)))

        # Past covariates: lag features from stored windows
        past_feat = None
        past_parts = []
        for col in past_cols:
            if past_cov_lags_dict and col in past_cov_lags_dict:
                past_parts.append(self._build_lag_features(past_cov_lags_dict[col]))
            else:
                # Placeholder: lags raw + 26 stat features = lags + 26
                past_parts.append(np.zeros((1, lags + 26)))
        if past_parts:
            past_feat = np.concatenate(past_parts, axis=1)
            parts.append(past_feat)

        if not parts:
            return None, None
        return np.concatenate(parts, axis=1), past_feat

    def predict(self, n, data=None, predict_kwargs=None, future_covariates=None):
        """
        Predict future values.

        Parameters
        ----------
        n : int
            Number of steps to predict.
        data : pd.DataFrame or None
            Input data for prediction. If None, uses last training data.
        predict_kwargs : ignored, for API compatibility.
        future_covariates : pd.DataFrame or None
            Future known covariate values for the forecast horizon.
            Must have at least n rows and columns matching known_covariates.

        Returns
        -------
        pd.DataFrame
        """
        id_col = self.all_configs.get('id_col')
        lags = self.all_configs['lags']

        # Parse future covariates into {col: np.array}
        known_cov_future = None
        if future_covariates is not None:
            known_cov_future = {}
            for col in getattr(self, '_known_cov_cols', []):
                if col in future_covariates.columns:
                    known_cov_future[col] = future_covariates[col].values.astype(np.float64)

        # Multi-series panel prediction
        if id_col is not None and hasattr(self, '_panel_raw_lags') and self._panel_raw_lags:
            if data is not None:
                check_time_col_is_timestamp(data, self.all_configs['time_col'])
                panel_windows = self._data_preprocess(data, mode='predict')
            else:
                panel_windows = self._panel_raw_lags

            all_results = []
            for sid, raw_window in panel_windows.items():
                raw_lags = raw_window.copy()
                x = self._build_lag_features(raw_lags)

                # Per-series covariate features
                sid_known = None
                if known_cov_future is not None and future_covariates is not None and id_col in future_covariates.columns:
                    sid_fc = future_covariates[future_covariates[id_col] == sid]
                    sid_known = {col: sid_fc[col].values.astype(np.float64)
                                 for col in self._known_cov_cols if col in sid_fc.columns}
                elif known_cov_future is not None:
                    sid_known = known_cov_future

                past_cov_dict = self._panel_past_cov_lags.get(sid, {})
                init_cov, past_feat = self._build_initial_cov_features(
                    sid_known, past_cov_dict
                )
                if init_cov is not None:
                    x = np.concatenate([x, init_cov], axis=1)

                preds = self._extend_predict(
                    x, n, raw_lags=raw_lags,
                    known_cov_future=sid_known, past_cov_features=past_feat
                )
                last_dt = self._panel_last_dt.get(sid, self.last_dt)
                if data is not None:
                    sdf = data[data[id_col] == sid]
                    if len(sdf) > 0:
                        last_dt = sdf[self.all_configs['time_col']].max()

                res = pd.DataFrame(preds, columns=[self.all_configs['target_col']])
                res[self.all_configs['time_col']] = \
                    last_dt + pd.to_timedelta(range(n + 1), unit='D')[1:]
                if self.all_configs['quantile'] is not None:
                    res = self.interval_predict(res)
                res = self.chosen_cols(res)
                res[id_col] = sid
                all_results.append(res)

            return pd.concat(all_results, ignore_index=True)

        # Single-series prediction (original behavior)
        if data is not None:
            check_time_col_is_timestamp(data, self.all_configs['time_col'])
            raise_if_not(
                ValueError, len(data) >= self.all_configs['lags'],
                'The length of the series must be >= lags.'
            )
            x = self._data_preprocess(
                data[[self.all_configs['time_col'], self.all_configs['target_col']]],
                mode='predict'
            )
            if x.ndim == 1:
                x = x.reshape(1, -1)
            else:
                x = x[-1:, :]
            # Extract raw lags (first `lags` columns) for iterative prediction
            raw_lags = x[:, :lags].copy()
            last_dt = data[self.all_configs['time_col']].max()
        else:
            x = self.x.copy()
            raw_lags = self._raw_lags.copy()
            last_dt = self.last_dt

        # Build covariate features for initial step
        init_cov, past_feat = self._build_initial_cov_features(
            known_cov_future, self._past_cov_lags if hasattr(self, '_past_cov_lags') else {}
        )
        if init_cov is not None:
            x = np.concatenate([x, init_cov], axis=1)

        res = self._extend_predict(
            x, n, raw_lags=raw_lags,
            known_cov_future=known_cov_future, past_cov_features=past_feat
        )
        raise_if_not(ValueError, len(res) == n, 'len(predictions) must == n')

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


