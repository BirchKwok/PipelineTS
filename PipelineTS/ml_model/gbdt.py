import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from spinesUtils.asserts import generate_function_kwargs, ParameterTypeAssert
from spinesUtils.asserts import raise_if_not
from spinesUtils.preprocessing import gc_collector

from PipelineTS.spinesTS.preprocessing import split_series, lag_splits
from PipelineTS.base.base import GBDTModelMixin, IntervalEstimationMixin
from PipelineTS.base.spines_base import SpinesMLModelMixin
from PipelineTS.utils import check_time_col_is_timestamp


class _SklearnCatBoostWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make CatBoostRegressor compatible with sklearn's RegressorChain.

    CatBoostRegressor doesn't implement __sklearn_tags__ required by sklearn >= 1.6.
    This wrapper inherits from BaseEstimator to provide that compatibility.
    """

    def __init__(self, **kwargs):
        self._cb_kwargs = kwargs
        self._model = None

    def fit(self, X, y, **fit_params):
        self._model = CatBoostRegressor(**self._cb_kwargs)
        self._model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_params(self, deep=True):
        return self._cb_kwargs.copy()

    def set_params(self, **params):
        self._cb_kwargs.update(params)
        return self


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

    def _data_preprocess(self, data, mode='train'):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])

        if mode == 'train':
            x, y = split_series(
                data[self.all_configs['target_col']],
                data[self.all_configs['target_col']],
                window_size=self.all_configs['lags'],
                pred_steps=self.all_configs['lags']
            )
            return self._build_lag_features(x), y
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

        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]
        self.last_dt = data[self.all_configs['time_col']].max()

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, mode='train')

        # Store raw lags (without features) for iterative prediction
        raw_lags = lag_splits(data[self.all_configs['target_col']],
                              window_size=self.all_configs['lags'])
        if raw_lags.ndim == 1:
            self._raw_lags = raw_lags.reshape(1, -1)
        else:
            self._raw_lags = raw_lags[-1:, :]
        self.x = self._build_lag_features(self._raw_lags)

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_gbrt(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n, raw_lags=None):
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
                current_res = self.model.predict(x)
                if current_res.ndim == 1:
                    current_res = current_res.reshape(1, -1)
                res.append(current_res.squeeze().tolist()[-1])
            return res

    def predict(self, n, data=None, predict_kwargs=None):
        """
        Predict future values.

        Parameters
        ----------
        n : int
            Number of steps to predict.
        data : pd.DataFrame or None
            Input data for prediction. If None, uses last training data.
        predict_kwargs : ignored, for API compatibility.

        Returns
        -------
        pd.DataFrame
        """
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
            raw_lags = x[:, :self.all_configs['lags']].copy()
            last_dt = data[self.all_configs['time_col']].max()
        else:
            x = self.x
            raw_lags = self._raw_lags.copy()
            last_dt = self.last_dt

        res = self._extend_predict(x, n, raw_lags=raw_lags)
        raise_if_not(ValueError, len(res) == n, 'len(predictions) must == n')

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


class CatBoostModel(_DirectGBDTMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            random_state=None,
            verbose=False,
            iterations=500,
            depth=6,
            learning_rate=0.08,
            l2_leaf_reg=3.0,
            bagging_temperature=0.5,
            subsample=0.8,
            **catboost_configs
    ):
        """
        CatBoostModel using native CatBoost with RegressorChain.

        Parameters
        ----------
        time_col : str
            Time column name.
        target_col : str
            Target column name.
        lags : int, optional, default: 1
            Number of lagged time steps.
        quantile : float or None, optional, default: 0.9
            Quantile for prediction intervals.
        random_state : int or None, optional
            Random seed.
        verbose : bool, optional, default: False
            Verbosity.
        iterations : int, optional, default: 500
            Number of boosting iterations.
        depth : int, optional, default: 6
            Tree depth.
        learning_rate : float, optional, default: 0.08
            Learning rate.
        l2_leaf_reg : float, optional, default: 3.0
            L2 regularization coefficient.
        bagging_temperature : float, optional, default: 0.5
            Bayesian bootstrap bagging temperature.
        subsample : float, optional, default: 0.8
            Subsample ratio of the training data.
        **catboost_configs
            Additional CatBoostRegressor configs.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_state,
            verbose=verbose,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            subsample=subsample,
            **catboost_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0)
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(_SklearnCatBoostWrapper(**self.all_configs['model_configs']))

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['iterations'] = min(100, cv_configs.get('iterations', 500))
        return RegressorChain(_SklearnCatBoostWrapper(**cv_configs))


class LightGBMModel(_DirectGBDTMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            random_state=None,
            verbose=-1,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            linear_tree=False,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            **lgbm_configs
    ):
        """
        LightGBMModel using native LightGBM with RegressorChain.

        Parameters
        ----------
        time_col : str
            Time column name.
        target_col : str
            Target column name.
        lags : int, optional, default: 1
            Number of lagged time steps.
        quantile : float or None, optional, default: 0.9
            Quantile for prediction intervals.
        random_state : int or None, optional
            Random seed.
        verbose : int, optional, default: -1
            Verbosity level.
        n_estimators : int, optional, default: 500
            Number of boosting iterations.
        learning_rate : float, optional, default: 0.05
            Learning rate.
        num_leaves : int, optional, default: 31
            Max number of leaves.
        linear_tree : bool, optional, default: True
            Whether to use linear tree.
        reg_alpha : float, optional, default: 0.1
            L1 regularization term on weights.
        reg_lambda : float, optional, default: 0.1
            L2 regularization term on weights.
        min_child_samples : int, optional, default: 5
            Minimum number of data needed in a child (leaf).
        subsample : float, optional, default: 0.8
            Subsample ratio of the training data.
        colsample_bytree : float, optional, default: 0.8
            Subsample ratio of columns when constructing each tree.
        **lgbm_configs
            Additional LGBMRegressor configs.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            LGBMRegressor,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=verbose,
            linear_tree=linear_tree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            subsample_freq=1,
            **lgbm_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0)
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(LGBMRegressor(**self.all_configs['model_configs']))

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 500))
        return RegressorChain(LGBMRegressor(**cv_configs))


class XGBoostModel(_DirectGBDTMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            random_state=None,
            verbose=0,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            **xgb_configs
    ):
        """
        XGBoostModel using native XGBoost with RegressorChain.

        Parameters
        ----------
        time_col : str
            Time column name.
        target_col : str
            Target column name.
        lags : int, optional, default: 1
            Number of lagged time steps.
        quantile : float or None, optional, default: 0.9
            Quantile for prediction intervals.
        random_state : int or None, optional
            Random seed.
        verbose : int, optional, default: 0
            Verbosity level.
        n_estimators : int, optional, default: 500
            Number of boosting iterations.
        learning_rate : float, optional, default: 0.05
            Learning rate.
        max_depth : int, optional, default: 6
            Max tree depth.
        reg_alpha : float, optional, default: 0.1
            L1 regularization term on weights.
        reg_lambda : float, optional, default: 1.0
            L2 regularization term on weights.
        subsample : float, optional, default: 0.8
            Subsample ratio of the training data.
        colsample_bytree : float, optional, default: 0.8
            Subsample ratio of columns when constructing each tree.
        **xgb_configs
            Additional XGBRegressor configs.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            XGBRegressor,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=verbose,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **xgb_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0)
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(XGBRegressor(**self.all_configs['model_configs']))

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 500))
        return RegressorChain(XGBRegressor(**cv_configs))


class RandomForestModel(_DirectGBDTMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            n_estimators=300,
            quantile=0.9,
            random_state=None,
            min_samples_leaf=2,
            max_features='sqrt',
            **rf_configs
    ):
        """
        RandomForestModel using native sklearn with RegressorChain.

        Parameters
        ----------
        time_col : str
            Time column name.
        target_col : str
            Target column name.
        lags : int, optional, default: 1
            Number of lagged time steps.
        n_estimators : int, optional, default: 300
            Number of trees.
        quantile : float or None, optional, default: 0.9
            Quantile for prediction intervals.
        random_state : int or None, optional
            Random seed.
        min_samples_leaf : int, optional, default: 2
            Minimum number of samples required at a leaf node.
        max_features : str or float, optional, default: 'sqrt'
            Number of features to consider for the best split.
        **rf_configs
            Additional RandomForestRegressor configs.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            RandomForestRegressor,
            n_estimators=n_estimators,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **rf_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0)
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(RandomForestRegressor(**self.all_configs['model_configs']))

    def _define_model_for_cv(self):
        cv_configs = dict(self.all_configs['model_configs'])
        cv_configs['n_estimators'] = min(100, cv_configs.get('n_estimators', 300))
        return RegressorChain(RandomForestRegressor(**cv_configs))
