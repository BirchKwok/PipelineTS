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

    def _data_preprocess(self, data, mode='train'):
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])

        if mode == 'train':
            x, y = split_series(
                data[self.all_configs['target_col']],
                data[self.all_configs['target_col']],
                window_size=self.all_configs['lags'],
                pred_steps=self.all_configs['lags']
            )
            return x, y
        else:
            return lag_splits(data[self.all_configs['target_col']],
                              window_size=self.all_configs['lags'])

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
        self.x = lag_splits(data[self.all_configs['target_col']],
                            window_size=self.all_configs['lags'])
        if self.x.ndim == 1:
            self.x = self.x.reshape(1, -1)
        else:
            self.x = self.x[-1:, :]

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_gbrt(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n):
        raise_if_not(TypeError, isinstance(n, int), 'n must be int.')
        raise_if_not(ValueError, x.ndim == 2, 'x must be 2D.')

        current_res = self.model.predict(x)
        if current_res.ndim == 1:
            current_res = current_res.reshape(1, -1)

        if n <= current_res.shape[1]:
            return current_res.squeeze().tolist()[:n]
        else:
            res = current_res.squeeze().tolist()
            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
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
            last_dt = data[self.all_configs['time_col']].max()
        else:
            x = self.x
            last_dt = self.last_dt

        res = self._extend_predict(x, n)
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
            learning_rate=0.1,
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
        learning_rate : float, optional, default: 0.1
            Learning rate.
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
            **catboost_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': 0
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(_SklearnCatBoostWrapper(**self.all_configs['model_configs']))


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
            learning_rate=0.1,
            num_leaves=31,
            linear_tree=True,
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
        learning_rate : float, optional, default: 0.1
            Learning rate.
        num_leaves : int, optional, default: 31
            Max number of leaves.
        linear_tree : bool, optional, default: True
            Whether to use linear tree.
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
            **lgbm_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': 0
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(LGBMRegressor(**self.all_configs['model_configs']))


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
            learning_rate=0.1,
            max_depth=6,
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
        learning_rate : float, optional, default: 0.1
            Learning rate.
        max_depth : int, optional, default: 6
            Max tree depth.
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
            **xgb_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': 0
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(XGBRegressor(**self.all_configs['model_configs']))


class RandomForestModel(_DirectGBDTMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            n_estimators=100,
            quantile=0.9,
            random_state=None,
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
        n_estimators : int, optional, default: 100
            Number of trees.
        quantile : float or None, optional, default: 0.9
            Quantile for prediction intervals.
        random_state : int or None, optional
            Random seed.
        **rf_configs
            Additional RandomForestRegressor configs.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            RandomForestRegressor,
            n_estimators=n_estimators,
            random_state=random_state,
            **rf_configs
        )

        self.last_dt = None
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': 0
        })
        self.x = None
        self.model = self._define_model()

    def _define_model(self):
        return RegressorChain(RandomForestRegressor(**self.all_configs['model_configs']))
