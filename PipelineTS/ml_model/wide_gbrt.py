from copy import deepcopy
import numpy as np
import pandas as pd
import re
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from PipelineTS.spinesTS.ml_model import GBRTPreprocessing
from sklearn.multioutput import RegressorChain
from spinesUtils.asserts import generate_function_kwargs, ParameterValuesAssert
from spinesUtils.asserts import raise_if_not
from spinesUtils.preprocessing import gc_collector

from PipelineTS.base.base import GBDTModelMixin, IntervalEstimationMixin
from PipelineTS.base.spines_base import SpinesMLModelMixin
from PipelineTS.utils import update_dict_without_conflict, check_time_col_is_timestamp


class WideGBRTModel(GBDTModelMixin, IntervalEstimationMixin, SpinesMLModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            n_estimators=100,
            quantile=0.9,
            random_state=None,
            differential_n=1,
            moving_avg_n=2,
            extend_daily_target_features=True,
            estimator=LGBMRegressor,
            **model_init_configs
    ):
        """
        Wide Gradient Boosting Regression Trees (GBRT) Model.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values to use as input features for training and prediction.
        n_estimators : int, optional, default: 100
            The number of boosting rounds (trees) in the GBRT model.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        random_state : int or None, optional, default: None
            The random seed for reproducibility.
        differential_n : int, optional, default: 0
            The number of differencing operations to apply to the target variable.
        moving_avg_n : int, optional, default: 0
            The window size for the moving average operation on the target variable.
        extend_daily_target_features : bool, optional, default: True
            Whether to extend the features with daily target-related features.
        estimator : sklearn.base.BaseEstimator, optional, default: LGBMRegressor
            The base estimator used for the GBRT model.
        **model_init_configs : dict
            Additional keyword arguments for configuring the base estimator.

        Attributes
        ----------
        estimator : BaseEstimator
            The base estimator for the GBRT model.
        processor : spinesTS.ml_model.GBRTPreprocessing
            The preprocessor for transforming input data.
        model : sklearn.multioutput.RegressorChain
            The GBRT model wrapped in a regressor chain.
        x : pandas.DataFrame or None
            The last input features used for training.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            estimator,
            n_estimators=n_estimators,
            random_state=random_state,
            **model_init_configs
        )

        if 'LGBMRegressor' in re.split("<|>|class| |\.|\'", str(estimator)):
            self.all_configs['model_configs'] = update_dict_without_conflict(self.all_configs['model_configs'], {
                'verbose': -1
            })

        self._estimator = estimator

        self.last_dt = None
        self.last_lags_dataframe = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0,
                'differential_n': differential_n,
                'moving_avg_n': moving_avg_n,
                'extend_daily_target_features': extend_daily_target_features,
                'built_in_scaler': MinMaxScaler()
            }
        )

        self.processor = GBRTPreprocessing(
            in_features=self.all_configs['lags'],
            out_features=self.all_configs['lags'],
            target_col=self.all_configs['target_col'],
            date_col=self.all_configs['time_col'],
            differential_n=self.all_configs['differential_n'],
            moving_avg_n=self.all_configs['moving_avg_n'],
            extend_daily_target_features=self.all_configs['extend_daily_target_features'],
            train_size=None
        )

        self.x = None

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the GBRT model using a regressor chain.

        Returns
        -------
        sklearn.multioutput.RegressorChain
            The GBRT model wrapped in a regressor chain.
        """
        return RegressorChain(self._estimator(**self.all_configs['model_configs']))

    @ParameterValuesAssert({
        'mode': ('train', 'predict')
    })
    def _data_preprocess(self, data, mode='train', update_last_dt=False):
        """
        Preprocess the input data for training or prediction.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data.
        mode : {'train', 'predict'}, optional, default: 'train'
            The mode indicating whether to preprocess data for training or prediction.
        update_last_dt : bool, optional, default: False
            Whether to update the last timestamp attribute.

        Returns
        -------
        tuple or numpy.ndarray
            If 'mode' is 'train', returns a tuple (x_train, y_train).
            If 'mode' is 'predict', returns lagged splits of the target column.
        """
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        if update_last_dt:
            self.last_dt = data[self.all_configs['time_col']].max()

        if mode == 'train':
            self.processor.fit(data)

            x, y = self.processor.transform(data, mode=mode)  # X, y
            return x.astype(np.float32), y.astype(np.float32)
        else:
            # x
            return self.processor.transform(data, mode=mode).astype(np.float32)

    @gc_collector()
    def fit(self, data, cv=5, fit_kwargs=None):
        """
        Fit the GBRT model to the training data.

        Parameters
        ----------
        data : pandas.DataFrame
            The training data.
        cv : int, optional, default: 5
            The number of cross-validation folds.
        fit_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the fitting process.

        Returns
        -------
        self
            Returns an instance of the fitted model.
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        # data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        self.last_lags_dataframe = data.iloc[-self.all_configs['lags']:, :]

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, 'train', update_last_dt=True)
        x = pd.DataFrame(x)

        self.x = pd.DataFrame(
            self._data_preprocess(data, 'predict')
        ).iloc[-1:, :]

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_gbrt(data, fit_kwargs=fit_kwargs, cv=cv)

        del x, y

        return self

    def _extend_predict(self, x, n):
        """
        Extrapolation prediction.

        Parameters
        ----------
        x : numpy.ndarray
            To-predict data with 2 dimensions.
        n : int
            Number of prediction steps.

        Returns
        -------
        list
            List of predicted values.
        """
        raise_if_not(TypeError, isinstance(x, np.ndarray), 'The input data must be a numpy.ndarray.')
        raise_if_not(ValueError, np.ndim(x) == 2, 'The input data must have 2 dimensions.')
        raise_if_not(TypeError, isinstance(n, int), 'The number of steps to predict must be an integer.')

        current_res = self.model.predict(x)  # np.ndarray
        if current_res.ndim == 1:
            current_res = current_res.reshape((1, -1))

        if n <= current_res.shape[1]:
            return current_res.squeeze().tolist()[:n]
        else:
            res = current_res.squeeze().tolist()
            last_data = self.last_lags_dataframe.copy()

            last_data[self.all_configs['time_col']] = pd.to_datetime(last_data[self.all_configs['time_col']])

            last_dt = deepcopy(self.last_dt)
            for i in range(n - self.all_configs['lags']):
                tmp_data = pd.DataFrame(columns=last_data.columns)
                tmp_data[self.all_configs['time_col']] = (last_dt +
                                                          pd.to_timedelta(range(self.all_configs['lags'] + 1),
                                                                          unit='D'))[1:]

                tmp_data[self.all_configs['target_col']] = res[-self.all_configs['lags']:]
                last_data = pd.concat((last_data.iloc[1:, :], tmp_data.iloc[:1, :]), axis=0)
                last_data = last_data.interpolate(
                    method='linear', limit_direction='forward', axis=0)

                last_dt = last_data[self.all_configs['time_col']].max()

                to_predict_x = pd.DataFrame(
                    self._data_preprocess(last_data, 'predict')
                ).iloc[-1:, :]

                current_res = self.model.predict(to_predict_x).squeeze()
                res.append(current_res[0])

            return res

    def predict(self, n, data=None):
        """
        Predict future values using the trained GBRT model.

        Parameters
        ----------
        n : int
            Number of steps to predict into the future.
        data : pandas.DataFrame or None, optional, default: None
            Additional data for prediction. If provided, the length of the series must be greater than or equal to lags.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the predicted values and corresponding timestamps.
        """
        if data is not None:
            check_time_col_is_timestamp(data, self.all_configs['time_col'])

            raise_if_not(
                ValueError, len(data) >= self.all_configs['lags'],
                'The length of the series must be greater than or equal to the lags.'
            )

            x = self._data_preprocess(data.iloc[-self.all_configs['lags']:, :], 'predict', update_last_dt=False)
            last_dt = data[self.all_configs['time_col']].max()
        else:
            x = self.x.values
            last_dt = self.last_dt

        res = self._extend_predict(x, n)  # list

        raise_if_not(ValueError, len(res) == n,
                     'The length of the predicted values must be equal to the number of steps.')

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
