from spinesTS.ml_model import MultiOutputRegressor as MOR, MultiStepRegressor as MSR
from sklearn.multioutput import RegressorChain
from spinesTS.preprocessing import split_series, lag_splits
from spinesUtils import generate_function_kwargs, ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.asserts import raise_if_not
from spinesUtils.preprocessing import reshape_if
from spinesTS.pipeline import Pipeline
from PipelineTS.base import GBDTModelMixin, IntervalEstimationMixin
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


class _MultiOutputModelMixin(GBDTModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            estimator=LGBMRegressor,
            **model_configs
    ):
        """
        A mixin class for multi-output regression models using scikit-learn's ensemble.RandomForest.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        estimator : sklearn.base.BaseEstimator, optional, default: LGBMRegressor
            The base estimator used for the multi-output regression model.
        **model_configs : dict
            Additional keyword arguments for configuring the base estimator.

        Attributes
        ----------
        _base_estimator : sklearn.base.BaseEstimator
            The base estimator for the multi-output regression model.
        all_configs : dict
            A dictionary containing all configuration parameters for the model.
        last_dt : pandas.Timestamp or None
            The last timestamp in the input data.
        last_lags_dataframe : pandas.DataFrame or None
            The DataFrame containing the last lagged values used for prediction.
        x : pandas.DataFrame or None
            The last input features used for training.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self._base_estimator = estimator
        self.all_configs['model_configs'] = generate_function_kwargs(
            self._base_estimator,
            **model_configs
        )

        self.last_dt = None
        self.last_lags_dataframe = None

        self.all_configs.update(
            {
                'lags': lags,
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

        self.x = None

    def _define_model(self):
        """
        Define the multi-output regression model using scikit-learn's ensemble.RandomForest.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError

    @ParameterTypeAssert({
        'mode': str
    })
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
            # x_train, y_train
            return split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                                window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])
        else:
            return lag_splits(data[self.all_configs['target_col']], window_size=self.all_configs['lags'])

    def fit(self, data, cv=5, fit_kwargs=None):
        """
        Fit the multi-output regression model to the training data.

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
        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        self.last_lags_dataframe = data.iloc[-(2 * self.all_configs['lags'] + 1):, :]

        if fit_kwargs is None:
            fit_kwargs = {}

        x, y = self._data_preprocess(data, update_last_dt=True, mode='train')
        x = pd.DataFrame(x)

        self.x = x.iloc[-1:, :]

        self.model.fit(x, y, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_mor(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n, predict_kwargs):
        """
        Extrapolation prediction.

        Parameters
        ----------
        x : numpy.ndarray
            Data to predict, must be 2-dimensional.
        n : int
            Number of prediction steps.
        predict_kwargs : dict
            Additional keyword arguments for the prediction process.

        Returns
        -------
        list
            List of predictions.
        """
        assert isinstance(n, int)
        assert x.ndim == 2

        current_res = self.model.predict(x, **predict_kwargs)
        current_res = reshape_if(current_res, current_res.ndim == 1, (1, -1))

        if n is None:
            return current_res.squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res.squeeze().tolist()[:n]
        else:
            res = current_res.squeeze().tolist()

            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
                current_res = self.model.predict(x, **predict_kwargs)
                current_res = reshape_if(current_res, current_res.ndim == 1, (1, -1))

                res.append(current_res.squeeze().tolist()[-1])

            return res

    def predict(self, n, data=None, predict_kwargs=None):
        """
        Predict future values using the trained model.

        Parameters
        ----------
        n : int
            Number of prediction steps.
        data : pandas.DataFrame or None, optional, default: None
            Additional data for prediction.
        predict_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the prediction process.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the predicted values and corresponding timestamps.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        if data is not None:
            raise_if_not(
                ValueError, len(data) >= self.all_configs['lags'],
                'The length of the series must be greater than or equal to the lags.'
            )

            x = self._data_preprocess(
                data, mode='predict', update_last_dt=False
            )[-1, :]
            x = reshape_if(x, x.ndim == 1, (1, -1))
            last_dt = data[self.all_configs['time_col']].max()
        else:
            x = reshape_if(self.x.values, self.x.values.ndim == 1, (1, -1))
            last_dt = self.last_dt

        res = self._extend_predict(x, n, predict_kwargs=predict_kwargs)  # list

        assert len(res) == n

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)


class MultiOutputRegressorModel(_MultiOutputModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            estimator=LGBMRegressor,
            **model_configs
    ):
        """
        Multi-output regression model using scikit-learn's ensemble.RandomForest with a regressor chain.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        estimator : sklearn.base.BaseEstimator, optional, default: LGBMRegressor
            The base estimator used for the multi-output regression model.
        **model_configs : dict
            Additional keyword arguments for configuring the base estimator.

        Attributes
        ----------
        model : spinesTS.pipeline.Pipeline
            The pipeline containing the multi-output regressor model.
        """
        super().__init__(
            time_col=time_col,
            target_col=target_col,
            lags=lags,
            quantile=quantile,
            estimator=estimator,
            **model_configs)

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the multi-output regressor model using a regressor chain.

        Returns
        -------
        spinesTS.pipeline.Pipeline
            The pipeline containing the multi-output regressor model.
        """
        return Pipeline([
            ('model', MOR(self._base_estimator(**self.all_configs['model_configs'])))
        ])


class MultiStepRegressorModel(_MultiOutputModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            estimator=LGBMRegressor,
            **model_configs
    ):
        """
        Multi-step regression model using scikit-learn's ensemble.RandomForest with a multi-step regressor.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        estimator : sklearn.base.BaseEstimator, optional, default: LGBMRegressor
            The base estimator used for the multi-step regression model.
        **model_configs : dict
            Additional keyword arguments for configuring the base estimator.

        Attributes
        ----------
        model : spinesTS.pipeline.Pipeline
            The pipeline containing the multi-step regressor model.
        """
        super().__init__(
            time_col=time_col,
            target_col=target_col,
            lags=lags,
            quantile=quantile,
            estimator=estimator,
            **model_configs)

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the multi-step regressor model using a multi-step regressor.

        Returns
        -------
        spinesTS.pipeline.Pipeline
            The pipeline containing the multi-step regressor model.
        """
        return Pipeline([
            ('model', MSR(self._base_estimator(**self.all_configs['model_configs'])))
        ])


class RegressorChainModel(_MultiOutputModelMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            estimator=LGBMRegressor,
            **model_configs
    ):
        """
        Regressor chain model using scikit-learn's ensemble.RandomForest.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values to use as input features for training and prediction.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        estimator : sklearn.base.BaseEstimator, optional, default: LGBMRegressor
            The base estimator used for the regressor chain model.
        **model_configs : dict
            Additional keyword arguments for configuring the base estimator.

        Attributes
        ----------
        model : sklearn.multioutput.RegressorChain
            The regressor chain model.
        """
        super().__init__(
            time_col=time_col,
            target_col=target_col,
            lags=lags,
            quantile=quantile,
            estimator=estimator,
            **model_configs)

        self.model = self._define_model()

    def _define_model(self):
        """
        Define the regressor chain model using a base estimator.

        Returns
        -------
        sklearn.multioutput.RegressorChain
            The regressor chain model.
        """
        return RegressorChain(self._base_estimator(**self.all_configs['model_configs']))
