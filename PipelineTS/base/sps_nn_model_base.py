import numpy as np
import pandas as pd
from spinesTS.preprocessing import split_series, lag_splits

from spinesUtils import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.asserts import raise_if_not
from spinesUtils.preprocessing import gc_collector, reshape_if

from PipelineTS.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp


class SpinesNNModelMixin(NNModelMixin, IntervalEstimationMixin):
    """
    SpinesNNModelMixin: A mixin class for integrating neural network models into the Spines framework.

    Parameters
    ----------
    time_col : str
        The column containing time information in the input data.
    target_col : str
        The column containing the target variable in the input data.
    accelerator : str or None, optional, default: None
        The accelerator to use for training (e.g., 'auto', 'cuda', 'cpu').

    Attributes
    ----------
    last_x : np.ndarray or None
        The last input sequence used for training or prediction.
    scaler : None
        Placeholder for a scaling object (e.g., MinMaxScaler) for future implementation.

    Methods
    -------
    _define_model()
        Abstract method to be implemented by subclasses for defining the neural network model.

    _data_preprocess(data, update_last_data=False, mode='train')
        Preprocesses the input data for training, validation, or prediction.

    fit(data, valid_data=None, cv=5, fit_kwargs=None)
        Fits the neural network model on the training data with optional validation data.

    _extend_predict(x, n, predict_kwargs)
        Extends predictions for extrapolation.

    predict(n, data=None, predict_kwargs=None)
        Makes predictions using the fitted neural network model.

    chosen_cols(data)
        Returns the selected columns from the input DataFrame.

    interval_predict(data)
        Calculates and adds the upper and lower quantile predictions to the DataFrame.

    calculate_confidence_interval_nn(data, fit_kwargs, cv)
        Calculates the confidence interval using cross-validated predictions.

    Examples
    --------
    # Instantiate SpinesNNModelMixin
    >>> nn_model = SpinesNNModelMixin(time_col='timestamp', target_col='value', accelerator='auto')
    """
    def __init__(self, time_col, target_col, accelerator=None):

        super().__init__(time_col, target_col, accelerator=accelerator)
        self.last_x = None
        self.scaler = None

    def _define_model(self):
        """
        Abstract method to be implemented by subclasses for defining the neural network model.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @ParameterValuesAssert({
        'mode': ('train', 'validation', 'predict')
    })
    def _data_preprocess(self, data, update_last_data=False, mode='train'):
        """
        Preprocesses the input data for training, validation, or prediction.

        Parameters
        ----------
        data : pd.DataFrame
            The input data in pandas DataFrame format.
        update_last_data : bool, optional, default: False
            Whether to update the last input sequence.
        mode : {'train', 'validation', 'predict'}, optional, default: 'train'
            The mode for data preprocessing.

        Returns
        -------
        x_train : np.ndarray
            The input features for training.
        y_train : np.ndarray
            The target variable for training.

        Raises
        ------
        ValueError
            If the length of the series is less than the specified lags.

        Examples
        --------
        # Preprocess training data
        >>> x_train, y_train = self._data_preprocess(train_data, update_last_data=True, mode='train')
        """
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        if update_last_data:
            self.last_dt = data[self.all_configs['time_col']].max()

        if mode == 'train':
            if update_last_data:
                self.last_x = data[self.all_configs['target_col']].iloc[-(2 * self.all_configs['lags']):]
            # x_train, y_train
            x_train, y_train = split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                                            window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            x_train = reshape_if(x_train, x_train.ndim == 1, (1, -1))
            y_train = reshape_if(y_train, y_train.ndim == 1, (1, -1))

            return x_train, y_train

        elif mode == 'validation':
            x, y = split_series(pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            x = reshape_if(x, x.ndim == 1, (1, -1))
            y = reshape_if(y, y.ndim == 1, (1, -1))

            return x, y

        else:
            x = lag_splits(data[self.all_configs['target_col']], window_size=self.all_configs['lags'])
            x = reshape_if(x, x.ndim == 1, (1, -1))

            return x

    @ParameterTypeAssert({
        'valid_data': (None, pd.DataFrame)
    })
    @gc_collector(3)
    def fit(self, data, valid_data=None, cv=5, fit_kwargs=None):
        """
        Fits the neural network model on the training data with optional validation data.

        Parameters
        ----------
        data : pd.DataFrame
            The training data in pandas DataFrame format.
        valid_data : pd.DataFrame or None, optional, default: None
            The validation data in pandas DataFrame format.
        cv : int, optional, default: 5
            The number of cross-validation folds.
        fit_kwargs : dict or None, optional, default: None
            Additional keyword arguments for fitting the model.

        Returns
        -------
        self : SpinesNNModelMixin
            Returns the instance itself.

        Examples
        --------
        # Fit the model on training data
        >>> nn_model.fit(train_data, valid_data=valid_data, cv=5, fit_kwargs={'epochs': 100})
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        if fit_kwargs is None:
            fit_kwargs = {}

        for fit_param in [
            'verbose', 'epochs', 'batch_size', 'patience',
            'min_delta', 'lr_scheduler', 'lr_scheduler_patience',
            'lr_factor', 'restore_best_weights', 'loss_type'
        ]:
            if fit_param not in fit_kwargs:
                fit_kwargs.update({fit_param: self.all_configs[fit_param]})

        x, y = self._data_preprocess(data, update_last_data=True, mode='train')

        self.x = data[self.all_configs['target_col']].iloc[-self.all_configs['lags']:]

        if valid_data is None:
            eval_set = [(x, y)]
        else:
            check_time_col_is_timestamp(valid_data, self.all_configs['time_col'])
            valid_x, valid_y = self._data_preprocess(valid_data, update_last_data=False, mode='validation')

            eval_set = [(valid_x, valid_y)]

        self.model.fit(x, y, eval_set=eval_set, **fit_kwargs)

        del x, y

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_nn(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _extend_predict(self, x, n, predict_kwargs):
        """
        Extends predictions for extrapolation.

        Parameters
        ----------
        x : np.ndarray
            The input sequence for prediction.
        n : int
            The number of time steps to predict.
        predict_kwargs : dict
            Additional keyword arguments for the prediction function.

        Returns
        -------
        res : np.ndarray
            The extrapolated prediction results.

        Raises
        ------
        AssertionError
            If the input arguments do not satisfy the specified conditions.

        Examples
        --------
        # Extend predictions for extrapolation
        >>> predictions = self._extend_predict(x, n, predict_kwargs={'verbose': True})
        """

        assert isinstance(n, int)
        assert x.ndim == 2

        current_res = self.model.predict(x, **predict_kwargs)

        current_res = reshape_if(current_res, current_res.ndim == 1, (1, -1))

        if n is None:
            return current_res.squeeze().tolist()
        elif n <= current_res.shape[1]:
            return current_res[-1][:n].tolist()
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
        Makes predictions using the fitted neural network model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.
        data : pd.DataFrame or None, optional, default: None
            The input data for prediction.
        predict_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the prediction function.

        Returns
        -------
        predictions : pd.DataFrame
            The DataFrame containing the predicted values.

        Examples
        --------
        # Make predictions using the fitted model
        >>> predictions = self.predict(n=10, data=test_data, predict_kwargs={'batch_size': 32})
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        if data is not None:
            check_time_col_is_timestamp(data, self.all_configs['time_col'])
            raise_if_not(
                ValueError, len(data) >= self.all_configs['lags'],
                'The length of the series must greater than or equal to the lags. '
            )

            x = self._data_preprocess(data.iloc[-self.all_configs['lags']:, :],
                                      update_last_data=False, mode='predict')
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
