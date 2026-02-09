import numpy as np
import pandas as pd
from PipelineTS.spinesTS.preprocessing import split_series, lag_splits, split_series_multivariate
from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert
from spinesUtils.asserts import raise_if, raise_if_not
from spinesUtils.preprocessing import gc_collector
from spinesUtils.logging import Logger

from PipelineTS.base.base import NNModelMixin, IntervalEstimationMixin
from PipelineTS.base.base_utils import generate_valid_data
from PipelineTS.utils import check_time_col_is_timestamp


logger = Logger(with_time=False)


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

    _data_preprocess(data, mode='train')
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
    def _data_preprocess(self, data, mode='train'):
        """
        Preprocesses the input data for training, validation, or prediction.

        Parameters
        ----------
        data : pd.DataFrame
            The input data in pandas DataFrame format.
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
        >>> x_train, y_train = self._data_preprocess(train_data, mode='train')
        """
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])

        if mode == 'train':
            # x_train, y_train
            x_train, y_train = split_series(data[self.all_configs['target_col']], data[self.all_configs['target_col']],
                                            window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            if x_train.ndim == 1:
                x_train = x_train.view((1, -1))
            if y_train.ndim == 1:
                y_train = y_train.view((1, -1))

            return x_train, y_train

        elif mode == 'validation':
            x, y = split_series(pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                pd.concat((self.last_x, data[self.all_configs['target_col']])),
                                window_size=self.all_configs['lags'], pred_steps=self.all_configs['lags'])

            if x.ndim == 1:
                x = x.view((1, -1))
            if y.ndim == 1:
                y = y.view((1, -1))

            return x, y

        else:
            x = lag_splits(data[self.all_configs['target_col']], window_size=self.all_configs['lags'])
            if x.ndim == 1:
                x = x.view((1, -1))

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

        self.x = data[self.all_configs['target_col']].iloc[-self.all_configs['lags']:]
        self.last_dt = data[self.all_configs['time_col']].max()

        x, y = self._data_preprocess(data, mode='train')

        if valid_data is None:
            eval_set = None  # [(x, y)]
        else:
            check_time_col_is_timestamp(valid_data, self.all_configs['time_col'])
            raise_if(
                ValueError, valid_data[self.all_configs['time_col']].min() <= data[self.all_configs['time_col']].max(),
                'validation data should be after the training data.')

            valid_data = generate_valid_data(data.copy(), valid_data, self.all_configs['lags'],
                                             self.all_configs['time_col'], self.all_configs['target_col'])

            valid_x, valid_y = self._data_preprocess(valid_data, mode='train')

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

        raise_if_not(ValueError, n > 0, 'n must be greater than 0.')
        raise_if_not(TypeError, isinstance(n, int), 'n must be an integer.')
        raise_if_not(AssertionError, x.ndim == 2, 'x must be a 2D array.')

        current_res = self.model.predict(x, **predict_kwargs)

        if current_res.ndim == 1:
            current_res = current_res.view((1, -1))

        if n <= current_res.shape[1]:
            return current_res[-1][:n].tolist()
        else:
            res = current_res.squeeze().tolist()
            for i in range(n - self.all_configs['lags']):
                x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
                current_res = self.model.predict(x, **predict_kwargs)
                if current_res.ndim == 1:
                    current_res = current_res.view((1, -1))

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
                                      mode='predict')
            last_dt = data[self.all_configs['time_col']].max()
        else:
            x_vals = self.x.values
            if x_vals.ndim == 1:
                x = x_vals.reshape((1, -1))
            else:
                x = x_vals.reshape((1, -1))
            last_dt = self.last_dt

        res = self._extend_predict(x, n, predict_kwargs=predict_kwargs)  # list

        raise_if_not(ValueError, len(res) == n, "The length of the predictions must equal to n.")

        res = pd.DataFrame(res, columns=[self.all_configs['target_col']])
        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(res.index.shape[0] + 1), unit='D')[1:]

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)

    def backtest(self, data):
        raise_if_not(ValueError, data.shape[0] >= 3 * self.all_configs['lags'],
                     "The length of the series must greater than or equal to 3 * lags. ")

        ...


class SpinesMultivariateNNModelMixin(NNModelMixin, IntervalEstimationMixin):
    """Mixin for multivariate NN models supporting three prediction modes:

    1. Univariate (single-input, single-output):
       target_col='y', feature_cols=None
    2. Multi-input, single-output:
       target_col='y', feature_cols=['a', 'b', 'y']
    3. Multi-input, multi-output:
       target_col=['a', 'b'], feature_cols=['a', 'b', 'c']

    Parameters
    ----------
    time_col : str
        Time column name.
    target_col : str or list of str
        Target column(s) to predict.
    feature_cols : list of str or None
        Input feature column(s). If None, uses target_col only (univariate).
    accelerator : str or None
        Compute device.
    """

    # Subclasses should set this before calling super().__init__.
    # True  = model outputs all input variates (e.g. ITransformer).
    # False = model outputs only target channels (e.g. SRSNet).
    _train_on_all_features = False

    def __init__(self, time_col, target_col, feature_cols=None, accelerator=None):
        self._multi_target = isinstance(target_col, list)
        self._target_col_list = target_col if self._multi_target else [target_col]
        self._primary_target = self._target_col_list[0] if self._multi_target else target_col

        if feature_cols is not None:
            self._feature_cols = list(feature_cols)
            self._multivariate = True
        else:
            self._feature_cols = list(self._target_col_list)
            self._multivariate = self._multi_target

        self._n_vars = len(self._feature_cols)
        self._n_targets = len(self._target_col_list)

        # Compute target column indices within feature_cols (for extracting from full output)
        self._target_indices_in_features = [
            self._feature_cols.index(t) for t in self._target_col_list
            if t in self._feature_cols
        ]

        super().__init__(time_col, self._primary_target, accelerator=accelerator)
        self.last_x = None
        self.scaler = None

    @property
    def _is_univariate(self):
        return not self._multivariate and not self._multi_target

    def _get_data_columns(self):
        """Return all required data columns (time + features)."""
        cols = [self.all_configs['time_col']] + self._feature_cols
        return list(dict.fromkeys(cols))

    def _get_features_and_targets(self, data):
        """Extract feature array (T, C) and target array from DataFrame.

        When _train_on_all_features is True, targets = features (all channels).
        Otherwise, targets are the target column(s) only.
        """
        features = data[self._feature_cols].values  # (T, C)
        if self._train_on_all_features:
            targets = features  # (T, C) — model learns to predict all variates
        elif self._multi_target:
            targets = data[self._target_col_list].values  # (T, n_targets)
        else:
            targets = data[self._primary_target].values  # (T,)
        return features, targets

    def _data_preprocess(self, data, mode='train'):
        """Preprocess data for multivariate time series.

        Returns
        -------
        For mode='train':
            X : np.ndarray, shape (N, lags, C) for multivariate, (N, lags) for univariate
            y : np.ndarray, shape (N, lags) for single-target, (N, lags, n_targets) for multi-target
        For mode='predict':
            X : np.ndarray, shape (1, lags, C) for multivariate, (1, lags) for univariate
        """
        data = data.copy()
        data[self.all_configs['time_col']] = pd.to_datetime(data[self.all_configs['time_col']])
        lags = self.all_configs['lags']

        if self._is_univariate:
            if mode in ('train', 'validation'):
                x, y = split_series(
                    data[self._primary_target], data[self._primary_target],
                    window_size=lags, pred_steps=lags
                )
                if x.ndim == 1:
                    x = x.reshape((1, -1))
                if y.ndim == 1:
                    y = y.reshape((1, -1))
                return x, y
            else:
                x = lag_splits(data[self._primary_target], window_size=lags)
                if x.ndim == 1:
                    x = x.reshape((1, -1))
                return x
        else:
            features, targets = self._get_features_and_targets(data)
            if mode in ('train', 'validation'):
                X, y = split_series_multivariate(
                    features, targets, window_size=lags, pred_steps=lags
                )
                return X, y
            else:
                x = features[-lags:]
                return x.reshape((1, lags, self._n_vars))

    @ParameterTypeAssert({
        'valid_data': (None, pd.DataFrame)
    })
    @gc_collector(3)
    def fit(self, data, valid_data=None, cv=5, fit_kwargs=None):
        """Fit the multivariate model.

        Parameters
        ----------
        data : pd.DataFrame
            Training data with time_col and feature_cols.
        valid_data : pd.DataFrame or None
            Validation data.
        cv : int
            Cross-validation folds for confidence interval.
        fit_kwargs : dict or None
            Additional fit keyword arguments.
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        required_cols = self._get_data_columns()
        data = data[required_cols].copy()

        if fit_kwargs is None:
            fit_kwargs = {}

        for fit_param in [
            'verbose', 'epochs', 'batch_size', 'patience',
            'min_delta', 'lr_scheduler', 'lr_scheduler_patience',
            'lr_factor', 'restore_best_weights', 'loss_type'
        ]:
            if fit_param not in fit_kwargs:
                fit_kwargs.update({fit_param: self.all_configs[fit_param]})

        self._last_features = data[self._feature_cols].iloc[-self.all_configs['lags']:].copy()
        self.last_dt = data[self.all_configs['time_col']].max()

        x, y = self._data_preprocess(data, mode='train')

        if valid_data is None:
            eval_set = None
        else:
            check_time_col_is_timestamp(valid_data, self.all_configs['time_col'])
            raise_if(
                ValueError,
                valid_data[self.all_configs['time_col']].min() <= data[self.all_configs['time_col']].max(),
                'validation data should be after the training data.'
            )
            valid_data = valid_data[required_cols].copy()

            lags = self.all_configs['lags']
            min_rows_needed = 2 * lags
            if valid_data.shape[0] < min_rows_needed:
                pad_len = min_rows_needed - valid_data.shape[0]
                pad_data = data.iloc[-pad_len:][required_cols].copy()
                valid_data = pd.concat([pad_data, valid_data], axis=0).reset_index(drop=True)

            valid_x, valid_y = self._data_preprocess(valid_data, mode='train')
            eval_set = [(valid_x, valid_y)]

        self.model.fit(x, y, eval_set=eval_set, **fit_kwargs)
        del x, y

        if self.all_configs.get('quantile') is not None:
            self.all_configs['quantile_error'] = \
                self._calculate_quantile_error(data, fit_kwargs=fit_kwargs, cv=cv)

        return self

    def _calculate_quantile_error(self, data, fit_kwargs=None, cv=5):
        """Calculate quantile error for confidence intervals."""
        from copy import deepcopy
        from PipelineTS.spinesTS.metrics import wmape

        if fit_kwargs is None:
            kwargs = {}
        else:
            kwargs = deepcopy(fit_kwargs)
        kwargs.update({'verbose': False})

        residuals = []
        n = len(data)
        lags = self.all_configs['lags']
        block_len = lags
        rng = np.random.RandomState(0)

        for _ in range(cv):
            n_blocks = max(1, n // block_len)
            all_block_starts = np.arange(0, n - block_len + 1)
            if len(all_block_starts) == 0:
                continue

            chosen = rng.choice(len(all_block_starts), size=n_blocks, replace=True)
            train_indices = set()
            for c in chosen:
                start = all_block_starts[c]
                for j in range(start, min(start + block_len, n)):
                    train_indices.add(j)

            test_indices = sorted(set(range(n)) - train_indices)
            train_indices = sorted(train_indices)

            if len(test_indices) > 0 and len(train_indices) >= block_len:
                train_data = data.iloc[train_indices, :].reset_index(drop=True)
                test_data = data.iloc[test_indices, :].reset_index(drop=True)

                try:
                    train_x, train_y = self._data_preprocess(train_data, mode='train')
                    test_x, test_y = self._data_preprocess(test_data, mode='train')
                except (ValueError, IndexError):
                    continue

                model = self._define_model()
                model.fit(train_x, train_y, eval_set=[(train_x, train_y)], **kwargs)
                res = model.predict(test_x).flatten()
                y_error = wmape(test_y.flatten(), res.flatten())
                residuals.append(y_error)

        if len(residuals) == 0:
            return 0.0
        return np.percentile(residuals, q=self.all_configs['quantile'])

    def _extend_predict(self, x, n, predict_kwargs):
        """Extend predictions, supporting both 2D (univariate) and 3D (multivariate) inputs."""
        raise_if_not(ValueError, n > 0, 'n must be greater than 0.')
        raise_if_not(TypeError, isinstance(n, int), 'n must be an integer.')

        if self._is_univariate:
            raise_if_not(AssertionError, x.ndim == 2, 'x must be a 2D array for univariate.')
            current_res = self.model.predict(x, **predict_kwargs)
            if current_res.ndim == 1:
                current_res = current_res.reshape((1, -1))

            if n <= current_res.shape[1]:
                return current_res[-1][:n].tolist()
            else:
                res = current_res.squeeze().tolist()
                lags = self.all_configs['lags']
                for i in range(n - lags):
                    x = np.concatenate((x[:, 1:], current_res[:, 0:1]), axis=1)
                    current_res = self.model.predict(x, **predict_kwargs)
                    if current_res.ndim == 1:
                        current_res = current_res.reshape((1, -1))
                    res.append(current_res.squeeze().tolist()[-1])
                return res
        else:
            raise_if_not(AssertionError, x.ndim == 3, 'x must be a 3D array for multivariate.')
            current_res = self.model.predict(x, **predict_kwargs)

            lags = self.all_configs['lags']

            if self._train_on_all_features:
                # Model outputs all variates: current_res shape (1, pred_steps, C)
                if current_res.ndim == 2 and self._n_vars > 1:
                    current_res = current_res.reshape((1, -1, self._n_vars))
                elif current_res.ndim == 2:
                    current_res = current_res.reshape((1, -1, 1))

                pred_steps = current_res.shape[1]
                # Extract only target columns from the full output
                tidx = self._target_indices_in_features

                if n <= pred_steps:
                    if len(tidx) == 1:
                        return current_res[0, :n, tidx[0]].tolist()
                    else:
                        return current_res[0, :n][:, tidx].tolist()
                else:
                    if len(tidx) == 1:
                        res = current_res[0, :, tidx[0]].tolist()
                    else:
                        res = current_res[0][:, tidx].tolist()
                    for i in range(n - lags):
                        # Feed full output back as next input
                        x = np.concatenate((x[:, 1:, :], current_res[:, 0:1, :]), axis=1)
                        current_res = self.model.predict(x, **predict_kwargs)
                        if current_res.ndim == 2 and self._n_vars > 1:
                            current_res = current_res.reshape((1, -1, self._n_vars))
                        elif current_res.ndim == 2:
                            current_res = current_res.reshape((1, -1, 1))
                        if len(tidx) == 1:
                            res.append(current_res[0, -1, tidx[0]].tolist())
                        else:
                            res.append(current_res[0, -1][tidx].tolist())
                    return res

            elif self._multi_target:
                if current_res.ndim == 2:
                    current_res = current_res.reshape((1, -1, self._n_targets))
                pred_steps = current_res.shape[1]

                if n <= pred_steps:
                    return current_res[0, :n, :].tolist()
                else:
                    res = current_res[0].tolist()
                    tidx = self._target_indices_in_features
                    for i in range(n - lags):
                        new_features = x[0, -1:, :].copy()
                        for idx, feat_idx in enumerate(tidx):
                            if idx < current_res.shape[-1]:
                                new_features[0, feat_idx] = current_res[0, 0, idx]
                        x = np.concatenate((x[:, 1:, :], new_features.reshape(1, 1, -1)), axis=1)
                        current_res = self.model.predict(x, **predict_kwargs)
                        if current_res.ndim == 2:
                            current_res = current_res.reshape((1, -1, self._n_targets))
                        res.append(current_res[0, -1, :].tolist())
                    return res
            else:
                # Multi-input single-output, model outputs targets only
                if current_res.ndim == 1:
                    current_res = current_res.reshape((1, -1))

                pred_steps = current_res.shape[1]

                if n <= pred_steps:
                    return current_res[0, :n].tolist()
                else:
                    res = current_res[0].tolist()
                    target_idx = self._target_indices_in_features[0] \
                        if self._target_indices_in_features else -1
                    for i in range(n - lags):
                        new_features = x[0, -1:, :].copy()
                        if target_idx >= 0:
                            new_features[0, target_idx] = current_res[0, 0]
                        x = np.concatenate((x[:, 1:, :], new_features.reshape(1, 1, -1)), axis=1)
                        current_res = self.model.predict(x, **predict_kwargs)
                        if current_res.ndim == 1:
                            current_res = current_res.reshape((1, -1))
                        res.append(current_res[0, -1].tolist())
                    return res

    def predict(self, n, data=None, predict_kwargs=None):
        """Predict future values.

        Parameters
        ----------
        n : int
            Number of time steps to predict.
        data : pd.DataFrame or None
            Input data. If None, uses stored last features.
        predict_kwargs : dict or None
            Additional prediction arguments.

        Returns
        -------
        pd.DataFrame with time_col and target column(s).
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        if data is not None:
            check_time_col_is_timestamp(data, self.all_configs['time_col'])
            lags = self.all_configs['lags']
            raise_if_not(
                ValueError, len(data) >= lags,
                'The length of the series must be >= lags.'
            )
            subset = data.iloc[-lags:]
            last_dt = data[self.all_configs['time_col']].max()
        else:
            subset = None
            last_dt = self.last_dt

        if self._is_univariate:
            if subset is not None:
                x = self._data_preprocess(subset, mode='predict')
            else:
                x_vals = self._last_features[self._primary_target].values
                x = x_vals.reshape((1, -1))
        else:
            if subset is not None:
                x = self._data_preprocess(subset, mode='predict')
            else:
                x = self._last_features.values.reshape((1, self.all_configs['lags'], self._n_vars))

        res_data = self._extend_predict(x, n, predict_kwargs=predict_kwargs)

        if self._multi_target:
            res = pd.DataFrame(res_data, columns=self._target_col_list)
        else:
            res = pd.DataFrame(res_data, columns=[self._primary_target])

        res[self.all_configs['time_col']] = \
            last_dt + pd.to_timedelta(range(len(res) + 1), unit='D')[1:]

        if self.all_configs.get('quantile') is not None and not self._multi_target:
            res = self.interval_predict(res)

        return self.chosen_cols(res)

    def _define_model(self):
        raise NotImplementedError

    def backtest(self, data):
        raise_if_not(ValueError, data.shape[0] >= 3 * self.all_configs['lags'],
                     "The length of the series must be >= 3 * lags.")
        ...


class SpinesMLModelMixin:
    """spinesTS ml model mixin class"""
    ...
