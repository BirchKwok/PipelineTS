import logging
from spinesUtils.asserts import generate_function_kwargs
from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class ProphetModel(StatisticModelMixin, IntervalEstimationMixin):
    """
    ProphetModel: A wrapper for the Facebook Prophet model with additional features.

    Parameters
    ----------
    time_col : str
        The column containing time information in the input data.
    target_col : str
        The column containing the target variable in the input data.
    lags : int, optional, default: 1
        The number of lagged values to use as input features for training and prediction.
    country_holidays : dict, optional, default: None
        A dictionary specifying country-specific holidays in the format {country: [list of holidays]}.
    quantile : float, optional, default: 0.9
        The quantile used for interval prediction. Set to None for point prediction.
    random_state : int, optional, default: 0
        The random seed for reproducibility.
    **prophet_configs
        Additional keyword arguments for configuring the Prophet model.

    Attributes
    ----------
    model : Prophet
        The Prophet model from the Facebook Prophet library.
    """

    from prophet import Prophet

    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            country_holidays=None,
            quantile=0.9,
            random_state=0,
            **prophet_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            ProphetModel.Prophet,
            holidays=country_holidays,
            **prophet_configs
        )

        self.model = self._define_model()

        self.all_configs.update({
            'quantile': quantile,
            'quantile_error': 0,
            'time_col': time_col,
            'target_col': target_col,
            'random_state': random_state,
            'lags': lags,
        })

    def _define_model(self):
        """
        Define the Prophet model from the Facebook Prophet library.

        Returns
        -------
        Prophet
            The Prophet model from the Facebook Prophet library.
        """
        return ProphetModel.Prophet(**self.all_configs['model_configs'])

    @staticmethod
    def _prophet_preprocessing(df, time_col, target_col):
        """
        Preprocess the input data for compatibility with the Prophet model.

        Parameters
        ----------
        df : pd.DataFrame
            The input data in pandas DataFrame format.
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.

        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame compatible with the Prophet model.
        """
        df_ = df[[time_col, target_col]]
        if 'ds' != time_col or 'y' != target_col:
            df_ = df_.rename(columns={time_col: 'ds', target_col: 'y'})
        return df_

    def fit(self, data, freq='D', cv=5, fit_kwargs=None):
        """
        Fit the Prophet model on the input data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data in pandas DataFrame format.
        freq : str, optional, default: 'D'
            The frequency of the time series data.
        cv : int, optional, default: 5
            The number of cross-validation folds.
        fit_kwargs : None, optional, default: None
            Additional keyword arguments for fitting the model.

        Returns
        -------
        self : ProphetModel
            Returns the instance itself.
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        if fit_kwargs is None:
            fit_kwargs = {}
        data = self._prophet_preprocessing(data, self.all_configs['time_col'], self.all_configs['target_col'])
        self.model.fit(data, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_prophet(data, cv=cv, freq=freq, fit_kwargs=fit_kwargs)
        return self

    def predict(self, n, freq='D', include_history=False):
        """
        Make predictions using the fitted Prophet model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.
        freq : str, optional, default: 'D'
            The frequency of the time series data.
        include_history : bool, optional, default: False
            Whether to include the historical data in the predictions.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the predicted values.
        """
        res = self.model.predict(
            self.model.make_future_dataframe(
                periods=n,
                freq=freq,
                include_history=include_history,
            )
        )[['ds', 'yhat']].rename(
            columns={
                'ds': self.all_configs['time_col'],
                'yhat': self.all_configs['target_col'],
            }
        )

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
