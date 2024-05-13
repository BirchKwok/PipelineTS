from darts.models import AutoARIMA
from spinesUtils.asserts import generate_function_kwargs

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.base.darts_base import DartsForecastMixin
from PipelineTS.utils import check_time_col_is_timestamp


class AutoARIMAModel(DartsForecastMixin, StatisticModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            start_p=8,
            max_p=12,
            start_q=1,
            seasonal=False,
            quantile=0.9,
            seasonal_length=12,
            n_jobs=-1,
            **darts_auto_arima_configs
    ):
        """
        AutoARIMAModel: A wrapper for the AutoARIMA model from the darts library with additional features.

        Parameters
        ----------
        time_col : str
            The column containing time information in the input data.
        target_col : str
            The column containing the target variable in the input data.
        lags : int, optional, default: 1
            The number of lagged values to use as input features for training and prediction.
        start_p : int, optional, default: 8
            The starting value for the order of autoregressive (AR) component.
        max_p : int, optional, default: 12
            The maximum value for the order of autoregressive (AR) component.
        start_q : int, optional, default: 1
            The starting value for the order of moving average (MA) component.
        seasonal : bool, optional, default: False
            Whether the time series exhibits seasonality.
        seasonal_length : int, optional, default: 12
            The length of the seasonal cycle, if seasonality is True.
        quantile : float, optional, default: 0.9
            The quantile used for interval prediction. Set to None for point prediction.
        n_jobs : int, optional, default: -1
            The number of jobs to run in parallel during model fitting.
        **darts_auto_arima_configs
            Additional keyword arguments for configuring the AutoARIMA model.

        Attributes
        ----------
        model : darts.models.AutoARIMA
            The AutoARIMA model from the darts library.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            AutoARIMA,
            start_p=start_p,
            max_p=max_p,
            start_q=start_q,
            seasonal=seasonal,
            seasonal_length=seasonal_length,
            n_jobs=n_jobs,
            **darts_auto_arima_configs
        )

        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,   # meanness, but only to follow coding conventions
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

    def _define_model(self):
        """
        Define the AutoARIMA model from the darts library.

        Returns
        -------
        darts.models.AutoARIMA
            The AutoARIMA model from the darts library.
        """
        return AutoARIMA(**self.all_configs['model_configs'])

    def fit(self, data, convert_dataframe_kwargs=None, cv=5, fit_kwargs=None):
        """
        Fit the AutoARIMA model on the input data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data in pandas DataFrame format.
        convert_dataframe_kwargs : None, optional, default: None
            Additional keyword arguments for converting the input data to the required format.
        cv : int, optional, default: 5
            The number of cross-validation folds.
        fit_kwargs : None, optional, default: None
            Additional keyword arguments for fitting the model.

        Returns
        -------
        self : AutoARIMAModel
            Returns the instance itself.
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        super().fit(
            data,
            convert_dataframe_kwargs=convert_dataframe_kwargs,
            fit_kwargs=fit_kwargs
        )

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    def predict(self, n, **kwargs):
        """
        Make predictions using the fitted AutoARIMA model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.
        kwargs : dict
            Additional keyword arguments for making predictions.
        """
        res = super().predict(n, **kwargs)
        res = self.rename_prediction(res)

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
