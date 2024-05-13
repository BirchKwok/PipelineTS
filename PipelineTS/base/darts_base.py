import numpy as np
import pandas as pd
from darts.models import (
    CatBoostModel as CBT,
    LightGBMModel as LGB,
    XGBModel as XGB,
    RandomForest as RF
)

from spinesUtils.asserts import ParameterTypeAssert
from spinesUtils.logging import Logger
from spinesUtils.asserts import raise_if_not, check_has_param
from spinesUtils.preprocessing import gc_collector

from PipelineTS.base.base_utils import generate_valid_data
from PipelineTS.utils import load_dataset_to_darts, check_time_col_is_timestamp

logger = Logger(with_time=False)


class DartsForecastMixin:
    @staticmethod
    def convert2dts_dataframe(
            df,
            time_col,
            target_col,
            **kwargs
    ):
        return load_dataset_to_darts(data=df, time_col=time_col, target_col=target_col, **kwargs)

    @staticmethod
    def convert2pd_dataframe(df):
        return df.pd_dataframe()

    @gc_collector()
    def _fit(self, data, valid_data=None, convert_dataframe_kwargs=None, fit_kwargs=None):
        """
        Fits the Darts forecasting model on the training data.

        Parameters
        ----------
        data : pd.DataFrame
            The training data in pandas DataFrame format.

        valid_data: None or pandas dataframe
        convert_dataframe_kwargs : dict or None, optional, default: None
            Additional keyword arguments for converting the DataFrame to Darts TimeSeries.
        fit_kwargs : dict or None, optional, default: None
            Additional keyword arguments for fitting the Darts model.
        convert_float32 : bool, optional, default: True
            Whether to convert the data to float32.

        Returns
        -------
        self : DartsForecastMixin
            Returns the instance itself.
        """

        if convert_dataframe_kwargs is None:
            convert_dataframe_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}

        if valid_data is not None:
            if valid_data.shape[0] < self.all_configs['lags'] * 2 and isinstance(self.model, (CBT, LGB, XGB, RF)):
                logger.info(
                    "The provided validation time series dataset is too short for obtaining even one training point."
                    "\nIt is recommended that the `lags` parameter be less than or equal to half the length of the "
                    "validation set."
                )
                valid_data = None
            else:
                valid_data = generate_valid_data(data, valid_data, lags=self.all_configs['lags'],
                                                 time_col=self.all_configs['time_col'],
                                                 target_col=self.all_configs['target_col'])

                valid_data = self.convert2dts_dataframe(valid_data, time_col=self.all_configs['time_col'],
                                                        target_col=self.all_configs['target_col'],
                                                        **convert_dataframe_kwargs).astype(np.float32)

        data = self.convert2dts_dataframe(data, time_col=self.all_configs['time_col'],
                                          target_col=self.all_configs['target_col'],
                                          **convert_dataframe_kwargs).astype(np.float32)

        if hasattr(self.model.fit, 'val_series'):
            self.model.fit(data, val_series=valid_data, **fit_kwargs)
        else:
            self.model.fit(data, **fit_kwargs)

        return self

    @ParameterTypeAssert({
        'data': pd.DataFrame,
        'valid_data': (None, pd.DataFrame),
        'cv': int,
        'convert_dataframe_kwargs': (None, dict),
        'fit_kwargs': (None, dict)
    })
    def fit(self, data, valid_data=None, cv=5, convert_dataframe_kwargs=None, fit_kwargs=None):
        """
        Fit the model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        valid_data: None or pandas dataframe

        cv : int, optional
            The number of cross-validation folds. Default is 5.

        convert_dataframe_kwargs : dict or None, optional
            Additional keyword arguments for converting the DataFrame. Default is None.

        fit_kwargs : dict or None, optional
            Additional keyword arguments for fitting the model. Default is None.

        Returns
        -------
        self
            Returns an instance of the fitted model.
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        self._fit(data, valid_data=valid_data,
                  convert_dataframe_kwargs=convert_dataframe_kwargs, fit_kwargs=fit_kwargs)

        # Calculate quantile error if quantile is specified
        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    def _predict(self, n, data=None, predict_kwargs=None, convert_dataframe_kwargs=None, convert_float32=True):
        """
        Makes predictions using the fitted Darts forecasting model.

        Parameters
        ----------
        n : int
            The number of time steps to predict.
        data : pd.DataFrame or None, optional, default: None
            The input data for prediction.
        predict_kwargs : dict or None, optional, default: None
            Additional keyword arguments for the prediction function.
        convert_dataframe_kwargs : dict or None, optional, default: None
            Additional keyword arguments for converting the DataFrame to Darts TimeSeries.
        convert_float32 : bool, optional, default: True
            Whether to convert the data to float32.

        Returns
        -------
        predictions : pd.DataFrame
            The DataFrame containing the predicted values.
        """
        if predict_kwargs is None:
            predict_kwargs = {}
        if check_has_param(self.model.predict, 'series') and data is not None:
            raise_if_not(
                ValueError, len(data) >= self.all_configs['lags'],
                'The length of the series must greater than or equal to the lags. '
            )

            convert_dataframe_kwargs = {} if convert_dataframe_kwargs is None else convert_dataframe_kwargs
            data = self.convert2dts_dataframe(
                data, time_col=self.all_configs['time_col'],
                target_col=self.all_configs['target_col'],
                **convert_dataframe_kwargs
            )

            if convert_float32:
                data = data.astype(np.float32)

            predict_kwargs.update({'series': data})

        return self.model.predict(
            n,
            **predict_kwargs
        ).pd_dataframe()

    @ParameterTypeAssert({
        'n': int,
        'data': (pd.DataFrame, None),
        'predict_kwargs': (None, dict),
        'convert_dataframe_kwargs': (None, dict),
    })
    def predict(self, n, data=None, predict_kwargs=None, convert_dataframe_kwargs=None):
        """
        Generate predictions for future time steps.

        Parameters
        ----------
        n : int
            The number of time steps to predict.

        data : pd.DataFrame or None, optional
            Additional data for prediction. Default is None.

        predict_kwargs : dict or None, optional
            Additional keyword arguments for prediction. Default is None.

        convert_dataframe_kwargs : dict or None, optional
            Additional keyword arguments for converting the DataFrame. Default is None.

        Returns
        -------
        pd.DataFrame
            Returns a DataFrame with predicted values and intervals.
        """
        if predict_kwargs is None:
            predict_kwargs = {}

        if data is not None:
            check_time_col_is_timestamp(data, self.all_configs['time_col'])

        # Generate predictions
        res = self._predict(n=n, data=data, predict_kwargs=predict_kwargs,
                            convert_dataframe_kwargs=convert_dataframe_kwargs)
        res = self.rename_prediction(res)

        # Generate prediction intervals if quantile is specified
        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)

    def rename_prediction(self, data):
        """
        Renames the prediction columns for better readability.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the predicted values.

        Returns
        -------
        data : pd.DataFrame
            The DataFrame with renamed columns.
        """
        data.columns.name = None
        data[self.all_configs['time_col']] = data.index.copy()

        data = data.reset_index(drop=True)

        for i in data.columns:
            if i == f"{self.all_configs['target_col']}_q0.50":
                data.rename(columns={i: f"{self.all_configs['target_col']}"}, inplace=True)

        if self.all_configs['quantile'] is not None:
            if self.all_configs['quantile'] < round(1 - self.all_configs['quantile'], 1):
                ratio = self.all_configs['quantile']
            else:
                ratio = round(1 - self.all_configs['quantile'], 1)

            if len(str(ratio).split('.')[-1]) == 1:
                left_ratio = str(ratio) + '0'
                right_ratio = str(1 - ratio) + '0'
            else:
                left_ratio = str(ratio)
                right_ratio = str(1 - ratio)

            for i in data.columns:
                if i == f"{self.all_configs['target_col']}_q{right_ratio}":
                    data.rename(columns={i: f"{self.all_configs['target_col']}_upper"}, inplace=True)

                elif i == f"{self.all_configs['target_col']}_q{left_ratio}":
                    data.rename(columns={i: f"{self.all_configs['target_col']}_lower"}, inplace=True)

        return self.chosen_cols(data)
