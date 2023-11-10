import sys
from copy import deepcopy

import numpy as np
from spinesUtils.asserts import *
from spinesUtils.preprocessing import gc_collector

from spinesTS.metrics import wmape
from spinesTS.utils import func_has_params

from PipelineTS.utils import load_dataset_to_darts


class DartsForecastMixin:
    @staticmethod
    def convert2dts_dataframe(
            df,
            time_col,
            target_col,
            **kwargs
    ):
        return load_dataset_to_darts(df, time_col, target_col, **kwargs)

    @staticmethod
    def convert2pd_dataframe(df):
        return df.pd_dataframe()

    @gc_collector()
    def fit(self, data, convert_dataframe_kwargs=None, fit_kwargs=None, convert_float32=True):
        """
        Fits the Darts forecasting model on the training data.

        Parameters
        ----------
        data : pd.DataFrame
            The training data in pandas DataFrame format.
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

        data = self.convert2dts_dataframe(data, time_col=self.all_configs['time_col'],
                                          target_col=self.all_configs['target_col'],
                                          **convert_dataframe_kwargs)

        if convert_float32:
            data = data.astype(np.float32)

        self.model.fit(data, **fit_kwargs)
        return self

    def predict(self, n, data=None, predict_kwargs=None, convert_dataframe_kwargs=None, convert_float32=True):
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
        if func_has_params(self.model.predict, 'series') and data is not None:
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


class GBDTModelMixin:
    def __init__(self, time_col, target_col):
        self.all_configs = {'model_configs': {}}
        self.sorted_cols = [
            time_col,
            target_col,
            f"{target_col}_lower",
            f"{target_col}_upper"
        ]

    def chosen_cols(self, data):
        if all(i in data.columns for i in self.sorted_cols):
            return data[self.sorted_cols]
        else:
            return data[[self.all_configs['time_col'],
                         *[i for i in data.columns if i != self.all_configs['time_col']]]]

    def _define_model(self):
        raise NotImplementedError


class StatisticModelMixin:
    def __init__(self, time_col, target_col):
        self.all_configs = {'model_configs': {}}
        self.sorted_cols = [
            time_col,
            target_col,
            f"{target_col}_lower",
            f"{target_col}_upper"
        ]

    def chosen_cols(self, data):
        if all(i in data.columns for i in self.sorted_cols):
            return data[self.sorted_cols]
        else:
            return data[[self.all_configs['time_col'],
                         *[i for i in data.columns if i != self.all_configs['time_col']]]]

    def _define_model(self):
        raise NotImplementedError


class NNModelMixin:
    def __init__(self, time_col, target_col, accelerator=None):
        self.all_configs = {'model_configs': {}}
        if accelerator is None:
            if sys.platform == 'darwin':
                self.accelerator = 'cpu'
            else:
                self.accelerator = 'auto'
        else:
            self.accelerator = accelerator

        self.sorted_cols = [
            time_col,
            target_col,
            f"{target_col}_lower",
            f"{target_col}_upper"
        ]

    def chosen_cols(self, data):
        if all(i in data.columns for i in self.sorted_cols):
            return data[self.sorted_cols]
        else:
            return data[[self.all_configs['time_col'],
                         *[i for i in data.columns if i != self.all_configs['time_col']]]]

    def _define_model(self):
        raise NotImplementedError


class IntervalEstimationMixin:
    def check_data(self, data):
        from darts.timeseries import TimeSeries
        if isinstance(data, TimeSeries):
            data = data.pd_dataframe()

        if len(data) < 2 * self.all_configs['lags']:
            raise ValueError("data length must be greater than or equal to 2 * lags.")

    def _split_train_valid_data(self, data, cv=5, is_prophet=False):
        self.check_data(data)

        if is_prophet:
            data = data[['ds', 'y']]
        else:
            data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        from mapie.subsample import BlockBootstrap

        cv = BlockBootstrap(n_resamplings=cv, length=self.all_configs['lags'], random_state=0)

        for train_index, test_index in cv.split(data):
            if len(test_index) > 0 and len(train_index) >= self.all_configs['lags']:
                yield (data.iloc[train_index, :].reset_index(drop=True),
                       data.iloc[test_index, :].reset_index(drop=True))

    @gc_collector(1)
    def calculate_confidence_interval_darts(self, data, cv=5, fit_kwargs=None, convert2dts_dataframe_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        if convert2dts_dataframe_kwargs is None:
            convert2dts_dataframe_kwargs = {}

        residuals = []

        for train_data, valid_data in self._split_train_valid_data(data, cv=cv):
            valid_y = valid_data[[self.all_configs['target_col']]].values

            train_ds = self.convert2dts_dataframe(
                data.reset_index(drop=True),
                time_col=self.all_configs['time_col'],
                target_col=self.all_configs['target_col'],
                **convert2dts_dataframe_kwargs
            ).astype(np.float32)

            model = self._define_model()

            model.fit(train_ds, **fit_kwargs)

            res = model.predict(len(valid_y)).pd_dataframe()[self.all_configs['target_col']].values

            y_cal_error = wmape(valid_y.flatten(), res.flatten())

            residuals.append(y_cal_error)

            del train_data, train_ds, valid_data, valid_y, model, y_cal_error, res

        quantile = np.percentile(residuals, q=self.all_configs['quantile'])

        return quantile

    @gc_collector(1)
    def _calculate_confidence_interval_sps(self, data, cv=5, fit_kwargs=None, train_data_process_kwargs=None,
                                           valid_data_process_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        if train_data_process_kwargs is None:
            train_data_process_kwargs = {}

        if valid_data_process_kwargs is None:
            valid_data_process_kwargs = {}

        residuals = []
        for train_data, valid_data in self._split_train_valid_data(data, cv=cv):
            data_x, data_y = self._data_preprocess(train_data, **train_data_process_kwargs)

            valid_data_x, valid_data_y = self._data_preprocess(valid_data, **valid_data_process_kwargs)

            model = self._define_model()

            if func_has_params(model.fit, 'eval_set'):
                model.fit(data_x, data_y, eval_set=[(data_x, data_y)], **fit_kwargs)
            else:
                model.fit(data_x, data_y, **fit_kwargs)

            res = model.predict(valid_data_x).flatten()

            y_cal_error = wmape(valid_data_y.flatten(), res.flatten())

            residuals.append(y_cal_error)

            del train_data, valid_data, data_x, data_y, valid_data_x, valid_data_y, model, res, y_cal_error

        quantile = np.percentile(residuals, q=self.all_configs['quantile'])

        return quantile

    def calculate_confidence_interval_mor(self, data, cv=5, fit_kwargs=None):
        return self._calculate_confidence_interval_sps(data, fit_kwargs=fit_kwargs, cv=cv)

    def calculate_confidence_interval_gbrt(self, data, cv=5, fit_kwargs=None):

        return self._calculate_confidence_interval_sps(data, fit_kwargs=fit_kwargs,
                                                       train_data_process_kwargs={'mode': 'train'},
                                                       valid_data_process_kwargs={'mode': 'train'},
                                                       cv=cv)

    def calculate_confidence_interval_nn(self, data, cv=5, fit_kwargs=None):
        if fit_kwargs is None:
            kwargs = {}
        else:
            kwargs = deepcopy(fit_kwargs)

        kwargs.update({'verbose': False})

        return self._calculate_confidence_interval_sps(
            data, fit_kwargs=kwargs, train_data_process_kwargs={'mode': 'train', 'update_last_data': False},
            valid_data_process_kwargs={'mode': 'train', 'update_last_data': False}, cv=cv)

    @gc_collector(1)
    def calculate_confidence_interval_prophet(self, data, cv=5, freq='D', fit_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        residuals = []
        for train_data, valid_data in self._split_train_valid_data(data, cv=cv, is_prophet=True):
            train_ds = train_data[['ds', 'y']]

            valid_data_y = valid_data['y'].values

            model = self._define_model()

            model.fit(train_ds, **fit_kwargs)

            res = model.predict(
                self.model.make_future_dataframe(
                    periods=len(valid_data_y),
                    freq=freq,
                    include_history=False,
                ))['yhat'].values

            y_cal_error = wmape(valid_data_y.flatten(), res.flatten())

            residuals.append(y_cal_error)

            del train_data, valid_data, train_ds, valid_data_y, model, res, y_cal_error

        quantile = np.percentile(residuals, q=self.all_configs['quantile'])

        return quantile

    def interval_predict(self, res):
        res[f"{self.all_configs['target_col']}_lower"] = \
            res[self.all_configs['target_col']].values * (1 - self.all_configs['quantile_error'])
        res[f"{self.all_configs['target_col']}_upper"] = \
            res[self.all_configs['target_col']].values * (1 + self.all_configs['quantile_error'])

        return self.chosen_cols(res)
