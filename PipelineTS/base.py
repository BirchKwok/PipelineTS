import sys

from spinesUtils.asserts import *
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from copy import deepcopy


def load_dataset_to_darts(
        data,
        time_col,
        target_col,
        fill_missing_dates=False,
        freq=None,
        fillna_value=None,
        static_covariates=None,
        hierarchy=None
):
    from darts.timeseries import TimeSeries

    return TimeSeries.from_dataframe(
        data,
        time_col=time_col,
        value_cols=target_col,
        fill_missing_dates=fill_missing_dates,
        freq=freq,
        fillna_value=fillna_value,
        static_covariates=static_covariates,
        hierarchy=hierarchy
    )


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

    def fit(self, data, convert_dataframe_kwargs=None, fit_kwargs=None, convert_float32=True):
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

    def predict(self, n, predict_likelihood_parameters=False, **kwargs):
        if 'predict_likelihood_parameters' in get_function_params_name(self.model.predict):
            return self.model.predict(
                n,
                predict_likelihood_parameters=predict_likelihood_parameters,
                **kwargs
            ).pd_dataframe()
        return self.model.predict(n, **kwargs).pd_dataframe()

    def rename_prediction(self, data):
        data.columns.name = None
        data[self.all_configs['time_col']] = data.index.copy()

        data = data.reset_index(drop=True)

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
            if i == f"{self.all_configs['target_col']}_q0.50":
                data.rename(columns={i: f"{self.all_configs['target_col']}"}, inplace=True)

            elif i == f"{self.all_configs['target_col']}_q{right_ratio}":
                data.rename(columns={i: f"{self.all_configs['target_col']}_upper"}, inplace=True)

            elif i == f"{self.all_configs['target_col']}_q{left_ratio}":
                data.rename(columns={i: f"{self.all_configs['target_col']}_lower"}, inplace=True)

        chosen_cols = [
            self.all_configs['time_col'],
            f"{self.all_configs['target_col']}",
            f"{self.all_configs['target_col']}_lower",
            f"{self.all_configs['target_col']}_upper"
        ]

        if all(i in data.columns for i in chosen_cols):
            return data[chosen_cols]
        else:
            return data[[self.all_configs['time_col'],
                         *[i for i in data.columns if i != self.all_configs['time_col']]]]


class GBDTModelMixin:
    def __init__(self):
        self.all_configs = {'model_configs': {}}


class StatisticModelMixin:
    def __init__(self):
        self.all_configs = {'model_configs': {}}


class NNModelMixin:
    def __init__(self, device=None):
        self.all_configs = {'model_configs': {}}
        if device is None:
            if sys.platform == 'darwin':
                self.device = 'cpu'
            else:
                self.device = 'auto'
        else:
            self.device = device


class IntervalEstimationMixin:
    def calculate_confidence_interval(self, data, estimator, cv=5,
                                      fit_kwargs=None, convert2dts_dataframe=True):
        if fit_kwargs is None:
            fit_kwargs = {}

        tscv = TimeSeriesSplit(n_splits=cv)

        data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        residuals = []

        for (train_index, test_index) in tscv.split(data):
            if convert2dts_dataframe:
                train_ds = self.convert2dts_dataframe(
                    data.iloc[train_index, :],
                    time_col=self.all_configs['time_col'],
                    target_col=self.all_configs['target_col']
                ).astype(np.float32)
            else:
                train_ds = data.iloc[train_index, :].astype(np.float32)

            test_v = data[self.all_configs['target_col']].iloc[test_index].values
            est = deepcopy(estimator)
            model = est(**self.all_configs['model_configs'])

            model.fit(train_ds, **fit_kwargs)

            res = model.predict(len(test_v)).pd_dataframe()

            error_rate = np.abs((res[self.all_configs['target_col']].values - test_v) / test_v)
            error_rate = np.where((error_rate == np.inf) | (error_rate == np.nan), 0., error_rate)

            residuals.extend(error_rate.tolist())

        quantile = np.percentile(residuals, self.all_configs['quantile'])
        if isinstance(quantile, (list, np.ndarray)):
            quantile = quantile[0]

        return quantile

    def interval_predict(self, res):
        res[f"{self.all_configs['target_col']}_lower"] = \
            res[self.all_configs['target_col']].values * (1 - self.all_configs['quantile_error'])
        res[f"{self.all_configs['target_col']}_upper"] = \
            res[self.all_configs['target_col']].values * (1 + self.all_configs['quantile_error'])

        chosen_cols = [
            self.all_configs['time_col'],
            f"{self.all_configs['target_col']}",
            f"{self.all_configs['target_col']}_lower",
            f"{self.all_configs['target_col']}_upper"
        ]

        return res[chosen_cols]
