import sys
from copy import deepcopy

import numpy as np
from spinesUtils.preprocessing import gc_collector

from PipelineTS.spinesTS.metrics import wmape
from spinesUtils.asserts import check_has_param


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

    def _split_train_valid_data(self, data, cv=5, is_prophet=False, is_gbrt=False):
        self.check_data(data)

        if is_prophet:
            data = data[['ds', 'y']]
        elif is_gbrt:
            ...
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
                                           valid_data_process_kwargs=None, is_gbrt=False):
        if fit_kwargs is None:
            fit_kwargs = {}

        if train_data_process_kwargs is None:
            train_data_process_kwargs = {}

        if valid_data_process_kwargs is None:
            valid_data_process_kwargs = {}

        residuals = []
        for train_data, valid_data in self._split_train_valid_data(data, cv=cv, is_gbrt=is_gbrt):
            data_x, data_y = self._data_preprocess(train_data, **train_data_process_kwargs)

            valid_data_x, valid_data_y = self._data_preprocess(valid_data, **valid_data_process_kwargs)

            model = self._define_model()

            if check_has_param(model.fit, 'eval_set'):
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
                                                       cv=cv, is_gbrt=True)

    def calculate_confidence_interval_nn(self, data, cv=5, fit_kwargs=None):
        if fit_kwargs is None:
            kwargs = {}
        else:
            kwargs = deepcopy(fit_kwargs)

        kwargs.update({'verbose': False})

        return self._calculate_confidence_interval_sps(
            data, fit_kwargs=kwargs, train_data_process_kwargs={'mode': 'train'},
            valid_data_process_kwargs={'mode': 'train'}, cv=cv)

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
