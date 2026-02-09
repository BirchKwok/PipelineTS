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
        if len(data) < 2 * self.all_configs['lags']:
            raise ValueError("data length must be greater than or equal to 2 * lags.")

    def _split_train_valid_data(self, data, cv=5, is_gbrt=False):
        self.check_data(data)

        if is_gbrt:
            ...
        else:
            data = data[[self.all_configs['time_col'], self.all_configs['target_col']]]

        n = len(data)
        block_len = self.all_configs['lags']
        rng = np.random.RandomState(0)

        for _ in range(cv):
            # Block bootstrap: sample blocks of length `block_len` to form train set
            n_blocks = max(1, n // block_len)
            # Use ~80% of blocks for training, rest for validation
            all_block_starts = np.arange(0, n - block_len + 1)
            if len(all_block_starts) == 0:
                continue

            chosen = rng.choice(len(all_block_starts), size=n_blocks, replace=True)
            train_indices = set()
            for c in chosen:
                start = all_block_starts[c]
                for j in range(start, min(start + block_len, n)):
                    train_indices.add(j)

            all_indices = set(range(n))
            test_indices = sorted(all_indices - train_indices)
            train_indices = sorted(train_indices)

            if len(test_indices) > 0 and len(train_indices) >= block_len:
                yield (data.iloc[train_indices, :].reset_index(drop=True),
                       data.iloc[test_indices, :].reset_index(drop=True))

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

    def interval_predict(self, res):
        res[f"{self.all_configs['target_col']}_lower"] = \
            res[self.all_configs['target_col']].values * (1 - self.all_configs['quantile_error'])
        res[f"{self.all_configs['target_col']}_upper"] = \
            res[self.all_configs['target_col']].values * (1 + self.all_configs['quantile_error'])

        return self.chosen_cols(res)
