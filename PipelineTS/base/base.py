import sys
from copy import deepcopy

import numpy as np
from spinesUtils.preprocessing import gc_collector

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

        signed_residuals = []
        for train_data, valid_data in self._split_train_valid_data(data, cv=cv, is_gbrt=is_gbrt):
            data_x, data_y = self._data_preprocess(train_data, **train_data_process_kwargs)

            valid_data_x, valid_data_y = self._data_preprocess(valid_data, **valid_data_process_kwargs)

            model = self._define_model()

            if check_has_param(model.fit, 'eval_set'):
                model.fit(data_x, data_y, eval_set=[(data_x, data_y)], **fit_kwargs)
            else:
                model.fit(data_x, data_y, **fit_kwargs)

            preds = model.predict(valid_data_x).flatten()
            actuals = valid_data_y.flatten()

            per_point_residuals = actuals - preds
            signed_residuals.extend(per_point_residuals.tolist())

            del train_data, valid_data, data_x, data_y, valid_data_x, valid_data_y, model, preds, actuals

        return self._compute_conformal_quantiles(
            signed_residuals, coverage=self.all_configs['quantile']
        )

    @staticmethod
    def _compute_conformal_quantiles(signed_residuals, coverage=0.9):
        """Compute asymmetric conformal quantiles from signed residuals.

        Uses the split conformal prediction framework with finite-sample
        correction to produce asymmetric (lower, upper) interval widths.

        Parameters
        ----------
        signed_residuals : list of float
            Per-point signed residuals (y_true - y_pred) collected from
            cross-validation folds.
        coverage : float, default 0.9
            Desired coverage level (e.g. 0.9 for 90% prediction interval).
            Corresponds to the ``quantile`` parameter in model configs.

        Returns
        -------
        tuple of (float, float)
            (q_lower, q_upper) where q_lower <= 0 and q_upper >= 0.
            Prediction intervals: [pred + q_lower, pred + q_upper].
        """
        if len(signed_residuals) == 0:
            return (0.0, 0.0)

        residuals = np.array(signed_residuals)
        n_cal = len(residuals)

        alpha = 1.0 - coverage
        # Conformal finite-sample correction: adjust quantile levels
        # so that coverage guarantee holds for n_cal calibration points
        q_lo_level = max(alpha / 2.0, 0.5 / n_cal)
        q_hi_level = min(1.0 - alpha / 2.0, 1.0 - 0.5 / n_cal)
        # Additional conformal correction: (1 + 1/n_cal) factor
        q_hi_level = min(1.0, q_hi_level * (1.0 + 1.0 / n_cal))

        q_lower = np.quantile(residuals, q=q_lo_level)
        q_upper = np.quantile(residuals, q=q_hi_level)

        # Ensure q_lower <= 0 <= q_upper for valid intervals
        q_lower = min(q_lower, 0.0)
        q_upper = max(q_upper, 0.0)

        return (float(q_lower), float(q_upper))

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
        """Apply conformal prediction intervals.

        Uses additive asymmetric intervals based on per-point signed residual
        quantiles from cross-validation calibration.

        Parameters
        ----------
        res : pd.DataFrame
            DataFrame with point predictions in target_col.

        Returns
        -------
        pd.DataFrame
            DataFrame with added _lower and _upper columns.
        """
        q_lower, q_upper = self.all_configs['quantile_error']

        res[f"{self.all_configs['target_col']}_lower"] = \
            res[self.all_configs['target_col']].values + q_lower
        res[f"{self.all_configs['target_col']}_upper"] = \
            res[self.all_configs['target_col']].values + q_upper

        return self.chosen_cols(res)
