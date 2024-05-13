import numpy as np
import pandas as pd
import scipy.stats
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from spinesUtils.asserts import raise_if_not

from PipelineTS.spinesTS.utils import check_is_fitted
from PipelineTS.spinesTS.preprocessing import split_series, lag_splits, moving_average
from PipelineTS.spinesTS.features_generator import DateExtendFeatures


class GBRTPreprocessing:
    """WideGBRT features engineering class"""
    def __init__(
        self, in_features, out_features, target_col,
        train_size=0.8, date_col=None, differential_n=0, moving_avg_n=0,
        extend_daily_target_features=True, use_scale=True
     ):
        self.cf = None
        self.input_features = in_features
        self.output_features = out_features
        self.target_col = target_col
        self.train_size = train_size
        self.date_col = date_col

        self.use_scale = use_scale

        self.extend_daily_target_features = extend_daily_target_features

        if self.extend_daily_target_features and self.use_scale:
            self.extend_daily_target_features_scaler = MinMaxScaler()

        # if self.use_scale:
        #     self.original_scaler = MinMaxScaler()

        raise_if_not(ValueError, isinstance(differential_n, int) and differential_n >= 0,
                     "differential_n must be a non-negative integer.")
        self.differential_n = differential_n

        raise_if_not(ValueError, isinstance(moving_avg_n, int) and moving_avg_n >= 0,
                     "moving_avg_n must be a non-negative integer.")
        self.moving_avg_n = moving_avg_n
        self.x_shape = None

        self.date_features_preprocessor = DateExtendFeatures(date_col=self.date_col, drop_date_col=True,
                                                             use_scale=self.use_scale)

        self.__spinesTS_is_fitted__ = False

    def process_date_col(self, x):
        """Processing date column"""
        return self.date_features_preprocessor.transform(x)

    def process_target_col(self, x, fit=False):
        if not x.ndim == 2:
            x = x.reshape(1, -1)

        mean_res = x.mean(axis=1).reshape((-1, 1))
        median_res = np.percentile(x, q=50, axis=1).reshape((-1, 1))
        min_res = x.min(axis=1).reshape((-1, 1))
        max_res = x.max(axis=1).reshape((-1, 1))
        p25 = np.percentile(x, q=25, axis=1).reshape((-1, 1))
        p75 = np.percentile(x, q=75, axis=1).reshape((-1, 1))
        std = np.std(x, axis=1).reshape((-1, 1))
        entropy = scipy.stats.entropy(x, base=2, axis=1).reshape((-1, 1))
        avg_diff = np.diff(x, n=1, axis=1).mean(axis=1).reshape((-1, 1))
        avg_abs_diff = np.abs(np.diff(x, n=1, axis=1)).mean(axis=1).reshape((-1, 1))
        avg_median_diff = np.percentile(np.diff(x, n=1, axis=1), q=50, axis=1).reshape((-1, 1))
        avg_abs_median_diff = np.percentile(np.abs(np.diff(x, n=1, axis=1)), q=50, axis=1).reshape((-1, 1))

        autocorrelation = scipy.signal.correlate(x, x, mode='same')
        autocorrelation_diff = scipy.signal.correlate(x, np.diff(x, n=1, axis=1), mode='same')

        percentile_count_under_75 = ((x < np.percentile(x, q=75, axis=1).reshape((-1, 1))).sum(axis=1)
                                     .astype(int).reshape((-1, 1)))
        percentile_count_under_25 = ((x < np.percentile(x, q=25, axis=1).reshape((-1, 1))).sum(axis=1)
                                     .astype(int).reshape((-1, 1)))
        percentile_count_under_90 = ((x < np.percentile(x, q=90, axis=1).reshape((-1, 1))).sum(axis=1)
                                     .astype(int).reshape((-1, 1)))
        percentile_count_over_90 = ((x > np.percentile(x, q=90, axis=1).reshape((-1, 1))).sum(axis=1)
                                    .astype(int).reshape((-1, 1)))

        peaks_count = np.array([len(signal.find_peaks(row)[0]) for row in x]).reshape((-1, 1))
        large_distance_peaks_count = np.array([len(signal.find_peaks(row, distance=150)[0]) for row in x]).reshape(
            (-1, 1))

        final_matrix = np.concatenate(
            (mean_res, median_res, max_res, min_res, p25, p75, std, entropy,
             avg_diff, avg_abs_diff, avg_median_diff, avg_abs_median_diff, autocorrelation, autocorrelation_diff,
             percentile_count_under_75, percentile_count_under_25, percentile_count_under_90, percentile_count_over_90,
             peaks_count, large_distance_peaks_count), axis=1)

        if self.use_scale:
            if fit:
                self.extend_daily_target_features_scaler.fit(final_matrix)
                return

            return self.extend_daily_target_features_scaler.transform(final_matrix)

        return final_matrix

    def check_x_types(self, x):
        raise_if_not(ValueError, isinstance(x, (pd.DataFrame, np.ndarray)),
                     "Only accept pandas.DataFrame or numpy.ndarray.")

        if not isinstance(x, pd.DataFrame):
            raise_if_not(ValueError, x.ndim == 2, "Only accept two-dim numpy.ndarray.")
            raise_if_not(ValueError, x.shape[1] == self.input_features,
                         "The shape of x does not match the input_features.")
            raise_if_not(ValueError, isinstance(self.target_col, int), "The target_col parameter must be an integer.")
            raise_if_not(ValueError, self.date_col is None or isinstance(self.date_col, int),
                         "The date_col parameter must be an integer or None.")

    def fit(self, x):
        self.check_x_types(x)
        self.x_shape = x.shape[1]
        self.date_features_preprocessor.fit(x)

        if self.use_scale:
            _tar = x[self.target_col].values.squeeze() if isinstance(x, pd.DataFrame) else x[:, self.target_col]
            _tar = lag_splits(
                _tar, window_size=self.input_features, skip_steps=1, pred_steps=1
            )

            if self.extend_daily_target_features:
                self.process_target_col(_tar, fit=True)

            # if self.differential_n > 0:
            #     _tar = np.diff(_tar, axis=1, n=self.differential_n)
            #
            # if self.moving_avg_n > 0:
            #     _tar = moving_average(_tar, window_size=self.moving_avg_n)
            #
            # self.original_scaler.fit(_tar)

        self.__spinesTS_is_fitted__ = True
        return self

    def _transform_train_mode(self, _tar, _non_lag_fea):
        """Transform data to fit WideGBRT model in train mode."""
        if self.train_size is None:
            x, y = split_series(_tar, _tar, self.input_features, self.output_features, train_size=self.train_size)

            if self.extend_daily_target_features:
                tar_fea_x = self.process_target_col(x)

            if self.differential_n > 0:
                x = np.diff(x, axis=1, n=self.differential_n)

            if self.moving_avg_n > 0:
                x = moving_average(x, window_size=self.moving_avg_n)

            # if self.use_scale:
            #     x = self.original_scaler.transform(x)

            x_non_lag, _ = split_series(_non_lag_fea, _tar, self.input_features,
                                        self.output_features, train_size=self.train_size)

            if x_non_lag.shape[1] > 0:
                x_non_lag = self._process_x_non_lag_dim(x_non_lag)
                if self.extend_daily_target_features:
                    x = np.concatenate((x, tar_fea_x, x_non_lag), axis=1)
                else:
                    x = np.concatenate((x, x_non_lag), axis=1)
            else:
                if self.extend_daily_target_features:
                    x = np.concatenate((x, tar_fea_x), axis=1)

            return x, y
        else:
            x_train, x_test, y_train, y_test = split_series(_tar, _tar, self.input_features,
                                                            self.output_features, train_size=self.train_size)
            if self.extend_daily_target_features:
                tar_fea_x_train = self.process_target_col(x_train)
                tar_fea_x_test = self.process_target_col(x_test)

            if self.differential_n > 0:
                x_train = np.diff(x_train, axis=1, n=self.differential_n)
                x_test = np.diff(x_test, axis=1, n=self.differential_n)

            if self.moving_avg_n > 0:
                x_train = moving_average(x_train, window_size=self.moving_avg_n)
                x_test = moving_average(x_test, window_size=self.moving_avg_n)

            # if self.use_scale:
            #     x_train = self.original_scaler.transform(x_train)
            #     x_test = self.original_scaler.transform(x_test)

            x_non_lag_train, x_non_lag_test, _, _ = split_series(_non_lag_fea, _tar, self.input_features,
                                                                 self.output_features, train_size=self.train_size)
            if len(x_non_lag_train) > 0 and len(x_non_lag_test) > 0:
                x_non_lag_train = self._process_x_non_lag_dim(x_non_lag_train)
                x_non_lag_test = self._process_x_non_lag_dim(x_non_lag_test)

                if self.extend_daily_target_features:
                    x_train = np.concatenate((x_train, tar_fea_x_train, x_non_lag_train), axis=1)
                    x_test = np.concatenate((x_test, tar_fea_x_test, x_non_lag_test), axis=1)
                else:
                    x_train = np.concatenate((x_train, x_non_lag_train), axis=1)
                    x_test = np.concatenate((x_test, x_non_lag_test), axis=1)
            else:
                if self.extend_daily_target_features:
                    x_train = np.concatenate((x_train, tar_fea_x_train), axis=1)
                    x_test = np.concatenate((x_test, tar_fea_x_test), axis=1)

            return x_train, x_test, y_train, y_test

    def _transform_predict_mode(self, _tar, _non_lag_fea):
        split_tar = lag_splits(_tar, window_size=self.input_features, skip_steps=1, pred_steps=1)

        if self.extend_daily_target_features:
            tar_fea_x = self.process_target_col(split_tar)

        if self.differential_n > 0:
            split_tar = np.diff(split_tar, axis=1, n=self.differential_n)

        if self.moving_avg_n > 0:
            split_tar = moving_average(split_tar, window_size=self.moving_avg_n)

        # if self.use_scale:
        #     split_tar = self.original_scaler.transform(split_tar)

        split_non_lag_fea = lag_splits(_non_lag_fea, window_size=self.input_features, skip_steps=1, pred_steps=1)

        if len(split_non_lag_fea) > 0:
            split_non_lag_fea = self._process_x_non_lag_dim(split_non_lag_fea)

            if self.extend_daily_target_features:
                x = np.concatenate((split_tar, tar_fea_x, split_non_lag_fea), axis=1)
            else:
                x = np.concatenate((split_tar, split_non_lag_fea), axis=1)
        else:
            x = np.concatenate((split_tar, tar_fea_x), axis=1) if self.extend_daily_target_features \
                else split_tar

        return x

    def transform(self, x, mode='train'):
        """Transform data to fit WideGBRT model.

        result's columns sequence:
        lag_1, lag_2, lag_3, ..., lag_n, x_col_1, x_col_2, ..., x_col_n, date_fea_1, date_fea_2, ..., date_fea_n

        Parameters
        ----------
        x: pandas.core.DataFrame or numpy.ndarray, the data that needs to be transformed
        mode: ('train', 'predict'), the way to transform data, default: 'train'

        Returns
        -------
        numpy.ndarray, x_train, x_test, y_train, y_test, when mode = 'train', else, x, y

        """
        raise_if_not(ValueError, mode in ('train', 'predict'), "mode must be 'train' or 'predict'.")
        check_is_fitted(self)

        self.check_x_types(x)

        if x.shape[1] != self.x_shape:
            raise ValueError("data shape does not match the shape of the data at the time of fitting.")

        _tar = x[self.target_col].values.squeeze() if isinstance(x, pd.DataFrame) else x[:, self.target_col]

        if isinstance(x, pd.DataFrame):
            if self.date_col is not None:
                x = self.process_date_col(x)
            # timestamp features and other features
            _non_lag_fea = x.loc[:, [i for i in x.columns if i != self.target_col]].values
        else:
            if self.date_col is not None:
                x = self.process_date_col(pd.DataFrame(x, columns=range(x.shape[1]))).values

            # timestamp features and other features
            _non_lag_fea = x[:, [i for i in range(x.shape[1]) if i != self.target_col]]

        return self._transform_train_mode(_tar, _non_lag_fea) if mode == 'train' \
            else self._transform_predict_mode(_tar, _non_lag_fea)

    @staticmethod
    def _process_x_non_lag_dim(x):
        if x[:, -1, :].squeeze().ndim == 1 and x[:, -1, :].ndim == 2:
            return x[:, -1, :]
        elif x[:, -1, :].squeeze().ndim == 1:
            return x[:, -1, :].squeeze(1)
        return x[:, -1, :].squeeze()
