import logging

import numpy as np
from prophet import Prophet
from spinesUtils import generate_function_kwargs
from sklearn.model_selection import TimeSeriesSplit

from PipelineTS.base import StatisticModelMixin, IntervalEstimationMixin

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class ProphetModel(StatisticModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=None,
            country_holidays=None,
            quantile=0.9,
            random_state=0,
            **prophet_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            Prophet,
            holidays=country_holidays,
            **prophet_configs
        )

        self.model = Prophet(**self.all_configs['model_configs'])

        self.all_configs.update({
            'quantile': quantile,
            'quantile_error': 0,
            'time_col': time_col,
            'target_col': target_col,
            'random_state': random_state,  # meanness, but only to follow coding conventions
            'lags': lags,  # meanness, but only to follow coding conventions
        })

    @staticmethod
    def _prophet_preprocessing(df, time_col, target_col):
        df_ = df[[time_col, target_col]]
        if 'ds' != time_col or 'y' != target_col:
            df_ = df_.rename(columns={time_col: 'ds', target_col: 'y'})

        return df_

    def fit(self, data, freq='D', cv=5, fit_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}
        data = self._prophet_preprocessing(data, self.all_configs['time_col'], self.all_configs['target_col'])
        self.model.fit(data, **fit_kwargs)

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_prophet(data, cv=cv, fit_kwargs=fit_kwargs)
        return self

    def calculate_confidence_interval_prophet(self, data, cv=5, freq='D', fit_kwargs=None):
        if fit_kwargs is None:
            fit_kwargs = {}

        tscv = TimeSeriesSplit(n_splits=cv)

        data = data[['ds', 'y']]

        residuals = []

        for (train_index, test_index) in tscv.split(data):
            train_ds = data.iloc[train_index, :]

            test_v = data['y'].iloc[test_index].values
            model = Prophet(**self.all_configs['model_configs'])

            model.fit(train_ds, **fit_kwargs)

            res = model.predict(
                self.model.make_future_dataframe(
                    periods=len(test_v),
                    freq=freq,
                    include_history=False,
                ))[['ds', 'yhat']]

            error_rate = np.abs((res['yhat'].values - test_v) / test_v)
            error_rate = np.where((error_rate == np.inf) | (error_rate == np.nan), 0., error_rate)

            residuals.extend(error_rate.tolist())

        quantile = np.percentile(residuals, self.all_configs['quantile'])
        if isinstance(quantile, (list, np.ndarray)):
            quantile = quantile[0]

        return quantile

    def predict(self, num_days, freq='D', include_history=False):
        res = self.model.predict(
            self.model.make_future_dataframe(
                periods=num_days,
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
