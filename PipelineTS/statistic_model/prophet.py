import logging
import random

import numpy as np
from prophet import Prophet
from spinesUtils import generate_function_kwargs

from PipelineTS.base import StatisticModelMixin

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class ProphetModel(StatisticModelMixin):
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
        super().__init__()

        self.all_configs['model_configs'] = generate_function_kwargs(
            Prophet,
            interval_width=quantile,
            holidays=country_holidays,
            **prophet_configs
        )

        self.model = Prophet(**self.all_configs['model_configs'])

        self.all_configs.update({
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'random_state': random_state,
            'lags': lags,  # meanness, but only to follow coding conventions
        })

        random.seed(random_state)
        np.random.seed(random_state)

    @staticmethod
    def _prophet_preprocessing(df, time_col, target_col):
        df_ = df[[time_col, target_col]]
        if 'ds' != time_col or 'y' != target_col:
            df_ = df_.rename(columns={time_col: 'ds', target_col: 'y'})

        return df_

    def fit(self, data, **kwargs):
        data = self._prophet_preprocessing(data, self.all_configs['time_col'], self.all_configs['target_col'])
        self.model.fit(data, **kwargs)
        return self

    def predict(self, num_days, freq='D', include_history=False):
        return self.model.predict(
            self.model.make_future_dataframe(
                periods=num_days,
                freq=freq,
                include_history=include_history,
            )
        )[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
            columns={
                'ds': self.all_configs['time_col'],
                'yhat': self.all_configs['target_col'],
                'yhat_lower': f"{self.all_configs['target_col']}_lower",
                'yhat_upper': f"{self.all_configs['target_col']}_upper"
            }
        )
