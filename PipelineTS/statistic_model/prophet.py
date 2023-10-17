import logging

from spinesUtils import generate_function_kwargs

from PipelineTS.base import StatisticModelMixin, IntervalEstimationMixin

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class ProphetModel(StatisticModelMixin, IntervalEstimationMixin):
    from prophet import Prophet

    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            country_holidays=None,
            quantile=0.9,
            random_state=0,
            **prophet_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)


        self.all_configs['model_configs'] = generate_function_kwargs(
            ProphetModel.Prophet,
            holidays=country_holidays,
            **prophet_configs
        )

        self.model = self._define_model()

        self.all_configs.update({
            'quantile': quantile,
            'quantile_error': 0,
            'time_col': time_col,
            'target_col': target_col,
            'random_state': random_state,  # meanness, but only to follow coding conventions
            'lags': lags,  # meanness, but only to follow coding conventions
        })

    def _define_model(self):
        return ProphetModel.Prophet(**self.all_configs['model_configs'])

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
                self.calculate_confidence_interval_prophet(data, cv=cv, freq=freq, fit_kwargs=fit_kwargs)
        return self

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
