from darts.models import AutoARIMA
from spinesUtils import generate_function_kwargs

from PipelineTS.base import DartsForecastMixin, StatisticModelMixin, IntervalEstimationMixin


class AutoARIMAModel(DartsForecastMixin, StatisticModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            start_p=8,
            max_p=12,
            start_q=1,
            seasonal=False,
            quantile=0.9,
            seasonal_length=12,
            n_jobs=-1,
            **darts_auto_arima_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        self.all_configs['model_configs'] = generate_function_kwargs(
            AutoARIMA,
            start_p=start_p,
            max_p=max_p,
            start_q=start_q,
            seasonal=seasonal,
            seasonal_length=seasonal_length,
            n_jobs=n_jobs,
            **darts_auto_arima_configs
        )

        self.model = self._define_model()

        self.all_configs.update(
            {
                'lags': lags,   # meanness, but only to follow coding conventions
                'quantile': quantile,
                'time_col': time_col,
                'target_col': target_col,
                'quantile_error': 0
            }
        )

    def _define_model(self):
        return AutoARIMA(**self.all_configs['model_configs'])

    def fit(self, data, convert_dataframe_kwargs=None, cv=5, fit_kwargs=None):
        super().fit(
            data,
            convert_dataframe_kwargs=convert_dataframe_kwargs,
            fit_kwargs=fit_kwargs
        )

        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self.calculate_confidence_interval_darts(data, fit_kwargs=fit_kwargs,
                                                         convert2dts_dataframe_kwargs=convert_dataframe_kwargs, cv=cv)

        return self

    def predict(self, n, **kwargs):
        res = super().predict(n, **kwargs)
        res = self.rename_prediction(res)

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
