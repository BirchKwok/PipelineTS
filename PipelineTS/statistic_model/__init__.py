from PipelineTS.statistic_model.auto_arima import AutoARIMAModel

try:
    from prophet import Prophet

    from PipelineTS.statistic_model.prophet import ProphetModel
except ImportError:
    ...
