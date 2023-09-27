from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


class ProphetModel:
    def __init__(self, data, country_holidays=None):
        self.model = Prophet(holidays=country_holidays)
        self.model.fit(data)

    def predict(self, num_days):
        """
        Predicts the future values of the time series.
        :param num_days: Number of days to predict.
        :return: The predicted values.
        """