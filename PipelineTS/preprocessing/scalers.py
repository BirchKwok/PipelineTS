from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from spinesTS.preprocessing import GaussRankScaler


class Scaler:
    def __init__(self, scaler_name='min_max'):
        """define scaler.

        :param scaler_name: str, default to 'min_max'. It can be 'min_max', 'gauss_rank', 'quantile', 'standard'.
                            The 'min_max' is for sklearn.preprocessing.MinMaxScaler,
                            the 'gauss_rank' is for spinesTS.preprocessing.GaussRankScaler,
                            the 'quantile' is for sklearn.preprocessing.QuantileTransformer,
                            the 'standard' is for sklearn.preprocessing.StandardScaler.
        """
        self.scaler_name = scaler_name

        if self.scaler_name == 'min_max':
            self.scaler = MinMaxScaler()
        elif self.scaler_name == 'gauss_rank':
            self.scaler = GaussRankScaler()
        elif self.scaler_name == 'quantile':
            self.scaler = QuantileTransformer()
        elif self.scaler_name == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError('Unknown scaler name')

    def fit(self, X):
        self.scaler.fit(X)

        return self

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)
