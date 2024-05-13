from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from PipelineTS.spinesTS.preprocessing import GaussRankScaler


class Scaler:
    def __init__(self, scaler_name='min_max'):
        """
        Initialize a scaler object.

        Parameters
        ----------
        scaler_name : str, default='min_max'
            The name of the scaler. It can be 'min_max', 'gauss_rank', 'quantile', or 'standard'.
            - 'min_max': MinMaxScaler from sklearn.preprocessing.
            - 'gauss_rank': GaussRankScaler from PipelineTS.spinesTS.preprocessing.
            - 'quantile': QuantileTransformer from sklearn.preprocessing.
            - 'standard': StandardScaler from sklearn.preprocessing.
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
        """
        Fit the scaler to the input data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data.

        Returns
        -------
        self : Scaler
            The fitted scaler object.
        """
        self.scaler.fit(X)
        return self

    def fit_transform(self, X):
        """
        Fit the scaler to the input data and transform it.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data.

        Returns
        -------
        transformed_data : array-like
            The transformed data.
        """
        return self.scaler.fit_transform(X)

    def transform(self, X):
        """
        Transform the input data using the fitted scaler.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input data.

        Returns
        -------
        transformed_data : array-like
            The transformed data.
        """
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        """
        Inverse transform the input data using the fitted scaler.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Transformed data.

        Returns
        -------
        original_data : array-like
            The original data before scaling.
        """
        return self.scaler.inverse_transform(X)
