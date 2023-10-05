from spinesTS.pipeline import Pipeline
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.asserts import augmented_isinstance
from spinesUtils.utils import Logger
import pandas as pd

from PipelineTS.statistic_model import ProphetModel


class BasePipeline:
    @ParameterTypeAssert({
        'df': pd.DataFrame,
        'target_col': str,
        'time_col': str,
        'forward_windows': int,
        'random_state': int,
        'n_jobs': int
    })
    def __init__(
            self,
            df,
            target_col,
            time_col,
            forward_windows,
            random_state=0,
            n_jobs=-1
    ):
        self.df = df
        self.target_col = target_col
        self.time_col = time_col
        self.forward_windows = forward_windows
        self.backward_windows = forward_windows
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self):
        ...

    def predict(self):
        ...

    def partial_predict(self, n):
        ...


class PipelineTS:
    def __init__(self, **configs):
        self.configs = configs

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
        """
        ...

    def predict(self, X):
        ...

    def partial_predict(self, n):
        ...
