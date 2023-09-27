from spinesTS.pipeline import Pipeline
from spinesUtils import ParameterTypeAssert, ParameterValuesAssert
from spinesUtils.asserts import augmented_isinstance
from spinesUtils.utils import Logger


class PipelineTS:
    def __init__(self):
        ...

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
