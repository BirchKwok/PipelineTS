import numpy as np
from spinesUtils.asserts import raise_if_not
from spinesUtils.feature_tools import vars_threshold, variation_threshold

from PipelineTS.spinesTS.metrics import r2_score


class ForecastingMixin:
    def extend_predict(self, x, n):
        """Extrapolation prediction.

        Parameters
        ----------
        x: to_predict data, must be 2 dims data
        n: predict steps, must be int

        Returns
        -------
        np.ndarray, which has 2 dims

        """
        raise_if_not(ValueError, isinstance(n, int), "n must be int")
        raise_if_not(ValueError, x.ndim == 2, "x must be 2 dims data")

        current_res = self.predict(x)

        if n is None:
            return current_res
        elif n <= current_res.shape[1]:
            return current_res[:, :n]
        else:
            res = [current_res]
            for i in range((n//res[0].shape[1])+1):
                current_res = self.predict(x)
                res.append(current_res)
                x = np.concatenate((x[:, current_res.shape[1]:], current_res), axis=-1)

            res = np.concatenate(res, axis=-1)[:, :n]

            return res

    def score(self, x, y, eval2d=True):
        if eval2d:
            return r2_score(y.T, self.predict(x).T)
        else:
            return r2_score(y, self.predict(x))
