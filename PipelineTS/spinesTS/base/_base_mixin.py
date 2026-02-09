import numpy as np
from spinesUtils.asserts import raise_if_not
from spinesUtils.feature_tools import vars_threshold, variation_threshold

from PipelineTS.spinesTS.metrics import r2_score


class ForecastingMixin:
    def extend_predict(self, x, n):
        """Extrapolation prediction.

        Parameters
        ----------
        x: to_predict data, 2D (batch, seq_len) or 3D (batch, seq_len, n_vars)
        n: predict steps, must be int

        Returns
        -------
        np.ndarray, 2D or 3D depending on input

        """
        raise_if_not(ValueError, isinstance(n, int), "n must be int")
        raise_if_not(ValueError, x.ndim in (2, 3), "x must be 2 or 3 dims data")

        is_multivariate = x.ndim == 3
        # seq_len axis is always axis=1
        seq_axis = 1

        current_res = self.predict(x)

        if n is None:
            return current_res

        pred_len = current_res.shape[seq_axis]

        if n <= pred_len:
            return current_res[:, :n] if not is_multivariate else current_res[:, :n, :]
        else:
            res = [current_res]
            for i in range((n // pred_len) + 1):
                current_res = self.predict(x)
                res.append(current_res)
                # Slide the window: drop oldest pred_len steps, append new prediction
                if is_multivariate:
                    x = np.concatenate((x[:, pred_len:, :], current_res), axis=seq_axis)
                else:
                    x = np.concatenate((x[:, pred_len:], current_res), axis=seq_axis)

            res = np.concatenate(res, axis=seq_axis)
            return res[:, :n] if not is_multivariate else res[:, :n, :]

    def score(self, x, y, eval2d=True):
        pred = self.predict(x)
        if pred.ndim == 3:
            # Multivariate: flatten to 2D for scoring
            B, L, N = pred.shape
            return r2_score(y.reshape(B, -1).T, pred.reshape(B, -1).T)
        if eval2d:
            return r2_score(y.T, pred.T)
        else:
            return r2_score(y, pred)
