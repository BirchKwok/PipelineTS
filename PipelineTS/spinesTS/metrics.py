import numpy as np
from sklearn.metrics import *
from torch import nn
import torch


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def rmse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs) ** 0.5


def mae(*args, **kwargs):
    return mean_absolute_error(*args, **kwargs)


def mse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs)


class WMAPELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(WMAPELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.abs(inputs - targets).sum() / (torch.abs(targets).sum() + 1e-8)


class CombinedQuantileLoss(nn.Module):
    """Combined pinball (quantile) loss for Conformalized Quantile Regression.

    Expects predictions of shape (B, 3*F) where F = target features,
    laid out as [q_lower | q_median | q_upper], and targets of shape (B, F).
    Automatically detects argument order.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.quantiles = [alpha / 2.0, 0.5, 1.0 - alpha / 2.0]

    def _pinball(self, pred, target, tau):
        errors = target - pred
        return torch.mean(torch.max(tau * errors, (tau - 1.0) * errors))

    def forward(self, a, b):
        # Detect which argument is predictions (3x size) vs targets
        if a.shape[-1] > b.shape[-1]:
            preds, targets = a, b
        elif b.shape[-1] > a.shape[-1]:
            preds, targets = b, a
        else:
            # Same size — fall back to (preds, targets) convention
            preds, targets = a, b

        f = targets.shape[-1]
        q_lower = preds[..., :f]
        q_median = preds[..., f:2 * f]
        q_upper = preds[..., 2 * f:]

        loss = (self._pinball(q_lower, targets, self.quantiles[0])
                + self._pinball(q_median, targets, self.quantiles[1])
                + self._pinball(q_upper, targets, self.quantiles[2]))
        return loss / 3.0


class RMSELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))
