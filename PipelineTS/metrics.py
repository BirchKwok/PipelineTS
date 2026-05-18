"""Evaluation metrics for time series point forecasts and prediction intervals.

All functions are vectorized with numpy for performance.
"""

import numpy as np

from spinesUtils.asserts import ParameterTypeAssert


# ---------------------------------------------------------------------------
#  Point forecast metrics
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE value (0–∞). Undefined when y_true contains zeros;
        zeros are excluded from the computation.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        sMAPE value in [0, 2]. Uses denominator (|y| + |ŷ|) / 2.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
    """Mean Absolute Scaled Error.

    Parameters
    ----------
    y_true : array-like
        True values of the test set.
    y_pred : array-like
        Predicted values.
    y_train : array-like
        Training set values (used for the naive scaling denominator).
    seasonality : int, default=1
        Seasonal period for the naive forecast. 1 = non-seasonal naive.

    Returns
    -------
    float
        MASE value. < 1 means better than naive.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        R² value (≤ 1). 1 = perfect, 0 = mean baseline, < 0 = worse than mean.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Median of absolute errors.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
#  Prediction interval metrics
# ---------------------------------------------------------------------------

@ParameterTypeAssert({
    'yt': np.ndarray,
    'left_pred': np.ndarray,
    'right_pred': np.ndarray
})
def quantile_acc(yt: np.ndarray, left_pred: np.ndarray, right_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of prediction intervals.

    Parameters
    ----------
    yt : np.ndarray
        The true values.

    left_pred : np.ndarray
        The left bound of the prediction interval.

    right_pred : np.ndarray
        The right bound of the prediction interval.

    Returns
    -------
    float
        The accuracy of the prediction intervals, computed as the ratio of correct predictions to the total number of samples.
    """
    return np.sum((yt >= left_pred) * (yt <= right_pred)) / yt.shape[0]


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Identical to quantile_acc but provided for standard naming.

    Parameters
    ----------
    y_true : array-like
        True values.
    lower : array-like
        Lower bound of prediction interval.
    upper : array-like
        Upper bound of prediction interval.

    Returns
    -------
    float
        Coverage ratio in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def pinaw(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Normalized Average Width (PINAW).

    Parameters
    ----------
    y_true : array-like
        True values (used for range normalization).
    lower : array-like
        Lower bound of prediction interval.
    upper : array-like
        Upper bound of prediction interval.

    Returns
    -------
    float
        Normalized average interval width. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0:
        return np.nan
    return float(np.mean(upper - lower) / y_range)


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  alpha: float = 0.1) -> float:
    """Winkler interval score.

    Rewards narrow intervals and penalizes missed coverage.

    Parameters
    ----------
    y_true : array-like
        True values.
    lower : array-like
        Lower bound of prediction interval.
    upper : array-like
        Upper bound of prediction interval.
    alpha : float, default=0.1
        Significance level (1 - coverage). E.g. 0.1 for 90% intervals.

    Returns
    -------
    float
        Average Winkler score. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    width = upper - lower
    penalty_lower = (2.0 / alpha) * (lower - y_true)
    penalty_upper = (2.0 / alpha) * (y_true - upper)

    score = width.copy()
    below = y_true < lower
    above = y_true > upper
    score[below] += penalty_lower[below]
    score[above] += penalty_upper[above]

    return float(np.mean(score))

# ---------------------------------------------------------------------------
#  Compatibility point metrics and torch losses used by PipelineTS models
# ---------------------------------------------------------------------------

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error.

    Computes ``sum(|y - ŷ|) / sum(|y|)``, a scale-free, outlier-robust
    alternative to MAPE that weights each error by the corresponding true
    value.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        wMAPE value in [0, ∞). Returns ``nan`` when all true values are zero.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAE value (≥ 0).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MSE value (≥ 0). Penalizes large errors more than MAE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE value (≥ 0). Same unit as the target variable.
    """
    return float(mse(y_true, y_pred) ** 0.5)


def business_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(y_true.size, y_pred.size)
    if n == 0:
        return np.nan
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    eps = 1e-12

    level_denom = np.sum(np.abs(y_true))
    if level_denom <= eps:
        level_denom = max(float(np.std(y_true) * y_true.size), eps)
    level_error = np.sum(np.abs(y_true - y_pred)) / level_denom

    if y_true.size < 2:
        return float(level_error)

    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    change_denom = np.sum(np.abs(dy_true))
    if change_denom <= eps:
        change_error = np.mean(np.abs(dy_pred)) / (np.std(y_true) + eps)
        direction_error = 0.0
    else:
        change_error = np.sum(np.abs(dy_true - dy_pred)) / change_denom
        true_sign = np.sign(dy_true)
        pred_sign = np.sign(dy_pred)
        weights = np.abs(dy_true)
        direction_error = np.sum(weights * ((true_sign != 0) & (true_sign != pred_sign))) / (np.sum(weights) + eps)

    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = np.linalg.norm(yt) * np.linalg.norm(yp)
    if denom <= eps:
        shape_error = 0.0 if np.linalg.norm(yt - yp) <= eps else 1.0
    else:
        corr = float(np.dot(yt, yp) / denom)
        corr = max(-1.0, min(1.0, corr))
        shape_error = (1.0 - corr) / 2.0

    level_error = min(float(level_error), 5.0)
    change_error = min(float(change_error), 5.0)
    direction_error = min(float(direction_error), 1.0)
    shape_error = min(float(shape_error), 1.0)

    return float(
        0.35 * level_error +
        0.30 * change_error +
        0.20 * direction_error +
        0.15 * shape_error
    )


def resolve_metric(metric):
    if callable(metric):
        return metric, getattr(metric, "__name__", "metric")
    name = str(metric).lower()
    registry = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "wmape": wmape,
        "medae": medae,
        "business": business_score,
        "business_score": business_score,
        "curve": business_score,
        "curve_score": business_score,
    }
    if name not in registry:
        raise ValueError(f"Unknown metric '{metric}'. Use one of {sorted(registry)} or pass a callable.")
    canonical = "business" if registry[name] is business_score else name
    return registry[name], canonical


try:
    from torch import nn
    import torch
except ImportError:
    nn = None
    torch = None


class _TorchLossUnavailable:
    def __init__(self, *args, **kwargs):
        raise ImportError("The torch backend is not installed. Install it with `pip install PipelineTS[torch]`.")


class WMAPELoss(nn.Module if nn is not None else _TorchLossUnavailable):
    def __init__(self, weight=None, size_average=True):
        super(WMAPELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.abs(inputs - targets).sum() / (torch.abs(targets).sum() + 1e-8)


class CombinedQuantileLoss(nn.Module if nn is not None else _TorchLossUnavailable):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.quantiles = [alpha / 2.0, 0.5, 1.0 - alpha / 2.0]

    def _pinball(self, pred, target, tau):
        errors = target - pred
        return torch.mean(torch.max(tau * errors, (tau - 1.0) * errors))

    def forward(self, a, b):
        if a.shape[-1] > b.shape[-1]:
            preds, targets = a, b
        elif b.shape[-1] > a.shape[-1]:
            preds, targets = b, a
        else:
            preds, targets = a, b

        f = targets.shape[-1]
        q_lower = preds[..., :f]
        q_median = preds[..., f:2 * f]
        q_upper = preds[..., 2 * f:]

        loss = (self._pinball(q_lower, targets, self.quantiles[0])
                + self._pinball(q_median, targets, self.quantiles[1])
                + self._pinball(q_upper, targets, self.quantiles[2]))
        return loss / 3.0


class RMSELoss(nn.Module if nn is not None else _TorchLossUnavailable):
    def __init__(self, weight=None, size_average=True):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))
