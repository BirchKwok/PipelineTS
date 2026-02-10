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
