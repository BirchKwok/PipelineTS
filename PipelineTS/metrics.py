import numpy as np

from spinesUtils.asserts import ParameterTypeAssert


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
