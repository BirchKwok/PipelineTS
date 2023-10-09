import numpy as np

from spinesUtils.asserts import ParameterTypeAssert


@ParameterTypeAssert({
    'yt': np.ndarray,
    'left_pred': np.ndarray,
    'right_pred': np.ndarray
})
def quantile_acc(yt, left_pred, right_pred):
    """计算预测区间准确率"""
    return np.sum((yt >= left_pred) * (yt <= right_pred)) / yt.shape[0]
