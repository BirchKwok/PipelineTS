"""
Comprehensive test suite for metrics in PipelineTS.

Tests:
- quantile_acc: accuracy of prediction intervals
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestQuantileAcc:
    def test_perfect_interval(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        left = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        right = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        acc = quantile_acc(yt, left, right)
        assert acc == 1.0, "All points should be within interval"

    def test_no_coverage(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([10.0, 20.0, 30.0])
        left = np.array([0.0, 0.0, 0.0])
        right = np.array([1.0, 1.0, 1.0])
        acc = quantile_acc(yt, left, right)
        assert acc == 0.0, "No points should be within interval"

    def test_partial_coverage(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([1.0, 5.0, 3.0, 10.0])
        left = np.array([0.0, 0.0, 0.0, 0.0])
        right = np.array([2.0, 4.0, 4.0, 4.0])
        acc = quantile_acc(yt, left, right)
        assert acc == 0.5, f"Expected 0.5, got {acc}"

    def test_boundary_values(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([1.0, 2.0])
        left = np.array([1.0, 2.0])
        right = np.array([1.0, 2.0])
        acc = quantile_acc(yt, left, right)
        assert acc == 1.0, "Boundary values should count as covered"

    def test_single_element(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([5.0])
        left = np.array([4.0])
        right = np.array([6.0])
        acc = quantile_acc(yt, left, right)
        assert acc == 1.0

    def test_return_type(self):
        from PipelineTS.metrics import quantile_acc
        yt = np.array([1.0, 2.0, 3.0])
        left = np.array([0.0, 1.0, 2.0])
        right = np.array([2.0, 3.0, 4.0])
        acc = quantile_acc(yt, left, right)
        assert isinstance(acc, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
