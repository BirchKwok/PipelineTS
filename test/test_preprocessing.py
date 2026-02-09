"""
Comprehensive test suite for preprocessing utilities in PipelineTS.

Tests:
- Scaler: min_max, gauss_rank, quantile, standard
- fit, fit_transform, transform, inverse_transform
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Scaler ───────────────────────────────────────────────────────────────────

class TestScaler:
    def _make_data(self):
        np.random.seed(42)
        return np.random.randn(50, 1).astype(np.float64)

    def test_min_max_scaler(self):
        from PipelineTS.preprocessing import Scaler
        scaler = Scaler('min_max')
        X = self._make_data()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == X.shape
        assert transformed.min() >= -1e-7
        assert transformed.max() <= 1.0 + 1e-7

    def test_standard_scaler(self):
        from PipelineTS.preprocessing import Scaler
        scaler = Scaler('standard')
        X = self._make_data()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == X.shape
        assert abs(transformed.mean()) < 0.1

    def test_quantile_scaler(self):
        from PipelineTS.preprocessing import Scaler
        scaler = Scaler('quantile')
        X = self._make_data()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == X.shape

    def test_gauss_rank_scaler(self):
        from PipelineTS.preprocessing import Scaler
        scaler = Scaler('gauss_rank')
        X = self._make_data()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == X.shape

    def test_fit_then_transform(self):
        from PipelineTS.preprocessing import Scaler
        scaler = Scaler('min_max')
        X = self._make_data()
        scaler.fit(X)
        transformed = scaler.transform(X)
        assert transformed.shape == X.shape

    def test_inverse_transform(self):
        from PipelineTS.preprocessing import Scaler
        scaler = Scaler('min_max')
        X = self._make_data()
        transformed = scaler.fit_transform(X)
        recovered = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, X, atol=1e-6)

    def test_unknown_scaler_raises(self):
        from PipelineTS.preprocessing import Scaler
        with pytest.raises(ValueError):
            Scaler('unknown_scaler')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
