from __future__ import annotations

from pathlib import Path


def main() -> None:
    import PipelineTS
    from PipelineTS import metrics
    from PipelineTS.base import ForecastingMixin, TorchModelMixin
    from PipelineTS.preprocessing import split_series, lag_splits, GaussRankScaler
    from PipelineTS.feature_engineering import DateExtendFeatures
    from PipelineTS.datasets import LoadElectric, BuiltInSeriesData
    from PipelineTS.ml_model.regressor_wrappers import MultiOutputRegressor, MultiStepRegressor
    from PipelineTS.ml_model.wide_gbrt_preprocessing import GBRTPreprocessing
    from PipelineTS.nn_model.backends import is_torch_available, is_mlx_available
    from PipelineTS.nn_model.backbones import NLinear, DLinear
    from PipelineTS.nn_model.layers import GlobalTemporalBlock
    from PipelineTS.pipeline.pipeline_models import get_all_available_models

    root = Path(__file__).resolve().parents[1]
    legacy_package = root / "PipelineTS" / ("spines" + "TS")
    assert not legacy_package.exists(), "legacy module package should be removed"
    assert callable(metrics.mae)
    assert ForecastingMixin is not None
    assert TorchModelMixin is not None
    assert split_series([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 1)[0].shape == (3, 2)
    assert lag_splits([1, 2, 3], 2).shape == (2, 2)
    assert GaussRankScaler is not None
    assert DateExtendFeatures is not None
    assert BuiltInSeriesData is not None
    assert LoadElectric() is not None
    assert MultiOutputRegressor is not None and MultiStepRegressor is not None
    assert GBRTPreprocessing is not None
    assert is_torch_available() or is_mlx_available()
    assert NLinear is not None and DLinear is not None
    assert GlobalTemporalBlock is not None
    models = get_all_available_models()
    assert "d_linear" in models and "n_linear" in models
    print("PASS core module refactor smoke")


if __name__ == "__main__":
    main()
