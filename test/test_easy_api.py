import numpy as np
import pandas as pd


def _make_data(n=48):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 10 + np.sin(np.arange(n) / 3.0) + rng.normal(0, 0.05, size=n)
    return pd.DataFrame({"ds": dates.astype(str), "y": y})


def test_infer_columns_and_preprocess():
    from PipelineTS import infer_target_col, infer_time_col, preprocess

    data = _make_data()
    dirty = pd.concat([data.iloc[[1]], data], ignore_index=True)
    dirty.loc[5, "y"] = np.nan

    assert infer_time_col(dirty) == "ds"
    assert infer_target_col(dirty, time_col="ds") == "y"

    cleaned, info = preprocess(dirty, return_info=True)
    assert len(cleaned) == len(data)
    assert cleaned["ds"].is_monotonic_increasing
    assert cleaned["y"].isna().sum() == 0
    assert info["time_col"] == "ds"
    assert info["target_col"] == "y"


def test_forecast_one_line_returns_predictions():
    from PipelineTS import forecast

    result, model = forecast(
        _make_data(),
        n=3,
        include_models="seasonal_naive",
        preset="fast",
        verbose=False,
        return_model=True,
    )

    assert len(result) == 3
    assert "ds" in result.columns
    assert "y" in result.columns
    assert model.inferred_columns_["time_col"] == "ds"
    assert model.inferred_columns_["target_col"] == "y"


def test_load_data_and_diagnose(tmp_path):
    from PipelineTS import diagnose, load_data

    path = tmp_path / "series.csv"
    _make_data().to_csv(path, index=False)

    loaded = load_data(path)
    report = diagnose(path, horizon=3)

    assert len(loaded) == 48
    assert report["time_col"] == "ds"
    assert report["target_col"] == "y"
    assert report["horizon"] == 3
    assert "readiness" in report["reports"]


def test_backtest_one_line_returns_summary():
    from PipelineTS import backtest

    result = backtest(
        _make_data(),
        n=3,
        n_splits=2,
        include_models="seasonal_naive",
        preset="fast",
        verbose=False,
    )

    assert len(result["results"]) == 2
    assert result["metric"] == "mae"
    assert result["summary"]["n_folds"] == 2
    assert result["time_col"] == "ds"
    assert result["target_col"] == "y"


def test_autoforecast_save_and_load(tmp_path):
    from PipelineTS import AutoForecast, forecast

    _, model = forecast(
        _make_data(),
        n=3,
        include_models="seasonal_naive",
        preset="fast",
        verbose=False,
        return_model=True,
    )
    path = tmp_path / "model.pts"
    model.save(str(path))

    loaded = AutoForecast.load(path)
    pred = loaded.predict(n=2)

    assert len(pred) == 2
    assert loaded.inferred_columns_["time_col"] == "ds"
    assert loaded.inferred_columns_["target_col"] == "y"
