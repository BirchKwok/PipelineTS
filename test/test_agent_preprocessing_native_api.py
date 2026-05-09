import pandas as pd

from PipelineTS.agent.executor import Executor
from PipelineTS.agent.session import Session
from PipelineTS.preprocessing import (
    baseline_forecast_report,
    clip_or_winsorize,
    difference_series,
    forecastability_report,
    leakage_risk_report,
    modeling_readiness_report,
    panel_structure_report,
    resample_time_series,
    smooth_series,
    sort_and_deduplicate,
    time_index_report,
    transform_target,
    trend_report,
)


def sample_data():
    return pd.DataFrame(
        {
            "ds": ["2024-01-03", "2024-01-01", "2024-01-01", "2024-01-04"],
            "y": [3.0, 1.0, 2.0, 100.0],
            "x": [0.3, 0.1, 0.2, 0.4],
        }
    )


def test_native_preprocessing_api_direct_call():
    raw = sample_data()

    clean = sort_and_deduplicate(raw, time_col="ds")
    assert list(clean["y"]) == [1.5, 3.0, 100.0]
    assert clean["ds"].is_monotonic_increasing

    regular = resample_time_series(clean, time_col="ds", freq="D")
    assert len(regular) == 4
    assert regular["y"].isna().sum() == 0

    transformed = transform_target(regular, target_col="y", method="log1p")
    assert "y_log1p" in transformed.columns

    diffed = difference_series(transformed, target_col="y", order=1)
    assert "y_diff1" in diffed.columns

    smoothed = smooth_series(diffed, target_col="y", method="rolling_mean", window=2)
    assert "y_rolling_mean_2" in smoothed.columns

    winsorized = clip_or_winsorize(smoothed, target_col="y", lower_q=0.05, upper_q=0.95)
    assert winsorized["y"].max() <= smoothed["y"].quantile(0.95) + 1e-12


def test_native_diagnostic_api_direct_call():
    data = sort_and_deduplicate(sample_data(), time_col="ds")

    index_report = time_index_report(data, time_col="ds")
    trend = trend_report(data, time_col="ds", target_col="y")

    assert isinstance(index_report, str)
    assert isinstance(trend, str)
    assert "Time Index" in index_report
    assert "Trend" in trend


def test_native_modeling_readiness_apis_direct_call():
    data = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=40, freq="D"),
            "y": list(range(40)),
            "promo": [0, 1] * 20,
            "future_y": list(range(1, 41)),
        }
    )

    forecastability = forecastability_report(data, target_col="y", horizon=5)
    baselines = baseline_forecast_report(data, time_col="ds", target_col="y", horizon=5)
    leakage = leakage_risk_report(data, time_col="ds", target_col="y", feature_cols=["promo", "future_y"], horizon=3)
    readiness = modeling_readiness_report(data, time_col="ds", target_col="y", horizon=5)

    assert "Forecastability Assessment" in forecastability
    assert "Baseline Forecast Benchmark" in baselines
    assert "Leakage Risk Assessment" in leakage
    assert "future_y" in leakage
    assert "Modeling Readiness Assessment" in readiness


def test_native_panel_structure_report_direct_call():
    data = pd.DataFrame(
        {
            "series": ["a"] * 10 + ["b"] * 8,
            "ds": list(pd.date_range("2024-01-01", periods=10, freq="D"))
            + list(pd.date_range("2024-01-03", periods=8, freq="D")),
            "y": list(range(10)) + list(range(8)),
        }
    )

    report = panel_structure_report(data, time_col="ds", target_col="y", id_col="series")

    assert "Panel / Multi-Series" in report
    assert "Series count: 2" in report


def test_agent_preprocessing_tools_delegate_to_native_api():
    session = Session(data=sample_data(), time_col="ds", target_col="y")
    executor = Executor(session)

    result = executor.dispatch("sort_and_deduplicate", {})
    assert "Data sorted and deduplicated" in result
    assert list(session.data["y"]) == [1.5, 3.0, 100.0]

    result = executor.dispatch("smooth_series", {"window": 2})
    assert "Smoothing" in result
    assert "y_rolling_mean_2" in session.data.columns

    result = executor.dispatch("clip_or_winsorize", {"lower_q": 0.05, "upper_q": 0.95})
    assert "Winsorization complete" in result


def test_agent_modeling_diagnostic_tools_delegate_to_native_api():
    data = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=40, freq="D"),
            "y": list(range(40)),
            "future_y": list(range(1, 41)),
        }
    )
    session = Session(data=data, time_col="ds", target_col="y")
    executor = Executor(session)

    assert "Forecastability Assessment" in executor.dispatch("assess_forecastability", {"horizon": 5})
    assert "Baseline Forecast Benchmark" in executor.dispatch("benchmark_baselines", {"horizon": 5})
    assert "Leakage Risk Assessment" in executor.dispatch("detect_leakage_risk", {"horizon": 3})
    assert "Modeling Readiness Assessment" in executor.dispatch("assess_modeling_readiness", {"horizon": 5})
