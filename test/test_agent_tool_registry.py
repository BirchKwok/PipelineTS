import inspect

from PipelineTS.agent.executor import Executor
from PipelineTS.agent.tools import ALL_TOOLS


def _tool_name(tool):
    return tool["function"]["name"]


def test_all_registered_tools_have_executor_handlers():
    tool_names = [_tool_name(tool) for tool in ALL_TOOLS]
    handlers = {
        name.removeprefix("_handle_")
        for name, member in inspect.getmembers(Executor, predicate=inspect.isfunction)
        if name.startswith("_handle_")
    }

    assert len(tool_names) == len(set(tool_names))
    assert sorted(set(tool_names) - handlers) == []


def test_native_preprocessing_tools_are_registered():
    tool_names = {_tool_name(tool) for tool in ALL_TOOLS}
    expected = {
        "analyze_time_index",
        "profile_series",
        "analyze_autocorrelation",
        "detect_seasonality",
        "analyze_trend",
        "detect_changepoints",
        "detect_distribution_shift",
        "analyze_volatility",
        "suggest_lag_features",
        "detect_calendar_effects",
        "analyze_covariates",
        "analyze_intermittency",
        "decompose_components",
        "recommend_timeseries_actions",
        "assess_forecastability",
        "benchmark_baselines",
        "analyze_panel_structure",
        "detect_leakage_risk",
        "assess_modeling_readiness",
        "sort_and_deduplicate",
        "resample_time_series",
        "transform_target",
        "difference_series",
        "smooth_series",
        "clip_or_winsorize",
        "set_covariates",
    }

    assert expected <= tool_names
