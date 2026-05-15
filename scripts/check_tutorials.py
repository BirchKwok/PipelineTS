import ast
import inspect
import json
import re
from pathlib import Path

from PipelineTS.pipeline import ModelPipeline, SmartRouter
from PipelineTS.plot import (
    TSPlotter,
    plot_acf_pacf,
    plot_decomposition,
    plot_forecast,
    plot_leaderboard,
    plot_leaderboard_detail,
    plot_model_comparison,
    plot_residuals,
    plot_series,
    plot_train_test_split,
)
from PipelineTS.preprocessing import (
    baseline_forecast_report,
    forecastability_report,
    leakage_risk_report,
    modeling_readiness_report,
    panel_structure_report,
    series_profile,
    time_index_report,
)

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

BANNED_TOKENS = {
    "torch_boosting_forest",
    "torch_bagging_forest",
    "torch_deep_forest",
    "deep_forest",
    "lightgbm",
    "torch_lightgbm",
    "torch_xgboost",
    "torch_catboost",
}

REQUIRED_API_TERMS = {
    "ModelPipeline.list_all_available_models",
    "SmartRouter.list_all_available_models",
    "PipelineConfigs",
    "predict_quantiles",
    "known_covariates",
    "past_covariates",
    "future_covariates",
    "id_col",
    "feature_cols",
    "save(",
    "load(",
    "update(",
    "plot_leaderboard",
    "TimeSeriesPreprocessor",
    "modeling_readiness_report",
    "TSPlotter",
    "hpo_strategy",
    "ensemble_strategy",
    "search_strategy",
    "autonomy_summary_",
}

INDUSTRIAL_TERMS = {
    "retail",
    "demand",
    "inventory",
    "manufacturing",
    "operations",
    "store",
    "promotion",
    "forecasting",
}

MODEL_PRESETS = {"light", "all", "nn", "ml"}
STRATEGY_VALUES = {
    "fast", "medium_quality", "high_quality", "best_quality",
    "basic", "auto", "thorough", "none", "weighted_avg", "median",
    "stacking", "multi_stack", "quick", "full",
}

CALLABLES_TO_CHECK = {
    "time_index_report": time_index_report,
    "series_profile": series_profile,
    "forecastability_report": forecastability_report,
    "baseline_forecast_report": baseline_forecast_report,
    "leakage_risk_report": leakage_risk_report,
    "modeling_readiness_report": modeling_readiness_report,
    "panel_structure_report": panel_structure_report,
    "plot_series": plot_series,
    "plot_forecast": plot_forecast,
    "plot_leaderboard": plot_leaderboard,
    "plot_leaderboard_detail": plot_leaderboard_detail,
    "plot_model_comparison": plot_model_comparison,
    "plot_residuals": plot_residuals,
    "plot_acf_pacf": plot_acf_pacf,
    "plot_decomposition": plot_decomposition,
    "plot_train_test_split": plot_train_test_split,
    "ModelPipeline": ModelPipeline,
    "SmartRouter": SmartRouter,
    "TSPlotter": TSPlotter,
}


def accepts_keyword(func, keyword):
    signature = inspect.signature(func)
    if keyword in signature.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


def cell_source(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def extract_include_models(source):
    names = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "include_models":
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                names.append(value.value)
            elif isinstance(value, (ast.List, ast.Tuple)):
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        names.append(elt.value)
    return names


def validate_direct_call_keywords(source, path_name, cell_idx, errors):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in CALLABLES_TO_CHECK:
            continue
        target = CALLABLES_TO_CHECK[node.func.id]
        for kw in node.keywords:
            if kw.arg is None:
                continue
            if not accepts_keyword(target, kw.arg):
                errors.append(
                    f"{path_name}: cell {cell_idx} passes unknown keyword "
                    f"'{kw.arg}' to {node.func.id}()"
                )


def main():
    available = set(ModelPipeline.list_all_available_models())
    allowed = available | MODEL_PRESETS | STRATEGY_VALUES
    all_text = []
    errors = []

    main_notebooks = sorted(TUTORIALS.glob("[0-9][0-9]_*.ipynb"))
    checkpoint_notebooks = sorted((TUTORIALS / ".ipynb_checkpoints").glob("[0-9][0-9]_*checkpoint.ipynb"))
    notebooks = main_notebooks + checkpoint_notebooks
    if len(main_notebooks) != 12:
        errors.append(f"Expected 12 tutorial notebooks, found {len(main_notebooks)}")
    if len(checkpoint_notebooks) != 12:
        errors.append(f"Expected 12 tutorial checkpoints, found {len(checkpoint_notebooks)}")

    for path in notebooks:
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue

        if nb.get("nbformat") != 4:
            errors.append(f"{path.name}: nbformat is not 4")

        if not nb.get("cells"):
            errors.append(f"{path.name}: notebook has no cells")

        text = "\n".join(cell_source(c) for c in nb.get("cells", []))
        all_text.append(text)
        lowered = text.lower()
        for token in BANNED_TOKENS:
            if token in lowered:
                errors.append(f"{path.name}: banned old model token found: {token}")

        for idx, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            try:
                compile(source, f"{path.name}:cell{idx}", "exec")
            except SyntaxError as exc:
                errors.append(f"{path.name}: cell {idx} syntax error: {exc}")

            validate_direct_call_keywords(source, path.name, idx, errors)

            for model_name in extract_include_models(source):
                if model_name not in allowed:
                    errors.append(
                        f"{path.name}: include_models uses unknown model/preset '{model_name}'"
                    )

    combined = "\n".join(all_text)
    for term in REQUIRED_API_TERMS:
        if term not in combined:
            errors.append(f"Missing required API coverage term: {term}")

    lowered_combined = combined.lower()
    for term in INDUSTRIAL_TERMS:
        if term not in lowered_combined:
            errors.append(f"Missing industrial scenario term: {term}")

    if errors:
        print("Tutorial validation failed:")
        for err in errors:
            print("-", err)
        raise SystemExit(1)

    print(f"Tutorial validation passed for {len(notebooks)} notebooks.")
    print("Available models checked:", sorted(available))


if __name__ == "__main__":
    main()
