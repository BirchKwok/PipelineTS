"""Session state management for the PipelineTS agent.

Tracks loaded data, trained models, column configuration, and plot outputs
across conversation turns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class Session:
    """Persistent state across agent conversation turns.

    Attributes
    ----------
    data : pd.DataFrame or None
        The currently loaded dataset.
    time_col : str or None
        Name of the datetime column.
    target_col : str or None
        Name of the target column to forecast.
    id_col : str or None
        Name of the series identifier column (multi-series / panel data).
    feature_cols : list of str
        Additional feature columns.
    known_covariates : list of str
        Known future covariate column names.
    past_covariates : list of str
        Past covariate column names.
    pipeline : ModelPipeline or None
        A fitted ModelPipeline instance (when train_pipeline is used).
    router : SmartRouter or None
        A fitted SmartRouter instance (when train_smart_router is used).
    single_model : object or None
        A single fitted model instance (when train_single_model is used).
    leaderboard : pd.DataFrame or None
        The model leaderboard DataFrame.
    best_model_name : str or None
        Name of the best model from the leaderboard.
    model_type : str or None
        One of 'pipeline', 'router', 'single', or None.
    data_filepath : str or None
        Original file path of the loaded data.
    last_plot_path : str or None
        Path of the most recently saved plot.
    messages : list of dict
        Conversation history (OpenAI format).
    metadata : dict
        Arbitrary metadata for extensibility.
    """

    data: Optional[pd.DataFrame] = None
    time_col: Optional[str] = None
    target_col: Optional[str] = None
    id_col: Optional[str] = None
    feature_cols: list = field(default_factory=list)
    known_covariates: list = field(default_factory=list)
    past_covariates: list = field(default_factory=list)

    pipeline: Any = None          # ModelPipeline
    router: Any = None            # SmartRouter
    single_model: Any = None      # Single fitted model
    leaderboard: Optional[pd.DataFrame] = None
    best_model_name: Optional[str] = None
    model_type: Optional[str] = None  # 'pipeline', 'router', 'single'

    data_filepath: Optional[str] = None
    last_plot_path: Optional[str] = None

    messages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    #  Data helpers
    # ------------------------------------------------------------------

    def has_data(self) -> bool:
        """Return True if data has been loaded."""
        return self.data is not None

    def has_data_file(self) -> bool:
        """Return True if a data file has been uploaded (via web UI) but not yet loaded into memory."""
        return bool(self.data_filepath)

    def has_model(self) -> bool:
        """Return True if any model has been trained."""
        return self.model_type is not None

    def get_model(self) -> Any:
        """Return the current model object (pipeline, router, or single)."""
        if self.model_type == "pipeline":
            return self.pipeline
        elif self.model_type == "router":
            return self.router
        elif self.model_type == "single":
            return self.single_model
        return None

    def get_best_model(self) -> Any:
        """Return the underlying best model from pipeline or the single model."""
        m = self.get_model()
        if m is None:
            return None
        if self.model_type == "pipeline" and hasattr(m, "best_model_"):
            return m.best_model_
        if self.model_type == "router" and hasattr(m, "pipeline_"):
            return m.pipeline_.best_model_ if m.pipeline_ else None
        return m

    def clear_model(self) -> None:
        """Remove trained models from session (keep data)."""
        self.pipeline = None
        self.router = None
        self.single_model = None
        self.leaderboard = None
        self.best_model_name = None
        self.model_type = None

    def to_dict(self) -> dict:
        """Serialize session state (excluding large objects) for display."""
        info = {
            "data_loaded": self.data is not None,
            "rows": len(self.data) if self.data is not None else 0,
            "columns": list(self.data.columns) if self.data is not None else [],
            "time_col": self.time_col,
            "target_col": self.target_col,
            "id_col": self.id_col,
            "feature_cols": self.feature_cols,
            "known_covariates": self.known_covariates,
            "past_covariates": self.past_covariates,
            "model_type": self.model_type,
            "best_model_name": self.best_model_name,
            "data_filepath": self.data_filepath,
            "last_plot_path": self.last_plot_path,
        }
        if self.leaderboard is not None:
            info["leaderboard_models"] = len(self.leaderboard)
            info["leaderboard_top3"] = (
                self.leaderboard["model"].iloc[:3].tolist()
                if "model" in self.leaderboard.columns
                else []
            )
        return info

    def status_summary(self) -> str:
        """Return a human-readable status summary."""
        d = self.to_dict()
        lines = []
        if d["data_loaded"]:
            lines.append(f"Data: {d['rows']} rows, columns={d['columns']}")
            lines.append(f"  time_col='{d['time_col']}', target_col='{d['target_col']}'")
            if d["id_col"]:
                lines.append(f"  id_col='{d['id_col']}' (multi-series)")
            if d["known_covariates"]:
                lines.append(f"  known_covariates={d['known_covariates']}")
        else:
            lines.append("Data: not loaded")
        if d["model_type"]:
            lines.append(f"Model type: {d['model_type']}")
            if d["best_model_name"]:
                lines.append(f"Best model: {d['best_model_name']}")
            if d.get("leaderboard_models"):
                lines.append(f"Leaderboard: {d['leaderboard_models']} models")
        else:
            lines.append("Model: not trained")
        return "\n".join(lines)

    def data_summary(self, n_rows: int = 5) -> str:
        """Return a DataFrame summary string."""
        if self.data is None:
            return "No data loaded."
        df = self.data
        lines = []
        lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        lines.append(f"Columns: {list(df.columns)}")
        lines.append(f"Dtypes:\n{df.dtypes.to_string()}")
        time_col = self.time_col[0] if isinstance(self.time_col, list) and self.time_col else self.time_col
        target_cols = self.target_col if isinstance(self.target_col, list) else ([self.target_col] if self.target_col else [])
        if time_col and time_col in df.columns:
            t = df[time_col]
            if pd.api.types.is_datetime64_any_dtype(t):
                lines.append(f"Time range: {t.min()} → {t.max()}")
                lines.append(f"Time freq: {pd.infer_freq(t)}")
            else:
                lines.append(f"Time range: {t.iloc[0]} → {t.iloc[-1]}")
        for target_col in target_cols:
            if target_col and target_col in df.columns:
                lines.append(
                    f"Target '{target_col}' stats: mean={df[target_col].mean():.4f}, "
                    f"std={df[target_col].std():.4f}, "
                    f"min={df[target_col].min():.4f}, "
                    f"max={df[target_col].max():.4f}"
                )
                lines.append(f"Missing target '{target_col}': {df[target_col].isna().sum()}")
        lines.append(f"\nFirst {n_rows} rows:\n{df.head(n_rows).to_string()}")
        lines.append(f"\nLast {n_rows} rows:\n{df.tail(n_rows).to_string()}")
        return "\n".join(lines)

    def default_plot_path(self, name: str) -> str:
        """Return a default file path for saving a plot.

        Uses the current working directory; prefixes with the data filename
        if available for context.
        """
        prefix = ""
        if self.data_filepath:
            base = os.path.splitext(os.path.basename(self.data_filepath))[0]
            prefix = f"{base}_"
        return f"{prefix}{name}.png"
