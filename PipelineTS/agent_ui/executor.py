"""Tool executor — maps LLM function calls to PipelineTS operations.

Each handler receives the session and the tool arguments, executes the
operation, and returns a string result for the LLM to incorporate into
its response.
"""

from __future__ import annotations

import os
import traceback

# Prevent macOS GUI crashes from matplotlib
import matplotlib
matplotlib.use("Agg")

from typing import Any

import numpy as np
import pandas as pd

from pathlib import Path

from PipelineTS.agent.session import Session
from PipelineTS.preprocessing import time_series_diagnostics as tsdiag
from PipelineTS.preprocessing import time_series_preprocessing as tsprep


def _as_columns(value) -> list:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def _primary_column(value) -> str:
    values = _as_columns(value)
    return values[0] if values else ""


def _column_payload(value):
    values = _as_columns(value)
    if len(values) > 1:
        return values
    return values[0] if values else ""


def _format_columns(value) -> str:
    return ", ".join(_as_columns(value))


class Executor:
    """Executes agent tools against a PipelineTS session.

    Each method corresponds to a tool name from tools.py and follows
    the signature:  handler(session, **kwargs) -> str

    The returned string is injected into the LLM conversation as the
    tool result.
    """

    def __init__(self, session: Session, plot_dir: Path = None):
        self.session = session
        self.plot_dir = plot_dir
        self.selected_data_context = None
        self.progress_callback = None

    def _emit_progress(self, stage: str, message: str, **payload):
        if not self.progress_callback:
            return
        event = {"stage": stage, "message": message}
        event.update(payload)
        self.progress_callback(event)

    def _pipeline_progress_callback(self, stage: str = "model_complete"):
        def callback(model_name, model, fit_info, idx, total):
            try:
                metric = fit_info.get("metric")
                train_cost = fit_info.get("train_cost")
                eval_cost = fit_info.get("eval_cost")
                metric_value = float(metric) if metric is not None else None
                train_seconds = float(train_cost) if train_cost is not None else None
                eval_seconds = float(eval_cost) if eval_cost is not None else None
                metric_text = f", metric={metric_value:.6g}" if metric_value is not None else ""
                self._emit_progress(
                    stage,
                    f"[{idx + 1}/{total}] Finished {model_name}{metric_text}",
                    current=idx + 1,
                    total=total,
                    model=model_name,
                    metric=metric_value,
                    train_seconds=train_seconds,
                    eval_seconds=eval_seconds,
                )
            except Exception:
                pass
        return callback

    def dispatch(self, tool_name: str, arguments: dict) -> str:
        """Route a tool call to the appropriate handler.

        Parameters
        ----------
        tool_name : str
            The function name from the LLM tool call.
        arguments : dict
            Parsed keyword arguments.

        Returns
        -------
        str
            Result string to feed back to the LLM.
        """
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return f"Error: unknown tool '{tool_name}'"
        try:
            scoped = self._selected_scope_for_tool(tool_name)
            if scoped is not None:
                scoped_df, scope_note, scope_ctx = scoped
                original_data = self.session.data
                original_time_col = self.session.time_col
                original_target_col = self.session.target_col
                selected_time_col = _primary_column(scope_ctx.get("time_col")) if isinstance(scope_ctx, dict) else ""
                selected_target_col = self._selected_context_target_col(scoped_df, scope_ctx)
                self.session.data = scoped_df
                if selected_time_col and selected_time_col in scoped_df.columns:
                    self.session.time_col = selected_time_col
                if selected_target_col and selected_target_col in scoped_df.columns:
                    self.session.target_col = selected_target_col
                try:
                    result = handler(**arguments)
                finally:
                    self.session.data = original_data
                    self.session.time_col = original_time_col
                    self.session.target_col = original_target_col
                return f"{scope_note}\n\n{result}"
            return handler(**arguments)
        except Exception as exc:
            tb = traceback.format_exc()
            return f"Error executing {tool_name}: {type(exc).__name__}: {exc}\n\nTraceback:\n{tb}"

    def _selected_scope_for_tool(self, tool_name: str):
        scoped_tools = {
            "inspect_data",
            "check_missing_values",
            "detect_outliers",
            "check_stationarity",
            "data_quality_report",
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
            "plot_time_series",
            "plot_acf_pacf",
            "plot_decomposition",
        }
        if tool_name not in scoped_tools:
            return None
        scope_ctx = self._selected_scope_context()
        if not scope_ctx:
            return None
        scoped_df = self._selected_scope_dataframe(scope_ctx)
        if scoped_df is None or scoped_df.empty:
            return None
        return scoped_df, self._selected_scope_note(len(scoped_df), scope_ctx), scope_ctx

    def _selected_scope_context(self) -> dict:
        ctx = self.selected_data_context
        if not isinstance(ctx, dict) or ctx.get("confirmed") is not True:
            return {}
        sections = [item for item in ctx.get("sections", []) if isinstance(item, dict)] if isinstance(ctx.get("sections"), list) else []
        if not sections:
            return ctx
        active_id = self.session.metadata.get("active_dataset_id") if isinstance(self.session.metadata, dict) else None
        selected_id = ctx.get("dataset_id")
        chosen = None
        if active_id:
            chosen = next((item for item in sections if item.get("dataset_id") == active_id), None)
        if chosen is None and selected_id:
            chosen = next((item for item in sections if item.get("dataset_id") == selected_id), None)
        if chosen is None:
            chosen = sections[0]
        merged = dict(ctx)
        merged.update(chosen)
        merged["confirmed"] = True
        return merged

    def _selected_scope_dataframe(self, ctx: dict):
        if not isinstance(ctx, dict):
            return None
        sections = [item for item in ctx.get("sections", []) if isinstance(item, dict)] if isinstance(ctx.get("sections"), list) else []
        if ctx.get("multi_dataset") is True and len(sections) > 1:
            return self._selected_multi_scope_dataframe(sections)
        file_scoped = self._selected_rows_dataframe_from_file(ctx)
        if file_scoped is not None and not file_scoped.empty:
            return file_scoped.reset_index(drop=True)
        df = self._selected_source_dataframe(ctx)
        if df is None or df.empty:
            return None
        return self._slice_dataframe_by_context(df, ctx)

    def _slice_dataframe_by_context(self, df: pd.DataFrame, ctx: dict):
        time_col = _primary_column(ctx.get("time_col")) or _primary_column(self.session.time_col)
        time_start = ctx.get("time_start")
        time_end = ctx.get("time_end")
        row_ranges = ctx.get("row_ranges") or []
        use_time_scope = len(row_ranges) <= 1
        if use_time_scope and time_col and time_col in df.columns and time_start and time_end:
            scoped = self._filter_dataframe_by_time(df, time_col, time_start, time_end)
            if scoped is not None and not scoped.empty:
                return scoped.reset_index(drop=True)

        rows = []
        for item in row_ranges:
            if not isinstance(item, dict):
                continue
            try:
                start = int(item.get("start", 0))
                end = int(item.get("end", start))
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            start = max(0, start)
            end = min(len(df) - 1, end)
            if start <= end:
                rows.extend(range(start, end + 1))
        if rows:
            return df.iloc[sorted(set(rows))].reset_index(drop=True)
        return None

    def _selected_multi_scope_dataframe(self, sections: list):
        frames = []
        for section in sections:
            scoped = self._selected_rows_dataframe_from_file(section)
            if scoped is None or scoped.empty:
                df = self._selected_source_dataframe(section)
                if df is None or df.empty:
                    continue
                scoped = self._slice_dataframe_by_context(df, section)
            if scoped is None or scoped.empty:
                continue
            scoped = scoped.copy()
            dataset_name = section.get("dataset_name") or section.get("dataset_id") or "Dataset"
            if "__dataset__" in scoped.columns:
                scoped["__dataset__"] = dataset_name
            else:
                scoped.insert(0, "__dataset__", dataset_name)
            frames.append(scoped)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True, sort=False)

    def _selected_rows_dataframe_from_file(self, ctx: dict):
        dataset_id = ctx.get("dataset_id")
        active_id = self.session.metadata.get("active_dataset_id") if isinstance(self.session.metadata, dict) else None
        if self.session.data is not None and not self.session.data.empty and (not dataset_id or not active_id or dataset_id == active_id):
            return None
        filepath = None
        if dataset_id:
            item = self._dataset_item_for_id(dataset_id)
            filepath = item.get("filepath") if item else None
        if not filepath and self.session.data is None:
            filepath = self.session.data_filepath
        if not filepath:
            return None
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return None
        row_ranges = ctx.get("row_ranges") or []
        if not row_ranges:
            return None
        frames = []
        for item in row_ranges:
            if not isinstance(item, dict):
                continue
            try:
                start = max(0, int(item.get("start", 0)))
                end = max(start, int(item.get("end", start)))
            except (TypeError, ValueError):
                continue
            try:
                chunk = pd.read_csv(path, skiprows=range(1, start + 1), nrows=end - start + 1)
            except Exception:
                return None
            if not chunk.empty:
                frames.append(chunk)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True, sort=False)

    def _selected_source_dataframe(self, ctx: dict):
        df = self.session.data
        dataset_id = ctx.get("dataset_id")
        active_id = self.session.metadata.get("active_dataset_id") if isinstance(self.session.metadata, dict) else None
        if df is not None and not df.empty and (not dataset_id or not active_id or dataset_id == active_id):
            return df
        filepath = None
        if dataset_id:
            item = self._dataset_item_for_id(dataset_id)
            filepath = item.get("filepath") if item else None
        if not filepath and df is None:
            filepath = self.session.data_filepath
        if filepath:
            path = Path(filepath)
            if path.exists() and path.is_file():
                try:
                    return pd.read_csv(path)
                except Exception:
                    return None
        if df is not None and not df.empty and not dataset_id:
            return df
        return None

    def _dataset_item_for_id(self, dataset_id: str):
        if not dataset_id or not isinstance(self.session.metadata, dict):
            return None
        datasets = self.session.metadata.get("datasets")
        if not isinstance(datasets, list):
            return None
        for item in datasets:
            if isinstance(item, dict) and item.get("id") == dataset_id:
                return item
        return None

    def _selected_context_target_col(self, df: pd.DataFrame, ctx: dict = None) -> str:
        ctx = ctx if isinstance(ctx, dict) else (self.selected_data_context if isinstance(self.selected_data_context, dict) else {})
        if ctx.get("multi_dataset") is True:
            target_col = _primary_column(ctx.get("target_col"))
            return target_col if target_col and target_col in df.columns else ""
        time_col = _primary_column(ctx.get("time_col")) or _primary_column(self.session.time_col)
        selected_cols = [col for col in _as_columns(ctx.get("selected_columns")) if col in df.columns]
        candidate_cols = [col for col in selected_cols if col != time_col]
        if len(candidate_cols) == 1:
            return candidate_cols[0]
        target_col = _primary_column(ctx.get("target_col"))
        if target_col and target_col in df.columns:
            return target_col
        return ""

    @staticmethod
    def _filter_dataframe_by_time(df: pd.DataFrame, time_col: str, time_start, time_end):
        try:
            values = pd.to_datetime(df[time_col], errors="coerce")
            start = pd.to_datetime(time_start, errors="coerce")
            end = pd.to_datetime(time_end, errors="coerce")
        except Exception:
            return None
        if pd.isna(start) or pd.isna(end):
            return None
        if end < start:
            start, end = end, start
        mask = (values >= start) & (values <= end)
        return df.loc[mask]

    def _selected_scope_note(self, n_rows: int, ctx: dict = None) -> str:
        ctx = ctx if isinstance(ctx, dict) else (self.selected_data_context if isinstance(self.selected_data_context, dict) else {})
        pieces = [f"Confirmed selected data scope applied: computed on {n_rows} selected row(s), not the full dataset."]
        if ctx.get("dataset_name"):
            pieces.append(f"Dataset: {ctx.get('dataset_name')}.")
        time_start = ctx.get("time_start")
        time_end = ctx.get("time_end")
        if time_start or time_end:
            pieces.append(f"Selected time range: {time_start} → {time_end}.")
        if ctx.get("target_col"):
            pieces.append(f"Selected target column in preview: {_format_columns(ctx.get('target_col'))}.")
        return " ".join(pieces)

    def _resolve_plot_path(self, save_path: str, default_name: str) -> str:
        if save_path:
            if self.plot_dir and not str(save_path).startswith(str(self.plot_dir)):
                return str(self.plot_dir / Path(save_path).name)
            return save_path
        name = self.session.default_plot_path(default_name)
        if self.plot_dir:
            return str(self.plot_dir / Path(name).name)
        return name

    def _plot_result(self, save_path: str, label: str) -> str:
        filename = Path(save_path).name
        return f"[PLOT]{label} saved.{{{filename}}}[/PLOT]"

    def _target_columns(self, columns=None) -> list:
        cols = _as_columns(columns) or _as_columns(self.session.target_col)
        return [c for c in cols if self.session.has_data() and c in self.session.data.columns]

    def _numeric_columns(self, columns=None, include_target: bool = True) -> list:
        if not self.session.has_data():
            return []
        if columns:
            candidates = _as_columns(columns)
        elif include_target:
            candidates = _as_columns(self.session.target_col)
        else:
            candidates = [
                c for c in self.session.data.select_dtypes(include=[np.number]).columns
                if c not in {self.session.time_col, self.session.id_col}
            ]
        return [
            c for c in candidates
            if c in self.session.data.columns and pd.api.types.is_numeric_dtype(self.session.data[c])
        ]

    def _mark_data_changed(self, df: pd.DataFrame) -> None:
        self.session.data = df
        self.session.clear_model()

    # ------------------------------------------------------------------
    #  Data Loading
    # ------------------------------------------------------------------

    def _handle_load_csv(
        self,
        filepath: str,
        time_col: str,
        target_col: str,
        id_col: str = None,
        sep: str = ",",
    ) -> str:
        time_col = _primary_column(time_col)
        target_cols = _as_columns(target_col)
        target_col = _column_payload(target_cols)
        if not time_col:
            return "Error: time_col is required."
        if not target_cols:
            return "Error: target_col is required."

        # ── Case 1: Data already loaded in session ──
        if self.session.has_data():
            df = self.session.data
            if time_col not in df.columns:
                return f"Error: time column '{time_col}' not found. Available: {list(df.columns)}"
            missing_targets = [col for col in target_cols if col not in df.columns]
            if missing_targets:
                return f"Error: target column(s) {missing_targets} not found. Available: {list(df.columns)}"

            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception:
                pass
            df = df.sort_values(time_col).reset_index(drop=True)

            self.session.data = df
            self.session.time_col = time_col
            self.session.target_col = target_col
            self.session.id_col = id_col
            self.session.clear_model()

            try:
                freq = pd.infer_freq(df[time_col])
            except Exception:
                freq = None
            freq_str = f", freq: {freq}" if freq else ""
            return (
                f"Data already in memory. Updated column mapping.\n"
                f"  {len(df)} rows × {len(df.columns)} cols{freq_str}\n"
                f"  time_col='{time_col}', target_col='{_format_columns(target_col)}'"
                + (f", id_col='{id_col}'" if id_col else "")
            )

        # ── Case 2: File exists on disk ──
        # If filepath doesn't exist, try the session's data_filepath (from web UI upload)
        if not os.path.exists(filepath) and self.session.data_filepath and os.path.exists(self.session.data_filepath):
            filepath = self.session.data_filepath

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep=sep)
            except Exception as e:
                return f"Error reading CSV: {e}"

            if time_col not in df.columns:
                return f"Error: column '{time_col}' not found. Available: {list(df.columns)}"
            missing_targets = [col for col in target_cols if col not in df.columns]
            if missing_targets:
                return f"Error: column(s) {missing_targets} not found. Available: {list(df.columns)}"

            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception as e:
                return f"Error converting '{time_col}' to datetime: {e}"

            df = df.sort_values(time_col).reset_index(drop=True)
            self.session.data = df
            self.session.time_col = time_col
            self.session.target_col = target_col
            self.session.id_col = id_col
            self.session.data_filepath = filepath
            self.session.clear_model()

            try:
                freq = pd.infer_freq(df[time_col])
            except Exception:
                freq = None
            freq_str = f", freq: {freq}" if freq else ""
            return (
                f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns{freq_str}.\n"
                f"  time_col='{time_col}', target_col='{_format_columns(target_col)}'"
                + (f", id_col='{id_col}'" if id_col else "")
            )

        # ── Case 3: File not found, no data in memory ──
        return (
            f"Error: file not found: '{filepath}'.\n"
            f"No data is currently loaded. Upload a CSV file through the web interface, "
            f"or use load_builtin_dataset to load an example dataset."
        )

    def _handle_load_builtin_dataset(self, dataset_name: str) -> str:
        mapping = {
            "electric": "LoadElectricDataSets",
            "messages_sent_hour": "LoadMessagesSentHourDataSets",
            "messages_sent": "LoadMessagesSentDataSets",
            "web_sales": "LoadWebSales",
            "supermarket_incoming": "LoadSupermarketIncoming",
        }
        loader_name = mapping.get(dataset_name)
        if loader_name is None:
            return f"Error: unknown dataset '{dataset_name}'. Options: {list(mapping.keys())}"

        try:
            from PipelineTS import dataset as ds_module

            loader = getattr(ds_module, loader_name)
            df = loader()
        except Exception as e:
            return f"Error loading built-in dataset: {e}"

        # Auto-detect columns
        time_col = None
        target_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
            elif col.lower() in ("value", "target", "y", "sales", "demand", "count"):
                target_col = col

        if time_col is None:
            # Try parsing first column
            try:
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                time_col = df.columns[0]
            except Exception:
                time_col = df.columns[0]

        if target_col is None:
            # Pick the last numeric column
            for col in reversed(df.columns):
                if pd.api.types.is_numeric_dtype(df[col]) and col != time_col:
                    target_col = col
                    break

        if target_col is None:
            target_col = df.columns[-1]

        df = df.sort_values(time_col).reset_index(drop=True) if time_col else df
        self.session.data = df
        self.session.time_col = time_col
        self.session.target_col = target_col
        self.session.id_col = None
        self.session.data_filepath = None
        self.session.clear_model()

        return (
            f"Loaded '{dataset_name}' dataset: {len(df)} rows, "
            f"columns={list(df.columns)}\n"
            f"  time_col='{time_col}', target_col='{target_col}'"
        )

    # ------------------------------------------------------------------
    #  Data Inspection
    # ------------------------------------------------------------------

    def _handle_inspect_data(self, n_rows: int = 5) -> str:
        if not self.session.has_data():
            return "No data loaded. Use load_csv or load_builtin_dataset first."
        return self.session.data_summary(n_rows=n_rows)

    def _handle_get_data_context(
        self,
        scope: str = None,
        columns: list = None,
        max_rows: int = 40,
        include_preview: bool = True,
    ) -> str:
        if not self.session.has_data() and not self.session.has_data_file():
            return "No data loaded or uploaded. Use load_csv, load_builtin_dataset, or upload a CSV first."

        ctx = self._selected_scope_context()
        valid_scopes = {
            "selected",
            "same_day",
            "full_dataset",
            "selected_vs_same_day",
            "selected_vs_full_dataset",
        }
        if not scope:
            scope = "selected_vs_same_day" if ctx else "full_dataset"
        if scope not in valid_scopes:
            return f"Error: invalid scope '{scope}'. Options: {sorted(valid_scopes)}"

        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            max_rows = 40
        max_rows = max(0, min(max_rows, 200))
        include_preview = bool(include_preview)

        source_ctx = ctx if ctx else {}
        source_df = self._selected_source_dataframe(source_ctx)
        if source_df is None or source_df.empty:
            return "No source data available for the requested context."
        source_df = source_df.copy()

        time_col = _primary_column(source_ctx.get("time_col")) or _primary_column(self.session.time_col)
        if time_col and time_col in source_df.columns:
            source_df[time_col] = pd.to_datetime(source_df[time_col], errors="coerce")
            source_df = source_df.sort_values(time_col).reset_index(drop=True)

        selected_df = None
        if ctx:
            selected_df = self._selected_scope_dataframe(ctx)
            if selected_df is not None and not selected_df.empty:
                selected_df = selected_df.copy()
                if time_col and time_col in selected_df.columns:
                    selected_df[time_col] = pd.to_datetime(selected_df[time_col], errors="coerce")
                    selected_df = selected_df.sort_values(time_col).reset_index(drop=True)

        need_selection = scope in {"selected", "selected_vs_same_day", "selected_vs_full_dataset"}
        if need_selection and (selected_df is None or selected_df.empty):
            return "No confirmed selected data is available for the requested selected-data context."

        numeric_columns = self._resolve_context_numeric_columns(source_df, selected_df, columns, source_ctx, time_col)
        if not numeric_columns:
            return "No numeric columns available for the requested data context."

        lines = ["Data Context (actual dataset evidence):"]
        lines.append(f"- Requested scope: {scope}")
        lines.append(f"- Numeric column(s): {', '.join(numeric_columns)}")
        if source_ctx.get("dataset_name"):
            lines.append(f"- Dataset: {source_ctx.get('dataset_name')}")
        if self.session.has_data():
            lines.append("- Data source: in-memory session dataframe")
        elif self.session.data_filepath:
            lines.append(f"- Data source: uploaded CSV file at {self.session.data_filepath}")
        lines.append("- Scope rule: this tool is not auto-limited to the confirmed selection; it reads the requested comparison scope from the actual data source.")

        same_day_df = None
        if scope in {"same_day", "selected_vs_same_day"}:
            same_day_df = self._same_day_context_dataframe(source_df, selected_df, source_ctx, time_col)
            if same_day_df is None or same_day_df.empty:
                return "Could not derive same-day/all-day context from the selected time range and source data."

        if scope in {"selected", "selected_vs_same_day", "selected_vs_full_dataset"}:
            self._append_context_scope(lines, "Confirmed selected rows", selected_df, numeric_columns, time_col, max_rows, include_preview)

        if scope in {"same_day", "selected_vs_same_day"}:
            self._append_context_scope(lines, "Same-day/all-day rows from source data", same_day_df, numeric_columns, time_col, max_rows, include_preview)
            self._append_context_comparison(lines, "Selected", selected_df, "Same-day/all-day", same_day_df, numeric_columns)
            self._append_selected_day_segments(lines, same_day_df, selected_df, numeric_columns, time_col)

        if scope in {"full_dataset", "selected_vs_full_dataset"}:
            self._append_context_scope(lines, "Full dataset rows from source data", source_df, numeric_columns, time_col, max_rows, include_preview)
            if scope == "selected_vs_full_dataset":
                self._append_context_comparison(lines, "Selected", selected_df, "Full dataset", source_df, numeric_columns)

        lines.append("Use only the figures above, the confirmed selection text, and other tool outputs as quantitative evidence. If a requested comparison is absent here, say it is not supported by the available data context.")
        return "\n".join(lines)

    def _resolve_context_numeric_columns(self, source_df, selected_df, columns, ctx, time_col) -> list:
        requested = _as_columns(columns)
        if not requested and isinstance(ctx, dict):
            requested = [
                col for col in _as_columns(ctx.get("selected_columns"))
                if col != time_col and col in source_df.columns
            ]
        if not requested and isinstance(ctx, dict):
            requested = _as_columns(ctx.get("target_col"))
        if not requested:
            requested = _as_columns(self.session.target_col)
        if not requested:
            requested = [
                col for col in source_df.select_dtypes(include=[np.number]).columns
                if col not in {time_col, self.session.id_col}
            ]

        out = []
        for col in requested:
            if col in out:
                continue
            if col in source_df.columns:
                series = pd.to_numeric(source_df[col], errors="coerce")
            elif selected_df is not None and col in selected_df.columns:
                series = pd.to_numeric(selected_df[col], errors="coerce")
            else:
                continue
            if series.notna().any():
                out.append(col)
        return out[:8]

    def _same_day_context_dataframe(self, source_df, selected_df, ctx, time_col):
        if not time_col or time_col not in source_df.columns:
            return None
        start = pd.to_datetime(ctx.get("time_start"), errors="coerce") if isinstance(ctx, dict) else pd.NaT
        end = pd.to_datetime(ctx.get("time_end"), errors="coerce") if isinstance(ctx, dict) else pd.NaT
        if (pd.isna(start) or pd.isna(end)) and selected_df is not None and time_col in selected_df.columns:
            selected_times = pd.to_datetime(selected_df[time_col], errors="coerce").dropna()
            if not selected_times.empty:
                start = selected_times.min()
                end = selected_times.max()
        if pd.isna(start) or pd.isna(end):
            return None
        if end < start:
            start, end = end, start
        source_times = pd.to_datetime(source_df[time_col], errors="coerce")
        start_day = start.normalize()
        end_day = end.normalize()
        mask = (source_times.dt.normalize() >= start_day) & (source_times.dt.normalize() <= end_day)
        return source_df.loc[mask].reset_index(drop=True)

    def _append_context_scope(self, lines, label, df, columns, time_col, max_rows, include_preview):
        lines.append("")
        lines.append(f"## {label}")
        lines.append(f"- Rows: {len(df)}")
        if time_col and time_col in df.columns:
            times = pd.to_datetime(df[time_col], errors="coerce").dropna()
            if not times.empty:
                lines.append(f"- Time range: {times.min()} → {times.max()}")
        lines.extend(self._numeric_summary_table(df, columns))
        if include_preview and max_rows > 0:
            lines.extend(self._preview_table(df, columns, time_col, max_rows))

    def _numeric_summary_table(self, df, columns) -> list:
        lines = []
        lines.append("")
        lines.append("| Column | Count | Mean | Std | Min | Max | First | Last | Change | Change % | Slope/step |")
        lines.append("|--------|-------|------|-----|-----|-----|-------|------|--------|----------|------------|")
        for col in columns:
            stats = self._numeric_stats(df, col)
            lines.append(
                "| {col} | {count} | {mean} | {std} | {min} | {max} | {first} | {last} | {change} | {change_pct} | {slope} |".format(
                    col=col,
                    count=stats["count"],
                    mean=self._format_context_value(stats["mean"]),
                    std=self._format_context_value(stats["std"]),
                    min=self._format_context_value(stats["min"]),
                    max=self._format_context_value(stats["max"]),
                    first=self._format_context_value(stats["first"]),
                    last=self._format_context_value(stats["last"]),
                    change=self._format_context_value(stats["change"]),
                    change_pct=self._format_context_value(stats["change_pct"], suffix="%"),
                    slope=self._format_context_value(stats["slope"]),
                )
            )
        return lines

    def _numeric_stats(self, df, col) -> dict:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        clean = values.dropna()
        if clean.empty:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "first": np.nan,
                "last": np.nan,
                "change": np.nan,
                "change_pct": np.nan,
                "slope": np.nan,
            }
        first = clean.iloc[0]
        last = clean.iloc[-1]
        change = last - first
        change_pct = (change / first * 100) if first != 0 else np.nan
        if len(clean) >= 2:
            x = np.arange(len(clean), dtype=float)
            slope = np.polyfit(x, clean.to_numpy(dtype=float), 1)[0]
        else:
            slope = np.nan
        return {
            "count": int(clean.size),
            "mean": clean.mean(),
            "std": clean.std(),
            "min": clean.min(),
            "max": clean.max(),
            "first": first,
            "last": last,
            "change": change,
            "change_pct": change_pct,
            "slope": slope,
        }

    def _append_context_comparison(self, lines, left_label, left_df, right_label, right_df, columns):
        lines.append("")
        lines.append(f"## Comparison: {left_label} vs {right_label}")
        lines.append("| Column | Selected mean | Reference mean | Mean diff | Selected change | Reference change | Selected slope | Reference slope |")
        lines.append("|--------|---------------|----------------|-----------|-----------------|------------------|----------------|-----------------|")
        for col in columns:
            left = self._numeric_stats(left_df, col)
            right = self._numeric_stats(right_df, col)
            lines.append(
                "| {col} | {left_mean} | {right_mean} | {mean_diff} | {left_change} | {right_change} | {left_slope} | {right_slope} |".format(
                    col=col,
                    left_mean=self._format_context_value(left["mean"]),
                    right_mean=self._format_context_value(right["mean"]),
                    mean_diff=self._format_context_value(left["mean"] - right["mean"]),
                    left_change=self._format_context_value(left["change"]),
                    right_change=self._format_context_value(right["change"]),
                    left_slope=self._format_context_value(left["slope"]),
                    right_slope=self._format_context_value(right["slope"]),
                )
            )

    def _append_selected_day_segments(self, lines, day_df, selected_df, columns, time_col):
        if not time_col or time_col not in day_df.columns or selected_df is None or time_col not in selected_df.columns:
            return
        day_times = pd.to_datetime(day_df[time_col], errors="coerce")
        selected_times = pd.to_datetime(selected_df[time_col], errors="coerce").dropna()
        if selected_times.empty:
            return
        start = selected_times.min()
        end = selected_times.max()
        segments = [
            ("Before selected window", day_df.loc[day_times < start]),
            ("Selected window", day_df.loc[(day_times >= start) & (day_times <= end)]),
            ("After selected window", day_df.loc[day_times > end]),
        ]
        lines.append("")
        lines.append("## Same-day segments around the selected window")
        lines.append("| Segment | Rows | Time range | Column | Mean | First | Last | Change |")
        lines.append("|---------|------|------------|--------|------|-------|------|--------|")
        for segment_name, segment_df in segments:
            time_range = ""
            if not segment_df.empty:
                segment_times = pd.to_datetime(segment_df[time_col], errors="coerce").dropna()
                if not segment_times.empty:
                    time_range = f"{segment_times.min()} → {segment_times.max()}"
            for col in columns:
                stats = self._numeric_stats(segment_df, col)
                lines.append(
                    "| {segment} | {rows} | {time_range} | {col} | {mean} | {first} | {last} | {change} |".format(
                        segment=segment_name,
                        rows=len(segment_df),
                        time_range=time_range,
                        col=col,
                        mean=self._format_context_value(stats["mean"]),
                        first=self._format_context_value(stats["first"]),
                        last=self._format_context_value(stats["last"]),
                        change=self._format_context_value(stats["change"]),
                    )
                )

    def _preview_table(self, df, columns, time_col, max_rows) -> list:
        preview_cols = []
        if time_col and time_col in df.columns:
            preview_cols.append(time_col)
        for col in columns:
            if col in df.columns and col not in preview_cols:
                preview_cols.append(col)
        if not preview_cols:
            return []
        if len(df) <= max_rows:
            preview = df[preview_cols]
            label = f"Row preview ({len(preview)} rows)"
        else:
            head_n = max_rows // 2
            tail_n = max_rows - head_n
            preview = pd.concat([df[preview_cols].head(head_n), df[preview_cols].tail(tail_n)])
            label = f"Row preview (first {head_n} and last {tail_n} of {len(df)} rows)"
        lines = ["", f"{label}:"]
        lines.append("| Row | " + " | ".join(preview_cols) + " |")
        lines.append("|-----|" + "|".join(["---"] * len(preview_cols)) + "|")
        for idx, row in preview.iterrows():
            values = [self._format_context_cell(row.get(col)) for col in preview_cols]
            lines.append(f"| {idx} | " + " | ".join(values) + " |")
        return lines

    @staticmethod
    def _format_context_value(value, suffix: str = "") -> str:
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.6g}{suffix}"
        return f"{value}{suffix}"

    def _format_context_cell(self, value) -> str:
        text = self._format_context_value(value)
        return str(text).replace("|", "\\|").replace("\n", " ")

    def _handle_check_missing_values(self) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesMissingHandler

        handler = TimeSeriesMissingHandler(time_col=self.session.time_col)
        try:
            report = handler.fit(self.session.data)
        except Exception as e:
            return f"Missing value detection failed: {e}"

        lines = ["Missing Value Report:"]
        lines.append(f"  Explicit NaN values: {report.get('n_explicit_nan', 'N/A')}")
        lines.append(f"  Implicit (time-gap) missing: {report.get('n_implicit_gaps', 'N/A')}")
        if report.get("missing_details"):
            lines.append(f"  Details: {report['missing_details']}")
        return "\n".join(lines)

    def _handle_detect_outliers(self, method: str = "iqr") -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesOutlierDetector

        detector = TimeSeriesOutlierDetector(
            time_col=self.session.time_col, method=method
        )
        try:
            mask = detector.fit(self.session.data, target_col=self.session.target_col)
        except Exception as e:
            return f"Outlier detection failed: {e}"

        if isinstance(mask, (pd.DataFrame, pd.Series)):
            mask_values = mask.to_numpy(dtype=bool)
        elif hasattr(mask, "__array__"):
            mask_values = np.asarray(mask, dtype=bool)
        else:
            mask_values = None

        n_outliers = int(mask_values.sum()) if mask_values is not None else "N/A"
        n_checked = int(mask_values.size) if mask_values is not None else len(mask)
        pct = (
            f" ({100 * n_outliers / n_checked:.2f}%)"
            if isinstance(n_outliers, int) and n_checked > 0
            else ""
        )
        return f"Outlier detection ({method}): {n_outliers} outliers found{pct}."

    def _handle_check_stationarity(self) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import StationarityTest

        tester = StationarityTest(significance_level=0.05)
        try:
            result = tester.fit(
                self.session.data[self.session.target_col].values
            )
        except Exception as e:
            return f"Stationarity test failed: {e}"

        lines = ["Stationarity Test (ADF + KPSS):"]
        lines.append(f"  Conclusion: {result.get('conclusion', 'N/A')}")
        lines.append(f"  Suggested differencing order: {result.get('suggested_d', 'N/A')}")
        lines.append(f"  Suggested action: {result.get('suggested_action', 'N/A')}")
        if "adf_statistic" in result:
            lines.append(
                f"  ADF statistic={result['adf_statistic']:.4f}, "
                f"p-value={result.get('adf_pvalue', 'N/A')}"
            )
        return "\n".join(lines)

    def _handle_data_quality_report(self) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesDataQualityReport

        reporter = TimeSeriesDataQualityReport(
            time_col=self.session.time_col, target_col=self.session.target_col
        )
        try:
            reporter.report(self.session.data)
        except Exception as e:
            return f"Data quality report failed: {e}"

        return "Data quality report generated (see console output)."

    def _handle_analyze_time_index(self) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.time_index_report(self.session.data, self.session.time_col)

    def _handle_profile_series(self) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.series_profile(self.session.data, self.session.target_col)

    def _handle_analyze_autocorrelation(self, max_lags: int = 40) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.autocorrelation_report(
            self.session.data,
            self.session.target_col,
            max_lags=max_lags,
        )

    def _handle_detect_seasonality(self, period: int = None, top_k: int = 5) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.seasonality_report(
            self.session.data,
            self.session.target_col,
            period=period,
            top_k=top_k,
        )

    def _handle_analyze_trend(self, window: int = None) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.trend_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            window=window,
        )

    def _handle_detect_changepoints(
        self,
        method: str = "auto",
        window: int = None,
        top_k: int = 5,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.changepoint_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            method=method,
            window=window,
            top_k=top_k,
        )

    def _handle_detect_distribution_shift(self, segments: int = 3) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.distribution_shift_report(
            self.session.data,
            self.session.target_col,
            segments=segments,
        )

    def _handle_analyze_volatility(self, window: int = None) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.volatility_report(
            self.session.data,
            self.session.target_col,
            window=window,
        )

    def _handle_suggest_lag_features(self, max_lags: int = 60, top_k: int = 10) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.lag_feature_report(
            self.session.data,
            self.session.target_col,
            max_lags=max_lags,
            top_k=top_k,
        )

    def _handle_detect_calendar_effects(
        self,
        granularity: str = "auto",
        top_k: int = 10,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.calendar_effect_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            granularity=granularity,
            top_k=top_k,
        )

    def _handle_analyze_covariates(
        self,
        covariates: list = None,
        max_lag: int = 12,
        top_k: int = 10,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.covariate_relationship_report(
            self.session.data,
            self.session.target_col,
            time_col=self.session.time_col,
            id_col=self.session.id_col,
            covariates=covariates,
            max_lag=max_lag,
            top_k=top_k,
        )

    def _handle_analyze_intermittency(self) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.intermittency_report(self.session.data, self.session.target_col)

    def _handle_decompose_components(self, period: int = None) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.decomposition_report(
            self.session.data,
            self.session.target_col,
            period=period,
        )

    def _handle_recommend_timeseries_actions(self) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.recommendation_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            id_col=self.session.id_col,
        )

    def _handle_assess_forecastability(self, horizon: int = None, seasonal_period: int = None) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.forecastability_report(
            self.session.data,
            self.session.target_col,
            horizon=horizon,
            seasonal_period=seasonal_period,
        )

    def _handle_benchmark_baselines(
        self,
        horizon: int = None,
        seasonal_period: int = None,
        test_size: int = None,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.baseline_forecast_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            id_col=self.session.id_col,
            horizon=horizon,
            seasonal_period=seasonal_period,
            test_size=test_size,
        )

    def _handle_analyze_panel_structure(self) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.panel_structure_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            self.session.id_col,
        )

    def _handle_detect_leakage_risk(self, horizon: int = None, corr_threshold: float = 0.98) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.leakage_risk_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            id_col=self.session.id_col,
            known_covariates=self.session.known_covariates,
            past_covariates=self.session.past_covariates,
            feature_cols=self.session.feature_cols,
            horizon=horizon,
            corr_threshold=corr_threshold,
        )

    def _handle_assess_modeling_readiness(self, horizon: int = None) -> str:
        if not self.session.has_data():
            return "No data loaded."
        return tsdiag.modeling_readiness_report(
            self.session.data,
            self.session.time_col,
            self.session.target_col,
            id_col=self.session.id_col,
            horizon=horizon,
            known_covariates=self.session.known_covariates,
            past_covariates=self.session.past_covariates,
            feature_cols=self.session.feature_cols,
        )

    # ------------------------------------------------------------------
    #  Preprocessing
    # ------------------------------------------------------------------

    def _handle_fill_missing_values(self, method: str = "linear") -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesMissingHandler

        handler = TimeSeriesMissingHandler(time_col=self.session.time_col)
        try:
            df = handler.transform(
                self.session.data, method=method
            )
        except Exception as e:
            return f"Missing value filling failed: {e}"

        self._mark_data_changed(df)
        return f"Missing values filled using '{method}' interpolation."

    def _handle_handle_outliers(self, strategy: str = "clip") -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesOutlierDetector

        detector = TimeSeriesOutlierDetector(
            time_col=self.session.time_col, method="iqr"
        )
        try:
            df = detector.transform(
                self.session.data,
                target_col=self.session.target_col,
                strategy=strategy,
            )
        except Exception as e:
            return f"Outlier handling failed: {e}"

        self._mark_data_changed(df)
        return f"Outliers handled using '{strategy}' strategy."

    def _handle_sort_and_deduplicate(
        self,
        duplicate_strategy: str = "mean",
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        if not self.session.time_col or self.session.time_col not in self.session.data.columns:
            return "No valid time column configured."

        before = len(self.session.data)
        source = self.session.data.copy()
        source[self.session.time_col] = pd.to_datetime(source[self.session.time_col], errors="coerce")
        invalid_times = int(source[self.session.time_col].isna().sum())
        valid_source = source.dropna(subset=[self.session.time_col])
        keys = [self.session.time_col]
        if self.session.id_col and self.session.id_col in valid_source.columns:
            keys = [self.session.id_col, self.session.time_col]
        duplicate_count = int(valid_source.duplicated(keys).sum())
        df = tsprep.sort_and_deduplicate(
            self.session.data,
            time_col=self.session.time_col,
            id_col=self.session.id_col,
            duplicate_strategy=duplicate_strategy,
        )
        self._mark_data_changed(df)
        return (
            f"Data sorted and deduplicated. Rows: {before} → {len(df)}. "
            f"Invalid timestamps removed: {invalid_times}. Duplicate key rows aggregated: {duplicate_count}."
        )

    def _handle_resample_time_series(
        self,
        freq: str = None,
        agg: str = "mean",
        fill_method: str = "linear",
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        if not self.session.time_col or self.session.time_col not in self.session.data.columns:
            return "No valid time column configured."

        before = len(self.session.data)
        result = tsprep.resample_time_series(
            self.session.data,
            time_col=self.session.time_col,
            freq=freq,
            id_col=self.session.id_col,
            agg=agg,
            fill_method=fill_method,
        )
        self._mark_data_changed(result)
        return (
            f"Resampled time series with freq='{freq or 'auto'}', agg='{agg}', fill_method='{fill_method}'. "
            f"Rows: {before} → {len(result)}. Models were cleared because data changed."
        )

    def _handle_transform_target(
        self,
        method: str,
        columns: list = None,
        suffix: str = None,
        replace: bool = False,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        cols = self._numeric_columns(columns)
        if not cols:
            return "No numeric target columns found."
        method = (method or "").lower()
        try:
            df = tsprep.transform_target(
                self.session.data,
                target_col=self.session.target_col,
                method=method,
                columns=columns,
                suffix=suffix,
                replace=replace,
            )
        except Exception as e:
            return f"Target transform failed: {e}"
        created = cols if replace else [f"{col}_{suffix or method}" for col in cols]
        self._mark_data_changed(df)
        return f"Target transform '{method}' applied to {cols}. Output column(s): {created}. Models were cleared."

    def _handle_difference_series(
        self,
        order: int = 1,
        seasonal_period: int = None,
        columns: list = None,
        suffix: str = None,
        drop_na: bool = False,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        cols = self._numeric_columns(columns)
        if not cols:
            return "No numeric target columns found."
        order = max(1, int(order or 1))
        try:
            df = tsprep.difference_series(
                self.session.data,
                target_col=self.session.target_col,
                order=order,
                seasonal_period=seasonal_period,
                columns=columns,
                suffix=suffix,
                drop_na=drop_na,
            )
        except Exception as e:
            return f"Differencing failed: {e}"
        created = []
        for col in cols:
            name_parts = [col, suffix or f"diff{order}"]
            if seasonal_period and seasonal_period > 1:
                name_parts.append(f"s{seasonal_period}")
            created.append("_".join(name_parts))
        self._mark_data_changed(df)
        return f"Differenced series created: {created}. drop_na={drop_na}. Models were cleared."

    def _handle_smooth_series(
        self,
        method: str = "rolling_mean",
        window: int = 7,
        columns: list = None,
        suffix: str = None,
        replace: bool = False,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        cols = self._numeric_columns(columns)
        if not cols:
            return "No numeric target columns found."
        window = max(2, int(window or 7))
        method = method or "rolling_mean"
        try:
            df = tsprep.smooth_series(
                self.session.data,
                target_col=self.session.target_col,
                method=method,
                window=window,
                columns=columns,
                suffix=suffix,
                replace=replace,
            )
        except Exception as e:
            return f"Smoothing failed: {e}"
        created = cols if replace else [f"{col}_{suffix or method}_{window}" for col in cols]
        self._mark_data_changed(df)
        return f"Smoothing '{method}' applied with window={window}. Output column(s): {created}. Models were cleared."

    def _handle_clip_or_winsorize(
        self,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        columns: list = None,
        replace: bool = True,
        suffix: str = "winsor",
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        cols = self._numeric_columns(columns)
        if not cols:
            return "No numeric target columns found."
        summaries = []
        source = self.session.data
        for col in cols:
            s = pd.to_numeric(source[col], errors="coerce")
            lo, hi = s.quantile([lower_q, upper_q])
            out_col = col if replace else f"{col}_{suffix}"
            clipped = int(((s < lo) | (s > hi)).sum())
            summaries.append(f"{out_col}: clipped {clipped} values to [{lo:.4g}, {hi:.4g}]")
        try:
            df = tsprep.clip_or_winsorize(
                self.session.data,
                target_col=self.session.target_col,
                lower_q=lower_q,
                upper_q=upper_q,
                columns=columns,
                replace=replace,
                suffix=suffix,
            )
        except Exception as e:
            return f"Winsorization failed: {e}"
        self._mark_data_changed(df)
        return "Winsorization complete. " + "; ".join(summaries) + ". Models were cleared."

    def _handle_set_covariates(
        self,
        known_covariates: list = None,
        past_covariates: list = None,
        feature_cols: list = None,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."
        columns = set(self.session.data.columns)
        known = [c for c in (known_covariates or []) if c in columns]
        past = [c for c in (past_covariates or []) if c in columns]
        features = [c for c in (feature_cols or []) if c in columns]
        missing = [
            c for c in (known_covariates or []) + (past_covariates or []) + (feature_cols or [])
            if c not in columns
        ]
        self.session.known_covariates = known
        self.session.past_covariates = past
        self.session.feature_cols = features
        self.session.clear_model()
        msg = (
            f"Covariate configuration updated. known_covariates={known}, "
            f"past_covariates={past}, feature_cols={features}. Models were cleared."
        )
        if missing:
            msg += f" Ignored missing columns: {missing}."
        return msg

    # ------------------------------------------------------------------
    #  Visualization
    # ------------------------------------------------------------------

    def _handle_plot_time_series(
        self, save_path: str = None, title: str = None
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.plot import plot_series

        save_path = self._resolve_plot_path(save_path, "plot_series")
        try:
            import matplotlib.pyplot as plt

            plot_series(
                self.session.data,
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                title=title,
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            return f"Plot failed: {e}"

        self.session.last_plot_path = save_path
        return self._plot_result(save_path, "Time series plot")

    def _handle_plot_acf_pacf(
        self, max_lags: int = 30, save_path: str = None
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.plot import plot_acf_pacf

        save_path = self._resolve_plot_path(save_path, "plot_acf_pacf")
        try:
            import matplotlib.pyplot as plt

            series_vals = self.session.data[self.session.target_col].values
            plot_acf_pacf(
                series_vals,
                max_lags=max_lags,
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            return f"ACF/PACF plot failed: {e}"

        self.session.last_plot_path = save_path
        return self._plot_result(save_path, "ACF/PACF plot")

    def _handle_plot_decomposition(
        self, period: int = None, save_path: str = None
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.plot import plot_decomposition

        save_path = self._resolve_plot_path(save_path, "plot_decomposition")
        try:
            import matplotlib.pyplot as plt

            plot_decomposition(
                self.session.data,
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                period=period,
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            return f"Decomposition plot failed: {e}"

        self.session.last_plot_path = save_path
        return self._plot_result(save_path, "Decomposition plot")

    # ------------------------------------------------------------------
    #  Feature Engineering
    # ------------------------------------------------------------------

    def _handle_create_features(
        self,
        use_lags: bool = False,
        lag_window: int = 12,
        use_fourier: bool = False,
        fourier_periods: list = None,
        use_calendar: bool = False,
        use_holidays: bool = False,
        holiday_country: str = "US",
    ) -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.feature_engineering import TimeSeriesFeatureEngineer

        try:
            engineer = TimeSeriesFeatureEngineer(
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                use_calendar=use_calendar,
                use_fourier=use_fourier,
                fourier_periods=fourier_periods or [7, 365],
                fourier_harmonics=2,
                use_holidays=use_holidays,
                holiday_country=holiday_country,
                use_lags=use_lags,
                lag_window=lag_window,
                lag_features=["mean", "std", "trend_slope", "ema"] if use_lags else None,
            )
            df = engineer.fit_transform(self.session.data)
        except Exception as e:
            return f"Feature engineering failed: {e}"

        self._mark_data_changed(df)
        new_cols = [
            c
            for c in self.session.data.columns
            if c not in (self.session.time_col, self.session.target_col)
        ]
        return (
            f"Features created. New columns ({len(new_cols)}): {new_cols[:20]}..."
            if len(new_cols) > 20
            else f"Features created. New columns ({len(new_cols)}): {new_cols}"
        )

    # ------------------------------------------------------------------
    #  Model Management
    # ------------------------------------------------------------------

    def _handle_list_available_models(self) -> str:
        from PipelineTS.pipeline.pipeline_models import get_all_available_models

        models = get_all_available_models()

        nn_models = [
            "d_linear", "deepar", "gau", "n_beats", "n_hits", "n_linear",
            "tcn", "tft", "patch_rnn", "stacking_rnn", "tide", "time2vec",
            "transformer", "itransformer", "srs_net",
        ]
        ml_models = [
            "catboost", "xgboost", "random_forest", "extra_forest",
            "gc_forest", "multi_output_model", "multi_step_model",
            "wide_gbrt", "torch_boosting_forest", "torch_bagging_forest",
            "deep_forest", "regressor_chain",
        ]
        stat_models = ["auto_arima", "prophet"]
        foundation_models = [
            "chronos_2", "chronos_2_synth", "chronos_2_small", "chronos_bolt_small",
        ]

        lines = ["Available Models:"]
        lines.append("Neural Network (NN):")
        for m in sorted(models):
            if m in nn_models:
                lines.append(f"  - {m}")
        lines.append("Machine Learning (ML):")
        for m in sorted(models):
            if m in ml_models:
                lines.append(f"  - {m}")
        lines.append("Statistical:")
        for m in sorted(models):
            if m in stat_models:
                lines.append(f"  - {m}")
        lines.append("Foundation (zero-shot):")
        for m in sorted(models):
            if m in foundation_models:
                lines.append(f"  - {m}")
        lines.append("Other:")
        for m in sorted(models):
            if m not in nn_models + ml_models + stat_models + foundation_models:
                lines.append(f"  - {m}")

        return "\n".join(lines)

    def _handle_train_pipeline(
        self,
        include_models: str = "light",
        lags: int = None,
        quantile: float = None,
        cv: int = 5,
        use_scaler: bool = True,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded. Use load_csv or load_builtin_dataset first."

        from PipelineTS.pipeline import ModelPipeline

        if lags is None:
            lags = min(24, max(6, len(self.session.data) // 10))

        try:
            self._emit_progress(
                "prepare",
                f"Preparing ModelPipeline (models={include_models}, lags={lags}, cv={cv})",
                rows=len(self.session.data),
                lags=lags,
                cv=cv,
            )
            pipeline = ModelPipeline(
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                lags=lags,
                quantile=quantile,
                include_models=include_models,
                cv=cv,
                scaler=use_scaler,
                id_col=self.session.id_col,
                known_covariates=self.session.known_covariates or None,
                past_covariates=self.session.past_covariates or None,
            )
            pipeline._on_model_complete_callback = self._pipeline_progress_callback("model_complete")
            self._emit_progress("fit_start", "ModelPipeline training started")
            leaderboard = pipeline.fit(self.session.data)
            self._emit_progress(
                "fit_complete",
                f"ModelPipeline training finished with {len(leaderboard)} successful model(s)",
                successful_models=len(leaderboard),
            )
        except Exception as e:
            return f"Pipeline training failed: {e}"

        self.session.pipeline = pipeline
        self.session.router = None
        self.session.single_model = None
        self.session.model_type = "pipeline"
        self.session.leaderboard = leaderboard

        if not leaderboard.empty:
            self.session.best_model_name = leaderboard.iloc[0]["model"]
            best_metric = leaderboard.iloc[0]["metric"]
            lines = [
                f"ModelPipeline trained successfully with {len(leaderboard)} models.",
                f"Best model: {self.session.best_model_name} (metric={best_metric:.6f})",
                f"\nTop 5 leaderboard:",
                leaderboard.head(5).to_string(index=False),
            ]
        else:
            lines = ["ModelPipeline completed but no models succeeded."]

        return "\n".join(lines)

    def _handle_train_smart_router(
        self,
        n_predict: int,
        preset: str = "medium_quality",
        quantile: float = None,
        time_limit: int = None,
        include_models: list = None,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded. Use load_csv or load_builtin_dataset first."

        from PipelineTS.pipeline import SmartRouter

        try:
            self._emit_progress(
                "prepare",
                f"Preparing SmartRouter (preset={preset}, n_predict={n_predict})",
                rows=len(self.session.data),
                n_predict=n_predict,
                preset=preset,
            )
            router = SmartRouter(
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                n_predict=n_predict,
                quantile=quantile,
                preset=preset,
                time_limit=time_limit,
                include_models=include_models,
                id_col=self.session.id_col,
                known_covariates=self.session.known_covariates or None,
                past_covariates=self.session.past_covariates or None,
            )
            original_on_model_trained = router._on_model_trained
            progress_callback = self._pipeline_progress_callback("router_model_complete")

            def on_model_trained(model_name, model, fit_info, idx, total):
                original_on_model_trained(model_name, model, fit_info, idx, total)
                progress_callback(model_name, model, fit_info, idx, total)

            router._on_model_trained = on_model_trained
            self._emit_progress("fit_start", "SmartRouter search and training started")
            router.fit(self.session.data)
            self._emit_progress("fit_complete", "SmartRouter training finished")
        except Exception as e:
            return f"SmartRouter training failed: {e}"

        self.session.router = router
        self.session.pipeline = None
        self.session.single_model = None
        self.session.model_type = "router"

        lb = getattr(router, "leader_board_", None)
        self.session.leaderboard = lb
        if lb is not None and not lb.empty:
            self.session.best_model_name = lb.iloc[0]["model"]

        strategy = getattr(router, "strategy_", {})
        models_used = strategy.get("models", [])
        ensemble_info = ""
        if getattr(router, "ensemble_", None) is not None:
            ensemble_info = f"\nEnsemble: {router.ensemble_}"

        lines = [
            f"SmartRouter trained successfully (preset='{preset}').",
            f"Data profile: {getattr(router, 'profile_', 'N/A')}",
            f"Selected models ({len(models_used)}): {models_used}",
            ensemble_info,
        ]
        if lb is not None and not lb.empty:
            lines.append(f"\nBest model: {lb.iloc[0]['model']}")
            lines.append(f"Leaderboard:\n{lb.head(5).to_string(index=False)}")

        return "\n".join(str(l) for l in lines)

    def _handle_train_single_model(
        self,
        model_name: str,
        lags: int = None,
        quantile: float = None,
    ) -> str:
        if not self.session.has_data():
            return "No data loaded. Use load_csv or load_builtin_dataset first."

        from PipelineTS.pipeline.pipeline_models import get_all_available_models

        available = get_all_available_models()
        if model_name not in available:
            similar = [k for k in available if model_name.lower() in k.lower()]
            hint = f" Did you mean: {similar}?" if similar else ""
            return f"Unknown model '{model_name}'. Use list_available_models to see options.{hint}"

        if lags is None:
            lags = min(24, max(6, len(self.session.data) // 10))

        model_cls = available[model_name]
        try:
            self._emit_progress(
                "prepare",
                f"Preparing {model_name} (lags={lags})",
                rows=len(self.session.data),
                model=model_name,
                lags=lags,
            )
            model = model_cls(
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                lags=lags,
                quantile=quantile,
            )
            self._emit_progress("fit_start", f"Training {model_name}")
            model.fit(self.session.data)
            self._emit_progress("fit_complete", f"Finished training {model_name}")
        except Exception as e:
            return f"Model training failed: {e}"

        self.session.single_model = model
        self.session.pipeline = None
        self.session.router = None
        self.session.model_type = "single"
        self.session.best_model_name = model_name
        self.session.leaderboard = None

        return f"Model '{model_name}' trained successfully (lags={lags})."

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def _handle_show_leaderboard(self, top_n: int = None) -> str:
        if self.session.leaderboard is None:
            return "No leaderboard available. Train a ModelPipeline or SmartRouter first."

        lb = self.session.leaderboard
        if top_n:
            lb = lb.head(top_n)
        return f"Leaderboard ({len(self.session.leaderboard)} models):\n{lb.to_string(index=False)}"

    def _handle_backtest_model(
        self,
        test_size: int,
        n_splits: int = 5,
        model_name: str = None,
    ) -> str:
        if not self.session.has_model():
            return "No model trained. Train a model first."

        from PipelineTS.evaluation import Backtester
        from PipelineTS.metrics import mae

        if model_name:
            if self.session.model_type == "pipeline":
                model = self.session.pipeline.get_model(model_name)
                if model is None:
                    return f"Model '{model_name}' not found."
            else:
                return "Specific model backtesting is only supported for ModelPipeline. Use the best model instead."
        else:
            model = self.session.get_best_model()

        if model is None:
            return "No model available for backtesting."

        try:
            bt = Backtester(
                model,
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                metric=mae,
                metric_name="MAE",
            )
            results = bt.fit(
                self.session.data, n_splits=n_splits, test_size=test_size, mode="expanding"
            )
            summary = bt.summary()
        except Exception as e:
            return f"Backtesting failed: {e}"

        lines = [
            f"Backtesting complete ({n_splits} splits, test_size={test_size}):",
            f"  Mean MAE: {summary.get('mean', 'N/A'):.4f}",
            f"  Std MAE:  {summary.get('std', 'N/A'):.4f}",
        ]
        return "\n".join(lines)

    def _handle_analyze_residuals(self, model_name: str = None) -> str:
        if not self.session.has_data() or not self.session.has_model():
            return "Data and trained model required."

        m = self.session.get_model()
        if m is None:
            return "No model available."

        try:
            pred = m.predict(len(self.session.data))
        except Exception:
            if self.session.model_type == "pipeline":
                pred = self.session.pipeline.predict(
                    len(self.session.data)
                )
            elif self.session.model_type == "router":
                pred = self.session.router.predict(
                    len(self.session.data)
                )
            elif self.session.model_type == "single":
                pred = self.session.single_model.predict(
                    len(self.session.data)
                )
            else:
                return "Cannot generate predictions."

        from PipelineTS.evaluation import ResidualAnalyzer
        import numpy as np

        y_true = self.session.data[self.session.target_col].values[
            -len(pred) :
        ]
        y_pred = pred[self.session.target_col].values[: len(y_true)]

        try:
            analyzer = ResidualAnalyzer(y_true, y_pred)
            stats = analyzer.statistics()
            normality = analyzer.normality_test()
            acorr = analyzer.autocorrelation()
        except Exception as e:
            return f"Residual analysis failed: {e}"

        lines = ["Residual Analysis:"]
        lines.append(f"  Mean residual: {stats.get('mean', np.nan):.6f}")
        lines.append(f"  Std residual:  {stats.get('std', np.nan):.6f}")
        lines.append(f"  Skewness:       {stats.get('skewness', np.nan):.4f}")
        lines.append(f"  Kurtosis:       {stats.get('kurtosis', np.nan):.4f}")
        if normality:
            lines.append(f"  Normality:       {normality}")
        if acorr:
            lines.append(f"  Autocorrelation: {acorr}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Prediction
    # ------------------------------------------------------------------

    def _handle_forecast(self, n: int, model_name: str = None) -> str:
        if not self.session.has_model():
            return "No model trained. Train a model first."

        try:
            if self.session.model_type == "pipeline":
                result = self.session.pipeline.predict(n, model_name=model_name)
            elif self.session.model_type == "router":
                result = self.session.router.predict(n)
            else:
                result = self.session.single_model.predict(n)
        except Exception as e:
            return f"Forecast failed: {e}"

        # Format result
        lines = [f"Forecast ({n} steps):"]
        if self.session.time_col in result.columns:
            lines.append(result.to_string(index=False))
        else:
            lines.append(result.to_string(index=False))
        return "\n".join(lines)

    def _handle_predict_with_intervals(
        self, n: int, levels: list = None, model_name: str = None
    ) -> str:
        if not self.session.has_model():
            return "No model trained."

        if levels is None:
            levels = [0.5, 0.8, 0.9, 0.95]

        try:
            if self.session.model_type == "pipeline":
                result = self.session.pipeline.predict_quantiles(
                    n, levels=levels, model_name=model_name
                )
            elif self.session.model_type == "router":
                result = self.session.router.predict_quantiles(n, levels=levels)
            else:
                # Single model: use predict_quantiles if available
                if hasattr(self.session.single_model, "predict_quantiles"):
                    result = self.session.single_model.predict_quantiles(n, levels=levels)
                else:
                    result = self.session.single_model.predict(n)
                    return f"Prediction ({n} steps) — intervals not available for this model:\n{result.to_string(index=False)}"
        except Exception as e:
            return f"Interval prediction failed: {e}"

        lines = [f"Prediction with intervals ({n} steps, levels={levels}):"]
        lines.append(result.to_string(index=False))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def _handle_save_model(self, filepath: str) -> str:
        m = self.session.get_model()
        if m is None:
            return "No model to save."

        from PipelineTS.io import save_model

        try:
            save_model(filepath, m)
        except Exception as e:
            return f"Save failed: {e}"

        return f"Model saved to: {filepath}"

    def _handle_load_model(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"

        from PipelineTS.io import load_model
        from PipelineTS.pipeline import ModelPipeline, SmartRouter

        try:
            obj = load_model(filepath)
        except Exception as e:
            return f"Load failed: {e}"

        if isinstance(obj, SmartRouter):
            self.session.router = obj
            self.session.pipeline = None
            self.session.single_model = None
            self.session.model_type = "router"
        elif isinstance(obj, ModelPipeline):
            self.session.pipeline = obj
            self.session.router = None
            self.session.single_model = None
            self.session.model_type = "pipeline"
        else:
            self.session.single_model = obj
            self.session.pipeline = None
            self.session.router = None
            self.session.model_type = "single"

        self.session.best_model_name = None
        return f"Model loaded from: {filepath} (type={self.session.model_type})"

    # ------------------------------------------------------------------
    #  Session
    # ------------------------------------------------------------------

    def _handle_get_session_status(self) -> str:
        return self.session.status_summary()
