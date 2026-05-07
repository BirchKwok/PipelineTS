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

import pandas as pd

from pathlib import Path

from PipelineTS.agent.session import Session


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
            return handler(**arguments)
        except Exception as exc:
            tb = traceback.format_exc()
            return f"Error executing {tool_name}: {type(exc).__name__}: {exc}\n\nTraceback:\n{tb}"

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

        n_outliers = int(mask.sum()) if hasattr(mask, "sum") else "N/A"
        pct = (
            f" ({100 * n_outliers / len(mask):.2f}%)"
            if isinstance(n_outliers, int) and len(mask) > 0
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

    # ------------------------------------------------------------------
    #  Preprocessing
    # ------------------------------------------------------------------

    def _handle_fill_missing_values(self, method: str = "linear") -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesMissingHandler

        handler = TimeSeriesMissingHandler(time_col=self.session.time_col)
        try:
            self.session.data = handler.transform(
                self.session.data, method=method
            )
        except Exception as e:
            return f"Missing value filling failed: {e}"

        return f"Missing values filled using '{method}' interpolation."

    def _handle_handle_outliers(self, strategy: str = "clip") -> str:
        if not self.session.has_data():
            return "No data loaded."

        from PipelineTS.preprocessing import TimeSeriesOutlierDetector

        detector = TimeSeriesOutlierDetector(
            time_col=self.session.time_col, method="iqr"
        )
        try:
            self.session.data = detector.transform(
                self.session.data,
                target_col=self.session.target_col,
                strategy=strategy,
            )
        except Exception as e:
            return f"Outlier handling failed: {e}"

        return f"Outliers handled using '{strategy}' strategy."

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
            self.session.data = engineer.fit_transform(self.session.data)
        except Exception as e:
            return f"Feature engineering failed: {e}"

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
            leaderboard = pipeline.fit(self.session.data)
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
            router.fit(self.session.data)
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
            model = model_cls(
                time_col=self.session.time_col,
                target_col=self.session.target_col,
                lags=lags,
                quantile=quantile,
            )
            model.fit(self.session.data)
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
        from PipelineTS.spinesTS.metrics import mae

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
