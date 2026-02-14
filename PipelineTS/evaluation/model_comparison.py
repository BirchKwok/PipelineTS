"""Model comparison visualization and multi-metric evaluation.

Provides bar charts, radar plots, and tabular comparison across models.
All computations are vectorized; plotting is optional (matplotlib).
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Union


class ModelComparison:
    """Compare multiple models on multiple metrics.

    Parameters
    ----------
    time_col : str
        Datetime column name.
    target_col : str
        Target variable column name.

    Examples
    --------
    >>> from PipelineTS.metrics import mape, smape, r2_score
    >>> comp = ModelComparison(time_col='date', target_col='value')
    >>> comp.add_result('TorchBoostingForest', y_true, y_pred_boost)
    >>> comp.add_result('TorchBaggingForest', y_true, y_pred_bag)
    >>> table = comp.fit(metrics={'MAPE': mape, 'sMAPE': smape, 'R2': r2_score})
    >>> comp.plot_bar()
    """

    def __init__(self, time_col: str, target_col: str):
        self.time_col = time_col
        self.target_col = target_col
        self._results = {}  # model_name -> (y_true, y_pred)
        self._interval_results = {}  # model_name -> (lower, upper)
        self._eval_table = None

    def add_result(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
    ) -> None:
        """Register a model's predictions for comparison.

        Parameters
        ----------
        model_name : str
            Display name for the model.
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.
        lower : array-like or None
            Lower prediction interval bound.
        upper : array-like or None
            Upper prediction interval bound.
        """
        self._results[model_name] = (
            np.asarray(y_true, dtype=np.float64),
            np.asarray(y_pred, dtype=np.float64),
        )
        if lower is not None and upper is not None:
            self._interval_results[model_name] = (
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
            )

    def fit(
        self,
        metrics: Optional[dict] = None,
        interval_metrics: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Evaluate all registered models on the given metrics.

        Parameters
        ----------
        metrics : dict or None
            Mapping of {metric_name: callable(y_true, y_pred) -> float}.
            If None, uses MAE and RMSE from spinesTS.
        interval_metrics : dict or None
            Mapping of {metric_name: callable(y_true, lower, upper) -> float}
            for interval evaluation. Applied only to models with interval results.

        Returns
        -------
        pd.DataFrame
            Comparison table with models as rows and metrics as columns.
        """
        if metrics is None:
            from PipelineTS.spinesTS.metrics import mae, rmse
            metrics = {'MAE': mae, 'RMSE': rmse}

        records = []
        for name, (yt, yp) in self._results.items():
            row = {'model': name}
            for mname, mfunc in metrics.items():
                try:
                    row[mname] = float(mfunc(yt, yp))
                except Exception:
                    row[mname] = np.nan
            if interval_metrics and name in self._interval_results:
                lower, upper = self._interval_results[name]
                for mname, mfunc in interval_metrics.items():
                    try:
                        row[mname] = float(mfunc(yt, lower, upper))
                    except Exception:
                        row[mname] = np.nan
            records.append(row)

        self._eval_table = pd.DataFrame(records)
        return self._eval_table

    def rank(self, metric_name: str, ascending: bool = True) -> pd.DataFrame:
        """Rank models by a specific metric.

        Parameters
        ----------
        metric_name : str
            Column name in the evaluation table.
        ascending : bool, default=True
            True if lower is better.

        Returns
        -------
        pd.DataFrame
            Sorted evaluation table with rank column.
        """
        if self._eval_table is None:
            raise RuntimeError("Call fit() first.")
        df = self._eval_table.sort_values(metric_name, ascending=ascending).reset_index(drop=True)
        df.insert(0, 'rank', range(1, len(df) + 1))
        return df

    def plot_bar(self, figsize: tuple = (12, 5), metric_cols: Optional[list] = None) -> None:
        """Plot grouped bar chart comparing models across metrics.

        Parameters
        ----------
        figsize : tuple, default=(12, 5)
            Figure size.
        metric_cols : list or None
            Metric columns to plot. If None, plots all numeric columns.
        """
        import matplotlib.pyplot as plt

        if self._eval_table is None:
            raise RuntimeError("Call fit() first.")

        df = self._eval_table.copy()
        if metric_cols is None:
            metric_cols = [c for c in df.columns if c != 'model']

        n_models = len(df)
        n_metrics = len(metric_cols)
        x = np.arange(n_models)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

        for i, col in enumerate(metric_cols):
            offset = (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(x + offset, df[col].values, width, label=col,
                          color=colors[i], edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(df['model'].values, rotation=30, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_radar(self, figsize: tuple = (8, 8), metric_cols: Optional[list] = None) -> None:
        """Plot radar (spider) chart comparing models.

        Metrics are min-max normalized to [0, 1] for visual comparison.

        Parameters
        ----------
        figsize : tuple, default=(8, 8)
            Figure size.
        metric_cols : list or None
            Metric columns to plot. If None, plots all numeric columns.
        """
        import matplotlib.pyplot as plt

        if self._eval_table is None:
            raise RuntimeError("Call fit() first.")

        df = self._eval_table.copy()
        if metric_cols is None:
            metric_cols = [c for c in df.columns if c != 'model']

        # Normalize metrics to [0, 1]
        norm_df = df[metric_cols].copy()
        for col in metric_cols:
            mn, mx = norm_df[col].min(), norm_df[col].max()
            if mx > mn:
                norm_df[col] = (norm_df[col] - mn) / (mx - mn)
            else:
                norm_df[col] = 0.5

        n_metrics = len(metric_cols)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        colors = plt.cm.Set1(np.linspace(0, 1, len(df)))

        for i, row in norm_df.iterrows():
            values = row[metric_cols].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, color=colors[i],
                    label=df.iloc[i]['model'])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_cols, size=9)
        ax.set_title('Model Comparison (normalized)', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_predictions(
        self,
        time_index: Optional[np.ndarray] = None,
        figsize: tuple = (14, 5),
        model_names: Optional[list] = None,
    ) -> None:
        """Plot predictions from multiple models against true values.

        Parameters
        ----------
        time_index : array-like or None
            X-axis values (timestamps or indices). If None, uses integer index.
        figsize : tuple, default=(14, 5)
            Figure size.
        model_names : list or None
            Subset of models to plot. If None, plots all.
        """
        import matplotlib.pyplot as plt

        if model_names is None:
            model_names = list(self._results.keys())

        fig, ax = plt.subplots(figsize=figsize)

        # Plot true values from the first model (all share the same y_true)
        first_name = model_names[0]
        y_true = self._results[first_name][0]
        x = time_index if time_index is not None else np.arange(len(y_true))
        ax.plot(x, y_true, 'k-', linewidth=1.5, label='Actual', zorder=10)

        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        for i, name in enumerate(model_names):
            _, y_pred = self._results[name]
            ax.plot(x[:len(y_pred)], y_pred, '--', linewidth=1.2,
                    color=colors[i], label=name, alpha=0.8)

        ax.set_title('Prediction Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel(self.target_col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Backward-compatible alias
    evaluate = fit
