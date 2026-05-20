"""Walk-forward backtesting framework for time series models.

Provides expanding and sliding window backtesting with per-fold metrics,
compatible with any PipelineTS model.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, Callable
from copy import deepcopy


class Backtester:
    """Walk-forward backtesting for time series models.

    Evaluates model performance by simulating sequential real-world forecasts:
    train on past data, predict forward, slide the window, repeat.

    Parameters
    ----------
    model : PipelineTS model instance
        Any model with fit() and predict() methods.
        The model is deep-copied per fold to avoid state leakage.
    time_col : str
        Datetime column name.
    target_col : str
        Target variable column name.
    metric : callable
        Scoring function with signature metric(y_true, y_pred) -> float.
    metric_name : str, default='metric'
        Display name of the metric.
    metric_less_is_better : bool, default=True
        Whether lower metric values are better.

    Examples
    --------
    >>> from PipelineTS.models.ml import TorchBoostingForestModel
    >>> from PipelineTS.metrics import mae
    >>> model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)
    >>> bt = Backtester(model, time_col='date', target_col='value', metric=mae)
    >>> results = bt.fit(data, n_splits=5, test_size=12, mode='expanding')
    >>> bt.summary()
    """

    def __init__(
        self,
        model,
        time_col: str,
        target_col: str,
        metric: Callable,
        metric_name: str = 'metric',
        metric_less_is_better: bool = True,
        id_col: Optional[str] = None,
    ):
        self.model = model
        self.time_col = time_col
        self.target_col = target_col
        self.metric = metric
        self.metric_name = metric_name
        self.metric_less_is_better = metric_less_is_better
        self.id_col = id_col
        self._results = None

    def fit(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size: int = 10,
        mode: Literal['expanding', 'sliding'] = 'expanding',
        train_size: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run backtesting.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.
        n_splits : int, default=5
            Number of train/test folds.
        test_size : int, default=10
            Number of time steps to forecast in each fold.
        mode : {'expanding', 'sliding'}
            'expanding': training window grows from the start.
            'sliding': training window has fixed size.
        train_size : int or None
            Fixed training size for 'sliding' mode. If None, auto-computed.
        verbose : bool, default=True
            Print per-fold progress.

        Returns
        -------
        pd.DataFrame
            Per-fold results with columns: fold, train_size, test_size, metric.
        """
        id_col = self.id_col

        # Multi-series panel backtesting
        if id_col is not None and id_col in data.columns:
            return self._fit_panel(
                data, n_splits=n_splits, test_size=test_size,
                mode=mode, train_size=train_size, verbose=verbose,
            )

        df = data.sort_values(self.time_col).reset_index(drop=True)
        n = len(df)

        # Compute fold boundaries
        folds = self._compute_folds(n, n_splits, test_size, mode, train_size)

        records = []
        all_preds = []
        all_actuals = []

        for i, (train_start, train_end, test_end) in enumerate(folds):
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            actual_test_size = len(test_df)

            if verbose:
                print(f"  Fold {i + 1}/{len(folds)}: "
                      f"train[{train_start}:{train_end}] ({train_end - train_start}) → "
                      f"test[{train_end}:{test_end}] ({actual_test_size})")

            fold_model = deepcopy(self.model)

            try:
                fold_model.fit(train_df)
                pred_df = fold_model.predict(actual_test_size)

                y_true = test_df[self.target_col].values
                y_pred = pred_df[self.target_col].values[:actual_test_size]

                score = self.metric(y_true, y_pred)

                all_preds.extend(y_pred.tolist())
                all_actuals.extend(y_true.tolist())

                records.append({
                    'fold': i + 1,
                    'train_size': train_end - train_start,
                    'test_size': actual_test_size,
                    self.metric_name: score,
                })
            except Exception as e:
                if verbose:
                    print(f"    ⚠ Fold {i + 1} failed: {e}")
                records.append({
                    'fold': i + 1,
                    'train_size': train_end - train_start,
                    'test_size': actual_test_size,
                    self.metric_name: np.nan,
                })

            del fold_model

        result_df = pd.DataFrame(records)
        self._results = result_df
        self._all_preds = np.array(all_preds) if all_preds else np.array([])
        self._all_actuals = np.array(all_actuals) if all_actuals else np.array([])

        return result_df

    def summary(self) -> dict:
        """Get summary statistics of backtesting results.

        Returns
        -------
        dict
            Keys: 'mean', 'std', 'min', 'max', 'median', 'n_folds', 'n_failed'.
        """
        if self._results is None:
            raise RuntimeError("Call fit() before summary().")

        scores = self._results[self.metric_name].dropna()
        return {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(scores.median()),
            'n_folds': len(self._results),
            'n_failed': int(self._results[self.metric_name].isna().sum()),
        }

    def _compute_folds(self, n, n_splits, test_size, mode, train_size):
        """Compute (train_start, train_end, test_end) tuples."""
        folds = []

        if mode == 'expanding':
            # Work backwards from the end
            total_test = n_splits * test_size
            min_train = max(test_size * 2, n - total_test)
            if min_train + total_test > n:
                min_train = n - total_test

            for i in range(n_splits):
                test_end = n - (n_splits - 1 - i) * test_size
                train_end = test_end - test_size
                train_start = 0
                if train_end <= 0 or test_end > n:
                    continue
                folds.append((train_start, train_end, test_end))

        elif mode == 'sliding':
            if train_size is None:
                train_size = max(test_size * 3, (n - n_splits * test_size))
            for i in range(n_splits):
                test_end = n - (n_splits - 1 - i) * test_size
                train_end = test_end - test_size
                train_start = max(0, train_end - train_size)
                if train_end <= train_start or test_end > n:
                    continue
                folds.append((train_start, train_end, test_end))

        return folds

    def _fit_panel(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size: int = 10,
        mode: Literal['expanding', 'sliding'] = 'expanding',
        train_size: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run backtesting on multi-series panel data.

        Splits each series independently at the same temporal fold boundaries
        (based on the shortest series), combines into panel train/test
        DataFrames, and evaluates predictions across all series.
        """
        id_col = self.id_col

        # Sort each series by time
        series_dfs = {}
        for sid, sdf in data.groupby(id_col):
            series_dfs[sid] = sdf.sort_values(self.time_col).reset_index(drop=True)

        # Compute folds based on the shortest series
        min_len = min(len(sdf) for sdf in series_dfs.values())
        folds = self._compute_folds(min_len, n_splits, test_size, mode, train_size)

        records = []
        all_preds = []
        all_actuals = []

        for i, (train_start, train_end, test_end) in enumerate(folds):
            # Build panel train/test by splitting each series at the same indices
            train_parts = []
            test_parts = []
            for sid, sdf in series_dfs.items():
                train_parts.append(sdf.iloc[train_start:train_end].copy())
                test_parts.append(sdf.iloc[train_end:test_end].copy())

            train_df = pd.concat(train_parts, ignore_index=True)
            test_df = pd.concat(test_parts, ignore_index=True)
            per_series_test_size = test_end - train_end

            if verbose:
                n_series = len(series_dfs)
                print(f"  Fold {i + 1}/{len(folds)}: "
                      f"train[{train_start}:{train_end}] ({train_end - train_start}/series, "
                      f"{len(train_df)} total) → "
                      f"test[{train_end}:{test_end}] ({per_series_test_size}/series, "
                      f"{len(test_df)} total, {n_series} series)")

            fold_model = deepcopy(self.model)

            try:
                fold_model.fit(train_df)
                pred_df = fold_model.predict(per_series_test_size)

                # Match predictions to actuals per series
                y_true_all = []
                y_pred_all = []
                for sid in series_dfs:
                    test_sid = test_df[test_df[id_col] == sid]
                    pred_sid = pred_df[pred_df[id_col] == sid] if id_col in pred_df.columns else pred_df
                    yt = test_sid[self.target_col].values
                    yp = pred_sid[self.target_col].values[:len(yt)]
                    y_true_all.extend(yt.tolist())
                    y_pred_all.extend(yp.tolist())

                y_true = np.array(y_true_all)
                y_pred = np.array(y_pred_all)
                score = self.metric(y_true, y_pred)

                all_preds.extend(y_pred.tolist())
                all_actuals.extend(y_true.tolist())

                records.append({
                    'fold': i + 1,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    self.metric_name: score,
                })
            except Exception as e:
                if verbose:
                    print(f"    ⚠ Fold {i + 1} failed: {e}")
                records.append({
                    'fold': i + 1,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    self.metric_name: np.nan,
                })

            del fold_model

        result_df = pd.DataFrame(records)
        self._results = result_df
        self._all_preds = np.array(all_preds) if all_preds else np.array([])
        self._all_actuals = np.array(all_actuals) if all_actuals else np.array([])

        return result_df

    # Backward-compatible alias
    run = fit
