"""Rolling (sliding window) prediction for time series models.

Provides rolling forecast that re-fits the model on a sliding window
of recent data, producing predictions that adapt to distribution shifts.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Callable, Union


class RollingPredictor:
    """Rolling window predictor that re-fits and forecasts sequentially.

    At each step, the model is re-trained on the most recent `train_size`
    observations and produces a `horizon`-step forecast. The window then
    advances by `step` observations.

    Parameters
    ----------
    model : PipelineTS model instance
        Any model with fit() and predict() methods. Deep-copied per window.
    time_col : str
        Datetime column name.
    target_col : str
        Target column name.
    train_size : int
        Number of observations in the training window.
    horizon : int
        Number of steps to forecast at each position.
    step : int, default=1
        Number of observations to advance the window per iteration.
    refit : bool, default=True
        If True, re-fit the model at each step. If False, only fit once
        on the first window (much faster, less adaptive).

    Examples
    --------
    >>> from PipelineTS.ml_model import TorchBoostingForestModel
    >>> model = TorchBoostingForestModel(time_col='date', target_col='value', lags=12)
    >>> rp = RollingPredictor(model, time_col='date', target_col='value',
    ...                       train_size=100, horizon=10, step=10)
    >>> results = rp.predict(data)
    """

    def __init__(
        self,
        model,
        time_col: str,
        target_col: str,
        train_size: int,
        horizon: int,
        step: int = 1,
        refit: bool = True,
    ):
        self.model = model
        self.time_col = time_col
        self.target_col = target_col
        self.train_size = train_size
        self.horizon = horizon
        self.step = step
        self.refit = refit

    def predict(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Execute rolling prediction across the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset sorted by time.
        verbose : bool, default=True
            Print progress.

        Returns
        -------
        pd.DataFrame
            Concatenated predictions with columns: time_col, target_col,
            '{target_col}_actual', 'window_id'. One row per predicted point.
        """
        df = data.sort_values(self.time_col).reset_index(drop=True)
        n = len(df)

        all_results = []
        window_id = 0
        fitted_model = None

        pos = self.train_size
        while pos + self.horizon <= n:
            train_df = df.iloc[pos - self.train_size:pos].copy()
            actual_df = df.iloc[pos:pos + self.horizon].copy()

            if self.refit or fitted_model is None:
                current_model = deepcopy(self.model)
                try:
                    current_model.fit(train_df)
                    fitted_model = current_model
                except Exception as e:
                    if verbose:
                        print(f"  Window {window_id}: fit failed - {e}")
                    pos += self.step
                    window_id += 1
                    continue

            try:
                pred_df = fitted_model.predict(self.horizon)

                result = pd.DataFrame({
                    self.time_col: actual_df[self.time_col].values,
                    self.target_col: pred_df[self.target_col].values[:self.horizon],
                    f'{self.target_col}_actual': actual_df[self.target_col].values,
                    'window_id': window_id,
                })

                # Include interval bounds if available
                lower_col = f'{self.target_col}_lower'
                upper_col = f'{self.target_col}_upper'
                if lower_col in pred_df.columns:
                    result[lower_col] = pred_df[lower_col].values[:self.horizon]
                if upper_col in pred_df.columns:
                    result[upper_col] = pred_df[upper_col].values[:self.horizon]

                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Window {window_id}: predict failed - {e}")

            if verbose and window_id % max(1, ((n - self.train_size) // self.step) // 10) == 0:
                print(f"  Window {window_id}: pos={pos}/{n}")

            pos += self.step
            window_id += 1

        if not all_results:
            return pd.DataFrame(columns=[self.time_col, self.target_col,
                                         f'{self.target_col}_actual', 'window_id'])

        combined = pd.concat(all_results, ignore_index=True)

        if verbose:
            y_true = combined[f'{self.target_col}_actual'].values
            y_pred = combined[self.target_col].values
            from PipelineTS.spinesTS.metrics import mae
            print(f"\n  Rolling MAE: {mae(y_true, y_pred):.4f} "
                  f"({window_id} windows, {len(combined)} predictions)")

        return combined

    def score(
        self,
        results: pd.DataFrame,
        metrics: Optional[dict] = None,
    ) -> dict:
        """Evaluate rolling prediction results.

        Parameters
        ----------
        results : pd.DataFrame
            Output from predict().
        metrics : dict or None
            {name: callable(y_true, y_pred)}. If None, uses MAE and RMSE.

        Returns
        -------
        dict
            {metric_name: {'overall': float, 'per_window': list}}
        """
        if metrics is None:
            from PipelineTS.spinesTS.metrics import mae, rmse
            metrics = {'MAE': mae, 'RMSE': rmse}

        y_true = results[f'{self.target_col}_actual'].values
        y_pred = results[self.target_col].values

        output = {}
        for mname, mfunc in metrics.items():
            overall = float(mfunc(y_true, y_pred))
            per_window = []
            for wid in results['window_id'].unique():
                mask = results['window_id'] == wid
                yt = results.loc[mask, f'{self.target_col}_actual'].values
                yp = results.loc[mask, self.target_col].values
                per_window.append(float(mfunc(yt, yp)))
            output[mname] = {'overall': overall, 'per_window': per_window}

        return output

    # Backward-compatible aliases
    run = predict
    evaluate = score
