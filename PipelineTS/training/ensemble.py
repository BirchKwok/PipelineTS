"""Ensemble methods for combining multiple PipelineTS models.

Provides weighted averaging and stacking ensembles. All prediction
aggregation is vectorized with numpy for performance.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Callable, Union


class WeightedEnsemble:
    """Weighted average ensemble of multiple PipelineTS models.

    Parameters
    ----------
    models : list of (name, model) tuples
        Pre-initialized PipelineTS model instances.
    time_col : str
        Datetime column name.
    target_col : str
        Target column name.
    weights : list of float or 'auto', default='auto'
        Model weights. 'auto' assigns inverse-error weights after fitting.
    metric : callable or None
        Scoring function for auto weight computation.
        Signature: metric(y_true, y_pred) -> float. Lower is better.

    Examples
    --------
    >>> from PipelineTS.ml_model import TorchBoostingForestModel, TorchBaggingForestModel
    >>> models = [
    ...     ('boosting', TorchBoostingForestModel(time_col='date', target_col='value', lags=12)),
    ...     ('bagging', TorchBaggingForestModel(time_col='date', target_col='value', lags=12)),
    ... ]
    >>> ens = WeightedEnsemble(models, time_col='date', target_col='value')
    >>> ens.fit(data)
    >>> result = ens.predict(10)
    """

    def __init__(
        self,
        models: list,
        time_col: str,
        target_col: str,
        weights: Union[list, str] = 'auto',
        metric: Optional[Callable] = None,
    ):
        self.models = models
        self.time_col = time_col
        self.target_col = target_col
        self._auto_weights = (weights == 'auto')
        if not self._auto_weights:
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

        if metric is None:
            from PipelineTS.metrics import mae
            self.metric = mae
        else:
            self.metric = metric

        self._fitted = False

    def fit(self, data: pd.DataFrame, valid_data: Optional[pd.DataFrame] = None) -> 'WeightedEnsemble':
        """Fit all models and optionally compute auto weights.

        Parameters
        ----------
        data : pd.DataFrame
            Training data.
        valid_data : pd.DataFrame or None
            If provided and weights='auto', used for weight computation.
            If None, uses the last 20% of data for validation.

        Returns
        -------
        self
        """
        for name, model in self.models:
            model.fit(data)

        if self._auto_weights:
            if valid_data is None:
                n = len(data)
                n_val = max(1, n // 5)
                valid_data = data.iloc[-n_val:]

            n_pred = len(valid_data)
            y_true = valid_data[self.target_col].values
            errors = []

            for name, model in self.models:
                try:
                    pred = model.predict(n_pred)
                    y_pred = pred[self.target_col].values[:len(y_true)]
                    err = self.metric(y_true, y_pred)
                    errors.append(max(err, 1e-10))
                except Exception:
                    errors.append(1e10)

            # Inverse-error weights
            inv_errors = [1.0 / e for e in errors]
            total = sum(inv_errors)
            self.weights = [w / total for w in inv_errors]

        self._fitted = True
        return self

    def predict(self, n: int) -> pd.DataFrame:
        """Generate weighted ensemble prediction.

        Parameters
        ----------
        n : int
            Number of steps to forecast.

        Returns
        -------
        pd.DataFrame
            Prediction with time_col, target_col, and optionally
            target_col_lower, target_col_upper.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        predictions = []
        for name, model in self.models:
            pred = model.predict(n)
            predictions.append(pred)

        # Weighted average of target column
        result = predictions[0][[self.time_col]].copy()
        weighted_sum = np.zeros(n)
        for i, pred in enumerate(predictions):
            values = pred[self.target_col].values[:n]
            weighted_sum[:len(values)] += self.weights[i] * values

        result[self.target_col] = weighted_sum

        # Average interval bounds if available
        lower_col = f'{self.target_col}_lower'
        upper_col = f'{self.target_col}_upper'
        has_lower = all(lower_col in p.columns for p in predictions)
        has_upper = all(upper_col in p.columns for p in predictions)

        if has_lower and has_upper:
            lower_sum = np.zeros(n)
            upper_sum = np.zeros(n)
            for i, pred in enumerate(predictions):
                lower_sum += self.weights[i] * pred[lower_col].values[:n]
                upper_sum += self.weights[i] * pred[upper_col].values[:n]
            result[lower_col] = lower_sum
            result[upper_col] = upper_sum

        return result

    def get_weights(self) -> dict:
        """Return model weights as a dict.

        Returns
        -------
        dict
            {model_name: weight}
        """
        return {name: w for (name, _), w in zip(self.models, self.weights)}


class StackingEnsemble:
    """Stacking ensemble that trains a meta-learner on base model predictions.

    Base models produce predictions via cross-validation. A simple
    ridge regression meta-learner combines them for the final forecast.

    Parameters
    ----------
    models : list of (name, model) tuples
        Pre-initialized PipelineTS model instances.
    time_col : str
        Datetime column name.
    target_col : str
        Target column name.
    n_folds : int, default=3
        Number of temporal CV folds for generating meta-features.

    Examples
    --------
    >>> ens = StackingEnsemble(models, time_col='date', target_col='value')
    >>> ens.fit(data)
    >>> result = ens.predict(10)
    """

    def __init__(
        self,
        models: list,
        time_col: str,
        target_col: str,
        n_folds: int = 3,
    ):
        self.models = models
        self.time_col = time_col
        self.target_col = target_col
        self.n_folds = n_folds
        self._meta_model = None
        self._fitted = False

    def fit(self, data: pd.DataFrame) -> 'StackingEnsemble':
        """Fit base models and meta-learner.

        Parameters
        ----------
        data : pd.DataFrame
            Training data.

        Returns
        -------
        self
        """
        from sklearn.linear_model import Ridge

        df = data.sort_values(self.time_col).reset_index(drop=True)
        n = len(df)

        # Generate meta-features via temporal CV
        fold_size = n // (self.n_folds + 1)
        meta_X = []
        meta_y = []

        for fold in range(self.n_folds):
            train_end = fold_size * (fold + 1) + fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start:
                continue

            train_fold = df.iloc[:train_end].copy()
            test_fold = df.iloc[test_start:test_end].copy()
            n_test = len(test_fold)

            row_preds = []
            for name, model in self.models:
                fold_model = deepcopy(model)
                try:
                    fold_model.fit(train_fold)
                    pred = fold_model.predict(n_test)
                    row_preds.append(pred[self.target_col].values[:n_test])
                except Exception:
                    row_preds.append(np.full(n_test, np.nan))
                del fold_model

            # Stack predictions as columns
            stacked = np.column_stack(row_preds)
            valid_rows = ~np.any(np.isnan(stacked), axis=1)
            if valid_rows.any():
                meta_X.append(stacked[valid_rows])
                meta_y.append(test_fold[self.target_col].values[:n_test][valid_rows])

        if meta_X:
            meta_X = np.vstack(meta_X)
            meta_y = np.concatenate(meta_y)

            self._meta_model = Ridge(alpha=1.0)
            self._meta_model.fit(meta_X, meta_y)
        else:
            # Fallback: equal weights
            self._meta_model = None

        # Refit all base models on full data
        for name, model in self.models:
            model.fit(data)

        self._fitted = True
        return self

    def predict(self, n: int) -> pd.DataFrame:
        """Generate stacking ensemble prediction.

        Parameters
        ----------
        n : int
            Number of steps to forecast.

        Returns
        -------
        pd.DataFrame
            Prediction DataFrame.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        base_preds = []
        time_col_values = None

        for name, model in self.models:
            pred = model.predict(n)
            base_preds.append(pred[self.target_col].values[:n])
            if time_col_values is None:
                time_col_values = pred[self.time_col].values[:n]

        stacked = np.column_stack(base_preds)

        if self._meta_model is not None:
            final_pred = self._meta_model.predict(stacked)
        else:
            final_pred = np.mean(stacked, axis=1)

        result = pd.DataFrame({
            self.time_col: time_col_values,
            self.target_col: final_pred,
        })

        return result
