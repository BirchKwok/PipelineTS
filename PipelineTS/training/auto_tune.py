"""Built-in hyperparameter tuning wrapper for PipelineTS models.

Uses Optuna when available, falls back to random search otherwise.
Designed for minimal overhead — each trial is a full fit+evaluate cycle.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional, Callable, Union


class AutoTune:
    """Hyperparameter tuning for any PipelineTS model.

    Parameters
    ----------
    model_class : class
        A PipelineTS model class (e.g. TorchBoostingForestModel, TCNModel).
    time_col : str
        Datetime column name.
    target_col : str
        Target column name.
    lags : int
        Number of lagged time steps.
    metric : callable
        Scoring function with signature metric(y_true, y_pred) -> float.
    metric_less_is_better : bool, default=True
        Whether lower metric values are better.
    n_trials : int, default=20
        Number of tuning trials.
    test_size : float or int, default=0.2
        Fraction or number of rows for the validation set (temporal split).
    fixed_params : dict or None, default=None
        Parameters that should not be tuned (passed to every trial).
    random_state : int, default=0
        Random seed for reproducibility.

    Examples
    --------
    >>> from PipelineTS.ml_model import TorchBoostingForestModel
    >>> from PipelineTS.spinesTS.metrics import mae
    >>> tuner = AutoTune(
    ...     model_class=TorchBoostingForestModel,
    ...     time_col='date', target_col='value', lags=12,
    ...     metric=mae, n_trials=30,
    ...     fixed_params={'verbose': False},
    ... )
    >>> best_model, best_params, history = tuner.fit(data, search_space={
    ...     'n_estimators': ('int', 50, 500),
    ...     'learning_rate': ('float', 0.01, 0.3, True),  # True = log scale
    ...     'max_depth': ('int', 3, 10),
    ... })
    """

    def __init__(
        self,
        model_class,
        time_col: str,
        target_col: str,
        lags: int,
        metric: Callable,
        metric_less_is_better: bool = True,
        n_trials: int = 20,
        test_size: Union[float, int] = 0.2,
        fixed_params: Optional[dict] = None,
        random_state: int = 0,
    ):
        self.model_class = model_class
        self.time_col = time_col
        self.target_col = target_col
        self.lags = lags
        self.metric = metric
        self.metric_less_is_better = metric_less_is_better
        self.n_trials = n_trials
        self.test_size = test_size
        self.fixed_params = fixed_params or {}
        self.random_state = random_state

    def fit(
        self,
        data: pd.DataFrame,
        search_space: dict,
        verbose: bool = True,
    ) -> tuple:
        """Run hyperparameter tuning.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.
        search_space : dict
            Parameter search space. Each value is a tuple:
            - ('int', low, high) for integer parameters
            - ('float', low, high) or ('float', low, high, True) for float (log=True for log scale)
            - ('categorical', [list of choices]) for categorical
        verbose : bool, default=True
            Print progress.

        Returns
        -------
        tuple of (best_model, best_params, history_df)
            - best_model: fitted model with best hyperparameters
            - best_params: dict of best hyperparameters
            - history_df: DataFrame with all trial results
        """
        # Temporal split
        df = data.sort_values(self.time_col).reset_index(drop=True)
        n = len(df)
        if isinstance(self.test_size, float):
            n_test = max(1, int(n * self.test_size))
        else:
            n_test = min(self.test_size, n - self.lags - 1)

        train_df = df.iloc[:n - n_test].copy()
        test_df = df.iloc[n - n_test:].copy()

        try:
            return self._run_optuna(train_df, test_df, n_test, search_space, verbose)
        except ImportError:
            if verbose:
                print("Optuna not available. Using random search.")
            return self._run_random(train_df, test_df, n_test, search_space, verbose)

    def _run_optuna(self, train_df, test_df, n_test, search_space, verbose):
        """Run tuning with Optuna."""
        import optuna

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        history = []

        def objective(trial):
            params = self._sample_optuna(trial, search_space)
            score = self._evaluate_params(params, train_df, test_df, n_test)
            history.append({**params, 'score': score})
            return score

        direction = 'minimize' if self.metric_less_is_better else 'maximize'
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=verbose)

        best_params = study.best_params
        history_df = pd.DataFrame(history)

        # Refit with best params on full data
        best_model = self._build_and_fit(best_params, train_df)

        if verbose:
            print(f"\nBest score: {study.best_value:.6f}")
            print(f"Best params: {best_params}")

        return best_model, best_params, history_df

    def _run_random(self, train_df, test_df, n_test, search_space, verbose):
        """Fallback random search."""
        rng = np.random.RandomState(self.random_state)
        history = []
        best_score = np.inf if self.metric_less_is_better else -np.inf
        best_params = {}

        for i in range(self.n_trials):
            params = self._sample_random(rng, search_space)
            score = self._evaluate_params(params, train_df, test_df, n_test)
            history.append({**params, 'score': score})

            is_better = (score < best_score) if self.metric_less_is_better else (score > best_score)
            if is_better:
                best_score = score
                best_params = params.copy()

            if verbose:
                print(f"  Trial {i + 1}/{self.n_trials}: score={score:.6f}  "
                      f"best={best_score:.6f}")

        history_df = pd.DataFrame(history)
        best_model = self._build_and_fit(best_params, train_df)

        if verbose:
            print(f"\nBest score: {best_score:.6f}")
            print(f"Best params: {best_params}")

        return best_model, best_params, history_df

    def _evaluate_params(self, params, train_df, test_df, n_test):
        """Build model, fit, predict, score."""
        try:
            model = self._build_model(params)
            model.fit(train_df)
            pred_df = model.predict(n_test)
            y_true = test_df[self.target_col].values
            y_pred = pred_df[self.target_col].values[:len(y_true)]
            return float(self.metric(y_true, y_pred))
        except Exception as e:
            return np.inf if self.metric_less_is_better else -np.inf

    def _build_model(self, params):
        """Instantiate model with given params."""
        all_params = {
            'time_col': self.time_col,
            'target_col': self.target_col,
            'lags': self.lags,
            **self.fixed_params,
            **params,
        }
        # Filter to only params the model accepts
        from spinesUtils.asserts import check_has_param
        filtered = {k: v for k, v in all_params.items()
                    if check_has_param(self.model_class, k)}
        return self.model_class(**filtered)

    def _build_and_fit(self, params, train_df):
        """Build and fit the final model."""
        model = self._build_model(params)
        model.fit(train_df)
        return model

    @staticmethod
    def _sample_optuna(trial, search_space):
        """Sample parameters using Optuna trial."""
        params = {}
        for name, spec in search_space.items():
            ptype = spec[0]
            if ptype == 'int':
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif ptype == 'float':
                log = spec[3] if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
            elif ptype == 'categorical':
                params[name] = trial.suggest_categorical(name, spec[1])
        return params

    @staticmethod
    def _sample_random(rng, search_space):
        """Sample parameters using random search."""
        params = {}
        for name, spec in search_space.items():
            ptype = spec[0]
            if ptype == 'int':
                params[name] = int(rng.randint(spec[1], spec[2] + 1))
            elif ptype == 'float':
                log = spec[3] if len(spec) > 3 else False
                if log:
                    params[name] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
                else:
                    params[name] = float(rng.uniform(spec[1], spec[2]))
            elif ptype == 'categorical':
                params[name] = spec[1][rng.randint(len(spec[1]))]
        return params

    # Backward-compatible alias
    run = fit
