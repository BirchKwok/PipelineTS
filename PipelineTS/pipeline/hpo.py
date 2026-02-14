"""Optuna-based Hyperparameter Optimization for PipelineTS models.

Provides per-model search spaces and a lightweight HPO runner that
integrates with SmartRouter and ModelPipeline via double-underscore
kwargs.

Requires: ``pip install optuna``
"""
import time
import warnings

import numpy as np
import pandas as pd


def _lazy_import_optuna():
    """Lazy import optuna with clear error message."""
    try:
        import optuna
        return optuna
    except ImportError:
        raise ImportError(
            "optuna is required for HPO. Install with: pip install optuna"
        )


# ---------------------------------------------------------------------------
#  Per-model search space definitions
# ---------------------------------------------------------------------------
# Each search space is a dict mapping param_name -> (type, *args).
# Types: 'int', 'float', 'loguniform', 'categorical'

NN_LIGHT_SEARCH_SPACE = {
    'learning_rate': ('loguniform', 1e-4, 1e-2),
    'epochs': ('int', 500, 3000),
}

NN_HEAVY_SEARCH_SPACE = {
    'learning_rate': ('loguniform', 5e-5, 5e-3),
    'epochs': ('int', 1000, 3000),
}

PROPHET_SEARCH_SPACE = {
    'changepoint_prior_scale': ('loguniform', 0.001, 0.5),
}

DEEP_FOREST_SEARCH_SPACE = {
    'n_trees': ('int', 16, 64),
    'tree_depth': ('int', 3, 6),
    'n_layers': ('int', 2, 5),
    'learning_rate': ('loguniform', 1e-3, 0.02),
    'n_epochs': ('int', 200, 800),
}

TORCH_TREE_SEARCH_SPACE = {
    'n_trees': ('int', 32, 192),
    'tree_depth': ('int', 3, 7),
    'learning_rate': ('loguniform', 1e-3, 0.05),
    'n_epochs': ('int', 200, 800),
}

TORCH_CASCADE_SEARCH_SPACE = {
    'n_trees': ('int', 16, 64),
    'tree_depth': ('int', 3, 6),
    'n_layers': ('int', 2, 5),
    'learning_rate': ('loguniform', 1e-3, 0.02),
    'n_epochs': ('int', 200, 800),
}

ARIMA_SEARCH_SPACE = {}  # AutoARIMA does its own grid search

# Model name -> search space mapping
MODEL_SEARCH_SPACES = {
    # All tree models now use differentiable torch trees
    'lightgbm': TORCH_TREE_SEARCH_SPACE,
    'xgboost': TORCH_TREE_SEARCH_SPACE,
    'catboost': TORCH_TREE_SEARCH_SPACE,
    'random_forest': TORCH_TREE_SEARCH_SPACE,
    'torch_boosting_forest': TORCH_TREE_SEARCH_SPACE,
    'torch_bagging_forest': TORCH_TREE_SEARCH_SPACE,
    'wide_gbrt': TORCH_TREE_SEARCH_SPACE,
    'multi_output_model': TORCH_TREE_SEARCH_SPACE,
    'multi_step_model': TORCH_TREE_SEARCH_SPACE,
    'regressor_chain': TORCH_TREE_SEARCH_SPACE,
    'deep_forest': DEEP_FOREST_SEARCH_SPACE,
    # NN light
    'd_linear': NN_LIGHT_SEARCH_SPACE,
    'n_linear': NN_LIGHT_SEARCH_SPACE,
    'tide': NN_LIGHT_SEARCH_SPACE,
    'tcn': NN_LIGHT_SEARCH_SPACE,
    # NN medium
    'n_beats': NN_HEAVY_SEARCH_SPACE,
    'n_hits': NN_HEAVY_SEARCH_SPACE,
    'stacking_rnn': NN_HEAVY_SEARCH_SPACE,
    'patch_rnn': NN_HEAVY_SEARCH_SPACE,
    'time2vec': NN_HEAVY_SEARCH_SPACE,
    'gau': NN_HEAVY_SEARCH_SPACE,
    # NN heavy
    'transformer': NN_HEAVY_SEARCH_SPACE,
    'tft': NN_HEAVY_SEARCH_SPACE,
    'itransformer': NN_HEAVY_SEARCH_SPACE,
    'srs_net': NN_HEAVY_SEARCH_SPACE,
    'deepar': NN_HEAVY_SEARCH_SPACE,
    # Statistic
    'prophet': PROPHET_SEARCH_SPACE,
    'auto_arima': ARIMA_SEARCH_SPACE,
    # Foundation
    'chronos_2': {},  # Chronos-2 family: zero-shot, no tunable hyperparams
    'chronos_2_synth': {},
    'chronos_2_small': {},
}


def _suggest_param(trial, name, spec):
    """Use an Optuna trial to suggest a parameter given its spec.

    Parameters
    ----------
    trial : optuna.Trial
    name : str
        Parameter name.
    spec : tuple
        (type, *args) where type is 'int', 'float', 'loguniform', 'categorical'.
    """
    ptype = spec[0]
    if ptype == 'int':
        return trial.suggest_int(name, spec[1], spec[2])
    elif ptype == 'float':
        return trial.suggest_float(name, spec[1], spec[2])
    elif ptype == 'loguniform':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


class OptunaHPO:
    """Lightweight Optuna HPO runner for PipelineTS.

    Tunes hyperparameters for selected models using Optuna's TPE sampler
    and returns results in double-underscore format ready for ModelPipeline.

    Parameters
    ----------
    time_col : str
    target_col : str
    lags : int
    metric : callable
        Evaluation metric function (e.g. mae).
    metric_less_is_better : bool
    n_trials : int
        Number of Optuna trials per model.
    timeout_per_model : float or None
        Max seconds per model's HPO. None = no limit.
    verbose : bool
    random_state : int
    """

    def __init__(
        self,
        time_col,
        target_col,
        lags,
        metric,
        metric_less_is_better=True,
        n_trials=10,
        timeout_per_model=None,
        verbose=True,
        random_state=0,
    ):
        self.time_col = time_col
        self.target_col = target_col
        self.lags = lags
        self.metric = metric
        self.metric_less_is_better = metric_less_is_better
        self.n_trials = n_trials
        self.timeout_per_model = timeout_per_model
        self.verbose = verbose
        self.random_state = random_state

        self.results_ = {}  # model_name -> {best_params, best_value, n_trials}

    def optimize(self, model_names, train_data, valid_data,
                 base_hyperparams=None, **pipeline_kwargs):
        """Run HPO for each model and return optimized double-underscore kwargs.

        Parameters
        ----------
        model_names : list of str
            Models to tune.
        train_data : pd.DataFrame
            Training data.
        valid_data : pd.DataFrame
            Validation data for evaluation.
        base_hyperparams : dict or None
            Existing double-underscore kwargs to use as baseline.
            HPO results will override these.
        **pipeline_kwargs
            Additional kwargs passed to ModelPipeline (scaler, accelerator, etc.)

        Returns
        -------
        dict
            Double-underscore kwargs with optimized hyperparameters.
        """
        optuna = _lazy_import_optuna()

        best_kwargs = dict(base_hyperparams or {})

        for model_name in model_names:
            search_space = MODEL_SEARCH_SPACES.get(model_name, {})
            if not search_space:
                # No tunable params for this model
                continue

            result = self._optimize_one_model(
                optuna, model_name, search_space,
                train_data, valid_data, pipeline_kwargs,
            )

            if result is not None:
                self.results_[model_name] = result
                # Merge best params into kwargs
                for param_name, param_value in result['best_params'].items():
                    best_kwargs[f'{model_name}__{param_name}'] = param_value

        return best_kwargs

    def _optimize_one_model(self, optuna, model_name, search_space,
                            train_data, valid_data, pipeline_kwargs):
        """Run Optuna optimization for a single model.

        Returns dict with best_params, best_value, n_trials or None.
        """
        from PipelineTS.pipeline.pipeline import ModelPipeline

        direction = 'minimize' if self.metric_less_is_better else 'maximize'

        # Suppress Optuna logs unless verbose
        optuna.logging.set_verbosity(
            optuna.logging.WARNING if not self.verbose else optuna.logging.INFO
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)

        n_valid = len(valid_data)

        def objective(trial):
            # Suggest hyperparameters
            trial_kwargs = {}
            for param_name, spec in search_space.items():
                val = _suggest_param(trial, param_name, spec)
                trial_kwargs[f'{model_name}__{param_name}'] = val

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    pipe = ModelPipeline(
                        time_col=self.time_col,
                        target_col=self.target_col,
                        lags=self.lags,
                        include_models=[model_name],
                        quantile=None,
                        metric=self.metric,
                        metric_less_is_better=self.metric_less_is_better,
                        random_state=self.random_state,
                        cv=2,  # minimal CV for speed during HPO
                        **pipeline_kwargs,
                        **trial_kwargs,
                    )
                    pipe.fit(train_data, valid_data=valid_data)

                    # Evaluate on validation data
                    preds = pipe.predict(n=n_valid)
                    y_true = valid_data[self.target_col].values[:n_valid]
                    y_pred = preds[self.target_col].values[:n_valid]
                    score = self.metric(y_true, y_pred)

                    return float(score)

            except Exception:
                # Return worst possible value on failure
                return float('inf') if self.metric_less_is_better else float('-inf')

        t0 = time.time()
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_per_model,
            show_progress_bar=False,
        )
        elapsed = time.time() - t0

        if len(study.trials) == 0 or study.best_trial is None:
            return None

        # Extract best params (without model prefix)
        best_params = {}
        for param_name in search_space:
            if param_name in study.best_params:
                best_params[param_name] = study.best_params[param_name]

        if self.verbose:
            from spinesUtils.logging import Logger
            logger = Logger(name='HPO')
            params_str = ', '.join(
                f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                for k, v in best_params.items()
            )
            logger.info(
                f"  {model_name}: best={study.best_value:.4f} "
                f"({len(study.trials)} trials, {elapsed:.1f}s) [{params_str}]"
            )

        return {
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'time': elapsed,
        }


def get_search_space(model_name):
    """Get the search space for a given model name.

    Returns an empty dict if the model has no tunable hyperparameters.
    """
    return MODEL_SEARCH_SPACES.get(model_name, {})
