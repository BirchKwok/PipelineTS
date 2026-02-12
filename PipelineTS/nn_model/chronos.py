"""ChronosModel: Zero-shot foundation model for time series forecasting.

Wraps Amazon's Chronos pretrained models (Chronos-2, Chronos-Bolt, Chronos-T5)
for seamless integration with PipelineTS.

Requires: pip install chronos-forecasting
"""

import warnings

import numpy as np
import pandas as pd
from spinesUtils.asserts import raise_if

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp


# Available model presets
CHRONOS_MODELS = {
    # Chronos-2 (latest, covariate support)
    'chronos-2': 'amazon/chronos-2',
    # Chronos-Bolt (fast, efficient)
    'chronos-bolt-tiny': 'amazon/chronos-bolt-tiny',
    'chronos-bolt-mini': 'amazon/chronos-bolt-mini',
    'chronos-bolt-small': 'amazon/chronos-bolt-small',
    'chronos-bolt-base': 'amazon/chronos-bolt-base',
    # Chronos-T5 (original)
    'chronos-t5-tiny': 'amazon/chronos-t5-tiny',
    'chronos-t5-mini': 'amazon/chronos-t5-mini',
    'chronos-t5-small': 'amazon/chronos-t5-small',
    'chronos-t5-base': 'amazon/chronos-t5-base',
    'chronos-t5-large': 'amazon/chronos-t5-large',
}

# Default model — good balance of speed and accuracy
_DEFAULT_MODEL = 'chronos-bolt-small'


def _import_chronos():
    """Lazy import of chronos package with helpful error message."""
    try:
        import chronos
        return chronos
    except ImportError:
        raise ImportError(
            "chronos-forecasting is required for ChronosModel. "
            "Install it with: pip install chronos-forecasting"
        )


class ChronosModel(StatisticModelMixin, IntervalEstimationMixin):
    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            model_name=None,
            device_map='auto',
            **chronos_configs
    ):
        """
        ChronosModel: Zero-shot foundation model for time series forecasting.

        Uses Amazon's Chronos pretrained models for zero-shot forecasting.
        No training is needed — the model uses pretrained weights to generate
        predictions directly from historical data.

        Parameters
        ----------
        time_col : str
            The column containing time information.
        target_col : str
            The column containing the target variable.
        lags : int, optional, default: 1
            Kept for API compatibility with PipelineTS.
        quantile : float or None, optional, default: 0.9
            Quantile level for prediction intervals. None for point prediction only.
        model_name : str or None, optional, default: None
            Chronos model to use. Options:
            - 'chronos-2': Latest Chronos-2 (supports covariates)
            - 'chronos-bolt-tiny/mini/small/base': Fast Chronos-Bolt models
            - 'chronos-t5-tiny/mini/small/base/large': Original T5-based models
            - Any HuggingFace model path
            If None, defaults to 'chronos-bolt-small'.
        device_map : str, optional, default: 'auto'
            Device placement strategy. 'auto', 'cpu', 'cuda', 'mps', etc.
        **chronos_configs
            Additional keyword arguments passed to the Chronos pipeline.
        """
        super().__init__(time_col=time_col, target_col=target_col)

        if model_name is None:
            model_name = _DEFAULT_MODEL

        # Resolve model name to HuggingFace path
        if model_name in CHRONOS_MODELS:
            hf_path = CHRONOS_MODELS[model_name]
        else:
            hf_path = model_name

        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
            'model_name': model_name,
            'hf_path': hf_path,
            'device_map': device_map,
        })

        self._pipeline = None
        self._train_data = None
        self._is_chronos2 = 'chronos-2' in hf_path

    def _define_model(self):
        """Not used — model is loaded from pretrained weights."""
        return None

    def _load_pipeline(self):
        """Load the Chronos pipeline (lazy, on first use)."""
        if self._pipeline is not None:
            return self._pipeline

        chronos = _import_chronos()
        hf_path = self.all_configs['hf_path']
        device_map = self.all_configs['device_map']

        # Select the right pipeline class based on model type
        if 'chronos-2' in hf_path:
            pipeline_cls = chronos.Chronos2Pipeline
        elif 'chronos-bolt' in hf_path:
            pipeline_cls = chronos.ChronosBoltPipeline
        else:
            pipeline_cls = chronos.ChronosPipeline

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipeline = pipeline_cls.from_pretrained(
                hf_path,
                device_map=device_map,
            )

        return self._pipeline

    def fit(self, data, cv=5, **kwargs):
        """
        Store training data for prediction. Chronos is a zero-shot model,
        so no actual training is performed.

        Parameters
        ----------
        data : pd.DataFrame
            Historical time series data.
        cv : int, optional, default: 5
            Number of CV folds for conformal interval calibration.

        Returns
        -------
        self
        """
        check_time_col_is_timestamp(data, self.all_configs['time_col'])

        # Store a copy of the training data for prediction context
        self._train_data = data.copy()

        # Pre-load the pipeline during fit to catch import errors early
        self._load_pipeline()

        # Calculate conformal prediction intervals if quantile is set
        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = \
                self._calculate_confidence_interval(data, cv=cv)

        return self

    def _calculate_confidence_interval(self, data, cv=5):
        """Calculate conformal prediction intervals via expanding-window CV.

        Uses the Chronos model itself for CV predictions to collect residuals
        for conformal calibration.
        """
        signed_residuals = []
        time_col = self.all_configs['time_col']
        target_col = self.all_configs['target_col']
        id_col = self.all_configs.get('id_col')

        n = len(data)
        fold_size = max(1, n // (cv + 1))

        for i in range(cv):
            train_end = n - (cv - i) * fold_size
            valid_end = train_end + fold_size
            if train_end < 10 or valid_end > n:
                continue

            train_df = data.iloc[:train_end].copy()
            valid_df = data.iloc[train_end:valid_end].copy()
            n_pred = len(valid_df)

            try:
                preds = self._raw_predict(
                    context_data=train_df, n=n_pred, id_col=id_col
                )
                if preds is not None and len(preds) > 0:
                    if id_col and id_col in valid_df.columns:
                        # Multi-series: match per-series
                        for sid in valid_df[id_col].unique():
                            v_mask = valid_df[id_col] == sid
                            p_mask = preds[id_col] == sid if id_col in preds.columns else slice(None)
                            y_true = valid_df.loc[v_mask, target_col].values
                            y_pred = preds.loc[p_mask, target_col].values[:len(y_true)]
                            signed_residuals.extend((y_true - y_pred).tolist())
                    else:
                        y_true = valid_df[target_col].values
                        y_pred = preds[target_col].values[:n_pred]
                        signed_residuals.extend((y_true - y_pred).tolist())
            except Exception:
                continue

        return IntervalEstimationMixin._compute_conformal_quantiles(
            signed_residuals, coverage=self.all_configs['quantile']
        )

    def _raw_predict(self, context_data, n, id_col=None, future_covariates=None):
        """Internal prediction using Chronos pipeline.

        Returns a DataFrame with time_col and target_col columns.
        """
        pipeline = self._load_pipeline()
        time_col = self.all_configs['time_col']
        target_col = self.all_configs['target_col']
        known_cols = getattr(self, '_known_cov_cols', [])

        # Use predict_df for Chronos-2 and Chronos-Bolt (high-level API)
        if hasattr(pipeline, 'predict_df'):
            # Chronos predict_df requires an id column; add a synthetic one if missing
            _synthetic_id = '__chronos_item_id__'
            ctx = context_data.copy()
            effective_id_col = id_col if (id_col and id_col in ctx.columns) else _synthetic_id
            if effective_id_col == _synthetic_id:
                ctx[_synthetic_id] = 'series_0'

            predict_kwargs = {
                'prediction_length': n,
                'timestamp_column': time_col,
                'target': target_col,
                'id_column': effective_id_col,
            }

            # Future covariates for Chronos-2
            future_df = None
            if self._is_chronos2 and known_cols and future_covariates is not None:
                future_df = future_covariates.copy()
                if effective_id_col == _synthetic_id and _synthetic_id not in future_df.columns:
                    future_df[_synthetic_id] = 'series_0'
                if time_col not in future_df.columns:
                    last_dt = ctx[time_col].max()
                    freq = getattr(self, '_freq', 'D')
                    future_dates = pd.date_range(
                        start=last_dt, periods=n + 1, freq=freq
                    )[1:]
                    future_df[time_col] = future_dates[:len(future_df)]

            if future_df is not None:
                predict_kwargs['future_df'] = future_df

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_df = pipeline.predict_df(ctx, **predict_kwargs)

            # Extract point predictions from the result
            if 'predictions' in pred_df.columns:
                pred_col = 'predictions'
            elif '0.5' in pred_df.columns:
                pred_col = '0.5'
            elif 'mean' in pred_df.columns:
                pred_col = 'mean'
            else:
                exclude = {time_col, effective_id_col, id_col, _synthetic_id}
                remaining = [c for c in pred_df.columns if c not in exclude]
                pred_col = remaining[0] if remaining else pred_df.columns[-1]

            result = pd.DataFrame({
                time_col: pred_df[time_col].values if time_col in pred_df.columns
                else pd.date_range(
                    start=context_data[time_col].max(),
                    periods=n + 1, freq=getattr(self, '_freq', 'D')
                )[1:],
                target_col: pred_df[pred_col].values,
            })

            if id_col and id_col in pred_df.columns:
                result[id_col] = pred_df[id_col].values

            return result

        else:
            # Chronos-T5: lower-level tensor API
            import torch

            if id_col and id_col in context_data.columns:
                # Multi-series: predict each series independently
                all_results = []
                for sid, sdf in context_data.groupby(id_col):
                    values = torch.tensor(
                        sdf[target_col].values, dtype=torch.float32
                    ).unsqueeze(0)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        forecast = pipeline.predict(values, prediction_length=n)

                    # forecast shape: (1, num_samples, prediction_length)
                    median = np.median(forecast[0].numpy(), axis=0)
                    last_dt = sdf[time_col].max()
                    freq = getattr(self, '_freq', 'D')
                    future_dates = pd.date_range(
                        start=last_dt, periods=n + 1, freq=freq
                    )[1:]

                    res = pd.DataFrame({
                        time_col: future_dates,
                        target_col: median,
                    })
                    res[id_col] = sid
                    all_results.append(res)
                return pd.concat(all_results, ignore_index=True)
            else:
                values = torch.tensor(
                    context_data[target_col].values, dtype=torch.float32
                ).unsqueeze(0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = pipeline.predict(values, prediction_length=n)

                median = np.median(forecast[0].numpy(), axis=0)
                last_dt = context_data[time_col].max()
                freq = getattr(self, '_freq', 'D')
                future_dates = pd.date_range(
                    start=last_dt, periods=n + 1, freq=freq
                )[1:]

                return pd.DataFrame({
                    time_col: future_dates,
                    target_col: median,
                })

    def predict(self, n, future_covariates=None, **kwargs):
        """
        Generate predictions using the Chronos foundation model.

        Parameters
        ----------
        n : int
            Number of future time steps to predict.
        future_covariates : pd.DataFrame or None
            Future known covariate values (only used with Chronos-2).

        Returns
        -------
        pd.DataFrame
            Predictions with time_col and target_col columns.
        """
        raise_if(ValueError, self._train_data is None,
                 "Model has not been fitted yet. Call fit() first.")

        id_col = self.all_configs.get('id_col')
        known_covs = self.all_configs.get('known_covariates') or []
        self._known_cov_cols = [c for c in known_covs if c in self._train_data.columns]

        # Infer frequency from training data
        time_col = self.all_configs['time_col']
        try:
            freq = pd.infer_freq(self._train_data[time_col].sort_values().unique())
            if freq is None:
                freq = 'D'
        except Exception:
            freq = 'D'
        self._freq = freq

        res = self._raw_predict(
            context_data=self._train_data,
            n=n,
            id_col=id_col,
            future_covariates=future_covariates,
        )

        if id_col and id_col in res.columns:
            # Multi-series: apply interval prediction per series
            all_results = []
            for sid in res[id_col].unique():
                sid_res = res[res[id_col] == sid].copy()
                if self.all_configs['quantile'] is not None:
                    sid_res = self.interval_predict(sid_res)
                sid_res = self.chosen_cols(sid_res)
                sid_res[id_col] = sid
                all_results.append(sid_res)
            return pd.concat(all_results, ignore_index=True)

        if self.all_configs['quantile'] is not None:
            res = self.interval_predict(res)

        return self.chosen_cols(res)
