"""Chronos-2 family: Zero-shot foundation models for time series forecasting.

Wraps Amazon/AutoGluon Chronos-2 pretrained models for seamless integration
with PipelineTS.  Three model sizes are provided:

- ``Chronos2Model``       — amazon/chronos-2           (120M params)
- ``Chronos2SynthModel``  — autogluon/chronos-2-synth  (120M params)
- ``Chronos2SmallModel``  — autogluon/chronos-2-small  (28M params)

Requires: pip install chronos-forecasting
"""

import warnings

import numpy as np
import pandas as pd
from spinesUtils.asserts import raise_if

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp


def _import_chronos():
    """Lazy import of chronos package with helpful error message."""
    try:
        import chronos
        return chronos
    except ImportError:
        raise ImportError(
            "chronos-forecasting is required for Chronos models. "
            "Install it with: pip install chronos-forecasting"
        )


class _ChronosBase(StatisticModelMixin, IntervalEstimationMixin):
    """Base class for all Chronos-2 family models.

    Subclasses only need to set ``_HF_PATH`` to the HuggingFace model id.
    """

    # Subclasses override this
    _HF_PATH: str = None

    def __init__(
            self,
            time_col,
            target_col,
            lags=1,
            quantile=0.9,
            device_map='auto',
            **chronos_configs
    ):
        super().__init__(time_col=time_col, target_col=target_col)

        hf_path = self._HF_PATH

        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
            'hf_path': hf_path,
            'device_map': device_map,
        })

        self._pipeline = None
        self._train_data = None

    def _define_model(self):
        """Not used — model is loaded from pretrained weights."""
        return None

    def _load_pipeline(self):
        """Load the Chronos-2 pipeline (lazy, on first use)."""
        if self._pipeline is not None:
            return self._pipeline

        chronos = _import_chronos()
        hf_path = self.all_configs['hf_path']
        device_map = self.all_configs['device_map']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipeline = chronos.Chronos2Pipeline.from_pretrained(
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
        """Internal prediction using Chronos-2 predict_df API.

        Returns a DataFrame with time_col and target_col columns.
        """
        pipeline = self._load_pipeline()
        time_col = self.all_configs['time_col']
        target_col = self.all_configs['target_col']
        known_cols = getattr(self, '_known_cov_cols', [])

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

        # Future covariates (all Chronos-2 models support this)
        future_df = None
        if known_cols and future_covariates is not None:
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


# ── Concrete model classes ──────────────────────────────────────────

class Chronos2Model(_ChronosBase):
    """Chronos-2: amazon/chronos-2 (120M params).

    The flagship Chronos-2 model with full covariate support.
    Best accuracy among the three variants.
    """
    _HF_PATH = 'amazon/chronos-2'


class Chronos2SynthModel(_ChronosBase):
    """Chronos-2-Synth: autogluon/chronos-2-synth (120M params).

    Trained on synthetic data. Same architecture as Chronos-2 but
    with a different training corpus.
    """
    _HF_PATH = 'autogluon/chronos-2-synth'


class Chronos2SmallModel(_ChronosBase):
    """Chronos-2-Small: autogluon/chronos-2-small (28M params).

    Lightweight variant of Chronos-2. Faster inference with smaller
    memory footprint, suitable for resource-constrained environments.
    """
    _HF_PATH = 'autogluon/chronos-2-small'


# Backward-compatible alias — defaults to Chronos2Model
ChronosModel = Chronos2Model
