import warnings

import numpy as np
import pandas as pd
from spinesUtils.asserts import raise_if

from PipelineTS.base.base import StatisticModelMixin, IntervalEstimationMixin
from PipelineTS.utils import check_time_col_is_timestamp


def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch is required for foundation time-series models. Install it with: pip install torch")


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM
        _patch_transformers_cache_compat()
        return AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers is required for Sundial and Time-MoE models. "
            "Install a compatible version, e.g. pip install transformers==4.40.1"
        )


def _patch_transformers_cache_compat():
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, 'get_max_length') or not hasattr(DynamicCache, 'get_max_cache_shape'):
        return

    def get_max_length(self):
        return self.get_max_cache_shape()

    DynamicCache.get_max_length = get_max_length


def _import_tirex():
    try:
        from tirex import load_model
        return load_model
    except ImportError:
        raise ImportError("tirex-ts is required for TiRexFoundationModel. Install it with: pip install tirex-ts")


class _FoundationBase(StatisticModelMixin, IntervalEstimationMixin):
    _HF_PATH = None

    def __init__(self, time_col, target_col, lags=512, quantile=0.9, device_map='auto', **model_configs):
        super().__init__(time_col=time_col, target_col=target_col)
        self.all_configs.update({
            'lags': lags,
            'quantile': quantile,
            'time_col': time_col,
            'target_col': target_col,
            'quantile_error': (0, 0),
            'hf_path': model_configs.pop('hf_path', self._HF_PATH),
            'device_map': device_map,
            'model_configs': model_configs,
        })
        self._pipeline = None
        self._train_data = None
        self._freq = 'D'

    def _define_model(self):
        return None

    def _load_pipeline(self):
        raise NotImplementedError

    def fit(self, data, cv=5, **kwargs):
        check_time_col_is_timestamp(data, self.all_configs['time_col'])
        self._train_data = data.copy()
        self._load_pipeline()
        if self.all_configs['quantile'] is not None:
            self.all_configs['quantile_error'] = self._calculate_confidence_interval(data, cv=cv)
        return self

    def _calculate_confidence_interval(self, data, cv=5):
        signed_residuals = []
        time_col = self.all_configs['time_col']
        target_col = self.all_configs['target_col']
        id_col = self.all_configs.get('id_col')
        n = len(data)
        fold_size = max(1, n // (cv + 1))
        for i in range(cv):
            train_end = n - (cv - i) * fold_size
            valid_end = train_end + fold_size
            if train_end < max(10, self.all_configs['lags'] // 2) or valid_end > n:
                continue
            train_df = data.iloc[:train_end].copy()
            valid_df = data.iloc[train_end:valid_end].copy()
            try:
                preds = self._raw_predict(train_df, len(valid_df), id_col=id_col)
                if preds is None or preds.empty:
                    continue
                if id_col and id_col in valid_df.columns and id_col in preds.columns:
                    for sid in valid_df[id_col].unique():
                        y_true = valid_df.loc[valid_df[id_col] == sid, target_col].values
                        y_pred = preds.loc[preds[id_col] == sid, target_col].values[:len(y_true)]
                        signed_residuals.extend((y_true - y_pred).tolist())
                else:
                    y_true = valid_df[target_col].values
                    y_pred = preds[target_col].values[:len(y_true)]
                    signed_residuals.extend((y_true - y_pred).tolist())
            except Exception:
                continue
        return IntervalEstimationMixin._compute_conformal_quantiles(
            signed_residuals, coverage=self.all_configs['quantile']
        )

    def _infer_freq(self):
        time_col = self.all_configs['time_col']
        try:
            freq = pd.infer_freq(self._train_data[time_col].sort_values().unique())
            self._freq = freq or 'D'
        except Exception:
            self._freq = 'D'

    def _context_arrays(self, context_data, id_col=None):
        target_col = self.all_configs['target_col']
        lags = self.all_configs['lags']
        if id_col and id_col in context_data.columns:
            series_ids, arrays = [], []
            for sid, part in context_data.groupby(id_col, sort=False):
                values = part[target_col].astype(float).values[-lags:]
                if len(values) > 0:
                    series_ids.append(sid)
                    arrays.append(values)
            return series_ids, arrays
        values = context_data[target_col].astype(float).values[-lags:]
        return [None], [values]

    def _result_frame(self, forecasts, series_ids, context_data, n, id_col=None):
        time_col = self.all_configs['time_col']
        target_col = self.all_configs['target_col']
        rows = []
        for i, values in enumerate(forecasts):
            if id_col and series_ids[i] is not None:
                part = context_data[context_data[id_col] == series_ids[i]]
            else:
                part = context_data
            last_dt = part[time_col].max()
            dates = pd.date_range(start=last_dt, periods=n + 1, freq=self._freq)[1:]
            frame = pd.DataFrame({time_col: dates, target_col: values[:n]})
            if id_col and series_ids[i] is not None:
                frame[id_col] = series_ids[i]
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)

    def _raw_predict(self, context_data, n, id_col=None):
        raise NotImplementedError

    def predict(self, n, **kwargs):
        raise_if(ValueError, self._train_data is None, "Model has not been fitted yet. Call fit() first.")
        self._infer_freq()
        id_col = self.all_configs.get('id_col')
        res = self._raw_predict(self._train_data, n, id_col=id_col)
        if id_col and id_col in res.columns:
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


class _TransformersGenerateFoundationBase(_FoundationBase):
    _NORMALIZE_INPUT = False
    _NUM_SAMPLES = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        AutoModelForCausalLM = _import_transformers()
        model_kwargs = dict(self.all_configs.get('model_configs', {}))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._pipeline = AutoModelForCausalLM.from_pretrained(
                self.all_configs['hf_path'],
                device_map=self.all_configs['device_map'],
                trust_remote_code=True,
                **model_kwargs,
            )
        return self._pipeline

    def _extract_forecast(self, output, n):
        arr = output.detach().cpu().numpy()
        if arr.ndim == 3:
            if arr.shape[-1] >= n:
                samples = arr[..., -n:]
            elif arr.shape[1] >= n:
                samples = np.transpose(arr[:, -n:, :], (0, 2, 1))
            else:
                samples = arr.reshape(arr.shape[0], -1, arr.shape[-1])[..., :n]
            return samples.mean(axis=1)
        if arr.ndim == 2:
            return arr[:, -n:] if arr.shape[1] >= n else arr[:, :n]
        return arr.reshape(1, -1)[:, -n:]

    def _raw_predict(self, context_data, n, id_col=None):
        torch = _import_torch()
        model = self._load_pipeline()
        series_ids, arrays = self._context_arrays(context_data, id_col=id_col)
        forecasts = []
        for values in arrays:
            seq = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
            mean = seq.mean(dim=-1, keepdim=True)
            std = seq.std(dim=-1, keepdim=True).clamp_min(1e-6)
            model_input = (seq - mean) / std if self._NORMALIZE_INPUT else seq
            kwargs = {'max_new_tokens': n}
            if self._NUM_SAMPLES is not None:
                kwargs['num_samples'] = self._NUM_SAMPLES
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                output = model.generate(model_input, **kwargs)
            pred = torch.as_tensor(self._extract_forecast(output, n), dtype=torch.float32)
            if self._NORMALIZE_INPUT:
                pred = pred * std + mean
            forecasts.append(pred.squeeze(0).detach().cpu().numpy())
        return self._result_frame(forecasts, series_ids, context_data, n, id_col=id_col)


class SundialModel(_TransformersGenerateFoundationBase):
    _HF_PATH = 'thuml/sundial-base-128m'
    _NUM_SAMPLES = 20


class TimeMoEModel(_TransformersGenerateFoundationBase):
    _HF_PATH = 'Maple728/TimeMoE-50M'
    _NORMALIZE_INPUT = True


class TiRexFoundationModel(_FoundationBase):
    _HF_PATH = 'NX-AI/TiRex'

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        load_model = _import_tirex()
        model_kwargs = dict(self.all_configs.get('model_configs', {}))
        self._pipeline = load_model(self.all_configs['hf_path'], **model_kwargs)
        return self._pipeline

    def _raw_predict(self, context_data, n, id_col=None):
        torch = _import_torch()
        model = self._load_pipeline()
        series_ids, arrays = self._context_arrays(context_data, id_col=id_col)
        forecasts = []
        for values in arrays:
            seq = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
            output = model.forecast(context=seq, prediction_length=n)
            if isinstance(output, tuple):
                _, mean = output
            else:
                mean = output
            pred = torch.as_tensor(mean, dtype=torch.float32).detach().cpu().numpy().reshape(-1)[-n:]
            forecasts.append(pred)
        return self._result_frame(forecasts, series_ids, context_data, n, id_col=id_col)
