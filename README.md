# PipelineTS

![PyPI](https://img.shields.io/pypi/v/PipelineTS)
![PyPI - License](https://img.shields.io/pypi/l/PipelineTS)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PipelineTS)
[![Downloads](https://pepy.tech/badge/pipelinets)](https://pepy.tech/project/pipelinets)

> **⚡ Simplest and most professional TimeSeries forecasting + AutoML.**  
> One line to forecast. Professional when you need it.

---

## Installation

```bash
pip install PipelineTS          # core (ML + stats)
pip install PipelineTS[torch]   # + PyTorch NN models
```

Python ≥ 3.9 required.

---

## Quick Start

### One-line forecast
```python
from PipelineTS import forecast

pred = forecast("sales.csv", n=12)   # auto-infers columns, cleans data, runs AutoML
```

### Diagnose → Forecast → Backtest
```python
from PipelineTS import diagnose, forecast, backtest

report = diagnose("sales.csv", horizon=12)   # data readiness check
pred   = forecast("sales.csv", n=12)         # AutoML prediction
score  = backtest("sales.csv", n=12)         # walk-forward evaluation

print(report["status"])      # READY / WARNING / NOT_READY
print(score["summary"])      # mean ± std of MAE across folds
```

### Progressive professional control
```python
from PipelineTS import AutoForecast

model = AutoForecast(
    time_col="date", target_col="value",
    horizon=12,
    preset="high_quality",   # fast | medium_quality | high_quality | best_quality
    quantile=0.9,            # 90% prediction intervals
)
model.fit(data)
pred = model.predict()

print(model.leader_board_)   # ranked model comparison
print(model.strategy_)       # routing decisions

model.save("forecaster.pts")
loaded = AutoForecast.load("forecaster.pts")
```

### ModelPipeline — full control
```python
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs
from sklearn.preprocessing import StandardScaler

pipeline = ModelPipeline(
    time_col="date", target_col="value",
    lags=12, quantile=0.9,
    include_models=["d_linear", "catboost", "prophet"],   # or 'all', 'nn', 'ml', 'light'
)
leaderboard = pipeline.fit(data)
pred = pipeline.predict(n=12)

# Per-model configuration
configs = PipelineConfigs([
    ("d_linear", "d_linear_std",  {"pipeline_configs": {"lags": 24, "scaler": StandardScaler()}}),
    ("catboost", "catboost_diff", {"pipeline_configs": {"differential_n": 1}}),
])
pipeline = ModelPipeline(..., configs=configs)
```

### SmartRouter — intelligent AutoML
```python
from PipelineTS.pipeline import SmartRouter

router = SmartRouter(
    time_col="date", target_col="value",
    n_predict=12,
    preset="medium_quality",          # auto-configures models, CV, search, ensemble
    ensemble_strategy="auto",
    quantile=0.9,
)
router.fit(data)
pred = router.predict(n=12)

# Multi-quantile intervals
result = router.predict_quantiles(n=12, levels=[0.5, 0.8, 0.9])

# Panel data (multiple series)
router = SmartRouter(time_col="date", target_col="sales", id_col="store_id")
router.fit(panel_data)

# Save / load
router.save("router.pts")
router = SmartRouter.load("router.pts")
```

---

## Features

| Capability | Details |
|---|---|
| **50+ built-in models** | 34 NN · ML · statistical · optional foundation models |
| **Zero-friction API** | `forecast()` auto-infers columns, cleans data, runs AutoML |
| **SmartRouter AutoML** | Profiles data → selects models, lags, scaler, hyperparams, ensemble |
| **Conformal intervals** | Distribution-free CQR for NNs + conformal prediction for all models |
| **Multi-quantile** | `predict_quantiles(levels=[0.5, 0.8, 0.9])` in one call |
| **Panel data** | `id_col` for native multi-series with per-series scaling |
| **Covariates** | Known future & past covariates for GBDT, Prophet, AutoARIMA |
| **Multivariate** | ITransformer / SRSNet: multi-input → multi-output |
| **Incremental learning** | `update(new_data)` warm-starts NNs, refits ML/stat models |
| **HPO** | Built-in Optuna search inside SmartRouter (`hpo_strategy='quick'`) |
| **Preprocessing toolkit** | Missing values, outlier detection, stationarity tests, frequency detection |
| **Feature engineering** | Fourier, calendar, holiday, lag rolling features |
| **Evaluation** | Walk-forward backtesting, residual analysis, multi-model comparison |
| **Visualization** | ACF/PACF, decomposition, forecast plot — with Chinese font auto-detection |
| **Save / load (.pts)** | Binary format with SHA-256 checksums and atomic writes |

---

## Models

### Neural Network (34)

| Key | Model |
|---|---|
| `d_linear` | DLinear — decomposition linear |
| `n_linear` | NLinear — simple linear mapping |
| `n_beats` | N-BEATS |
| `n_hits` | N-HiTS — hierarchical interpolation |
| `tft` | Temporal Fusion Transformer |
| `transformer` | Transformer encoder |
| `tide` | Time-series Dense Encoder |
| `gau` | Gated Attention Unit |
| `stacking_rnn` | RWKV linear RNN + gated residual |
| `time2vec` | Time2Vec + RWKV |
| `patch_rnn` | Patch-based RNN |
| `tcn` | Temporal Convolutional Network |
| `itransformer` | Inverted Transformer (multivariate) |
| `srs_net` | Selective Representation Space Net (multivariate) |
| `deepar` | DeepAR — RWKV encoder + Gaussian head |
| `timexer` / `time_mixer` | Modern modular TimeXer / TimeMixer variants |
| `timesnet` / `msgnet` | Frequency and multi-scale convolutional variants |
| `pyraformer` / `informer` / `reformer` | Efficient transformer-style variants |
| `etsformer` / `autoformer` / `fedformer` | Decomposition and frequency-aware variants |
| `lightts` / `tsmixer` / `wpmixer` | Lightweight mixer variants |
| `patchtst` / `multi_patch_former` | Patch transformer variants |
| `nonstationary_transformer` | Non-stationarity-aware transformer variant |
| `timefilter` / `seg_rnn` / `tirex` | Filtering, segmented RNN, and trend-residual variants |

Modern NN variants share modular reusable layers and a central spec registry to avoid duplicated wrapper/backbone boilerplate.

### Machine Learning (8)

| Key | Model |
|---|---|
| `torch_boosting_forest` | GPU gradient boosting (MART/DART) |
| `torch_bagging_forest` | GPU bagging forest |
| `deep_forest` | gcForest cascade ensemble |
| `wide_gbrt` | Wide-table GBRT with 40+ lag features |
| `multi_output_model` | Multi-output sklearn regressor |
| `multi_step_model` | Multi-step sklearn regressor |
| `regressor_chain` | Chained regressors |
| `gc_forest` | Classic gcForest |

### Statistical (2+)

| Key | Model |
|---|---|
| `prophet` | Custom Prophet-like with ridge regression |
| `auto_arima` | Auto ARIMA — native implementation, no statsmodels |
| `theta` / `ets` / `stat_ensemble` | Fast lightweight baselines |

### Foundation — zero-shot (6, optional)

```bash
pip install chronos-forecasting      # Chronos-2
pip install tirex-ts                 # TiRex foundation
pip install transformers==4.40.1     # Sundial / Time-MoE
```

| Key | Model |
|---|---|
| `chronos_2` | Amazon Chronos-2 (120M, covariate support) |
| `chronos_2_small` | Chronos-2-Small (28M) |
| `chronos_2_synth` | Chronos-2-Synth |
| `tirex_foundation` | NX-AI TiRex zero-shot foundation model |
| `sundial` | THUML Sundial / Timer 3.0 |
| `time_moe` | Time-MoE mixture-of-experts foundation model |

---

## Documentation

Full API reference, guides, and tutorials live in the [`docs/`](docs/) directory and the [`tutorials/`](tutorials/) folder.

| Resource | Link |
|---|---|
| Quick Start | [docs/quickstart.md](docs/quickstart.md) |
| All Models | [docs/models.md](docs/models.md) |
| Pipeline & PipelineConfigs | [docs/pipeline.md](docs/pipeline.md) |
| SmartRouter | [docs/advanced.md](docs/advanced.md) |
| Preprocessing | [docs/preprocessing.md](docs/preprocessing.md) |
| Feature Engineering | [docs/feature_engineering.md](docs/feature_engineering.md) |
| Evaluation & Metrics | [docs/evaluation.md](docs/evaluation.md) |
| Visualization | [docs/visualization.md](docs/visualization.md) |
| **Zero-Friction API** | [docs/api_reference.md#zero-friction-api](docs/api_reference.md) |
| API Reference | [docs/api_reference.md](docs/api_reference.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

**Tutorials (Jupyter notebooks):**

| # | Topic |
|---|---|
| 00 | [Zero-Friction API](tutorials/00_EasyAPI.ipynb) — `load_data`, `diagnose`, `forecast`, `AutoForecast`, `backtest` |
| 01 | [Quick Start Guide](tutorials/01_QuickStart_Guide.ipynb) |
| 02 | [All Models](tutorials/02_All_Models_Guide.ipynb) |
| 03 | [Multivariate Prediction](tutorials/03_Multivariate_Prediction.ipynb) |
| 04 | [Advanced Pipeline](tutorials/04_Advanced_Pipeline.ipynb) |
| 05 | [Preprocessing & Data](tutorials/05_Preprocessing_and_Data.ipynb) |
| 06 | [Hyperparameter Tuning](tutorials/06_Hyperparameter_Tuning.ipynb) |
| 07 | [Benchmarks](tutorials/07_Benchmarks.ipynb) |
| 08 | [Visualization](tutorials/08_Visualization.ipynb) |
| 09 | [Multi-Quantile Intervals](tutorials/09_Multi_Quantile_Intervals.ipynb) |
| 10 | [Multi-Series & Covariates](tutorials/10_Multi_Series_Covariates.ipynb) |
| 11 | [Incremental Learning](tutorials/11_Incremental_Learning.ipynb) |
| 12 | [SmartRouter & Pipeline Deep Dive](tutorials/12_SmartRouter_and_Pipeline.ipynb) |

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
