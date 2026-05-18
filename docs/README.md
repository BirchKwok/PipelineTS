# PipelineTS Documentation
# PipelineTS 文档

Welcome to the PipelineTS documentation. This guide covers everything you need to get started and make the most of the library.
欢迎阅读 PipelineTS 文档。本指南涵盖了入门和充分利用本库所需的全部内容。

---

## Contents / 目录

### Getting Started / 入门

- **[Zero-Friction API / 零配置 API](api_reference.md#zero-friction-api-pipelinets)**
  One-line `forecast()`, `diagnose()`, `AutoForecast`, and `backtest()` — no configuration needed.
  一行完成 `forecast()`、`diagnose()`、`AutoForecast` 和 `backtest()` — 无需配置。

- **[Installation Guide / 安装指南](installation.md)**
  How to install PipelineTS and its dependencies.
  如何安装 PipelineTS 及其依赖项。

- **[Quick Start Guide / 快速入门指南](quickstart.md)**
  Progressive guide from `forecast()` to `ModelPipeline` to `SmartRouter`.
  从 `forecast()` 到 `ModelPipeline` 到 `SmartRouter` 的逐步指南。

### Core Guides / 核心指南

- **[Model Reference / 模型参考](models.md)**
  Detailed reference for all 25+ built-in models, including GPU-accelerated tree models, parameters, and usage examples.
  所有 25+ 个内置模型的详细参考，包括 GPU 加速树模型、参数和使用示例。

- **[Pipeline Usage / 管道使用](pipeline.md)**
  How to use `ModelPipeline` for automatic model selection and comparison.
  如何使用 `ModelPipeline` 进行自动模型选择和比较。

- **[Preprocessing & Data / 数据预处理](preprocessing.md)**
  Built-in datasets, data scalers, sequence splitting, and evaluation metrics.
  内置数据集、数据缩放器、序列分割和评估指标。

### Visualization / 可视化

- **[Visualization / 可视化](visualization.md)**
  Comprehensive plotting toolkit with Chinese font support, forecast charts, leaderboard, residual diagnostics, and more.
  全面的绘图工具包，支持中文字体、预测图表、排行榜、残差诊断等。

### Advanced Topics / 高级主题

- **[Multivariate Prediction / 多变量预测](multivariate.md)**
  Multi-input/multi-output forecasting with ITransformer and SRSNet.
  使用 ITransformer 和 SRSNet 进行多输入/多输出预测。

- **[Advanced Features / 高级功能](advanced.md)**
  Hyperparameter tuning, GPU tree models, adaptive complexity, differencing, custom scalers, multi-quantile, covariates, incremental learning, HPO, and more.
  超参数调优、GPU 树模型、自适应复杂度、差分、自定义缩放器、多分位数、协变量、增量学习、HPO 等高级功能。

- **[API Reference / API 参考](api_reference.md)**
  Complete API reference for all public classes and functions.
  所有公共类和函数的完整 API 参考。

---

## Interactive Tutorials / 交互式教程

For hands-on learning, check out the [tutorials/](../tutorials/) directory which contains Jupyter notebooks covering:
如需动手实践，请查阅 [tutorials/](../tutorials/) 目录中的 Jupyter Notebook 教程：

| # | Tutorial / 教程 | Description / 描述 |
|---|---|---|
| 00 | [Zero-Friction API](../tutorials/00_EasyAPI.ipynb) | `load_data`, `infer_*`, `preprocess`, `diagnose`, `forecast`, `AutoForecast`, `backtest` |
| 01 | [Quick Start Guide](../tutorials/01_QuickStart_Guide.ipynb) | Basic usage and core workflow / 基本用法和核心工作流 |
| 02 | [All Models Guide](../tutorials/02_All_Models_Guide.ipynb) | Usage of all models including GPU tree models / 所有模型的用法（含 GPU 树模型） |
| 03 | [Multivariate Prediction](../tutorials/03_Multivariate_Prediction.ipynb) | Multi-input/multi-output forecasting / 多输入/多输出预测 |
| 04 | [Advanced Pipeline](../tutorials/04_Advanced_Pipeline.ipynb) | PipelineConfigs, scalers, metrics / 管道配置、缩放器、指标 |
| 05 | [Preprocessing & Data](../tutorials/05_Preprocessing_and_Data.ipynb) | Datasets, scalers, sequence splitting / 数据集、缩放器、序列分割 |
| 06 | [Hyperparameter Tuning](../tutorials/06_Hyperparameter_Tuning.ipynb) | Optuna integration for tuning / 使用 Optuna 进行超参数调优 |
| 07 | [Benchmarks](../tutorials/07_Benchmarks.ipynb) | Model benchmarking across datasets / 跨数据集的模型基准测试 |
| 08 | [Visualization](../tutorials/08_Visualization.ipynb) | Full visualization toolkit with Chinese fonts / 全面可视化工具包（含中文字体） |
| 09 | [Multi-Quantile Intervals](../tutorials/09_Multi_Quantile_Intervals.ipynb) | Multi-level prediction intervals / 多分位数预测区间 |
| 10 | [Multi-Series & Covariates](../tutorials/10_Multi_Series_Covariates.ipynb) | Panel data and external covariates / 面板数据与外部协变量 |
| 11 | [Incremental Learning](../tutorials/11_Incremental_Learning.ipynb) | Update models with new data / 增量学习更新模型 |
| 12 | [SmartRouter & Pipeline](../tutorials/12_SmartRouter_and_Pipeline.ipynb) | Core engines: ModelPipeline & SmartRouter deep dive / 核心引擎深度指南 |
