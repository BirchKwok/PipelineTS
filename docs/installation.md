# Installation Guide
# 安装指南

---

## Requirements / 环境要求

PipelineTS requires Python >= 3.9.
PipelineTS 需要 Python >= 3.9。

Supported Python versions: 3.9, 3.10, 3.11, 3.12, 3.13.
支持的 Python 版本：3.9、3.10、3.11、3.12、3.13。

---

## Install via pip / 通过 pip 安装

```bash
pip install PipelineTS
```

This will install PipelineTS along with all core dependencies.
这将安装 PipelineTS 及其所有核心依赖。

---

## Core Dependencies / 核心依赖

The following packages are automatically installed:
以下包会被自动安装：

| Package / 包 | Minimum Version / 最低版本 | Purpose / 用途 |
|---|---|---|
| `numpy` | >= 1.24.3 | Numerical computation / 数值计算 |
| `pandas` | >= 2.0.3 | Data manipulation / 数据处理 |
| `scikit-learn` | >= 1.3.0 | ML utilities and metrics / 机器学习工具和指标 |
| `torch` | >= 1.8.0 | Neural network backend / 神经网络后端 |
| `xgboost` | >= 1.6.0 | XGBoost model / XGBoost 模型 |
| `lightgbm` | >= 3.3.5 | LightGBM model / LightGBM 模型 |
| `catboost` | >= 1.2.2 | CatBoost model / CatBoost 模型 |
| `statsmodels` | >= 0.14.0 | Statistical models (ARIMA) / 统计模型 (ARIMA) |
| `matplotlib` | >= 3.7.1 | Plotting / 绑图 |
| `scipy` | >= 1.7.3 | Scientific computing / 科学计算 |

---

## Optional Dependencies / 可选依赖

### Prophet / Prophet 模型

To use the `ProphetModel`, install the `prophet` package separately:
要使用 `ProphetModel`，需要单独安装 `prophet` 包：

```bash
pip install prophet
```

### Optuna (for Hyperparameter Tuning) / Optuna（用于超参数调优）

To use Optuna for hyperparameter tuning (see [Advanced Features](advanced.md)):
要使用 Optuna 进行超参数调优（参见 [高级功能](advanced.md)）：

```bash
pip install optuna
```

---

## GPU Acceleration / GPU 加速

PipelineTS neural network models support multiple computing backends:
PipelineTS 神经网络模型支持多种计算后端：

| Backend / 后端 | Accelerator Value / accelerator 值 | Note / 备注 |
|---|---|---|
| Auto-detect / 自动检测 | `'auto'` | Recommended / 推荐 |
| CPU | `'cpu'` | Always available / 始终可用 |
| CUDA (NVIDIA GPU) | `'cuda'` | Requires CUDA-enabled PyTorch / 需要支持 CUDA 的 PyTorch |
| MPS (Apple Silicon) | `'mps'` | macOS with Apple Silicon / macOS Apple 芯片 |

To use CUDA, install the CUDA-enabled version of PyTorch:
要使用 CUDA，请安装支持 CUDA 的 PyTorch 版本：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Verify Installation / 验证安装

```python
import PipelineTS
print(PipelineTS.__version__)

from PipelineTS.pipeline import ModelPipeline
print(ModelPipeline.list_all_available_models())
```

If the above runs without errors, PipelineTS is installed correctly.
如果以上代码运行无误，则 PipelineTS 安装成功。

---

## Troubleshooting / 故障排除

### Common Issues / 常见问题

**Problem**: `ImportError: No module named 'prophet'`
**问题**：`ImportError: No module named 'prophet'`

**Solution**: Prophet is optional. Install it with `pip install prophet`, or exclude it from the pipeline:
**解决方案**：Prophet 是可选的。使用 `pip install prophet` 安装，或在管道中排除它：

```python
pipeline = ModelPipeline(..., exclude_models=['prophet'])
```

**Problem**: MPS backend error on macOS for certain models
**问题**：macOS 上某些模型的 MPS 后端报错

**Solution**: Force CPU backend for those models:
**解决方案**：为这些模型强制使用 CPU 后端：

```python
pipeline = ModelPipeline(
    ...,
    n_hits__accelerator='cpu',
    tft__accelerator='cpu',
)
```
