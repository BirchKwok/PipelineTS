# Model Reference
# 模型参考

PipelineTS includes 25+ built-in time series forecasting models across five categories.
PipelineTS 包含 25+ 个内置时间序列预测模型，分为五大类。

All models share a unified API: `fit(data)` for training and `predict(n)` for forecasting.
所有模型共享统一的 API：`fit(data)` 用于训练，`predict(n)` 用于预测。

---

## Common Parameters / 通用参数

The following parameters are shared by all models:
以下参数为所有模型共有：

| Parameter / 参数 | Type / 类型 | Description / 描述 |
|---|---|---|
| `time_col` | str | Name of the time column / 时间列名 |
| `target_col` | str or list | Name of the target column(s) / 目标列名（或列表） |
| `lags` | int | Number of past time steps used as input features / 用作输入特征的历史时间步数 |
| `quantile` | float or None | Coverage level for prediction intervals (e.g., 0.9 for 90%). None = point prediction only / 预测区间覆盖率（如 0.9 表示 90%）。None = 仅点预测 |
| `random_state` | int | Random seed for reproducibility / 随机种子，用于可复现性 |

---

## Neural Network Models / 神经网络模型

All NN models additionally support:
所有 NN 模型还支持以下参数：

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `epochs` | 1000 | Maximum training epochs / 最大训练轮数 |
| `patience` | 100 | Early stopping patience / 早停耐心值 |
| `verbose` | False | Whether to show training progress / 是否显示训练进度 |
| `learning_rate` | 0.001 | Learning rate / 学习率 |
| `use_gtb` | False | Enable GlobalTemporalBlock (FreqMix + Attention + SwiGLU) / 启用全局时序块 |
| `gtb_d_model` | 64 | GTB hidden dimension / GTB 隐藏维度 |
| `routing_mode` | `'static'` | GTB routing: `'static'` (all experts) or `'adaptive'` (MoE top-K) / GTB 路由：`'static'`（全部专家）或 `'adaptive'`（MoE top-K） |

### Modern modular NN family / 现代模块化 NN 模型族

The following registry keys are available through one shared backbone/layer implementation: `timexer`, `time_mixer`, `timesnet`, `pyraformer`, `etsformer`, `lightts`, `patchtst`, `tsmixer`, `nonstationary_transformer`, `fedformer`, `autoformer`, `informer`, `reformer`, `multi_patch_former`, `wpmixer`, `timefilter`, `msgnet`, `seg_rnn`, and `tirex`.
以下注册名通过同一套共享 backbone/layer 实现提供：`timexer`、`time_mixer`、`timesnet`、`pyraformer`、`etsformer`、`lightts`、`patchtst`、`tsmixer`、`nonstationary_transformer`、`fedformer`、`autoformer`、`informer`、`reformer`、`multi_patch_former`、`wpmixer`、`timefilter`、`msgnet`、`seg_rnn` 和 `tirex`。

Public wrapper classes such as `PatchTSTModel`, `TimesNetModel`, and `TimeMixerModel` are generated from a central model spec registry, reducing duplicated training boilerplate while keeping the same `fit()` / `predict()` interface.
公开包装类（如 `PatchTSTModel`、`TimesNetModel`、`TimeMixerModel`）由统一模型规格注册表生成，减少重复训练模板代码，同时保持相同的 `fit()` / `predict()` 接口。

### NLinearModel

Simple linear mapping model. The fastest NN model, suitable as a baseline.
简单线性映射模型。最快的神经网络模型，适合作为基准。

```python
from PipelineTS.nn_model import NLinearModel

model = NLinearModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, epochs=50, patience=10, verbose=False
)
model.fit(data)
result = model.predict(10)
```

### DLinearModel

Decomposition linear model that separates trend and seasonal components.
分解线性模型，将序列分解为趋势和季节性分量。

```python
from PipelineTS.nn_model import DLinearModel

model = DLinearModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, epochs=50, patience=10, verbose=False
)
```

### NBeatsModel

N-BEATS architecture supporting both generic and interpretable modes.
N-BEATS 架构，支持通用和可解释两种模式。

```python
from PipelineTS.nn_model import NBeatsModel

model = NBeatsModel(
    time_col='date', target_col='value', lags=12,
    generic_architecture=True,  # True: generic, False: interpretable
                                # True: 通用模式, False: 可解释模式
    num_stacks=2, num_blocks=1, num_layers=2, layer_widths=64,
    quantile=0.9, epochs=50, verbose=False
)
```

### NHitsModel

N-HiTS model with hierarchical interpolation for efficient multi-step forecasting.
N-HiTS 模型，使用分层插值结构提高多步预测效率。

```python
from PipelineTS.nn_model import NHitsModel

model = NHitsModel(
    time_col='date', target_col='value', lags=12,
    num_stacks=2, num_blocks=1, num_layers=2, layer_widths=64,
    quantile=0.9, epochs=50, verbose=False
)
```

### TFTModel

Temporal Fusion Transformer combining LSTM and multi-head attention.
时序融合 Transformer，结合 LSTM 和多头注意力机制。

```python
from PipelineTS.nn_model import TFTModel

model = TFTModel(
    time_col='date', target_col='value', lags=12,
    hidden_size=32, lstm_layers=1, n_heads=2,
    quantile=0.9, epochs=50, verbose=False
)
```

### TransformerModel

Classic Transformer encoder architecture for time series.
经典 Transformer 编码器架构。

```python
from PipelineTS.nn_model import TransformerModel

model = TransformerModel(
    time_col='date', target_col='value', lags=12,
    d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64,
    quantile=0.9, epochs=50, verbose=False
)
```

### TiDEModel

Time-series Dense Encoder with fully-connected encoder-decoder structure.
时序密集编码器，基于全连接的编解码器结构。

```python
from PipelineTS.nn_model import TiDEModel

model = TiDEModel(
    time_col='date', target_col='value', lags=12,
    num_encoder_layers=2, num_decoder_layers=2,
    hidden_size=64, decoder_output_dim=16,
    quantile=0.9, epochs=50, verbose=False
)
```

### GAUModel

Gated Attention Unit model with gated attention mechanism.
门控注意力单元模型，使用门控注意力机制。

```python
from PipelineTS.nn_model import GAUModel

model = GAUModel(
    time_col='date', target_col='value', lags=12,
    level=3,
    quantile=0.9, epochs=50, verbose=False
)
```

### StackingRNNModel

RWKV (linear RNN) encoder with gated residual blocks and RevIN normalization. Uses parallel linear temporal mixing (no sequential recurrence), followed by gated residual refinement with SiLU activation, plus a direct residual shortcut.
RWKV（线性 RNN）编码器 + 门控残差块 + RevIN 归一化。使用并行线性时序混合（无顺序递归），经过带 SiLU 激活的门控残差精炼，加上直接残差快捷连接。

```python
from PipelineTS.nn_model import StackingRNNModel

model = StackingRNNModel(
    time_col='date', target_col='value', lags=12,
    blocks=3,           # Number of gated residual blocks / 门控残差块数量
    d_model=48,         # Hidden dimension / 隐藏层维度
    quantile=0.9, epochs=50, verbose=False
)
```

### Time2VecModel

Trend-seasonal decomposition combined with StableTime2Vec periodic encoding and RWKV temporal mixing. The input is decomposed via moving average into trend and seasonal components; the trend path uses a linear projection while the seasonal path applies log-spaced Time2Vec periodic features followed by RWKV encoder blocks. Includes RevIN normalization and direct residual shortcut.
趋势-季节分解 + StableTime2Vec 周期编码 + RWKV 时序混合。输入通过移动平均分解为趋势和季节分量；趋势路径使用线性投影，季节路径使用对数间距的 Time2Vec 周期特征 + RWKV 编码器。包含 RevIN 归一化和直接残差快捷连接。

```python
from PipelineTS.nn_model import Time2VecModel

model = Time2VecModel(
    time_col='date', target_col='value', lags=12,
    num_layers=2,       # Number of RWKV blocks / RWKV 块数量
    quantile=0.9, epochs=50, verbose=False
)
```

### PatchRNNModel

Patch-based RNN that segments the input sequence into patches before feeding to LSTM.
基于 Patch 的 RNN，将输入序列分块后输入 LSTM。

```python
from PipelineTS.nn_model import PatchRNNModel

model = PatchRNNModel(
    time_col='date', target_col='value', lags=12,
    kernel_size=4,
    quantile=0.9, epochs=50, verbose=False
)
```

### TCNModel

Temporal Convolutional Network with dilated causal convolutions.
时序卷积网络，使用膨胀因果卷积。

```python
from PipelineTS.nn_model import TCNModel

model = TCNModel(
    time_col='date', target_col='value', lags=12,
    kernel_size=3,
    quantile=0.9, epochs=50, verbose=False
)
```

### ITransformerModel

Inverted Transformer that treats each variable as a token. Supports multivariate prediction.
反转 Transformer，将每个变量视为一个 token。支持多变量预测。

```python
from PipelineTS.nn_model import ITransformerModel

model = ITransformerModel(
    time_col='date', target_col='value', lags=12,
    d_model=32, n_heads=2, d_ff=64, e_layers=1,
    feature_cols=None,  # Set for multivariate mode / 设置为多变量模式
    quantile=0.9, epochs=50, verbose=False
)
```

### SRSNetModel

Selective Representation Space Network with multi-scale adaptive patches. Supports multivariate prediction.
选择性表征空间网络，使用多尺度自适应 patch。支持多变量预测。

```python
from PipelineTS.nn_model import SRSNetModel

model = SRSNetModel(
    time_col='date', target_col='value', lags=12,
    d_model=32, n_heads=2,
    feature_cols=None,  # Set for multivariate mode / 设置为多变量模式
    quantile=0.9, epochs=50, verbose=False
)
```

### DeepARModel

Probabilistic time series forecasting with autoregressive recurrent networks. Uses a modern RWKV (linear RNN) encoder instead of traditional LSTM, combined with a Gaussian probabilistic output head. During training, the model learns distribution parameters (μ, σ) via Gaussian NLL loss; at inference, point predictions use the learned mean.
概率时间序列预测模型，使用自回归循环网络。采用现代 RWKV（线性 RNN）编码器替代传统 LSTM，结合高斯概率输出头。训练时通过高斯负对数似然损失学习分布参数（μ, σ）；推理时使用学习到的均值作为点预测。

**Architecture / 架构:**

```
Input → RevIN → Per-timestep Embedding → RWKV Encoder → Attention-weighted Pooling
→ Gated Residual Blocks → Gaussian Head (μ, σ) → RevIN Denormalize
```

- **RWKV Encoder**: Stacked RWKVBlocks (GatedTimeMixing + SiLU-gated ChannelMixing), all nn.Linear ops, O(T) parallel temporal mixing, no sequential recurrence
- **RWKV 编码器**：堆叠的 RWKVBlock（门控时序混合 + SiLU 门控通道混合），全部 nn.Linear 操作，O(T) 并行时序混合，无顺序递归
- **Gated Residual Blocks**: LayerNorm → sigmoid(gate) * SiLU(up) → dropout → residual
- **门控残差块**：LayerNorm → sigmoid(gate) * SiLU(up) → dropout → 残差连接
- **Gaussian Head**: Shared trunk with separate μ/σ heads; σ guaranteed positive via softplus
- **高斯输出头**：共享主干 + 独立的 μ/σ 头；σ 通过 softplus 保证正值
- **RevIN**: Instance normalization for non-stationary time series
- **RevIN**：实例归一化，处理非平稳时间序列
- **Direct residual shortcut**: Linear projection from input to output for gradient flow
- **直接残差快捷连接**：从输入到输出的线性投影，改善梯度流

**Model-specific parameters / 模型特有参数:**

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `d_model` | 64 | Hidden dimension for RWKV encoder and gated residual blocks / RWKV 编码器和门控残差块的隐藏维度 |
| `n_blocks` | 3 | Number of gated residual refinement blocks / 门控残差精炼块数量 |
| `n_rwkv_blocks` | 3 | Number of RWKV temporal mixing blocks in the encoder / 编码器中 RWKV 时序混合块的数量 |
| `dropout` | 0.1 | Dropout rate / Dropout 比率 |

```python
from PipelineTS.nn_model import DeepARModel

model = DeepARModel(
    time_col='date', target_col='value', lags=12,
    d_model=64,          # Hidden dimension / 隐藏维度
    n_blocks=3,          # Gated residual blocks / 门控残差块数量
    n_rwkv_blocks=3,     # RWKV encoder blocks / RWKV 编码器块数量
    dropout=0.1,         # Dropout rate / Dropout 比率
    quantile=0.9, epochs=50, verbose=False
)
model.fit(data)
result = model.predict(10)
```

**Key differences from the original DeepAR paper / 与原始 DeepAR 论文的主要区别:**

- Replaces LSTM with RWKV linear RNN encoder — fully parallelizable, O(T) complexity
- 用 RWKV 线性 RNN 编码器替代 LSTM — 完全可并行化，O(T) 复杂度
- Attention-weighted pooling instead of last hidden state — learns which timesteps matter most
- 注意力加权池化替代最后隐藏状态 — 学习哪些时间步最重要
- Gated residual refinement blocks for richer feature transformation
- 门控残差精炼块实现更丰富的特征变换
- Supports CQR (Conformalized Quantile Regression) for adaptive prediction intervals
- 支持 CQR（保形分位数回归）生成自适应预测区间

**Reference / 参考:**
Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks", International Journal of Forecasting, 2020.

---

## GPU-Accelerated Tree Models / GPU 加速树模型

PipelineTS features fully **GPU-accelerated differentiable tree ensembles** built in PyTorch. All tree parameters are stored as batched tensors and the forward pass uses `torch.einsum` — **zero Python loops** over individual trees. GPU acceleration is automatic: CUDA > MPS > CPU fallback.

PipelineTS 采用基于 PyTorch 的**全 GPU 加速可微分树集成**。所有树参数以批量张量存储，前向传播使用 `torch.einsum` —— 对单棵树**零 Python 循环**。GPU 加速为自动模式：CUDA > MPS > CPU 降级。

All tree models automatically build rich lag features (26+ features per window) including statistics, trends, and autocorrelation.
所有树模型自动构建丰富的滞后特征（每个窗口 26+ 个特征），包括统计量、趋势和自相关。

**Architecture highlights / 架构亮点:**

- **Oblivious decision trees**: All trees share the same split structure, enabling efficient batched computation.
- **斜向决策树**：所有树共享相同的分裂结构，实现高效批量计算。
- **Linear skip connection**: `output = trees(x) + linear(x)` — trees learn the non-linear residual on top of a strong linear baseline.
- **线性跳跃连接**：`output = trees(x) + linear(x)` —— 树在强线性基线之上学习非线性残差。
- **Feature temperature annealing**: Softmax temperature on feature logits annealed 1.0→0.1 during training, producing increasingly tree-like hard feature selection.
- **特征温度退火**：特征 logits 的 softmax 温度在训练中从 1.0 退火到 0.1，产生越来越像树的硬特征选择。
- **Structural-break adaptation**: YDF-inspired exponential recency sample weighting for regime changes.
- **结构性断点适应**：受 YDF 启发的指数近因样本加权，用于处理分布变化。

**GPU optimization features / GPU 优化特性:**

| Feature / 特性 | Description / 描述 |
|---|---|
| **AMP (Mixed Precision)** | `torch.amp.autocast` + `GradScaler` for CUDA, auto-enabled when n≥128 / CUDA 混合精度，n≥128 时自动启用 |
| **torch.compile** | PyTorch 2.0+ `reduce-overhead` mode for CUDA when n≥256 / PyTorch 2.0+ 编译加速 |
| **pin_memory** | Efficient CPU→GPU data transfers via `pin_memory()` + `non_blocking=True` / 高效 CPU→GPU 数据传输 |
| **inference_mode** | `torch.inference_mode()` for faster prediction / 更快的推理模式 |

### TorchBoostingForestModel

GPU-accelerated gradient boosting forest with staged residual learning (MART/DART). Uses differentiable oblivious trees trained end-to-end. Each boosting stage trains on the residual error from all previous stages, with a GrowNet-style corrective step for joint fine-tuning.

GPU 加速梯度提升森林，具有分阶段残差学习（MART/DART）。使用端到端训练的可微分斜向树。每个提升阶段在前序所有阶段的残差误差上训练，并使用 GrowNet 风格的修正步骤进行联合微调。

**Model-specific parameters / 模型特有参数:**

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `n_trees` | 64 | Number of trees per boosting stage / 每个提升阶段的树数量 |
| `tree_depth` | 5 | Depth of each oblivious decision tree / 每棵斜向决策树的深度 |
| `learning_rate` | 0.08 | Learning rate for AdamW optimizer / AdamW 优化器的学习率 |
| `n_epochs` | 200 | Maximum training epochs per stage / 每阶段最大训练轮数 |
| `batch_size` | 0 | Batch size (0 = full batch) / 批大小（0 = 全批量） |
| `early_stop_patience` | 15 | Early stopping patience / 早停耐心值 |
| `dropout` | 0.0 | Tree-level dropout / 树级别的 Dropout |
| `weight_decay` | 1e-4 | L2 regularization / L2 正则化 |
| `boosting_stages` | 3 | Number of sequential residual boosting stages / 顺序残差提升阶段数 |
| `boosting_shrinkage` | 0.5 | Shrinkage per stage / 每阶段收缩率 |
| `accelerator` | None | `'cuda'`, `'mps'`, `'cpu'`, or None (auto-detect) / 加速器 |
| `auto_complexity` | False | Enable adaptive complexity auto-tuning / 启用自适应复杂度自动调优 |
| `verbose` | False | Show training progress / 显示训练进度 |

```python
from PipelineTS.ml_model import TorchBoostingForestModel

model = TorchBoostingForestModel(
    time_col='date', target_col='value', lags=16,
    quantile=0.9,
    accelerator='cuda',       # GPU acceleration / GPU 加速
    n_trees=64,
    tree_depth=5,
    boosting_stages=3,
    boosting_shrinkage=0.5,
    auto_complexity=False,     # Set True for auto-tuning / 设为 True 自动调优
)
model.fit(data)
result = model.predict(10)
```


### TorchBaggingForestModel

GPU-accelerated bagging forest. Each tree votes independently, and tree-level dropout during training decorrelates the ensemble — analogous to random subspace selection in classical bagging methods.

GPU 加速袋装森林。每棵树独立投票，训练期间的树级别 Dropout 去相关集成 —— 类似于经典袋装方法的随机子空间选择。


| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `n_trees` | 128 | Number of trees in the ensemble / 集成中的树数量 |
| `tree_depth` | 5 | Depth of each oblivious decision tree / 每棵树的深度 |
| `dropout` | 0.15 | Tree-level dropout for decorrelation / 去相关的树级 Dropout |
| `n_epochs` | 300 | Maximum training epochs / 最大训练轮数 |
| `auto_complexity` | False | Enable adaptive complexity auto-tuning / 启用自适应复杂度自动调优 |

```python
from PipelineTS.ml_model import TorchBaggingForestModel

model = TorchBaggingForestModel(
    time_col='date', target_col='value', lags=16,
    quantile=0.9,
    accelerator='cuda',
    n_trees=128,
    dropout=0.15,
)
model.fit(data)
result = model.predict(10)
```

### DeepForestModel

GPU-accelerated Deep Forest (gcForest) — multi-layer cascade of differentiable tree ensembles (Zhou & Feng 2017). Each layer's tree outputs are concatenated with the original features and fed to the next layer, all trained end-to-end via backpropagation. Includes a learnable residual scale parameter that provides gradient shortcuts to all layers.

GPU 加速深度森林（gcForest）—— 多层级联可微分树集成（Zhou & Feng 2017）。每层树的输出与原始特征拼接后输入下一层，全部通过反向传播端到端训练。包含可学习的残差缩放参数，为所有层提供梯度快捷通道。

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `n_trees` | 32 | Trees per cascade layer / 每层级联的树数量 |
| `tree_depth` | 4 | Depth of each tree / 每棵树的深度 |
| `n_layers` | 3 | Number of cascade layers / 级联层数 |
| `dropout` | 0.1 | Tree-level dropout / 树级 Dropout |
| `n_epochs` | 200 | Maximum training epochs / 最大训练轮数 |
| `auto_complexity` | False | Enable adaptive complexity auto-tuning / 启用自适应复杂度自动调优 |

```python
from PipelineTS.ml_model import DeepForestModel

model = DeepForestModel(
    time_col='date', target_col='value', lags=16,
    quantile=0.9,
    accelerator='cuda',
    n_trees=32,
    n_layers=3,
    tree_depth=4,
)
model.fit(data)
result = model.predict(10)
```

```python
# Pipeline usage / 管道使用
from PipelineTS.pipeline import ModelPipeline

pipe = ModelPipeline(
    time_col='date', target_col='value', lags=16,
    include_models=['deep_forest'],
    deep_forest__n_layers=4,
    deep_forest__n_trees=48,
)
pipe.fit(data)
```

### Adaptive Complexity Auto-Tuning / 自适应复杂度自动调优

All three GPU tree models support `auto_complexity=True`, which dynamically selects optimal `tree_depth` and `n_trees` based on data characteristics. An `_AdaptiveComplexityController` analyzes the training data and selects one of five complexity profiles:

三个 GPU 树模型均支持 `auto_complexity=True`，可根据数据特征动态选择最优的 `tree_depth` 和 `n_trees`。`_AdaptiveComplexityController` 分析训练数据并选择五个复杂度配置之一：

| Profile / 配置 | Depth / 深度 | Trees / 树数 | When / 适用场景 |
|---|---|---|---|
| `minimal` | 2–3 | 8–24 | Tiny data (n < 60) / 极小数据 |
| `light` | 3–4 | 16–48 | Small data (n < 150) / 小数据 |
| `moderate` | 4–5 | 32–64 | Medium data (n < 400) / 中等数据 |
| `heavy` | 5–6 | 48–96 | Large data (n < 1000) / 大数据 |
| `maximal` | 6–7 | 64–128 | Very large data (n ≥ 1000) / 超大数据 |

The controller considers: **data size**, **noise ratio** (linear residual variance), **nonlinearity** (smooth vs linear fit), **autocorrelation** (lag-1), and **ensemble mode** (cascade gets lighter per-layer trees).

控制器综合考虑：**数据规模**、**噪声比率**（线性残差方差）、**非线性度**（平滑 vs 线性拟合）、**自相关**（滞后-1）和**集成模式**（级联模式每层使用更轻量的树）。

```python
from PipelineTS.ml_model import TorchBoostingForestModel

model = TorchBoostingForestModel(
    time_col='date', target_col='value', lags=16,
    auto_complexity=True,     # Enable auto-tuning / 启用自动调优
    verbose=True,             # Print complexity selection reasons / 打印复杂度选择原因
)
model.fit(data)

# Inspect the auto-selected complexity / 查看自动选择的复杂度
info = model.model.complexity_info
print(f"Profile: {info['profile']}")          # e.g., 'moderate'
print(f"Auto depth: {info['tree_depth']}")    # e.g., 5
print(f"Auto trees: {info['n_trees']}")       # e.g., 64
print(f"Reasons: {info['reasons']}")          # e.g., ['medium_data(n=350)->moderate', ...]
```

---

## Other ML Models / 其他 ML 模型

### WideGBRTModel

Wide-table GBRT with automatically constructed rich time series features (~40 features). Supports differencing.
宽表 GBRT 模型，自动构建丰富的时序特征（约 40 个特征）。支持差分操作。

```python
from PipelineTS.ml_model import WideGBRTModel

model = WideGBRTModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, n_estimators=200, verbose=-1,
    differential_n=1,  # Order of differencing / 差分阶数
)
```

### MultiOutputRegressorModel / MultiStepRegressorModel / RegressorChainModel

Multi-output regression wrappers for multi-step forecasting.
多输出回归封装器，用于多步预测。

```python
from PipelineTS.ml_model import (
    MultiOutputRegressorModel,
    MultiStepRegressorModel,
    RegressorChainModel
)

# All share the same interface / 都共享相同接口
model = MultiOutputRegressorModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9, verbose=-1
)
model.fit(data)
result = model.predict(10)
```

You can specify a custom estimator:
可以指定自定义估计器：

```python
from sklearn.ensemble import GradientBoostingRegressor

model = MultiOutputRegressorModel(
    time_col='date', target_col='value', lags=12,
    estimator=GradientBoostingRegressor, verbose=0
)
```

---

## Statistical Models / 统计模型

### ProphetModel

Custom Prophet-like decomposable time series model (not Facebook Prophet). Uses piecewise linear trend with automatic changepoint detection, Fourier-based seasonality with FFT auto-detection, and optional causal rolling lag features (7 features: rolling mean, std, trend slope, momentum, half-ratio, EMA, autocorrelation). All parameters solved via ridge regression (closed-form), making it 100x+ faster than Facebook Prophet.
自定义类 Prophet 可分解时序模型（非 Facebook Prophet）。使用分段线性趋势 + 自动变点检测、基于傅里叶的季节性 + FFT 自动检测，以及可选的因果滚动滞后特征（7 个特征：滚动均值、标准差、趋势斜率、动量、半比率、EMA、自相关）。所有参数通过岭回归（解析解）求解，比 Facebook Prophet 快 100 倍以上。

```python
from PipelineTS.statistic_model import ProphetModel

model = ProphetModel(
    time_col='date', target_col='value', lags=12,
    quantile=0.9,
    auto_seasonality=True,       # Auto-detect seasonality via FFT / 通过 FFT 自动检测季节性
    use_lag_features=True,       # Enable causal rolling lag features / 启用因果滚动滞后特征
    lag_window='auto',           # Auto-determine window size / 自动确定窗口大小
    changepoint_prior_scale=0.05, # Smaller = smoother trend / 越小趋势越平滑
)
model.fit(data, cv=2)
result = model.predict(10)
```

### AutoARIMAModel

Automatic ARIMA parameter search.
自动搜索最佳 ARIMA 参数。

```python
from PipelineTS.statistic_model import AutoARIMAModel

model = AutoARIMAModel(
    time_col='date', target_col='value', lags=12,
    start_p=0, max_p=3, start_q=0, max_q=3,
    seasonal=False, quantile=0.9
)
model.fit(data, cv=2)
result = model.predict(10)
```

---

## Foundation Models / 基础模型 (optional / 可选)

Foundation models are large pretrained models that perform **zero-shot forecasting** — no training on your data is needed.
基础模型是大型预训练模型，执行**零样本预测** —— 无需在您的数据上训练。

> Optional dependencies: `pip install chronos-forecasting` for Chronos-2, `pip install tirex-ts` for TiRex foundation, and `pip install transformers==4.40.1` for Sundial / Time-MoE.
> 可选依赖：Chronos-2 使用 `pip install chronos-forecasting`，TiRex foundation 使用 `pip install tirex-ts`，Sundial / Time-MoE 使用 `pip install transformers==4.40.1`。

### Foundation family / 基础模型家族

The foundation adapters wrap Chronos-2, TiRex, Sundial, and Time-MoE checkpoints through lazy-loaded optional dependencies:
基础模型适配器通过延迟加载的可选依赖封装 Chronos-2、TiRex、Sundial 和 Time-MoE 检查点：

| Class / 类 | Pipeline Key / 管道键名 | HuggingFace Path | Size / 大小 |
|---|---|---|---|
| `Chronos2Model` | `chronos_2` | `amazon/chronos-2` | 120M |
| `Chronos2SynthModel` | `chronos_2_synth` | `autogluon/chronos-2-synth` | 120M |
| `Chronos2SmallModel` | `chronos_2_small` | `autogluon/chronos-2-small` | 28M |
| `TiRexFoundationModel` | `tirex_foundation` | `NX-AI/TiRex` | 35M |
| `SundialModel` | `sundial` | `thuml/sundial-base-128m` | 128M |
| `TimeMoEModel` | `time_moe` | `Maple728/TimeMoE-50M` | 50M |

`ChronosModel` is a backward-compatible alias for `Chronos2Model`.
`ChronosModel` 是 `Chronos2Model` 的向后兼容别名。

**Common parameters / 通用参数:**

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `lags` | 1 | Kept for API compatibility / 保留用于 API 兼容 |
| `quantile` | 0.9 | Conformal interval coverage / 共形区间覆盖率 |
| `device_map` | `'auto'` | Device placement: `'auto'`, `'cpu'`, `'cuda'`, `'mps'` |

```python
from PipelineTS.nn_model import Chronos2Model, Chronos2SynthModel, Chronos2SmallModel
from PipelineTS.nn_model import TiRexFoundationModel, SundialModel, TimeMoEModel

# Standalone usage / 独立使用
model = Chronos2SmallModel(
    time_col='date', target_col='value', quantile=0.9
)
model.fit(data, cv=2)
result = model.predict(10)

# Pipeline usage / 管道使用
from PipelineTS.pipeline import ModelPipeline

pipe = ModelPipeline(
    time_col='date', target_col='value', lags=12,
    include_models=['chronos_2_small', 'tirex_foundation', 'sundial', 'time_moe'],
    quantile=0.9,
)
pipe.fit(data)
pipe.predict(n=10)
```

**Key features / 主要特点:**
- **Zero-shot**: No training needed, uses pretrained weights / 零样本：无需训练，使用预训练权重
- **Covariate support**: Chronos-2 models support known future covariates / 协变量支持：Chronos-2 模型支持已知未来协变量
- **Multi-series**: Supports panel data via `id_col` / 多序列：通过 `id_col` 支持面板数据
- **Conformal intervals**: Prediction intervals via conformal calibration / 共形区间：通过共形校准实现预测区间

---

## Prediction Intervals / 预测区间

All models support prediction intervals via the `quantile` parameter.
所有模型通过 `quantile` 参数支持预测区间。

- **ML and Statistical models**: Use Conformal Prediction with asymmetric intervals.
- **ML 和统计模型**：使用保形预测，生成非对称区间。

- **NN models**: Use Conformalized Quantile Regression (CQR) for adaptive, input-dependent intervals.
- **NN 模型**：使用保形分位数回归（CQR），生成自适应的、依赖输入的区间。

Set `quantile=None` for point predictions only.
设置 `quantile=None` 仅进行点预测。
