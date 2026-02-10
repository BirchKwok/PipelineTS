# Changelog

## v1.1.0 (2025-02-10)

### New Features / 新功能

- **GlobalTemporalBlock (GTB)**: Optional plug-in module for all 12 univariate NN models combining three expert components — FreqMixingBlock (frequency-domain mixing via FFT), GatedLinearAttention (efficient linear attention), and SwiGLU FFN — with residual connections and RevIN normalization.
- **GlobalTemporalBlock (GTB)**：所有 12 个单变量 NN 模型的可选插件模块，组合三个专家组件——FreqMixingBlock（FFT 频域混合）、GatedLinearAttention（高效线性注意力）和 SwiGLU FFN——带残差连接和 RevIN 归一化。

- **MoE Adaptive Routing**: Learned sparse top-K expert selection (inspired by DeepSeek-V2 / Switch Transformer) with load-balancing auxiliary loss. The `ExpertRouter` dynamically activates 2 of 3 GTB experts per sample. Use `routing_mode='adaptive'` to enable.
- **MoE 自适应路由**：学习型稀疏 top-K 专家选择（灵感来自 DeepSeek-V2 / Switch Transformer），带负载均衡辅助损失。`ExpertRouter` 动态激活每个样本 3 个 GTB 专家中的 2 个。使用 `routing_mode='adaptive'` 启用。

- **DeepAR Probabilistic Forecasting**: Added DeepAR model for probabilistic time series forecasting.
- **DeepAR 概率预测**：新增 DeepAR 模型用于概率时间序列预测。

### Parameters / 参数

All 12 univariate NN models now accept:
所有 12 个单变量 NN 模型现在接受：

| Parameter / 参数 | Default / 默认值 | Description / 描述 |
|---|---|---|
| `use_gtb` | `False` | Enable GlobalTemporalBlock / 启用全局时序块 |
| `gtb_d_model` | `64` | GTB hidden dimension / GTB 隐藏维度 |
| `routing_mode` | `'static'` | `'static'` (all experts) or `'adaptive'` (MoE top-K) / 静态或自适应 |

### Benchmark Highlights / 基准测试亮点

On Electric_Production dataset (lags=16, predict=16), adaptive MoE routing achieves notable improvements:
在 Electric_Production 数据集上（lags=16, predict=16），自适应 MoE 路由取得显著提升：

| Model | Baseline MSE | Adaptive MSE | Improvement |
|-------|-------------|-------------|-------------|
| PatchRNN | 31.0 | 22.1 | **-28.6%** |
| StackingRNN | 24.4 | 21.1 | **-13.5%** |
| DLinear | 26.5 | 23.0 | **-13.3%** |
| TiDE | 21.7 | 19.7 | **-9.3%** |

---

## v1.0.0

- Initial release with 24 built-in models (14 NN, 7 ML, 2 statistical, 1 ensemble pipeline)
- Conformal Prediction intervals with CQR for neural networks
- Multivariate forecasting (ITransformer, SRSNet)
- Rich feature engineering (26+ lag features for GBDT/ML, rolling features for Prophet)
- Data preprocessing toolkit (missing values, outliers, stationarity, frequency detection)
- Evaluation framework (backtesting, residual analysis, model comparison)
- Training utilities (AutoTune, WeightedEnsemble, StackingEnsemble)
- Prediction utilities (RollingPredictor, ModelExplainer)
