# PipelineTS

一站式时间序列分析工具，支持时序数据预处理、特征工程、模型训练、模型评估、模型预测等。

## 安装

```bash
conda install -c conda-forge prophet

python -m pip install PipelineTS
```

## 快速开始

```python
from PipelineTS.dataset import LoadWebSales
init_data = LoadWebSales()[['date', 'type_a']]

valid_data = init_data.iloc[-30:, :]
data = init_data.iloc[:-30, :]

from PipelineTS.pipeline import PipelineTS
# list all models
PipelineTS.list_models()

from sklearn.metrics import mean_absolute_error
pipeline = PipelineTS(
    time_col='date', 
    target_col='type_a', 
    lags=30, 
    random_state=42, 
    metric=mean_absolute_error, 
    metric_less_is_better=True
)

# training all models
pipeline.fit(data, valid_df=valid_data)

# use best model to predict next 30 steps data point
res = pipeline.predict(30)

```

### 数据准备

```python
# TODO
```

### 预处理

```python
# TODO
```

### 特征工程

```python
# TODO
```

### 模型训练

```python
# TODO
```

### 模型评估    

```python
# TODO
```

### 模型预测

```python
# TODO
```

### 模型部署

```python
# TODO
```