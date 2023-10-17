# PipelineTS

一站式时间序列分析工具，支持时序数据预处理、特征工程、模型训练、模型评估、模型预测等。

## 安装

```bash
# if you don't want to use the prophet model 如果你不想使用prophet模型
# run this
python -m pip install PipelineTS[core]

# if you want to use all models 如果你想使用所有模型
# run this
python -m pip install PipelineTS[all]
```

## 快速开始

### 查看可用模型
```python
from PipelineTS.dataset import LoadWebSales

init_data = LoadWebSales()[['date', 'type_a']]

valid_data = init_data.iloc[-30:, :]
data = init_data.iloc[:-30, :]
device = 'cpu'

from PipelineTS.pipeline import ModelPipeline

# list all models
ModelPipeline.list_models()
```

```
[output]:
['prophet',
 'auto_arima',
 'catboost',
 'lightgbm',
 'xgboost',
 'wide_gbrt',
 'd_linear',
 'n_linear',
 'n_beats',
 'n_hits',
 'tcn',
 'tft',
 'gau',
 'stacking_rnn',
 'time2vec',
 'multi_output_model',
 'multi_step_model',
 'transformer',
 'random_forest',
 'tide']
```

### 开始训练
```python
from sklearn.metrics import mean_absolute_error

pipeline = ModelPipeline(
    time_col='date',
    target_col='type_a',
    lags=30,
    random_state=42,
    metric=mean_absolute_error,
    metric_less_is_better=True,
    device=device
)

# training all models
pipeline.fit(data, valid_df=valid_data)

# use best model to predict next 30 steps data point
res = pipeline.predict(30)

```

## 数据准备

```python

from PipelineTS.dataset import LoadMessagesSentDataSets
import pandas as pd
# convert time col, the date column is assumed to be date_col
time_col = 'date_col'
target_col = 'ta'
lags = 60  # 往前的窗口大小，数据将会被切割成lags天的多条序列进行训练
n = 40 # 需要预测多少步，在这个例子里为需要预测多少天

# you can also load data with pandas
# init_data = pd.read_csv('/path/to/your/data.csv')
init_data = LoadMessagesSentDataSets()[[time_col, target_col]]

init_data[time_col] = pd.to_datetime(init_data[time_col], format='%Y-%m-%d')

# 划分训练集和测试集
valid_data = init_data.iloc[-n:, :]
data = init_data.iloc[:-n, :]
print("data shape: ", data.shape, ", valid data shape: ", valid_data.shape)
data.tail(5)

# 数据可视化
from PipelineTS.plot import plot_data_period
plot_data_period(
    data.iloc[-300:, :], 
    valid_data, 
    time_col=time_col, 
    target_col=target_col, 
    labels=['Train data', 'Valid_data']
)

```
![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic1.png)

## 单个模型的训练和预测

```python
from PipelineTS.nn_model import TiDEModel
tide = TiDEModel(
    time_col=time_col, target_col=target_col, lags=lags, random_state=42, 
    quantile=0.9, enable_progress_bar=False, enable_model_summary=False
)
tide.fit(data)
tide.predict(n)
```

## PipelineTS 模块

```python
# 如果需要配置模型
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

# 第一个为模型的名称，需要在PipelineTS.list_models()列表中，第二个为dict类型
# dict可以有三个key: 'init_configs', 'fit_configs', 'predict_configs'，也可以任意一个，剩余的会自动补全为默认参数
# 其中init_configs为模型初始化参数，fit_configs为模型训练时参数，predict_configs为模型预测时参数
pipeline_configs = PipelineConfigs([
    ('lightgbm', {'init_configs': {'verbose': -1, 'linear_tree': True}}),
    ('multi_output_model', {'init_configs': {'verbose': -1}}),
    ('multi_step_model', {'init_configs': {'verbose': -1}}),
    ('multi_output_model', {
        'init_configs': {'estimator': XGBRegressor, 'random_state': 42, 'kwargs': {'verbosity': 0}}
    }
     ),
    ('multi_output_model', {
        'init_configs': {'estimator': CatBoostRegressor, 'random_state': 42, 'verbose': False}
    }
     ),
])
```
<table>
<thead>
<tr><th style="text-align: right;">  </th><th>model_name        </th><th>model_name_with_index  </th><th>model_configs                                                                                                                                                    </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;"> 0</td><td>lightgbm          </td><td>lightgbm_1             </td><td>{&#x27;init_configs&#x27;: {&#x27;verbose&#x27;: -1, &#x27;linear_tree&#x27;: True}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}                                                                 </td></tr>
<tr><td style="text-align: right;"> 1</td><td>multi_output_model</td><td>multi_output_model_1   </td><td>{&#x27;init_configs&#x27;: {&#x27;verbose&#x27;: -1}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}                                                                                      </td></tr>
<tr><td style="text-align: right;"> 2</td><td>multi_output_model</td><td>multi_output_model_2   </td><td>{&#x27;init_configs&#x27;: {&#x27;estimator&#x27;: &lt;class &#x27;xgboost.sklearn.XGBRegressor&#x27;&gt;, &#x27;random_state&#x27;: 42, &#x27;kwargs&#x27;: {&#x27;verbosity&#x27;: 0}}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}</td></tr>
<tr><td style="text-align: right;"> 3</td><td>multi_output_model</td><td>multi_output_model_3   </td><td>{&#x27;init_configs&#x27;: {&#x27;estimator&#x27;: &lt;class &#x27;catboost.core.CatBoostRegressor&#x27;&gt;, &#x27;random_state&#x27;: 42, &#x27;verbose&#x27;: False}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}       </td></tr>
<tr><td style="text-align: right;"> 4</td><td>multi_step_model  </td><td>multi_step_model_1     </td><td>{&#x27;init_configs&#x27;: {&#x27;verbose&#x27;: -1}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}                                                                                      </td></tr>
</tbody>
</table>

### 非区间预测

```python
from sklearn.metrics import mean_absolute_error

from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col=time_col,
    target_col=target_col,
    lags=lags,
    random_state=42,
    metric=mean_absolute_error,
    metric_less_is_better=True,
    configs=pipeline_configs,
    include_init_config_model=False,
    use_standard_scale=False,  # False for MinMaxScaler, True for StandardScaler, None means no data be scaled
    with_quantile_prediction=False,
    device=device,
    # models=['wide_gbrt']  # 支持指定模型
)

pipeline.fit(data, valid_data)
```

#### 获取PipelineTS中的模型参数
```python
# Gets all configurations for the specified model， default to best model
pipeline.get_models(model_name='wide_gbrt').all_configs
```

#### 绘制预测结果
```python
# use best model to predict next 30 steps data point
prediction = pipeline.predict(n)  # 可以使用model_name指定pipeline中已训练好的模型

plot_data_period(init_data.iloc[-100:, :], prediction, 
                 time_col=time_col, target_col=target_col)
```

![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic2.png)

### 区间预测

```python
from sklearn.metrics import mean_absolute_error

from PipelineTS.pipeline import ModelPipeline

pipeline = ModelPipeline(
    time_col=time_col,
    target_col=target_col,
    lags=lags,
    random_state=42,
    metric=mean_absolute_error,
    metric_less_is_better=True,
    configs=pipeline_configs,
    include_init_config_model=False,
    use_standard_scale=False,
    with_quantile_prediction=True,  # turn on the quantile prediction switch, if you like
    device=device,
    # models=['wide_gbrt']  # 支持指定模型
)

pipeline.fit(data, valid_data)
```

#### 绘制预测结果
```python
# use best model to predict next 30 steps data point
prediction = pipeline.predict(n, model_name=None)  # 可以使用model_name指定pipeline中已训练好的模型

plot_data_period(init_data.iloc[-100:, :], prediction, 
                 time_col=time_col, target_col=target_col)
```
![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic3.png)
