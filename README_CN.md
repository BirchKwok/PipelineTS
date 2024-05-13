# PipelineTS

![PyPI](https://img.shields.io/pypi/v/PipelineTS)
![PyPI - License](https://img.shields.io/pypi/l/PipelineTS)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/PipelineTS)
[![Downloads](https://pepy.tech/badge/pipelinets)](https://pepy.tech/project/pipelinets)
[![Downloads](https://pepy.tech/badge/pipelinets/month)](https://pepy.tech/project/pipelinets)
[![Downloads](https://pepy.tech/badge/pipelinets/week)](https://pepy.tech/project/pipelinets)

[\[English Documentation\]](https://github.com/BirchKwok/PipelineTS/blob/main/README.md)


一站式时间序列分析工具，支持时间序列数据预处理、特征工程、模型训练、模型评估、模型预测等。基于spinesTS和dart。

## 安装

```bash
# 如果你不想使用prophet模型
# 运行下述语句
python -m pip install PipelineTS[core]

# 如果你想使用所有模型
# 运行下述语句
python -m pip install PipelineTS[all]
```

## 快速开始 [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/QuickStart.ipynb)

### 列出所有可用模型
```python
from PipelineTS.dataset import LoadWebSales

init_data = LoadWebSales()[['date', 'type_a']]

valid_data = init_data.iloc[-30:, :]
data = init_data.iloc[:-30, :]
accelerator = 'auto' # 指定计算设备

from PipelineTS.pipeline import ModelPipeline

# 列出所有模型
ModelPipeline.list_all_available_models()
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
    accelerator=accelerator  # 支持以下值以指定计算设备平台: `auto`, `cpu`, `tpu`, `cuda`, `mps`.
)

# 训练所有模型
pipeline.fit(data, valid_data=valid_data)

# 使用最好的模型预测接下来的30步时间点
res = pipeline.predict(30)

```

## 单个模型的训练和预测
### 不预测指定的序列 [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/modeling.ipynb)

<details>
<summary>Code</summary>

```python

from PipelineTS.dataset import LoadMessagesSentDataSets
import pandas as pd
# 转换time_col, 传入模型的time_col被假定为timestamp格式，强烈建议转为pd.Timestamp格式
time_col = 'date'
target_col = 'ta'
lags = 60  # 往前的窗口大小，数据将会被切割成lags天的多条序列进行训练
n = 40 # 需要预测多少步，在这个例子里为需要预测多少天

# 你一样可以通过pandas加载数据
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

# 训练和预测
from PipelineTS.nn_model import TiDEModel
tide = TiDEModel(
    time_col=time_col, target_col=target_col, lags=lags, random_state=42, 
    quantile=0.9, enable_progress_bar=False, enable_model_summary=False
)
tide.fit(data)
tide.predict(n)

```
</details>

### 预测指定序列 [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/modeling-with-predict-specify-series.ipynb)

<details>
<summary>Code</summary>

```python

from PipelineTS.dataset import LoadMessagesSentDataSets
import pandas as pd

# 转换time_col, 传入模型的time_col被假定为timestamp格式，强烈建议转为pd.Timestamp格式
time_col = 'date'
target_col = 'ta'
lags = 60  # 往前的窗口大小，数据将会被切割成lags天的多条序列进行训练
n = 40  # 需要预测多少步，在这个例子里为需要预测多少天

# 你一样可以通过pandas加载数据
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

# 训练和预测
from PipelineTS.nn_model import TiDEModel

tide = TiDEModel(
    time_col=time_col, target_col=target_col, lags=lags, random_state=42,
    quantile=0.9, enable_progress_bar=False, enable_model_summary=False
)
tide.fit(data)
tide.predict(n, data=valid_data)
```

</details>


## ModelPipeline 模块

```python
# 如果需要配置模型
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

# 如果希望一次性尝试模型的多个配置，以便进行比较或者调参，请使用PipelineConfigs
# 此功能允许自定义每个ModelPipeline.list_all_available_models()的模型，
# 第一个为模型的名称，需要在ModelPipeline.list_all_available_models()列表中，
# 如果你想自定义模型的名称，那第二个参数可以是模型名字字符串，否则，第二个必须为dict类型
# dict可以有三个key: 'init_configs', 'fit_configs', 'predict_configs'，也可以任意一个，剩余的会自动补全为默认参数
# 其中init_configs为模型初始化参数，fit_configs为模型训练时参数，predict_configs为模型预测时参数
pipeline_configs = PipelineConfigs([
    ('lightgbm', 'lightgbm_linear_tree', {'init_configs': {'verbose': -1, 'linear_tree': True}}),
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
<tr><th style="text-align: right;">  </th><th>model_name        </th><th>model_name_after_rename  </th><th>model_configs                                                                                                                                                    </th></tr>
</thead>
<tbody>
<tr><td style="text-align: right;"> 0</td><td>lightgbm          </td><td>lightgbm_linear_tree     </td><td>{&#x27;init_configs&#x27;: {&#x27;verbose&#x27;: -1, &#x27;linear_tree&#x27;: True}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}                                                                 </td></tr>
<tr><td style="text-align: right;"> 1</td><td>multi_output_model</td><td>multi_output_model_1     </td><td>{&#x27;init_configs&#x27;: {&#x27;verbose&#x27;: -1}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}                                                                                      </td></tr>
<tr><td style="text-align: right;"> 2</td><td>multi_output_model</td><td>multi_output_model_2     </td><td>{&#x27;init_configs&#x27;: {&#x27;estimator&#x27;: &lt;class &#x27;xgboost.sklearn.XGBRegressor&#x27;&gt;, &#x27;random_state&#x27;: 42, &#x27;kwargs&#x27;: {&#x27;verbosity&#x27;: 0}}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}</td></tr>
<tr><td style="text-align: right;"> 3</td><td>multi_output_model</td><td>multi_output_model_3     </td><td>{&#x27;init_configs&#x27;: {&#x27;estimator&#x27;: &lt;class &#x27;catboost.core.CatBoostRegressor&#x27;&gt;, &#x27;random_state&#x27;: 42, &#x27;verbose&#x27;: False}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}       </td></tr>
<tr><td style="text-align: right;"> 4</td><td>multi_step_model  </td><td>multi_step_model_1       </td><td>{&#x27;init_configs&#x27;: {&#x27;verbose&#x27;: -1}, &#x27;fit_configs&#x27;: {}, &#x27;predict_configs&#x27;: {}}                                                                                      </td></tr>
</tbody>
</table>

### 非区间预测 [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/pipeline.ipynb)

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
    scaler=False,  # False 为 MinMaxScaler, True 为 StandardScaler, None 表示数据未被缩放
    # include_models=['d_linear', 'random_forest', 'n_linear', 'n_beats'],  # 可以指定使用模型
    # exclude_models=['catboost', 'tcn', 'transformer'],  # 可以指定排除模型，注意，include_models和exclude_models不能同时指定
    accelerator=accelerator,
    # 现在我们可以直接指定"modelname__'init_params'" 参数去初始化ModelPipeline中的模型
    # 注意，模型名字后面要接双下划线. 当"modelname__'init_params'"与ModelPipeline类的默认参数重复时，ModelPipeline类的默认参数将会被忽略
    d_linear__lags=50,
    n_linear__random_state=1024,
    n_beats__num_blocks=3,
    random_forest__n_estimators=200,
    n_hits__accelerator='cpu', # 因为n_hits模型在mac的mps后端上运行会引发报错，因此这里指定为cpu后端
    tft__accelerator='cpu', # tft模型也是同样的问题，但如果你用的是cuda后端，可以直接忽略这两个参数配置
)

pipeline.fit(data, valid_data)
```

#### 获取ModelPipeline中的模型参数
```python
# 获取指定模型的所有配置信息，默认为最佳模型
pipeline.get_model_all_configs(model_name='wide_gbrt')
```

#### 绘制预测结果
```python
# 使用最好的模型预测接下来的n个时间点
prediction = pipeline.predict(n)  # 可以使用model_name指定pipeline中已训练好的模型

plot_data_period(init_data.iloc[-100:, :], prediction, 
                 time_col=time_col, target_col=target_col)
```

![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic2.png)

### 区间预测 [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/pipeline-with-quantile-prediction.ipynb)

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
    scaler=False,
    with_quantile_prediction=True,  # 打开区间预测的开关，训练时间会有所延长
    accelerator=accelerator,
    # models=['wide_gbrt']  # 支持指定模型
    n_hits__accelerator='cpu',
    tft__accelerator='cpu',
)

pipeline.fit(data, valid_data)
```

#### 绘制预测结果
```python
# 使用最好的模型预测接下来的n个时间点
prediction = pipeline.predict(n, model_name=None)  # 可以使用model_name指定pipeline中已训练好的模型

plot_data_period(init_data.iloc[-100:, :], prediction, 
                 time_col=time_col, target_col=target_col)
```
![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic3.png)


## 模型、ModelPipeline的保存和加载
```python
from PipelineTS.io import load_model, save_model

# save
save_model(path='/path/to/save/your/fitted_model_or_pipeline.zip', model=pipeline)
# load
pipeline = load_model('/path/to/save/your/fitted_model_or_pipeline.zip')


```
