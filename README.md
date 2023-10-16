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
device='cpu'

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
    metric_less_is_better=True,
    device=device
)

# training all models
pipeline.fit(data, valid_df=valid_data)

# use best model to predict next 30 steps data point
res = pipeline.predict(30)

```

### 数据准备

```python
# load data with pandas
import pandas as pd
df = pd.read_csv('/path/to/your/data.csv')

# convert time col, the date column is assumed to be date_col
time_col = 'date_col'
target_col = 'target'
lags = 30  # 往前的窗口大小，数据将会被切割成lags天的多条序列进行训练
n = 30 # 需要预测多少步，在这个例子里为需要预测多少天
df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d')

```

### 模型训练和预测

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
from PipelineTS.pipeline import PipelineTS, PipelineConfigs

# list all models
PipelineTS.list_models()

# 第一个为模型的名称，需要在PipelineTS.list_models()列表中，第二个为dict类型
# dict可以有三个key: 'init_configs', 'fit_configs', 'predict_configs'，也可以任意一个，剩余的会自动补全为默认参数
# 其中init_configs为模型初始化参数，fit_configs为模型训练时参数，predict_configs为模型预测时参数
pipeline_configs = PipelineConfigs([
    ('lightgbm', {'init_configs': {'verbose': -1, 'linear_tree': True}}),
    ('multi_output_model', {'init_configs': {'verbose': -1}}),
    ('multi_step_model', {'init_configs': {'verbose': -1}}),
    ('multi_output_model', {
        'init_configs': {'estimator': XGBRegressor, 'random_state': 42, 'kwargs':{'verbosity':0}}
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

```python
from sklearn.metrics import mean_absolute_error

from PipelineTS.pipeline import PipelineTS

pipeline2 = PipelineTS(
    time_col=time_col, 
    target_col=target_col, 
    lags=lags, 
    random_state=42, 
    metric=mean_absolute_error, 
    metric_less_is_better=True,
    configs=pipeline_configs,
    include_init_config_model=False,
    use_standard_scale=False,
    with_quantile_prediction=True,   # turn on the quantile prediction switch, if you like
    device=device,
    # models=['wide_gbrt']  # 支持指定模型
)
```