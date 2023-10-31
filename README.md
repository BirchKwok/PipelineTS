# PipelineTS
[\[中文文档\]](https://github.com/BirchKwok/PipelineTS/blob/main/README_CN.md)

One-stop time series analysis tool, supporting time series data preprocessing, feature engineering, model training, model evaluation, model prediction, etc. Based on spinesTS and darts.
## Installation

```bash
# if you don't want to use the prophet model
# run this
python -m pip install PipelineTS[core]

# if you want to use all models
# run this
python -m pip install PipelineTS[all]
```

## Quick Start [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/QuickStart.ipynb)

### list all available models
```python
from PipelineTS.dataset import LoadWebSales

init_data = LoadWebSales()[['date', 'type_a']]

valid_data = init_data.iloc[-30:, :]
data = init_data.iloc[:-30, :]
accelerator = 'auto'  # Specify Computing Device

from PipelineTS.pipeline import ModelPipeline

# list all models
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

### Training
```python
from sklearn.metrics import mean_absolute_error

pipeline = ModelPipeline(
    time_col='date',
    target_col='type_a',
    lags=30,
    random_state=42,
    metric=mean_absolute_error,
    metric_less_is_better=True,
    accelerator=accelerator,  # Supported values for accelerator: `auto`, `cpu`, `tpu`, `cuda`, `mps`.
)

# training all models
pipeline.fit(data, valid_data=valid_data)

# use best model to predict next 30 steps data point
res = pipeline.predict(30)

```
## Training and prediction of a single model
###  Without predict specify series [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/modeling.ipynb)

#### Data Preprocessing

```python

from PipelineTS.dataset import LoadMessagesSentDataSets
import pandas as pd
# convert time col, the date column is assumed to be date_col
time_col = 'date'
target_col = 'ta'
lags = 60  # Ahead of the window size, the data will be split into multiple sequences of lags for training
n = 40 # How many steps to predict, in this case how many days to predict

# you can also load data with pandas
# init_data = pd.read_csv('/path/to/your/data.csv')
init_data = LoadMessagesSentDataSets()[[time_col, target_col]]

init_data[time_col] = pd.to_datetime(init_data[time_col], format='%Y-%m-%d')

# split trainning set and test set
valid_data = init_data.iloc[-n:, :]
data = init_data.iloc[:-n, :]
print("data shape: ", data.shape, ", valid data shape: ", valid_data.shape)
data.tail(5)

# data visualization
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

#### Training

```python
from PipelineTS.nn_model import TiDEModel
tide = TiDEModel(
    time_col=time_col, target_col=target_col, lags=lags, random_state=42, 
    quantile=0.9, enable_progress_bar=False, enable_model_summary=False
)
tide.fit(data)
tide.predict(n)
```

### With predict specify series [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/modeling-with-predict-specify-series.ipynb)
```python
tide.predict(n, series=valid_data)
```


## PipelineTS Module

```python
# If you need to configure the model
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from PipelineTS.pipeline import ModelPipeline, PipelineConfigs

# If you want to try multiple configurations of a model at once for comparison or tuning purposes, you can use `PipelineConfigs`.
# This feature allows for customizing the models returned by each `ModelPipeline.list_all_available_models()` call.
# The first one is the name of the model, which needs to be in the list of available models provided by PipelineTS.list_all_available_models(). 
# If you want to customize the name of the model, then the second argument can be a string of the model name, 
# otherwise, the second one is of type dict. The dict can have three keys: 'init_configs', 'fit_configs', 'predict_configs', or any combination of them. 
# The remaining keys will be automatically filled with default parameters.
# Among them, 'init_configs' represents the initialization parameters of the model, 'fit_configs' represents the parameters during model training, 
# and 'predict_configs' represents the parameters during model prediction.

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

### Non-Interval Forecasting [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/pipeline.ipynb)

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
    # include_models=['d_linear', 'random_forest', 'n_linear', 'n_beats'],  # specifying the model used
    # exclude_models=['catboost', 'tcn', 'transformer'],  # exclude specified models
    # Note that `include_models` and `exclude_models` cannot be specified simultaneously.
    accelerator=accelerator,
    # Now we can directly input the "modelname__'init_params'" parameter to instantiate the models in ModelPipeline.
    # Note that it is double underline. 
    # When it is duplicated with the ModelPipeline class keyword parameter, the ModelPipeline clas keyword parameter is ignored
    d_linear__lags=50,
    n_linear__random_state=1024,
    n_beats__num_blocks=3,
    random_forest__n_estimators=200,
    n_hits__accelerator='cpu', # Since using mps backend for n_hits model on mac gives an error, cpu backend is used as an alternative
    tft__accelerator='cpu', # tft, same question, but if you use cuda backend, you can just ignore this two configurations.
)

pipeline.fit(data, valid_data)
```

#### Get the model parameters in PipelineTS
```python
# Gets all configurations for the specified model， default to best model
pipeline.get_model_all_configs(model_name='wide_gbrt')
```

#### Plotting the forecast results
```python
# use best model to predict next 30 steps data point
prediction = pipeline.predict(n, model_name=None)  # You can use `model_name` to specify the pre-trained model in the pipeline when using Python.

plot_data_period(init_data.iloc[-100:, :], prediction, 
                 time_col=time_col, target_col=target_col)
```

![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic2.png)

### Interval prediction [\[notebook\]](https://github.com/BirchKwok/PipelineTS/blob/main/examples/pipeline-with-quantile-prediction.ipynb)

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
    accelerator=accelerator,
    # models=['wide_gbrt']  # Specify the model
    n_hits__accelerator='cpu',
    tft__accelerator='cpu',
)

pipeline.fit(data, valid_data)
```

#### Plotting the forecast results
```python
# use best model to predict next 30 steps data point
prediction = pipeline.predict(n, model_name=None)  # You can use `model_name` to specify the pre-trained model in the pipeline when using Python.

plot_data_period(init_data.iloc[-100:, :], prediction, 
                 time_col=time_col, target_col=target_col)
```
![image1](https://github.com/BirchKwok/PipelineTS/blob/main/pics/pic3.png)


## Model and pipeline saving and loading
```python
from PipelineTS.io import load_model, save_model

# save
save_model(path='/path/to/save/your/fitted_model_or_pipeline.zip', model=pipeline)
# load
pipeline = load_model('/path/to/save/your/fitted_model_or_pipeline.zip')


```
