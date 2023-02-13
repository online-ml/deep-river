# Bike-sharing forecasting

In this tutorial we're going to forecast the number of bikes in 5 bike stations from the city of Toulouse. We'll do so by building a simple model step by step. The dataset contains 182,470 observations. Let's first take a peak at the data.


```python
from pprint import pprint
from river import datasets

dataset = datasets.Bikes()

for x, y in dataset:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break
```

    {'clouds': 75,
     'description': 'light rain',
     'humidity': 81,
     'moment': datetime.datetime(2016, 4, 1, 0, 0, 7),
     'pressure': 1017.0,
     'station': 'metro-canal-du-midi',
     'temperature': 6.54,
     'wind': 9.3}
    Number of available bikes: 1


Let's start by using a simple linear regression on the numeric features. We can select the numeric features and discard the rest of the features using a `Select`. Linear regression is very likely to go haywire if we don't scale the data, so we'll use a `StandardScaler` to do just that. We'll evaluate the model by measuring the mean absolute error. Finally we'll print the score every 20,000 observations. 


```python
from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import optim

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))

metric = metrics.MAE()

evaluate.progressive_val_score(dataset.take(50000), model, metric, print_every=5_000)
```

    [5,000] MAE: 4.258099
    [10,000] MAE: 4.495612
    [15,000] MAE: 4.752074
    [20,000] MAE: 4.912727
    [25,000] MAE: 4.934188
    [30,000] MAE: 5.164331
    [35,000] MAE: 5.320877
    [40,000] MAE: 5.333554
    [45,000] MAE: 5.354958
    [50,000] MAE: 5.378699





    MAE: 5.378699



The model doesn't seem to be doing that well, but then again we didn't provide a lot of features. Generally, a good idea for this kind of problem is to look at an average of the previous values. For example, for each station we can look at the average number of bikes per hour. To do so we first have to extract the hour from the  `moment` field. We can then use a `TargetAgg` to aggregate the values of the target.


```python
from river import feature_extraction
from river import stats

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(1e-2))

metric = metrics.MAE()

evaluate.progressive_val_score(dataset.take(50000), model, metric, print_every=5_000)
```

    [5,000] MAE: 69.042914
    [10,000] MAE: 36.269638
    [15,000] MAE: 25.241059
    [20,000] MAE: 19.781737
    [25,000] MAE: 16.605912
    [30,000] MAE: 14.402878
    [35,000] MAE: 12.857216
    [40,000] MAE: 11.647737
    [45,000] MAE: 10.646566
    [50,000] MAE: 9.94726





    MAE: 9.94726



By adding a single feature, we've managed to significantly reduce the mean absolute error. At this point you might think that the model is getting slightly complex, and is difficult to understand and test. Pipelines have the advantage of being terse, but they aren't always to debug. Thankfully `river` has some ways to relieve the pain.

The first thing we can do it to visualize the pipeline, to get an idea of how the data flows through it.


```python
model
```




<div><div class="river-component river-pipeline"><div class="river-component river-union"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">['clouds', 'humidity', 'pressure', 'temperature', 'wind']</pre></summary><code class="river-estimator-params">(
  clouds
  humidity
  pressure
  temperature
  wind
)

</code></details><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">get_hour</pre></summary><code class="river-estimator-params">
def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">y_mean_by_station_and_hour</pre></summary><code class="river-estimator-params">(
  by=['station', 'hour']
  how=Mean ()
  target_name="y"
)

</code></details></div></div><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">LinearRegression</pre></summary><code class="river-estimator-params">(
  optimizer=SGD (
    lr=Constant (
      learning_rate=0.01
    )
  )
  loss=Squared ()
  l2=0.
  l1=0.
  intercept_init=0.
  intercept_lr=Constant (
    learning_rate=0.01
  )
  clip_gradient=1e+12
  initializer=Zeros ()
)

</code></details></div><style scoped>
.river-estimator {
    padding: 1em;
    border-style: solid;
    background: white;
}

.river-pipeline {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: linear-gradient(#000, #000) no-repeat center / 3px 100%;
}

.river-union {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white;
}

.river-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1em;
    border-style: solid;
    background: white;
}

.river-wrapper > .river-estimator {
    margin-top: 1em;
}

/* Vertical spacing between steps */

.river-component + .river-component {
    margin-top: 2em;
}

.river-union > .river-estimator {
    margin-top: 0;
}

.river-union > .pipeline {
    margin-top: 0;
}

/* Spacing within a union of estimators */

.river-union > .river-component + .river-component {
    margin-left: 1em;
}

/* Typography */

.river-estimator-params {
    display: block;
    white-space: pre-wrap;
    font-size: 120%;
    margin-bottom: -1em;
}

.river-estimator > .river-estimator-params,
.river-wrapper > .river-details > river-estimator-params {
    background-color: white !important;
}

.river-estimator-name {
    display: inline;
    margin: 0;
    font-size: 130%;
}

/* Toggle */

.river-summary {
    display: flex;
    align-items:center;
    cursor: pointer;
}

.river-summary > div {
    width: 100%;
}
</style></div>



The `debug_one` method shows what happens to an input set of features, step by step.

And now comes the catch. Up until now we've been using the `progressive_val_score` method from the `evaluate` module. What this does is that it sequentially predicts the output of an observation and updates the model immediately afterwards. This way of proceeding is often used for evaluating online learning models. But in some cases it is the wrong approach.

When evaluating a machine learning model, the goal is to simulate production conditions in order to get a trust-worthy assessment of the performance of the model. In our case, we typically want to forecast the number of bikes available in a station, say, 30 minutes ahead. Then, once the 30 minutes have passed, the true number of available bikes will be available and we will be able to update the model using the features available 30 minutes ago.

What we really want is to evaluate the model by forecasting 30 minutes ahead and only updating the model once the true values are available. This can be done using the `moment` and `delay` parameters in the  `progressive_val_score` method. The idea is that each observation in the stream of the data is shown twice to the model: once for making a prediction, and once for updating the model when the true value is revealed. The `moment` parameter determines which variable should be used as a timestamp, while the `delay` parameter controls the duration to wait before revealing the true values to the model.


```python
import datetime as dt

evaluate.progressive_val_score(
    dataset=dataset.take(50000),
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=5_000
)
```

    [5,000] MAE: 4.675207
    [10,000] MAE: 4.352476
    [15,000] MAE: 4.193511
    [20,000] MAE: 4.203433
    [25,000] MAE: 4.226929
    [30,000] MAE: 4.191629
    [35,000] MAE: 4.227425
    [40,000] MAE: 4.195404
    [45,000] MAE: 4.102599
    [50,000] MAE: 4.117846





    MAE: 4.117846



The performance is a bit worse, which is to be expected. Indeed, the task is more difficult: the model is only shown the ground truth 30 minutes after making a prediction.

# Goin' Deep
## Rebuilding Linear Regression in PyTorch


```python
from deep_river.regression import Regressor
from river import feature_extraction
from river import stats
import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, n_features, outputSize=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= Regressor(
    module=LinearRegression,
    loss_fn='mse',
    optimizer_fn='sgd',
    lr=1e-2,
)
```


```python
import datetime as dt

evaluate.progressive_val_score(
    dataset=dataset.take(50000),
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=5_000
)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[13], line 3
          1 import datetime as dt
    ----> 3 evaluate.progressive_val_score(
          4     dataset=dataset.take(50000),
          5     model=model.clone(),
          6     metric=metrics.MAE(),
          7     moment='moment',
          8     delay=dt.timedelta(minutes=30),
          9     print_every=5_000
         10 )


    File ~/Documents/environments/deep-river39/lib/python3.9/site-packages/river/evaluate/progressive_validation.py:341, in progressive_val_score(dataset, model, metric, moment, delay, print_every, show_time, show_memory, **print_kwargs)
        190 """Evaluates the performance of a model on a streaming dataset.
        191 
        192 This method is the canonical way to evaluate a model's performance. When used correctly, it
       (...)
        327 
        328 """
        330 checkpoints = iter_progressive_val_score(
        331     dataset=dataset,
        332     model=model,
       (...)
        338     measure_memory=show_memory,
        339 )
    --> 341 for checkpoint in checkpoints:
        343     msg = f"[{checkpoint['Step']:,d}] {metric}"
        344     if show_time:


    File ~/Documents/environments/deep-river39/lib/python3.9/site-packages/river/evaluate/progressive_validation.py:167, in iter_progressive_val_score(dataset, model, metric, moment, delay, step, measure_time, measure_memory)
         80 def iter_progressive_val_score(
         81     dataset: base.typing.Dataset,
         82     model,
       (...)
         88     measure_memory=False,
         89 ) -> typing.Generator:
         90     """Evaluates the performance of a model on a streaming dataset and yields results.
         91 
         92     This does exactly the same as `evaluate.progressive_val_score`. The only difference is that
       (...)
        164 
        165     """
    --> 167     yield from _progressive_validation(
        168         dataset,
        169         model,
        170         metric,
        171         checkpoints=itertools.count(step, step) if step else iter([]),
        172         moment=moment,
        173         delay=delay,
        174         measure_time=measure_time,
        175         measure_memory=measure_memory,
        176     )


    File ~/Documents/environments/deep-river39/lib/python3.9/site-packages/river/evaluate/progressive_validation.py:47, in _progressive_validation(dataset, model, metric, checkpoints, moment, delay, measure_time, measure_memory)
         45 # Case 1: no ground truth, just make a prediction
         46 if y is None:
    ---> 47     preds[i] = pred_func(x=x, **kwargs)
         48     continue
         50 # Case 2: there's a ground truth, model and metric can be updated


    File ~/Documents/environments/deep-river39/lib/python3.9/site-packages/river/compose/pipeline.py:594, in Pipeline.predict_one(self, x, **params)
        585 """Call `transform_one` on the first steps and `predict_one` on the last step.
        586 
        587 Parameters
       (...)
        591 
        592 """
        593 x, last_step = self._transform_one(x)
    --> 594 return last_step.predict_one(x, **params)


    File ~/Documents/projects/IncrementalLearning/deep-river/deep_river/regression/regressor.py:177, in Regressor.predict_one(self, x)
        175 if not self.module_initialized:
        176     self.kwargs["n_features"] = len(x)
    --> 177     self.initialize_module(**self.kwargs)
        178 x_t = dict2tensor(x, self.device)
        179 self.module.eval()


    File ~/Documents/projects/IncrementalLearning/deep-river/deep_river/base.py:142, in DeepEstimator.initialize_module(self, **kwargs)
        127 """
        128 Parameters
        129 ----------
       (...)
        139   The initialized component.
        140 """
        141 if not isinstance(self.module_cls, torch.nn.Module):
    --> 142     self.module = self.module_cls(
        143         **self._filter_kwargs(self.module_cls, kwargs)
        144     )
        146 self.module.to(self.device)
        147 self.optimizer = self.optimizer_fn(
        148     self.module.parameters(), lr=self.lr
        149 )


    TypeError: 'NoneType' object is not callable


## Building RNN Models


```python
from deep_river.regression import Regressor, RollingRegressor
from river import feature_extraction
from river import stats
import torch

class RnnModule(torch.nn.Module):

    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.n_features=n_features
        self.rnn = torch.nn.RNN(input_size=n_features, hidden_size=hidden_size, num_layers=1)
        self.fc = torch.nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X, **kwargs):
        output, hn  = self.rnn(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= RollingRegressor(
    module=RnnModule,
    loss_fn='mse',
    optimizer_fn='sgd',
    lr=1e-2,
    hidden_size=20,
    window_size=32,
)
```


```python
import datetime as dt

evaluate.progressive_val_score(
    dataset=dataset.take(50000),
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=5_000
)
```

## Building LSTM Models


```python
class LstmModule(torch.nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.fc = torch.nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= RollingRegressor(
    module=LstmModule,
    loss_fn='mse',
    optimizer_fn='sgd',
    lr=1e-2,
    hidden_size=20,
    window_size=32,
)
```


```python
import datetime as dt

evaluate.progressive_val_score(
    dataset=dataset.take(50000),
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=5_000
)
```
