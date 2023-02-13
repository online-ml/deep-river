# LSTM Regression Model


```python
from deep_river.regression import RollingRegressor
from river import metrics, compose, preprocessing, datasets, stats, feature_extraction
from torch import nn
from tqdm import tqdm
```


```python
def get_hour(x):
    x['hour'] = x['moment'].hour
    return x
```

## Simple RNN Regression Model


```python
class RnnModule(nn.Module):

    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.n_features=n_features
        self.rnn = nn.RNN(input_size=n_features, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X, **kwargs):
        output, hn  = self.rnn(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])
```


```python
dataset = datasets.Bikes()
metric = metrics.MAE()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= RollingRegressor(
    module=RnnModule,
    loss_fn='mse',
    optimizer_fn='sgd',
    window_size=20,
    lr=1e-2,
    hidden_size=32,  # parameters of MyModule can be overwritten
    append_predict=True,
)
model_pipeline
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

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">RollingRegressor</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="mse_loss"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.01
  window_size=20
  append_predict=True
  device="cpu"
  seed=42
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




```python
for x, y in tqdm(dataset.take(5000)):
    y_pred = model_pipeline.predict_one(x)
    metric = metric.update(y_true=y, y_pred=y_pred)
    model_pipeline = model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
```

    5000it [00:11, 451.42it/s]

    MAE: 3.94


    


## LSTM Regression Model


```python
class LstmModule(nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])

```


```python
dataset = datasets.Bikes()
metric = metrics.MAE()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= RollingRegressor(
    module=LstmModule,
    loss_fn='mse',
    optimizer_fn='sgd',
    window_size=20,
    lr=1e-2,
    hidden_size=32,  # parameters of MyModule can be overwritten
    append_predict=True,
)
model_pipeline
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

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">RollingRegressor</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="mse_loss"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.01
  window_size=20
  append_predict=True
  device="cpu"
  seed=42
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




```python
for x, y in tqdm(dataset.take(5000)):
    y_pred = model_pipeline.predict_one(x)
    metric = metric.update(y_true=y, y_pred=y_pred)
    model_pipeline = model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
```

    5000it [00:22, 225.22it/s]

    MAE: 2.81


    

