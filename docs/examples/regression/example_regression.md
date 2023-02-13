# Simple Regression Model


```python
from river import metrics, compose, preprocessing, datasets, stats, feature_extraction
from deep_river.regression import Regressor
from torch import nn
from pprint import pprint
from tqdm import tqdm
```


```python
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



```python
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(n_features,5)
        self.nonlin = nn.ReLU()
        self.dense1 = nn.Linear(5, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.softmax(X)
        return X

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x
```


```python
metric = metrics.MAE()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= Regressor(module=MyModule, loss_fn="mse", optimizer_fn='sgd')
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

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">Regressor</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="mse_loss"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.001
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
    metric.update(y_true=y, y_pred=y_pred)
    model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
```

    5000it [00:04, 1029.49it/s]

    MAE: 6.83


    

