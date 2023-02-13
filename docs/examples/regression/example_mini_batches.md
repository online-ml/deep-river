```python
import pandas as pd
from river import datasets
from deep_river import regression
from torch import nn
from river import compose
from river import preprocessing
from itertools import islice
from sklearn import metrics
```


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

def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch
```


```python
dataset = datasets.Bikes()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= regression.Regressor(module=MyModule, loss_fn="mse", optimizer_fn="sgd")
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">['clouds', 'humidity', 'pressure', 'temperature', 'wind']</pre></summary><code class="river-estimator-params">(
  clouds
  humidity
  pressure
  temperature
  wind
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
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
y_trues = []
y_preds = []
for batch in batcher(dataset.take(5000),5):
    x,y = zip(*batch)
    x = pd.DataFrame(x)
    y_trues.extend(y)
    y_preds.extend(model_pipeline.predict_many(X=x))
    model_pipeline.learn_many(X=x, y=y)
```

    /Users/kulbach/Documents/environments/deep-river39/lib/python3.9/site-packages/river/preprocessing/scale.py:238: RuntimeWarning: invalid value encountered in scalar power
      stds = np.array([self.vars[c] ** 0.5 for c in X.columns])



```python
metrics.mean_squared_error(y_true=y_trues,y_pred=y_preds)
```




    102.4412




```python

```
