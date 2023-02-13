# Mini Batches
Iterate over a data stream in mini batches


```python
import pandas as pd
from river import datasets
from deep_river import classification
from torch import nn
from river import compose
from river import preprocessing
from itertools import islice
from sklearn import metrics
```


```python
dataset = datasets.Phishing()
```


```python
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(n_features,5)
        self.nonlin = nn.ReLU()
        self.dense1 = nn.Linear(5, 2)
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
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    classification.Classifier(module=MyModule,loss_fn="binary_cross_entropy",optimizer_fn="sgd")
)
model
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">Classifier</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="binary_cross_entropy"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.001
  output_is_logit=True
  is_class_incremental=False
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
for batch in batcher(dataset,5):
    x,y = zip(*batch)
    x = pd.DataFrame(x)
    y_trues.extend(y)
    y = pd.Series(y)
    y_preds.extend(model.predict_many(X=x))
    model = model.learn_many(x, y)    # make the model learn
```


```python
metrics.accuracy_score(
    y_pred=[str(i) for i in y_preds],
    y_true=[str(i) for i in y_trues]
)
```




    0.4192


