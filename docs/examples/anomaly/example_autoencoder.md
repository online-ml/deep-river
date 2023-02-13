# Simple Fully Connected Autoencoder


```python
from river import compose, preprocessing, metrics, datasets

from deep_river.anomaly import Autoencoder
from torch import nn, manual_seed
```


```python
_ = manual_seed(42)
dataset = datasets.CreditCard().take(5000)
metric = metrics.ROCAUC(n_thresholds=50)

class MyAutoEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super(MyAutoEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.nonlin = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, **kwargs):
        X = self.linear1(X)
        X = self.nonlin(X)
        X = self.linear2(X)
        return self.sigmoid(X)

model_pipeline = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    Autoencoder(module=MyAutoEncoder, lr=0.005)
)
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">MinMaxScaler</pre></summary><code class="river-estimator-params">()

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">Autoencoder</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="mse_loss"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.005
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
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```

    ROCAUC: 0.7447



```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```

    ROCAUC: 0.7447



```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```

    ROCAUC: 0.7447



```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```

    ROCAUC: 0.7447

