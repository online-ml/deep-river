# Simple Classification Model


```python
from river import metrics, datasets, compose, preprocessing
from deep_river.classification import Classifier
from torch import nn
from tqdm import tqdm
```


```python
dataset = datasets.Phishing()
metric = metrics.Accuracy()

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

model_pipeline = compose.Pipeline(
    preprocessing.StandardScaler,
    Classifier(module=MyModule,loss_fn="binary_cross_entropy",optimizer_fn='adam')
)
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">Classifier</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="binary_cross_entropy"
  optimizer_fn=&lt;class 'torch.optim.adam.Adam'&gt;
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
for x,y in tqdm(dataset.take(5000)):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model_pipeline = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get()}')
```

    1250it [00:00, 1367.42it/s]

    Accuracy: 0.6728


    

