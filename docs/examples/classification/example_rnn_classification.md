# RNN Classification Models
This example shows the application of RNN models in river-torch with and without usage of an incremental class adaption strategy.


```python
from deep_river.classification import RollingClassifier
from river import metrics, compose, preprocessing, datasets
import torch
from tqdm import tqdm
```

## RNN Model


```python
class RnnModule(torch.nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        #self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1)
        self.rnn = torch.nn.RNN(input_size=n_features, hidden_size=hidden_size, num_layers=1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        out, hn  = self.rnn(X)  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.rnn.hidden_size)
        return self.softmax(hn)
```

### Classification without incremental class adapation strategy


```python
dataset = datasets.Keystroke()
metric = metrics.Accuracy()
optimizer_fn = torch.optim.SGD

model_pipeline = preprocessing.StandardScaler()
model_pipeline |= RollingClassifier(
    module=RnnModule,
    loss_fn="binary_cross_entropy",
    optimizer_fn=torch.optim.SGD,
    window_size=20,
    lr=1e-2,
    append_predict=True,
    is_class_incremental=False,
)
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">RollingClassifier</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="binary_cross_entropy"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.01
  output_is_logit=True
  is_class_incremental=False
  device="cpu"
  seed=42
  window_size=20
  append_predict=True
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
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get():.2f}')
```

    20400it [00:30, 666.50it/s]

    Accuracy: 0.02


    


### Classification with incremental class adaption strategy


```python
dataset = datasets.Keystroke()
metric = metrics.Accuracy()
optimizer_fn = torch.optim.SGD

model_pipeline = preprocessing.StandardScaler()
model_pipeline |= RollingClassifier(
    module=RnnModule,
    loss_fn="binary_cross_entropy",
    optimizer_fn=torch.optim.SGD,
    window_size=20,
    lr=1e-2,
    append_predict=True,
    is_class_incremental=True,
)
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">RollingClassifier</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="binary_cross_entropy"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.01
  output_is_logit=True
  is_class_incremental=True
  device="cpu"
  seed=42
  window_size=20
  append_predict=True
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
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get():.2f}')
```

    20400it [00:31, 652.13it/s]

    Accuracy: 0.09


    


## LSTM Model


```python
class LstmModule(torch.nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.lstm.hidden_size)
        return self.softmax(hn)
```

### Classifcation without incremental class adaption strategy


```python
dataset = datasets.Keystroke()
metric = metrics.Accuracy()
optimizer_fn = torch.optim.SGD

model_pipeline = preprocessing.StandardScaler()
model_pipeline |= RollingClassifier(
    module=LstmModule,
    loss_fn="binary_cross_entropy",
    optimizer_fn=torch.optim.SGD,
    window_size=20,
    lr=1e-2,
    append_predict=True,
)
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">RollingClassifier</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="binary_cross_entropy"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.01
  output_is_logit=True
  is_class_incremental=False
  device="cpu"
  seed=42
  window_size=20
  append_predict=True
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
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get():.2f}')
```

    20400it [00:59, 344.86it/s]

    Accuracy: 0.02


    


### Classifcation with incremental class adaption strategy


```python
dataset = datasets.Keystroke()
metric = metrics.Accuracy()
optimizer_fn = torch.optim.SGD

model_pipeline = preprocessing.StandardScaler()
model_pipeline |= RollingClassifier(
    module=LstmModule,
    loss_fn="binary_cross_entropy",
    optimizer_fn=torch.optim.SGD,
    window_size=20,
    lr=1e-2,
    append_predict=True,
    is_class_incremental=True
)
model_pipeline
```




<div><div class="river-component river-pipeline"><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">StandardScaler</pre></summary><code class="river-estimator-params">(
  with_std=True
)

</code></details><details class="river-component river-estimator"><summary class="river-summary"><pre class="river-estimator-name">RollingClassifier</pre></summary><code class="river-estimator-params">(
  module=None
  loss_fn="binary_cross_entropy"
  optimizer_fn=&lt;class 'torch.optim.sgd.SGD'&gt;
  lr=0.01
  output_is_logit=True
  is_class_incremental=True
  device="cpu"
  seed=42
  window_size=20
  append_predict=True
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
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get()}:.2f')
```

    20400it [01:07, 300.25it/s]

    Accuracy: 0.13857843137254902:.2f


    

