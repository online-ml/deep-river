# RNN Classification Models
This example shows the application of RNN models in river-torch with and without usage of an incremental class adaption strategy.


```python
from river_torch.classification import RollingClassifier
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


```python
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get():.2f }')
```

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


```python
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get():.2f}')
```

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


```python
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get()}')
```

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


```python
for x,y in tqdm(dataset):
    y_pred = model_pipeline.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model_pipeline.learn_one(x, y)    # make the model learn
print(f'Accuracy: {metric.get()}')
```


```python

```
