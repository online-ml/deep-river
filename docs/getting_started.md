# Getting started
We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="https://online-ml.github.io/deep-river">Documentation</a>.

##💈Installation

River is meant to work with Python 3.8 and above. Installation can be done via `pip`:

```sh
pip install deep-river
```
or
```sh
pip install "river[deep]"
```

You can install the latest development version from GitHub, as so:

```sh
pip install git+https://github.com/online-ml/deep-river --upgrade
```

Or, through SSH:

```sh
pip install git+ssh://git@github.com/online-ml/deep-river.git --upgrade
```

Feel welcome to [open an issue on GitHub](https://github.com/online-ml/deep-river/issues/new) if you are having any trouble.


## 💻 Usage

### Classification

```python
>>> from river import metrics, datasets, preprocessing, compose
>>> from deep_river import classification
>>> from torch import nn
>>> from torch import optim
>>> from torch import manual_seed

>>> _ = manual_seed(42)

>>> class MyModule(nn.Module):
...     def __init__(self, n_features):
...         super(MyModule, self).__init__()
...         self.dense0 = nn.Linear(n_features, 5)
...         self.nonlin = nn.ReLU()
...         self.dense1 = nn.Linear(5, 2)
...         self.softmax = nn.Softmax(dim=-1)
...
...     def forward(self, X, **kwargs):
...         X = self.nonlin(self.dense0(X))
...         X = self.nonlin(self.dense1(X))
...         X = self.softmax(X)
...         return X

>>> model_pipeline = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     classification.Classifier(module=MyModule, loss_fn='binary_cross_entropy', optimizer_fn='adam')
...     )

>>> dataset = datasets.Phishing()
>>> metric = metrics.Accuracy()

>>> for x, y in dataset:
...     y_pred = model_pipeline.predict_one(x)  # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model_pipeline = model_pipeline.learn_one(x, y)  # make the model learn
>>>     print(f"Accuracy: {metric.get():.4f}")
Accuracy: 0.6728

```

### Regression

```python
>>> from river import metrics, compose, preprocessing, datasets
>>> from deep_river.regression import Regressor
>>> from torch import nn
>>> from pprint import pprint
>>> from tqdm import tqdm

>>> dataset = datasets.Bikes()
>>> metric = metrics.MAE()

>>>

class MyModule(nn.Module):


    ...


def __init__(self, n_features):


    ...
super(MyModule, self).__init__()
...
self.dense0 = nn.Linear(n_features, 5)
...
self.nonlin = nn.ReLU()
...
self.dense1 = nn.Linear(5, 1)
...
self.softmax = nn.Softmax(dim=-1)
...
...


def forward(self, X, **kwargs):


    ...
X = self.nonlin(self.dense0(X))
...
X = self.nonlin(self.dense1(X))
...
X = self.softmax(X)
...
return X
>> > model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
>> > model_pipeline |= preprocessing.StandardScaler()
>> > model_pipeline |= Regressor(module=MyModule, loss_fn="mse", optimizer_fn='sgd')
>> > for x, y in dataset.take(5000):
    ...
y_pred = model_pipeline.predict_one(x)
...
metric.update(y_true=y, y_pred=y_pred)
...
model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
MAE: 6.83
```

### Anomaly Detection

```python
>> > from deep_river.anomaly import Autoencoder
>> > from river import metrics
>> > from river.datasets import CreditCard
>> > from torch import nn
>> > import math
>> > from river.compose import Pipeline
>> > from river.preprocessing import MinMaxScaler

>> > dataset = CreditCard().take(5000)
>> > metric = metrics.ROCAUC(n_thresholds=50)

>> >

class MyAutoEncoder(nn.Module):


    ...


def __init__(self, n_features, latent_dim=3):


    ...
super(MyAutoEncoder, self).__init__()
...
self.linear1 = nn.Linear(n_features, latent_dim)
...
self.nonlin = nn.LeakyReLU()
...
self.linear2 = nn.Linear(latent_dim, n_features)
...
self.sigmoid = nn.Sigmoid()
...
...


def forward(self, X, **kwargs):


    ...
X = self.linear1(X)
...
X = self.nonlin(X)
...
X = self.linear2(X)
...
return self.sigmoid(X)

>> > ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
>> > scaler = MinMaxScaler()
>> > model = Pipeline(scaler, ae)

>> > for x, y in dataset:
    ...
score = model.score_one(x)
...
model = model.learn_one(x=x)
...
metric = metric.update(y, score)
...
>> > print(f"ROCAUC: {metric.get():.4f}")
ROCAUC: 0.7447
```
