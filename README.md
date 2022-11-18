<p align="center">
  <img height="150px" src="https://raw.githubusercontent.com/online-ml/river-torch/master/docs/img/logo.png" alt="incremental dl logo">
</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/river-torch">
    <a href="https://codecov.io/gh/online-ml/river-torch" > 
        <img src="https://codecov.io/gh/online-ml/river-torch/branch/master/graph/badge.svg?token=ZKUIISZAYA"/> 
    </a>
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dw/river-torch">
    <img alt="GitHub" src="https://img.shields.io/github/license/online-ml/river-torch">
</p>
<p align="center">
    river-torch is a Python library for online deep learning.
    River-torch's ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks.
    It combines the <a href="https://www.riverml.xyz">river</a> API with the capabilities of designing neural networks based on <a href="https://pytorch.org">PyTorch</a>.
</p>

## 💈 Installation

```shell
pip install river-torch
```
or
```shell
pip install "river[torch]"
```
You can install the latest development version from GitHub as so:

```shell
pip install https://github.com/online-ml/river-torch/archive/refs/heads/master.zip
```

## 🍫 Quickstart

We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="https://online-ml.github.io/river-torch">Documentation</a>.

### Classification

```python
>>> from river import metrics, datasets, preprocessing, compose
>>> from river_torch import classification
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
... )

>>> dataset = datasets.Phishing()
>>> metric = metrics.Accuracy()

>>> for x, y in dataset:
...     y_pred = model_pipeline.predict_one(x)  # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model_pipeline = model_pipeline.learn_one(x,y)  # make the model learn
>>> print(f"Accuracy: {metric.get():.4f}")
Accuracy: 0.6728

```

### Anomaly Detection

```python
>>> from river_torch.anomaly import Autoencoder
>>> from river import metrics
>>> from river.datasets import CreditCard
>>> from torch import nn
>>> import math
>>> from river.compose import Pipeline
>>> from river.preprocessing import MinMaxScaler

>>> dataset = CreditCard().take(5000)
>>> metric = metrics.ROCAUC(n_thresholds=50)

>>> class MyAutoEncoder(nn.Module):
...     def __init__(self, n_features, latent_dim=3):
...         super(MyAutoEncoder, self).__init__()
...         self.linear1 = nn.Linear(n_features, latent_dim)
...         self.nonlin = nn.LeakyReLU()
...         self.linear2 = nn.Linear(latent_dim, n_features)
...         self.sigmoid = nn.Sigmoid()
...
...     def forward(self, X, **kwargs):
...         X = self.linear1(X)
...         X = self.nonlin(X)
...         X = self.linear2(X)
...         return self.sigmoid(X)

>>> ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
>>> scaler = MinMaxScaler()
>>> model = Pipeline(scaler, ae)

>>> for x, y in dataset:
...    score = model.score_one(x)
...    model = model.learn_one(x=x)
...    metric = metric.update(y, score)
...
>>> print(f"ROCAUC: {metric.get():.4f}")
ROCAUC: 0.7447

```

## 🏫 Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>
