![GitHub last commit](https://img.shields.io/github/last-commit/kulbachcedric/DeepRiver)
[![unit-tests](https://github.com/kulbachcedric/DeepRiver/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/kulbachcedric/DeepRiver/actions/workflows/unit-tests.yml)
![Codecov](https://img.shields.io/codecov/c/github/kulbachcedric/DeepRiver)
[![docs](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/mkdocs.yml/badge.svg)](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/unit_test.yml)

<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>

<p align="center">
    DeepRiver is a Python library for online deep learning.
    DeepRivers ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks.
    It combines the <a href="https://www.riverml.xyz">river</a> API with the capabilities of designing neural networks based on <a href="https://pytorch.org">PyTorch</a>.
</p>

## 💈 Installation

```shell
pip install river-torch
```

You can install the latest development version from GitHub as so:

```shell
pip install https://github.com/online-ml/river-torch --upgrade
```

Or, through SSH:

```shell
pip install git@github.com:online-ml/river-torch.git --upgrade
```

## 🍫 Quickstart

We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="https://online-ml.github.io/river-torch">Documentation</a>.

### Classification

```python
>>> from river.datasets import Phishing
>>> from river import metrics
>>> from river import preprocessing
>>> from river import compose
>>> from river_torch import classification
>>> from torch import nn
>>> from torch import optim
>>> from torch import manual_seed

>>> _ = manual_seed(42)


>>> def build_torch_mlp_classifier(n_features):  # build neural architecture
...     net = nn.Sequential(
...         nn.Linear(n_features, 5),
...         nn.Linear(5, 5),
...         nn.Linear(5, 5),
...         nn.Linear(5, 5),
...         nn.Linear(5, 1),
...         nn.Sigmoid()
...     )
...     return net


>>> model = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     classification.Classifier(build_fn=build_torch_mlp_classifier, loss_fn='binary_cross_entropy', optimizer_fn=optim.Adam, lr=1e-3)
... )

>>> dataset = Phishing()
>>> metric = metrics.Accuracy()

>>> for x, y in dataset:
...     y_pred = model.predict_one(x)  # make a prediction
...     metric = metric.update(y, y_pred)  # update the metric
...     model = model.learn_one(x, y)  # make the model learn

>>> print(f'Accuracy: {metric.get()}')
Accuracy: 0.8304

```

### Anomaly Detection

```python
>>> import math
>>> from river import metrics, preprocessing
>>> from river.datasets import CreditCard
>>> from river_torch.anomaly import Autoencoder
>>> from river_torch.utils import get_activation_fn
>>> from torch import manual_seed, nn

>>> _ = manual_seed(42)

>>> def get_ae(n_features=3, dropout=0.1):
...     latent_dim = math.ceil(n_features / 2)
...     net = nn.Sequential(
...         nn.Dropout(p=dropout),
...         nn.Linear(in_features=n_features, out_features=latent_dim),
...         nn.ReLU(),
...         nn.Linear(in_features=latent_dim, out_features=n_features),
...         nn.Sigmoid()
...     )
...     return net

>>> dataset = CreditCard().take(5000)
>>> metric = metrics.ROCAUC()
>>> scaler = preprocessing.MinMaxScaler()

>>> model = Autoencoder(build_fn=get_ae, lr=0.01)

>>> for x, y in dataset:
...     x = scaler.learn_one(x).transform_one(x)
...     score = model.score_one(x)
...     metric = metric.update(y_true=y, y_pred=score)
...     model = model.learn_one(x=x)

```

## 🏫 Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>
