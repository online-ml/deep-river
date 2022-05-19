[![unit-tests](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/python-app.yml)
[![docs](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/main.yml/badge.svg)](https://github.com/kulbachcedric/IncrementalTorch/actions/workflows/main.yml)

<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>

<p align="center">
    IncrementalTorch is a Python library for online deep learning.
    IncrementalTorch ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks.
    It combines the <a href="https://www.riverml.xyz">river</a> API with the capabilities of designing neural networks based on <a href="https://pytorch.org">PyTorch</a>. 
</p>

## 💈 Installation
```shell
pip install IncrementalTorch
```

## 🍫 Quickstart
We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
```python
from river import datasets
from river import metrics
from river import preprocessing
from river import compose
from IncrementalTorch import classification
from torch import nn
from torch import optim
from torch import manual_seed
_ = manual_seed(0)

def build_torch_mlp_classifier(n_features):     # build neural architecture
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )
    return net

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    classification.PyTorch2RiverClassifier(build_fn=build_torch_mlp_classifier,loss_fn='bce',optimizer_fn=optim.Adam,learning_rate=1e-3)
) 

dataset = datasets.Phishing()
metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model.learn_one(x, y)      # make the model learn

print(metric)
```

## 🏫 Affiliations
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>
