import subprocess
import sys

import numpy as np
import torch


def test_numpy_tensor_round_trip():
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = torch.from_numpy(values)
    np.testing.assert_array_equal(tensor.numpy(), values)


def test_core_learning_does_not_import_optional_dependencies():
    script = """
import importlib.abc
import importlib.util
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"pytest", "sklearn", "torchviz", "tqdm"}:
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(module.__name__)

sys.meta_path.insert(0, BlockOptionalImports())
import torch
from deep_river.regression import Regressor
model = Regressor(torch.nn.Sequential(torch.nn.Linear(1, 1)), "mse", "sgd")
model.learn_one({"x": 1.0}, 2.0)
assert isinstance(model.predict_one({"x": 1.0}), float)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
