from river import base
import numpy as np

class ScoreStandardizer(base.Transformer):
    def __init__(self, momentum=0.99, with_std=True):
        self.with_std = with_std
        self.momentum = momentum
        self.mean = None
        self.var = 0

    def learn_one(self, x):
        if self.mean is None:
            self.mean = x
        else:
            last_diff = x - self.mean
            self.mean += (1 - self.momentum) * last_diff
            if self.with_std:
                self.var = self.momentum * (self.var + (1 - self.momentum) * last_diff ** 2)

    def transform_one(self, x):
        x_centered = x - self.mean
        if self.with_std:
            x_standardized = np.divide(x_centered, self.var ** 0.5, where=self.var > 0)
        else:
            x_standardized = x_centered
        return x_standardized

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)