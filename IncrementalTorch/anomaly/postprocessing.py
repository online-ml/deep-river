from collections import deque

import numpy as np
from river import base


class ExponentialStandardizer(base.Transformer):
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
                self.var = self.momentum * (
                        self.var + (1 - self.momentum) * last_diff ** 2
                )

    def transform_one(self, x):
        x_centered = 0 if self.mean is None else x - self.mean
        if self.with_std:
            x_standardized = np.divide(x_centered, self.var ** 0.5, where=self.var > 0)
        else:
            x_standardized = x_centered
        return x_standardized

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)


class ExponentialMeanScaler(ExponentialStandardizer):
    def __init__(self, momentum=0.99) -> None:
        super().__init__(momentum=momentum, with_std=False)

    def transform_one(self, x):
        x_centered = 0 if self.mean is None else x / self.mean
        return x_centered


class WindowedStandardizer(base.Transformer):
    def __init__(self, window_size=250) -> None:
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0
        self.dsquared = 0

    def learn_one(self, x):
        if self.current_size < self.window_size:
            self.values.append(x)
            self.current_size += 1
            mean_old = self.mean
            self.mean += (x - mean_old) / self.current_size
            self.dsquared += (x - self.mean) * (x - mean_old)

        else:
            x_old = self.values.popleft()
            mean_old = self.mean
            self.mean += (x - x_old) / self.window_size
            self.dsquared += (x - x_old) * (x + x_old - self.mean - mean_old)
            self.values.append(x)

    def transform_one(self, x):
        x_centered = 0 if self.mean is None else x - self.mean
        x_standardized = np.divide(x_centered, self.var ** 0.5, where=self.var > 0)
        return x_standardized

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)

    @property
    def var(self):
        if self.current_size < 1:
            return 0
        return self.dsquared / self.current_size


class WindowedMeanScaler(base.Transformer):
    def __init__(self, window_size=250) -> None:
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def learn_one(self, x):
        if self.current_size < self.window_size:
            self.values.append(x)
            self.current_size += 1
            self.mean += (x - self.mean) / self.current_size

        else:
            x_old = self.values.popleft()
            self.mean += (x - x_old) / self.window_size
            self.values.append(x)

    def transform_one(self, x):
        return 0 if self.mean is None else x / self.mean

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)


class WindowedMinMaxScaler(base.Transformer):
    def __init__(self, window_size=250) -> None:
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
        self.current_size = 0
        self.max = -np.Inf
        self.min = np.Inf

    def learn_one(self, x):
        if x > self.max:
            self.max = x

        if x < self.min:
            self.min = x

    def transform_one(self, x: dict) -> dict:
        diff = self.max - self.min
        if diff > 0:
            return x - self.min / diff
        else:
            return 0

    def learn_transform_one(self, x):
        self.learn_one(x)
        return self.transform_one(x)
