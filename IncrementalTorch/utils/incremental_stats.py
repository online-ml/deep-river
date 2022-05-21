import math
from collections import deque


class WindowedVarianceMeter:
    def __init__(self, window_size:int=250) -> None:
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0
        self.dsquared = 0

    def __len__(self):
        return len(self.values)

    def update(self, x_new):
        if self.current_size < self.window_size:
            self.values.append(x_new)
            self.current_size += 1
            mean_old = self.mean
            self.mean += (x_new - mean_old) / self.current_size
            self.dsquared += (x_new - self.mean) * (x_new - mean_old)

        else:
            x_old = self.values.popleft()
            mean_old = self.mean
            self.mean += (x_new - x_old) / self.window_size
            self.dsquared += (x_new - x_old) * (x_new + x_old - self.mean - mean_old)
            self.values.append(x_new)

    @property
    def population_var(self):
        if self.current_size < 1:
            return 0
        return self.dsquared / self.current_size

    @property
    def population_std(self):
        if self.current_size < 1:
            return 0
        return math.sqrt(self.dsquared / self.current_size)

    @property
    def sample_var(self):
        if self.current_size <= 1:
            return 0
        else:
            return self.dsquared / (self.current_size - 1)

    @property
    def sample_std(self):
        if self.current_size <= 1:
            return 0
        else:
            return math.sqrt(self.dsquared / (self.current_size - 1))


class WindowedMeanMeter:
    """

    """
    def __init__(self, window_size:int = 250) -> None:
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, x_new):
        if self.current_size < self.window_size:
            self.values.append(x_new)
            self.current_size += 1
            self.mean += (x_new - self.mean) / self.current_size

        else:
            x_old = self.values.popleft()
            self.mean += (x_new - x_old) / self.window_size
            self.values.append(x_new)
