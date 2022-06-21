from river import stream

from . import base


class MNIST(base.FileDataset):
    def __init__(self):
        super().__init__(
            n_samples=50_000,
            n_features=784,
            filename="mnist.csv.zip",
            task=base.BINARY_CLF,
        )
        self.converters = {f"Pixel {i}": float for i in range(1, 785)}
        self.converters["is_anom"] = int

    def __iter__(self):
        return stream.iter_csv(self.path, target="is_anom", converters=self.converters)
