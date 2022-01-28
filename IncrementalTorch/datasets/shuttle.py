from river import stream

from . import base


class Shuttle(base.FileDataset):
    def __init__(self):
        super().__init__(
            n_samples=49097,
            n_features=9,
            filename="shuttle.csv.zip",
            task=base.BINARY_CLF,
        )
        self.converters = {f"V{i}": float for i in range(1, 10)}
        self.converters["is_anom"] = int

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="is_anom",
            converters=self.converters,
        )
