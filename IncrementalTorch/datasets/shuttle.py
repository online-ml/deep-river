from river import stream
from . import base


class Shuttle(base.FileDataset):
    def __init__(self):
        super().__init__(
            n_samples=58000,
            n_features=9,
            filename="shuttle.csv.zip",
            task=base.BINARY_CLF,
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="is_anom",
            converters={f"V{i}": float for i in range(1, 10)},
        )
