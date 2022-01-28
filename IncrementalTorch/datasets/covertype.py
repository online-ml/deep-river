from river import stream

from . import base


class Covertype(base.FileDataset):
    def __init__(self):
        super().__init__(
            n_samples=286_048,
            n_features=10,
            filename="covertype.csv.zip",
            task=base.BINARY_CLF,
        )
        self.converters = {
            "Elevation": float,
            "Aspect": float,
            "Slope": float,
            "Horizontal_Distance_To_Hydrology": float,
            "Vertical_Distance_To_Hydrology": float,
            "Horizontal_Distance_To_Roadways": float,
            "Hillshade_9am": float,
            "Hillshade_Noon": float,
            "Hillshade_3pm": float,
            "Horizontal_Distance_To_Fire_Points": float,
            "Is_Anomaly": int,
        }

    def __iter__(self):
        return stream.iter_csv(self.path, target="Is_Anomaly", converters=self.converters)
