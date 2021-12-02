from river import stream
import base


class Covertype(base.FileDataset):
    def __init__(self):
        super().__init__(
            n_samples=581_012,
            n_features=54,
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
        }
        for i in range(1, 5): self.converters[f"Wilderness_Area{i}"] = int
        for i in range(1, 41): self.converters[f"Soil_Type{i}"] = int


    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target="is_anom",
            converters=self.converters
        )
