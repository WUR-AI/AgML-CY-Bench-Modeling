import pickle
import logging
from pathlib import Path

import numpy as np

from cybench.models.model import BaseModel
from cybench.datasets.dataset import PandasDataset
from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET

log = logging.getLogger(__name__)


class AverageYieldModel(BaseModel):
    """Predicts the average training-set yield, grouped by location.

    For unseen test locations, finds the nearest training location
    using coordinate columns (loc_x/y/z or longitude/latitude) and
    returns that neighbor's average. Falls back to the global average
    only when no location features are available.

    Operates on PandasDataset; compatible with Hydra instantiation.

    Parameters
    ----------
    name : str
        Model name, used for saving artifacts.
    group_by : str
        Index level to group training targets by when computing
        averages (e.g. ``admin`` for per-location averages).
    """

    def __init__(self, name: str = "average_yield", group_by: str = KEY_LOC):
        self.name = name
        if group_by == "admin": group_by = KEY_LOC
        self._group_by = group_by
        self._averages = None
        self._global_avg = None
        self._location_columns = None
        self._location_df = None
        self._train_loc_coords = None

    def fit(self, dataset: PandasDataset, **fit_params) -> dict:
        x, y = dataset.xy
        y = y.reset_index()
        self._averages = y.groupby(self._group_by)[KEY_TARGET].mean()
        self._global_avg = y[KEY_TARGET].mean()

        # Build per-location coordinate lookup for nearest-neighbor fallback
        if 'loc_x' in x.columns:
            self._location_columns = ['loc_x', 'loc_y', 'loc_z']
        elif 'longitude' in x.columns:
            self._location_columns = ['longitude', 'latitude']
        else:
            self._location_columns = None

        if self._location_columns is not None:
            self._location_df = x[self._location_columns]
            # One coordinate row per training location (deduplicate over years)
            self._train_loc_coords = (
                self._location_df
                .groupby(level=KEY_LOC)
                .first()
            )
        return {}

    def predict(self, dataset: PandasDataset, **predict_params):
        x, y = dataset.xy
        locs = y.index.get_level_values(KEY_LOC)
        predictions = np.empty(len(locs))

        for i, loc in enumerate(locs):
            if loc in self._averages.index:
                predictions[i] = self._averages[loc]
            else:
                predictions[i] = self._nearest_neighbor_avg(loc, x)

        return predictions, {}

    def _nearest_neighbor_avg(self, loc, test_x):
        """Look up the average of the nearest training location by coordinates.

        Falls back to the global average when no coordinate columns are available
        or the test location has no coordinate data.
        """
        if self._train_loc_coords is None or self._location_columns is None:
            return self._global_avg

        # Get coordinates for the query location from the test features
        if loc not in test_x.index.get_level_values(KEY_LOC):
            return self._global_avg

        query = test_x.loc[loc, self._location_columns].values
        if query.ndim > 1:
            query = query[0]

        # Euclidean distance to every training location
        train_coords = self._train_loc_coords.values
        dists = np.linalg.norm(train_coords - query, axis=1)
        nearest_loc = self._train_loc_coords.index[np.argmin(dists)]

        log.info("Unseen location %s -> nearest neighbor %s", loc, nearest_loc)
        return self._averages[nearest_loc]

    def save(self, path):
        with open(Path(path) / f"{self.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)