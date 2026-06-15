from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle
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

    def __init__(
        self,
        name: str = "average_yield",
        group_by: str = KEY_LOC,
        **_ignored,
    ):
        # Hydra model configs may include metadata keys (e.g. framework).
        # Accept and ignore unknown kwargs to keep instantiation robust.
        self.name = name
        if group_by == "admin": group_by = KEY_LOC
        self._group_by = group_by
        self._averages: pd.Series | None = None
        self._global_avg: float | None = None
        self._location_columns: list[str] | None = None
        self._location_df: pd.DataFrame | None = None
        self._train_loc_coords: pd.DataFrame | None = None

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        x, y = dataset.xy
        y = y.reset_index()
        self._averages = cast(pd.Series, y.groupby(self._group_by)[KEY_TARGET].mean())
        self._global_avg = float(y[KEY_TARGET].mean())

        # Build per-location coordinate lookup for nearest-neighbor fallback
        if 'loc_x' in x.columns:
            self._location_columns = ['loc_x', 'loc_y', 'loc_z']
        elif 'longitude' in x.columns:
            self._location_columns = ['longitude', 'latitude']
        else:
            self._location_columns = None

        if self._location_columns is not None:
            location_df = cast(pd.DataFrame, x[self._location_columns])
            self._location_df = location_df
            # One coordinate row per training location (deduplicate over years)
            self._train_loc_coords = cast(
                pd.DataFrame,
                location_df.groupby(level=KEY_LOC).first(),
            )
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        averages = self._averages
        if averages is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")

        x, y = dataset.xy
        locs = y.index.get_level_values(KEY_LOC)
        predictions = np.empty(len(locs))

        for i, loc in enumerate(locs):
            if loc in averages.index:
                predictions[i] = averages[loc]
            else:
                predictions[i] = self._nearest_neighbor_avg(loc, x)

        return predictions, {}

    def _nearest_neighbor_avg(self, loc, test_x: pd.DataFrame) -> float:
        """Look up the average of the nearest training location by coordinates.

        Falls back to the global average when no coordinate columns are available
        or the test location has no coordinate data.
        """
        averages = self._averages
        if averages is None or self._global_avg is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")

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
        return float(averages[nearest_loc])

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str = "average_yield") -> AverageYieldModel:
        return load_pickle(model_path, name)
