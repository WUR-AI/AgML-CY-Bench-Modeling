from __future__ import annotations

from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Any

import pandas as pd
import numpy as np
import numpy.typing as npt

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
)


class BaseDataset(ABC):
    """
    Abstract base class defining the interface for custom datasets.
    All datasets must implement the split_on_years method.
    """

    @abstractmethod
    def split_on_years(
        self, years_split: tuple[Iterable[int], Iterable[int]]
    ) -> tuple['BaseDataset', 'BaseDataset']:
        """
        Split the dataset into two subsets based on year ranges.

        :param years_split: Tuple of two lists, e.g., ([2012, 2014], [2015, 2017])
                           First list defines years for first subset,
                           second list defines years for second subset
        :return: Tuple of two dataset instances (subset1, subset2)
        """
        pass

    @property
    @abstractmethod
    def years(self) -> set[Any]:
        """Obtain a set containing all years occurring in the dataset."""
        ...

    @property
    @abstractmethod
    def location_ids(self) -> set[Any]:
        """Obtain a set containing all location ids occurring in the dataset."""
        ...

    @property
    @abstractmethod
    def targets(self) -> npt.NDArray[Any]:
        """Obtain a numpy array of targets or labels."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        ...


class PandasDataset(BaseDataset):
    """Tabular dataset for use with sklearn, xgboost, lightgbm, and similar libraries.

    All features are stored in a single flat DataFrame (one row per location-year),
    ready to pass directly into model.fit(x, y).

    Parameters
    ----------
    y : pd.DataFrame
        Yield targets, indexed by (KEY_LOC, KEY_YEAR).
    x : pd.DataFrame
        All features, indexed by (KEY_LOC, KEY_YEAR). Produced by
        DataFactory after tabularization and merging of all sources.
    cfg : optional
        Dataset config, stored for reference.
    """

    def __init__(self, cfg, y: pd.DataFrame, x: pd.DataFrame, normalizer=None):
        self.cfg = cfg
        self.normalizer = normalizer
        self.y = y
        self.x = x
        self.indices = x.index.to_frame()

        # Align on index — drops any (loc, year) pairs not present in both
        self.x, self.y = x.align(y, join="inner", axis=0)

        # Downcast float64 -> float32 to halve memory usage.
        self.x = self.x.astype(
            {col: "float32" for col, dtype in self.x.dtypes.items() if dtype == "float64"}
        )

    def split_on_years(
        self, years_split: tuple[Iterable[int], Iterable[int]]
    ) -> tuple["PandasDataset", "PandasDataset"]:
        """Split into two datasets by year.

        Parameters
        ----------
        years_split : ([train_years], [test_years])

        Returns
        -------
        Two PandasDataset instances.
        """
        years1, years2 = years_split

        def _subset(years):
            mask = self.y.index.get_level_values(KEY_YEAR).isin(years)
            return PandasDataset(
                cfg=self.cfg,
                y=self.y.loc[mask],
                x=self.x.loc[mask],
                normalizer=self.normalizer,
            )

        return _subset(years1), _subset(years2)

    @property
    def xy(self):
        return self.x, self.y

    @property
    def years(self) -> set[Any]:
        return set(self.y.index.get_level_values(KEY_YEAR))

    @property
    def location_ids(self) -> set[Any]:
        return set(self.y.index.get_level_values(KEY_LOC))

    @property
    def targets(self) -> npt.NDArray[Any]:
        return self.y[KEY_TARGET].to_numpy()

    @property
    def feature_names(self) -> list[str]:
        return self.x.columns.tolist()

    def select_features(self, columns: list[str]) -> "PandasDataset":
        """Return a copy with only the given feature columns."""
        missing = set(columns) - set(self.x.columns)
        if missing:
            raise ValueError(f"Unknown feature columns: {sorted(missing)}")
        return PandasDataset(
            cfg=self.cfg,
            y=self.y,
            x=self.x.loc[:, list(columns)],
            normalizer=self.normalizer,
        )

    def __len__(self) -> int:
        return len(self.y)

    def save(self, cache_path: str) -> None:
        """Pickle the entire dataset to a single file.

        Stores everything — DataFrames, normalizer, config, and any future
        attributes — without needing to update this method when the class grows.

        Parameters
        ----------
        cache_path : str
            Full file path to write, e.g. ``cache/hash.pkl``.
            Parent directory is created if it does not exist.
        """
        import os, pickle
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(cache_path: str) -> "PandasDataset":
        """Load a pickled dataset from a file.

        Parameters
        ----------
        cache_path : str
            Path previously passed to ``save()``.
        """
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)
