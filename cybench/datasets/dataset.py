from abc import abstractmethod, ABC
from typing import Tuple

import pandas as pd
import numpy as np

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    KEY_CROP_SEASON,
    CROP_CALENDAR_DATES,
)


class BaseDataset(ABC):
    """
    Abstract base class defining the interface for custom datasets.
    All datasets must implement the split_on_years method.
    """

    @abstractmethod
    def split_on_years(self, years_split: Tuple[list, list]) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Split the dataset into two subsets based on year ranges.

        :param years_split: Tuple of two lists, e.g., ([2012, 2014], [2015, 2017])
                           First list defines years for first subset,
                           second list defines years for second subset
        :return: Tuple of two dataset instances (subset1, subset2)
        """
        pass

    @property
    def years(self) -> set:
        """
        Obtain a set containing all years occurring in the dataset
        """
        pass

    @property
    def location_ids(self) -> set:
        """
        Obtain a set containing all location ids occurring in the dataset
        """
        pass

    @property
    def targets(self) -> np.ndarray[tuple[int], np.dtype[np.number]]:
        """
        Obtain a numpy array of targets or labels
        """
        pass


class Dataset(BaseDataset):
    def __init__(
        self,
        cfg,
        df_y: pd.DataFrame = None,
        dfs_x: dict = None,
    ):
        """
        Dataset class for regional yield forecasting

        Targets/inputs are provided using properly formatted pandas dataframes.

        :param cfg: DatasetConfig defining all data related settings from data sources till feature design
        :param df_y: pandas.DataFrame that contains yield statistics
                            Dataframe should meet the following requirements:
                                - The column containing yield targets should be named properly
                                  Expected column name is stored in `config.KEY_TARGET`
                                - The dataframe is indexed by (location id, year) using the correct naming
                                  Expected names are stored in `config.KEY_LOC`, `config.KEY_YEAR`, resp.
        :param dfs_x: dict of data source to pandas.Dataframe objects each containing inputs
                            Dataframes should meet the following requirements:
                                - inputs are assumed to be numeric
                                - Columns should be named by their respective feature names
                                - Dataframes cannot have overlapping column (i.e. feature) names
                                - Each dataframe can be indexed in three different ways:
                                    - By location only -- for static location inputs
                                    - By location and year -- for yearly occurring inputs
                                    - By location, year, and some extra level assumed to be temporal (e.g. daily,
                                      dekadal, ...)
                                  The index levels should be named properly, i.e.
                                    - `config.KEY_LOC` for the location
                                    - `config.KEY_YEAR` for the year
                                    - the name of the extra optional temporal level is ignored and has no requirement
        """
        self.cfg = cfg

        # If no data is given, create an empty dataset
        if df_y is None:
            df_y = self._empty_df_target()
        if dfs_x is None:
            dfs_x = {}

        # Validate input data
        assert self._validate_dfs(df_y, dfs_x)

        self._df_y = df_y
        self._dfs_x = dfs_x

        self._max_season_window_length = None
        if KEY_CROP_SEASON in self._dfs_x:
            self._max_season_window_length = self._dfs_x[KEY_CROP_SEASON][
                "season_window_length"
            ].max()

        # Sort all data for faster lookups
        self._df_y.sort_index(inplace=True)
        for x in self._dfs_x:
            self._dfs_x[x].sort_index(inplace=True)

        # Bool value that specifies whether missing data values are allowed
        # For now always set to False
        self._allow_incomplete = False

    @property
    def crop(self):
        return self.cfg.crop

    @property
    def years(self) -> set:
        """
        Obtain a set containing all years occurring in the dataset
        """
        return set([year for _, year in self._df_y.index.values])

    @property
    def location_ids(self) -> set:
        """
        Obtain a set containing all location ids occurring in the dataset
        """
        return set([loc for loc, _ in self._df_y.index.values])

    @property
    def feature_names(self) -> set:
        """
        Obtain a set containing all feature names
        """
        return set.union(*[set(self._dfs_x[x].columns) for x in self._dfs_x])

    def targets(self) -> np.array:
        """
        Obtain a numpy array of targets or labels
        """
        return self._df_y[KEY_TARGET].values

    def indices(self) -> list:
        return self._df_y.index.values

    @property
    def max_season_window_length(self) -> int:
        return self._max_season_window_length

    def __getitem__(self, index) -> dict:
        """
        Get a single data point in the dataset

        Data point is returned as a dict

        :param index: index for accessing the data. Can be an int or the (location, year) that specify the data
        :return:
        """
        # Index is either integer or tuple of (year, location)
        if isinstance(index, int):
            sample_y = self._df_y.iloc[index]
            loc_id, year = sample_y.name

        elif isinstance(index, tuple):
            assert len(index) == 2
            loc_id, year = index
            sample_y = self._df_y.loc[index]

        else:
            raise Exception(f"Unsupported index type {type(index)}")

        # Get the target label for the specified sample
        sample = {
            KEY_YEAR: year,
            KEY_LOC: loc_id,
            KEY_TARGET: sample_y[KEY_TARGET],
        }

        # crop season dates are datetime objects
        if KEY_CROP_SEASON in self._dfs_x:
            sample_cc = self._dfs_x[KEY_CROP_SEASON].loc[(loc_id, year)]
            data_cc = {k: sample_cc[k] for k in CROP_CALENDAR_DATES}
            sample = {**data_cc, **sample}

        # Get feature data corresponding to the label
        data_x = self._get_feature_data(loc_id, year)
        # Merge label and feature data
        sample = {**data_x, **sample}

        return sample

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset
        """
        return len(self._df_y)

    def __iter__(self):
        """
        Iterate through the samples in the dataset
        """
        for i in range(len(self)):
            yield self[i]

    def _get_feature_data(self, loc_id: str, year: int) -> dict:
        """
        Helper function for obtaining feature data corresponding to some index
        :param loc_id: location index value
        :param year: year index value
        :return: a dict containing all feature data corresponding to the specified index
        """
        data = {
            KEY_DATES: dict(),
        }
        # For all feature dataframes
        for x in self._dfs_x:
            # handled in __getitem__()
            if x == KEY_CROP_SEASON:
                continue

            df = self._dfs_x[x]
            # Check in which category the dataframe fits:
            #   (1) static data -- indexed only by location
            #   (2) yearly data -- indexed by (location, year)
            #   (3) yearly temporal data -- indexed by (location, year, "some extra temporal level")
            n_levels = len(df.index.names)
            assert (
                1 <= n_levels <= 3
            )  # Make sure the dataframe fits one of the categories

            # (1) static data
            if n_levels == 1:
                if self._allow_incomplete:
                    # If value is missing, skip this feature
                    if loc_id not in df.index:
                        continue

                data = {
                    **df.loc[loc_id].to_dict(),
                    **data,
                }

            # (2) yearly data
            if n_levels == 2:
                if self._allow_incomplete:
                    # If value is missing, skip this feature
                    if (loc_id, year) not in df.index:
                        continue

                data = {
                    **df.loc[loc_id, year].to_dict(),
                    **data,
                }

            # (3) yearly temporal data
            if n_levels == 3:
                # Select data matching the location and year
                df_loc = df.xs((loc_id, year), drop_level=True)

                if self._allow_incomplete and len(df_loc) == 0:
                    # If value is missing, skip this feature
                    continue

                # Data in temporal dimension is assumed to be sorted
                # Obtain the values contained in the filtered dataframe
                data_loc = {key: df_loc[key].values for key in df_loc.columns}
                dates = {key: df_loc.index.values for key in df_loc.columns}

                data = {
                    **data_loc,
                    **data,
                }
                data[KEY_DATES] = {
                    **dates,
                    **data[KEY_DATES],
                }

        return data

    def get_normalization_params(self, normalization="standard"):
        """
        Compute normalization parameters for input data.
        :param normalization: normalization method, default standard or z-score
        :return: a dict containing normalization parameters (e.g. mean and std)
        """
        norm_params = {}
        for x, df in self._dfs_x.items():
            if x == KEY_CROP_SEASON:
                continue

            for c in df.columns:
                if normalization == "standard":
                    if len(df.index) > 1:
                        norm_params[c] = {"mean": df[c].mean(), "std": df[c].std()}
                    else:
                        # only one value, set to a small number to avoid division by zero
                        norm_params[c] = {"mean": df[c].mean(), "std": 1e-6}

                elif normalization == "min-max":
                    norm_params[c] = {"min": df[c].min(), "max": df[c].max()}
                else:
                    raise Exception(f"Unsupported normalization {normalization}")

        return norm_params

    @staticmethod
    def _empty_df_target() -> pd.DataFrame:
        """
        Helper function that creates an empty (but rightly formatted) dataframe for yield statistics
        """
        df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(([], []), names=[KEY_LOC, KEY_YEAR]),
            columns=[KEY_TARGET],
        )
        return df

    @staticmethod
    def _validate_dfs(df_y: pd.DataFrame, dfs_x: dict) -> bool:
        """
        Helper function that implements some checks on whether the input dataframes are correctly formatted

        :param df_y: dataframe containing yield statistics
        :param dfs_x: dict of data source to dataframes each containing feature data
        :return: a bool indicating whether the test has passed
        """

        # TODO -- more checks are to be implemented
        #   - Correct column names
        #   - Correct index names
        #   - Missing data

        # Make sure columns are named properly
        if len(dfs_x) > 0:
            column_names = set.union(*[set(dfs_x[x].columns) for x in dfs_x])
            assert KEY_LOC not in column_names
            assert KEY_YEAR not in column_names
            assert KEY_TARGET not in column_names
            assert KEY_DATES not in column_names

        # Make sure there are no overlaps in feature names
        if len(dfs_x) > 1:
            assert len(set.intersection(*[set(dfs_x[x].columns) for x in dfs_x])) == 0

        return True

    @staticmethod
    def _filter_df_on_index(df: pd.DataFrame, keys: list, level: int):
        """
        Helper method for filtering a dataframe based on the occurrence of certain values in a specified index

        :param df: the dataframe that should be filtered
        :param keys: the values on which it should filter
        :param level: the index level in which samples should be filtered
        :return: a filtered dataframe
        """
        if not isinstance(df.index, pd.MultiIndex):
            return df.loc[keys]
        else:
            return pd.concat(
                [df.xs(key, level=level, drop_level=False) for key in keys]
            )

    @staticmethod
    def _split_df_on_index(df: pd.DataFrame, split: tuple, level: int):
        df.sort_index(inplace=True)

        keys1, keys2 = split

        df_1 = Dataset._filter_df_on_index(df, keys1, level)
        df_2 = Dataset._filter_df_on_index(df, keys2, level)

        return df_1, df_2

    def split_on_years(self, years_split: tuple) -> tuple:
        """
        Create two new datasets based on the provided split in years

        :param years_split: tuple e.g ([2012, 2014], [2013, 2015])
        :return: two data sets
        """
        data_dfs1 = {}
        data_dfs2 = {}

        # Check existing index.
        # TODO: There might be a better way to do this.
        index_years = self.years
        years_split = (
            list(set(years_split[0]).intersection(index_years)),
            list(set(years_split[1]).intersection(index_years)),
        )

        for x in self._dfs_x:
            src_df = self._dfs_x[x]
            n_levels = len(src_df.index.names)
            if (n_levels) >= 2:
                src_df_1, src_df_2 = self._split_df_on_index(
                    src_df, years_split, level=1
                )
            else:
                src_df_1 = src_df.copy()
                src_df_2 = src_df.copy()
            data_dfs1[x] = src_df_1
            data_dfs2[x] = src_df_2

        df_y_1, df_y_2 = self._split_df_on_index(self._df_y, years_split, level=1)
        return (
            Dataset(
                self.cfg,
                df_y=df_y_1,
                dfs_x=data_dfs1,
            ),
            Dataset(
                self.cfg,
                df_y=df_y_2,
                dfs_x=data_dfs2,
            ),
        )


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

    def split_on_years(self, years_split: Tuple[list, list]) -> Tuple["PandasDataset", "PandasDataset"]:
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
    def years(self) -> set:
        return set(self.y.index.get_level_values(KEY_YEAR))

    @property
    def location_ids(self) -> set:
        return set(self.y.index.get_level_values(KEY_LOC))

    @property
    def targets(self) -> np.ndarray:
        return self.y[KEY_TARGET].to_numpy()

    @property
    def feature_names(self) -> list:
        return self.x.columns.tolist()

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