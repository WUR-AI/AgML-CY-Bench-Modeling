import os
from functools import reduce

import numpy as np
import pandas as pd

from cybench.config import DatasetConfig, DATASETS, PATH_DATA_DIR, KEY_LOC, KEY_YEAR, KEY_TARGET
from cybench.datasets.alignment import compute_crop_season_window, ensure_same_categories_union, \
    align_to_crop_season_window_numpy, restore_category_to_string, align_to_crop_season_window, align_inputs_and_labels
from cybench.datasets.dataset import Dataset


class DataFactory:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

        # test
        assert self.cfg.crop in DATASETS, f"Crop type '{self.cfg.crop}' is not supported. See DATASETS in config.py"
        assert self.cfg.country in DATASETS[self.cfg.crop], f"Country '{self.cfg.country}' is not supported for crop type '{self.cfg.crop}'. See DATASETS in config.py"

    def build(self) -> Dataset:
        if isinstance(self.cfg.country, list):
            df_y = pd.DataFrame()
            dfs_x = {}
            for country in self.cfg.country:
                df_y_cn, dfs_x_cn = self.load_dfs(crop=self.cfg.crop, country_code=country)

                df_y = pd.concat([df_y, df_y_cn], axis=0)

                if len(dfs_x) == 0:
                    dfs_x = dfs_x_cn
                else:
                    for x, df_temporal_cn in dfs_x_cn.items():
                        dfs_x[x] = pd.concat([dfs_x_cn[x], df_temporal_cn], axis=0)
        else:
            df_y, dfs_x = self.load_dfs(crop=self.cfg.crop, country_code=self.cfg.country)

        return Dataset(
            cfg=self.cfg,
            df_y=df_y,
            dfs_x=dfs_x,
        )

    def load_dfs(self,
                 crop: str,
                 country_code: str,
                 use_memory_optimization: bool = True) -> tuple:
        """Load data from CSV files for crop and country.
        Expects CSV files in PATH_DATA_DIR/<crop>/<country_code>/.

        Args:
            crop (str): crop name
            country_code (str): 2-letter country code
            use_memory_optimization (bool): use (slower) memory-optimized function for crop season alignment

        Returns:
            a tripel (target DataFrame, dataframe of non-temporal data, dict of input DataFrames)
        """

        # targets
        df_y = self.load_target(crop=crop, country_code=country_code)

        # non-temporal
        df_non_temporal = self.load_non_temporal(crop=crop, country_code=country_code)

        # temporal
        dfs_x = self.load_temporal(crop=crop, country_code=country_code)

        dfs_x["non_temporal"] = df_non_temporal
        df_y, dfs_temporal = align_inputs_and_labels(df_y, dfs_x)

        return df_y, dfs_x

    def load_target(self, crop: str, country_code: str):
        path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

        df_y = pd.read_csv(
            os.path.join(path_data_cn, "_".join(["yield", crop, country_code]) + ".csv"),
            header=0,
        )
        # TODO delete next line after fixing yield data filtering based on sophisticates conditions in /data_preparation. Yield data should be analysis ready
        df_y = df_y[df_y.harvest_area > 0]
        df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
        df_y = df_y[[KEY_LOC, KEY_YEAR, KEY_TARGET]]
        df_y = df_y.dropna(axis=0)
        assert not df_y.empty, "Yield data is empty."

        df_y.set_index([KEY_LOC, KEY_YEAR], inplace=True)

        if self.cfg.target.transform == "log":
            df_y[KEY_TARGET] = np.log(df_y[KEY_TARGET] + 1e-3)

        if self.cfg.target.residualize:
            raise NotImplementedError("TODO")
        return df_y

    def load_non_temporal(self, crop: str, country_code: str):
        path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

        df_ls = []
        for file_name, values in self.cfg.non_temporal.sources.items():
            df_x = pd.read_csv(
                os.path.join(path_data_cn, "_".join([file_name, crop, country_code]) + ".csv"),
                usecols=[KEY_LOC] + values.features,
                header=0,
            )
            if "transfrom" in values:
                raise NotImplementedError("TODO")
            df_ls.append(df_x)
        non_temp_df = reduce(lambda x, y: pd.merge(x, y, on=KEY_LOC), df_ls)
        non_temp_df.set_index([KEY_LOC], inplace=True)
        return non_temp_df

    def load_temporal(self, crop: str, country_code: str):
        path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

        # crop calendar
        df_crop_cal = pd.read_csv(
            os.path.join(
                path_data_cn, "_".join(["crop_calendar", crop, country_code]) + ".csv"
            ),
            header=0,
        )
        df_crop_cal = compute_crop_season_window(
            df=df_crop_cal,
            min_year=self.cfg.min_year,
            max_year=self.cfg.max_year,
            start_of_sequence=self.cfg.temporal.season.start_of_sequence,
            end_of_sequence=self.cfg.temporal.season.end_of_sequence,
        )

        dfs_x = {}
        for file_name, source_cfg in self.cfg.temporal.sources.items():
            df_ts = self.load_and_preprocess_time_series_data(
                crop=crop,
                country_code=country_code,
                file_name=file_name,
                source_cfg=source_cfg,
                crop_season_df=df_crop_cal,
                use_memory_optimization=True,
            )
            dfs_x[file_name] = df_ts
        return dfs_x

    def load_and_preprocess_time_series_data(
            self,
            crop,
            country_code,
            file_name,
            source_cfg,
            crop_season_df,
            use_memory_optimization=False,
            verbose=False,
    ):
        """A helper function to load and preprocess time series data.

        Args:
            crop (str): crop name
            country_code (str): 2-letter country code
            ts_input (str): time series input (used to name data file)
            index_cols (list): columns used as index
            ts_cols (list): columns with time series variables
            crop_season_df (pd.DataFrame): crop calendar data
            use_memory_optimization (bool): use (slower) memory-optimized function for crop season alignment
            verbose (bool): output detailed processing information.


        Returns:
            the same DataFrame after preprocessing and aligning to crop season
        """
        index_cols = [KEY_LOC, KEY_YEAR] + ["date"]

        path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)
        file_path = os.path.join(path_data_cn, "_".join([file_name, crop, country_code]) + ".csv")
        if verbose:
            print(f'load {file_path}')
        df_ts = pd.read_csv(file_path, header=0)
        df_ts["date"] = pd.to_datetime(df_ts["date"], format="%Y%m%d")
        df_ts[KEY_YEAR] = df_ts["date"].dt.year
        df_ts = df_ts[index_cols + source_cfg.features]

        if use_memory_optimization:
            crop_season_keys = {
                (loc, year): idx
                for loc, year, idx in zip(
                    crop_season_df[KEY_LOC], crop_season_df[KEY_YEAR], crop_season_df.index
                )
            }
            df_ts, crop_season_df = ensure_same_categories_union(df_ts, crop_season_df)
            keep_mask, years = align_to_crop_season_window_numpy(
                df_ts[KEY_LOC].values,
                df_ts[KEY_YEAR].values,
                df_ts["date"].values,
                crop_season_keys,
                crop_season_df["sos_date"].values,
                crop_season_df["eos_date"].values,
                crop_season_df["start_of_sequence_date"].values,
                crop_season_df["end_of_sequence_date"].values,
            )
            assert len(keep_mask) == len(df_ts)
            df_ts[KEY_YEAR] = years
            df_ts = df_ts.loc[keep_mask]
            df_ts = restore_category_to_string(df_ts)
            df_ts.set_index(index_cols, inplace=True)
            crop_season_df = restore_category_to_string(crop_season_df)
        else:
            df_ts = align_to_crop_season_window(df_ts, crop_season_df)
            df_ts.set_index(index_cols, inplace=True)
        return df_ts




