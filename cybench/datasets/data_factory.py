import os
from functools import reduce
import numpy as np
import pandas as pd
import torch

from cybench.config import DATASETS, PATH_DATA_DIR, KEY_LOC, KEY_YEAR, KEY_TARGET
from cybench.datasets.alignment import compute_crop_season_window, ensure_same_categories_union, \
    align_to_crop_season_window_numpy, restore_category_to_string, align_to_crop_season_window, align_inputs_and_labels, \
    interpolate_time_series_data, make_aligned_tensors
from cybench.datasets.dataset import Dataset, PandasDataset
from cybench.datasets.feature_design import FEATURE_FUNCTIONS
from cybench.datasets.feature_transformation import feature_transform
from cybench.datasets.normalizer import Normalizer
from cybench.datasets.torch_dataset import TorchDataset
from cybench.util.store_and_cache import cfg_to_hash


class DataFactory:
    def __init__(self, cfg):# DatasetConfig):
        self.cfg = cfg
        crop_name = self.cfg.crop.name

        # test
        assert crop_name in DATASETS, f"Crop type '{crop_name}' is not supported. See DATASETS in config.py"

        if isinstance(self.cfg.country, str):
            assert self.cfg.country in DATASETS[crop_name], f"Country '{self.cfg.country}' is not supported for crop type '{crop_name}'. See DATASETS in config.py"
        else:
            for c in self.cfg.country:
                assert c in DATASETS[crop_name], f"Country '{self.cfg.country}' is not supported for crop type '{crop_name}'. See DATASETS in config.py"

    def build(self) -> Dataset:
        # Caching Strategy: Check existing
        use_cache = getattr(self.cfg, 'use_cache', False)
        use_memory_optimization = getattr(self.cfg, 'use_memory_optimization', True)
        cache_dir = os.path.join(PATH_DATA_DIR, "cache")

        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            dataset_hash = cfg_to_hash(self.cfg, add_str=self.cfg.name)

            if self.cfg.framework == "torch":
                cache_path = os.path.join(cache_dir, f"{dataset_hash}.pt")
                if os.path.exists(cache_path):
                    return torch.load(cache_path, weights_only=False)

            elif self.cfg.framework == "pandas":
                cache_path = os.path.join(cache_dir, f"{dataset_hash}.pkl")
                if os.path.exists(cache_path):
                    return PandasDataset.load(cache_path)
            else:
                raise NotImplementedError(f"You try to load a cached dataset using an unknown framework: {self.cfg.framework}")

        if isinstance(self.cfg.country, str):
            df_y, dfs_x = self.load_dfs(crop=self.cfg.crop, country_code=self.cfg.country, use_memory_optimization=use_memory_optimization)
        else:
            df_y = pd.DataFrame()
            dfs_x = {}
            for country in self.cfg.country:
                df_y_cn, dfs_x_cn = self.load_dfs(crop=self.cfg.crop, country_code=country, use_memory_optimization=use_memory_optimization)

                df_y = pd.concat([df_y, df_y_cn], axis=0)

                if len(dfs_x) == 0:
                    dfs_x = dfs_x_cn
                else:
                    for x, df_temporal_cn in dfs_x_cn.items():
                        dfs_x[x] = pd.concat([dfs_x[x], df_temporal_cn], axis=0)

        normalizer = getattr(self.cfg, 'normalizer', None)
        if normalizer is not None:
            normalizer = Normalizer(self.cfg.normalizer)
            if normalizer.name == "fit":
                dfs_x = normalizer.fit_normalize(dfs_x)
                normalizer.name = "fitted"
            else:
                dfs_x = normalizer.normalize(dfs_x)

        if self.cfg.framework == "torch":
            # unifies and interpolate time-series dataframes into a single dataframe
            df_ts = interpolate_time_series_data(dfs_x)
            # align datasets and cast to torch tensors
            aligned_tensors, column_names, doy_tensor = make_aligned_tensors(
                df_y=df_y,
                df_non_temporal=dfs_x["non_temporal"],
                df_ts=df_ts,
                normalizer=normalizer
            )

            dataset = TorchDataset(
                aligned_tensors=aligned_tensors,
                doy_tensor=doy_tensor,
                column_names=column_names,
                indices=df_y.index.to_frame(index=False),
                normalizer=normalizer,
            )
            # Caching Strategy: Save result
            if use_cache:
                torch.save(dataset, cache_path)

        elif self.cfg.framework == "pandas":
            united_df_x = pd.concat(
                [self._tabularize(df_x) for name, df_x in dfs_x.items() if name != "non_temporal"],
                axis=1,
            )
            united_df_x = (
                united_df_x.reset_index()
                .merge(dfs_x["non_temporal"].reset_index(), on=KEY_LOC, how="left")
                .set_index([KEY_LOC, KEY_YEAR])
            )
            dataset = PandasDataset(
                cfg=self.cfg,
                x=united_df_x,
                y=df_y,
                normalizer=normalizer,
            )
            if use_cache:
                dataset.save(cache_path)
        else:
            dataset = Dataset(
                cfg=self.cfg,
                df_y=df_y,
                dfs_x=dfs_x,
            )

        return dataset

    def load_dfs(self,
                 crop: str,
                 country_code: str,
                 use_memory_optimization: bool = True) -> tuple:
        """Load data from CSV files for crop and country.
        Expects CSV files in PATH_DATA_DIR/<crop>/<country_code>/.

        Args:
            crop (dict): crop config
            country_code (str): 2-letter country code
            use_memory_optimization (bool): use (slower) memory-optimized function for crop season alignment

        Returns:
            a tripel (target DataFrame, dataframe of non-temporal data, dict of input DataFrames)
        """

        # targets
        df_y = self.load_target(crop=crop.name, country_code=country_code)

        # non-temporal
        df_non_temporal = self.load_non_temporal(crop=crop.name, country_code=country_code)

        # temporal
        dfs_x = self.load_temporal(crop=crop, country_code=country_code, use_memory_optimization=use_memory_optimization)

        dfs_x["non_temporal"] = df_non_temporal
        df_y, dfs_temporal = align_inputs_and_labels(df_y, dfs_x)

        return df_y, dfs_x

    def load_target(self, crop: str, country_code: str):
        path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

        if "filter_samples" in self.cfg.target.keys() and self.cfg.target["filter_samples"]:
            df_y = pd.read_csv(
                os.path.join(path_data_cn, "_".join(["yield_quality", crop, country_code]) + ".csv"),
                header=0,
            )
            df_y = df_y[~df_y[self.cfg.target["filter_samples"]].any(axis=1)]
        else:
            df_y = pd.read_csv(
                os.path.join(path_data_cn, "_".join(["yield", crop, country_code]) + ".csv"),
                header=0,
            )
        df_y = df_y.rename(columns={"harvest_year": KEY_YEAR})
        df_y = df_y[[KEY_LOC, KEY_YEAR, KEY_TARGET]]
        df_y = df_y.dropna(axis=0)
        assert not df_y.empty, f"Yield data is empty in ({country_code}, {crop})."

        df_y = df_y[(df_y.year >= self.cfg.min_year) & (df_y.year <= self.cfg.max_year)]
        df_y.set_index([KEY_LOC, KEY_YEAR], inplace=True)

        assert not df_y.isnull().values.any(), "Unexpected NaN in df_y"
        return df_y

    def load_non_temporal(self, crop: str, country_code: str):
        path_data_cn = os.path.join(PATH_DATA_DIR, crop, country_code)

        df_ls = []
        for file_name, values in self.cfg.non_temporal.sources.items():
            if file_name == "climate_vars":
                assert np.any(["climate_vars" in file_name for file_name in os.listdir(path_data_cn)]), f"Your configuration is using 'climate-variables' which are non-native to the CY-Bench data and have to be pre-processed. Execute the climate_variables_preprocess.py script."
            df_x = pd.read_csv(
                os.path.join(path_data_cn, "_".join([file_name, crop, country_code]) + ".csv"),
                usecols=[KEY_LOC] + values.select,
                header=0,
            )
            if "transform" in values:
                for transform in values.transform:
                    df_x = feature_transform(df_x, transform)
            df_ls.append(df_x)
        non_temp_df = reduce(lambda x, y: pd.merge(x, y, on=KEY_LOC), df_ls)
        non_temp_df.set_index([KEY_LOC], inplace=True)

        # fill nan values
        non_temp_df.fillna(non_temp_df.mean())
        return non_temp_df

    def load_temporal(self, crop: dict, country_code: str, use_memory_optimization=True):
        path_data_cn = os.path.join(PATH_DATA_DIR, crop.name, country_code)

        # crop calendar
        df_crop_cal = pd.read_csv(
            os.path.join(
                path_data_cn, "_".join(["crop_calendar", crop.name, country_code]) + ".csv"
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
            df_ts = self.load_and_process_time_series_data(
                crop=crop,
                country_code=country_code,
                file_name=file_name,
                source_cfg=source_cfg,
                aggregate=getattr(self.cfg.temporal, 'aggregate', None),
                crop_season_df=df_crop_cal,
                use_memory_optimization=use_memory_optimization,
            )

            assert not df_ts.isnull().values.any(), f"Unexpected NaN in df_ts ({file_name})"
            dfs_x[file_name] = df_ts
        return dfs_x

    def load_and_process_time_series_data(
            self,
            crop,
            country_code,
            file_name,
            source_cfg,
            aggregate,
            crop_season_df,
            use_memory_optimization=True,
            verbose=False,
    ):
        """A helper function to load and preprocess time series data.

        Args:
            crop (dict): crop configuration
            country_code (str): 2-letter country code
            file_name (str): file name based on data source. E.g. meteo, soil_moisture ...
            source_cfg (dict): source configuration that determines which feature to load, create and aggregate
            aggregate (int): number of days that are aggregated to one feature
            crop_season_df (pd.DataFrame): crop calendar data
            use_memory_optimization (bool): use (slower) memory-optimized function for crop season alignment
            verbose (bool): output detailed processing information.


        Returns:
            the same DataFrame after preprocessing and aligning to crop season
        """
        index_cols = [KEY_LOC, KEY_YEAR] + ["date"]

        path_data_cn = os.path.join(PATH_DATA_DIR, crop.name, country_code)
        file_path = os.path.join(path_data_cn, "_".join([file_name, crop.name, country_code]) + ".csv")
        if verbose:
            print(f'load {file_path}')
        df_ts = pd.read_csv(file_path, header=0)
        df_ts["date"] = pd.to_datetime(df_ts["date"], format="%Y%m%d")
        df_ts[KEY_YEAR] = df_ts["date"].dt.year
        df_ts = df_ts[index_cols + source_cfg.select]

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
            crop_season_df = restore_category_to_string(crop_season_df)
        else:
            df_ts = align_to_crop_season_window(df_ts, crop_season_df)

        if hasattr(source_cfg, 'create') and source_cfg.create:
            df_ts = self._create_features(df_ts, source_cfg.create, self.cfg.crop)

        if aggregate is not None:
            df_ts = self._aggregate_time_series(df_ts, source_cfg.aggregate, aggregate)
        else:
            df_ts.set_index(index_cols, inplace=True)
        return df_ts


    @staticmethod
    def _create_features(
            df_ts: pd.DataFrame,
            create_cfg: list,
            crop_params,
    ) -> pd.DataFrame:
        """Derive new columns in config order before aggregation.

        Entries execute sequentially so later ones can reference columns
        produced by earlier ones (e.g. cum_gdd safely references gdd).

        Args:
            df_ts:       flat DataFrame [KEY_LOC, KEY_YEAR, date, <selected cols>]
            create_cfg:  list of OmegaConf nodes, each with {name, type, input}
            crop_params: OmegaConf node from crops/<crop>.yaml
        """
        group_keys = [KEY_LOC, KEY_YEAR]

        for entry in create_cfg:
            if entry.type not in FEATURE_FUNCTIONS:
                available = ', '.join(FEATURE_FUNCTIONS)
                raise ValueError(
                    f"Feature type '{entry.type}' not found. "
                    f"Available: {available}. "
                    f"Add it to FEATURE_FUNCTIONS in feature_design.py."
                )
            fn = FEATURE_FUNCTIONS[entry.type]
            df_ts[entry.name] = fn(df_ts, entry.input, group_keys, crop_params)

        return df_ts

    @staticmethod
    def _aggregate_time_series(
            df_ts: pd.DataFrame,
            agg_function: dict,
            aggregate: int,
    ) -> pd.DataFrame:
        """Aggregate a time series into fixed N-day windows per (loc, year).

        Windows are EOS-anchored: the last interval always ends at EOS.
        Shorter seasons simply produce fewer leading windows,
        so the late-season signal is always preserved when features are aligned.

        Args:
            df_ts: DataFrame.
            agg_function: dict mapping column -> str | list[str] (OmegaConf-safe).
            aggregate: window size in days.

        Returns:
            Aggregated DataFrame with MultiIndex structure.
        """
        group_keys = [KEY_LOC, KEY_YEAR]

        season_end = (
            df_ts.groupby(group_keys, observed=True)["date"]
            .transform("max")
        )
        day_offset = (season_end - df_ts["date"]).dt.days  # 0 at EOS, grows backwards
        window_idx = day_offset // aggregate
        df_ts["date"] = season_end - pd.to_timedelta(window_idx * aggregate, unit="D")

        # Split agg_function into single-fn and multi-fn columns
        single_agg: dict[str, str] = {}  # col -> "mean" | "sum" | …
        multi_agg: dict[str, list] = {}  # col -> ["sum", "max", …]

        for col, fn in agg_function.items():
            if isinstance(fn, str):
                single_agg[col] = fn
            else:  # list / ListConfig
                multi_agg[col] = list(fn)

        grouped = df_ts.groupby(group_keys + ["date"], observed=True, sort=True)

        parts: list[pd.DataFrame] = []

        if single_agg:
            # Returns a DataFrame with MultiIndex = index_cols, cols = feature cols
            agg_result = grouped[list(single_agg.keys())].agg(single_agg)
            agg_result.columns = [f"{col}_{fn}" for col, fn in single_agg.items()]
            parts.append(agg_result)
        for col, fns in multi_agg.items():
            # agg on a SeriesGroupBy returns a DataFrame when given a list
            agg_result = grouped[col].agg(fns)
            agg_result.columns = [f"{col}_{fn}" for fn in fns]
            parts.append(agg_result)

        df_agg = parts[0]
        for part in parts[1:]:
            df_agg = df_agg.join(part, how="left")

        # index is already [KEY_LOC, KEY_YEAR, "date"] — matches original contract
        return df_agg


    @staticmethod
    def _tabularize(df_ts: pd.DataFrame) -> pd.DataFrame:
        """Pivot aggregated time series into one flat row per (adm_id, year).

        Window index is EOS-anchored: 0 = last window (EOS), 1 = one before, …
        Column names follow the pattern <feature>_0, <feature>_1, …
        adm_ids with shorter seasons produce NaN in their earliest columns.

        Args:
            df_ts: aggregated DataFrame with MultiIndex [KEY_LOC, KEY_YEAR, date]

        Returns:
            Flat DataFrame with MultiIndex [KEY_LOC, KEY_YEAR]
        """
        df = df_ts.reset_index()

        # Rank windows per (loc, year): 0 = EOS (latest date), 1 = one before, ...
        # rank(ascending=False) gives 1 to the largest date, so subtract 1 for 0-based.
        # Using method='dense' ensures no gaps even if dates are irregular.
        group_keys = [KEY_LOC, KEY_YEAR]
        df["window"] = (
            df.groupby(group_keys, observed=True)["date"]
            .rank(method="dense", ascending=False)
            .astype(int) - 1
        )

        feature_cols = [c for c in df.columns if c not in group_keys + ["date", "window"]]

        # pivot_table handles missing windows per adm_id gracefully — NaN fills in.
        pivoted = df.pivot_table(
            index=group_keys,
            columns="window",
            values=feature_cols,
            aggfunc="first",  # each (loc, year, window) is already unique
            observed=True,
        )

        # Flatten MultiIndex columns: (ndvi_mean, 0) -> ndvi_mean_0
        pivoted.columns = [f"{col}_{w}" for col, w in pivoted.columns]

        return pivoted
