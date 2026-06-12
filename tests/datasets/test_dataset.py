import os
from typing import cast

import pandas as pd
import pytest
from hydra import compose, initialize
import copy
from omegaconf import open_dict

from cybench.datasets.dataset import PandasDataset
from cybench.datasets.data_factory import DataFactory
from cybench.config import (
    PATH_DATA_DIR,
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    # SOIL_PROPERTIES,
    # LOCATION_PROPERTIES,
    # METEO_INDICATORS,
    # RS_FPAR,
    # RS_NDVI,
    # SOIL_MOISTURE_INDICATORS,
)

@pytest.fixture
def dataset_cfg():
    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="default",
            overrides=[
                "crop=maize",
                "country=NL",
                "framework=pandas",
                "target.filter_samples=null",
                # 'temporal.sources.meteo.select=["tmin","tmax","tavg","prec","rad","cwb"]',
            ],
        )
    return cfg


@pytest.fixture
def dataset(dataset_cfg):
    return DataFactory(dataset_cfg).build()


def test_dataset_item(dataset):
    assert isinstance(dataset, PandasDataset)
    assert len(dataset) > 0

    x, y = dataset.xy
    assert KEY_TARGET in y.columns
    assert len(x) == len(y)

    temporal_cols = [
        col
        for source in dataset.cfg.temporal.sources.values()
        for col in source.select
    ]
    for col in temporal_cols:
        tabular_cols = [c for c in x.columns if c.startswith(f"{col}_")]
        assert tabular_cols, f"No tabularized columns for temporal feature {col}"
        assert bool(x[tabular_cols].notna().to_numpy().any())


def test_split():
    data_path_county_features = os.path.join(PATH_DATA_DIR, "features", "maize", "US")
    train_csv = os.path.join(data_path_county_features, "grain_maize_US_train.csv")
    train_df = pd.read_csv(train_csv, index_col=[KEY_LOC, KEY_YEAR])
    train_yields = cast(pd.DataFrame, train_df[[KEY_TARGET]].copy())
    feature_cols = [c for c in train_df.columns if c != KEY_TARGET]
    train_features = cast(pd.DataFrame, train_df.loc[:, feature_cols].copy())
    dataset_cv = PandasDataset(
        cfg=None,
        y=train_yields,
        x=train_features,
    )

    even_years = {x for x in dataset_cv.years if x % 2 == 0}
    odd_years = dataset_cv.years - even_years

    ds1, ds2 = dataset_cv.split_on_years((even_years, odd_years))
    assert ds1.years == even_years
    assert ds2.years == odd_years


def _load_dfs_multi_country(factory: DataFactory, cfg, countries: list[str]):
    """Mirror DataFactory.build() multi-country concat without tabularization."""
    df_y = pd.DataFrame()
    dfs_x = {}
    for country in countries:
        df_y_cn, dfs_x_cn = factory.load_dfs(
            crop=cfg.crop,
            country_code=country,
        )
        df_y = pd.concat([df_y, df_y_cn], axis=0)
        if not dfs_x:
            dfs_x = dfs_x_cn
        else:
            for name, df in dfs_x_cn.items():
                dfs_x[name] = pd.concat([dfs_x[name], df], axis=0)
    return df_y, dfs_x


def test_load_dfs_crop(dataset_cfg):
    """Every yield (loc, year) must be present in all aligned feature tables."""
    factory = DataFactory(dataset_cfg)
    df_y, dfs_x = _load_dfs_multi_country(factory, dataset_cfg, ["NL", "ES"])

    df_y.sort_index(inplace=True)
    for name in dfs_x:
        dfs_x[name] = dfs_x[name].sort_index()

    for loc_year in df_y.index:
        for df_x in dfs_x.values():
            if len(df_x.index.names) == 1:
                assert loc_year[0] in df_x.index
            else:
                assert loc_year in df_x.index


def test_load(dataset_cfg):
    cfg1 = copy.deepcopy(dataset_cfg)
    with open_dict(cfg1):
        cfg1.country = "NL"
    ds1 = DataFactory(cfg1).build()

    cfg2 = copy.deepcopy(dataset_cfg)
    with open_dict(cfg2):
        cfg2.country = "ES"
    ds2 = DataFactory(cfg2).build()

    cfg3 = copy.deepcopy(dataset_cfg)
    with open_dict(cfg3):
        cfg3.country = ["NL", "ES"]
    ds3 = DataFactory(cfg3).build()

    assert len(ds3) == len(ds1) + len(ds2)


def test_memory_optimization(dataset_cfg):
    cfg = copy.deepcopy(dataset_cfg)
    with open_dict(cfg):
        cfg.country = "ES"
        cfg.use_memory_optimization = False
    dataset_no_optimization = cast(PandasDataset, DataFactory(cfg).build())
    y_no_optimization, x_no_optimization = dataset_no_optimization.y, dataset_no_optimization.x

    with open_dict(cfg):
        cfg.country = "ES"
        cfg.use_memory_optimization = True
    dataset_memory_optimized = cast(PandasDataset, DataFactory(cfg).build())
    y_memory_optimized, x_memory_optimized = dataset_memory_optimized.y, dataset_memory_optimized.x
    assert y_no_optimization.equals(y_memory_optimized)

    for col in x_no_optimization.columns:
        assert x_no_optimization[col].equals(x_memory_optimized[col])
