"""Tests for tabular NaN handling in DataFactory."""

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize
from omegaconf import open_dict

from cybench.config import KEY_LOC, KEY_YEAR, KEY_TARGET
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import PandasDataset


def test_build_pandas_dataset_without_temporal_sources():
    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="default",
            overrides=[
                "crop=maize",
                "country=NL",
                "framework=pandas",
                "target.filter_samples=null",
                "use_cache=false",
            ],
        )
    with open_dict(cfg.temporal):
        cfg.temporal.sources.clear()

    dataset = DataFactory(cfg).build()
    assert isinstance(dataset, PandasDataset)
    assert len(dataset) > 0
    assert len(dataset.x.columns) > 0


def test_drop_sparse_tabular_columns_removes_high_nan_windows():
    index = pd.MultiIndex.from_product(
        [["A", "B"], [2018, 2019]],
        names=[KEY_LOC, KEY_YEAR],
    )
    x = pd.DataFrame(
        {
            "gdd_sum_0": [1.0, 2.0, 3.0, 4.0],
            "gdd_sum_1": [1.0, 2.0, 3.0, 4.0],
            "ndvi_mean_8": [np.nan, np.nan, 1.0, 2.0],
            "latitude": [10.0, 11.0, 12.0, 13.0],
        },
        index=index,
    )

    filtered, dropped = DataFactory._drop_sparse_tabular_columns(x, max_nan_rate=0.05)

    assert "ndvi_mean_8" in dropped
    assert "gdd_sum_0" in filtered.columns
    assert "latitude" in filtered.columns
    assert not bool(np.any(filtered.isna().to_numpy()))


def test_drop_sparse_tabular_columns_keeps_all_when_below_threshold():
    index = pd.MultiIndex.from_product([["A", "B"], [2018]], names=[KEY_LOC, KEY_YEAR])
    x = pd.DataFrame({"feat_0": [1.0, 2.0], "feat_1": [1.0, np.nan]}, index=index)

    filtered, dropped = DataFactory._drop_sparse_tabular_columns(x, max_nan_rate=0.5)

    assert dropped == []
    assert list(filtered.columns) == ["feat_0", "feat_1"]
