"""Tests for yield quality flag generation and loading."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from hydra import compose, initialize

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR, PATH_DATA_DIR
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.yield_quality import (
    FLAG_AREA,
    FLAG_CONSECUTIVE,
    FLAG_YIELD,
    assess_yield_dataframe,
    build_yield_quality_file,
    detect_outliers_with_polyfit,
    flag_consecutive_values,
)


def test_build_yield_quality_flags_non_positive_yield(tmp_path):
    yield_file = tmp_path / "yield_maize_XX.csv"
    pd.DataFrame(
        {
            "crop_name": ["maize", "maize", "maize"],
            KEY_LOC: ["XX01", "XX02", "XX03"],
            "harvest_year": [2020, 2021, 2022],
            KEY_TARGET: [5.0, 0.0, -1.0],
        }
    ).to_csv(yield_file, index=False)

    quality_file = tmp_path / "yield_quality_maize_XX.csv"
    build_yield_quality_file(yield_file, quality_file)

    df_q = pd.read_csv(quality_file)
    assert list(df_q.columns[-3:]) == [FLAG_CONSECUTIVE, FLAG_AREA, FLAG_YIELD]
    assert not bool(df_q.loc[0, FLAG_YIELD])
    assert bool(df_q.loc[1, FLAG_YIELD])
    assert bool(df_q.loc[2, FLAG_YIELD])


def test_flag_consecutive_values_detects_linear_trend():
    years = np.arange(2010, 2022)
    group = pd.DataFrame(
        {
            "harvest_year": years,
            KEY_TARGET: np.linspace(3.0, 8.0, len(years)),
        }
    )
    flags = flag_consecutive_values(group, min_consecutive=5)
    assert flags.all()


def test_flag_consecutive_values_keeps_variable_series():
    group = pd.DataFrame(
        {
            "harvest_year": [2018, 2019, 2020, 2021, 2022, 2023],
            KEY_TARGET: [4.0, 6.5, 3.2, 7.1, 4.8, 6.0],
        }
    )
    flags = flag_consecutive_values(group, min_consecutive=5)
    assert not flags.any()


def test_detect_outliers_with_polyfit_high_only():
    years = np.arange(2010, 2020)
    yields = np.full(len(years), 5.0)
    yields[5] = 50.0
    group = pd.DataFrame({"harvest_year": years, KEY_TARGET: yields})
    flags = detect_outliers_with_polyfit(
        group, KEY_TARGET, direction="high", threshold=2.0
    )
    assert flags.iloc[5]
    assert not flags.drop(index=5).any()


def test_assess_yield_dataframe_preserves_all_rows():
    df = pd.DataFrame(
        {
            KEY_LOC: ["A1", "A1", "A2"],
            "harvest_year": [2020, 2021, 2020],
            KEY_TARGET: [5.0, np.nan, 0.0],
        }
    )
    out, _ = assess_yield_dataframe(df, min_usable_year=None)
    assert len(out) == 3
    assert out.loc[1, FLAG_CONSECUTIVE]
    assert out.loc[2, FLAG_YIELD]


def test_data_factory_applies_yield_quality_filter(caplog, monkeypatch):
    import cybench.config as config
    import cybench.datasets.data_factory as data_factory_mod

    monkeypatch.setattr(config, "PATH_DATA_DIR", PATH_DATA_DIR)
    monkeypatch.setattr(data_factory_mod, "PATH_DATA_DIR", PATH_DATA_DIR)

    path_data_cn = os.path.join(PATH_DATA_DIR, "maize", "NL")
    yield_file = os.path.join(path_data_cn, "yield_maize_NL.csv")
    quality_file = os.path.join(path_data_cn, "yield_quality_maize_NL.csv")
    if not os.path.exists(quality_file):
        build_yield_quality_file(yield_file, quality_file)

    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="default",
            overrides=["crop=maize", "country=NL", "framework=pandas"],
        )

    with caplog.at_level("INFO"):
        dataset = DataFactory(cfg).build()

    assert len(dataset) > 0
    assert (dataset.targets > 0).all()
    assert any(
        "Removed" in record.message and "quality flags" in record.message
        for record in caplog.records
    )
