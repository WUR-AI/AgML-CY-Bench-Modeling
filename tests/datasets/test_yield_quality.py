"""Tests for yield quality flag generation and loading."""

import pandas as pd
from hydra import compose, initialize

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.data_factory import DataFactory
from data_preparation.assess_yield_quality import build_yield_quality_file


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
    build_yield_quality_file(str(yield_file), str(quality_file))

    df_q = pd.read_csv(quality_file)
    assert list(df_q.columns[-3:]) == [
        "flag_consecutive_yield",
        "flag_area_outlier",
        "flag_yield_outlier",
    ]
    assert not bool(df_q.loc[0, "flag_yield_outlier"])
    assert bool(df_q.loc[1, "flag_yield_outlier"])
    assert bool(df_q.loc[2, "flag_yield_outlier"])


def test_data_factory_applies_yield_quality_filter(caplog):
    import cybench.config as config
    import cybench.datasets.data_factory as data_factory_mod

    config.PATH_DATA_DIR = "cybench/testdata"
    data_factory_mod.PATH_DATA_DIR = "cybench/testdata"

    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="default",
            overrides=["crop=maize", "country=NL", "framework=pandas"],
        )

    with caplog.at_level("INFO"):
        dataset = DataFactory(cfg).build()

    assert len(dataset) > 0
    assert (dataset.y[KEY_TARGET] > 0).all()
    assert any("Removed" in record.message and "quality flags" in record.message for record in caplog.records)
