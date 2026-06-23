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
    QUALITY_KEY_COLUMNS,
    assess_yield_dataframe,
    build_yield_quality_file,
    configured_outlier_threshold,
    configured_yield_quality_settings,
    detect_outliers_with_polyfit,
    filter_samples_from_target,
    flag_consecutive_values,
    load_yield_target_config,
    merge_yield_with_quality,
    process_yield_quality_files,
    slim_quality_dataframe,
    apply_yield_quality_filter,
    viz_flag_columns,
    yield_quality_settings_from_target,
)


def test_build_yield_quality_flags_non_positive_yield(tmp_path):
    yield_file = tmp_path / "yield_maize_XX.csv"
    pd.DataFrame(
        {
            "crop_name": ["maize", "maize", "maize"],
            "country_code": ["XX", "XX", "XX"],
            KEY_LOC: ["XX01", "XX02", "XX03"],
            "harvest_year": [2020, 2021, 2022],
            KEY_TARGET: [5.0, 0.0, -1.0],
        }
    ).to_csv(yield_file, index=False)

    quality_file = tmp_path / "yield_quality_maize_XX.csv"
    build_yield_quality_file(yield_file, quality_file)

    df_q = pd.read_csv(quality_file)
    assert list(df_q.columns) == list(QUALITY_KEY_COLUMNS) + [
        FLAG_CONSECUTIVE,
        FLAG_AREA,
        FLAG_YIELD,
    ]
    assert KEY_TARGET not in df_q.columns
    merged = merge_yield_with_quality(pd.read_csv(yield_file), df_q)
    assert not bool(merged.loc[0, FLAG_YIELD])
    assert bool(merged.loc[1, FLAG_YIELD])
    assert bool(merged.loc[2, FLAG_YIELD])


def test_slim_quality_dataframe_keeps_keys_and_flags_only():
    df = pd.DataFrame(
        {
            "crop_name": ["maize"],
            "country_code": ["XX"],
            KEY_LOC: ["XX01"],
            "harvest_year": [2020],
            KEY_TARGET: [5.0],
            "harvest_area": [100.0],
            FLAG_CONSECUTIVE: [False],
            FLAG_AREA: [False],
            FLAG_YIELD: [True],
        }
    )
    slim = slim_quality_dataframe(df)
    assert list(slim.columns) == list(QUALITY_KEY_COLUMNS) + [
        FLAG_CONSECUTIVE,
        FLAG_AREA,
        FLAG_YIELD,
    ]
    assert KEY_TARGET not in slim.columns


def test_merge_yield_with_quality_supports_slim_sidecar(tmp_path):
    yield_file = tmp_path / "yield_maize_XX.csv"
    pd.DataFrame(
        {
            "crop_name": ["maize"],
            "country_code": ["XX"],
            KEY_LOC: ["XX01"],
            "harvest_year": [2020],
            KEY_TARGET: [5.0],
        }
    ).to_csv(yield_file, index=False)

    quality_file = tmp_path / "yield_quality_maize_XX.csv"
    pd.DataFrame(
        {
            "crop_name": ["maize"],
            "country_code": ["XX"],
            KEY_LOC: ["XX01"],
            "harvest_year": [2020],
            FLAG_YIELD: [True],
            FLAG_CONSECUTIVE: [False],
            FLAG_AREA: [False],
        }
    ).to_csv(quality_file, index=False)

    merged = merge_yield_with_quality(pd.read_csv(yield_file), pd.read_csv(quality_file))
    assert merged.loc[0, KEY_TARGET] == 5.0
    assert bool(merged.loc[0, FLAG_YIELD])


def test_merge_yield_with_quality_handles_duplicate_keys():
    yield_df = pd.DataFrame(
        {
            "crop_name": ["maize", "maize", "maize"],
            "country_code": ["ML", "ML", "ML"],
            KEY_LOC: ["ML01", "ML01", "ML02"],
            "harvest_year": [2020, 2020, 2021],
            KEY_TARGET: [5.0, 6.0, 7.0],
        }
    )
    quality_df = pd.DataFrame(
        {
            "crop_name": ["maize", "maize", "maize"],
            "country_code": ["ML", "ML", "ML"],
            KEY_LOC: ["ML01", "ML01", "ML02"],
            "harvest_year": [2020, 2020, 2021],
            FLAG_YIELD: [False, True, False],
            FLAG_CONSECUTIVE: [False, False, False],
            FLAG_AREA: [False, False, False],
        }
    )

    merged = merge_yield_with_quality(yield_df, quality_df)
    assert len(merged) == 3
    assert list(merged[FLAG_YIELD]) == [False, True, False]


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


def test_assess_yield_dataframe_outlier_threshold():
    years = np.arange(2010, 2020)
    yields = np.full(len(years), 5.0)
    yields[5] = 50.0
    df = pd.DataFrame({KEY_LOC: ["A1"] * len(years), "harvest_year": years, KEY_TARGET: yields})

    strict, _ = assess_yield_dataframe(df, outlier_threshold=2.0)
    lenient, _ = assess_yield_dataframe(df, outlier_threshold=10.0)

    assert bool(strict.iloc[5][FLAG_YIELD])
    assert not bool(lenient.iloc[5][FLAG_YIELD])


def test_yield_quality_settings_from_hydra_config():
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="assess_yield_quality",
            overrides=["target.quality.outlier_threshold=4.5"],
        )

    settings = yield_quality_settings_from_target(cfg)
    assert settings.outlier_threshold == 4.5
    assert settings.polyfit_degree == 2


def test_configured_yield_quality_settings_reads_yield_yaml():
    cfg = load_yield_target_config()
    quality = cfg["quality"]
    settings = configured_yield_quality_settings()
    assert settings.outlier_threshold == float(quality["outlier_threshold"])
    assert settings.polyfit_degree == int(quality["polyfit_degree"])
    assert settings.consecutive_threshold_factor == float(quality["consecutive_threshold_factor"])
    assert settings.consecutive_min_years == int(quality["consecutive_min_years"])
    assert settings.min_usable_year == int(quality["min_usable_year"])
    assert configured_outlier_threshold() == settings.outlier_threshold
    assert cfg["filter_samples"] == [FLAG_YIELD]
    assert viz_flag_columns() == [FLAG_YIELD]


def test_filter_samples_from_target_hydra_override():
    with initialize(version_base=None, config_path="../../cybench/conf/dataset"):
        cfg = compose(
            config_name="visualize_yield_quality",
            overrides=["target.filter_samples=[flag_consecutive_yield,flag_area_outlier]"],
        )
    assert filter_samples_from_target(cfg) == [FLAG_CONSECUTIVE, FLAG_AREA]
    assert viz_flag_columns(cfg) == [FLAG_CONSECUTIVE, FLAG_AREA]


def test_assess_yield_dataframe_yield_flag_split():
    df = pd.DataFrame(
        {
            "crop_name": ["maize", "maize", "maize"],
            "country_code": ["XX", "XX", "XX"],
            KEY_LOC: ["A1", "A1", "A2"],
            "harvest_year": [2020, 2021, 2020],
            KEY_TARGET: [5.0, 0.0, 50.0],
        }
    )
    years = np.arange(2010, 2020)
    spike = pd.DataFrame(
        {
            "crop_name": ["maize"] * len(years),
            "country_code": ["XX"] * len(years),
            KEY_LOC: ["B1"] * len(years),
            "harvest_year": years,
            KEY_TARGET: np.full(len(years), 5.0),
        }
    )
    spike.loc[5, KEY_TARGET] = 50.0
    df = pd.concat([df, spike], ignore_index=True)

    _, summary = assess_yield_dataframe(df, min_usable_year=None, outlier_threshold=2.0)
    assert summary is not None
    assert summary.n_yield_invalid == 1
    assert summary.n_yield_poly_outlier >= 1
    assert summary.n_yield_outlier == summary.n_yield_invalid + summary.n_yield_poly_outlier


def test_apply_yield_quality_filter_drops_flagged_rows(tmp_path):
    data_dir = tmp_path / "data"
    country_dir = data_dir / "maize" / "XX"
    country_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "crop_name": ["maize", "maize"],
            "country_code": ["XX", "XX"],
            KEY_LOC: ["XX01", "XX02"],
            "harvest_year": [2020, 2021],
            FLAG_YIELD: [True, False],
            FLAG_CONSECUTIVE: [False, False],
            FLAG_AREA: [False, False],
        }
    ).to_csv(country_dir / "yield_quality_maize_XX.csv", index=False)

    preds = pd.DataFrame(
        {
            KEY_LOC: ["XX01", "XX02"],
            KEY_YEAR: [2020, 2021],
            KEY_TARGET: [5.0, 6.0],
            "Ridge": [4.8, 5.9],
        }
    )
    filtered, n_removed = apply_yield_quality_filter(
        preds,
        "maize",
        "XX",
        data_dir=data_dir,
        quality_flags=[FLAG_YIELD],
    )
    assert n_removed == 1
    assert len(filtered) == 1
    assert filtered.iloc[0][KEY_LOC] == "XX02"


def test_run_yield_quality_visualizations(tmp_path):
    from data_preparation.visualize_yield_quality import run_yield_quality_visualizations

    crop, country = "wheat", "ES"
    data_root = tmp_path / crop / country
    data_root.mkdir(parents=True)
    yield_file = data_root / f"yield_{crop}_{country}.csv"
    years = np.arange(2010, 2020)
    yields = np.full(len(years), 5.0)
    yields[5] = 50.0
    pd.DataFrame(
        {
            "crop_name": [crop] * len(years),
            "country_code": [country] * len(years),
            KEY_LOC: ["ES241"] * len(years),
            "harvest_year": years,
            KEY_TARGET: yields,
        }
    ).to_csv(yield_file, index=False)

    settings = configured_yield_quality_settings()
    build_yield_quality_file(
        yield_file,
        data_root / f"yield_quality_{crop}_{country}.csv",
        settings=settings,
    )

    out_root = tmp_path / "viz"
    paths = run_yield_quality_visualizations(
        tmp_path,
        [crop],
        settings=settings,
        output_root=out_root,
        countries=[country],
        only_if_flagged=True,
    )
    assert paths
    assert (out_root / f"yield_quality_{crop}_{country}.png").is_file()


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
