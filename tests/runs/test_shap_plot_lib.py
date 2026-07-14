"""Tests for SHAP plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from cybench.runs.analysis.shap_plot_lib import (
    feature_rows_from_summary,
    load_feature_table,
    meta_group_shares,
    parse_feature_name,
    timing_table,
)


def test_parse_tabular_feature():
    parsed = parse_feature_name("cum_ndvi_max_8")
    assert parsed.variable_group == "cum_ndvi"
    assert parsed.statistic == "max"
    assert parsed.window == 8
    assert parsed.meta_group == "Vegetation"


def test_parse_torch_features():
    ctx = parse_feature_name("ctx:latitude")
    assert ctx.variable_group == "latitude"
    assert ctx.meta_group == "Static"
    ts = parse_feature_name("ts:ndvi")
    assert ts.variable_group == "ndvi"
    assert ts.meta_group == "Vegetation"


def test_meta_group_shares_normalize_and_aggregate():
    rows = [
        {
            "crop": "maize",
            "country": "NL",
            "model": "random_forest",
            "horizon": "eos",
            "origin": 2020,
            "feature": "cum_ndvi_max_8",
            "variable_group": "cum_ndvi",
            "statistic": "max",
            "window": 8,
            "channel": "tabular",
            "meta_group": "Vegetation",
            "mean_abs_shap": 0.6,
            "rank": 1,
        },
        {
            "crop": "maize",
            "country": "NL",
            "model": "random_forest",
            "horizon": "eos",
            "origin": 2020,
            "feature": "tmin_min_8",
            "variable_group": "tmin",
            "statistic": "min",
            "window": 8,
            "channel": "tabular",
            "meta_group": "Temperature",
            "mean_abs_shap": 0.4,
            "rank": 2,
        },
    ]
    frame = pd.DataFrame(rows)
    shares = meta_group_shares(frame)
    veg = shares.loc[shares["meta_group"] == "Vegetation"].iloc[0]
    temp = shares.loc[shares["meta_group"] == "Temperature"].iloc[0]
    assert veg["median_share_pct"] == pytest.approx(60.0)
    assert temp["median_share_pct"] == pytest.approx(40.0)


def test_timing_table_groups_windows():
    rows = [
        {
            "crop": "maize",
            "country": "NL",
            "model": "random_forest",
            "horizon": "eos",
            "origin": 2020,
            "feature": "cum_ndvi_max_8",
            "variable_group": "cum_ndvi",
            "statistic": "max",
            "window": 8,
            "channel": "tabular",
            "meta_group": "Vegetation",
            "mean_abs_shap": 0.5,
            "rank": 1,
        },
        {
            "crop": "maize",
            "country": "NL",
            "model": "random_forest",
            "horizon": "eos",
            "origin": 2020,
            "feature": "cum_ndvi_max_0",
            "variable_group": "cum_ndvi",
            "statistic": "max",
            "window": 0,
            "channel": "tabular",
            "meta_group": "Vegetation",
            "mean_abs_shap": 0.5,
            "rank": 2,
        },
    ]
    timing = timing_table(pd.DataFrame(rows))
    w8 = timing.loc[timing["window"] == 8, "median_share_pct"].iloc[0]
    w0 = timing.loc[timing["window"] == 0, "median_share_pct"].iloc[0]
    assert w8 == pytest.approx(50.0)
    assert w0 == pytest.approx(50.0)


def test_load_feature_table_from_summary_yaml(tmp_path: Path):
    summary = {
        "crop": "maize",
        "country": "NL",
        "model": "random_forest",
        "horizon": "eos",
        "origins": [
            {
                "test_years": [2020],
                "features": [
                    {"name": "cum_ndvi_max_8", "mean_abs_shap": 0.2, "rank": 1},
                ],
            }
        ],
    }
    model_dir = tmp_path / "maize_NL" / "random_forest"
    model_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(summary), model_dir / "shap_summary.yaml")

    frame = load_feature_table([model_dir / "shap_summary.yaml"])
    assert len(frame) == 1
    assert frame.iloc[0]["variable_group"] == "cum_ndvi"


def test_feature_rows_from_summary():
    summary = {
        "crop": "maize",
        "country": "DE",
        "model": "tabpfn",
        "horizon": "eos",
        "origins": [
            {
                "test_years": [2019],
                "features": [
                    {"name": "prec_sum_3", "mean_abs_shap": 0.1, "rank": 1},
                ],
            }
        ],
    }
    rows = feature_rows_from_summary(summary)
    assert rows[0]["origin"] == 2019
    assert rows[0]["variable_group"] == "prec"
