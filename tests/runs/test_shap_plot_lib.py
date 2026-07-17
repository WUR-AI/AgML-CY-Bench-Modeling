"""Tests for SHAP plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from cybench.runs.analysis.shap_plot_lib import (
    chrono_window_columns,
    feature_rows_from_summary,
    load_feature_table,
    meta_group_shares,
    parse_feature_name,
    timing_table,
    window_relative_to_eos,
)


def test_parse_tabular_feature():
    parsed = parse_feature_name("cum_ndvi_max_8")
    assert parsed.variable_group == "cum_ndvi"
    assert parsed.statistic == "max"
    assert parsed.window == 8
    assert parsed.meta_group == "Vegetation"


def test_parse_onehot_drainage_class():
    parsed = parse_feature_name("drainage_class_4")
    assert parsed.raw == "drainage_class"
    assert parsed.variable_group == "drainage_class"
    assert parsed.meta_group == "Static"
    ctx = parse_feature_name("ctx:drainage_class_2")
    assert ctx.raw == "ctx:drainage_class"
    assert ctx.variable_group == "drainage_class"
    assert ctx.meta_group == "Static"


def test_feature_rows_coalesce_drainage_onehots():
    summary = {
        "crop": "maize",
        "country": "NL",
        "model": "transformer_lf",
        "horizon": "eos",
        "origins": [
            {
                "test_years": [2020],
                "features": [
                    {"name": "ctx:drainage_class_3", "mean_abs_shap": 0.1, "rank": 1},
                    {"name": "ctx:drainage_class_4", "mean_abs_shap": 0.2, "rank": 2},
                    {"name": "ctx:awc", "mean_abs_shap": 0.15, "rank": 3},
                ],
            }
        ],
    }
    rows = feature_rows_from_summary(summary)
    by_feat = {row["feature"]: row["mean_abs_shap"] for row in rows}
    assert by_feat["ctx:drainage_class"] == pytest.approx(0.3)
    assert by_feat["ctx:awc"] == pytest.approx(0.15)
    assert len(rows) == 2


def test_window_relative_to_eos_is_chronological():
    assert window_relative_to_eos(0) == 0
    assert window_relative_to_eos(1) == -1
    assert window_relative_to_eos(8) == -8
    assert chrono_window_columns([0, 2, 1, 8]) == [8, 2, 1, 0]


def test_plot_meta_group_families_keeps_ylabel(tmp_path: Path):
    from cybench.runs.analysis.shap_plot_lib import plot_meta_group_families

    shares = pd.DataFrame(
        [
            {
                "crop": "maize",
                "model": "random_forest",
                "horizon": "eos",
                "meta_group": "Vegetation",
                "median_share_pct": 60.0,
            },
            {
                "crop": "maize",
                "model": "random_forest",
                "horizon": "eos",
                "meta_group": "Temperature",
                "median_share_pct": 40.0,
            },
            {
                "crop": "maize",
                "model": "transformer_lf",
                "horizon": "eos",
                "meta_group": "Vegetation",
                "median_share_pct": 30.0,
            },
            {
                "crop": "maize",
                "model": "transformer_lf",
                "horizon": "eos",
                "meta_group": "Temperature",
                "median_share_pct": 70.0,
            },
        ]
    )
    out = tmp_path / "meta.png"
    plot_meta_group_families(
        shares,
        crop="maize",
        horizon="eos",
        models=["random_forest", "transformer_lf"],
        output_path=out,
    )
    assert out.is_file()


def test_plot_timing_heatmaps_uses_relative_chrono_axis(tmp_path: Path, monkeypatch):
    from cybench.runs.analysis import shap_plot_lib as lib

    captured: dict = {}

    def fake_heatmap(data, ax=None, **kwargs):
        captured["columns"] = list(data.columns)
        if ax is not None:
            ax.imshow([[0.0]])

    monkeypatch.setattr(lib.sns, "heatmap", fake_heatmap)
    timing = pd.DataFrame(
        [
            {
                "crop": "maize",
                "model": "random_forest",
                "horizon": "eos",
                "variable_group": "ndvi",
                "window": w,
                "median_share_pct": 10.0 + w,
            }
            for w in (0, 1, 2)
        ]
    )
    out = tmp_path / "timing.png"
    lib.plot_timing_heatmaps(
        timing,
        crop="maize",
        horizon="eos",
        models=["random_forest"],
        output_path=out,
    )
    assert captured["columns"] == [-2, -1, 0]
    assert out.is_file()


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
