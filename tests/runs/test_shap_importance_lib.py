"""Tests for SHAP importance helpers (no SHAP runtime required)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from cybench.runs.analysis.shap_importance_lib import (
    aggregate_feature_importance,
    find_screening_split_dir,
    find_walk_forward_run_dir,
    model_run_name,
)


def test_model_run_name_random_forest():
    assert model_run_name("random_forest") == "random_forest"


def test_find_screening_and_walk_forward_dirs(tmp_path: Path):
    baselines = tmp_path / "baselines_NL_eos_v2"
    baselines.mkdir()
    screen = baselines / "maize_NL_random_forest_screening_eos_20260626_000900"
    split = screen / "2016_2017_2018_2019_2020"
    split.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"name": "random_forest"}), split / "optimal_model.yaml")

    wf = baselines / "maize_NL_random_forest_walk_forward_eos_20260626_001038"
    wf.mkdir()
    (wf / "2020").mkdir()

    found_screen = find_screening_split_dir(
        baselines, crop="maize", country="NL", model_slug="random_forest", horizon="eos"
    )
    assert found_screen == split

    found_wf = find_walk_forward_run_dir(
        baselines, crop="maize", country="NL", model_slug="random_forest", horizon="eos"
    )
    assert found_wf == wf


def test_aggregate_feature_importance_median_across_origins():
    records = [
        {
            "model": "random_forest",
            "test_years": [2019],
            "features": [
                {"name": "prec_sum_1", "mean_abs_shap": 0.4, "rank": 1},
                {"name": "gdd_sum_2", "mean_abs_shap": 0.2, "rank": 2},
            ],
        },
        {
            "model": "random_forest",
            "test_years": [2020],
            "features": [
                {"name": "prec_sum_1", "mean_abs_shap": 0.6, "rank": 1},
                {"name": "gdd_sum_2", "mean_abs_shap": 0.1, "rank": 2},
            ],
        },
    ]
    agg = aggregate_feature_importance(records)
    assert list(agg.columns) == [
        "model",
        "feature",
        "median_mean_abs_shap",
        "mean_rank",
        "n_origins",
        "aggregate_rank",
    ]
    prec = agg.loc[agg["feature"] == "prec_sum_1"].iloc[0]
    assert prec["median_mean_abs_shap"] == pytest.approx(0.5)
    assert int(prec["n_origins"]) == 2


def test_find_screening_split_dir_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_screening_split_dir(
            tmp_path,
            crop="maize",
            country="NL",
            model_slug="random_forest",
            horizon="eos",
        )
