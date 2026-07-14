"""Tests for SHAP importance helpers (no SHAP runtime required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from cybench.runs.analysis.shap_importance_lib import (
    _aggregate_shapiq_first_order,
    _mean_abs_feature_importance,
    _rank_features,
    aggregate_feature_importance,
    find_saved_model_artifact,
    find_screening_split_dir,
    find_walk_forward_run_dir,
    interpretability_for_model,
    model_run_name,
    resolve_shap_sample_limits,
)


def test_mean_abs_feature_importance_squeezes_output_dim():
    # GradientExplainer often returns (n_eval, n_feat, 1) for a scalar head.
    shap_ctx = np.random.rand(80, 7, 1)
    out = _mean_abs_feature_importance(shap_ctx, n_features=7)
    assert out.shape == (7,)


def test_mean_abs_feature_importance_temporal():
    shap_ts = np.random.rand(80, 24, 10)
    out = _mean_abs_feature_importance(shap_ts, n_features=10)
    assert out.shape == (10,)


def test_rank_features_from_2d_mean_vector():
    names = ["a", "b", "c"]
    mean_abs = np.array([[0.1, 0.5, 0.2]])
    rows = _rank_features(names, mean_abs)
    assert rows[0]["name"] == "b"


def test_aggregate_shapiq_first_order():
    class _FakeExplanation:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def get_n_order_values(self, order: int) -> list[float]:
            assert order == 1
            return self._values

    explanations = [
        _FakeExplanation([0.4, 0.1, 0.2]),
        _FakeExplanation([0.6, 0.0, 0.1]),
    ]
    out = _aggregate_shapiq_first_order(explanations, n_features=3)
    assert out.tolist() == pytest.approx([0.5, 0.05, 0.15])


def test_interpretability_for_model_families():
    assert interpretability_for_model("random_forest")["method"] == "tree_shap"
    assert interpretability_for_model("random_forest")["explainer_label"] == "TreeSHAP"
    assert interpretability_for_model("tabpfn")["method"] == "tabpfn_shapley"
    assert interpretability_for_model("tabpfn")["explainer_label"] == "TabPFNShapley"
    assert interpretability_for_model("transformer_lf")["method"] == "gradient_shap"
    assert interpretability_for_model("tabicl")["method"] == "sklearn_permutation"
    unknown = interpretability_for_model("xgboost")
    assert unknown["method"] == "permutation_shap"


def test_resolve_shap_sample_limits_tabpfn():
    bg, ev = resolve_shap_sample_limits(
        "tabpfn", max_background=50, max_eval_samples=80
    )
    assert bg == 25
    assert ev == 20
    bg_rf, ev_rf = resolve_shap_sample_limits(
        "random_forest", max_background=50, max_eval_samples=80
    )
    assert bg_rf == 50
    assert ev_rf == 80


def test_model_run_name_random_forest():
    assert model_run_name("random_forest") == "random_forest"


def test_find_saved_model_artifact_torch_and_sklearn(tmp_path: Path):
    wf = tmp_path / "wf"
    rep = wf / "2017" / "42"
    rep.mkdir(parents=True)
    (rep / "transformer_lf.pt").write_bytes(b"pt")
    (rep / "random_forest.pkl").write_bytes(b"pkl")

    assert find_saved_model_artifact(
        wf, test_year=2017, seed=42, model_name="transformer_lf"
    ) == rep / "transformer_lf.pt"
    assert find_saved_model_artifact(
        wf, test_year=2017, seed=42, model_name="random_forest"
    ) == rep / "random_forest.pkl"
    assert find_saved_model_artifact(
        wf, test_year=2018, seed=42, model_name="transformer_lf"
    ) is None
    assert find_saved_model_artifact(
        None, test_year=2017, seed=42, model_name="transformer_lf"
    ) is None


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
