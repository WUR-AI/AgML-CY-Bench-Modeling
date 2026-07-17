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
    _sum_abs_over_time_feature_importance,
    MODEL_MANIFEST,
    aggregate_feature_importance,
    coalesce_onehot_feature_name,
    coalesce_onehot_shap_ranks,
    collect_shap_output_dir,
    discover_shap_collect_cases,
    gather_origin_records,
    find_saved_model_artifact,
    find_screening_split_dir,
    find_walk_forward_run_dir,
    interpretability_for_model,
    model_run_name,
    resolve_shap_sample_limits,
    DEFAULT_MAX_BACKGROUND,
    DEFAULT_MAX_EVAL_SAMPLES,
    ICL_MAX_BACKGROUND,
    ICL_MAX_EVAL_SAMPLES,
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


def test_sum_abs_over_time_feature_importance():
    # One sample, two timesteps, two features.
    # Feature 0: |SHAP| = 1 then 3 → sum 4
    # Feature 1: |SHAP| = 2 then 2 → sum 4
    shap_ts = np.array(
        [
            [[1.0, 2.0], [3.0, 2.0]],
            [[1.0, 0.0], [1.0, 4.0]],
        ]
    )
    out = _sum_abs_over_time_feature_importance(shap_ts, n_features=2)
    # Sample0: [4, 4], sample1: [2, 4] → mean [3, 4]
    assert out.tolist() == pytest.approx([3.0, 4.0])
    mean_out = _mean_abs_feature_importance(shap_ts, n_features=2)
    # Mean over time then samples: sample0 [2, 2], sample1 [1, 2] → [1.5, 2]
    assert mean_out.tolist() == pytest.approx([1.5, 2.0])
    assert out[0] > mean_out[0]


def test_rank_features_from_2d_mean_vector():
    names = ["a", "b", "c"]
    mean_abs = np.array([[0.1, 0.5, 0.2]])
    rows = _rank_features(names, mean_abs)
    assert rows[0]["name"] == "b"


def test_coalesce_onehot_drainage_class():
    assert coalesce_onehot_feature_name("drainage_class_4") == "drainage_class"
    assert coalesce_onehot_feature_name("ctx:drainage_class_2") == "ctx:drainage_class"
    assert coalesce_onehot_feature_name("ctx:awc") == "ctx:awc"
    assert coalesce_onehot_feature_name("ndvi_mean_3") == "ndvi_mean_3"

    ranks = coalesce_onehot_shap_ranks(
        [
            {"name": "drainage_class_1", "mean_abs_shap": 0.1, "rank": 1},
            {"name": "drainage_class_4", "mean_abs_shap": 0.3, "rank": 2},
            {"name": "awc", "mean_abs_shap": 0.2, "rank": 3},
            {"name": "ctx:drainage_class_3", "mean_abs_shap": 0.05, "rank": 4},
            {"name": "ctx:drainage_class_5", "mean_abs_shap": 0.15, "rank": 5},
        ]
    )
    by_name = {row["name"]: row["mean_abs_shap"] for row in ranks}
    assert by_name["drainage_class"] == pytest.approx(0.4)
    assert by_name["ctx:drainage_class"] == pytest.approx(0.2)
    assert by_name["awc"] == pytest.approx(0.2)
    assert ranks[0]["name"] == "drainage_class"


def test_rank_features_coalesces_onehot_dummies():
    names = ["drainage_class_1", "drainage_class_4", "awc"]
    mean_abs = np.array([0.1, 0.3, 0.25])
    rows = _rank_features(names, mean_abs)
    assert [row["name"] for row in rows] == ["drainage_class", "awc"]
    assert rows[0]["mean_abs_shap"] == pytest.approx(0.4)


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
    assert interpretability_for_model("tabicl")["explainer_label"] == "PermutationImportance"
    assert interpretability_for_model("tabdpt")["method"] == "sklearn_permutation"
    unknown = interpretability_for_model("xgboost")
    assert unknown["method"] == "permutation_shap"


def test_resolve_shap_sample_limits_tabpfn():
    bg, ev = resolve_shap_sample_limits(
        "tabpfn",
        max_background=500,
        max_eval_samples=500,
        n_train=10_000,
        n_test=800,
    )
    assert bg == ICL_MAX_BACKGROUND
    assert ev == ICL_MAX_EVAL_SAMPLES
    bg_rf, ev_rf = resolve_shap_sample_limits(
        "random_forest",
        max_background=500,
        max_eval_samples=500,
        n_train=10_000,
        n_test=800,
    )
    assert bg_rf == 500
    assert ev_rf == 500


def test_resolve_shap_sample_limits_uses_all_rows_when_below_cap():
    bg, ev = resolve_shap_sample_limits(
        "transformer_lf",
        max_background=500,
        max_eval_samples=500,
        n_train=84,
        n_test=8,
    )
    assert bg == 84
    assert ev == 8


def test_model_manifest_includes_tabular_foundation_models():
    for slug in ("tabpfn", "tabicl", "tabdpt"):
        entry = MODEL_MANIFEST[slug]
        assert entry["framework"] == "pandas"
        assert entry["feature_design"] is True
        assert entry["needs_gpu"] is True


def test_resolve_shap_sample_limits_tabicl_and_tabdpt():
    for slug in ("tabicl", "tabdpt"):
        bg, ev = resolve_shap_sample_limits(
            slug,
            max_background=500,
            max_eval_samples=500,
            n_train=1000,
            n_test=200,
        )
        assert bg == ICL_MAX_BACKGROUND
        assert ev == ICL_MAX_EVAL_SAMPLES


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


def test_collect_shap_output_dir_from_parallel_origins(tmp_path: Path):
    model_dir = tmp_path / "maize_NL" / "tabicl"
    for year, prec_shap in ((2019, 0.3), (2020, 0.5)):
        origin_dir = model_dir / f"origin_{year}"
        origin_dir.mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "crop": "maize",
                    "country": "NL",
                    "model": "tabicl",
                    "horizon": "eos",
                    "seed": 42,
                    "train_years": list(range(2008, year)),
                    "test_years": [year],
                    "n_train": 10,
                    "n_test": 2,
                    "reproduction": {"corr_saved_preds": 1.0, "max_abs_pred_diff": 0.0},
                    "explainer": "PermutationImportance",
                    "features": [
                        {"name": "prec_sum_1", "mean_abs_shap": prec_shap, "rank": 1},
                    ],
                }
            ),
            origin_dir / "shap_importance.yaml",
        )

    summaries = collect_shap_output_dir(
        tmp_path, crop="maize", country="NL", models=["tabicl"]
    )
    assert len(summaries) == 1
    assert summaries[0]["n_origins"] == 2
    assert (model_dir / "shap_summary.yaml").is_file()
    assert (model_dir / "shap_aggregate.csv").is_file()
    agg = pd.read_csv(model_dir / "shap_aggregate.csv")
    prec = agg.loc[agg["feature"] == "prec_sum_1"].iloc[0]
    assert prec["median_mean_abs_shap"] == pytest.approx(0.4)
    assert (tmp_path / "maize_NL" / "shap_aggregate_all_models.csv").is_file()

    records = gather_origin_records(model_dir)
    assert len(records) == 2


def test_discover_shap_collect_cases(tmp_path: Path):
    case_root = tmp_path / "shap_importance" / "maize_NL_eos" / "maize_NL" / "random_forest"
    (case_root / "origin_2020").mkdir(parents=True)
    (case_root / "origin_2020" / "shap_importance.yaml").write_text(
        "crop: maize\ncountry: NL\n", encoding="utf-8"
    )
    (tmp_path / "shap_importance" / "maize_NL_eos" / "maize_NL" / "empty_model").mkdir(
        parents=True
    )
    (tmp_path / "shap_importance" / "not_a_case").mkdir()

    cases = discover_shap_collect_cases(tmp_path / "shap_importance")
    assert len(cases) == 1
    assert cases[0].crop == "maize"
    assert cases[0].country == "NL"
    assert cases[0].models == ("random_forest",)
    assert cases[0].n_origins == 1
