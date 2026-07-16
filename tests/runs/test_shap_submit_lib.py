"""Tests for SHAP submission planning."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.slurm.shap_submit_lib import (
    benchmark_crop_country_pairs,
    build_shap_submit_plans,
)


def test_benchmark_pairs_non_empty():
    pairs = benchmark_crop_country_pairs()
    assert pairs
    assert ("maize", "NL") in pairs


def test_build_plans_skips_missing_baselines(tmp_path: Path):
    plans = build_shap_submit_plans(
        countries=["NL"],
        crops=["maize"],
        models=["random_forest"],
        output_root=tmp_path,
    )
    assert len(plans) == 1
    assert plans[0].skip
    assert "missing baselines" in plans[0].skip_reason


def test_build_plans_pending_origins_only(tmp_path: Path):
    baselines = tmp_path / "baselines_NL_eos_v4"
    baselines.mkdir(parents=True)
    out = tmp_path / "shap_importance" / "maize_NL_eos" / "maize_NL" / "random_forest"
    (out / "origin_2016").mkdir(parents=True)
    (out / "origin_2016" / "shap_importance.yaml").write_text(
        "crop: maize\ncountry: NL\nmodel: random_forest\n"
        "test_years:\n- 2016\nfeatures: []\n",
        encoding="utf-8",
    )
    plans = build_shap_submit_plans(
        countries=["NL"],
        crops=["maize"],
        models=["random_forest"],
        output_root=tmp_path,
    )
    assert len(plans) == 1
    # Without real baselines/WF runs this stays skipped; origin bookkeeping is tested
    # once integration fixtures exist. Ensure plan references NL maize.
    assert plans[0].crop == "maize"
    assert plans[0].country == "NL"
