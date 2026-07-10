"""Tests for country-level AI benefit bootstrap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cybench.runs.analysis.country_significance_lib import (
    analyze_country_ai_benefit,
    bootstrap_country_ai_metrics,
    bootstrap_family_vs_naive_stats,
    bootstrap_one_sided_significant,
    build_family_vs_naive_significance,
    country_ai_benefit_frame,
    family_vs_naive_country_deltas,
    prepare_work_for_family_vs_naive,
)
from cybench.runs.analysis.model_family_radar_lib import build_radar_slice


def _row(model: str, country: str, nrmse: float, *, crop: str = "maize") -> dict:
    return {
        "crop": crop,
        "country": country,
        "model": model,
        "batch_horizon": "eos",
        "nrmse": nrmse,
    }


def test_country_ai_benefit_frame_picks_best_per_side():
    df = pd.DataFrame(
        [
            _row("trend", "DE", 0.20),
            _row("lpjml_bc", "DE", 0.16),
            _row("lightgbm", "DE", 0.14),
            _row("trend", "FR", 0.22),
            _row("average", "FR", 0.18),
            _row("xgboost", "FR", 0.19),
        ]
    )
    frame = country_ai_benefit_frame(df, batch_horizon="eos", crop="maize")
    by_country = {r["country"]: r for r in frame.to_dict(orient="records")}
    assert by_country["DE"]["traditional_model"] == "lpjml_bc"
    assert by_country["DE"]["ai_model"] == "lightgbm"
    assert by_country["DE"]["delta_pct"] == pytest.approx(12.5)
    assert by_country["DE"]["ai_wins"] is True
    assert by_country["FR"]["traditional_model"] == "average"
    assert by_country["FR"]["ai_wins"] is False


def test_seed_average_before_country_comparison():
    df = pd.DataFrame(
        [
            {**_row("trend", "DE", 0.20), "seed": 0},
            {**_row("trend", "DE", 0.24), "seed": 1},
            {**_row("lightgbm", "DE", 0.14), "seed": 0},
            {**_row("lightgbm", "DE", 0.18), "seed": 1},
        ]
    )
    frame = country_ai_benefit_frame(df, batch_horizon="eos", crop="maize")
    assert len(frame) == 1
    assert frame.loc[0, "nrmse_trad"] == pytest.approx(0.22)
    assert frame.loc[0, "nrmse_ai"] == pytest.approx(0.16)


def test_bootstrap_joint_metrics_ci():
    frame = pd.DataFrame(
        {
            "delta_abs": [0.02, 0.01, 0.03, 0.0, -0.01, 0.015],
            "delta_pct": [10.0, 5.0, 12.0, 0.0, -4.0, 6.0],
            "ai_wins": [True, True, True, False, False, True],
        }
    )
    boot = bootstrap_country_ai_metrics(frame, n_bootstrap=3000, seed=1)
    assert boot["median_delta_pct"] == pytest.approx(5.5)
    assert boot["win_rate"] == pytest.approx(4 / 6)
    assert boot["delta_pct_ci_lo"] <= boot["median_delta_pct"] <= boot["delta_pct_ci_hi"]
    assert boot["win_rate_ci_lo"] <= boot["win_rate"] <= boot["win_rate_ci_hi"]


def test_bootstrap_family_vs_naive_stats_includes_p_value():
    deltas = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    stats = bootstrap_family_vs_naive_stats(deltas, n_bootstrap=2000, seed=1)
    assert stats["significant"] is True
    assert stats["median_delta"] == pytest.approx(0.03)
    assert stats["p_one_sided"] is not None
    assert stats["p_one_sided"] < 0.05
    assert stats["ci_lo"] <= stats["median_delta"] <= stats["ci_hi"]


def test_bootstrap_family_vs_naive_stats_marks_significantly_worse():
    deltas = np.array([-0.05, -0.04, -0.03, -0.02, -0.01])
    stats = bootstrap_family_vs_naive_stats(deltas, n_bootstrap=2000, seed=3)
    assert stats["significant_worse"] is True
    assert stats["significant"] is False
    assert stats["p_one_sided_worse"] is not None
    assert stats["p_one_sided_worse"] < 0.05


def test_bootstrap_one_sided_not_significant_when_mixed():
    deltas = np.array([0.05, 0.04, -0.03, -0.02, 0.01])
    assert bootstrap_one_sided_significant(deltas, n_bootstrap=2000, seed=2) is False


def test_family_vs_naive_country_deltas_nrmse_direction():
    df = pd.DataFrame(
        [
            {**_row("trend", "DE", 0.20), "nrmse": 0.20},
            {**_row("lightgbm", "DE", 0.14), "nrmse": 0.14},
            {**_row("trend", "FR", 0.22), "nrmse": 0.22},
            {**_row("lightgbm", "FR", 0.18), "nrmse": 0.18},
        ]
    )
    reps = {"Naive baselines": "trend", "Feature-Engineered ML": "lightgbm"}
    deltas = family_vs_naive_country_deltas(
        df,
        family_model="lightgbm",
        naive_model="trend",
        metric="nrmse",
        higher_is_better=False,
    )
    assert deltas.tolist() == pytest.approx([0.06, 0.04])


def test_family_vs_naive_skips_country_without_naive_rep():
    df = pd.DataFrame(
        [
            {**_row("average", "DE", 0.20), "nrmse": 0.20},
            {**_row("lightgbm", "DE", 0.14), "nrmse": 0.14},
            {**_row("trend", "FR", 0.22), "nrmse": 0.22},
            {**_row("lightgbm", "FR", 0.18), "nrmse": 0.18},
        ]
    )
    deltas = family_vs_naive_country_deltas(
        df,
        family_model="lightgbm",
        naive_model="trend",
        metric="nrmse",
        higher_is_better=False,
    )
    assert len(deltas) == 1
    assert deltas[0] == pytest.approx(0.04)


def test_table_median_gap_matches_table_cells():
    df = pd.DataFrame(
        [
            {**_row("trend", "DE", 0.20), "r2": 0.10},
            {**_row("tabpfn", "DE", 0.16), "r2": 0.45},
            {**_row("trend", "FR", 0.22), "r2": 0.12},
            {**_row("tabpfn", "FR", 0.18), "r2": 0.40},
            {**_row("trend", "NL", 0.21), "r2": 0.50},
            {**_row("tabpfn", "NL", 0.17), "r2": 0.05},
            {**_row("trend", "PL", 0.20), "r2": 0.48},
            {**_row("tabpfn", "PL", 0.19), "r2": 0.06},
        ]
    )
    reps = {"Naive baselines": "trend", "Tabular Foundation": "tabpfn"}
    slice_ = build_radar_slice(df, batch_horizon="eos")
    tabpfn = next(f for f in slice_["families"] if f["family"] == "Tabular Foundation")
    assert tabpfn["vs_naive"]["r2"]["table_median_gap"] == pytest.approx(
        tabpfn["raw"]["r2"] - next(f for f in slice_["families"] if f["is_naive"])["raw"]["r2"]
    )


def test_build_family_vs_naive_significance_marks_improvement():
    df = pd.DataFrame(
        [
            {**_row("trend", "DE", 0.20), "nrmse": 0.20, "r2": 0.40},
            {**_row("lightgbm", "DE", 0.14), "nrmse": 0.14, "r2": 0.55},
            {**_row("trend", "FR", 0.22), "nrmse": 0.22, "r2": 0.38},
            {**_row("lightgbm", "FR", 0.18), "nrmse": 0.18, "r2": 0.52},
            {**_row("trend", "NL", 0.21), "nrmse": 0.21, "r2": 0.39},
            {**_row("lightgbm", "NL", 0.17), "nrmse": 0.17, "r2": 0.51},
        ]
    )
    reps = {
        "Naive baselines": "trend",
        "Feature-Engineered ML": "lightgbm",
    }
    sig = build_family_vs_naive_significance(
        df, reps, metrics=("nrmse", "r2"), n_bootstrap=3000, seed=0
    )
    assert sig["Feature-Engineered ML"]["nrmse"]["significant"] is True
    assert sig["Feature-Engineered ML"]["r2"]["significant"] is True
    assert sig["Feature-Engineered ML"]["nrmse"]["p_one_sided"] is not None


def test_bootstrap_family_vs_naive_stats_single_country_returns_median():
    deltas = np.array([0.04])
    stats = bootstrap_family_vs_naive_stats(deltas, n_bootstrap=2000, seed=1)
    assert stats["n_countries"] == 1
    assert stats["median_delta"] == pytest.approx(0.04)
    assert stats["ci_lo"] is None
    assert stats["significant"] is False


def test_family_vs_naive_uses_spatial_agg_fallback():
    df = pd.DataFrame(
        [
            {**_row("trend", "DE", 0.20), "r_spatial_agg": 0.40},
            {**_row("lightgbm", "DE", 0.14), "r_spatial_agg": 0.55},
            {**_row("trend", "FR", 0.22), "r_spatial_agg": 0.38},
            {**_row("lightgbm", "FR", 0.18), "r_spatial_agg": 0.52},
        ]
    )
    deltas = family_vs_naive_country_deltas(
        df,
        family_model="lightgbm",
        naive_model="trend",
        metric="r_spatial",
        higher_is_better=True,
    )
    assert deltas.tolist() == pytest.approx([0.15, 0.14])


def test_build_family_vs_naive_significance_without_nrmse_filter():
    df = pd.DataFrame(
        [
            {**_row("trend", "DE", 0.20), "nrmse": np.nan, "r_spatial": 0.40},
            {**_row("lightgbm", "DE", 0.14), "nrmse": np.nan, "r_spatial": 0.55},
            {**_row("trend", "FR", 0.22), "nrmse": np.nan, "r_spatial": 0.38},
            {**_row("lightgbm", "FR", 0.18), "nrmse": np.nan, "r_spatial": 0.52},
        ]
    )
    from cybench.runs.analysis.global_insights_lib import _filter_summary_work
    from cybench.runs.analysis.model_family_radar_lib import build_radar_slice

    work_sig = prepare_work_for_family_vs_naive(
        _filter_summary_work(df, batch_horizon="eos", require_valid_nrmse=False)
    )
    reps = {"Naive baselines": "trend", "Feature-Engineered ML": "lightgbm"}
    sig = build_family_vs_naive_significance(work_sig, reps, metrics=("r_spatial",))
    assert sig["Feature-Engineered ML"]["r_spatial"]["n_countries"] == 2
    assert sig["Feature-Engineered ML"]["r_spatial"]["median_delta"] is not None

    # NRMSE-filtered work would drop these rows entirely.
    filtered = _filter_summary_work(df, batch_horizon="eos", require_valid_nrmse=True)
    assert filtered.empty


def test_analyze_country_ai_benefit_returns_bootstrap_fields():
    df = pd.DataFrame(
        [
            _row("trend", "DE", 0.20),
            _row("lightgbm", "DE", 0.14),
            _row("trend", "FR", 0.18),
            _row("lightgbm", "FR", 0.19),
            _row("trend", "US", 0.21),
            _row("xgboost", "US", 0.17),
        ]
    )
    res = analyze_country_ai_benefit(
        df, batch_horizon="eos", crop="maize", n_bootstrap=500, seed=0
    )
    assert res["n_countries"] == 3
    assert res["n_ai_wins"] == 2
    assert res["win_rate"] == pytest.approx(2 / 3)
    assert res["median_delta_pct"] is not None
    assert res["delta_pct_ci_lo"] is not None
    assert res["median_nrmse_trad"] == pytest.approx(0.20)
    assert res["median_nrmse_ai"] == pytest.approx(0.17)
    assert res["median_delta_abs"] is not None
    assert res["delta_abs_ci_lo"] is not None
