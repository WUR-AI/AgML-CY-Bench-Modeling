"""Tests for country-level AI benefit bootstrap analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cybench.runs.analysis.country_significance_lib import (
    analyze_country_ai_benefit,
    bootstrap_country_ai_metrics,
    country_ai_benefit_frame,
)


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
