"""Tests for model-family radar aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.analysis.build_model_family_radar_dashboard import build_radar_html
from cybench.runs.analysis.global_insights_lib import compat_legacy_summary_columns
from cybench.runs.analysis.model_family_radar_lib import (
    RADAR_NORMALIZATION_NOTE,
    build_radar_payload,
    build_sample_scatter_slice,
    pick_representatives,
    relative_scores,
    summarize_sample_scatter,
)


def _summary_row(model: str, **metrics: float) -> dict:
    base = {
        "crop": "maize",
        "country": "DE",
        "model": model,
        "batch_horizon": "eos",
        "n_samples": 500,
        "n_train": 2800,
        "r": 0.55,
        "r2": 0.5,
        "nrmse": 0.18,
        "r_spatial": 0.72,
        "r_temporal": 0.25,
        "r_res": 0.2,
        "r2_spatial": 0.4,
        "r2_spatial_agg": 0.45,
        "r2_temporal": 0.3,
        "r2_temporal_agg": 0.35,
        "r2_anomaly": 0.2,
        "r2_res": 0.25,
    }
    base.update(metrics)
    return base


def test_pick_representatives_by_lowest_nrmse():
    df = pd.DataFrame(
        [
            _summary_row("lpjml_bc", nrmse=0.22),
            _summary_row("twso_bc", nrmse=0.18),
            _summary_row("lightgbm", nrmse=0.15),
            _summary_row("xgboost", nrmse=0.14),
            _summary_row("average", nrmse=0.25),
            _summary_row("trend", nrmse=0.23),
        ]
    )
    reps = pick_representatives(df)
    assert reps["Process-Based"] == "twso_bc"
    assert reps["Feature-Engineered ML"] == "xgboost"
    assert reps["Naive baselines"] == "trend"


def test_pick_representatives_prefers_override():
    df = pd.DataFrame(
        [
            _summary_row("lpjml_bc", nrmse=0.18),
            _summary_row("twso_bc", nrmse=0.22),
            _summary_row("lightgbm", nrmse=0.15),
        ]
    )
    reps = pick_representatives(df, overrides={"Process-Based": "lpjml_bc"})
    assert reps["Process-Based"] == "lpjml_bc"


def test_relative_scores_span_unit_interval():
    raw = pd.DataFrame(
        {
            "r": [0.2, 0.8],
            "r_spatial": [0.1, 0.9],
            "r_temporal": [0.3, 0.7],
            "r_res": [0.0, 1.0],
        }
    )
    rel = relative_scores(raw)
    assert rel["Overall"].min() == 0.0
    assert rel["Overall"].max() == 1.0
    assert rel["Anomaly"].tolist() == [0.0, 1.0]


def test_legacy_summary_columns_map_to_agg_metrics():
    df = pd.DataFrame(
        [
            {
                "model": "trend",
                "r2_spatial": 0.69,
                "r2_temporal": -0.75,
                "r_spatial": 0.89,
                "r_temporal": -0.29,
                "r2_res": -1.0,
            }
        ]
    )
    out = compat_legacy_summary_columns(df)
    assert out["r2_spatial_agg"].iloc[0] == 0.69
    assert out["r2_temporal_agg"].iloc[0] == -0.75
    assert out["r_spatial_agg"].iloc[0] == 0.89
    assert out["r_temporal_agg"].iloc[0] == -0.29
    assert "r_spatial" not in out.columns
    assert "r_temporal" not in out.columns


def test_build_radar_payload_structure(tmp_path: Path):
    for hz in ("eos", "mid"):
        d = tmp_path / f"paper_walk_forward_de_{hz}_v1"
        d.mkdir(parents=True)
        pd.DataFrame(
            [
                _summary_row("average", r=0.2, nrmse=0.25),
                _summary_row("trend", r=0.3, nrmse=0.22),
                _summary_row(
                    "lpjml_bc",
                    r=0.5,
                    r_spatial=0.62,
                    r_temporal=0.33,
                    r_res=0.2,
                    nrmse=0.22,
                ),
                _summary_row(
                    "lightgbm",
                    r=0.8,
                    r_spatial=0.74,
                    r_temporal=0.1,
                    r_res=0.5,
                    nrmse=0.15,
                ),
                _summary_row(
                    "transformer_lf",
                    r=0.7,
                    r_spatial=0.73,
                    r_temporal=0.21,
                    r_res=0.4,
                    nrmse=0.17,
                ),
                _summary_row(
                    "tabpfn",
                    r=0.75,
                    r_spatial=0.77,
                    r_temporal=0.31,
                    r_res=0.45,
                    nrmse=0.16,
                ),
            ]
        ).to_csv(d / "walk_forward_summary.csv", index=False)

    payload = build_radar_payload(tmp_path, version=1)
    assert payload["n_rows"] == 12
    eos_all = payload["by_horizon"]["eos"]["all"]
    families = {f["family"]: f for f in eos_all["families"]}
    assert len(eos_all["families"]) == 5
    assert set(families) == {
        "Naive baselines",
        "Process-Based",
        "Feature-Engineered ML",
        "Sequence / Deep TS",
        "Tabular Foundation",
    }
    assert families["Naive baselines"]["display_name"] == "Trend"
    assert families["Feature-Engineered ML"]["relative"]["Overall"] == 1.0
    assert families["Process-Based"]["relative"]["Overall"] == pytest.approx(0.4, abs=0.01)
    assert RADAR_NORMALIZATION_NOTE in payload["normalization_note"]
    assert "sample_scatter" in payload
    eos_scatter = payload["sample_scatter"]["eos"]["all"]
    assert "families" in eos_scatter
    assert any(f["family"] == "Tabular Foundation" for f in eos_scatter["families"])
    assert eos_scatter["summary"]["n_points"] >= 4


def test_build_sample_scatter_slice_uses_family_representatives():
    df = pd.DataFrame(
        [
            _summary_row("average", nrmse=0.20, n_train=4000, batch_horizon="eos"),
            _summary_row("lightgbm", nrmse=0.16, n_train=4000, batch_horizon="eos"),
            _summary_row("xgboost", nrmse=0.14, n_train=5000, batch_horizon="eos"),
            _summary_row("average", nrmse=0.22, n_train=6000, batch_horizon="eos", country="FR"),
            _summary_row("tabpfn", nrmse=0.11, n_train=6000, batch_horizon="eos", country="FR"),
            _summary_row("average", nrmse=0.21, n_train=3000, batch_horizon="mid", country="FR"),
            _summary_row("lightgbm", nrmse=0.20, n_train=3000, batch_horizon="mid", country="FR"),
        ]
    )
    eos = build_sample_scatter_slice(df, batch_horizon="eos")
    ml = next(f for f in eos if f["family"] == "Feature-Engineered ML")
    assert ml["model"] == "xgboost"
    assert len(ml["points"]) == 1
    assert ml["points"][0]["n_train"] == 5000
    assert ml["points"][0]["relative_nrmse"] == 0.7
    tab = next(f for f in eos if f["family"] == "Tabular Foundation")
    assert tab["model"] == "tabpfn"
    assert tab["points"][0]["relative_nrmse"] == pytest.approx(0.5, rel=1e-3)
    assert "lightgbm" not in {p["model"] for fam in eos for p in fam["points"]}


def test_build_radar_payload_includes_relative_scatter_metric(tmp_path: Path):
    d = tmp_path / "paper_walk_forward_de_eos_v1"
    d.mkdir(parents=True)
    pd.DataFrame(
        [
            _summary_row("average", nrmse=0.2),
            _summary_row("lightgbm", nrmse=0.16),
        ]
    ).to_csv(d / "walk_forward_summary.csv", index=False)
    payload = build_radar_payload(tmp_path, version=1)
    assert payload["sample_scatter_metric"]["key"] == "relative_nrmse"


def test_summarize_sample_scatter_reports_percentiles():
    fams = build_sample_scatter_slice(
        pd.DataFrame(
            [
                _summary_row("average", nrmse=0.20, n_train=100, batch_horizon="eos"),
                _summary_row("lightgbm", nrmse=0.16, n_train=100, batch_horizon="eos"),
                _summary_row("average", nrmse=0.20, n_train=10000, batch_horizon="eos", country="US"),
                _summary_row("lightgbm", nrmse=0.10, n_train=10000, batch_horizon="eos", country="US"),
            ]
        ),
        batch_horizon="eos",
    )
    summary = summarize_sample_scatter(fams)
    assert summary["n_points"] == 4
    assert summary["x_min"] == 100
    assert summary["x_max"] == 10000


def test_build_radar_html_embeds_payload(tmp_path: Path):
    d = tmp_path / "paper_walk_forward_de_eos_v1"
    d.mkdir(parents=True)
    pd.DataFrame([_summary_row("lightgbm")]).to_csv(d / "walk_forward_summary.csv", index=False)
    payload = build_radar_payload(tmp_path, version=1)
    html = build_radar_html(payload)
    assert "Model family comparison" in html
    assert '"Feature-Engineered ML"' in html
