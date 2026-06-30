"""Tests for model-family radar aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.analysis.build_model_family_radar_dashboard import build_radar_html
from cybench.runs.analysis.model_family_radar_lib import (
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
        "r2": 0.5,
        "nrmse": 0.18,
        "r2_spatial": 0.4,
        "r2_spatial_agg": 0.45,
        "r2_temporal": 0.3,
        "r2_temporal_agg": 0.35,
        "r2_anomaly": 0.2,
    }
    base.update(metrics)
    return base


def test_pick_representatives_prefers_override():
    df = pd.DataFrame(
        [
            _summary_row("lpjml_bc", r2=0.4),
            _summary_row("twso_bc", r2=0.9),
            _summary_row("lightgbm", r2=0.8),
        ]
    )
    reps = pick_representatives(df)
    assert reps["Process-Based"] == "lpjml_bc"
    assert reps["Feature-Engineered ML"] == "lightgbm"


def test_relative_scores_span_unit_interval():
    raw = pd.DataFrame(
        {
            "r2": [0.2, 0.8],
            "r2_spatial": [0.1, 0.9],
            "r2_temporal": [0.3, 0.7],
            "r2_anomaly": [0.0, 1.0],
        }
    )
    rel = relative_scores(raw)
    assert rel["Overall"].min() == 0.0
    assert rel["Overall"].max() == 1.0
    assert rel["Anomaly"].tolist() == [0.0, 1.0]


def test_build_radar_payload_structure(tmp_path: Path):
    for hz in ("eos", "mid"):
        d = tmp_path / f"paper_walk_forward_de_{hz}_v1"
        d.mkdir(parents=True)
        pd.DataFrame(
            [
                _summary_row("average_yield", r2=0.1, nrmse=0.25),
                _summary_row("lpjml_bc", r2=0.5, r2_spatial=0.4, r2_temporal=0.3, r2_anomaly=0.2, nrmse=0.22),
                _summary_row("lightgbm", r2=0.8, r2_spatial=0.7, r2_temporal=0.6, r2_anomaly=0.5, nrmse=0.15),
                _summary_row("transformer_lf", r2=0.7, r2_spatial=0.6, r2_temporal=0.5, r2_anomaly=0.4, nrmse=0.17),
                _summary_row("tabpfn", r2=0.75, r2_spatial=0.65, r2_temporal=0.55, r2_anomaly=0.45, nrmse=0.16),
            ]
        ).to_csv(d / "walk_forward_summary.csv", index=False)

    payload = build_radar_payload(tmp_path, version=1)
    assert payload["n_rows"] == 10
    eos_all = payload["by_horizon"]["eos"]["all"]
    families = {f["family"]: f for f in eos_all["families"]}
    assert set(families) == {
        "Process-Based",
        "Feature-Engineered ML",
        "Sequence / Deep TS",
        "Tabular Foundation",
    }
    assert families["Feature-Engineered ML"]["relative"]["Overall"] == 1.0
    assert families["Process-Based"]["relative"]["Overall"] == 0.0
    assert "sample_scatter" in payload
    eos_scatter = payload["sample_scatter"]["eos"]["all"]
    assert "families" in eos_scatter
    assert any(f["family"] == "Tabular Foundation" for f in eos_scatter["families"])
    assert eos_scatter["summary"]["n_points"] >= 4


def test_build_sample_scatter_slice_uses_family_representatives():
    df = pd.DataFrame(
        [
            _summary_row("average_yield", nrmse=0.20, n_train=4000, batch_horizon="eos"),
            _summary_row("lightgbm", nrmse=0.16, n_train=4000, batch_horizon="eos"),
            _summary_row("xgboost", nrmse=0.14, n_train=5000, batch_horizon="eos"),
            _summary_row("average_yield", nrmse=0.22, n_train=6000, batch_horizon="eos", country="FR"),
            _summary_row("tabpfn", nrmse=0.11, n_train=6000, batch_horizon="eos", country="FR"),
            _summary_row("average_yield", nrmse=0.21, n_train=3000, batch_horizon="mid", country="FR"),
            _summary_row("lightgbm", nrmse=0.20, n_train=3000, batch_horizon="mid", country="FR"),
        ]
    )
    eos = build_sample_scatter_slice(df, batch_horizon="eos")
    ml = next(f for f in eos if f["family"] == "Feature-Engineered ML")
    assert ml["model"] == "lightgbm"
    assert len(ml["points"]) == 1
    assert ml["points"][0]["n_train"] == 4000
    assert ml["points"][0]["relative_nrmse"] == 0.8
    tab = next(f for f in eos if f["family"] == "Tabular Foundation")
    assert tab["model"] == "tabpfn"
    assert tab["points"][0]["relative_nrmse"] == pytest.approx(0.5, rel=1e-3)
    assert "xgboost" not in {p["model"] for fam in eos for p in fam["points"]}


def test_build_radar_payload_includes_relative_scatter_metric(tmp_path: Path):
    d = tmp_path / "paper_walk_forward_de_eos_v1"
    d.mkdir(parents=True)
    pd.DataFrame(
        [
            _summary_row("average_yield", nrmse=0.2),
            _summary_row("lightgbm", nrmse=0.16),
        ]
    ).to_csv(d / "walk_forward_summary.csv", index=False)
    payload = build_radar_payload(tmp_path, version=1)
    assert payload["sample_scatter_metric"]["key"] == "relative_nrmse"


def test_summarize_sample_scatter_reports_percentiles():
    fams = build_sample_scatter_slice(
        pd.DataFrame(
            [
                _summary_row("average_yield", nrmse=0.20, n_train=100, batch_horizon="eos"),
                _summary_row("lightgbm", nrmse=0.16, n_train=100, batch_horizon="eos"),
                _summary_row("average_yield", nrmse=0.20, n_train=10000, batch_horizon="eos", country="US"),
                _summary_row("lightgbm", nrmse=0.10, n_train=10000, batch_horizon="eos", country="US"),
            ]
        ),
        batch_horizon="eos",
    )
    summary = summarize_sample_scatter(fams)
    assert summary["n_points"] == 2
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
