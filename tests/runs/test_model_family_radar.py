"""Tests for model-family radar aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cybench.runs.analysis.build_model_family_radar_dashboard import build_radar_html
from cybench.runs.analysis.model_family_radar_lib import (
    build_radar_payload,
    build_sample_scatter_slice,
    pick_representatives,
    relative_scores,
)


def _summary_row(model: str, **metrics: float) -> dict:
    base = {
        "crop": "maize",
        "country": "DE",
        "model": model,
        "batch_horizon": "eos",
        "n_samples": 500,
        "r2": 0.5,
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
                _summary_row("lpjml_bc", r2=0.5, r2_spatial=0.4, r2_temporal=0.3, r2_anomaly=0.2),
                _summary_row("lightgbm", r2=0.8, r2_spatial=0.7, r2_temporal=0.6, r2_anomaly=0.5),
                _summary_row("transformer_lf", r2=0.7, r2_spatial=0.6, r2_temporal=0.5, r2_anomaly=0.4),
                _summary_row("tabpfn", r2=0.75, r2_spatial=0.65, r2_temporal=0.55, r2_anomaly=0.45),
            ]
        ).to_csv(d / "walk_forward_summary.csv", index=False)

    payload = build_radar_payload(tmp_path, version=1)
    assert payload["n_rows"] == 8
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
    assert any(f["family"] == "Tabular Foundation" for f in eos_scatter)


def test_build_sample_scatter_slice_groups_by_family():
    df = pd.DataFrame(
        [
            _summary_row("lightgbm", n_samples=400, r2=0.8, batch_horizon="eos"),
            _summary_row("tabpfn", n_samples=600, r2=0.75, batch_horizon="eos"),
            _summary_row("lightgbm", n_samples=300, r2=0.7, batch_horizon="mid", country="FR"),
        ]
    )
    eos = build_sample_scatter_slice(df, batch_horizon="eos")
    ml = next(f for f in eos if f["family"] == "Feature-Engineered ML")
    assert len(ml["points"]) == 1
    assert ml["points"][0]["n_samples"] == 400
    tab = next(f for f in eos if f["family"] == "Tabular Foundation")
    assert tab["points"][0]["metrics"]["r2"] == 0.75


def test_build_radar_html_embeds_payload(tmp_path: Path):
    d = tmp_path / "paper_walk_forward_de_eos_v1"
    d.mkdir(parents=True)
    pd.DataFrame([_summary_row("lightgbm")]).to_csv(d / "walk_forward_summary.csv", index=False)
    payload = build_radar_payload(tmp_path, version=1)
    html = build_radar_html(payload)
    assert "Model family comparison" in html
    assert '"Feature-Engineered ML"' in html
