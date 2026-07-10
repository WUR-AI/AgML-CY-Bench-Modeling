"""Tests for model-family radar aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.analysis.build_model_family_radar_dashboard import build_radar_html
from cybench.runs.analysis.global_insights_lib import compat_legacy_summary_columns
from cybench.runs.analysis.model_family_radar_lib import (
    RADAR_ABSOLUTE_NOTE,
    RADAR_NORMALIZATION_NOTE,
    absolute_scores,
    ai_error_reduction_pct,
    build_family_dataset_rows,
    build_paper_family_table_latex,
    build_radar_payload,
    build_radar_slice,
    build_sample_scatter_slice,
    pick_representatives,
    relative_scores,
    summarize_sample_scatter,
    _ai_benefit_map_slice,
)
from cybench.runs.analysis.country_significance_lib import build_country_bootstrap_payload


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
        "r_anomaly": 0.2,
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
    assert reps["Process-Based"] == "lpjml_bc"
    assert reps["Feature-Engineered ML"] == "xgboost"
    assert reps["Naive baselines"] == "trend"


def test_pick_representatives_uses_country_equal_median_not_pooled_rows():
    """Countries with more crops must not dominate representative selection."""
    df = pd.DataFrame(
        [
            _summary_row("lstm_lf", nrmse=0.22, country="US", crop="maize"),
            _summary_row("lstm_lf", nrmse=0.22, country="US", crop="wheat"),
            _summary_row("lstm_lf", nrmse=0.22, country="US", crop="soy"),
            _summary_row("lstm_lf", nrmse=0.40, country="DE", crop="maize"),
            _summary_row("tst_lf", nrmse=0.226, country="US", crop="maize"),
            _summary_row("tst_lf", nrmse=0.226, country="DE", crop="maize"),
        ]
    )
    reps = pick_representatives(df)
    assert reps["Sequence / Deep TS"] == "tst_lf"


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


def test_ai_error_reduction_pct_formula():
    assert ai_error_reduction_pct(0.15, 0.20) == 25.0
    assert ai_error_reduction_pct(0.22, 0.20) == -10.0
    assert ai_error_reduction_pct(0.20, 0.20) == 0.0
    assert ai_error_reduction_pct(0.15, 0.0) is None


def test_ai_benefit_map_uses_best_traditional_including_lpjml():
    df = pd.DataFrame(
        [
            _summary_row("trend", nrmse=0.20, country="DE"),
            _summary_row("lpjml_bc", nrmse=0.16, country="DE"),
            _summary_row("lightgbm", nrmse=0.14, country="DE"),
            _summary_row("average", nrmse=0.18, country="FR"),
            _summary_row("lpjml_bc", nrmse=0.22, country="FR"),
            _summary_row("xgboost", nrmse=0.19, country="FR"),
        ]
    )
    slice_payload = _ai_benefit_map_slice(df, batch_horizon="eos")
    by_country = {row["country"]: row for row in slice_payload["countries"]}
    de = by_country["DE"]
    assert de["traditional_family"] == "Process-Based"
    assert de["traditional_model"] == "lpjml_bc"
    assert de["nrmse_traditional"] == pytest.approx(0.16)
    assert de["benefit_pct"] == pytest.approx(12.5)
    fr = by_country["FR"]
    assert fr["traditional_family"] == "Naive baselines"
    assert fr["traditional_model"] == "average"
    assert fr["benefit_pct"] == pytest.approx(-5.56, abs=0.01)


def test_relative_scores_span_unit_interval():
    raw = pd.DataFrame(
        {
            "nrmse": [0.2, 0.8],
            "r_spatial": [0.1, 0.9],
            "r_temporal": [0.3, 0.7],
            "r_res": [0.0, 1.0],
        }
    )
    rel = relative_scores(raw)
    assert rel["Overall"].tolist() == [1.0, 0.0]
    assert rel["Overall"].min() == 0.0
    assert rel["Overall"].max() == 1.0
    assert rel["Anomaly"].tolist() == [0.0, 1.0]


def test_absolute_scores_use_fixed_scales():
    raw = pd.DataFrame(
        {
            "nrmse": [0.1, 0.2, 0.30],
            "r_spatial": [-0.2, 0.5, 1.0],
            "r_temporal": [0.0, 0.5, 1.0],
            "r_res": [0.25, 0.5, 0.75],
        }
    )
    abs_scores = absolute_scores(raw)
    assert abs_scores["Overall"].tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert abs_scores["Spatial"].tolist() == [0.0, 0.5, 1.0]
    assert abs_scores["Temporal"].tolist() == [0.0, 0.5, 1.0]
    assert abs_scores["Anomaly"].tolist() == [0.25, 0.5, 0.75]


def test_absolute_scores_clamps_out_of_range_nrmse():
    raw = pd.DataFrame(
        {
            "nrmse": [0.05, 0.40],
            "r_spatial": [0.5, 0.5],
            "r_temporal": [0.5, 0.5],
            "r_res": [0.5, 0.5],
        }
    )
    abs_scores = absolute_scores(raw)
    assert abs_scores["Overall"].iloc[0] == 1.0
    assert abs_scores["Overall"].iloc[1] == 0.0


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


def test_build_family_dataset_rows_includes_all_view_metrics():
    df = pd.DataFrame(
        [
            _summary_row("lightgbm", crop="maize", country="DE"),
            _summary_row("xgboost", crop="maize", country="FR"),
        ]
    )
    reps = {"Feature-Engineered ML": "lightgbm"}
    rows = build_family_dataset_rows(df, reps)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "maize_DE"
    assert set(rows[0]["metrics"]) == {"nrmse", "r2", "r_spatial", "r_temporal", "r_res"}


def test_build_radar_slice_includes_country_iqr():
    df = pd.DataFrame(
        [
            _summary_row("lightgbm", nrmse=0.10, country="DE"),
            _summary_row("lightgbm", nrmse=0.30, country="FR"),
            _summary_row("lightgbm", nrmse=0.20, country="US"),
        ]
    )
    radar = build_radar_slice(
        df, batch_horizon="eos", representatives={"Feature-Engineered ML": "lightgbm"}
    )
    fam = radar["families"][0]
    country_vals = pd.Series([0.10, 0.20, 0.30])
    assert fam["raw"]["nrmse"] == pytest.approx(0.20)
    assert fam["iqr"]["nrmse"]["q25"] == pytest.approx(float(country_vals.quantile(0.25)))
    assert fam["iqr"]["nrmse"]["q75"] == pytest.approx(float(country_vals.quantile(0.75)))


def test_build_radar_payload_structure(tmp_path: Path):
    for hz in ("eos", "mid"):
        d = tmp_path / f"paper_walk_forward_de_{hz}_v1"
        d.mkdir(parents=True)
        pd.DataFrame(
            [
                _summary_row("average", nrmse=0.25),
                _summary_row("trend", nrmse=0.22),
                _summary_row(
                    "lpjml_bc",
                    nrmse=0.22,
                    r_spatial=0.62,
                    r_temporal=0.33,
                    r_res=0.2,
                    r2_res=0.12,
                ),
                _summary_row(
                    "lightgbm",
                    nrmse=0.15,
                    r_spatial=0.74,
                    r_temporal=0.1,
                    r_res=0.5,
                    r2_res=0.28,
                ),
                _summary_row(
                    "transformer_lf",
                    nrmse=0.17,
                    r_spatial=0.73,
                    r_temporal=0.21,
                    r_res=0.4,
                    r2_res=0.18,
                ),
                _summary_row(
                    "tabpfn",
                    nrmse=0.16,
                    r_spatial=0.77,
                    r_temporal=0.31,
                    r_res=0.45,
                    r2_res=0.22,
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
    assert families["Process-Based"]["relative"]["Overall"] == pytest.approx(0.0, abs=0.01)
    assert families["Feature-Engineered ML"]["absolute"]["Overall"] == pytest.approx(0.75, abs=0.01)
    assert "absolute" in families["Naive baselines"]
    assert len(eos_all["dataset_rows"]) == 5
    assert eos_all["dataset_rows"][0]["metrics"]["nrmse"] is not None
    ml_fam = next(f for f in eos_all["families"] if f["family"] == "Feature-Engineered ML")
    assert "vs_naive_sig" in ml_fam
    assert "vs_naive" in ml_fam
    assert isinstance(ml_fam["vs_naive_sig"]["nrmse"], bool)
    assert "median_delta" in ml_fam["vs_naive"]["nrmse"]
    assert "p_one_sided" in ml_fam["vs_naive"]["nrmse"]
    naive_fam = next(f for f in eos_all["families"] if f["family"] == "Naive baselines")
    assert naive_fam["vs_naive_sig"]["nrmse"] is False
    assert payload["family_vs_naive_sig_note"]
    assert RADAR_NORMALIZATION_NOTE in payload["relative_note"]
    assert RADAR_ABSOLUTE_NOTE in payload["absolute_note"]
    assert payload["radar_scales"]["absolute"]["Overall"]["lo"] == 0.1
    assert "sample_scatter" in payload
    assert "benchmark_map_isos" in payload
    assert "DE" in payload["benchmark_map_isos"]
    assert "winner_maps" in payload
    assert "ai_benefit_maps" in payload
    assert "all" in payload["winner_maps"]["eos"]
    assert "Overall" in payload["winner_maps"]["eos"]["all"]
    benefit_rows = payload["ai_benefit_maps"]["eos"]["all"]["countries"]
    assert isinstance(benefit_rows, list)
    if benefit_rows:
        assert "benefit_pct" in benefit_rows[0]
        assert "nrmse_traditional" in benefit_rows[0]
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


def test_build_country_bootstrap_payload():
    df = pd.DataFrame(
        [
            _summary_row("trend", country="DE", crop="maize", nrmse=0.20),
            _summary_row("lightgbm", country="DE", crop="maize", nrmse=0.14),
            _summary_row("trend", country="FR", crop="maize", nrmse=0.22),
            _summary_row("lightgbm", country="FR", crop="maize", nrmse=0.19),
            _summary_row("trend", country="DE", crop="wheat", nrmse=0.18),
            _summary_row("tabicl", country="DE", crop="wheat", nrmse=0.14),
        ]
    )
    payload = build_country_bootstrap_payload(df, n_bootstrap=200, seed=0)
    assert "eos" in payload["by_horizon"]
    assert "maize" in payload["by_horizon"]["eos"]
    assert payload["by_horizon"]["eos"]["maize"]["n_countries"] == 2


def test_build_radar_payload_includes_country_bootstrap(tmp_path: Path):
    d = tmp_path / "paper_walk_forward_de_eos_v1"
    d.mkdir(parents=True)
    pd.DataFrame(
        [
            _summary_row("trend", country="DE", nrmse=0.20),
            _summary_row("lightgbm", country="DE", nrmse=0.14),
        ]
    ).to_csv(d / "walk_forward_summary.csv", index=False)
    payload = build_radar_payload(tmp_path, version=1)
    assert "country_bootstrap" in payload
    assert payload["country_bootstrap"]["n_bootstrap"] == 10_000


def test_build_radar_html_embeds_payload(tmp_path: Path):
    d = tmp_path / "paper_walk_forward_de_eos_v1"
    d.mkdir(parents=True)
    pd.DataFrame([_summary_row("lightgbm")]).to_csv(d / "walk_forward_summary.csv", index=False)
    payload = build_radar_payload(tmp_path, version=1)
    html = build_radar_html(payload)
    assert "Model family comparison" in html
    assert '"Feature-Engineered ML"' in html
    assert "data-mode=\"absolute\"" in html
    assert "data-mode=\"benefit\"" in html
    assert 'id="map-export-svg"' in html
    assert 'id="map-export-png"' in html
    assert 'id="table-export-latex"' in html
    assert "buildMetricsTableLatex" in html
    assert 'id="bootstrap-export-latex"' in html
    assert "buildBootstrapTableLatex" in html
    assert "booktabs" in html


def test_build_radar_slice_includes_r2_in_family_raw():
    df = pd.DataFrame(
        [
            _summary_row("xgboost", country="DE", crop="maize", nrmse=0.2, r2=0.51),
            _summary_row("xgboost", country="US", crop="maize", nrmse=0.22, r2=0.48),
        ]
    )
    sl = build_radar_slice(df, batch_horizon="eos", crop="maize")
    xgb = next(f for f in sl["families"] if f["model"] == "xgboost")
    assert xgb["raw"]["r2"] == pytest.approx(0.495, abs=0.01)


def test_build_paper_family_table_latex_includes_r2():
    rows = []
    for crop, country, model, nrmse, r2 in [
        ("maize", "DE", "trend", 0.24, 0.31),
        ("maize", "US", "trend", 0.22, 0.35),
        ("maize", "DE", "random_forest", 0.25, 0.42),
        ("maize", "US", "random_forest", 0.23, 0.44),
        ("wheat", "DE", "tabicl", 0.16, 0.55),
        ("wheat", "FR", "tabicl", 0.17, 0.52),
    ]:
        rows.append(_summary_row(model, crop=crop, country=country, nrmse=nrmse, r2=r2))
    df = pd.DataFrame(rows)
    latex = build_paper_family_table_latex(df, batch_horizon="eos", crops=("maize", "wheat"))

    assert "\\textbf{Maize}" in latex
    assert "\\textbf{Wheat}" in latex
    assert "$R^2$" in latex
    assert "Random Forest" in latex
    assert "TabICL" in latex
