"""Tests for global insights aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.analysis.export_family_models_horizon_table import build_markdown_table
from cybench.runs.analysis.global_insights_lib import (
    aggregate_model_leaderboard,
    attach_baseline_metrics,
    build_dashboard_hrefs,
    build_family_models_horizon_table,
    build_horizon_skill_curves_payload,
    build_insights_payload,
    build_crop_comparison_payload,
    build_model_country_matrix,
    compare_crops_pairwise,
    compare_horizons,
    dashboard_href_for_paper_dir,
    horizons_in_data,
    is_baseline_model,
    load_summary_frame,
    median_model_metrics_across_countries,
    quantile_model_metrics_across_countries,
    parse_paper_dir_name,
)


def test_parse_paper_dir_name():
    assert parse_paper_dir_name("paper_walk_forward_de_eos_v1") == ("DE", "eos", 1)
    assert parse_paper_dir_name("paper_walk_forward_pl_mid_v1") == ("PL", "mid", 1)
    assert parse_paper_dir_name("paper_walk_forward_br_qtr_v2") == ("BR", "qtr", 2)
    assert parse_paper_dir_name("paper_walk_forward") is None


def test_dashboard_href_for_paper_dir():
    assert dashboard_href_for_paper_dir("paper_walk_forward_au_mid_v2") == (
        "au_walk_forward_mid_v2/dashboard.html"
    )
    assert dashboard_href_for_paper_dir("not_a_paper_dir") is None


def test_build_dashboard_hrefs(tmp_path: Path):
    for cc, hz in [("de", "eos"), ("de", "mid")]:
        d = tmp_path / f"paper_walk_forward_{cc}_{hz}_v1"
        d.mkdir(parents=True)
        pd.DataFrame([{"crop": "maize", "model": "ridge", "nrmse": 0.1}]).to_csv(
            d / "walk_forward_summary.csv", index=False
        )
    hrefs = build_dashboard_hrefs(tmp_path, version=1)
    assert hrefs["DE"]["eos"] == "de_walk_forward_eos_v1/dashboard.html"
    assert hrefs["DE"]["mid"] == "de_walk_forward_mid_v1/dashboard.html"


def test_is_baseline_model():
    assert is_baseline_model("average")
    assert is_baseline_model("AverageYieldModel")
    assert not is_baseline_model("ridge")


def test_aggregate_model_leaderboard_mean_nrmse():
    df = pd.DataFrame(
        [
            {"model": "ridge", "nrmse": 0.10, "r2": 0.9, "n_samples": 100, "country": "DE", "batch_horizon": "eos", "crop": "maize"},
            {"model": "ridge", "nrmse": 0.30, "r2": 0.5, "n_samples": 10, "country": "FR", "batch_horizon": "eos", "crop": "maize"},
            {"model": "xgboost", "nrmse": 0.20, "r2": 0.7, "n_samples": 50, "country": "DE", "batch_horizon": "eos", "crop": "maize"},
            {"model": "ridge", "nrmse": 0.25, "r2": 0.6, "n_samples": 40, "country": "DE", "batch_horizon": "mid", "crop": "maize"},
        ]
    )
    board = aggregate_model_leaderboard(df, batch_horizon="eos")
    assert list(board["model"]) == ["ridge", "xgboost"]
    assert board.loc[board["model"] == "ridge", "median_nrmse"].iloc[0] == 0.2

    maize = aggregate_model_leaderboard(df, batch_horizon="eos", crop="maize")
    assert list(maize["model"]) == ["ridge", "xgboost"]

    mid_board = aggregate_model_leaderboard(df, batch_horizon="mid")
    assert list(mid_board["model"]) == ["ridge"]


def test_baseline_beat_rate():
    df = pd.DataFrame(
        [
            {
                "model": "average",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.30,
                "r2": 0.2,
                "n_samples": 50,
            },
            {
                "model": "ridge",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "n_samples": 50,
            },
            {
                "model": "xgboost",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.40,
                "r2": -0.1,
                "n_samples": 50,
            },
        ]
    )
    tagged = attach_baseline_metrics(df)
    assert bool(tagged.loc[tagged["model"] == "ridge", "beats_baseline"].iloc[0])
    assert not bool(tagged.loc[tagged["model"] == "xgboost", "beats_baseline"].iloc[0])

    board = aggregate_model_leaderboard(df, batch_horizon="eos")
    ridge = board.loc[board["model"] == "ridge"].iloc[0]
    xgb = board.loc[board["model"] == "xgboost"].iloc[0]
    assert ridge["beat_baseline_rate"] == 1.0
    assert xgb["beat_baseline_rate"] == 0.0
    avg = board.loc[board["model"] == "average"].iloc[0]
    assert pd.isna(avg["beat_baseline_rate"])
    assert not bool(tagged.loc[tagged["model"] == "average", "skilled"].iloc[0])


def test_skilled_only_leaderboard():
    """Skilled filter drops unskilled rows; baseline models are never kept."""
    df = pd.DataFrame(
        [
            {
                "model": "average",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.20,
                "r2": 0.3,
                "n_samples": 50,
            },
            {
                "model": "ridge",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "n_samples": 50,
            },
            {
                "model": "ridge",
                "crop": "wheat",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.50,
                "r2": -0.2,
                "n_samples": 50,
            },
        ]
    )
    board = aggregate_model_leaderboard(df, batch_horizon="eos", skilled_only=True)
    assert len(board) == 1
    assert board.iloc[0]["model"] == "ridge"
    assert board.iloc[0]["n_datasets"] == 1


def test_build_model_country_matrix():
    df = pd.DataFrame(
        [
            {
                "model": "average",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.30,
                "r2": 0.2,
                "n_samples": 50,
            },
            {
                "model": "ridge",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "n_samples": 50,
            },
            {
                "model": "ridge",
                "crop": "wheat",
                "country": "FR",
                "batch_horizon": "eos",
                "nrmse": 0.20,
                "r2": 0.7,
                "n_samples": 30,
            },
        ]
    )
    matrix = build_model_country_matrix(df, batch_horizon="eos", crop="maize")
    assert matrix["models"] == ["average", "ridge"]
    assert matrix["countries"] == ["DE"]
    ridge_cell = next(c for c in matrix["cells"] if c["model"] == "ridge")
    average_cell = next(c for c in matrix["cells"] if c["model"] == "average")
    assert ridge_cell["axes"]["overall"]["nrmse"] == 0.10
    assert average_cell["axes"]["overall"]["nrmse"] == 0.30
    assert matrix["model_totals"]["ridge"]["overall"]["nrmse"] == 0.10


def test_build_model_country_matrix_spatial_axis():
    df = pd.DataFrame(
        [
            {
                "model": "ridge",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "r_spatial": 0.62,
                "n_samples": 50,
            },
            {
                "model": "ridge",
                "crop": "wheat",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.20,
                "r2": 0.7,
                "r_spatial": 0.35,
                "n_samples": 50,
            },
        ]
    )
    matrix = build_model_country_matrix(df, batch_horizon="eos")
    ridge_cell = next(c for c in matrix["cells"] if c["model"] == "ridge")
    assert ridge_cell["axes"]["spatial"]["r"] == 0.485
    assert matrix["model_totals"]["ridge"]["spatial"]["r"] == 0.485


def test_model_median_by_country_matches_matrix():
    """Leaderboard and matrix Median column: median of per-country NRMSE values."""
    df = pd.DataFrame(
        [
            {
                "model": "tabpfn",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "n_samples": 50,
            },
            {
                "model": "tabpfn",
                "crop": "maize",
                "country": "FR",
                "batch_horizon": "eos",
                "nrmse": 0.30,
                "r2": 0.7,
                "n_samples": 50,
            },
            {
                "model": "tabpfn",
                "crop": "wheat",
                "country": "FR",
                "batch_horizon": "eos",
                "nrmse": 0.50,
                "r2": 0.5,
                "n_samples": 50,
            },
        ]
    )
    board = aggregate_model_leaderboard(df, batch_horizon="eos")
    matrix = build_model_country_matrix(df, batch_horizon="eos")
    expected = float(board.loc[board["model"] == "tabpfn", "median_nrmse"].iloc[0])
    assert matrix["model_totals"]["tabpfn"]["overall"]["nrmse"] == expected
    assert expected == 0.25
    de_cell = next(c for c in matrix["cells"] if c["country"] == "DE")
    fr_cell = next(c for c in matrix["cells"] if c["country"] == "FR")
    assert de_cell["axes"]["overall"]["nrmse"] == 0.10
    assert fr_cell["axes"]["overall"]["nrmse"] == 0.40
    assert (de_cell["median_nrmse"] + fr_cell["median_nrmse"]) / 2 == 0.25


def test_compare_horizons_eos_better():
    df = pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "DE",
                "model": "ridge",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "n_samples": 50,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "ridge",
                "batch_horizon": "mid",
                "nrmse": 0.20,
                "r2": 0.7,
                "n_samples": 50,
            },
        ]
    )
    detail, summary = compare_horizons(df)
    assert len(detail) == 1
    assert detail["delta_nrmse"].iloc[0] == 0.10
    assert bool(detail["eos_better"].iloc[0]) is True
    assert summary.loc[summary["model"] == "ridge", "eos_win_rate"].iloc[0] == 1.0


def test_compare_crops_pairwise_shared_countries_only():
    df = pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "DE",
                "model": "ridge",
                "batch_horizon": "eos",
                "nrmse": 0.10,
                "r2": 0.9,
                "n_samples": 50,
            },
            {
                "crop": "wheat",
                "country": "DE",
                "model": "ridge",
                "batch_horizon": "eos",
                "nrmse": 0.20,
                "r2": 0.7,
                "n_samples": 50,
            },
            {
                "crop": "maize",
                "country": "US",
                "model": "ridge",
                "batch_horizon": "eos",
                "nrmse": 0.25,
                "r2": 0.6,
                "n_samples": 40,
            },
            {
                "crop": "wheat",
                "country": "US",
                "model": "ridge",
                "batch_horizon": "eos",
                "nrmse": 0.15,
                "r2": 0.8,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "FR",
                "model": "ridge",
                "batch_horizon": "eos",
                "nrmse": 0.30,
                "r2": 0.5,
                "n_samples": 30,
            },
        ]
    )
    detail, summary = compare_crops_pairwise(df, crop_a="maize", crop_b="wheat")
    assert len(detail) == 2
    assert set(detail["country"]) == {"DE", "US"}
    de = detail[detail["country"] == "DE"].iloc[0]
    assert de["delta_nrmse"] == pytest.approx(0.10)
    assert bool(de["crop_a_better"]) is True
    us = detail[detail["country"] == "US"].iloc[0]
    assert bool(us["crop_a_better"]) is False
    assert summary.loc[summary["model"] == "ridge", "crop_a_win_rate"].iloc[0] == 0.5

    payload = build_crop_comparison_payload(df)
    overall = payload["eos"]["maize_vs_wheat"]["overall"]
    assert overall["n_countries"] == 2
    assert overall["crop_a_win_rate"] == 0.5


def test_load_summary_frame_from_tmp(tmp_path: Path):
    de_dir = tmp_path / "paper_walk_forward_de_eos_v1"
    de_dir.mkdir()
    pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "DE",
                "model": "ridge",
                "horizon": "eos",
                "dataset": "maize_DE",
                "nrmse": 0.12,
                "r2": 0.8,
                "n_samples": 40,
                "n_regions": 4,
                "n_years": 10,
            }
        ]
    ).to_csv(de_dir / "walk_forward_summary.csv", index=False)
    frame = load_summary_frame([de_dir / "walk_forward_summary.csv"])
    assert len(frame) == 1
    assert frame["country"].iloc[0] == "DE"
    assert frame["batch_horizon"].iloc[0] == "eos"


def test_build_insights_payload_structure(tmp_path: Path):
    for cc, hz in [("de", "eos"), ("de", "mid")]:
        d = tmp_path / f"paper_walk_forward_{cc}_{hz}_v1"
        d.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "crop": "maize",
                    "country": "DE",
                    "model": "average",
                    "nrmse": 0.25,
                    "r2": 0.3,
                    "n_samples": 40,
                },
                {
                    "crop": "maize",
                    "country": "DE",
                    "model": "ridge",
                    "nrmse": 0.15 if hz == "eos" else 0.22,
                    "r2": 0.8 if hz == "eos" else 0.6,
                    "n_samples": 40,
                },
            ]
        ).to_csv(d / "walk_forward_summary.csv", index=False)

    payload = build_insights_payload(tmp_path, version=1)
    assert "leaderboards" in payload
    assert payload["dashboard_hrefs"]["DE"]["eos"] == "de_walk_forward_eos_v1/dashboard.html"
    assert payload["dashboard_hrefs"]["DE"]["mid"] == "de_walk_forward_mid_v1/dashboard.html"
    assert len(payload["leaderboards"]["eos"]["all"]) == 2
    models = {r["model"] for r in payload["leaderboards"]["eos"]["all"]}
    assert models == {"average", "ridge"}
    assert payload["model_country"]["eos"]["all"]["models"] == ["average", "ridge"]
    assert "model_country_skilled" in payload
    ridge_board = next(r for r in payload["leaderboards"]["eos"]["all"] if r["model"] == "ridge")
    ridge_matrix = payload["model_country"]["eos"]["all"]["model_totals"]["ridge"]["overall"]["nrmse"]
    assert ridge_board["median_nrmse"] == ridge_matrix
    assert len(payload["matrix_axes"]) == 4
    assert payload["matrix_axes"][0]["id"] == "overall"
    assert "country_map_cc" in payload
    assert payload["country_map_cc"]["DE"] == "DE"
    assert "benchmark_map_isos" in payload
    assert "metric_map_scales" in payload
    assert payload["metric_map_scales"]["nrmse"]["higher_better"] is False
    assert "horizon_delta_scales" in payload
    assert payload["horizon_delta_scales"]["nrmse"]["higher_better"] is True


def _three_horizon_fixture(tmp_path: Path) -> pd.DataFrame:
    """DE+FR with eos/mid/qtr; US only eos (excluded from curves)."""
    rows_by_hz = {
        "mid": {"ridge": 0.30, "trend": 0.34, "xgboost": 0.28},
        "qtr": {"ridge": 0.22, "trend": 0.30, "xgboost": 0.20},
        "eos": {"ridge": 0.15, "trend": 0.28, "xgboost": 0.14},
    }
    pattern_by_hz = {
        "mid": {"r_spatial": 0.55, "r_temporal": 0.35, "r_res": 0.25},
        "qtr": {"r_spatial": 0.60, "r_temporal": 0.50, "r_res": 0.30},
        "eos": {"r_spatial": 0.65, "r_temporal": 0.70, "r_res": 0.35},
    }

    def _row(model: str, cc: str, hz: str, nrmse: float, r2: float) -> dict:
        pat = pattern_by_hz[hz]
        return {
            "crop": "maize",
            "country": cc.upper(),
            "model": model,
            "nrmse": nrmse,
            "r2": r2,
            "r_spatial": pat["r_spatial"],
            "r_temporal": pat["r_temporal"],
            "r_res": pat["r_res"],
            "n_samples": 40,
        }

    for cc, hz in [("de", "eos"), ("de", "mid"), ("de", "qtr"), ("fr", "eos"), ("fr", "mid"), ("fr", "qtr")]:
        d = tmp_path / f"paper_walk_forward_{cc}_{hz}_v1"
        d.mkdir(parents=True, exist_ok=True)
        metrics = rows_by_hz[hz]
        pd.DataFrame(
            [
                _row("ridge", cc, hz, metrics["ridge"], 0.7),
                _row("trend", cc, hz, metrics["trend"], 0.3),
                _row("xgboost", cc, hz, metrics["xgboost"], 0.75),
                _row("average", cc, hz, metrics["trend"] + 0.02, 0.2),
                _row("lstm_lf", cc, hz, metrics["ridge"] + 0.05, 0.6),
            ]
        ).to_csv(d / "walk_forward_summary.csv", index=False)
    # US: eos only
    us = tmp_path / "paper_walk_forward_us_eos_v1"
    us.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "US",
                "model": "ridge",
                "nrmse": 0.12,
                "r2": 0.8,
                "n_samples": 50,
            }
        ]
    ).to_csv(us / "walk_forward_summary.csv", index=False)
    from cybench.runs.analysis.global_insights_lib import discover_summary_tables, load_summary_frame

    paths = discover_summary_tables(tmp_path, version=1)
    return load_summary_frame(paths)


def test_horizon_skill_curves_inner_join_countries():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        df = _three_horizon_fixture(Path(tmp))
        payload = build_horizon_skill_curves_payload(df)
        assert len(payload["horizons"]) == 3
        assert [h["id"] for h in payload["horizons"]] == ["mid", "qtr", "eos"]
        maize = payload["by_crop"]["maize"]
        assert maize["n_countries"] == 2
        assert set(maize["countries"]) == {"DE", "FR"}
        assert "US" in maize["excluded_countries"]

        xgb = next(f for f in maize["families"] if f["model"] == "xgboost")
        assert xgb["plot"] is True
        assert xgb.get("eos_only") is not True
        nrmse_by_hz = {
            p["horizon"]: p["metrics"]["nrmse"]["median"] for p in xgb["points"]
        }
        assert nrmse_by_hz["mid"] > nrmse_by_hz["qtr"] > nrmse_by_hz["eos"]
        temporal_by_hz = {
            p["horizon"]: p["metrics"]["r_temporal"]["median"] for p in xgb["points"]
        }
        assert temporal_by_hz["mid"] < temporal_by_hz["qtr"] < temporal_by_hz["eos"]
        assert len(payload["axes"]) == 4
        assert payload["axes"][2]["id"] == "temporal"
        model_slugs = {m["model"] for m in maize["models"]}
        assert "ridge" in model_slugs
        assert "xgboost" in model_slugs
        assert "trend" in model_slugs
        xgb_all = next(m for m in maize["models"] if m["model"] == "xgboost")
        assert xgb_all["family"] == "Feature-Engineered ML"
        assert xgb_all["is_representative"] is True
        assert "models_table_note" in payload
        assert "models_plot_note" in payload


def test_horizon_skill_curves_eos_only_lpjml_in_table_not_plot(tmp_path: Path):
    df = _three_horizon_fixture(tmp_path)
    lpj_rows = [
        {
            "crop": "maize",
            "country": cc,
            "model": "lpjml_bc",
            "batch_horizon": "eos",
            "nrmse": 0.19,
            "r2": 0.5,
            "n_samples": 40,
        }
        for cc in ("DE", "FR")
    ]
    df = pd.concat([df, pd.DataFrame(lpj_rows)], ignore_index=True)

    payload = build_horizon_skill_curves_payload(df)
    maize = payload["by_crop"]["maize"]
    lpj = next(f for f in maize["families"] if f["model"] == "lpjml_bc")
    assert lpj["eos_only"] is True
    assert lpj["plot"] is False
    assert len(lpj["points"]) == 1
    assert lpj["points"][0]["horizon"] == "eos"
    assert lpj["points"][0]["metrics"]["nrmse"]["median"] == 0.19
    plot_models = {f["model"] for f in maize["families"] if f["plot"]}
    assert "lpjml_bc" not in plot_models
    lpj_all = next(m for m in maize["models"] if m["model"] == "lpjml_bc")
    assert lpj_all["eos_only"] is True
    assert lpj_all["family"] == "Process-Based"
    assert "plot_excluded_note" in payload


def test_horizon_models_include_partial_early_coverage():
    df = pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "DE",
                "model": "tabpfn",
                "batch_horizon": "eos",
                "nrmse": 0.12,
                "r2": 0.8,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "tabpfn",
                "batch_horizon": "early",
                "nrmse": 0.20,
                "r2": 0.5,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "xgboost",
                "batch_horizon": "early",
                "nrmse": 0.18,
                "r2": 0.55,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "xgboost",
                "batch_horizon": "eos",
                "nrmse": 0.11,
                "r2": 0.82,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "FR",
                "model": "xgboost",
                "batch_horizon": "early",
                "nrmse": 0.19,
                "r2": 0.52,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "FR",
                "model": "xgboost",
                "batch_horizon": "eos",
                "nrmse": 0.13,
                "r2": 0.78,
                "n_samples": 40,
            },
        ]
    )
    payload = build_horizon_skill_curves_payload(df)
    models = payload["by_crop"]["maize"]["models"]
    tabpfn = next(m for m in models if m["model"] == "tabpfn")
    assert any(p["horizon"] == "early" for p in tabpfn["points"])
    assert any(p["horizon"] == "eos" for p in tabpfn["points"])
    early_pt = next(p for p in tabpfn["points"] if p["horizon"] == "early")
    assert early_pt["n_countries"] == 1


def test_build_insights_payload_includes_qtr_and_curves(tmp_path: Path):
    _three_horizon_fixture(tmp_path)
    payload = build_insights_payload(tmp_path, version=1)
    assert "qtr" in payload["available_horizons"]
    assert "qtr" in payload["leaderboards"]
    assert payload["horizon_skill_curves"]["horizons"]
    assert payload["horizon_skill_curves"]["by_crop"]["maize"]["n_countries"] == 2
    assert payload["horizon_skill_curves"]["by_crop"]["maize"]["models"]
    assert "horizon_summary" not in payload
    assert "overall_horizon" not in payload


def test_median_model_metrics_matches_insights_matrix():
    from cybench.runs.analysis.model_family_radar_lib import build_radar_slice

    df = pd.DataFrame(
        [
            {
                "model": "trend",
                "crop": "maize",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.22,
                "r_spatial": 0.90,
                "r_temporal": 0.3,
                "r_res": 0.2,
                "n_samples": 50,
            },
            {
                "model": "trend",
                "crop": "wheat",
                "country": "DE",
                "batch_horizon": "eos",
                "nrmse": 0.22,
                "r_spatial": 0.50,
                "r_temporal": 0.3,
                "r_res": 0.2,
                "n_samples": 50,
            },
            {
                "model": "trend",
                "crop": "maize",
                "country": "US",
                "batch_horizon": "eos",
                "nrmse": 0.22,
                "r_spatial": 0.71,
                "r_temporal": 0.3,
                "r_res": 0.2,
                "n_samples": 50,
            },
            {
                "model": "trend",
                "crop": "wheat",
                "country": "US",
                "batch_horizon": "eos",
                "nrmse": 0.22,
                "r_spatial": 0.71,
                "r_temporal": 0.3,
                "r_res": 0.2,
                "n_samples": 50,
            },
            {
                "model": "trend",
                "crop": "soy",
                "country": "US",
                "batch_horizon": "eos",
                "nrmse": 0.22,
                "r_spatial": 0.71,
                "r_temporal": 0.3,
                "r_res": 0.2,
                "n_samples": 50,
            },
        ]
    )
    matrix = build_model_country_matrix(df, batch_horizon="eos")
    insights_spatial = matrix["model_totals"]["trend"]["spatial"]["r"]

    medians = median_model_metrics_across_countries(df, ["r_spatial"], models=["trend"])
    assert medians.loc["trend", "r_spatial"] == insights_spatial

    radar = build_radar_slice(
        df, batch_horizon="eos", representatives={"Naive baselines": "trend"}
    )
    trend = next(f for f in radar["families"] if f["model"] == "trend")
    assert trend["raw"]["r_spatial"] == insights_spatial
    country_vals = pd.Series([0.70, 0.71])
    assert trend["iqr"]["r_spatial"]["q25"] == pytest.approx(
        float(country_vals.quantile(0.25)), abs=0.001
    )
    assert trend["iqr"]["r_spatial"]["q75"] == pytest.approx(
        float(country_vals.quantile(0.75)), abs=0.001
    )
    # DE=0.70, US=0.71 -> 0.705; pooled row median would be 0.71
    assert insights_spatial == pytest.approx(0.705, abs=0.001)


def test_quantile_model_metrics_across_countries():
    df = pd.DataFrame(
        [
            {"model": "m", "country": "DE", "crop": "maize", "nrmse": 0.10},
            {"model": "m", "country": "DE", "crop": "wheat", "nrmse": 0.30},
            {"model": "m", "country": "US", "crop": "maize", "nrmse": 0.20},
            {"model": "m", "country": "FR", "crop": "maize", "nrmse": 0.40},
        ]
    )
    medians = median_model_metrics_across_countries(df, ["nrmse"], models=["m"])
    q25, q75 = quantile_model_metrics_across_countries(df, ["nrmse"], models=["m"])
    country_vals = pd.Series([0.20, 0.20, 0.40])
    assert medians.loc["m", "nrmse"] == pytest.approx(0.20)
    assert q25.loc["m", "nrmse"] == pytest.approx(float(country_vals.quantile(0.25)))
    assert q75.loc["m", "nrmse"] == pytest.approx(float(country_vals.quantile(0.75)))


def test_build_family_models_horizon_table_lists_all_models_per_family():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        df = _three_horizon_fixture(Path(tmp))
        table = build_family_models_horizon_table(df, metrics=("nrmse", "r2"))
        assert not table.empty
        ml = table[table["family"] == "Feature-Engineered ML"]
        assert set(ml["model"]) == {"ridge", "xgboost"}
        deep = table[table["family"] == "Sequence / Deep TS"]
        assert list(deep["model"]) == ["lstm_lf"]
        assert "early_nrmse_median" not in table.columns
        assert "mid_nrmse_median" in table.columns
        xgb = ml[ml["model"] == "xgboost"].iloc[0]
        assert xgb["mid_nrmse_median"] > xgb["eos_nrmse_median"]


def test_build_family_models_horizon_table_allows_incomplete_early():
    df = pd.DataFrame(
        [
            {
                "crop": "maize",
                "country": "DE",
                "model": "tabpfn",
                "batch_horizon": "eos",
                "nrmse": 0.12,
                "r2": 0.8,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "FR",
                "model": "tabpfn",
                "batch_horizon": "eos",
                "nrmse": 0.14,
                "r2": 0.75,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "tabpfn",
                "batch_horizon": "early",
                "nrmse": 0.20,
                "r2": 0.5,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "xgboost",
                "batch_horizon": "early",
                "nrmse": 0.18,
                "r2": 0.55,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "FR",
                "model": "xgboost",
                "batch_horizon": "early",
                "nrmse": 0.19,
                "r2": 0.52,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "DE",
                "model": "xgboost",
                "batch_horizon": "eos",
                "nrmse": 0.11,
                "r2": 0.82,
                "n_samples": 40,
            },
            {
                "crop": "maize",
                "country": "FR",
                "model": "xgboost",
                "batch_horizon": "eos",
                "nrmse": 0.13,
                "r2": 0.78,
                "n_samples": 40,
            },
        ]
    )
    table = build_family_models_horizon_table(df, metrics=("nrmse",))
    tabpfn = table[table["model"] == "tabpfn"].iloc[0]
    xgb = table[table["model"] == "xgboost"].iloc[0]
    assert tabpfn["n_countries_early"] == 1
    assert pd.isna(tabpfn["early_nrmse_median"]) is False
    assert tabpfn["early_nrmse_median"] == pytest.approx(0.20)
    assert xgb["n_countries_early"] == 2
    md = build_markdown_table(
        table, metric="nrmse", horizons=("early", "eos"), crop_label="all crops"
    )
    assert "## Tabular Foundation" in md
    assert "TabPFN" in md
