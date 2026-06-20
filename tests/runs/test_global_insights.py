"""Tests for global insights aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cybench.runs.analysis.global_insights_lib import (
    aggregate_model_leaderboard,
    attach_baseline_metrics,
    build_insights_payload,
    build_model_country_matrix,
    compare_horizons,
    is_baseline_model,
    load_summary_frame,
    parse_paper_dir_name,
)


def test_parse_paper_dir_name():
    assert parse_paper_dir_name("paper_walk_forward_de_eos_v1") == ("DE", "eos", 1)
    assert parse_paper_dir_name("paper_walk_forward_pl_mid_v1") == ("PL", "mid", 1)
    assert parse_paper_dir_name("paper_walk_forward") is None


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
    assert ridge_cell["median_nrmse"] == 0.10
    assert average_cell["median_nrmse"] == 0.30
    assert matrix["model_totals"]["ridge"]["median_nrmse"] == 0.10


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
    assert matrix["model_totals"]["tabpfn"]["median_nrmse"] == expected
    assert expected == 0.25
    de_cell = next(c for c in matrix["cells"] if c["country"] == "DE")
    fr_cell = next(c for c in matrix["cells"] if c["country"] == "FR")
    assert de_cell["median_nrmse"] == 0.10
    assert fr_cell["median_nrmse"] == 0.40
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
    assert len(payload["leaderboards"]["eos"]["all"]) == 2
    models = {r["model"] for r in payload["leaderboards"]["eos"]["all"]}
    assert models == {"average", "ridge"}
    assert payload["model_country"]["eos"]["all"]["models"] == ["average", "ridge"]
    assert "model_country_skilled" in payload
    ridge_board = next(r for r in payload["leaderboards"]["eos"]["all"] if r["model"] == "ridge")
    ridge_matrix = payload["model_country"]["eos"]["all"]["model_totals"]["ridge"]["median_nrmse"]
    assert ridge_board["median_nrmse"] == ridge_matrix
