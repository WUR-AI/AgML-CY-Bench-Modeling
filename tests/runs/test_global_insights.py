"""Tests for global insights aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cybench.runs.analysis.global_insights_lib import (
    aggregate_model_leaderboard,
    compare_horizons,
    load_summary_frame,
    parse_paper_dir_name,
)


def test_parse_paper_dir_name():
    assert parse_paper_dir_name("paper_walk_forward_de_eos_v1") == ("DE", "eos", 1)
    assert parse_paper_dir_name("paper_walk_forward_pl_mid_v1") == ("PL", "mid", 1)
    assert parse_paper_dir_name("paper_walk_forward") is None


def test_aggregate_model_leaderboard_weighted_nrmse():
    df = pd.DataFrame(
        [
            {"model": "ridge", "nrmse": 0.10, "r2": 0.9, "n_samples": 100, "country": "DE"},
            {"model": "ridge", "nrmse": 0.30, "r2": 0.5, "n_samples": 10, "country": "FR"},
            {"model": "xgboost", "nrmse": 0.20, "r2": 0.7, "n_samples": 50, "country": "DE"},
        ]
    )
    board = aggregate_model_leaderboard(df)
    assert list(board["model"]) == ["ridge", "xgboost"]
    assert board.loc[board["model"] == "ridge", "weighted_nrmse"].iloc[0] < 0.15
    assert int(board.loc[board["model"] == "ridge", "total_samples"].iloc[0]) == 110


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
