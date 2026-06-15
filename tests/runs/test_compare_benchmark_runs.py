from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.analysis.benchmark_run_catalog import (
    discover_benchmark_runs,
    load_run_metrics,
    parse_benchmark_run_dir,
    parse_group_spec,
)
from cybench.runs.analysis.compare_benchmark_runs import compare_groups


def test_parse_group_spec():
    assert parse_group_spec("wf=walk_forward/eos") == ("wf", "walk_forward", "eos")
    assert parse_group_spec("scr=screening") == ("scr", "screening", None)


def test_parse_benchmark_run_dir():
    run = parse_benchmark_run_dir(
        "maize_NL_ridge_walk_forward_eos_20260615_135937",
        Path("/tmp/x"),
    )
    assert run is not None
    assert run.phase == "walk_forward"
    assert run.horizon == "eos"
    assert run.model == "ridge"


def test_compare_groups_screening_vs_walk_forward(tmp_path: Path):
    baselines = tmp_path / "baselines"
    screen = baselines / "maize_NL_ridge_screening_eos_20260615_120000"
    wf = baselines / "maize_NL_ridge_walk_forward_eos_20260615_130000"
    split = screen / "2016_2017" / "42"
    split.mkdir(parents=True)
    (split / "report_metrics.yaml").write_text(
        "n_regions: 5\nn_years: 2\nregion_year:\n  r: 0.5\n  r2: 0.2\n  nrmse: 0.20\n"
        "spatial:\n  r: 0.4\n  r2: 0.1\ntemporal:\n  r: 0.3\n  r2: 0.0\n"
        "  r_res: 0.1\n  r2_res: -0.1\n"
    )
    wf_split = wf / "2016" / "42"
    wf_split.mkdir(parents=True)
    pd.DataFrame(
        {"adm_id": ["NL-01"], "year": [2016], "targets": [10.0], "preds": [9.0]}
    ).to_csv(wf_split / "test_preds.csv", index=False)
    wf_split2 = wf / "2017" / "42"
    wf_split2.mkdir(parents=True)
    pd.DataFrame(
        {"adm_id": ["NL-01"], "year": [2017], "targets": [11.0], "preds": [10.0]}
    ).to_csv(wf_split2 / "test_preds.csv", index=False)

    df = compare_groups(
        baselines,
        [
            ("screening", "screening", "eos"),
            ("walk_forward", "walk_forward", "eos"),
        ],
    )
    assert len(df) == 1
    assert "screening__nrmse" in df.columns
    assert "walk_forward__nrmse" in df.columns
    assert "delta__nrmse" in df.columns
    assert "delta__r2" in df.columns


def test_compare_groups_cross_horizon(tmp_path: Path):
    baselines = tmp_path / "baselines"
    eos = baselines / "maize_NL_ridge_walk_forward_eos_20260615_120000"
    mid = baselines / "maize_NL_ridge_walk_forward_mid_season_20260615_130000"
    for run_dir, target in ((eos, 10.0), (mid, 11.0)):
        split = run_dir / "2016" / "42"
        split.mkdir(parents=True)
        pd.DataFrame(
            {"adm_id": ["NL-01"], "year": [2016], "targets": [target], "preds": [9.0]}
        ).to_csv(split / "test_preds.csv", index=False)

    df = compare_groups(
        baselines,
        [
            ("eos", "walk_forward", "eos"),
            ("mid", "walk_forward", "mid_season"),
        ],
    )
    assert len(df) == 1
    assert df.loc[0, "eos__horizon"] == "eos"
    assert df.loc[0, "mid__horizon"] == "mid_season"
    assert "delta__nrmse" in df.columns


def test_discover_benchmark_runs_filters_horizon(tmp_path: Path):
    baselines = tmp_path / "baselines"
    (baselines / "maize_NL_ridge_walk_forward_eos_20260615_120000").mkdir(parents=True)
    (baselines / "maize_NL_ridge_walk_forward_mid_season_20260615_120000").mkdir(parents=True)
    eos = discover_benchmark_runs(baselines, phase="walk_forward", horizon="eos")
    assert len(eos) == 1
    assert eos[0].horizon == "eos"


def test_screening_metrics_path_skips_single_year_folders(tmp_path: Path):
    from cybench.runs.analysis.benchmark_run_catalog import _screening_metrics_path

    wf_split = tmp_path / "2016" / "42"
    wf_split.mkdir(parents=True)
    (wf_split / "report_metrics.yaml").write_text("region_year: {}\n")

    screen_split = tmp_path / "2016_2017_2018" / "42"
    screen_split.mkdir(parents=True)
    (screen_split / "report_metrics.yaml").write_text(
        "n_regions: 10\nn_years: 3\nregion_year:\n  r: 0.5\n  r2: 0.2\n  nrmse: 0.1\n"
    )

    assert _screening_metrics_path(tmp_path) == screen_split / "report_metrics.yaml"
