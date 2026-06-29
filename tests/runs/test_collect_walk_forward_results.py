from pathlib import Path

import pandas as pd

from cybench.runs.analysis.benchmark_run_catalog import parse_benchmark_run_dir
from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.yield_quality import FLAG_YIELD
from cybench.runs.analysis.collect_walk_forward_results import (
    _filter_runs,
    aggregate_flat_metrics_across_seeds,
    discover_run_seeds,
    load_pooled_predictions,
    resolve_model_column,
    summary_rows_to_dashboard_records,
)


def test_filter_runs_by_country_and_horizon():
    runs = []
    for name in (
        "maize_AO_ridge_walk_forward_eos_20260101_120000",
        "maize_DE_ridge_walk_forward_eos_20260101_120000",
        "maize_AO_ridge_walk_forward_mid_season_20260101_120000",
    ):
        run = parse_benchmark_run_dir(name, Path("/tmp"))
        assert run is not None
        runs.append(run)
    filtered = _filter_runs(runs, country="AO", horizon="eos")
    assert len(filtered) == 1
    assert filtered[0].country == "AO"
    assert filtered[0].horizon == "eos"


def test_resolve_model_column_from_repo_config():
    assert resolve_model_column(Path("/nonexistent"), "ridge") == "Ridge"
    assert resolve_model_column(Path("/nonexistent"), "xgboost") == "XGBoostModel"


def test_discover_run_seeds(tmp_path: Path):
    run_dir = tmp_path / "maize_NL_ridge_walk_forward_eos_20260615_135937"
    for year, seed, pred in ((2016, 42, 9.5), (2016, 43, 9.6), (2017, 42, 8.5)):
        split = run_dir / str(year) / str(seed)
        split.mkdir(parents=True)
        pd.DataFrame(
            {
                "adm_id": ["NL-01"],
                "year": [year],
                "targets": [10.0],
                "preds": [pred],
            }
        ).to_csv(split / "test_preds.csv", index=False)
    assert discover_run_seeds(run_dir) == [42, 43]


def test_load_pooled_predictions_selects_seed(tmp_path: Path):
    run = parse_benchmark_run_dir(
        "maize_NL_ridge_walk_forward_eos_20260615_135937",
        tmp_path,
    )
    assert run is not None
    for seed, pred in ((42, 9.5), (43, 9.0)):
        split = tmp_path / "2016" / str(seed)
        split.mkdir(parents=True)
        pd.DataFrame(
            {
                "adm_id": ["NL-01"],
                "year": [2016],
                "targets": [10.0],
                "preds": [pred],
            }
        ).to_csv(split / "test_preds.csv", index=False)

    df, model_col = load_pooled_predictions(tmp_path, model_slug=run.model, seed=43)
    assert model_col == "Ridge"
    assert df["Ridge"].iloc[0] == 9.0


def test_aggregate_flat_metrics_across_seeds():
    per_seed = [
        {"r": 0.80, "r2": 0.60, "nrmse": 0.10, "n_regions": 5, "n_years": 2, "n_samples": 10},
        {"r": 0.70, "r2": 0.50, "nrmse": 0.12, "n_regions": 5, "n_years": 2, "n_samples": 10},
    ]
    summary = aggregate_flat_metrics_across_seeds(per_seed, [42, 43])
    assert summary["n_seeds"] == 2
    assert summary["r"] == 0.75
    assert abs(summary["r_std"] - 0.070710678) < 1e-6
    assert summary["n_regions"] == 5


def test_collect_walk_forward_run_multi_seed(tmp_path: Path):
    from cybench.runs.analysis.collect_walk_forward_results import collect_walk_forward_run

    run_dir = tmp_path / "maize_NL_ridge_walk_forward_eos_20260615_135937"
    run_dir.mkdir()
    run = parse_benchmark_run_dir(run_dir.name, run_dir)
    assert run is not None
    for seed, preds in ((42, (9.0, 11.0)), (43, (8.5, 11.5))):
        split = run_dir / "2016" / str(seed)
        split.mkdir(parents=True)
        pd.DataFrame(
            {
                "adm_id": ["NL-01", "NL-02"],
                "year": [2016, 2016],
                "targets": [10.0, 12.0],
                "preds": list(preds),
            }
        ).to_csv(split / "test_preds.csv", index=False)

    result = collect_walk_forward_run(
        run,
        data_dir=tmp_path / "data",
        quality_flags=[],
        apply_qc=False,
    )
    assert result is not None
    summary, per_seed_rows, plot_df, model_col, plot_seed = result
    assert model_col == "Ridge"
    assert plot_seed == 42
    assert summary["n_seeds"] == 2
    assert len(per_seed_rows) == 2
    assert summary["nrmse_std"] is not None
    assert not pd.isna(summary["nrmse_std"])
    assert len(plot_df) == 2


def test_load_pooled_predictions_from_test_preds(tmp_path: Path):
    run = parse_benchmark_run_dir(
        "maize_NL_ridge_walk_forward_eos_20260615_135937",
        tmp_path,
    )
    assert run is not None
    split = tmp_path / "2016" / "42"
    split.mkdir(parents=True)
    pd.DataFrame(
        {
            "adm_id": ["NL-01"],
            "year": [2016],
            "targets": [10.0],
            "preds": [9.5],
        }
    ).to_csv(split / "test_preds.csv", index=False)

    df, model_col = load_pooled_predictions(tmp_path, model_slug=run.model)
    assert model_col == "Ridge"
    assert len(df) == 1
    assert df["yield"].iloc[0] == 10.0
    assert df["Ridge"].iloc[0] == 9.5


def test_collect_applies_quality_filter(tmp_path: Path, monkeypatch):
    import cybench.config as config
    import cybench.datasets.yield_quality as yq

    data_dir = tmp_path / "data"
    country_dir = data_dir / "maize" / "NL"
    country_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "crop_name": ["maize"],
            "country_code": ["NL"],
            KEY_LOC: ["NL-01"],
            "harvest_year": [2016],
            FLAG_YIELD: [True],
            "flag_consecutive_yield": [False],
            "flag_area_outlier": [False],
        }
    ).to_csv(country_dir / "yield_quality_maize_NL.csv", index=False)

    monkeypatch.setattr(config, "PATH_DATA_DIR", str(data_dir))
    monkeypatch.setattr(yq, "PATH_DATA_DIR", str(data_dir))

    run_dir = tmp_path / "maize_NL_ridge_walk_forward_eos_20260615_135937"
    split = run_dir / "2016" / "42"
    split.mkdir(parents=True)
    pd.DataFrame(
        {
            "adm_id": ["NL-01", "NL-02"],
            "year": [2016, 2016],
            "targets": [10.0, 11.0],
            "preds": [9.5, 10.5],
        }
    ).to_csv(split / "test_preds.csv", index=False)

    df, model_col = load_pooled_predictions(run_dir, model_slug="ridge")
    filtered, n_removed = yq.apply_yield_quality_filter(
        df,
        "maize",
        "NL",
        data_dir=data_dir,
        quality_flags=[FLAG_YIELD],
    )
    assert n_removed == 1
    assert len(filtered) == 1
    assert filtered.iloc[0][KEY_LOC] == "NL-02"


def test_summary_rows_to_dashboard_records(tmp_path: Path):
    rows = [
        {
            "model": "ridge",
            "model_col": "Ridge",
            "dataset": "maize_NL",
            "horizon": "eos",
            "n_regions": 11,
            "n_years": 5,
            "r": 0.04,
            "r2": -1.46,
            "nrmse": 0.31,
            "r_spatial": -0.12,
            "r2_spatial": -0.93,
            "r_temporal": 0.18,
            "r2_temporal": -5.10,
            "r2_yearly_median": 0.42,
            "r_res": 0.05,
            "r2_res": -1.76,
            "n_seeds": 3,
            "nrmse_std": 0.012,
            "r_std": 0.004,
        }
    ]
    assets = (
        tmp_path / "preds" / "ridge_eos" / "report_assets"
    )
    assets.mkdir(parents=True)
    (assets / "maize_NL_scatter.png").write_bytes(b"png")

    records = summary_rows_to_dashboard_records(rows, tmp_path)
    assert len(records) == 10
    assert records[0]["view"] == "region_year"
    nrmse_rec = next(r for r in records if r["view"] == "region_year" and r["metric"] == "nrmse")
    assert nrmse_rec["value"] == 0.31
    assert nrmse_rec["value_std"] == 0.012
    r_rec = next(r for r in records if r["view"] == "region_year" and r["metric"] == "r")
    assert r_rec["value_std"] == 0.004
    r2_rec = next(r for r in records if r["view"] == "region_year" and r["metric"] == "r2")
    assert r2_rec["value_std"] is None
    scatter_recs = [r for r in records if r.get("images", {}).get("scatter")]
    assert scatter_recs
    assert "maize_NL" in scatter_recs[0]["images"]["scatter"]
