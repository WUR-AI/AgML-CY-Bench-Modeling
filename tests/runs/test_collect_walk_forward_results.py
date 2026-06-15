from pathlib import Path

import pandas as pd

from cybench.runs.analysis.benchmark_run_catalog import parse_benchmark_run_dir
from cybench.runs.analysis.collect_walk_forward_results import (
    load_pooled_predictions,
    resolve_model_column,
    summary_rows_to_dashboard_records,
)


def test_resolve_model_column_from_repo_config():
    assert resolve_model_column(Path("/nonexistent"), "ridge") == "Ridge"
    assert resolve_model_column(Path("/nonexistent"), "xgboost") == "XGBoostModel"


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
            "r_res": 0.05,
            "r2_res": -1.76,
        }
    ]
    assets = (
        tmp_path / "preds" / "ridge_eos" / "report_assets"
    )
    assets.mkdir(parents=True)
    (assets / "maize_NL_scatter.png").write_bytes(b"png")

    records = summary_rows_to_dashboard_records(rows, tmp_path)
    assert len(records) == 9
    assert records[0]["view"] == "region_year"
    scatter_recs = [r for r in records if r.get("images", {}).get("scatter")]
    assert scatter_recs
    assert "maize_NL" in scatter_recs[0]["images"]["scatter"]
