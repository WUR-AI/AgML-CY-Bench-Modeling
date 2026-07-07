"""Tests for pre-baked regional dashboard maps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.viz.region_map_lib import (
    build_region_map_payload,
    dataset_country_code,
    infer_pred_column,
    load_dataset_year_csvs,
    region_means,
    strip_map_pngs_from_records,
)


def test_dataset_country_code():
    assert dataset_country_code("maize_DE") == "DE"
    assert dataset_country_code("wheat_NL") == "NL"
    assert dataset_country_code("bad") == ""


def test_region_means_and_infer_pred_column():
    df = pd.DataFrame(
        {
            "adm_id": ["R1", "R1", "R2"],
            "year": [2020, 2021, 2020],
            "yield": [10.0, 12.0, 8.0],
            "Ridge": [9.5, 11.0, 7.8],
        }
    )
    assert region_means(df, "yield") == {"R1": 11.0, "R2": 8.0}
    assert infer_pred_column(df, model_col="Ridge") == "Ridge"


def test_load_dataset_year_csvs(tmp_path: Path):
    preds = tmp_path / "preds" / "ridge_eos"
    preds.mkdir(parents=True)
    pd.DataFrame(
        {"adm_id": ["R1"], "year": [2020], "yield": [10.0], "Ridge": [9.0]}
    ).to_csv(preds / "maize_DE_h10_year_2020.csv", index=False)
    pd.DataFrame(
        {"adm_id": ["R1"], "year": [2021], "yield": [11.0], "Ridge": [10.0]}
    ).to_csv(preds / "maize_DE_h10_year_2021.csv", index=False)
    loaded = load_dataset_year_csvs(preds, "maize_DE")
    assert loaded is not None
    assert len(loaded) == 2


def test_build_region_map_payload(tmp_path: Path):
    output_dir = tmp_path / "paper_walk_forward_de_eos_v1"
    preds = output_dir / "preds" / "ridge_eos"
    preds.mkdir(parents=True)
    for year, y, p in ((2020, 10.0, 9.0), (2021, 11.0, 10.5)):
        pd.DataFrame(
            {
                "adm_id": ["DE01", "DE02"],
                "year": [year, year],
                "yield": [y, y + 1],
                "Ridge": [p, p + 0.2],
            }
        ).to_csv(preds / f"maize_DE_h10_year_{year}.csv", index=False)

    summary_rows = [
        {
            "dataset": "maize_DE",
            "model": "ridge",
            "horizon": "eos",
            "model_col": "Ridge",
        }
    ]
    payload = build_region_map_payload(output_dir, summary_rows)
    assert "maize_DE" in payload["datasets"]
    ds = payload["datasets"]["maize_DE"]
    assert ds["country"] == "DE"
    assert ds["actual"]["DE01"] == pytest.approx(10.5)
    assert ds["models"]["ridge"]["DE01"] == pytest.approx(9.75)


def test_strip_map_pngs_from_records():
    records = [
        {
            "model": "ridge",
            "images": {
                "map_actual": "a.png",
                "map_pred": "b.png",
                "scatter": "c.png",
            },
        }
    ]
    strip_map_pngs_from_records(records)
    assert "map_actual" not in records[0]["images"]
    assert "map_pred" not in records[0]["images"]
    assert records[0]["images"]["scatter"] == "c.png"
