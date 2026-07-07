"""Tests for pre-baked regional dashboard maps."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cybench.runs.viz.region_map_lib import (
    benchmark_locs_by_country,
    build_region_map_payload,
    bundle_region_map_assets,
    dataset_country_code,
    dataset_crop,
    infer_pred_column,
    load_dataset_year_csvs,
    prepare_geometry_for_geojson,
    region_means,
    region_values_by_year,
)


def test_dataset_country_code():
    assert dataset_country_code("maize_DE") == "DE"
    assert dataset_country_code("wheat_NL") == "NL"
    assert dataset_country_code("bad") == ""


def test_dataset_crop():
    assert dataset_crop("maize_DE") == "maize"
    assert dataset_crop("wheat_NL") == "wheat"


def test_prepare_geometry_keeps_ring_orientation():
    from shapely.geometry import Polygon

    cw = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    kept = prepare_geometry_for_geojson(cw)
    assert kept is not None
    ring = list(kept.exterior.coords)
    signed = sum(
        (ring[i + 1][0] - ring[i][0]) * (ring[i + 1][1] + ring[i][1])
        for i in range(len(ring) - 1)
    )
    assert signed > 0


def test_prepare_geometry_drops_dateline_spanning_polygon():
    from shapely.geometry import Polygon

    huge = Polygon([(-179, 50), (179, 50), (179, 55), (-179, 55), (-179, 50)])
    assert prepare_geometry_for_geojson(huge) is None


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
    assert region_values_by_year(df, "yield") == {
        "2020": {"R1": 10.0, "R2": 8.0},
        "2021": {"R1": 12.0},
    }
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
    assert ds["crop"] == "maize"
    assert ds["yield_range"] == {"min": 0, "max": 14}
    assert payload["yield_ranges"]["maize"]["max"] == 14
    assert ds["actual"]["DE01"] == pytest.approx(10.5)
    assert ds["models"]["ridge"]["DE01"] == pytest.approx(9.75)
    assert ds["years"] == [2020, 2021]
    assert ds["actual_by_year"]["2020"]["DE01"] == pytest.approx(10.0)
    assert ds["actual_by_year"]["2021"]["DE01"] == pytest.approx(11.0)
    assert ds["models_by_year"]["ridge"]["2020"]["DE01"] == pytest.approx(9.0)
    assert ds["models_by_year"]["ridge"]["2021"]["DE01"] == pytest.approx(10.5)


def test_bundle_region_map_survives_referenced_assets(tmp_path: Path, monkeypatch):
    """GeoJSON must be written after bundle_referenced_assets (which wipes assets/)."""
    from cybench.runs.analysis.collect_walk_forward_results import (
        write_model_comparison_dashboard,
    )

    output_dir = tmp_path / "paper_walk_forward_us_eos_v2"
    preds = output_dir / "preds" / "ridge_eos"
    preds.mkdir(parents=True)
    pd.DataFrame(
        {
            "adm_id": ["US-01-001"],
            "year": [2020],
            "yield": [10.0],
            "Ridge": [9.5],
        }
    ).to_csv(preds / "maize_US_h10_year_2020.csv", index=False)

    scatter_src = preds / "report_assets"
    scatter_src.mkdir(parents=True)
    scatter_src.joinpath("maize_US_scatter.png").write_bytes(b"png")

    summary_rows = [
        {
            "dataset": "maize_US",
            "model": "ridge",
            "horizon": "eos",
            "model_col": "Ridge",
            "nrmse": 0.1,
            "r2": 0.8,
            "r_spatial": 0.5,
            "r_spatial_agg": 0.5,
            "r_temporal": 0.4,
            "r_temporal_agg": 0.4,
            "r_res": 0.3,
            "r2_res": 0.2,
            "n_regions": 1,
            "n_years": 1,
            "n_samples": 1,
        }
    ]

    def _fake_export(country_code: str, dest: Path, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        locs = kwargs.get("locations") or {"US-01-001"}
        features = [
            {
                "type": "Feature",
                "properties": {"loc": loc},
                "geometry": {"type": "Point", "coordinates": [0, 0]},
            }
            for loc in sorted(locs)
        ]
        dest.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}),
            encoding="utf-8",
        )
        return dest

    monkeypatch.setattr(
        "cybench.runs.viz.region_map_lib.export_region_geojson",
        _fake_export,
    )

    write_model_comparison_dashboard(output_dir, summary_rows, bundle_assets=True)
    geojson = output_dir / "assets" / "regions_US.geojson"
    assert geojson.is_file(), "regions_US.geojson must survive asset bundling"
    html = (output_dir / "compare_models.html").read_text(encoding="utf-8")
    assert "regions_US.geojson" in html
    assert "border_US.geojson" not in html
    assert (output_dir / "assets" / "maize_US_scatter.png").is_file()


def test_benchmark_locs_by_country():
    payload = {
        "datasets": {
            "maize_DE": {
                "country": "DE",
                "actual": {"DE01": 1.0},
                "models": {"ridge": {"DE01": 0.9, "DE02": 0.8}},
            },
            "wheat_NL": {
                "country": "NL",
                "actual": {"NL01": 2.0},
                "models": {},
            },
        }
    }
    locs = benchmark_locs_by_country(payload)
    assert locs["DE"] == {"DE01", "DE02"}
    assert locs["NL"] == {"NL01"}


def test_bundle_region_map_assets_filters_locations(tmp_path: Path, monkeypatch):
    payload = build_region_map_payload(
        tmp_path,
        [{"dataset": "maize_DE", "model": "ridge", "horizon": "eos"}],
    )
    payload["datasets"] = {
        "maize_DE": {
            "country": "DE",
            "crop": "maize",
            "actual": {"DE01": 1.0},
            "models": {"ridge": {"DE01": 0.9}},
        }
    }
    payload["geojson_by_country"] = {"DE": ""}
    seen: dict[str, set[str] | None] = {}

    def _fake_export(country_code: str, dest: Path, **kwargs):
        seen["locations"] = kwargs.get("locations")
        dest.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        return dest

    monkeypatch.setattr(
        "cybench.runs.viz.region_map_lib.export_region_geojson",
        _fake_export,
    )
    out = bundle_region_map_assets(payload, tmp_path)
    assert seen["locations"] == {"DE01"}
    assert out["geojson_by_country"]["DE"] == "assets/regions_DE.geojson"
    assert "border_geojson_by_country" not in out
