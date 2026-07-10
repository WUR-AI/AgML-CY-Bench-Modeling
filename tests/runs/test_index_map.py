"""Tests for map-based dashboard index."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.analysis.index_map_lib import (
    build_index_map_payload,
    export_world_geojson,
    group_walk_forward_entries,
    map_iso_for_cybencH,
)
from cybench.util.geo import world_shape_path
from cybench.runs.analysis.publish_dashboard_bundle import IndexEntry


def test_map_iso_alias_greece():
    assert map_iso_for_cybencH("EL") == "GR"
    assert map_iso_for_cybencH("DE") == "DE"


def test_group_walk_forward_entries():
    entries = [
        IndexEntry(
            href="de_walk_forward_eos_v1/dashboard.html",
            slug="de_walk_forward_eos_v1",
            title="Germany",
            subtitle="eos",
            country_code="DE",
            kind="walk_forward",
        ),
        IndexEntry(
            href="de_walk_forward_mid_v1/dashboard.html",
            slug="de_walk_forward_mid_v1",
            title="Germany",
            subtitle="mid",
            country_code="DE",
            kind="walk_forward",
        ),
        IndexEntry(
            href="el_walk_forward_eos_v1/dashboard.html",
            slug="el_walk_forward_eos_v1",
            title="Greece",
            subtitle="eos",
            country_code="EL",
            kind="walk_forward",
        ),
    ]
    grouped = group_walk_forward_entries(entries)
    de = next(r for r in grouped if r["cc"] == "DE")
    el = next(r for r in grouped if r["cc"] == "EL")
    assert de["eos"] and de["mid"]
    assert el["map_cc"] == "GR"
    assert el["eos"]


def test_group_walk_forward_entries_qtr_and_latest_version():
    entries = [
        IndexEntry(
            href="de_walk_forward_qtr_v1/dashboard.html",
            slug="de_walk_forward_qtr_v1",
            title="Germany",
            subtitle="qtr",
            country_code="DE",
            kind="walk_forward",
        ),
        IndexEntry(
            href="de_walk_forward_qtr_v2/dashboard.html",
            slug="de_walk_forward_qtr_v2",
            title="Germany",
            subtitle="qtr",
            country_code="DE",
            kind="walk_forward",
        ),
    ]
    grouped = group_walk_forward_entries(entries)
    de = next(r for r in grouped if r["cc"] == "DE")
    assert de["qtr"] == "de_walk_forward_qtr_v2/dashboard.html"


def test_export_world_geojson_includes_france(tmp_path: Path):
    try:
        world_shape_path("110")
    except FileNotFoundError:
        return
    dest = tmp_path / "world_countries.geojson"
    export_world_geojson(dest, simplify=0.2)
    text = dest.read_text(encoding="utf-8")
    assert '"ISO_A2": "FR"' in text or '"ISO_A2":"FR"' in text
    assert '"ISO_A2": "AQ"' not in text and '"ISO_A2":"AQ"' not in text

    import geopandas as gpd

    world = gpd.read_file(dest)
    fr = world[world["ISO_A2"] == "FR"]
    assert not fr.empty
    for _, row in fr.iterrows():
        c = row.geometry.centroid
        assert c.y > 30, f"FR polygon should be metropolitan Europe, got lat={c.y}"
        assert c.x > -15, f"FR polygon should not be in South America, got lon={c.x}"
    # French Guiana should be detached from FR (neutral gray on the map).
    overseas = world[world["ISO_A2"] == "XX"]
    assert not overseas.empty
    assert any(row.geometry.centroid.x < -30 for _, row in overseas.iterrows())


def test_export_world_geojson_excludes_alaska_from_us(tmp_path: Path):
    try:
        world_shape_path("110")
    except FileNotFoundError:
        return
    dest = tmp_path / "world_countries.geojson"
    export_world_geojson(dest, simplify=0.2)
    import geopandas as gpd

    world = gpd.read_file(dest)
    us = world[world["ISO_A2"] == "US"]
    assert not us.empty
    for _, row in us.iterrows():
        c = row.geometry.centroid
        assert c.y < 55, f"US polygon should be CONUS, got lat={c.y}"
        assert c.x > -130, f"US polygon should not be in Alaska, got lon={c.x}"
    overseas = world[world["ISO_A2"] == "XX"]
    assert not overseas.empty
    assert any(row.geometry.centroid.y > 55 for _, row in overseas.iterrows())


def test_build_index_map_payload(tmp_path: Path):
    (tmp_path / "insights.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "model_families.html").write_text("<html></html>", encoding="utf-8")
    entries = [
        IndexEntry(
            href="pl_walk_forward_eos_v1/dashboard.html",
            slug="pl_walk_forward_eos_v1",
            title="Poland",
            subtitle="eos",
            country_code="PL",
            kind="walk_forward",
        ),
    ]
    payload = build_index_map_payload(entries, publish_root=tmp_path)
    assert payload["has_insights"] is True
    assert payload["has_model_families"] is True
    assert payload["n_countries"] == 1
    assert payload["countries"][0]["cc"] == "PL"
