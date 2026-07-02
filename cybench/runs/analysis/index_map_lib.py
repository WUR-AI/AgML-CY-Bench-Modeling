"""Build map payload and world GeoJSON for the dashboard index page."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from cybench.runs.analysis.publish_dashboard_bundle import IndexEntry, _COUNTRY_NAMES

_BUNDLED_GEOJSON = (
    Path(__file__).resolve().parent.parent / "viz" / "data" / "world_countries_110m.geojson"
)
_SLUG_RE = re.compile(
    r"^([a-z]{2})_walk_forward_(eos|mid|mid_season)(?:_v\d+)?$", re.IGNORECASE
)

# CY-Bench country codes that differ from Natural Earth ISO_A2 (e.g. EL = Greece).
_CYBENCH_TO_MAP_ISO: dict[str, str] = {
    "EL": "GR",
}

# Natural Earth admin-0 polygons sometimes include overseas departments under the
# parent ISO (e.g. French Guiana under FR). CY-Bench countries are metropolitan;
# clip to these WGS84 bounding boxes (min_lon, min_lat, max_lon, max_lat).
_METROPOLITAN_BBOX_WGS84: dict[str, tuple[float, float, float, float]] = {
    "FR": (-5.5, 41.0, 10.0, 51.5),
}
_OVERSEAS_MAP_ISO = "XX"


def _explode_metropolitan_map_units(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode multipolygons; re-tag overseas parts so they are not benchmark-colored."""
    if frame.empty:
        return frame
    exploded = frame.explode(index_parts=True).reset_index(drop=True)
    if not _METROPOLITAN_BBOX_WGS84:
        return exploded

    iso = exploded["ISO_A2"].astype(str)
    centroids = exploded.geometry.representative_point()
    for map_iso, (min_lon, min_lat, max_lon, max_lat) in _METROPOLITAN_BBOX_WGS84.items():
        parent = iso == map_iso
        if not parent.any():
            continue
        inside = (
            (centroids.x >= min_lon)
            & (centroids.x <= max_lon)
            & (centroids.y >= min_lat)
            & (centroids.y <= max_lat)
        )
        overseas = parent & ~inside
        if overseas.any():
            exploded.loc[overseas, "ISO_A2"] = _OVERSEAS_MAP_ISO
    return exploded


def map_iso_for_cybencH(cc: str) -> str:
    key = cc.upper()
    return _CYBENCH_TO_MAP_ISO.get(key, key)


def country_display_name(cc: str) -> str:
    return _COUNTRY_NAMES.get(cc.lower(), cc.upper())


def _horizon_key_from_slug(slug: str) -> str | None:
    match = _SLUG_RE.match(slug)
    if not match:
        return None
    hz = match.group(2).lower()
    if hz in {"mid", "mid_season"}:
        return "mid"
    return hz


def group_walk_forward_entries(entries: list[IndexEntry]) -> list[dict[str, Any]]:
    """Group walk-forward index entries by ISO2 country code."""
    by_cc: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.kind != "walk_forward" or not entry.country_code:
            continue
        cc = entry.country_code.upper()
        hz = _horizon_key_from_slug(entry.slug)
        if hz is None:
            continue
        row = by_cc.setdefault(
            cc,
            {
                "cc": cc,
                "map_cc": map_iso_for_cybencH(cc),
                "name": country_display_name(cc),
                "eos": None,
                "mid": None,
            },
        )
        row[hz] = entry.href
    return sorted(by_cc.values(), key=lambda r: r["name"])


def build_index_map_payload(
    entries: list[IndexEntry],
    *,
    publish_root: Path | None = None,
) -> dict[str, Any]:
    screening = [e for e in entries if e.kind == "screening"]
    countries = group_walk_forward_entries(entries)
    return {
        "countries": countries,
        "n_countries": len(countries),
        "has_insights": bool(publish_root and (publish_root / "insights.html").is_file()),
        "has_model_families": bool(
            publish_root and (publish_root / "model_families.html").is_file()
        ),
        "has_screening": bool(screening),
        "screening_href": screening[0].href if screening else None,
    }


def export_world_geojson(dest: Path, *, simplify: float = 0.08) -> Path:
    """Write simplified admin-0 countries GeoJSON for the index map."""
    import geopandas as gpd

    from cybench.util.geo import world_shape_path

    shp = world_shape_path("110")
    world = gpd.read_file(shp).to_crs(4326)
    iso_col = next(
        (c for c in ("ISO_A2", "iso_a2") if c in world.columns),
        None,
    )
    iso_eh_col = next(
        (c for c in ("ISO_A2_EH", "iso_a2_eh") if c in world.columns),
        None,
    )
    wb_col = next((c for c in ("WB_A2", "wb_a2") if c in world.columns), None)
    name_col = next(
        (c for c in ("NAME", "name", "ADMIN") if c in world.columns),
        None,
    )
    if iso_col is None:
        raise ValueError(f"No ISO_A2 column in {shp}; columns={list(world.columns)}")

    keep = world[[iso_col, name_col, "geometry"]].copy() if name_col else world[[iso_col, "geometry"]].copy()
    keep = keep.rename(columns={iso_col: "ISO_A2", name_col: "NAME"} if name_col else {iso_col: "ISO_A2"})
    if "NAME" not in keep.columns:
        keep["NAME"] = keep["ISO_A2"]
    # Natural Earth marks some countries (e.g. France, Norway) as ISO_A2 "-99".
    iso = keep["ISO_A2"].astype(str)
    bad = iso.isna() | (iso == "-99") | (iso == "nan")
    if iso_eh_col is not None:
        iso = iso.where(~bad, world[iso_eh_col].astype(str))
        bad = iso.isna() | (iso == "-99") | (iso == "nan")
    if wb_col is not None:
        iso = iso.where(~bad, world[wb_col].astype(str))
    keep["ISO_A2"] = iso
    keep = keep[keep["ISO_A2"].notna() & (keep["ISO_A2"] != "-99") & (keep["ISO_A2"] != "nan")]
    keep["geometry"] = keep.geometry.simplify(simplify, preserve_topology=True)
    keep = _explode_metropolitan_map_units(keep)
    dest.parent.mkdir(parents=True, exist_ok=True)
    keep.to_file(dest, driver="GeoJSON")
    return dest


def ensure_world_geojson(publish_root: Path) -> str:
    """Copy or export world GeoJSON into publish_root/assets; return relative href."""
    assets_dir = publish_root / "assets"
    dest = assets_dir / "world_countries.geojson"
    rel = "assets/world_countries.geojson"

    try:
        export_world_geojson(dest)
        return rel
    except (ImportError, FileNotFoundError, OSError) as exc:
        if _BUNDLED_GEOJSON.is_file():
            assets_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_BUNDLED_GEOJSON, dest)
            return rel
        raise RuntimeError(
            "Could not build world map geometry. Install geopandas + Natural Earth "
            "shapefiles under data_preparation/, or add "
            f"{_BUNDLED_GEOJSON.name} to the engine repo."
        ) from exc


def build_index_map_html(payload: dict[str, Any], *, geojson_href: str) -> str:
    template_path = Path(__file__).resolve().parent.parent / "viz" / "index_map_template.html"
    template = template_path.read_text(encoding="utf-8")
    payload = {**payload, "geojson_href": geojson_href}
    return template.replace("__MAP_DATA_JSON__", json.dumps(payload))
