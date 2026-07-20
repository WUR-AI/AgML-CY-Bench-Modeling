"""Build map payload and world GeoJSON for the dashboard index page."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cybench.runs.analysis.publish_dashboard_bundle import IndexEntry, _COUNTRY_NAMES

if TYPE_CHECKING:
    import geopandas as gpd

_BUNDLED_GEOJSON = (
    Path(__file__).resolve().parent.parent / "viz" / "data" / "world_countries_50m.geojson"
)
_BUNDLED_AGML_LOGO = (
    Path(__file__).resolve().parent.parent / "viz" / "assets" / "agml-logo.png"
)
_SLUG_RE = re.compile(
    r"^([a-z]{2})_walk_forward_(eos|early|early_season|mid|mid_season|qtr|quarter_season)(?:_v\d+)?$", re.IGNORECASE
)

# CY-Bench country codes that differ from Natural Earth ISO_A2 (e.g. EL = Greece).
_CYBENCH_TO_MAP_ISO: dict[str, str] = {
    "EL": "GR",
}

# Natural Earth admin-0 polygons sometimes include overseas departments under the
# parent ISO (e.g. French Guiana under FR). CY-Bench countries are metropolitan;
# clip via cybench.util.geo.explode_metropolitan_map_units.
_OVERSEAS_MAP_ISO = "XX"
_EXCLUDED_MAP_ISOS = frozenset({"AQ"})


def map_iso_for_cybencH(cc: str) -> str:
    key = cc.upper()
    return _CYBENCH_TO_MAP_ISO.get(key, key)


def country_display_name(cc: str) -> str:
    return _COUNTRY_NAMES.get(cc.lower(), cc.upper())


def _version_from_slug(slug: str) -> int:
    match = re.search(r"_v(\d+)$", slug, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _horizon_key_from_slug(slug: str) -> str | None:
    match = _SLUG_RE.match(slug)
    if not match:
        return None
    hz = match.group(2).lower()
    if hz in {"mid", "mid_season"}:
        return "mid"
    if hz in {"qtr", "quarter_season"}:
        return "qtr"
    if hz in {"early", "early_season"}:
        return "early"
    return hz


def group_walk_forward_entries(
    entries: list[IndexEntry],
    *,
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Group walk-forward index entries by ISO2 country code."""
    from cybench.util.benchmark_scope import is_benchmark_evaluation_country

    walk_forward = [
        e
        for e in entries
        if e.kind == "walk_forward" and e.country_code
    ]
    # Higher vN last so it wins when the same country×horizon appears twice.
    walk_forward.sort(key=lambda e: _version_from_slug(e.slug))

    by_cc: dict[str, dict[str, Any]] = {}
    for entry in walk_forward:
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
                "qtr": None,
                "early": None,
            },
        )
        row[hz] = entry.href
    return [
        row
        for row in sorted(by_cc.values(), key=lambda r: r["name"])
        if is_benchmark_evaluation_country(row["cc"], data_dir=data_dir)
    ]


def build_landing_stats(
    output_root: Path | None,
    *,
    version: int = 4,
) -> dict[str, Any]:
    """Derive landing-page headline counts from walk-forward summaries."""
    from cybench.runs.analysis.global_insights_lib import (
        discover_summary_tables,
        load_summary_frame,
        matrix_axes_payload,
    )

    n_axes = matrix_axes_payload()
    n_skill_dimensions = len(n_axes)
    empty: dict[str, Any] = {
        "n_country_crop_datasets": None,
        "n_maize_datasets": None,
        "n_wheat_datasets": None,
        "n_regions": None,
        "n_models": None,
        "n_skill_dimensions": n_skill_dimensions,
        "skill_dimension_labels": [str(ax["label"]) for ax in n_axes],
    }
    if output_root is None:
        return empty
    paths = discover_summary_tables(Path(output_root), version=version)
    if not paths:
        return empty
    df = load_summary_frame(paths)
    if df.empty or "model" not in df.columns:
        return empty

    crop_col = "crop" if "crop" in df.columns else None
    country_col = "country" if "country" in df.columns else None
    n_country_crop = None
    n_maize = None
    n_wheat = None
    if crop_col and country_col:
        pairs = df[[crop_col, country_col]].drop_duplicates()
        n_country_crop = int(len(pairs))
        crops = pairs[crop_col].astype(str).str.casefold()
        n_maize = int((crops == "maize").sum())
        n_wheat = int(crops.str.contains("wheat", regex=False).sum())

    n_regions = None
    if crop_col and country_col and "n_regions" in df.columns:
        per_dataset = (
            df.groupby([crop_col, country_col], dropna=False)["n_regions"]
            .max()
            .dropna()
        )
        if len(per_dataset):
            n_regions = int(per_dataset.sum())

    return {
        "n_country_crop_datasets": n_country_crop,
        "n_maize_datasets": n_maize,
        "n_wheat_datasets": n_wheat,
        "n_regions": n_regions,
        "n_models": int(df["model"].nunique()),
        "n_skill_dimensions": n_skill_dimensions,
        "skill_dimension_labels": [str(ax["label"]) for ax in n_axes],
    }


def build_index_map_payload(
    entries: list[IndexEntry],
    *,
    publish_root: Path | None = None,
    output_root: Path | None = None,
    version: int = 4,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    screening = [e for e in entries if e.kind == "screening"]
    countries = group_walk_forward_entries(entries, data_dir=data_dir)
    return {
        "countries": countries,
        "n_countries": len(countries),
        "stats": build_landing_stats(output_root, version=version),
        "has_insights": bool(publish_root and (publish_root / "insights.html").is_file()),
        "has_screening": bool(screening),
        "screening_href": screening[0].href if screening else None,
    }


def export_world_geojson(
    dest: Path,
    *,
    scale: str | None = None,
    simplify: float | None = None,
) -> Path:
    """Write admin-0 countries GeoJSON for dashboard world maps."""
    import geopandas as gpd

    from cybench.util.geo import explode_metropolitan_map_units, world_shape_path

    shp = world_shape_path(scale)
    resolved_scale = scale or _scale_from_shape_path(shp)
    if simplify is None:
        # 50m/10m are already generalised; extra simplify blurs coastlines.
        simplify = 0.0 if resolved_scale in {"10", "50"} else 0.05

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
    keep = keep[~keep["ISO_A2"].isin(_EXCLUDED_MAP_ISOS)]
    if simplify > 0:
        keep["geometry"] = keep.geometry.simplify(simplify, preserve_topology=True)
    keep = explode_metropolitan_map_units(keep)
    dest.parent.mkdir(parents=True, exist_ok=True)
    keep.to_file(dest, driver="GeoJSON")
    return dest


def _scale_from_shape_path(shp: str) -> str:
    for token in ("10", "50", "110"):
        if f"ne_{token}m_" in shp:
            return token
    return "110"


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


def ensure_agml_logo(publish_root: Path) -> str:
    """Copy bundled AgML logo into publish_root/assets; return relative href."""
    if not _BUNDLED_AGML_LOGO.is_file():
        raise FileNotFoundError(f"Missing bundled AgML logo: {_BUNDLED_AGML_LOGO}")
    assets_dir = publish_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / "agml-logo.png"
    shutil.copy2(_BUNDLED_AGML_LOGO, dest)
    return "assets/agml-logo.png"


def build_index_map_html(payload: dict[str, Any], *, geojson_href: str) -> str:
    template_path = Path(__file__).resolve().parent.parent / "viz" / "index_map_template.html"
    template = template_path.read_text(encoding="utf-8")
    payload = {**payload, "geojson_href": geojson_href}
    return template.replace("__MAP_DATA_JSON__", json.dumps(payload))
