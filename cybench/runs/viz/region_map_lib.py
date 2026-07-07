"""Pre-bake regional map geometry and yield values for country dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cybench.config import CROP_YIELD_RANGES, KEY_LOC, KEY_TARGET, KEY_YEAR

try:
    from shapely.geometry.base import BaseGeometry
except ImportError:  # pragma: no cover
    BaseGeometry = object  # type: ignore[misc,assignment]

# Skip geometries that break map fitting (dateline-spanning admin units).
_MAX_MAP_LON_SPAN = 60.0
_MAX_MAP_LAT_SPAN = 40.0

_NON_VALUE_COLS = frozenset(
    {KEY_LOC, "adm_id", "year", "country_code", "crop", KEY_TARGET, "yield"}
)


def dataset_country_code(dataset: str) -> str:
    parts = str(dataset).split("_")
    if len(parts) >= 2 and len(parts[1]) == 2:
        return parts[1].upper()
    return ""


def dataset_crop(dataset: str) -> str:
    parts = str(dataset).split("_")
    return parts[0].lower() if parts else ""


def prepare_geometry_for_geojson(geometry: BaseGeometry | None) -> BaseGeometry | None:
    """Drop pathological footprints; ring winding is fixed in the dashboard (D3/SVG)."""
    if geometry is None or geometry.is_empty:
        return None
    minx, miny, maxx, maxy = geometry.bounds
    if (maxx - minx) > _MAX_MAP_LON_SPAN or (maxy - miny) > _MAX_MAP_LAT_SPAN:
        return None
    return geometry


def export_region_geojson(
    country_code: str,
    dest: Path,
    *,
    simplify: float = 0.01,
    locations: set[str] | frozenset[str] | None = None,
) -> Path | None:
    """Write simplified admin-region GeoJSON for benchmark locations in one country."""
    try:
        from cybench.util.geo import get_shapes_from_polygons
    except ImportError:
        return None

    country = country_code.upper()
    try:
        gdf = get_shapes_from_polygons(country)
    except (FileNotFoundError, OSError, ValueError):
        return None

    loc_col = KEY_LOC if KEY_LOC in gdf.columns else "adm_id"
    if loc_col not in gdf.columns:
        return None

    slim = gdf[[loc_col, "geometry"]].copy()
    slim["geometry"] = slim.geometry.simplify(simplify, preserve_topology=True)
    slim["geometry"] = slim.geometry.map(prepare_geometry_for_geojson)
    slim = slim[slim.geometry.notna()].copy()
    if slim.empty:
        return None
    slim = slim.rename(columns={loc_col: "loc"})
    slim["loc"] = slim["loc"].astype(str)
    if locations is not None:
        slim = slim[slim["loc"].isin(locations)]
    if slim.empty:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    slim.to_file(dest, driver="GeoJSON")
    return dest


def preds_dir_for_row(output_dir: Path, row: dict[str, Any]) -> Path | None:
    model = str(row["model"])
    horizon = row.get("horizon")
    path = output_dir / "preds" / f"{model}_{horizon}"
    return path if path.is_dir() else None


def load_dataset_year_csvs(preds_dir: Path, dataset: str) -> pd.DataFrame | None:
    files = sorted(preds_dir.glob(f"{dataset}_h*_year_*.csv"))
    if not files:
        return None
    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except OSError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def infer_pred_column(
    df: pd.DataFrame,
    *,
    model_col: str | None,
) -> str | None:
    if model_col and model_col in df.columns:
        return model_col
    candidates = [c for c in df.columns if c not in _NON_VALUE_COLS]
    return candidates[0] if len(candidates) == 1 else None


def region_means(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    loc_col = KEY_LOC if KEY_LOC in df.columns else "adm_id"
    if loc_col not in df.columns or value_col not in df.columns:
        return {}
    sub = df[[loc_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return {}
    grouped = sub.groupby(loc_col, sort=True)[value_col].mean()
    return {str(k): round(float(v), 4) for k, v in grouped.items()}


def region_values_by_year(df: pd.DataFrame, value_col: str) -> dict[str, dict[str, float]]:
    """Per-calendar-year regional values; outer keys are year strings."""
    loc_col = KEY_LOC if KEY_LOC in df.columns else "adm_id"
    year_col = KEY_YEAR if KEY_YEAR in df.columns else "year"
    if (
        loc_col not in df.columns
        or year_col not in df.columns
        or value_col not in df.columns
    ):
        return {}
    sub = df[[year_col, loc_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return {}
    grouped = sub.groupby([year_col, loc_col], sort=True)[value_col].mean()
    out: dict[str, dict[str, float]] = {}
    for (year, loc), val in grouped.items():
        out.setdefault(str(int(year)), {})[str(loc)] = round(float(val), 4)
    return out


def benchmark_locs_by_country(map_payload: dict[str, Any]) -> dict[str, set[str]]:
    """Union of region ids with yield data, grouped by country code."""
    out: dict[str, set[str]] = {}
    for ds in map_payload.get("datasets", {}).values():
        country = ds.get("country")
        if not country:
            continue
        locs: set[str] = set(ds.get("actual", {}))
        for model_preds in ds.get("models", {}).values():
            locs.update(model_preds)
        if locs:
            out.setdefault(str(country), set()).update(locs)
    return out


def build_region_map_payload(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build map values from pooled year CSVs under output_dir/preds/."""
    output_dir = output_dir.resolve()
    datasets: dict[str, Any] = {}
    countries_needed: set[str] = set()

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    for dataset, rows in sorted(by_dataset.items()):
        country = dataset_country_code(dataset)
        if not country:
            continue

        actual: dict[str, float] = {}
        actual_by_year: dict[str, dict[str, float]] = {}
        models: dict[str, dict[str, float]] = {}
        models_by_year: dict[str, dict[str, dict[str, float]]] = {}
        years_seen: set[int] = set()

        for row in rows:
            preds_dir = preds_dir_for_row(output_dir, row)
            if preds_dir is None:
                continue
            df = load_dataset_year_csvs(preds_dir, dataset)
            if df is None or df.empty:
                continue

            target_col = KEY_TARGET if KEY_TARGET in df.columns else "yield"
            if not actual and target_col in df.columns:
                actual = region_means(df, target_col)
                actual_by_year = region_values_by_year(df, target_col)
                years_seen.update(int(y) for y in actual_by_year)

            pred_col = infer_pred_column(
                df, model_col=str(row.get("model_col") or "") or None
            )
            if pred_col is None:
                continue
            model_name = str(row["model"])
            model_preds = region_means(df, pred_col)
            if model_preds:
                models[model_name] = model_preds
            model_by_year = region_values_by_year(df, pred_col)
            if model_by_year:
                models_by_year[model_name] = model_by_year
                years_seen.update(int(y) for y in model_by_year)

        if not actual and not models:
            continue

        countries_needed.add(country)
        crop = dataset_crop(dataset)
        yield_range = CROP_YIELD_RANGES.get(crop)
        datasets[dataset] = {
            "country": country,
            "crop": crop,
            "yield_range": dict(yield_range) if yield_range else None,
            "years": sorted(years_seen),
            "actual": actual,
            "actual_by_year": actual_by_year,
            "models": models,
            "models_by_year": models_by_year,
            "n_regions": len(actual or next(iter(models.values()), {})),
        }

    return {
        "geojson_by_country": {cc: "" for cc in sorted(countries_needed)},
        "yield_ranges": CROP_YIELD_RANGES,
        "datasets": datasets,
        "note": (
            "Maps default to multi-year regional means; use the year slider for single years. "
            "Map extent covers benchmark regions only. "
            "Colors autoscale to pooled actual+pred across all years. "
            "Geometry is simplified admin boundaries from cybench/data/polygons."
        ),
    }


def bundle_region_map_assets(
    map_payload: dict[str, Any],
    output_dir: Path,
    *,
    assets_dirname: str = "assets",
) -> dict[str, Any]:
    """Export GeoJSON per country into assets/ and set relative hrefs."""
    if not map_payload.get("datasets"):
        return map_payload

    assets_dir = Path(output_dir) / assets_dirname
    assets_dir.mkdir(parents=True, exist_ok=True)
    geojson_by_country: dict[str, str] = {}
    locs_by_country = benchmark_locs_by_country(map_payload)

    for country in map_payload.get("geojson_by_country", {}):
        dest = assets_dir / f"regions_{country}.geojson"
        exported = export_region_geojson(
            country,
            dest,
            locations=locs_by_country.get(country),
        )
        if exported is not None:
            geojson_by_country[country] = f"{assets_dirname}/{dest.name}"

    return {
        **map_payload,
        "geojson_by_country": geojson_by_country,
    }

def write_region_map_sidecar(
    output_dir: Path,
    map_payload: dict[str, Any],
    *,
    filename: str = "region_map_data.json",
) -> Path | None:
    if not map_payload.get("datasets"):
        return None
    path = Path(output_dir) / filename
    path.write_text(json.dumps(map_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
