import os

import geopandas as gpd

from cybench.config import PATH_POLYGONS_DIR

# CY-Bench country codes that differ from Natural Earth ISO_A2 (e.g. EL = Greece).
_CYBENCH_TO_MAP_ISO: dict[str, str] = {
    "EL": "GR",
}
_OVERSEAS_MAP_ISO = "XX"
_EXCLUDED_MAP_ISOS = frozenset({"AQ"})
# Clip metropolitan extent for countries whose admin-0 polygon includes overseas territories.
_METROPOLITAN_BBOX_WGS84: dict[str, tuple[float, float, float, float]] = {
    "FR": (-5.5, 41.0, 10.0, 51.5),
}


def map_iso_for_country(country_code: str) -> str:
    key = country_code.upper()
    return _CYBENCH_TO_MAP_ISO.get(key, key)


def _normalize_world_iso(world: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    iso_col = next((c for c in ("ISO_A2", "iso_a2") if c in world.columns), None)
    iso_eh_col = next((c for c in ("ISO_A2_EH", "iso_a2_eh") if c in world.columns), None)
    wb_col = next((c for c in ("WB_A2", "wb_a2") if c in world.columns), None)
    if iso_col is None:
        raise ValueError(f"No ISO_A2 column in world shapefile; columns={list(world.columns)}")

    keep = world[[iso_col, "geometry"]].copy().rename(columns={iso_col: "ISO_A2"})
    iso = keep["ISO_A2"].astype(str)
    bad = iso.isna() | (iso == "-99") | (iso == "nan")
    if iso_eh_col is not None:
        iso = iso.where(~bad, world[iso_eh_col].astype(str))
        bad = iso.isna() | (iso == "-99") | (iso == "nan")
    if wb_col is not None:
        iso = iso.where(~bad, world[wb_col].astype(str))
    keep["ISO_A2"] = iso
    return keep[
        keep["ISO_A2"].notna()
        & (keep["ISO_A2"] != "-99")
        & (keep["ISO_A2"] != "nan")
        & ~keep["ISO_A2"].isin(_EXCLUDED_MAP_ISOS)
    ].copy()


def _explode_metropolitan_map_units(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode multipolygons; re-tag overseas parts so they are not included in borders."""
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


def get_country_border_gdf(country_code: str, *, scale: str = "110") -> gpd.GeoDataFrame:
    """Natural Earth admin-0 outline for one CY-Bench country (metropolitan where applicable)."""
    map_iso = map_iso_for_country(country_code)
    world = gpd.read_file(world_shape_path(scale)).to_crs(4326)
    keep = _explode_metropolitan_map_units(_normalize_world_iso(world))
    country = keep[keep["ISO_A2"] == map_iso].copy()
    if country.empty:
        raise ValueError(f"No Natural Earth polygon for country code {country_code!r} (ISO {map_iso})")
    return country[["ISO_A2", "geometry"]]


def get_shapes_from_polygons(region):
    """
    Load administrative boundaries from the polygons folder.

    Folder structure assumed:
        POLYGONS_DIR/COUNTRY/COUNTRY.shp

    :param region: 2-letter country code
    :return: GeoDataFrame with an 'adm_id' column
    """
    region_dir = os.path.join(PATH_POLYGONS_DIR, region)
    shp_path = os.path.join(region_dir, f"{region}.shp")

    if not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Shapefile for region '{region}' not found at {shp_path}"
        )

    gdf = gpd.read_file(shp_path)

    # Ensure a column 'adm_id' exists
    if "adm_id" not in gdf.columns:
        # fallback: use first column that looks like an ID
        id_cols = [c for c in gdf.columns if "id" in c.lower() or "ID" in c]
        if id_cols:
            gdf["adm_id"] = gdf[id_cols[0]]
        else:
            # fallback: create a numeric index as ID
            gdf["adm_id"] = range(len(gdf))

    # Project to EPSG 4326
    if region == "BR":
        gdf = gdf.set_crs(epsg=4326)

    gdf = gdf.to_crs(4326)

    return gdf


def world_shape_path(scale: str | None = None) -> str:
    """Natural Earth admin-0 outline for map backgrounds (grey context behind regions).

    Default preference: 50m, then 10m, then 110m. Override with *scale* or
  CYBENCH_WORLD_MAP_SCALE (50 | 10 | 110).
    """
    import os

    from cybench.config import REPO_DIR

    preferred = scale or os.environ.get("CYBENCH_WORLD_MAP_SCALE", "50")
    order: list[str] = []
    if preferred in {"10", "50", "110"}:
        order.append(preferred)
    for s in ("50", "10", "110"):
        if s not in order:
            order.append(s)
    root = os.path.join(REPO_DIR, "data_preparation")
    for s in order:
        path = os.path.join(root, f"ne_{s}m_admin_0_countries", f"ne_{s}m_admin_0_countries.shp")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No Natural Earth country shapefile found under data_preparation/. "
        "Install ne_50m_admin_0_countries (recommended) or ne_110m_admin_0_countries. "
        "See cybench/runs/slurm/README.md § Polygons for maps."
    )
