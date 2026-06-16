import os
import geopandas as gpd

from cybench.config import PATH_POLYGONS_DIR


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
