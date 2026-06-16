#!/usr/bin/env python3
import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from cybench.util.geo import get_shapes_from_polygons, world_shape_path
from cybench.config import KEY_LOC, KEY_TARGET
from cybench.evaluation.aggregated_metrics import calc_r_r2, get_metrics_dict

# -----------------------------
# Configuration
# -----------------------------
CORE_FRACTION = 0.0
YEAR_COVERAGE_THRESHOLD = 0.0
BASELINE_MODEL = "AverageYieldModel"
NON_MODEL_COLS = {KEY_LOC, "year", KEY_TARGET, "country_code", "crop"}
MIN_REGIONS_THRESHOLD = 10  # Rows with fewer regions will be greyed out in the table


def _complete_rows_mask(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    return cast(pd.Series, cast(pd.DataFrame, df.loc[:, columns]).notna().all(axis=1))


def _series_int_at(series: pd.Series, key: object, default: int = 0) -> int:
    value = series.get(key, default)
    if value is None or pd.isna(value):
        return default
    return int(value)


def _index_as_list(index: pd.Index) -> List[int]:
    return [int(x) for x in index]


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
    return np.asarray(values, dtype=float)


# -----------------------------
# Data Discovery & Loading
# -----------------------------
def discover_inputs(results_dir: str) -> Dict[str, List[str]]:
    """
    Scans results_dir for CSVs and groups them by crop_region key.
    """
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    groups: Dict[str, List[str]] = {}
    pat = re.compile(r"^([A-Za-z]+)_([A-Z]{2})(?:_.*)?\.csv$")

    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".csv"):
            continue

        match = pat.match(fn)
        if match:
            crop, region = match.groups()
            key = f"{crop}_{region}"
            groups.setdefault(key, []).append(os.path.join(results_dir, fn))

    return groups


def load_and_clean_data(
    csv_files: List[str], target_model_to_plot: str, min_years: int, dataset_key: str = ""
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads CSVs and applies strict filtering:
    1. Enforce numeric types for Target and ALL Models.
    2. Skip YEARS where < 50% of the dataset's regions have valid data (for all columns).
    3. Drop remaining individual rows with missing data.
    """
    try:
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    except Exception as e:
        return None, f"read_error: {e}"

    # 1. Check basic requirements
    required_fixed = [KEY_LOC, "year", KEY_TARGET]
    if not all(c in df.columns for c in required_fixed):
        return (
            None,
            f"missing_fixed_cols: {[c for c in required_fixed if c not in df.columns]}",
        )

    if target_model_to_plot not in df.columns:
        return None, f"target_model_missing: {target_model_to_plot}"

    # 2. Identify ALL Model Columns (Dynamic Scan)
    all_cols = set(df.columns)
    candidate_models = [c for c in all_cols if c not in NON_MODEL_COLS]
    if target_model_to_plot not in candidate_models:
        candidate_models.append(target_model_to_plot)

    # Check Columns: Target + All Models
    cols_to_check = [KEY_TARGET] + candidate_models

    # 3. Coerce to Numeric
    for c in cols_to_check:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------------------------------------------------
    # 4. Filter Years based on Spatial Coverage
    # ---------------------------------------------------------

    # Calculate mask of valid rows (finite in ALL relevant cols)
    valid_mask = _complete_rows_mask(df, cols_to_check)

    # If a region is entirely NaN for all years, it shouldn't count towards the "total" we expect.
    valid_locations_universe = df.loc[valid_mask, KEY_LOC].unique()
    total_regions = len(valid_locations_universe)

    if total_regions > 0:
        # Count unique regions having valid data per year
        regions_per_year = cast(
            pd.Series, df[valid_mask].groupby("year")[KEY_LOC].nunique()
        )
        all_regions_per_year = df.groupby("year")[KEY_LOC].nunique()

        # Report how many regions are removed per year because of missing model/target values.
        removed_regions_per_year = (all_regions_per_year - regions_per_year).fillna(0).astype(int)
        removed_regions_per_year = removed_regions_per_year[removed_regions_per_year > 0]
        if len(removed_regions_per_year) > 0:
            key_prefix = f"{dataset_key}: " if dataset_key else ""
            print(
                f"[INFO] {key_prefix}regions removed by validity filter per year "
                f"(year:count) -> "
                + ", ".join(f"{int(y)}:{int(c)}" for y, c in removed_regions_per_year.items())
            )

        # Determine valid years
        min_required = int(np.ceil(YEAR_COVERAGE_THRESHOLD * total_regions))
        valid_years = _index_as_list(
            cast(pd.Series, regions_per_year[regions_per_year >= min_required]).index
        )
        all_years = sorted(df["year"].unique())
        key_prefix = f"{dataset_key}: " if dataset_key else ""

        per_year_coverage = []
        for year in all_years:
            available_regions = _series_int_at(regions_per_year, year)
            status = "KEEP" if available_regions >= min_required else "SKIP"
            per_year_coverage.append(
                f"{int(year)}={available_regions}/{total_regions} ({status})"
            )
        print(
            f"[INFO] {key_prefix}year coverage check "
            f"(threshold={min_required}/{total_regions} regions): "
            + ", ".join(per_year_coverage)
        )

        skipped_years = sorted(set(df["year"].unique()) - set(valid_years))
        if skipped_years:
            print(
                f"[INFO] {key_prefix}skipping year(s) for low spatial coverage "
                f"(threshold={min_required}/{total_regions} regions): {skipped_years}"
            )

        # Filter the DataFrame to keep only valid years
        df = df[df["year"].isin(valid_years)].copy()
    else:
        # If no valid regions exist at all, return empty early
        return None, "no_valid_regions_found"

    # ---------------------------------------------------------
    # 5. Strict Row Filtering (Intersection of Validity)
    # ---------------------------------------------------------
    # Now that we only have "good" years, we still drop any
    # specific rows that might be missing data (e.g. a single region in a good year).

    n_before = len(df)
    df = cast(
        pd.DataFrame,
        df.loc[_complete_rows_mask(cast(pd.DataFrame, df), cols_to_check)],
    ).copy()
    n_after = len(df)

    if n_after == 0:
        return None, f"empty_after_strict_filter (dropped {n_before} rows)"

    if df["year"].nunique() < min_years:
        return None, f"too_few_years: {df['year'].nunique()}"

    return df, "ok"


# -----------------------------
# Reporting
# -----------------------------
def generate_markdown_table(stats_list: List[dict]) -> str:
    """Generates a Markdown table across region-year, spatial, temporal, anomaly views."""

    md = (
        "| Dataset | N_regions | N_years | "
        "Region-Year | Region-Year | Region-Year | "
        "Spatial | Spatial | "
        "Temporal | Temporal | "
        "Anomaly | Anomaly |\n"
    )
    md += (
        "|  |  |  | "
        "r | R² | NRMSE | "
        "r | R² | "
        "r | R² | "
        "r | R² |\n"
    )
    md += (
        "| :--- | ---: | ---: | "
        "---: | ---: | ---: | "
        "---: | ---: | "
        "---: | ---: | "
        "---: | ---: |\n"
    )

    for s in stats_list:
        d_name = s["dataset"]
        n_reg = s["n_regions"]
        n_yrs = s["n_years"]
        mod = s["metrics_model"]
        sp = s["metrics_spatial"]

        # Check threshold
        is_faint = n_reg < MIN_REGIONS_THRESHOLD

        # Format helpers
        def fmt(val):
            return f"{val:.2f}" if pd.notnull(val) else "-"

        def style(text):
            # Apply grey color if below threshold
            if is_faint:
                return f'<span style="color:gray">{text}</span>'
            return text

        row = (
            f"| {style(d_name)} "
            f"| {style(str(n_reg))} "
            f"| {style(str(n_yrs))} "
            f"| {style(fmt(mod['r']))} "
            f"| {style(fmt(mod['r2']))} "
            f"| {style(fmt(mod['nrmse']))} "
            f"| {style(fmt(sp['r']))} "
            f"| {style(fmt(sp['r2']))} "
            f"| {style(fmt(s['r_time_model']))} "
            f"| {style(fmt(s['r2_time_model']))} "
            f"| {style(fmt(mod['r_res']))} "
            f"| {style(fmt(mod['r2_res']))} |"
        )
        md += row + "\n"

    return md


def generate_html_table(stats_list: List[dict]) -> str:
    """Generates an HTML table with grouped headers using colspan."""

    def fmt(val):
        return f"{val:.2f}" if pd.notnull(val) else "-"

    html = """
<html>
<head>
<meta charset="utf-8" />
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: right; }
th { background: #f5f5f5; font-weight: 600; }
th.left, td.left { text-align: left; }
.small-n { color: gray; }
</style>
</head>
<body>
<table>
  <thead>
    <tr>
      <th class="left" rowspan="2">Dataset</th>
      <th rowspan="2">N_regions</th>
      <th rowspan="2">N_years</th>
      <th colspan="3">Region-Year</th>
      <th colspan="2">Spatial</th>
      <th colspan="2">Temporal</th>
      <th colspan="2">Anomaly</th>
    </tr>
    <tr>
      <th>r</th><th>R²</th><th>NRMSE</th>
      <th>r</th><th>R²</th>
      <th>r</th><th>R²</th>
      <th>r</th><th>R²</th>
    </tr>
  </thead>
  <tbody>
"""

    for s in stats_list:
        d_name = s["dataset"]
        n_reg = s["n_regions"]
        n_yrs = s["n_years"]
        mod = s["metrics_model"]
        sp = s["metrics_spatial"]
        cls = ' class="small-n"' if n_reg < MIN_REGIONS_THRESHOLD else ""

        html += (
            f"    <tr{cls}>"
            f"<td class='left'>{d_name}</td>"
            f"<td>{n_reg}</td>"
            f"<td>{n_yrs}</td>"
            f"<td>{fmt(mod['r'])}</td>"
            f"<td>{fmt(mod['r2'])}</td>"
            f"<td>{fmt(mod['nrmse'])}</td>"
            f"<td>{fmt(sp['r'])}</td>"
            f"<td>{fmt(sp['r2'])}</td>"
            f"<td>{fmt(s['r_time_model'])}</td>"
            f"<td>{fmt(s['r2_time_model'])}</td>"
            f"<td>{fmt(mod['r_res'])}</td>"
            f"<td>{fmt(mod['r2_res'])}</td>"
            "</tr>\n"
        )

    html += """  </tbody>
</table>
</body>
</html>
"""
    return html


def save_panel_images(
    fig: Figure, axes: npt.NDArray[Any], output_dir: str, dataset_key: str
) -> Dict[str, str]:
    """Save each subplot panel as a separate PNG and return relative paths."""
    os.makedirs(output_dir, exist_ok=True)
    renderer = FigureCanvasAgg(fig).get_renderer()

    panel_names = ["map_actual", "map_pred", "scatter", "temporal"]
    out_paths: Dict[str, str] = {}

    for panel_name, ax in zip(panel_names, axes):
        axis = cast(Axes, ax)
        tight_bbox = axis.get_tightbbox(renderer)
        if tight_bbox is None:
            tight_bbox = axis.bbox
        bbox = tight_bbox.transformed(fig.dpi_scale_trans.inverted())
        fn = f"{dataset_key}_{panel_name}.png"
        fp = os.path.join(output_dir, fn)
        fig.savefig(fp, dpi=160, bbox_inches=bbox)
        out_paths[panel_name] = fp

    return out_paths


def generate_local_report_html(stats_list: List[dict], pdf_filename: str) -> str:
    """Generate an interactive local HTML report."""
    rows = []
    for s in stats_list:
        m = s["metrics_model"]
        sp = s["metrics_spatial"]
        paths = s.get("panel_paths", {})
        rows.append(
            f"""
      <tr data-dataset="{s['dataset']}"
          data-map-actual="{paths.get('map_actual', '')}"
          data-map-pred="{paths.get('map_pred', '')}"
          data-scatter="{paths.get('scatter', '')}"
          data-temporal="{paths.get('temporal', '')}">
        <td>{s['dataset']}</td>
        <td>{s['n_regions']}</td>
        <td>{s['n_years']}</td>
        <td>{m['r']:.2f}</td>
        <td>{m['r2']:.2f}</td>
        <td>{m['nrmse']:.2f}</td>
        <td>{sp['r']:.2f}</td>
        <td>{sp['r2']:.2f}</td>
        <td>{s['r_time_model']:.2f}</td>
        <td>{s['r2_time_model']:.2f}</td>
        <td>{m['r_res']:.2f}</td>
        <td>{m['r2_res']:.2f}</td>
      </tr>
"""
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CY-Bench Local Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f5f5f5; }}
    tr:hover {{ background: #fafafa; cursor: pointer; }}
    tr.active {{ background: #e8f2ff; }}
    .toolbar {{ display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }}
    .toolbar button {{ padding: 6px 10px; border: 1px solid #bbb; background: white; cursor: pointer; }}
    .toolbar button.active {{ background: #1f6feb; color: white; border-color: #1f6feb; }}
    .img-wrap {{ border: 1px solid #ddd; padding: 8px; min-height: 360px; }}
    .img-wrap img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    .meta {{ color: #555; margin-bottom: 10px; }}
  </style>
</head>
<body>
  <h2>CY-Bench evaluation report</h2>
  <div class="meta">PDF bundle: <a href="{pdf_filename}" target="_blank">{pdf_filename}</a></div>
  <table>
    <thead>
      <tr>
        <th rowspan="2">Dataset</th>
        <th rowspan="2">N_regions</th>
        <th rowspan="2">N_years</th>
        <th colspan="3">Region-Year</th>
        <th colspan="2">Spatial</th>
        <th colspan="2">Temporal</th>
        <th colspan="2">Anomaly</th>
      </tr>
      <tr>
        <th>r</th><th>R²</th><th>NRMSE</th>
        <th>r</th><th>R²</th>
        <th>r</th><th>R²</th>
        <th>r</th><th>R²</th>
      </tr>
    </thead>
    <tbody id="table-body">
{''.join(rows)}
    </tbody>
  </table>
  <div class="toolbar">
    <button data-panel="map_actual" class="active">Map actual</button>
    <button data-panel="map_pred">Map prediction</button>
    <button data-panel="scatter">Scatter</button>
    <button data-panel="temporal">Temporal</button>
  </div>
  <div class="meta" id="selection-label">Select a dataset row.</div>
  <div class="img-wrap">
    <img id="panel-image" alt="Selected panel" />
  </div>
  <script>
    const rows = Array.from(document.querySelectorAll("#table-body tr"));
    const buttons = Array.from(document.querySelectorAll(".toolbar button"));
    const label = document.getElementById("selection-label");
    const panelImage = document.getElementById("panel-image");
    let activeDataset = null;
    let activePanel = "map_actual";

    function refresh() {{
      if (!activeDataset) {{
        panelImage.removeAttribute("src");
        return;
      }}
      const imgPath = activeDataset.dataset[activePanel.replace(/_([a-z])/g, (_, c) => c.toUpperCase())];
      panelImage.src = imgPath;
      label.textContent = `${{activeDataset.dataset.dataset}} | panel: ${{activePanel}}`;
    }}

    rows.forEach((row) => {{
      row.addEventListener("click", () => {{
        rows.forEach(r => r.classList.remove("active"));
        row.classList.add("active");
        activeDataset = row;
        refresh();
      }});
    }});

    buttons.forEach((btn) => {{
      btn.addEventListener("click", () => {{
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activePanel = btn.dataset.panel;
        refresh();
      }});
    }});

    if (rows.length > 0) {{
      rows[0].click();
    }}
  </script>
</body>
</html>
"""


# -----------------------------
# Plotting
# -----------------------------
def process_dataset(
    dataset_key: str,
    df: pd.DataFrame,
    model: str,
    world: gpd.GeoDataFrame,
) -> Tuple[Figure, dict, npt.NDArray[Any]]:

    # 1. Core Year Filtering
    years = sorted(df["year"].unique())
    core_years = years[1:-1]
    n_core = len(core_years)

    if n_core == 0:
        raise ValueError("Not enough years to define core years (need > 2).")

    coverage = cast(
        pd.Series,
        df[df["year"].isin(core_years)].groupby(KEY_LOC)["year"].nunique(),
    )
    threshold = int(np.ceil(CORE_FRACTION * n_core))
    loc_coverage = cast(pd.Series, coverage[coverage >= threshold])
    valid_locs = loc_coverage.index.astype(str).tolist()

    df_filtered = cast(pd.DataFrame, df[df[KEY_LOC].isin(valid_locs)].copy())

    # 2. Geometry Prep
    try:
        crop, region_code = dataset_key.split("_")[:2]
    except ValueError:
        crop, region_code = "Unknown", "XX"

    shapes = get_shapes_from_polygons(region=region_code)
    geo_df = shapes[[KEY_LOC, "geometry"]].merge(
        df_filtered.groupby(KEY_LOC)[[KEY_TARGET, model]].mean().reset_index(),
        on=KEY_LOC,
        how="inner",
    )

    if geo_df.empty:
        raise ValueError(f"No geometry matches found for regions in {dataset_key}.")

    bounds = geo_df.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.05
    pad_y = (bounds[3] - bounds[1]) * 0.05
    bounds = (
        bounds[0] - pad_x,
        bounds[2] + pad_x,
        bounds[1] - pad_y,
        bounds[3] + pad_y,
    )

    # 3. Stats Calculation (Primary Model)
    metrics_model = get_metrics_dict(df_filtered, KEY_TARGET, model)

    # 3b. Stats Calculation (Baseline)
    has_baseline = BASELINE_MODEL in df_filtered.columns
    if has_baseline:
        metrics_base = get_metrics_dict(df_filtered, KEY_TARGET, BASELINE_MODEL)
    else:
        metrics_base = None

    # Temporal stats (Spatial Mean per Year)
    # Group for Target and Main Model
    ts = df_filtered.groupby("year")[[KEY_TARGET, model]].mean()

    # Join Baseline with a suffix to avoid collision if model == baseline
    if has_baseline:
        ts_base = df_filtered.groupby("year")[BASELINE_MODEL].mean()
        ts_base.name = f"{BASELINE_MODEL}_Base"
        ts = ts.join(ts_base)

    ts = ts.sort_index()

    r_time_model, r2_time_model = calc_r_r2(
        _as_float_array(ts[KEY_TARGET]),
        _as_float_array(ts[model]),
    )

    # Spatial stats (mean over years per location)
    spatial = cast(pd.DataFrame, df_filtered.groupby(KEY_LOC)[[KEY_TARGET, model]].mean())
    r_spatial_model, r2_spatial_model = calc_r_r2(
        _as_float_array(spatial[KEY_TARGET]),
        _as_float_array(spatial[model]),
    )

    # For baseline temporal correlation, we use the renamed column
    base_col_name = f"{BASELINE_MODEL}_Base"
    if has_baseline and base_col_name in ts.columns:
        r_time_base = calc_r_r2(
            _as_float_array(ts[KEY_TARGET]),
            _as_float_array(ts[base_col_name]),
        )[0]
    else:
        r_time_base = np.nan

    n_samples = len(df_filtered)
    n_regions = int(cast(pd.Series, df_filtered[KEY_LOC]).nunique())
    n_years = int(cast(pd.Series, df_filtered["year"]).nunique())

    # Save stats dict
    stats = {
        "dataset": dataset_key,
        "n_samples": n_samples,
        "n_regions": n_regions,
        "n_years": n_years,
        "metrics_model": metrics_model,
        "metrics_baseline": metrics_base,
        "r_time_model": r_time_model,
        "r2_time_model": r2_time_model,
        "r_time_base": r_time_base,
        "metrics_spatial": {"r": r_spatial_model, "r2": r2_spatial_model},
    }

    # 4. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5), constrained_layout=True)
    fig.suptitle(f"{dataset_key} (Model: {model})", fontsize=16)

    # Map 1: GT
    ax = axes[0]
    world.plot(ax=ax, color="lightgrey", edgecolor="k", linewidth=0.1)
    geo_df.plot(column=KEY_TARGET, ax=ax, legend=True, legend_kwds={"shrink": 0.5})
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(f"Ground Truth (Mean)\n($N_{{reg}}={n_regions}$)")
    ax.axis("off")

    # Map 2: Pred
    ax = axes[1]
    world.plot(ax=ax, color="lightgrey", edgecolor="k", linewidth=0.1)
    geo_df.plot(column=model, ax=ax, legend=True, legend_kwds={"shrink": 0.5})
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_title(f"Prediction (Mean)\n({model})")
    ax.axis("off")

    # Scatter
    ax = axes[2]
    y_true = _as_float_array(df_filtered[KEY_TARGET])
    y_pred = _as_float_array(df_filtered[model])

    if len(y_true) > 500:
        ax.hexbin(y_true, y_pred, gridsize=50, cmap="Blues", mincnt=1)
    else:
        ax.scatter(y_true, y_pred, alpha=0.6, s=15, label="Model")

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)

    ax.set_title("Scatter (All Region-Years)")
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield")

    # Stats Text Box
    txt_lines = [f"$N={n_samples}$"]

    def fmt_line(label, m):
        return (
            f"**{label}**: $r={m['r']:.2f}, R^2={m['r2']:.2f}$ | "
            f"$r_{{res}}={m['r_res']:.2f}, R^2_{{res}}={m['r2_res']:.2f}$"
        )

    txt_lines.append(fmt_line("Model", metrics_model))

    if metrics_base:
        txt_lines.append(fmt_line("Base", metrics_base))
    else:
        txt_lines.append("(Baseline not found)")

    stats_text = "\n".join(txt_lines)

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Time Series
    ax = axes[3]
    ax.plot(ts.index, ts[KEY_TARGET], "-o", linewidth=2, label="Actual")
    ax.plot(
        ts.index, ts[model], "--o", linewidth=2, label=f"Model ($r={r_time_model:.2f}$)"
    )

    if has_baseline:
        # We plot the explicitly renamed base column
        ax.plot(
            ts.index,
            ts[base_col_name],
            ":o",
            color="green",
            alpha=0.7,
            label=f"Base ($r={r_time_base:.2f}$)",
        )

    ax.set_title(f"Spatial Mean over Time ($N_{{yrs}}={n_years}$)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlabel("Year")

    return fig, stats, axes


# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate 1x4 evaluation plots for crop datasets."
    )

    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing the CSV result files.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="Optional list of dataset keys."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model column name to PLOT.",
    )
    parser.add_argument(
        "--save_individual", action="store_true", help="Save individual PNG/JSON files."
    )
    parser.add_argument(
        "--min_years",
        type=int,
        default=3,
        help="Minimum distinct years required per dataset (default: 3).",
    )
    parser.add_argument("--output_pdf", help="Custom path for combined PDF.")

    args = parser.parse_args()

    # 1. Discovery
    all_groups = discover_inputs(args.results_dir)

    if not all_groups:
        print(f"[INFO] No valid CSV files found in {args.results_dir}")
        return

    # 2. Filtering Logic
    if args.datasets:
        requested = set(args.datasets)
        datasets_to_run = {k: v for k, v in all_groups.items() if k in requested}
        if len(datasets_to_run) < len(requested):
            print(f"[WARN] Some requested datasets were not found.")
    else:
        datasets_to_run = all_groups

    if not datasets_to_run:
        print("[INFO] No datasets matched. Exiting.")
        return

    # 3. Setup Output
    if args.output_pdf:
        pdf_path = args.output_pdf
    else:
        pdf_path = os.path.join(args.results_dir, "evaluation_plots.pdf")

    print("[INFO] Loading world geometry...")
    world = gpd.read_file(world_shape_path())

    print(f"[INFO] Processing {len(datasets_to_run)} dataset(s). Output: {pdf_path}")
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)

    # Accumulator for final table
    all_stats_list = []
    panel_dir = os.path.join(args.results_dir, "report_assets")

    with PdfPages(pdf_path) as pdf:
        for key in sorted(datasets_to_run.keys()):
            files = datasets_to_run[key]
            print(f"--> {key}...", end=" ", flush=True)

            # Load
            df, msg = load_and_clean_data(
                files, args.model, min_years=args.min_years, dataset_key=key
            )
            if df is None:
                print(f"[SKIP] {msg}")
                continue

            # Process
            try:
                fig, stats, axes = process_dataset(key, df, args.model, world)
                pdf.savefig(fig)

                # Append stats for table generation
                all_stats_list.append(stats)

                if args.save_individual:
                    fig.savefig(
                        os.path.join(args.results_dir, f"{key}_plot.png"),
                        dpi=100,
                        bbox_inches="tight",
                    )
                    with open(
                        os.path.join(args.results_dir, f"{key}_stats.json"), "w"
                    ) as f:
                        json.dump(stats, f, indent=2)

                # Always save per-panel PNGs for local interactive report.
                panel_paths_abs = save_panel_images(fig, axes, panel_dir, key)
                stats["panel_paths"] = {
                    k: os.path.relpath(v, args.results_dir).replace(os.sep, "/")
                    for k, v in panel_paths_abs.items()
                }

                plt.close(fig)
                print("[OK]")

            except Exception as e:
                print(f"[FAIL] {e}")
                import traceback

                traceback.print_exc()

    # 4. Generate and Print Markdown Table
    if all_stats_list:
        md_table = generate_markdown_table(all_stats_list)

        table_path = os.path.join(args.results_dir, "summary_table.md")
        with open(table_path, "w") as f:
            f.write(md_table)

        html_table = generate_html_table(all_stats_list)
        html_table_path = os.path.join(args.results_dir, "summary_table.html")
        with open(html_table_path, "w") as f:
            f.write(html_table)

        report_html = generate_local_report_html(
            all_stats_list, os.path.basename(pdf_path)
        )
        report_html_path = os.path.join(args.results_dir, "report.html")
        with open(report_html_path, "w") as f:
            f.write(report_html)

        print(f"\n[DONE] Markdown table saved to: {table_path}")
        print(f"[DONE] HTML table saved to: {html_table_path}")
        print(f"[DONE] Local interactive report saved to: {report_html_path}")
        print(f"[DONE] Report panel assets saved to: {panel_dir}")
    else:
        print("\n[WARN] No stats collected. No table generated.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
