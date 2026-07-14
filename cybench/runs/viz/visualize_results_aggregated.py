#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from cybench.config import KEY_LOC, KEY_TARGET
from cybench.evaluation.aggregated_metrics import (
    MIN_SLICE_REGIONS,
    calc_r_r2,
    compute_report_metrics,
    get_metrics_dict,
)

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


def _load_dashboard_metrics_overlay(
    results_dir: str | None, dataset_key: str
) -> dict[str, Any] | None:
    """Region-year metrics from collect (walk_forward_summary), if dashboard_metrics.json exists."""
    if not results_dir:
        return None
    path = os.path.join(results_dir, "dashboard_metrics.json")
    if not os.path.isfile(path):
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    entry = raw.get(dataset_key)
    return entry if isinstance(entry, dict) else None


def _region_year_metrics_from_overlay(overlay: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("r", "r2", "nrmse"):
        val = overlay.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[key] = float(val)
    return out


def load_and_clean_data(
    csv_files: List[str],
    target_model_to_plot: str,
    min_years: int,
    dataset_key: str = "",
    *,
    trust_collect_export: bool = False,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads CSVs and applies strict filtering:
    1. Enforce numeric types for Target and ALL Models.
    2. Unless ``trust_collect_export``, skip YEARS where < 50% of regions have
       valid data (for all columns). Collect exports already QC-filtered rows;
       re-applying this step changes N and r relative to the dashboard table.
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
    if not trust_collect_export:
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
        "Spatial | "
        "Temporal | "
        "Anomaly |\n"
    )
    md += (
        "|  |  |  | "
        "r | R² | NRMSE | "
        "R² (med/yr) | R² (agg) | "
        "R² (med/reg) | R² (agg) | "
        "R² (med/reg) |\n"
    )
    md += (
        "| :--- | ---: | ---: | "
        "---: | ---: | ---: | "
        "---: | ---: | "
        "---: | ---: | "
        "---: |\n"
    )

    for s in stats_list:
        d_name = s["dataset"]
        n_reg = s["n_regions"]
        n_yrs = s["n_years"]
        mod = s["metrics_model"]
        sp = s["metrics_spatial"]
        tm = s["metrics_temporal"]
        an = s["metrics_anomaly"]

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
            f"| {style(fmt(sp['r2_typical_year']))} "
            f"| {style(fmt(sp.get('r2_aggregate', sp.get('r2_climatology'))))} "
            f"| {style(fmt(tm['r2_typical_region']))} "
            f"| {style(fmt(tm['r2_aggregate']))} "
            f"| {style(fmt(an['r2_typical_region']))} |"
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
      <th colspan="1">Anomaly</th>
    </tr>
    <tr>
      <th>r</th><th>R²</th><th>NRMSE</th>
      <th>R² (med/yr)</th><th>R² (agg)</th>
      <th>R² (med/reg)</th><th>R² (agg)</th>
      <th>R² (med/reg)</th>
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
        tm = s["metrics_temporal"]
        an = s["metrics_anomaly"]
        cls = ' class="small-n"' if n_reg < MIN_REGIONS_THRESHOLD else ""

        html += (
            f"    <tr{cls}>"
            f"<td class='left'>{d_name}</td>"
            f"<td>{n_reg}</td>"
            f"<td>{n_yrs}</td>"
            f"<td>{fmt(mod['r'])}</td>"
            f"<td>{fmt(mod['r2'])}</td>"
            f"<td>{fmt(mod['nrmse'])}</td>"
            f"<td>{fmt(sp['r2_typical_year'])}</td>"
            f"<td>{fmt(sp.get('r2_aggregate', sp.get('r2_climatology')))}</td>"
            f"<td>{fmt(tm['r2_typical_region'])}</td>"
            f"<td>{fmt(tm['r2_aggregate'])}</td>"
            f"<td>{fmt(an['r2_typical_region'])}</td>"
            "</tr>\n"
        )

    html += """  </tbody>
</table>
</body>
</html>
"""
    return html


ALL_PANELS = ("scatter", "temporal")
DASHBOARD_PANELS = ALL_PANELS
DEFAULT_PANEL_DPI = 160

# Switch to hexbin when overplotting would obscure structure.
SCATTER_HEX_THRESHOLD = 500
SCATTER_HEX_GRIDSIZE = 50
# Padding around per-panel PNG exports (inches). Union bbox avoids clipping labels.
PANEL_EXPORT_PAD_INCHES = 0.2


def parse_panels(raw: str) -> tuple[str, ...]:
    panels = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = set(panels) - set(ALL_PANELS)
    if unknown:
        raise ValueError(f"Unknown panel(s): {sorted(unknown)}; allowed: {ALL_PANELS}")
    if not panels:
        raise ValueError("At least one panel is required")
    return panels


def _panel_export_bbox(fig: Figure, axis: Axes, renderer) -> Bbox:
    """Bounding box for exporting one subplot as a standalone PNG.

    ``get_tightbbox`` alone can clip axis labels when ``set_box_aspect`` is used,
    and may bleed into neighbouring panels in a multi-panel figure. Union with the
    axes window extent and pad on all sides.
    """
    window = axis.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    tight = axis.get_tightbbox(renderer)
    if tight is not None:
        tight = tight.transformed(fig.dpi_scale_trans.inverted())
        bbox = Bbox.union([window, tight])
    else:
        bbox = window
    return bbox.padded(PANEL_EXPORT_PAD_INCHES)


def save_panel_images(
    fig: Figure,
    panel_axes: dict[str, Axes],
    output_dir: str,
    dataset_key: str,
) -> Dict[str, str]:
    """Save each subplot panel as a separate PNG and return absolute paths."""
    os.makedirs(output_dir, exist_ok=True)
    renderer = FigureCanvasAgg(fig).get_renderer()

    out_paths: Dict[str, str] = {}

    for panel_name, ax in panel_axes.items():
        axis = cast(Axes, ax)
        bbox = _panel_export_bbox(fig, axis, renderer)
        fn = f"{dataset_key}_{panel_name}.png"
        fp = os.path.join(output_dir, fn)
        fig.savefig(fp, dpi=DEFAULT_PANEL_DPI, bbox_inches=bbox, pad_inches=0)
        out_paths[panel_name] = fp

    return out_paths


def generate_local_report_html(stats_list: List[dict], pdf_filename: str) -> str:
    """Generate an interactive local HTML report."""
    rows = []
    for s in stats_list:
        m = s["metrics_model"]
        sp = s["metrics_spatial"]
        tm = s["metrics_temporal"]
        an = s["metrics_anomaly"]
        paths = s.get("panel_paths", {})
        rows.append(
            f"""
      <tr data-dataset="{s['dataset']}"
          data-scatter="{paths.get('scatter', '')}"
          data-temporal="{paths.get('temporal', '')}">
        <td>{s['dataset']}</td>
        <td>{s['n_regions']}</td>
        <td>{s['n_years']}</td>
        <td>{m['r']:.2f}</td>
        <td>{m['r2']:.2f}</td>
        <td>{m['nrmse']:.2f}</td>
        <td>{sp['r2_typical_year']:.2f}</td>
        <td>{sp.get('r2_aggregate', sp.get('r2_climatology', float('nan'))):.2f}</td>
        <td>{tm['r2_typical_region']:.2f}</td>
        <td>{tm['r2_aggregate']:.2f}</td>
        <td>{an['r2_typical_region']:.2f}</td>
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
        <th colspan="1">Anomaly</th>
      </tr>
      <tr>
        <th>r</th><th>R²</th><th>NRMSE</th>
        <th>R² (med/yr)</th><th>R² (agg)</th>
        <th>R² (med/reg)</th><th>R² (agg)</th>
        <th>R² (med/reg)</th>
      </tr>
    </thead>
    <tbody id="table-body">
{''.join(rows)}
    </tbody>
  </table>
  <div class="toolbar">
    <button data-panel="scatter" class="active">Scatter</button>
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
    let activePanel = "scatter";

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
def _yearly_cross_region_stats(
    df: pd.DataFrame,
    value_col: str,
    *,
    year_col: str = "year",
    min_regions: int = MIN_SLICE_REGIONS,
) -> pd.DataFrame:
    """Per-year mean and std of *value_col* across regions."""
    grouped = df.groupby(year_col)[value_col]
    out = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(ddof=1),
            "n_regions": grouped.count(),
        }
    ).sort_index()
    out.loc[out["n_regions"] < min_regions, "std"] = np.nan
    return out


def _plot_mean_std_band(
    ax: Axes,
    stats: pd.DataFrame,
    *,
    color: str,
    mean_linestyle: str,
    label: str,
    zorder: int = 2,
) -> None:
    years = stats.index.to_numpy()
    mean = stats["mean"].to_numpy(dtype=float)
    std = stats["std"].to_numpy(dtype=float)
    ax.plot(
        years,
        mean,
        mean_linestyle,
        color=color,
        linewidth=2.0,
        markersize=5,
        marker="o",
        label=label,
        zorder=zorder + 1,
    )
    lower = mean - std
    upper = mean + std
    has_band = np.isfinite(std)
    if np.any(has_band):
        ax.fill_between(
            years,
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=zorder,
        )


def _plot_scatter_panel(
    ax: Axes,
    df_filtered: pd.DataFrame,
    model: str,
    *,
    metrics_model: dict,
    metrics_base: dict | None,
    n_samples: int,
    metrics_note: str | None = None,
) -> None:
    y_true = _as_float_array(df_filtered[KEY_TARGET])
    y_pred = _as_float_array(df_filtered[model])

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = (hi - lo) * 0.02 or 1.0
    lim = (lo - pad, hi + pad)

    # Limits + square axes box must be set before hexbin: bin geometry follows the
    # data-to-display transform. set_aspect alone only equalizes data units inside
    # the subplot slot (which is often non-square in the wide multi-panel figure).
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)

    n = len(y_true)
    if n > SCATTER_HEX_THRESHOLD:
        ax.hexbin(
            y_true,
            y_pred,
            gridsize=SCATTER_HEX_GRIDSIZE,
            cmap="Blues",
            mincnt=1,
            linewidths=0,
            alpha=0.85,
            extent=(*lim, *lim),
        )
    else:
        alpha = min(0.75, max(0.35, 30 / np.sqrt(max(n, 1))))
        ax.scatter(
            y_true,
            y_pred,
            alpha=alpha,
            s=15,
            c="#2166ac",
            edgecolors="none",
            rasterized=True,
        )

    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.55, linewidth=1.0, zorder=5)

    ax.set_title("Scatter (all region-years)")
    ax.set_xlabel("Actual yield")
    ax.set_ylabel("Predicted yield")

    def fmt_region_year(m: dict) -> str:
        return f"r = {m['r']:.3f}, R² = {m['r2']:.3f}, NRMSE = {m['nrmse']:.3f}"

    txt_lines = [f"N = {n_samples:,}", f"Model: {fmt_region_year(metrics_model)}"]
    if metrics_note:
        txt_lines.append(metrics_note)
    if metrics_base:
        txt_lines.append(f"Baseline: {fmt_region_year(metrics_base)}")

    ax.text(
        0.05,
        0.95,
        "\n".join(txt_lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def process_dataset(
    dataset_key: str,
    df: pd.DataFrame,
    model: str,
    *,
    panels: tuple[str, ...] = ALL_PANELS,
    results_dir: str | None = None,
) -> Tuple[Figure, dict, dict[str, Axes]]:

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
    # Region-year metrics and scatter use the full cleaned frame (matches collect / table).
    # df_filtered: locations with sufficient core-year coverage (temporal spatial means).
    df_metrics = df

    overlay = _load_dashboard_metrics_overlay(results_dir, dataset_key)
    metrics_note: str | None = None
    if overlay and {"r", "r2", "nrmse"}.issubset(
        {k for k, v in overlay.items() if v is not None}
    ):
        metrics_model = _region_year_metrics_from_overlay(overlay)
        n_seeds = int(overlay["n_seeds"]) if overlay.get("n_seeds") else 1
        if n_seeds > 1:
            metrics_note = f"(matches table: mean over {n_seeds} seeds)"
        elif overlay.get("plot_seed") is not None:
            metrics_note = f"(matches table; plot seed {overlay['plot_seed']})"
        table_n = overlay.get("n_samples")
        if table_n is not None and int(table_n) != len(df_metrics):
            metrics_note = (
                (metrics_note + " ") if metrics_note else ""
            ) + f"[WARN] N={len(df_metrics)} here vs {int(table_n)} in table"
    else:
        metrics_model = get_metrics_dict(df_metrics, KEY_TARGET, model)

    # Stats calculation (Baseline)
    has_baseline = BASELINE_MODEL in df_metrics.columns
    if has_baseline:
        metrics_base = get_metrics_dict(df_metrics, KEY_TARGET, BASELINE_MODEL)
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

    report = compute_report_metrics(df_metrics, KEY_TARGET, model, year_col="year")
    r_time_model = report["temporal"]["r_aggregate"]
    r2_time_model = report["temporal"]["r2_aggregate"]

    # For baseline temporal correlation, we use the renamed column
    base_col_name = f"{BASELINE_MODEL}_Base"
    if has_baseline and base_col_name in ts.columns:
        r_time_base = calc_r_r2(
            _as_float_array(ts[KEY_TARGET]),
            _as_float_array(ts[base_col_name]),
        )[0]
    else:
        r_time_base = np.nan

    n_samples = len(df_metrics)
    n_regions = int(cast(pd.Series, df_metrics[KEY_LOC]).nunique())
    n_years = int(cast(pd.Series, df_metrics["year"]).nunique())

    # Save stats dict
    stats = {
        "dataset": dataset_key,
        "n_samples": n_samples,
        "n_regions": n_regions,
        "n_years": n_years,
        "metrics_model": metrics_model,
        "metrics_baseline": metrics_base,
        "metrics_spatial": report["spatial"],
        "metrics_temporal": report["temporal"],
        "metrics_anomaly": report["anomaly"],
        "r_time_model": r_time_model,
        "r2_time_model": r2_time_model,
        "r_time_base": r_time_base,
    }

    n_panels = len(panels)
    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 6.5), constrained_layout=True
    )
    if n_panels == 1:
        axes_list = [cast(Axes, axes_arr)]
    else:
        axes_list = [cast(Axes, ax) for ax in axes_arr]
    fig.suptitle(f"{dataset_key} (Model: {model})", fontsize=16)

    panel_axes: dict[str, Axes] = {}
    for panel_name, ax in zip(panels, axes_list):
        panel_axes[panel_name] = ax
        if panel_name == "scatter":
            _plot_scatter_panel(
                ax,
                df_metrics,
                model,
                metrics_model=metrics_model,
                metrics_base=metrics_base,
                n_samples=n_samples,
                metrics_note=metrics_note,
            )
        elif panel_name == "temporal":
            r_med_reg = report["temporal"]["r_typical_region"]
            actual_yr = _yearly_cross_region_stats(df_filtered, KEY_TARGET)
            model_yr = _yearly_cross_region_stats(df_filtered, model)
            _plot_mean_std_band(
                ax,
                actual_yr,
                color="#1f2328",
                mean_linestyle="-",
                label="Actual (mean ± std across regions)",
            )
            model_label = (
                f"Model (mean ± std, r med/reg = {r_med_reg:.2f})"
                if pd.notnull(r_med_reg)
                else "Model (mean ± std across regions)"
            )
            _plot_mean_std_band(
                ax,
                model_yr,
                color="#2166ac",
                mean_linestyle="--",
                label=model_label,
            )
            if has_baseline:
                base_yr = _yearly_cross_region_stats(df_filtered, BASELINE_MODEL)
                _plot_mean_std_band(
                    ax,
                    base_yr,
                    color="#1a7f37",
                    mean_linestyle=":",
                    label="Baseline (mean ± std)",
                )
            ax.set_title(
                f"Cross-region spread by year ($N_{{reg}}$={n_regions}, "
                f"$N_{{yrs}}$={n_years})"
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Year")
            ax.text(
                0.02,
                0.02,
                f"Band: ±1 std across regions (years with ≥{MIN_SLICE_REGIONS} regions)",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
                verticalalignment="bottom",
            )

    return fig, stats, panel_axes


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
        "--dashboard-assets",
        action="store_true",
        help="Scatter + temporal PNGs only (published dashboard; no PDF/maps/reports).",
    )
    parser.add_argument(
        "--panels",
        help=f"Comma-separated panels (default: all). Choices: {', '.join(ALL_PANELS)}",
    )
    parser.add_argument(
        "--min_years",
        type=int,
        default=3,
        help="Minimum distinct years required per dataset (default: 3).",
    )
    parser.add_argument(
        "--output_pdf",
        help="Optional combined PDF path (skipped with --dashboard-assets).",
    )

    args = parser.parse_args()

    if args.dashboard_assets:
        panels = DASHBOARD_PANELS
        write_reports = False
        pdf_path = None
    elif args.panels:
        panels = parse_panels(args.panels)
        write_reports = True
        pdf_path = args.output_pdf or os.path.join(args.results_dir, "evaluation_plots.pdf")
    else:
        panels = ALL_PANELS
        write_reports = True
        pdf_path = args.output_pdf or os.path.join(args.results_dir, "evaluation_plots.pdf")

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

    if pdf_path:
        print(f"[INFO] Processing {len(datasets_to_run)} dataset(s). PDF: {pdf_path}")
        os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    else:
        print(
            f"[INFO] Processing {len(datasets_to_run)} dataset(s). "
            f"Panels: {', '.join(panels)} (no PDF)"
        )

    all_stats_list = []
    panel_dir = os.path.join(args.results_dir, "report_assets")

    def _run_dataset(key: str, files: list[str], pdf: PdfPages | None) -> None:
        trust_collect = os.path.isfile(
            os.path.join(args.results_dir, "dashboard_metrics.json")
        )
        df, msg = load_and_clean_data(
            files,
            args.model,
            min_years=args.min_years,
            dataset_key=key,
            trust_collect_export=trust_collect,
        )
        if df is None:
            print(f"[SKIP] {msg}")
            return
        try:
            fig, stats, panel_axes = process_dataset(
                key, df, args.model, panels=panels, results_dir=args.results_dir
            )
            if pdf is not None:
                pdf.savefig(fig)
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

            # Export each panel from its own figure so crops do not bleed into neighbours.
            panel_paths_abs: Dict[str, str] = {}
            for panel_name in panels:
                pfig, _, paxes = process_dataset(
                    key,
                    df,
                    args.model,
                    panels=(panel_name,),
                    results_dir=args.results_dir,
                )
                try:
                    panel_paths_abs.update(
                        save_panel_images(pfig, paxes, panel_dir, key)
                    )
                finally:
                    plt.close(pfig)
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

    dataset_items = sorted(datasets_to_run.items())
    if pdf_path:
        with PdfPages(pdf_path) as pdf:
            for key, files in dataset_items:
                print(f"--> {key}...", end=" ", flush=True)
                _run_dataset(key, files, pdf)
    else:
        for key, files in dataset_items:
            print(f"--> {key}...", end=" ", flush=True)
            _run_dataset(key, files, None)

    if write_reports and all_stats_list:
        md_table = generate_markdown_table(all_stats_list)

        table_path = os.path.join(args.results_dir, "summary_table.md")
        with open(table_path, "w") as f:
            f.write(md_table)

        html_table = generate_html_table(all_stats_list)
        html_table_path = os.path.join(args.results_dir, "summary_table.html")
        with open(html_table_path, "w") as f:
            f.write(html_table)

        report_html = generate_local_report_html(
            all_stats_list, os.path.basename(pdf_path or "evaluation_plots.pdf")
        )
        report_html_path = os.path.join(args.results_dir, "report.html")
        with open(report_html_path, "w") as f:
            f.write(report_html)

        print(f"\n[DONE] Markdown table saved to: {table_path}")
        print(f"[DONE] HTML table saved to: {html_table_path}")
        print(f"[DONE] Local interactive report saved to: {report_html_path}")
        print(f"[DONE] Report panel assets saved to: {panel_dir}")
    elif all_stats_list:
        print(f"\n[DONE] Report panel assets saved to: {panel_dir}")
    else:
        print("\n[WARN] No stats collected.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
