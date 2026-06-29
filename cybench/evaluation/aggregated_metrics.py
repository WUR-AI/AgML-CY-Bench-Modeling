"""Region-year, spatial, temporal, and anomaly metrics (reporting views)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR

# Minimum points per slice for median view metrics (and temporal panel regional lines).
MIN_SLICE_YEARS = 3
MIN_SLICE_REGIONS = 3


def calc_r_r2(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return float("nan"), float("nan")

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else float("nan")
    else:
        r2 = float(1.0 - ss_res / ss_tot)
    return r, r2


def calc_nrmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """NRMSE normalized by mean absolute observed value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.mean(np.abs(y_true)))
    return float("nan") if denom == 0 else float(rmse / denom)


def calc_median_yearly_r2(
    df: pd.DataFrame,
    target_col: str,
    model_col: str,
    *,
    loc_col: str = KEY_LOC,
    year_col: str = KEY_YEAR,
    min_regions: int = MIN_SLICE_REGIONS,
) -> float:
    """Median of per-year R², where each year's R² is computed across regions."""
    yearly_r2: list[float] = []
    for _, year_df in df.groupby(year_col):
        if year_df[loc_col].nunique() < min_regions:
            continue
        _, r2 = calc_r_r2(
            year_df[target_col].values,
            year_df[model_col].values,
        )
        yearly_r2.append(r2)
    if not yearly_r2:
        return float("nan")
    return float(np.nanmedian(yearly_r2))


def calc_median_regional_r2(
    df: pd.DataFrame,
    target_col: str,
    model_col: str,
    *,
    loc_col: str = KEY_LOC,
    year_col: str = KEY_YEAR,
    min_years: int = MIN_SLICE_YEARS,
) -> float:
    """Median of per-region R², where each region's R² is computed across years."""
    del year_col  # API symmetry with yearly helper; years come from row groups.
    regional_r2: list[float] = []
    for _, loc_df in df.groupby(loc_col):
        if len(loc_df) < min_years:
            continue
        _, r2 = calc_r_r2(
            loc_df[target_col].values,
            loc_df[model_col].values,
        )
        regional_r2.append(r2)
    if not regional_r2:
        return float("nan")
    return float(np.nanmedian(regional_r2))


def calc_median_regional_r2_res(
    df: pd.DataFrame,
    target_col: str,
    model_col: str,
    *,
    loc_col: str = KEY_LOC,
    min_years: int = MIN_SLICE_YEARS,
) -> float:
    """Median of per-region R² on location-de-meaned yields (anomaly view)."""
    loc_means = df.groupby(loc_col)[target_col].mean()
    work = df.copy()
    work["_true_res"] = work[target_col] - work[loc_col].map(loc_means)
    work["_pred_res"] = work[model_col] - work[loc_col].map(loc_means)
    regional_r2: list[float] = []
    for _, loc_df in work.groupby(loc_col):
        if len(loc_df) < min_years:
            continue
        _, r2 = calc_r_r2(loc_df["_true_res"].values, loc_df["_pred_res"].values)
        regional_r2.append(r2)
    if not regional_r2:
        return float("nan")
    return float(np.nanmedian(regional_r2))


def get_metrics_dict(
    df: pd.DataFrame,
    target_col: str,
    model_col: str,
    *,
    loc_col: str = KEY_LOC,
) -> dict[str, float]:
    """Region-year and location-de-meaned (anomaly) metrics."""
    y_true = df[target_col].values
    y_pred = df[model_col].values
    r, r2 = calc_r_r2(y_true, y_pred)
    nrmse = calc_nrmse(y_true, y_pred)

    loc_means = df.groupby(loc_col)[target_col].mean()
    y_true_res = df[target_col] - df[loc_col].map(loc_means)
    y_pred_res = df[model_col] - df[loc_col].map(loc_means)
    r_res, r2_res = calc_r_r2(y_true_res, y_pred_res)

    return {
        "r": r,
        "r2": r2,
        "nrmse": nrmse,
        "r_res": r_res,
        "r2_res": r2_res,
    }


def compute_report_metrics(
    df: pd.DataFrame,
    target_col: str,
    model_col: str,
    *,
    loc_col: str = KEY_LOC,
    year_col: str = KEY_YEAR,
) -> dict[str, Any]:
    """
    Metrics used by visualize_results_aggregated / build_results_dashboard.

    Views:
      - region_year: pooled region-year rows (r, R², NRMSE, pooled anomaly r/R²)
      - spatial: typical-year cross-region R² (median over years); climatology map r/R²
      - temporal: typical-region cross-year R² (median over regions); aggregate series r/R²
      - anomaly: typical-region R² on residuals; pooled anomaly r/R²
    """
    complete = df[target_col].notna() & df[model_col].notna()
    df = df.loc[complete].copy()
    region_year = get_metrics_dict(df, target_col, model_col, loc_col=loc_col)

    spatial_clim = df.groupby(loc_col)[[target_col, model_col]].mean()
    r_spatial_clim, r2_spatial_clim = calc_r_r2(
        spatial_clim[target_col].values,
        spatial_clim[model_col].values,
    )

    temporal_agg = df.groupby(year_col)[[target_col, model_col]].mean().sort_index()
    r_temporal_agg, r2_temporal_agg = calc_r_r2(
        temporal_agg[target_col].values,
        temporal_agg[model_col].values,
    )

    return {
        "n_regions": int(df[loc_col].nunique()),
        "n_years": int(df[year_col].nunique()),
        "n_samples": int(len(df)),
        "region_year": region_year,
        "spatial": {
            "r2_typical_year": calc_median_yearly_r2(
                df, target_col, model_col, loc_col=loc_col, year_col=year_col
            ),
            "r_climatology": r_spatial_clim,
            "r2_climatology": r2_spatial_clim,
        },
        "temporal": {
            "r2_typical_region": calc_median_regional_r2(
                df, target_col, model_col, loc_col=loc_col, year_col=year_col
            ),
            "r_aggregate": r_temporal_agg,
            "r2_aggregate": r2_temporal_agg,
        },
        "anomaly": {
            "r2_typical_region": calc_median_regional_r2_res(
                df, target_col, model_col, loc_col=loc_col
            ),
            "r_pooled": region_year["r_res"],
            "r2_pooled": region_year["r2_res"],
        },
    }


def format_report_metrics(metrics: dict[str, Any]) -> str:
    """Single-line summary matching the aggregated report table."""
    ry = metrics["region_year"]
    sp = metrics["spatial"]
    tm = metrics["temporal"]
    an = metrics["anomaly"]
    return (
        f"region-year r={ry['r']:.2f} R²={ry['r2']:.2f} NRMSE={ry['nrmse']:.2f} | "
        f"spatial R²(med/yr)={sp['r2_typical_year']:.2f} "
        f"clim R²={sp['r2_climatology']:.2f} | "
        f"temporal R²(med/reg)={tm['r2_typical_region']:.2f} "
        f"agg R²={tm['r2_aggregate']:.2f} | "
        f"anomaly R²(med/reg)={an['r2_typical_region']:.2f} "
        f"pooled R²={an['r2_pooled']:.2f}"
    )
