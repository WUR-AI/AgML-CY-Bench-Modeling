"""Region-year, spatial, temporal, and anomaly metrics (reporting views)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR


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
    r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
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
      - region_year: pooled region-year rows (r, R², NRMSE, anomaly r/R²)
      - spatial: per-location means across years
      - temporal: per-year means across locations
    """
    region_year = get_metrics_dict(df, target_col, model_col, loc_col=loc_col)

    spatial = df.groupby(loc_col)[[target_col, model_col]].mean()
    r_spatial, r2_spatial = calc_r_r2(
        spatial[target_col].values,
        spatial[model_col].values,
    )

    temporal = df.groupby(year_col)[[target_col, model_col]].mean().sort_index()
    r_time, r2_time = calc_r_r2(
        temporal[target_col].values,
        temporal[model_col].values,
    )

    return {
        "n_regions": int(df[loc_col].nunique()),
        "n_years": int(df[year_col].nunique()),
        "n_samples": int(len(df)),
        "region_year": region_year,
        "spatial": {"r": r_spatial, "r2": r2_spatial},
        "temporal": {"r": r_time, "r2": r2_time},
    }


def format_report_metrics(metrics: dict[str, Any]) -> str:
    """Single-line summary matching the aggregated report table."""
    ry = metrics["region_year"]
    sp = metrics["spatial"]
    tm = metrics["temporal"]
    return (
        f"region-year r={ry['r']:.2f} R²={ry['r2']:.2f} NRMSE={ry['nrmse']:.2f} | "
        f"spatial r={sp['r']:.2f} R²={sp['r2']:.2f} | "
        f"temporal r={tm['r']:.2f} R²={tm['r2']:.2f} | "
        f"anomaly r={ry['r_res']:.2f} R²={ry['r2_res']:.2f}"
    )
