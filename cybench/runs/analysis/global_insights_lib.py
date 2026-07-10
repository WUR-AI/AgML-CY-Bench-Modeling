"""Aggregate walk-forward summaries for cross-country and horizon comparisons."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from cybench.runs.analysis.index_map_lib import map_iso_for_cybencH

_PAPER_DIR_RE = re.compile(
    r"^paper_walk_forward_(?P<country>[a-z]{2})_(?P<horizon>eos|mid|qtr|early)_v(?P<version>\d+)$"
)

_BASELINE_MODELS = frozenset({"average", "averageyieldmodel", "average_yield"})

# Lead time decreases left → right (less season observed → more observed).
# Code: early = ~25% observed; mid = 50%; qtr = ~75% observed (25% left); eos = ~100%.
BATCH_HORIZON_ORDER: tuple[str, ...] = ("early", "mid", "qtr", "eos")

HORIZON_DISPLAY_LABELS: dict[str, str] = {
    "early": "Early season (~25% observed)",
    "mid": "Mid-season (~50% observed)",
    "qtr": "Late season (~75% observed, 25% left)",
    "eos": "End of season (~100% observed)",
}

# Process baselines evaluated only at end-of-season (no mid/qtr walk-forward runs).
EOS_ONLY_HORIZON_CURVE_MODELS: frozenset[str] = frozenset({"lpjml_bc"})

# Fixed choropleth scales (aligned with insights heatmap cell coloring).
METRIC_MAP_SCALES: dict[str, dict[str, Any]] = {
    "nrmse": {"lo": 0.0, "hi": 0.5, "higher_better": False, "label": "NRMSE"},
    "r2": {"lo": 0.0, "hi": 1.0, "higher_better": True, "label": "R²"},
    "r": {"lo": -1.0, "hi": 1.0, "higher_better": True, "label": "r"},
}

MAP_COVERAGE_NOTE = (
    "CY-Bench countries use a blue-grey base; other land is neutral grey. "
    "The United States and France are metropolitan only (Alaska, French Guiana, "
    "and other overseas territories are not colored)."
)


def horizons_in_data(df: pd.DataFrame) -> tuple[str, ...]:
    """Horizons present in a summary frame, ordered by increasing season progress."""
    if df.empty or "batch_horizon" not in df.columns:
        return ()
    present = set(df["batch_horizon"].astype(str))
    return tuple(hz for hz in BATCH_HORIZON_ORDER if hz in present)

# Evaluation views for the model×country heatmap (aligned with country dashboards / radar).
MODEL_COUNTRY_AXES: tuple[dict[str, Any], ...] = (
    {
        "id": "overall",
        "label": "Overall",
        "note": "Region×year pooled metrics (all samples).",
        "metrics": (
            {"id": "r2", "column": "r2", "label": "R²", "higher_better": True},
            {"id": "nrmse", "column": "nrmse", "label": "NRMSE", "higher_better": False},
        ),
    },
    {
        "id": "spatial",
        "label": "Spatial",
        "note": "Median per-year Pearson r across regions (typical-year slice).",
        "metrics": (
            {"id": "r", "column": "r_spatial", "label": "r", "higher_better": True},
        ),
    },
    {
        "id": "temporal",
        "label": "Temporal",
        "note": "Median per-region Pearson r across years (typical-region slice).",
        "metrics": (
            {"id": "r", "column": "r_temporal", "label": "r", "higher_better": True},
        ),
    },
    {
        "id": "anomaly",
        "label": "Anomaly",
        "note": "Pooled Pearson r on location-demeaned yields (all region×year residuals). "
        "Distinct from temporal median per-region r; measures whether anomalous years line up globally.",
        "metrics": (
            {"id": "r", "column": "r_res", "label": "r", "higher_better": True},
        ),
    },
)

_NUMERIC_SUMMARY_COLS = (
    "nrmse",
    "r2",
    "n_samples",
    "n_train",
    "n_regions",
    "n_years",
    "r2_spatial",
    "r_spatial",
    "r_spatial_agg",
    "r2_spatial_agg",
    "r2_temporal",
    "r_temporal",
    "r_temporal_agg",
    "r2_temporal_agg",
    "r2_anomaly",
    "r_anomaly",
    "r2_res",
    "r_res",
)


def is_baseline_model(model: object) -> bool:
    return str(model).lower().replace("-", "_") in _BASELINE_MODELS


def parse_paper_dir_name(name: str) -> tuple[str, str, int] | None:
    match = _PAPER_DIR_RE.match(name)
    if not match:
        return None
    return match.group("country").upper(), match.group("horizon"), int(match.group("version"))


def dashboard_href_for_paper_dir(paper_dir_name: str) -> str | None:
    """Relative GitHub Pages path to a country dashboard (e.g. ``de_walk_forward_eos_v1/dashboard.html``)."""
    parsed = parse_paper_dir_name(paper_dir_name)
    if parsed is None:
        return None
    country, hz, ver = parsed
    slug = f"{country.lower()}_walk_forward_{hz}_v{ver}"
    return f"{slug}/dashboard.html"


def build_dashboard_hrefs(output_root: Path, *, version: int = 2) -> dict[str, dict[str, str]]:
    """Map CY-Bench country code -> horizon (``eos``/``mid``) -> dashboard HTML href."""
    hrefs: dict[str, dict[str, str]] = {}
    for path in discover_summary_tables(output_root, version=version):
        parsed = parse_paper_dir_name(path.parent.name)
        if parsed is None:
            continue
        country, hz, _ver = parsed
        rel = dashboard_href_for_paper_dir(path.parent.name)
        if rel:
            hrefs.setdefault(country, {})[hz] = rel
    return hrefs


def discover_summary_tables(output_root: Path, *, version: int = 2) -> list[Path]:
    """Return walk_forward_summary.csv paths under paper_walk_forward_* dirs."""
    if not output_root.is_dir():
        return []
    paths: list[Path] = []
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_paper_dir_name(entry.name)
        if parsed is None or parsed[2] != version:
            continue
        summary = entry / "walk_forward_summary.csv"
        if summary.is_file():
            paths.append(summary)
    return paths


def compat_legacy_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map pre-v2 walk_forward_summary columns to current metric names.

    Older collects stored aggregate spatial/temporal metrics in ``r2_spatial``,
    ``r2_temporal``, ``r_spatial``, and ``r_temporal``. Current schema keeps
    slice medians in ``r_spatial`` / ``r_temporal`` and aggregates in
    ``*_agg`` columns — legacy aggregate r columns are renamed, not reused as slices.
    """
    if df.empty:
        return df
    out = df.copy()
    legacy_r2_agg = "r2_spatial_agg" not in out.columns and "r2_spatial" in out.columns
    if legacy_r2_agg:
        out["r2_spatial_agg"] = out["r2_spatial"]
        out["r2_temporal_agg"] = out.get("r2_temporal")
    if "r2_res" not in out.columns and "r2_anomaly" in out.columns:
        out["r2_res"] = out["r2_anomaly"]
    if legacy_r2_agg:
        if "r_spatial_agg" not in out.columns and "r_spatial" in out.columns:
            out["r_spatial_agg"] = out["r_spatial"]
            out = out.drop(columns=["r_spatial"])
        if "r_temporal_agg" not in out.columns and "r_temporal" in out.columns:
            out["r_temporal_agg"] = out["r_temporal"]
            out = out.drop(columns=["r_temporal"])
    return out


def _series_for_matrix_column(grp: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric series for a matrix column."""
    if column not in grp.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(grp[column], errors="coerce")


def _median_in_group(grp: pd.DataFrame, column: str) -> float | None:
    vals = _series_for_matrix_column(grp, column).dropna()
    if vals.empty:
        return None
    return float(vals.median())


def _country_values_for_model(model_grp: pd.DataFrame, metric: str) -> list[float]:
    """Per-country values: median within country (across crops when crop=all)."""
    country_vals: list[float] = []
    if "country" in model_grp.columns:
        for _, country_grp in model_grp.groupby("country", sort=True):
            val = _median_in_group(country_grp, metric)
            if val is not None:
                country_vals.append(val)
    else:
        val = _median_in_group(model_grp, metric)
        if val is not None:
            country_vals.append(val)
    return country_vals


def median_model_metrics_across_countries(
    work: pd.DataFrame,
    metrics: tuple[str, ...] | list[str],
    *,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Per-model medians using the same rule as the insights matrix Median column.

    For each country, take the median within that country (across crops when crop=all).
    The model summary is the median across those country values — each country weighs equally.
    """
    if work.empty or "model" not in work.columns:
        return pd.DataFrame()
    frame = work
    if models:
        frame = frame[frame["model"].isin(models)]
    if frame.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in frame.columns]
    if not present:
        return pd.DataFrame()

    rows: dict[str, dict[str, float]] = {}
    for model, model_grp in frame.groupby("model", sort=True):
        row: dict[str, float] = {}
        for metric in present:
            country_vals = _country_values_for_model(model_grp, metric)
            row[metric] = (
                float(pd.Series(country_vals).median()) if country_vals else float("nan")
            )
        rows[str(model)] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def quantile_model_metrics_across_countries(
    work: pd.DataFrame,
    metrics: tuple[str, ...] | list[str],
    *,
    models: list[str] | None = None,
    low_q: float = 0.25,
    high_q: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model Q25/Q75 across per-country values (same country rule as median)."""
    empty = pd.DataFrame()
    if work.empty or "model" not in work.columns:
        return empty, empty
    frame = work
    if models:
        frame = frame[frame["model"].isin(models)]
    if frame.empty:
        return empty, empty
    present = [m for m in metrics if m in frame.columns]
    if not present:
        return empty, empty

    q25_rows: dict[str, dict[str, float]] = {}
    q75_rows: dict[str, dict[str, float]] = {}
    for model, model_grp in frame.groupby("model", sort=True):
        row25: dict[str, float] = {}
        row75: dict[str, float] = {}
        for metric in present:
            country_vals = _country_values_for_model(model_grp, metric)
            if country_vals:
                series = pd.Series(country_vals, dtype=float)
                row25[metric] = float(series.quantile(low_q))
                row75[metric] = float(series.quantile(high_q))
            else:
                row25[metric] = float("nan")
                row75[metric] = float("nan")
        q25_rows[str(model)] = row25
        q75_rows[str(model)] = row75
    return (
        pd.DataFrame.from_dict(q25_rows, orient="index"),
        pd.DataFrame.from_dict(q75_rows, orient="index"),
    )


def _axis_metrics_for_group(grp: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    axes: dict[str, dict[str, float | None]] = {}
    for axis in MODEL_COUNTRY_AXES:
        metrics: dict[str, float | None] = {}
        for spec in axis["metrics"]:
            metrics[str(spec["id"])] = _median_in_group(grp, str(spec["column"]))
        axes[str(axis["id"])] = metrics
    return axes


def matrix_axes_payload() -> list[dict[str, Any]]:
    """JSON-serializable axis definitions for the insights heatmap UI."""
    out: list[dict[str, Any]] = []
    for axis in MODEL_COUNTRY_AXES:
        out.append(
            {
                "id": axis["id"],
                "label": axis["label"],
                "note": axis["note"],
                "metrics": [
                    {
                        "id": m["id"],
                        "label": m["label"],
                        "higher_better": m["higher_better"],
                    }
                    for m in axis["metrics"]
                ],
            }
        )
    return out


def load_summary_frame(summary_paths: list[Path]) -> pd.DataFrame:
    """Load and tag rows from multiple country/horizon summary CSVs."""
    frames: list[pd.DataFrame] = []
    for path in summary_paths:
        parsed = parse_paper_dir_name(path.parent.name)
        if parsed is None:
            continue
        country, batch_hz, version = parsed
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["country"] = country
        df["batch_horizon"] = batch_hz
        df["version"] = version
        df["paper_dir"] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in _NUMERIC_SUMMARY_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return compat_legacy_summary_columns(out)


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    s = series[mask]
    w = weights[mask]
    return float((s * w).sum() / w.sum())


def attach_baseline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add baseline NRMSE and skill flags per crop×country×horizon."""
    if df.empty:
        return df.copy()

    out = df.copy()
    baseline = out[out["model"].apply(is_baseline_model)]
    key_cols = ["crop", "country", "batch_horizon"]
    if baseline.empty:
        out["baseline_nrmse"] = float("nan")
        out["beats_baseline"] = False
        out["skilled"] = out["r2"].fillna(float("-inf")) > 0 if "r2" in out.columns else False
        return out

    bl = (
        baseline.groupby(key_cols, as_index=False)["nrmse"]
        .min()
        .rename(columns={"nrmse": "baseline_nrmse"})
    )
    out = out.merge(bl, on=key_cols, how="left")
    out["beats_baseline"] = out["nrmse"] < out["baseline_nrmse"]
    if "r2" in out.columns:
        out["skilled"] = out["beats_baseline"] | (out["r2"].fillna(float("-inf")) > 0)
    else:
        out["skilled"] = out["beats_baseline"]
    out.loc[out["model"].apply(is_baseline_model), "skilled"] = False
    return out


def _beat_baseline_rate(model: str, grp: pd.DataFrame) -> float | None:
    if is_baseline_model(model):
        return None
    comparable = grp[grp["baseline_nrmse"].notna()]
    if comparable.empty:
        return float("nan")
    return float(comparable["beats_baseline"].mean())


def _filter_summary_work(
    df: pd.DataFrame,
    *,
    batch_horizon: str | None = None,
    crop: str | None = None,
    skilled_only: bool = False,
    require_valid_nrmse: bool = True,
) -> pd.DataFrame:
    work = attach_baseline_metrics(df)
    if batch_horizon is not None:
        work = work[work["batch_horizon"] == batch_horizon]
    if crop:
        work = work[work["crop"] == crop]
    if skilled_only:
        work = work[work["skilled"]]
    if require_valid_nrmse and "nrmse" in work.columns:
        work = work[work["nrmse"].notna()]
    return work.copy()


def _model_median_by_country(work: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-model median of per-country axis metrics.

    Each country contributes one value per axis metric: the median within that country
    (relevant when crop=all spans multiple crops in the same country). The model summary
    is then the median across those country values.
    """
    totals: dict[str, dict[str, Any]] = {}
    for model, model_grp in work.groupby("model", sort=True):
        by_country: list[dict[str, dict[str, float | None]]] = []
        for _, country_grp in model_grp.groupby("country", sort=True):
            by_country.append(_axis_metrics_for_group(country_grp))

        axes: dict[str, dict[str, float | None]] = {}
        for axis in MODEL_COUNTRY_AXES:
            axis_id = str(axis["id"])
            axes[axis_id] = {}
            for spec in axis["metrics"]:
                metric_id = str(spec["id"])
                country_vals = [
                    c[axis_id][metric_id]
                    for c in by_country
                    if c[axis_id].get(metric_id) is not None
                ]
                axes[axis_id][metric_id] = (
                    round(float(pd.Series(country_vals).median()), 4) if country_vals else None
                )
        totals[str(model)] = axes
    return totals


def aggregate_model_leaderboard(
    df: pd.DataFrame,
    *,
    batch_horizon: str | None = None,
    crop: str | None = None,
    skilled_only: bool = False,
) -> pd.DataFrame:
    """Rank models by median NRMSE across countries (one value per country)."""
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()

    work = _filter_summary_work(
        df, batch_horizon=batch_horizon, crop=crop, skilled_only=skilled_only
    )
    if work.empty:
        return pd.DataFrame()

    by_country = _model_median_by_country(work)
    rows: list[dict[str, Any]] = []
    for model, grp in work.groupby("model", sort=True):
        beat_rate = _beat_baseline_rate(str(model), grp)
        totals = by_country[str(model)]
        overall = totals.get("overall", {})

        rows.append(
            {
                "model": model,
                "median_nrmse": overall.get("nrmse"),
                "median_r2": overall.get("r2"),
                "beat_baseline_rate": beat_rate,
                "n_datasets": int(len(grp)),
                "n_countries": int(grp["country"].nunique()) if "country" in grp else 0,
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("median_nrmse", ascending=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def _crop_keys(df: pd.DataFrame) -> list[str]:
    if df.empty or "crop" not in df.columns:
        return []
    return sorted({str(c) for c in df["crop"].dropna().unique()})


def build_leaderboards_by_crop(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
    skilled_only: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Leaderboards keyed by crop name; ``all`` aggregates every crop."""
    boards: dict[str, list[dict[str, Any]]] = {
        "all": _df_records(
            aggregate_model_leaderboard(
                df, batch_horizon=batch_horizon, skilled_only=skilled_only
            )
        )
    }
    for crop in _crop_keys(df):
        boards[crop] = _df_records(
            aggregate_model_leaderboard(
                df, batch_horizon=batch_horizon, crop=crop, skilled_only=skilled_only
            )
        )
    return boards


def build_model_country_matrix(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
    crop: str | None = None,
    skilled_only: bool = False,
) -> dict[str, Any]:
    """Model × country matrix (median metrics per evaluation axis)."""
    work = _filter_summary_work(
        df, batch_horizon=batch_horizon, crop=crop, skilled_only=skilled_only
    )
    if work.empty:
        return {"models": [], "countries": [], "cells": [], "model_totals": {}}

    cells: list[dict[str, Any]] = []
    for (model, country), grp in work.groupby(["model", "country"], sort=True):
        beat_rate = _beat_baseline_rate(str(model), grp)
        axes = _axis_metrics_for_group(grp)
        overall = axes.get("overall", {})
        cells.append(
            {
                "model": model,
                "country": country,
                "axes": axes,
                # Legacy flat keys for overall (leaderboard parity).
                "median_nrmse": overall.get("nrmse"),
                "median_r2": overall.get("r2"),
                "n_datasets": int(len(grp)),
                "beat_baseline_rate": beat_rate,
            }
        )

    models = sorted({c["model"] for c in cells})
    countries = sorted({c["country"] for c in cells})
    return {
        "models": models,
        "countries": countries,
        "cells": cells,
        "model_totals": _model_median_by_country(work),
    }


def build_model_country_by_crop(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
    skilled_only: bool = False,
) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {
        "all": build_model_country_matrix(
            df, batch_horizon=batch_horizon, skilled_only=skilled_only
        ),
    }
    for crop in _crop_keys(df):
        matrices[crop] = build_model_country_matrix(
            df, batch_horizon=batch_horizon, crop=crop, skilled_only=skilled_only
        )
    return matrices


def compare_horizons(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare eos vs mid-season NRMSE per (crop, country, model).

    Returns (per_pair_detail, per_model_summary).
    Delta = mid_nrmse - eos_nrmse (positive ⇒ end-of-season is better).
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    eos = df[df["batch_horizon"] == "eos"].copy()
    mid = df[df["batch_horizon"] == "mid"].copy()
    if eos.empty or mid.empty:
        return pd.DataFrame(), pd.DataFrame()

    key_cols = ["crop", "country", "model"]
    for col in key_cols:
        if col not in eos.columns or col not in mid.columns:
            return pd.DataFrame(), pd.DataFrame()

    eos_keyed = eos[key_cols + ["nrmse", "r2", "n_samples"]].rename(
        columns={"nrmse": "eos_nrmse", "r2": "eos_r2", "n_samples": "eos_samples"}
    )
    mid_keyed = mid[key_cols + ["nrmse", "r2", "n_samples"]].rename(
        columns={"nrmse": "mid_nrmse", "r2": "mid_r2", "n_samples": "mid_samples"}
    )
    merged = eos_keyed.merge(mid_keyed, on=key_cols, how="inner")
    merged = merged[merged["eos_nrmse"].notna() & merged["mid_nrmse"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["delta_nrmse"] = merged["mid_nrmse"] - merged["eos_nrmse"]
    merged["delta_r2"] = merged["mid_r2"] - merged["eos_r2"]
    merged["eos_better"] = merged["delta_nrmse"] > 0
    merged["dataset"] = merged["crop"] + "_" + merged["country"]

    pair_weights = merged[["eos_samples", "mid_samples"]].min(axis=1).fillna(1).clip(lower=1)
    merged["pair_weight"] = pair_weights

    model_rows: list[dict[str, Any]] = []
    for model, grp in merged.groupby("model", sort=True):
        weights = grp["pair_weight"]
        model_rows.append(
            {
                "model": model,
                "n_pairs": int(len(grp)),
                "eos_win_rate": float(grp["eos_better"].mean()),
                "mean_delta_nrmse": float(grp["delta_nrmse"].mean()),
                "weighted_delta_nrmse": _weighted_mean(grp["delta_nrmse"], weights),
                "mean_eos_nrmse": float(grp["eos_nrmse"].mean()),
                "mean_mid_nrmse": float(grp["mid_nrmse"].mean()),
            }
        )
    summary = pd.DataFrame(model_rows).sort_values(
        ["weighted_delta_nrmse", "mean_delta_nrmse"], ascending=[False, False]
    )
    detail = merged.sort_values(["model", "country", "crop"]).reset_index(drop=True)
    return detail, summary


def compare_crops_pairwise(
    df: pd.DataFrame,
    *,
    crop_a: str,
    crop_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare NRMSE for two crops in countries that have both (inner join on country × model).

    Returns (per_pair_detail, per_model_summary).
    Delta = crop_b_nrmse − crop_a_nrmse (positive ⇒ crop_a has lower NRMSE).
    """
    if df.empty or crop_a == crop_b:
        return pd.DataFrame(), pd.DataFrame()

    key_cols = ["country", "model"]
    for col in key_cols + ["crop", "nrmse"]:
        if col not in df.columns:
            return pd.DataFrame(), pd.DataFrame()

    a_df = df[df["crop"] == crop_a][key_cols + ["nrmse", "r2", "n_samples"]].rename(
        columns={"nrmse": "crop_a_nrmse", "r2": "crop_a_r2", "n_samples": "crop_a_samples"}
    )
    b_df = df[df["crop"] == crop_b][key_cols + ["nrmse", "r2", "n_samples"]].rename(
        columns={"nrmse": "crop_b_nrmse", "r2": "crop_b_r2", "n_samples": "crop_b_samples"}
    )
    if a_df.empty or b_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = a_df.merge(b_df, on=key_cols, how="inner")
    merged = merged[merged["crop_a_nrmse"].notna() & merged["crop_b_nrmse"].notna()].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["crop_a"] = crop_a
    merged["crop_b"] = crop_b
    merged["delta_nrmse"] = merged["crop_b_nrmse"] - merged["crop_a_nrmse"]
    merged["delta_r2"] = merged["crop_b_r2"] - merged["crop_a_r2"]
    merged["crop_a_better"] = merged["delta_nrmse"] > 0
    merged["dataset_a"] = crop_a + "_" + merged["country"]
    merged["dataset_b"] = crop_b + "_" + merged["country"]

    pair_weights = merged[["crop_a_samples", "crop_b_samples"]].min(axis=1).fillna(1).clip(lower=1)
    merged["pair_weight"] = pair_weights

    model_rows: list[dict[str, Any]] = []
    for model, grp in merged.groupby("model", sort=True):
        weights = grp["pair_weight"]
        model_rows.append(
            {
                "model": model,
                "n_pairs": int(len(grp)),
                "n_countries": int(grp["country"].nunique()),
                "crop_a_win_rate": float(grp["crop_a_better"].mean()),
                "mean_delta_nrmse": float(grp["delta_nrmse"].mean()),
                "weighted_delta_nrmse": _weighted_mean(grp["delta_nrmse"], weights),
                "mean_crop_a_nrmse": float(grp["crop_a_nrmse"].mean()),
                "mean_crop_b_nrmse": float(grp["crop_b_nrmse"].mean()),
            }
        )
    summary = pd.DataFrame(model_rows).sort_values(
        ["weighted_delta_nrmse", "mean_delta_nrmse"], ascending=[False, False]
    )
    detail = merged.sort_values(["model", "country"]).reset_index(drop=True)
    return detail, summary


HORIZON_SKILL_VALUE_COLUMNS: tuple[str, ...] = (
    "nrmse",
    "r2",
    "r_spatial",
    "r_temporal",
    "r_res",
)


def _horizon_skill_axes() -> list[dict[str, Any]]:
    """Four evaluation views for horizon curves (aligned with model×country heatmap)."""
    axes: list[dict[str, Any]] = []
    for axis in MODEL_COUNTRY_AXES:
        if axis["id"] == "overall":
            metrics = [
                {
                    "id": "nrmse",
                    "column": "nrmse",
                    "label": "NRMSE",
                    "higher_better": False,
                    "has_iqr": True,
                },
                {
                    "id": "r2",
                    "column": "r2",
                    "label": "R²",
                    "higher_better": True,
                    "has_iqr": True,
                },
                {
                    "id": "skill_vs_trend",
                    "column": "skill_vs_trend",
                    "label": "Skill vs trend",
                    "higher_better": True,
                    "has_iqr": False,
                },
            ]
        else:
            metrics = [
                {
                    "id": m["id"],
                    "column": m["column"],
                    "label": m["label"],
                    "higher_better": m["higher_better"],
                    "has_iqr": True,
                }
                for m in axis["metrics"]
            ]
        axes.append(
            {
                "id": axis["id"],
                "label": axis["label"],
                "note": axis["note"],
                "metrics": metrics,
            }
        )
    return axes


def _median_iqr_stats(values: list[float]) -> dict[str, float | None]:
    series = pd.Series(values, dtype=float)
    if series.empty:
        return {"median": None, "q25": None, "q75": None}
    return {
        "median": round(float(series.median()), 4),
        "q25": round(float(series.quantile(0.25)), 4),
        "q75": round(float(series.quantile(0.75)), 4),
    }


def _wide_country_model_horizon_metrics(
    work: pd.DataFrame,
    horizons: tuple[str, ...],
    *,
    crop: str | None,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Inner-join median metrics across horizons on country×model (fair country set)."""
    frame = work[work["nrmse"].notna()].copy()
    if crop:
        frame = frame[frame["crop"] == crop]
    cols = [c for c in value_columns if c in frame.columns]
    if not cols:
        return pd.DataFrame()
    group_keys = ["country", "model"]

    wide: pd.DataFrame | None = None
    for hz in horizons:
        hz_df = frame[frame["batch_horizon"] == hz]
        if hz_df.empty:
            return pd.DataFrame()
        agg = hz_df.groupby(group_keys, as_index=False)[cols].median()
        rename = {c: f"{c}_{hz}" for c in cols}
        part = agg.rename(columns=rename)
        wide = part if wide is None else wide.merge(part, on=group_keys, how="inner")
    if wide is None:
        return pd.DataFrame()
    if crop:
        wide.insert(0, "crop", crop)
    return wide


def _point_metrics_from_wide_rows(
    model_rows: pd.DataFrame,
    *,
    trend_nrmse_by_country: dict[tuple[str, str], float],
    horizons: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for hz in horizons:
        metrics: dict[str, Any] = {}
        n_countries = 0
        for col_base in value_columns:
            col = f"{col_base}_{hz}"
            vals: list[float] = []
            for _, row in model_rows.iterrows():
                val = row.get(col)
                if pd.notna(val):
                    vals.append(float(val))
            if col_base == "nrmse":
                n_countries = len(vals)
            metrics[col_base] = _median_iqr_stats(vals)

        skill_vals: list[float] = []
        nrmse_col = f"nrmse_{hz}"
        for _, row in model_rows.iterrows():
            country = str(row["country"])
            val = row.get(nrmse_col)
            if pd.isna(val):
                continue
            nrmse = float(val)
            trend_val = trend_nrmse_by_country.get((country, hz))
            if trend_val and trend_val > 0:
                skill_vals.append(1.0 - nrmse / trend_val)
        skill_stats = _median_iqr_stats(skill_vals)
        metrics["skill_vs_trend"] = {"median": skill_stats["median"]}

        points.append({"horizon": hz, "metrics": metrics, "n_countries": n_countries})
    return points


def _family_curve_points(
    wide: pd.DataFrame,
    *,
    model: str,
    trend_model: str,
    horizons: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    model_rows = wide[wide["model"] == model]
    trend_rows = wide[wide["model"] == trend_model]
    if model_rows.empty:
        return []

    trend_nrmse_by_country: dict[tuple[str, str], float] = {}
    for _, row in trend_rows.iterrows():
        country = str(row["country"])
        for hz in horizons:
            col = f"nrmse_{hz}"
            val = row.get(col)
            if pd.notna(val):
                trend_nrmse_by_country[(country, hz)] = float(val)

    return _point_metrics_from_wide_rows(
        model_rows,
        trend_nrmse_by_country=trend_nrmse_by_country,
        horizons=horizons,
        value_columns=value_columns,
    )


def _eos_only_family_points(
    work: pd.DataFrame,
    *,
    model: str,
    trend_model: str,
    value_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Median EOS metrics for models not run at earlier forecast horizons."""
    eos = work[
        (work["model"] == model) & (work["batch_horizon"] == "eos") & work["nrmse"].notna()
    ]
    if eos.empty:
        return []

    trend_eos = work[
        (work["model"] == trend_model) & (work["batch_horizon"] == "eos") & work["nrmse"].notna()
    ]
    trend_by_country: dict[str, float] = {}
    if not trend_eos.empty and "country" in trend_eos.columns:
        for country, grp in trend_eos.groupby("country"):
            med = grp["nrmse"].median()
            if pd.notna(med) and float(med) > 0:
                trend_by_country[str(country)] = float(med)

    metrics: dict[str, Any] = {}
    n_countries = 0
    cols = [c for c in value_columns if c in eos.columns]
    if "country" in eos.columns:
        for col_base in cols:
            vals: list[float] = []
            for country, grp in eos.groupby("country"):
                med = grp[col_base].median()
                if pd.notna(med):
                    vals.append(float(med))
            if col_base == "nrmse":
                n_countries = len(vals)
            metrics[col_base] = _median_iqr_stats(vals)
    else:
        for col_base in cols:
            med = eos[col_base].median()
            if pd.notna(med):
                metrics[col_base] = _median_iqr_stats([float(med)])
                if col_base == "nrmse":
                    n_countries = 1

    skill_vals: list[float] = []
    if "country" in eos.columns and "nrmse" in eos.columns:
        for country, grp in eos.groupby("country"):
            med = grp["nrmse"].median()
            if pd.isna(med):
                continue
            nrmse = float(med)
            trend_val = trend_by_country.get(str(country))
            if trend_val and trend_val > 0:
                skill_vals.append(1.0 - nrmse / trend_val)
    skill_stats = _median_iqr_stats(skill_vals)
    metrics["skill_vs_trend"] = {"median": skill_stats["median"]}

    return [{"horizon": "eos", "metrics": metrics, "n_countries": n_countries}]


def _build_model_horizon_entry(
    *,
    model: str,
    wide: pd.DataFrame,
    work: pd.DataFrame,
    trend_model: str,
    horizons: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> dict[str, Any] | None:
    eos_only = model in EOS_ONLY_HORIZON_CURVE_MODELS
    if eos_only:
        points = _eos_only_family_points(
            work,
            model=model,
            trend_model=trend_model,
            value_columns=value_columns,
        )
    else:
        points = _family_curve_points(
            wide,
            model=model,
            trend_model=trend_model,
            horizons=horizons,
            value_columns=value_columns,
        )
    if not points:
        return None
    n_horizons_with_data = sum(
        1 for p in points if (p.get("metrics") or {}).get("nrmse", {}).get("median") is not None
    )
    return {
        "model": model,
        "points": points,
        "eos_only": eos_only,
        "plot": (not eos_only) and n_horizons_with_data >= 2,
    }


def _build_all_model_horizon_entries(
    work: pd.DataFrame,
    wide: pd.DataFrame,
    *,
    trend_model: str,
    horizons: tuple[str, ...],
    value_columns: tuple[str, ...],
    model_display_names: dict[str, str],
) -> list[dict[str, Any]]:
    if work.empty or "model" not in work.columns:
        return []
    entries: list[dict[str, Any]] = []
    for model in sorted(work["model"].astype(str).unique()):
        entry = _build_model_horizon_entry(
            model=model,
            wide=wide,
            work=work,
            trend_model=trend_model,
            horizons=horizons,
            value_columns=value_columns,
        )
        if entry is None:
            continue
        entry["display"] = model_display_names.get(model, model)
        entries.append(entry)
    return entries


def build_horizon_skill_curves_payload(
    df: pd.DataFrame,
    *,
    representatives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Horizon vs performance curves for fixed family representatives (fair country set)."""
    from cybench.runs.analysis.model_family_radar_lib import (
        FAMILY_COLORS,
        FAMILY_ORDER,
        MODEL_DISPLAY_NAMES,
        pick_representatives,
    )

    horizons = horizons_in_data(df)
    if len(horizons) < 2:
        return {
            "horizons": [],
            "by_crop": {},
            "representatives": {},
            "note": "Need at least two forecast horizons with collected summaries.",
        }

    rep_source_hz = "eos" if "eos" in horizons else horizons[-1]
    rep_work = _filter_summary_work(df, batch_horizon=rep_source_hz)
    reps = pick_representatives(rep_work, overrides=representatives)
    trend_model = reps.get("Naive baselines", "trend")
    rep_models = set(reps.values()) | {trend_model}

    crop_keys = ["all", *_crop_keys(df)]
    value_columns = tuple(c for c in HORIZON_SKILL_VALUE_COLUMNS if c in df.columns)
    by_crop: dict[str, Any] = {}
    for crop_key in crop_keys:
        crop_filter = None if crop_key == "all" else crop_key
        crop_work = _filter_summary_work(df, crop=crop_filter)
        wide_all = _wide_country_model_horizon_metrics(
            crop_work, horizons, crop=crop_filter, value_columns=value_columns
        )
        countries = (
            sorted(wide_all["country"].astype(str).unique()) if not wide_all.empty else []
        )
        any_countries = (
            set(crop_work["country"].astype(str).unique()) if "country" in crop_work.columns else set()
        )
        excluded = sorted(any_countries - set(countries))

        models = _build_all_model_horizon_entries(
            crop_work,
            wide_all,
            trend_model=trend_model,
            horizons=horizons,
            value_columns=value_columns,
            model_display_names=MODEL_DISPLAY_NAMES,
        )

        work = crop_work[crop_work["model"].isin(rep_models)]
        wide = (
            wide_all[wide_all["model"].isin(rep_models)].copy()
            if not wide_all.empty
            else wide_all
        )

        families: list[dict[str, Any]] = []
        for family in FAMILY_ORDER:
            model = reps.get(family)
            if not model or model == trend_model:
                continue
            entry = _build_model_horizon_entry(
                model=model,
                wide=wide,
                work=work,
                trend_model=trend_model,
                horizons=horizons,
                value_columns=value_columns,
            )
            if entry is None:
                continue
            families.append(
                {
                    "family": family,
                    "model": model,
                    "display": MODEL_DISPLAY_NAMES.get(model, model),
                    "color": FAMILY_COLORS.get(family, "#666"),
                    **entry,
                }
            )

        by_crop[crop_key] = {
            "n_countries": len(countries),
            "countries": countries,
            "excluded_countries": excluded,
            "families": families,
            "models": models,
        }

    rep_labels = ", ".join(f"{f}: {m}" for f, m in reps.items())
    return {
        "horizons": [
            {"id": hz, "label": HORIZON_DISPLAY_LABELS.get(hz, hz), "order": i}
            for i, hz in enumerate(horizons)
        ],
        "representatives": reps,
        "representatives_source_horizon": rep_source_hz,
        "representatives_summary": rep_labels,
        "by_crop": by_crop,
        "note": (
            "Fixed model per family (representatives chosen at "
            f"{HORIZON_DISPLAY_LABELS.get(rep_source_hz, rep_source_hz)}). "
            "Curves use only countries with data at every plotted horizon "
            "(inner join on country×model). Each country contributes one median per metric "
            "(median across crops when crop=all). IQR band = spread across countries. "
            "Switch axis to compare overall (NRMSE/R²), spatial, temporal, or anomaly views; "
            "temporal r often shifts most with forecast lead time. "
            "LPJmL is end-of-season only and appears in the table but not on the curve plot. "
            "Hyperparameters are tuned per horizon (screening at each lead time)."
        ),
        "plot_excluded_note": (
            "EOS-only baselines (LPJmL) are omitted from the curve plot; see the tables for "
            "their end-of-season median."
        ),
        "models_table_note": (
            "Median per metric across countries. Multi-horizon models use only countries with "
            "data at every collected horizon (inner join). EOS-only baselines show end-of-season "
            "values only."
        ),
        "axes": _horizon_skill_axes(),
        "metric_notes": {
            "nrmse": "Median pooled NRMSE across countries (lower is better).",
            "r2": "Median pooled R² across countries (higher is better).",
            "skill_vs_trend": (
                "Median skill = 1 − NRMSE_model / NRMSE_trend per country (higher is better)."
            ),
            "r_spatial": "Median spatial r across countries (typical-year slice; higher is better).",
            "r_temporal": (
                "Median temporal r across countries (typical-region slice; often horizon-sensitive)."
            ),
            "r_res": "Median anomaly r (pooled demeaned residuals; higher is better).",
        },
    }


def report_model_horizon_pairs(
    df: pd.DataFrame,
    model: str,
    *,
    crop: str | None = None,
    min_sample_ratio: float | None = None,
) -> dict[str, Any]:
    """Paired mid/qtr/eos report for one model (fair country intersection).

    Inner-joins on (country, model) across all horizons present in ``df``.
    Optional ``min_sample_ratio`` drops pairs where min(n_samples)/max(n_samples)
    across horizons falls below the threshold (e.g. 0.95).
    """
    horizons = horizons_in_data(df)
    if len(horizons) < 2:
        return {"model": model, "error": "Need at least two horizons in summaries."}

    work = _filter_summary_work(df, crop=crop)
    work = work[work["model"] == model]
    if work.empty:
        return {"model": model, "crop": crop, "error": f"No rows for model {model!r}."}

    group_keys = ["country", "model"]
    wide: pd.DataFrame | None = None
    samples_wide: pd.DataFrame | None = None
    for hz in horizons:
        hz_df = work[work["batch_horizon"] == hz]
        if hz_df.empty:
            return {"model": model, "crop": crop, "error": f"No {hz} rows for {model}."}
        nrmse_part = hz_df.groupby(group_keys, as_index=False)["nrmse"].median().rename(
            columns={"nrmse": f"nrmse_{hz}"}
        )
        wide = nrmse_part if wide is None else wide.merge(nrmse_part, on=group_keys, how="inner")
        if "n_samples" in hz_df.columns:
            samp_part = hz_df.groupby(group_keys, as_index=False)["n_samples"].sum().rename(
                columns={"n_samples": f"n_samples_{hz}"}
            )
            samples_wide = (
                samp_part if samples_wide is None else samples_wide.merge(samp_part, on=group_keys, how="inner")
            )

    if wide is None or wide.empty:
        return {"model": model, "crop": crop, "error": "No country intersection across horizons."}

    if samples_wide is not None:
        for _, row in samples_wide.iterrows():
            vals = [row.get(f"n_samples_{hz}") for hz in horizons]
            vals = [float(v) for v in vals if pd.notna(v) and float(v) > 0]
            if len(vals) >= 2:
                ratio = min(vals) / max(vals)
                wide.loc[wide["country"] == row["country"], "_sample_ratio"] = ratio

    if min_sample_ratio is not None and "_sample_ratio" in wide.columns:
        wide = wide[wide["_sample_ratio"].fillna(0) >= min_sample_ratio].copy()

    crop_work = _filter_summary_work(df, crop=crop)
    any_countries = (
        set(crop_work[crop_work["model"] == model]["country"].astype(str).unique())
        if "country" in crop_work.columns
        else set()
    )
    paired_countries = sorted(wide["country"].astype(str).unique())
    excluded = sorted(any_countries - set(paired_countries))

    detail_rows: list[dict[str, Any]] = []
    for _, row in wide.sort_values("country").iterrows():
        country = str(row["country"])
        nrmse_by_hz = {hz: float(row[f"nrmse_{hz}"]) for hz in horizons}
        best_hz = min(nrmse_by_hz, key=nrmse_by_hz.get)  # type: ignore[arg-type]
        entry: dict[str, Any] = {
            "country": country,
            **{f"nrmse_{hz}": round(nrmse_by_hz[hz], 4) for hz in horizons},
            "best_horizon": best_hz,
        }
        if samples_wide is not None:
            sw = samples_wide[samples_wide["country"] == country]
            if not sw.empty:
                for hz in horizons:
                    col = f"n_samples_{hz}"
                    if col in sw.columns:
                        val = sw.iloc[0][col]
                        entry[col] = int(val) if pd.notna(val) else None
                if "_sample_ratio" in row.index and pd.notna(row["_sample_ratio"]):
                    entry["sample_ratio"] = round(float(row["_sample_ratio"]), 4)
        detail_rows.append(entry)

    summary: dict[str, Any] = {"n_paired_countries": len(paired_countries)}
    for hz in horizons:
        col = f"nrmse_{hz}"
        vals = wide[col].astype(float)
        summary[f"median_{col}"] = round(float(vals.median()), 4)
        summary[f"mean_{col}"] = round(float(vals.mean()), 4)

    if "eos" in horizons:
        for hz in horizons:
            if hz == "eos":
                continue
            col = f"nrmse_{hz}"
            delta = wide[col] - wide["nrmse_eos"]
            wins = delta > 0  # lower NRMSE is better; positive delta => eos better
            summary[f"eos_better_than_{hz}_rate"] = round(float(wins.mean()), 4)
            summary[f"mean_delta_{hz}_minus_eos"] = round(float(delta.mean()), 4)
            summary[f"median_delta_{hz}_minus_eos"] = round(float(delta.median()), 4)

    return {
        "model": model,
        "crop": crop or "all",
        "horizons": list(horizons),
        "horizon_labels": {hz: HORIZON_DISPLAY_LABELS.get(hz, hz) for hz in horizons},
        "n_paired_countries": len(paired_countries),
        "paired_countries": paired_countries,
        "excluded_countries": excluded,
        "summary": summary,
        "detail": detail_rows,
        "interpretation": (
            f"Paired comparison for {model}: only countries with walk-forward summaries "
            f"at all horizons ({', '.join(horizons)}). "
            "delta = horizon_nrmse − eos_nrmse; positive ⇒ end-of-season has lower NRMSE. "
            "best_horizon = lowest NRMSE within country."
        ),
    }


def build_crop_comparison_payload(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per horizon, pairwise crop NRMSE on shared countries only."""
    if df.empty:
        return {}
    crops = _crop_keys(df)
    out: dict[str, dict[str, Any]] = {}
    for hz in horizons_in_data(df):
        hz_df = df[df["batch_horizon"] == hz]
        pairs: dict[str, Any] = {}
        for i, crop_a in enumerate(crops):
            for crop_b in crops[i + 1 :]:
                detail, summary = compare_crops_pairwise(hz_df, crop_a=crop_a, crop_b=crop_b)
                pair_key = f"{crop_a}_vs_{crop_b}"
                pairs[pair_key] = {
                    "crop_a": crop_a,
                    "crop_b": crop_b,
                    "detail": _df_records(detail),
                    "summary": _df_records(summary),
                    "overall": _overall_crop_pair_stats(detail, crop_a, crop_b),
                }
        out[hz] = pairs
    return out


def build_insights_payload(output_root: Path, *, version: int = 2) -> dict[str, Any]:
    """Build JSON-serializable payload for the global insights dashboard."""
    paths = discover_summary_tables(output_root, version=version)
    df = load_summary_frame(paths)

    available_horizons = horizons_in_data(df)
    leaderboards: dict[str, dict[str, list[dict[str, Any]]]] = {}
    leaderboards_skilled: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_country: dict[str, dict[str, dict[str, Any]]] = {}
    model_country_skilled: dict[str, dict[str, dict[str, Any]]] = {}
    for hz in available_horizons:
        leaderboards[hz] = build_leaderboards_by_crop(df, batch_horizon=hz, skilled_only=False)
        leaderboards_skilled[hz] = build_leaderboards_by_crop(
            df, batch_horizon=hz, skilled_only=True
        )
        model_country[hz] = build_model_country_by_crop(df, batch_horizon=hz, skilled_only=False)
        model_country_skilled[hz] = build_model_country_by_crop(
            df, batch_horizon=hz, skilled_only=True
        )

    countries = sorted(df["country"].unique()) if "country" in df.columns else []
    crops = _crop_keys(df)
    baseline_models = sorted({str(m) for m in df["model"].unique() if is_baseline_model(m)})
    horizon_labels = {hz: HORIZON_DISPLAY_LABELS.get(hz, hz) for hz in available_horizons}

    country_map_cc = {str(cc): map_iso_for_cybencH(str(cc)) for cc in countries}
    benchmark_map_isos = sorted(set(country_map_cc.values()))
    return {
        "output_root": str(output_root.resolve()),
        "dashboard_hrefs": build_dashboard_hrefs(output_root, version=version),
        "matrix_axes": matrix_axes_payload(),
        "available_horizons": list(available_horizons),
        "horizon_labels": horizon_labels,
        "n_summary_files": len(paths),
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()) if "country" in df.columns else 0,
        "countries": countries,
        "crops": crops,
        "baseline_models": baseline_models,
        "leaderboards": leaderboards,
        "leaderboards_skilled": leaderboards_skilled,
        "model_country": model_country,
        "model_country_skilled": model_country_skilled,
        "horizon_skill_curves": build_horizon_skill_curves_payload(df),
        "crop_comparison": build_crop_comparison_payload(df),
        "country_map_cc": country_map_cc,
        "benchmark_map_isos": benchmark_map_isos,
        "map_coverage_note": MAP_COVERAGE_NOTE,
        "metric_map_scales": METRIC_MAP_SCALES,
    }


def _overall_crop_pair_stats(
    detail: pd.DataFrame, crop_a: str, crop_b: str
) -> dict[str, Any]:
    if detail.empty:
        return {
            "crop_a": crop_a,
            "crop_b": crop_b,
            "interpretation": (
                f"No paired {crop_a}/{crop_b} comparisons in countries with both crops."
            ),
        }
    weights = detail["pair_weight"]
    paired_countries = sorted(detail["country"].unique())
    crop_a_label = crop_a.replace("_", " ")
    crop_b_label = crop_b.replace("_", " ")
    return {
        "crop_a": crop_a,
        "crop_b": crop_b,
        "n_pairs": int(len(detail)),
        "n_countries": int(len(paired_countries)),
        "paired_countries": paired_countries,
        "crop_a_win_rate": float(detail["crop_a_better"].mean()),
        "mean_delta_nrmse": float(detail["delta_nrmse"].mean()),
        "weighted_delta_nrmse": float(_weighted_mean(detail["delta_nrmse"], weights)),
        "mean_crop_a_nrmse": float(detail["crop_a_nrmse"].mean()),
        "mean_crop_b_nrmse": float(detail["crop_b_nrmse"].mean()),
        "interpretation": (
            f"Paired comparison in countries with both crops. "
            f"delta_nrmse = {crop_b_label} − {crop_a_label}; "
            f"positive values mean {crop_a_label} has lower NRMSE. "
            f"Crop A win % = share of country×model pairs where {crop_a_label} wins."
        ),
    }


def _df_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out.where(pd.notna(out), None).to_dict(orient="records")
