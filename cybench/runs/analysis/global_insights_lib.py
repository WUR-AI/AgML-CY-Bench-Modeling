"""Aggregate walk-forward summaries for cross-country and horizon comparisons."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

_PAPER_DIR_RE = re.compile(
    r"^paper_walk_forward_(?P<country>[a-z]{2})_(?P<horizon>eos|mid)_v(?P<version>\d+)$"
)

_BASELINE_MODELS = frozenset({"average", "averageyieldmodel", "average_yield"})

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
        "note": "R² on regional means (aggregate across years).",
        "metrics": (
            {"id": "r2", "column": "r2_spatial_agg", "label": "R²", "higher_better": True},
        ),
    },
    {
        "id": "temporal",
        "label": "Temporal",
        "note": "R² on yearly national means (aggregate across regions).",
        "metrics": (
            {"id": "r2", "column": "r2_temporal_agg", "label": "R²", "higher_better": True},
        ),
    },
    {
        "id": "anomaly",
        "label": "Anomaly",
        "note": "Pooled R² on location-de-meaned yields (r2_res, else r2_anomaly).",
        "metrics": (
            {"id": "r2", "column": "r2_res", "label": "R²", "higher_better": True},
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
    "r2_spatial_agg",
    "r2_temporal",
    "r2_temporal_agg",
    "r2_anomaly",
    "r2_res",
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


def build_dashboard_hrefs(output_root: Path, *, version: int = 1) -> dict[str, dict[str, str]]:
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


def discover_summary_tables(output_root: Path, *, version: int = 1) -> list[Path]:
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
    """Map pre-v2 walk_forward_summary columns to current aggregate metric names.

    Older collects stored aggregate spatial/temporal R² in ``r2_spatial`` /
    ``r2_temporal``; current schema uses ``r2_spatial_agg`` / ``r2_temporal_agg``.
    """
    if df.empty:
        return df
    out = df.copy()
    if "r2_spatial_agg" not in out.columns and "r2_spatial" in out.columns:
        out["r2_spatial_agg"] = out["r2_spatial"]
    if "r2_temporal_agg" not in out.columns and "r2_temporal" in out.columns:
        out["r2_temporal_agg"] = out["r2_temporal"]
    if "r2_res" not in out.columns and "r2_anomaly" in out.columns:
        out["r2_res"] = out["r2_anomaly"]
    return out


def _series_for_matrix_column(grp: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric series for a matrix column, with anomaly fallbacks."""
    if column == "r2_res":
        if "r2_res" in grp.columns:
            s = pd.to_numeric(grp["r2_res"], errors="coerce")
            if "r2_anomaly" in grp.columns:
                return s.fillna(pd.to_numeric(grp["r2_anomaly"], errors="coerce"))
            return s
        if "r2_anomaly" in grp.columns:
            return pd.to_numeric(grp["r2_anomaly"], errors="coerce")
        return pd.Series(dtype=float)
    if column not in grp.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(grp[column], errors="coerce")


def _median_in_group(grp: pd.DataFrame, column: str) -> float | None:
    vals = _series_for_matrix_column(grp, column).dropna()
    if vals.empty:
        return None
    return float(vals.median())


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
) -> pd.DataFrame:
    work = attach_baseline_metrics(df)
    if batch_horizon is not None:
        work = work[work["batch_horizon"] == batch_horizon]
    if crop:
        work = work[work["crop"] == crop]
    if skilled_only:
        work = work[work["skilled"]]
    return work[work["nrmse"].notna()].copy()


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


def build_insights_payload(output_root: Path, *, version: int = 1) -> dict[str, Any]:
    """Build JSON-serializable payload for the global insights dashboard."""
    paths = discover_summary_tables(output_root, version=version)
    df = load_summary_frame(paths)
    horizon_detail, horizon_summary = compare_horizons(df)

    leaderboards: dict[str, dict[str, list[dict[str, Any]]]] = {}
    leaderboards_skilled: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_country: dict[str, dict[str, dict[str, Any]]] = {}
    model_country_skilled: dict[str, dict[str, dict[str, Any]]] = {}
    for hz in ("eos", "mid"):
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
    return {
        "output_root": str(output_root.resolve()),
        "dashboard_hrefs": build_dashboard_hrefs(output_root, version=version),
        "matrix_axes": matrix_axes_payload(),
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
        "horizon_summary": _df_records(horizon_summary),
        "horizon_detail": _df_records(horizon_detail),
        "overall_horizon": _overall_horizon_stats(horizon_detail),
    }


def _overall_horizon_stats(detail: pd.DataFrame) -> dict[str, Any]:
    if detail.empty:
        return {}
    weights = detail["pair_weight"]
    return {
        "n_pairs": int(len(detail)),
        "eos_win_rate": float(detail["eos_better"].mean()),
        "mean_delta_nrmse": float(detail["delta_nrmse"].mean()),
        "weighted_delta_nrmse": float(_weighted_mean(detail["delta_nrmse"], weights)),
        "interpretation": (
            "delta_nrmse = mid − eos; positive values mean end-of-season (nowcast) "
            "has lower NRMSE than mid-season."
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
