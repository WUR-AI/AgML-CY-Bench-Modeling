"""Aggregate walk-forward metrics into model-family radar chart payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cybench.runs.analysis.benchmark_run_catalog import HIGHER_IS_BETTER, LOWER_IS_BETTER
from cybench.runs.analysis.global_insights_lib import (
    attach_baseline_metrics,
    discover_summary_tables,
    is_baseline_model,
    load_summary_frame,
)

# Scientific views → headline metric per axis (from aggregated_metrics / METRIC_KEYS).
EVALUATION_VIEWS: tuple[dict[str, str], ...] = (
    {
        "label": "Overall",
        "metric": "r2",
        "question": "Can the model predict crop yields accurately?",
    },
    {
        "label": "Spatial",
        "metric": "r2_spatial",
        "question": "For a typical year, can it reproduce spatial productivity patterns?",
    },
    {
        "label": "Temporal",
        "metric": "r2_temporal",
        "question": "For a typical region, can it reproduce year-to-year yield dynamics?",
    },
    {
        "label": "Anomaly",
        "metric": "r2_anomaly",
        "question": "Can it predict deviations from a region's expected yield?",
    },
)

VIEW_METRICS: tuple[str, ...] = tuple(v["metric"] for v in EVALUATION_VIEWS)

MODEL_FAMILIES: dict[str, list[str]] = {
    "Process-Based": ["lpjml_bc", "twso_bc"],
    "Feature-Engineered ML": ["lightgbm", "xgboost", "random_forest", "ridge"],
    "Sequence / Deep TS": [
        "transformer_lf",
        "lstm_lf",
        "patchtst_lf",
        "dlinear_lf",
        "nlinear_lf",
        "autoformer_lf",
        "informer_lf",
        "tst_lf",
        "cnn_lf",
    ],
    "Tabular Foundation": ["tabpfn", "tabicl", "tabdpt"],
}

DEFAULT_REPRESENTATIVES: dict[str, str] = {
    "Process-Based": "lpjml_bc",
    "Feature-Engineered ML": "lightgbm",
    "Sequence / Deep TS": "transformer_lf",
    "Tabular Foundation": "tabpfn",
}

FAMILY_COLORS: dict[str, str] = {
    "Process-Based": "#e76f51",
    "Feature-Engineered ML": "#2a9d8f",
    "Sequence / Deep TS": "#457b9d",
    "Tabular Foundation": "#9b5de5",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "lpjml_bc": "LPJmL",
    "twso_bc": "TWSO",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "ridge": "Ridge",
    "transformer_lf": "Transformer",
    "lstm_lf": "LSTM",
    "patchtst_lf": "PatchTST",
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
}


def _metric_higher_is_better(metric: str) -> bool:
    if metric in HIGHER_IS_BETTER:
        return True
    if metric in LOWER_IS_BETTER:
        return False
    return True


def _median_per_model(df: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Median metric per model (one value per crop×country row in *df*)."""
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()
    work = df[~df["model"].apply(is_baseline_model)]
    if work.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in work.columns]
    if not present:
        return pd.DataFrame()
    grouped = work.groupby("model", sort=True)[present].median()
    return grouped


def pick_representatives(
    df: pd.DataFrame,
    *,
    selection_metric: str = "r2",
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Pick one model slug per family (default: best median overall R² in frame)."""
    overrides = dict(overrides or DEFAULT_REPRESENTATIVES)
    chosen: dict[str, str] = {}
    models_in_frame = set(df["model"].astype(str)) if "model" in df.columns else set()

    for family, candidates in MODEL_FAMILIES.items():
        override = overrides.get(family)
        if override and override in models_in_frame:
            chosen[family] = override
            continue
        sub = df[df["model"].isin(candidates)] if "model" in df.columns else pd.DataFrame()
        if sub.empty or selection_metric not in sub.columns:
            continue
        med = sub.groupby("model")[selection_metric].median()
        med = med[med.notna()]
        if med.empty:
            continue
        chosen[family] = str(med.idxmax())
    return chosen


def relative_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Min–max normalize each view column across all models in *raw* (higher radius = better)."""
    out = raw.copy()
    for view in EVALUATION_VIEWS:
        label = view["label"]
        metric = view["metric"]
        if metric not in out.columns:
            out[label] = float("nan")
            continue
        vals = out[metric].astype(float)
        higher = _metric_higher_is_better(metric)
        lo = float(vals.min())
        hi = float(vals.max())
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            out[label] = 0.5
        else:
            norm = (vals - lo) / (hi - lo)
            out[label] = norm if higher else 1.0 - norm
    return out


def _family_records(
    medians: pd.DataFrame,
    representatives: dict[str, str],
) -> list[dict[str, Any]]:
    view_labels = [v["label"] for v in EVALUATION_VIEWS]
    rel_all = relative_scores(medians.copy()) if not medians.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for family, model in representatives.items():
        if model not in medians.index:
            continue
        raw_row = medians.loc[model]
        raw: dict[str, float | None] = {}
        for view in EVALUATION_VIEWS:
            val = raw_row.get(view["metric"])
            raw[view["metric"]] = None if pd.isna(val) else round(float(val), 4)
        relative = {
            label: (
                round(float(rel_all[label].loc[model]), 4)
                if label in rel_all.columns and model in rel_all.index
                else None
            )
            for label in view_labels
        }
        rows.append(
            {
                "family": family,
                "model": model,
                "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                "color": FAMILY_COLORS.get(family, "#666"),
                "raw": raw,
                "relative": relative,
            }
        )
    return rows


def family_for_model(model: str) -> str | None:
    for family, models in MODEL_FAMILIES.items():
        if model in models:
            return family
    return None


SAMPLE_SCATTER_METRIC: dict[str, Any] = {
    "key": "relative_nrmse",
    "label": "NRMSE / average yield",
    "baseline_model": "average_yield",
    "lower_is_better": True,
    "reference": 1.0,
}


def build_sample_scatter_slice(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
    crop: str | None = None,
    representatives: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One representative per family: relative NRMSE vs average_yield by training size."""
    work = df[df["batch_horizon"] == batch_horizon].copy() if "batch_horizon" in df.columns else df
    if crop:
        work = work[work["crop"] == crop]
    if work.empty:
        return []

    for key in ("nrmse", "n_train"):
        if key in work.columns:
            work[key] = pd.to_numeric(work[key], errors="coerce")

    work = attach_baseline_metrics(work)
    reps = pick_representatives(work, overrides=representatives)

    families_out: list[dict[str, Any]] = []
    for family, model in reps.items():
        sub = work[work["model"] == model]
        if sub.empty:
            continue
        points: list[dict[str, Any]] = []
        for _, row in sub.iterrows():
            n_train = row.get("n_train")
            baseline_nrmse = row.get("baseline_nrmse")
            nrmse = row.get("nrmse")
            if pd.isna(n_train) or int(n_train) <= 0:
                continue
            if pd.isna(baseline_nrmse) or float(baseline_nrmse) <= 0 or pd.isna(nrmse):
                continue
            rel = float(nrmse) / float(baseline_nrmse)
            points.append(
                {
                    "model": model,
                    "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "country": str(row.get("country", "")),
                    "crop": str(row.get("crop", "")),
                    "dataset": f"{row.get('crop', '')}_{row.get('country', '')}",
                    "n_train": int(n_train),
                    "nrmse": round(float(nrmse), 4),
                    "baseline_nrmse": round(float(baseline_nrmse), 4),
                    "relative_nrmse": round(rel, 4),
                }
            )
        if points:
            families_out.append(
                {
                    "family": family,
                    "model": model,
                    "display_name": MODEL_DISPLAY_NAMES.get(model, model),
                    "color": FAMILY_COLORS.get(family, "#666"),
                    "points": points,
                }
            )
    return families_out


def summarize_sample_scatter(families: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline stats for the training-size scatter (percentiles + rank correlation)."""
    rows: list[dict[str, Any]] = []
    for fam in families:
        for p in fam.get("points") or []:
            rel = p.get("relative_nrmse")
            n_train = p.get("n_train")
            if rel is None or n_train is None:
                continue
            rows.append(
                {
                    "family": fam.get("family"),
                    "n_train": int(n_train),
                    "relative_nrmse": float(rel),
                }
            )
    if not rows:
        return {}

    frame = pd.DataFrame(rows)
    n = len(frame)
    p05 = float(frame["n_train"].quantile(0.05))
    p95 = float(frame["n_train"].quantile(0.95))
    core = frame[(frame["n_train"] >= p05) & (frame["n_train"] <= p95)]
    rho = float(core["n_train"].corr(core["relative_nrmse"], method="spearman")) if len(core) >= 5 else float("nan")

    per_family: dict[str, float | None] = {}
    for family, grp in core.groupby("family"):
        if len(grp) < 4:
            per_family[str(family)] = None
        else:
            per_family[str(family)] = round(
                float(grp["n_train"].corr(grp["relative_nrmse"], method="spearman")), 3
            )

    return {
        "n_points": n,
        "n_outliers_x": int(n - len(core)),
        "x_p05": int(round(p05)),
        "x_p50": int(round(float(frame["n_train"].median()))),
        "x_p95": int(round(p95)),
        "x_min": int(frame["n_train"].min()),
        "x_max": int(frame["n_train"].max()),
        "spearman_rho_core": None if pd.isna(rho) else round(rho, 3),
        "spearman_family_core": per_family,
    }


def build_sample_scatter_payload(
    df: pd.DataFrame,
    *,
    representatives: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    by_horizon: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for hz in ("eos", "mid"):
        if "batch_horizon" in df.columns and hz not in set(df["batch_horizon"].astype(str)):
            continue
        by_crop: dict[str, Any] = {}
        all_fams = build_sample_scatter_slice(
            df, batch_horizon=hz, representatives=representatives
        )
        by_crop["all"] = {
            "families": all_fams,
            "summary": summarize_sample_scatter(all_fams),
        }
        if "crop" in df.columns:
            for crop in sorted({str(c) for c in df["crop"].dropna().unique()}):
                fams = build_sample_scatter_slice(
                    df, batch_horizon=hz, crop=crop, representatives=representatives
                )
                by_crop[crop] = {
                    "families": fams,
                    "summary": summarize_sample_scatter(fams),
                }
        by_horizon[hz] = by_crop
    return by_horizon


def build_radar_slice(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
    crop: str | None = None,
    representatives: dict[str, str] | None = None,
) -> dict[str, Any]:
    work = df[df["batch_horizon"] == batch_horizon].copy() if "batch_horizon" in df.columns else df
    if crop:
        work = work[work["crop"] == crop]
    medians = _median_per_model(work, VIEW_METRICS)
    reps = pick_representatives(work, overrides=representatives)
    families = _family_records(medians, reps)
    return {
        "batch_horizon": batch_horizon,
        "crop": crop or "all",
        "n_datasets": int(len(work)),
        "representatives": reps,
        "families": families,
    }


def build_radar_payload(
    output_root: Path,
    *,
    version: int = 1,
    representatives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """JSON payload for the model-family radar dashboard."""
    paths = discover_summary_tables(output_root, version=version)
    df = load_summary_frame(paths)
    for metric in VIEW_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    by_horizon: dict[str, dict[str, Any]] = {}
    for hz in ("eos", "mid"):
        if "batch_horizon" in df.columns and hz not in set(df["batch_horizon"].astype(str)):
            continue
        by_crop: dict[str, Any] = {
            "all": build_radar_slice(df, batch_horizon=hz, representatives=representatives),
        }
        if "crop" in df.columns:
            for crop in sorted({str(c) for c in df["crop"].dropna().unique()}):
                by_crop[crop] = build_radar_slice(
                    df, batch_horizon=hz, crop=crop, representatives=representatives
                )
        by_horizon[hz] = by_crop

    crops = sorted({str(c) for c in df["crop"].dropna().unique()}) if "crop" in df.columns else []
    return {
        "output_root": str(output_root.resolve()),
        "n_summary_files": len(paths),
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()) if "country" in df.columns else 0,
        "countries": sorted(df["country"].unique()) if "country" in df.columns else [],
        "crops": crops,
        "views": list(EVALUATION_VIEWS),
        "family_catalog": {
            family: {"models": models, "color": FAMILY_COLORS.get(family, "#666")}
            for family, models in MODEL_FAMILIES.items()
        },
        "by_horizon": by_horizon,
        "sample_scatter_metric": SAMPLE_SCATTER_METRIC,
        "sample_scatter": build_sample_scatter_payload(df, representatives=representatives),
        "normalization_note": (
            "Each axis is min–max normalized across all models in the selected horizon "
            "and crop filter. Radar vertices show one representative per family; radii "
            "indicate where that representative sits relative to the full model field."
        ),
    }
