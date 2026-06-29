"""Aggregate walk-forward metrics into model-family radar chart payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cybench.runs.analysis.benchmark_run_catalog import HIGHER_IS_BETTER, LOWER_IS_BETTER
from cybench.runs.analysis.global_insights_lib import discover_summary_tables, load_summary_frame

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
    present = [m for m in metrics if m in df.columns]
    if not present:
        return pd.DataFrame()
    grouped = df.groupby("model", sort=True)[present].median()
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
        "normalization_note": (
            "Each axis is min–max normalized across all models in the selected horizon "
            "and crop filter. Radar vertices show one representative per family; radii "
            "indicate where that representative sits relative to the full model field."
        ),
    }
