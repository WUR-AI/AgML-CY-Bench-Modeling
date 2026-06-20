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


def is_baseline_model(model: object) -> bool:
    return str(model).lower().replace("-", "_") in _BASELINE_MODELS


def parse_paper_dir_name(name: str) -> tuple[str, str, int] | None:
    match = _PAPER_DIR_RE.match(name)
    if not match:
        return None
    return match.group("country").upper(), match.group("horizon"), int(match.group("version"))


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
    for col in ("nrmse", "r2", "n_samples", "n_regions", "n_years"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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
    return out


def aggregate_model_leaderboard(
    df: pd.DataFrame,
    *,
    batch_horizon: str | None = None,
    skilled_only: bool = False,
) -> pd.DataFrame:
    """Rank models using sample-weighted mean NRMSE and baseline-relative skill."""
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()

    work = attach_baseline_metrics(df)
    work = work[~work["model"].apply(is_baseline_model)]
    if batch_horizon:
        work = work[work["batch_horizon"] == batch_horizon]
    if skilled_only:
        work = work[work["skilled"]]
    work = work[work["nrmse"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    if "n_samples" not in work.columns:
        work["n_samples"] = 1.0
    work["n_samples"] = work["n_samples"].fillna(1).clip(lower=1)

    rows: list[dict[str, Any]] = []
    for model, grp in work.groupby("model", sort=True):
        comparable = grp[grp["baseline_nrmse"].notna()]
        if len(comparable):
            beat_rate = float(comparable["beats_baseline"].mean())
            weighted_beat = _weighted_mean(
                comparable["beats_baseline"].astype(float), comparable["n_samples"]
            )
        else:
            beat_rate = float("nan")
            weighted_beat = float("nan")

        rows.append(
            {
                "model": model,
                "weighted_nrmse": _weighted_mean(grp["nrmse"], grp["n_samples"]),
                "mean_nrmse": float(grp["nrmse"].mean()),
                "median_nrmse": float(grp["nrmse"].median()),
                "std_nrmse": float(grp["nrmse"].std(ddof=0)) if len(grp) > 1 else 0.0,
                "mean_r2": float(grp["r2"].mean()) if "r2" in grp else float("nan"),
                "beat_baseline_rate": beat_rate,
                "weighted_beat_baseline_rate": weighted_beat,
                "skilled_rate": float(grp["skilled"].mean()) if "skilled" in grp else float("nan"),
                "n_datasets": int(len(grp)),
                "n_countries": int(grp["country"].nunique()) if "country" in grp else 0,
                "total_samples": int(grp["n_samples"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["weighted_nrmse", "mean_nrmse"], ascending=[True, True])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def build_model_country_matrix(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
) -> dict[str, Any]:
    """Model × country matrix for heatmap (weighted NRMSE per model×country)."""
    work = attach_baseline_metrics(df)
    work = work[~work["model"].apply(is_baseline_model)]
    work = work[work["batch_horizon"] == batch_horizon]
    work = work[work["nrmse"].notna()].copy()
    if work.empty:
        return {"models": [], "countries": [], "cells": []}

    if "n_samples" not in work.columns:
        work["n_samples"] = 1.0
    work["n_samples"] = work["n_samples"].fillna(1).clip(lower=1)

    cells: list[dict[str, Any]] = []
    for (model, country), grp in work.groupby(["model", "country"], sort=True):
        comparable = grp[grp["baseline_nrmse"].notna()]
        cells.append(
            {
                "model": model,
                "country": country,
                "weighted_nrmse": _weighted_mean(grp["nrmse"], grp["n_samples"]),
                "mean_r2": float(grp["r2"].mean()) if "r2" in grp else None,
                "n_datasets": int(len(grp)),
                "beat_baseline_rate": float(comparable["beats_baseline"].mean())
                if len(comparable)
                else None,
            }
        )

    models = sorted({c["model"] for c in cells})
    countries = sorted({c["country"] for c in cells})
    return {"models": models, "countries": countries, "cells": cells}


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

    leaderboards: dict[str, list[dict[str, Any]]] = {}
    leaderboards_skilled: dict[str, list[dict[str, Any]]] = {}
    model_country: dict[str, dict[str, Any]] = {}
    for hz in ("eos", "mid"):
        board = aggregate_model_leaderboard(df, batch_horizon=hz, skilled_only=False)
        board_skilled = aggregate_model_leaderboard(df, batch_horizon=hz, skilled_only=True)
        leaderboards[hz] = _df_records(board)
        leaderboards_skilled[hz] = _df_records(board_skilled)
        model_country[hz] = build_model_country_matrix(df, batch_horizon=hz)

    countries = sorted(df["country"].unique()) if "country" in df.columns else []
    baseline_models = sorted({str(m) for m in df["model"].unique() if is_baseline_model(m)})
    return {
        "output_root": str(output_root.resolve()),
        "n_summary_files": len(paths),
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()) if "country" in df.columns else 0,
        "countries": countries,
        "baseline_models": baseline_models,
        "leaderboards": leaderboards,
        "leaderboards_skilled": leaderboards_skilled,
        "model_country": model_country,
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
