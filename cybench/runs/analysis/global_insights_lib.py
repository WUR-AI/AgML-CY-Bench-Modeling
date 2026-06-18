"""Aggregate walk-forward summaries for cross-country and horizon comparisons."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

_PAPER_DIR_RE = re.compile(
    r"^paper_walk_forward_(?P<country>[a-z]{2})_(?P<horizon>eos|mid)_v(?P<version>\d+)$"
)


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


def aggregate_model_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models across countries using sample-weighted mean NRMSE."""
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()

    work = df[df["nrmse"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    if "n_samples" not in work.columns:
        work["n_samples"] = 1.0
    work["n_samples"] = work["n_samples"].fillna(1).clip(lower=1)

    rows: list[dict[str, Any]] = []
    for model, grp in work.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "weighted_nrmse": _weighted_mean(grp["nrmse"], grp["n_samples"]),
                "mean_nrmse": float(grp["nrmse"].mean()),
                "median_nrmse": float(grp["nrmse"].median()),
                "mean_r2": float(grp["r2"].mean()) if "r2" in grp else float("nan"),
                "n_datasets": int(len(grp)),
                "n_countries": int(grp["country"].nunique()) if "country" in grp else 0,
                "total_samples": int(grp["n_samples"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["weighted_nrmse", "mean_nrmse"], ascending=[True, True])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


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
    leaderboard = aggregate_model_leaderboard(df)
    horizon_detail, horizon_summary = compare_horizons(df)

    def _df_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        out = frame.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].round(4)
        return out.where(pd.notna(out), None).to_dict(orient="records")

    countries = sorted(df["country"].unique()) if "country" in df.columns else []
    return {
        "output_root": str(output_root.resolve()),
        "n_summary_files": len(paths),
        "n_rows": int(len(df)),
        "n_countries": int(df["country"].nunique()) if "country" in df.columns else 0,
        "countries": countries,
        "leaderboard": _df_records(leaderboard),
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
