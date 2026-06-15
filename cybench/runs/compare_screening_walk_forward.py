#!/usr/bin/env python3
"""Compare pooled test metrics: screening (one model) vs walk-forward (per-year refit).

Both evaluate the same held-out years, but:
  - Screening: one model fit on train+val, predicts all test years together.
  - Walk-forward: refit before each test year (expanding train window).

Usage::

    poetry run python cybench/runs/compare_screening_walk_forward.py \\
        --baselines-dir ../output/baselines \\
        --output ../output/screening_vs_walk_forward.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from cybench.config import KEY_TARGET
from cybench.evaluation.aggregated_metrics import compute_report_metrics
from cybench.runs.collect_walk_forward_results import (
    discover_walk_forward_runs,
    flatten_report_metrics,
    load_pooled_predictions,
)
from cybench.util.prediction_horizon import parse_run_name_suffix

SCREENING_TOKEN = "_screening_"


@dataclass(frozen=True)
class ScreeningRun:
    crop: str
    country: str
    model: str
    horizon: str | None
    timestamp: str
    path: Path

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        return (self.crop, self.country, self.model, self.horizon)


def _parse_screening_dir(name: str, path: Path) -> ScreeningRun | None:
    if SCREENING_TOKEN not in name:
        return None
    prefix, suffix = name.split(SCREENING_TOKEN, 1)
    parts = prefix.split("_", 2)
    if len(parts) < 3:
        return None
    crop, country, model = parts
    horizon, timestamp = parse_run_name_suffix(suffix)
    return ScreeningRun(
        crop=crop,
        country=country,
        model=model,
        horizon=horizon,
        timestamp=timestamp,
        path=path,
    )


def discover_screening_runs(baselines_dir: Path) -> dict[tuple[str, str, str, str | None], ScreeningRun]:
    by_key: dict[tuple[str, str, str, str | None], ScreeningRun] = {}
    for entry in sorted(baselines_dir.iterdir()):
        if not entry.is_dir():
            continue
        run = _parse_screening_dir(entry.name, entry)
        if run is None:
            continue
        if run.key not in by_key or run.timestamp > by_key[run.key].timestamp:
            by_key[run.key] = run
    return by_key


def _screening_metrics_path(run_dir: Path, seed: int = 42) -> Path | None:
    """Pooled test split folder is multi-year (e.g. 2016_2017_...), not a single year."""
    for path in sorted(run_dir.glob(f"*/{seed}/report_metrics.yaml")):
        split_name = path.parent.parent.name
        if re.fullmatch(r"\d{4}", split_name):
            continue
        return path
    return None


def load_screening_flat(run_dir: Path, seed: int = 42) -> dict[str, Any] | None:
    metrics_path = _screening_metrics_path(run_dir, seed=seed)
    if metrics_path is None:
        return None
    raw = OmegaConf.to_container(OmegaConf.load(metrics_path))
    if not isinstance(raw, dict):
        return None
    ry = raw.get("region_year", {})
    sp = raw.get("spatial", {})
    tm = raw.get("temporal", {})
    return {
        "n_regions": raw.get("n_regions"),
        "n_years": raw.get("n_years"),
        "screening_r": ry.get("r"),
        "screening_r2": ry.get("r2"),
        "screening_nrmse": ry.get("nrmse"),
        "screening_r_spatial": sp.get("r"),
        "screening_r2_spatial": sp.get("r2"),
        "screening_r_temporal": tm.get("r"),
        "screening_r2_temporal": tm.get("r2"),
    }


def load_walk_forward_flat(run_dir: Path, model_slug: str) -> dict[str, Any] | None:
    try:
        df, model_col = load_pooled_predictions(run_dir, model_slug=model_slug)
    except ValueError:
        return None
    metrics = compute_report_metrics(df, target_col=KEY_TARGET, model_col=model_col)
    flat = flatten_report_metrics(metrics)
    return {
        "n_regions": flat.get("n_regions"),
        "n_years": flat.get("n_years"),
        "walk_forward_r": flat.get("r"),
        "walk_forward_r2": flat.get("r2"),
        "walk_forward_nrmse": flat.get("nrmse"),
        "walk_forward_r_spatial": flat.get("r_spatial"),
        "walk_forward_r2_spatial": flat.get("r2_spatial"),
        "walk_forward_r_temporal": flat.get("r_temporal"),
        "walk_forward_r2_temporal": flat.get("r2_temporal"),
    }


def compare_runs(
    baselines_dir: Path,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    screening = discover_screening_runs(baselines_dir)
    walk_forward = {
        (r.crop, r.country, r.model, r.horizon): r
        for r in discover_walk_forward_runs(baselines_dir, latest_only=True)
    }

    rows: list[dict[str, Any]] = []
    keys = sorted(set(screening) & set(walk_forward))
    for key in keys:
        s_run = screening[key]
        wf_run = walk_forward[key]
        s_flat = load_screening_flat(s_run.path, seed=seed)
        wf_flat = load_walk_forward_flat(wf_run.path, model_slug=s_run.model)
        if s_flat is None or wf_flat is None:
            continue
        row: dict[str, Any] = {
            "crop": s_run.crop,
            "country": s_run.country,
            "model": s_run.model,
            "horizon": s_run.horizon,
            "dataset": f"{s_run.crop}_{s_run.country}",
            "screening_run": str(s_run.path),
            "walk_forward_run": str(wf_run.path),
            **s_flat,
            **wf_flat,
        }
        if row.get("screening_r2") is not None and row.get("walk_forward_r2") is not None:
            row["delta_r2"] = float(row["walk_forward_r2"]) - float(row["screening_r2"])
        if row.get("screening_nrmse") is not None and row.get("walk_forward_nrmse") is not None:
            row["delta_nrmse"] = float(row["walk_forward_nrmse"]) - float(
                row["screening_nrmse"]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path("../output/baselines"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../output/screening_vs_walk_forward.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = compare_runs(args.baselines_dir.resolve(), seed=args.seed)
    if df.empty:
        print("[WARN] No paired screening + walk-forward runs found.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.4f")

    print(f"[DONE] Wrote {len(df)} paired rows to {args.output}")
    if "delta_r2" in df.columns:
        improved = int((df["delta_r2"] > 0).sum())
        print(
            f"[INFO] Walk-forward higher region-year R² in {improved}/{len(df)} pairs "
            f"(negative delta = screening better)"
        )
    cols = ["dataset", "model", "screening_r2", "walk_forward_r2", "delta_r2"]
    show = [c for c in cols if c in df.columns]
    print(df[show].sort_values("delta_r2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
