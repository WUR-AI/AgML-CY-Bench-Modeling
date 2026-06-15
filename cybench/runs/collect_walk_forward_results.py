#!/usr/bin/env python3
"""Collect walk-forward Hydra runs into paper-ready metrics and prediction CSVs.

Walk-forward writes one split folder per forecast year. For the paper we pool
those years and compute the same report metrics / plots as screening, but each
year uses its own refit model (true walk-forward).

Typical workflow::

    poetry run python cybench/runs/collect_walk_forward_results.py \\
        --baselines-dir ../output/baselines \\
        --output-dir ../output/paper_walk_forward \\
        --plot

Then open ``../output/paper_walk_forward/plots/<model>/report.html``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import OmegaConf

from cybench.config import KEY_COUNTRY, KEY_LOC, KEY_TARGET, KEY_YEAR, REPO_DIR
from cybench.evaluation.aggregated_metrics import (
    compute_report_metrics,
    format_report_metrics,
)
from cybench.util.prediction_horizon import parse_run_name_suffix

PHASE_TOKEN = "_walk_forward_"
YEAR_CSV_RE = re.compile(
    r"^[a-z]+_[A-Z]{2}(?:_h[a-z0-9_]+)?_year_(?:\d+_)*\d+\.csv$"
)


@dataclass(frozen=True)
class WalkForwardRun:
    crop: str
    country: str
    model: str
    horizon: str | None
    timestamp: str
    path: Path

    @property
    def dataset_key(self) -> str:
        return f"{self.crop}_{self.country}"


def _parse_walk_forward_dir(name: str, path: Path) -> WalkForwardRun | None:
    if PHASE_TOKEN not in name:
        return None
    prefix, suffix = name.split(PHASE_TOKEN, 1)
    parts = prefix.split("_", 2)
    if len(parts) < 3:
        return None
    crop, country, model = parts
    horizon, timestamp = parse_run_name_suffix(suffix)
    return WalkForwardRun(
        crop=crop,
        country=country,
        model=model,
        horizon=horizon,
        timestamp=timestamp,
        path=path,
    )


def discover_walk_forward_runs(
    baselines_dir: Path,
    *,
    latest_only: bool = True,
) -> list[WalkForwardRun]:
    if not baselines_dir.is_dir():
        raise FileNotFoundError(f"Baselines directory not found: {baselines_dir}")

    by_key: dict[tuple[str, str, str, str | None], WalkForwardRun] = {}
    for entry in sorted(baselines_dir.iterdir()):
        if not entry.is_dir():
            continue
        run = _parse_walk_forward_dir(entry.name, entry)
        if run is None:
            continue
        key = (run.crop, run.country, run.model, run.horizon)
        if key not in by_key or run.timestamp > by_key[key].timestamp:
            by_key[key] = run

    runs = sorted(by_key.values(), key=lambda r: (r.crop, r.country, r.model, r.horizon or ""))
    if not latest_only:
        all_runs: list[WalkForwardRun] = []
        for entry in sorted(baselines_dir.iterdir()):
            if not entry.is_dir():
                continue
            run = _parse_walk_forward_dir(entry.name, entry)
            if run is not None:
                all_runs.append(run)
        return sorted(
            all_runs,
            key=lambda r: (r.crop, r.country, r.model, r.horizon or "", r.timestamp),
        )
    return runs


def _model_column_from_config(run_dir: Path) -> str | None:
    for cfg_path in sorted(run_dir.glob("*/model_config.yaml")):
        cfg = OmegaConf.load(cfg_path)
        target = OmegaConf.select(cfg, "_target_")
        if isinstance(target, str) and target:
            return target.rsplit(".", 1)[-1]
    return None


def _load_year_csvs(run_dir: Path) -> pd.DataFrame | None:
    year_files = sorted(
        p for p in run_dir.glob("*.csv") if YEAR_CSV_RE.match(p.name)
    )
    if not year_files:
        return None
    frames = [pd.read_csv(p) for p in year_files]
    return pd.concat(frames, ignore_index=True)


def _load_split_preds(run_dir: Path, model_col: str) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for split_dir in sorted(run_dir.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.isdigit():
            continue
        pred_paths = sorted(split_dir.glob("*/test_preds.csv"))
        if not pred_paths:
            continue
        raw = pd.read_csv(pred_paths[0])
        if "targets" not in raw.columns or "preds" not in raw.columns:
            continue
        out = raw.copy()
        if KEY_YEAR not in out.columns and "year" in out.columns:
            out = out.rename(columns={"year": KEY_YEAR})
        if KEY_LOC not in out.columns and "adm_id" in out.columns:
            out = out.rename(columns={"adm_id": KEY_LOC})
        country = run_dir.name.split("_")[1] if "_" in run_dir.name else None
        if KEY_COUNTRY not in out.columns and country:
            out[KEY_COUNTRY] = country
        formatted = out.rename(columns={"targets": KEY_TARGET, "preds": model_col})
        keep = [c for c in [KEY_COUNTRY, KEY_LOC, KEY_YEAR, KEY_TARGET, model_col] if c in formatted.columns]
        frames.append(formatted[keep])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_pooled_predictions(run_dir: Path) -> tuple[pd.DataFrame, str]:
    model_col = _model_column_from_config(run_dir)
    df = _load_year_csvs(run_dir)
    if df is not None:
        if model_col is None:
            candidates = [c for c in df.columns if c not in {KEY_COUNTRY, KEY_LOC, KEY_YEAR, KEY_TARGET, "year", "adm_id", "yield"}]
            if len(candidates) != 1:
                raise ValueError(f"Could not infer prediction column in {run_dir}")
            model_col = candidates[0]
        if KEY_TARGET not in df.columns and "yield" in df.columns:
            df = df.rename(columns={"yield": KEY_TARGET})
        if KEY_YEAR not in df.columns and "year" in df.columns:
            df = df.rename(columns={"year": KEY_YEAR})
        if KEY_LOC not in df.columns and "adm_id" in df.columns:
            df = df.rename(columns={"adm_id": KEY_LOC})
        return df, model_col

    if model_col is None:
        raise ValueError(f"No model_config.yaml found under {run_dir}")
    df = _load_split_preds(run_dir, model_col)
    if df is None:
        raise ValueError(f"No walk-forward predictions found in {run_dir}")
    return df, model_col


def flatten_report_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    ry = metrics.get("region_year", {})
    sp = metrics.get("spatial", {})
    tm = metrics.get("temporal", {})
    return {
        "n_regions": metrics.get("n_regions"),
        "n_years": metrics.get("n_years"),
        "n_samples": metrics.get("n_samples"),
        "r": ry.get("r"),
        "r2": ry.get("r2"),
        "nrmse": ry.get("nrmse"),
        "r_res": ry.get("r_res"),
        "r2_res": ry.get("r2_res"),
        "r_spatial": sp.get("r"),
        "r2_spatial": sp.get("r2"),
        "r_temporal": tm.get("r"),
        "r2_temporal": tm.get("r2"),
    }


def export_year_csvs(df: pd.DataFrame, run: WalkForwardRun, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    year_col = KEY_YEAR if KEY_YEAR in df.columns else "year"
    horizon_part = f"_h{run.horizon}" if run.horizon else ""
    for year, year_df in df.groupby(year_col):
        out_name = f"{run.dataset_key}{horizon_part}_year_{int(year)}.csv"
        year_df.to_csv(dest_dir / out_name, index=False, float_format="%.6f")


def run_visualize(
    preds_dir: Path,
    model_col: str,
    *,
    min_years: int,
) -> None:
    script = Path(__file__).resolve().parent / "visualize_results_aggregated.py"
    plots_dir = preds_dir.parent / "plots" / model_col
    plots_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--results_dir",
        str(preds_dir),
        "--model",
        model_col,
        "--min_years",
        str(min_years),
        "--save_individual",
        "--output_pdf",
        str(plots_dir / "evaluation_plots.pdf"),
    ]
    print(f"[INFO] Plotting {model_col} from {preds_dir}")
    proc = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(f"[WARN] Plotting failed for {model_col}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path("../output/baselines"),
        help="Hydra baselines root (default: ../output/baselines)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write summary CSV, pooled preds, and optional plots",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every timestamp, not only the latest per crop/country/model",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run visualize_results_aggregated.py per model after export",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=3,
        help="Minimum years required for plotting (passed to visualize script)",
    )
    args = parser.parse_args()

    baselines_dir = args.baselines_dir.resolve()
    output_dir = args.output_dir.resolve()
    preds_root = output_dir / "preds"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_walk_forward_runs(baselines_dir, latest_only=not args.all_runs)
    if not runs:
        print(f"[WARN] No walk_forward runs found in {baselines_dir}")
        return

    summary_rows: list[dict[str, Any]] = []
    models_to_plot: dict[str, Path] = {}

    for run in runs:
        try:
            df, model_col = load_pooled_predictions(run.path)
        except ValueError as exc:
            print(f"[SKIP] {run.path.name}: {exc}")
            continue

        metrics = compute_report_metrics(df, target_col=KEY_TARGET, model_col=model_col)
        flat = flatten_report_metrics(metrics)
        row = {
            "crop": run.crop,
            "country": run.country,
            "model": run.model,
            "horizon": run.horizon,
            "model_col": model_col,
            "dataset": run.dataset_key,
            "timestamp": run.timestamp,
            "run_dir": str(run.path),
            **flat,
        }
        summary_rows.append(row)

        model_preds_dir = preds_root / (
            f"{run.model}_{run.horizon}" if run.horizon else run.model
        )
        export_year_csvs(df, run, model_preds_dir)
        models_to_plot[model_col] = model_preds_dir

        metrics_path = output_dir / "metrics" / f"{run.dataset_key}_{run.model}.yaml"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        print(
            f"[OK] {run.dataset_key} | {run.model} | "
            f"{format_report_metrics(metrics)}"
        )

    summary_path = output_dir / "walk_forward_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, float_format="%.4f")
    print(f"\n[DONE] Summary: {summary_path} ({len(summary_rows)} rows)")
    print(f"[DONE] Pooled year CSVs: {preds_root}")

    manifest = {
        "baselines_dir": str(baselines_dir),
        "n_runs": len(summary_rows),
        "models": sorted({r["model"] for r in summary_rows}),
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    if args.plot:
        for model_col, preds_dir in sorted(models_to_plot.items()):
            plot_preds_dir = output_dir / "plots" / model_col / "preds"
            if plot_preds_dir.exists():
                shutil.rmtree(plot_preds_dir)
            shutil.copytree(preds_dir, plot_preds_dir)
            run_visualize(plot_preds_dir, model_col, min_years=args.min_years)


if __name__ == "__main__":
    main()
