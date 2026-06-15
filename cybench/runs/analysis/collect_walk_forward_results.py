#!/usr/bin/env python3
"""Collect walk-forward Hydra runs into paper-ready metrics and prediction CSVs.

Walk-forward writes one split folder per forecast year. For the paper we pool
those years and compute the same report metrics / plots as screening, but each
year uses its own refit model (true walk-forward).

Typical workflow::

    poetry run python cybench/runs/analysis/collect_walk_forward_results.py \\
        --baselines-dir ../output/baselines \\
        --output-dir ../output/paper_walk_forward \\
        --plot

Then open ``../output/paper_walk_forward/plots/<model>/report.html``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
from cybench.runs.analysis.benchmark_run_catalog import (
    BenchmarkRun,
    discover_benchmark_runs,
    flatten_report_metrics,
)

from cybench.runs.viz.build_results_dashboard import (
    build_html,
    bundle_referenced_assets,
)


def _model_column_from_hydra(run_dir: Path) -> str | None:
    hydra_cfg = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return None
    cfg = OmegaConf.load(hydra_cfg)
    target = OmegaConf.select(cfg, "model._target_")
    if isinstance(target, str) and target:
        return target.rsplit(".", 1)[-1]
    return None


def _model_column_from_repo_config(model_slug: str) -> str | None:
    conf_path = Path(REPO_DIR) / "cybench" / "conf" / "model" / f"{model_slug}.yaml"
    if not conf_path.exists():
        return None
    cfg = OmegaConf.load(conf_path)
    target = OmegaConf.select(cfg, "_target_")
    if isinstance(target, str) and target:
        return target.rsplit(".", 1)[-1]
    return None


def resolve_model_column(run_dir: Path, model_slug: str) -> str:
    """Class name used as prediction column in test_preds.csv (from Hydra config)."""
    col = _model_column_from_hydra(run_dir)
    if col:
        return col
    col = _model_column_from_repo_config(model_slug)
    if col:
        return col
    raise ValueError(
        f"Could not resolve prediction column for model={model_slug!r} in {run_dir}"
    )


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


def load_pooled_predictions(run_dir: Path, *, model_slug: str) -> tuple[pd.DataFrame, str]:
    model_col = resolve_model_column(run_dir, model_slug)
    df = _load_split_preds(run_dir, model_col)
    if df is None:
        raise ValueError(f"No walk-forward predictions found in {run_dir}")
    return df, model_col


def _panel_search_dirs(output_dir: Path, row: dict[str, Any]) -> list[Path]:
    """Directories that may contain report_assets/ from visualize_results_aggregated."""
    model = str(row["model"])
    model_col = str(row.get("model_col") or model)
    horizon = row.get("horizon")
    candidates = [output_dir / "preds" / f"{model}_{horizon}"]
    candidates.append(output_dir / "plots" / model_col)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        out.append(path)
    return out


def _panel_images_for_dataset(
    output_dir: Path, row: dict[str, Any], dataset: str
) -> dict[str, str]:
    panel_names = ("map_actual", "map_pred", "scatter", "temporal")
    for base in _panel_search_dirs(output_dir, row):
        assets = base / "report_assets"
        rel: dict[str, str] = {}
        for name in panel_names:
            fp = assets / f"{dataset}_{name}.png"
            if fp.exists():
                rel[name] = os.path.relpath(fp, output_dir).replace(os.sep, "/")
        if rel:
            return rel
    return {}


def summary_rows_to_dashboard_records(
    summary_rows: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    """Flatten summary rows into records for build_results_dashboard.build_html."""
    metric_map = [
        ("region_year", "r", "r"),
        ("region_year", "r2", "r2"),
        ("region_year", "nrmse", "nrmse"),
        ("spatial", "r", "r_spatial"),
        ("spatial", "r2", "r2_spatial"),
        ("temporal", "r", "r_temporal"),
        ("temporal", "r2", "r2_temporal"),
        ("anomaly", "r", "r_res"),
        ("anomaly", "r2", "r2_res"),
    ]
    records: list[dict[str, Any]] = []
    for row in summary_rows:
        dataset = str(row["dataset"])
        images = _panel_images_for_dataset(output_dir, row, dataset)
        common = {
            "model": str(row["model"]),
            "dataset": dataset,
            "n_regions": row.get("n_regions"),
            "n_years": row.get("n_years"),
            "images": images,
        }
        for view, metric, key in metric_map:
            raw = row.get(key)
            value: float | None
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                value = None
            else:
                value = float(raw)
            records.append({**common, "view": view, "metric": metric, "value": value})
    return records


def write_model_comparison_dashboard(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    *,
    bundle_assets: bool = True,
) -> Path:
    records = summary_rows_to_dashboard_records(summary_rows, output_dir)
    if not records:
        raise ValueError("No summary rows available for dashboard.")
    html_dir = str(output_dir)
    if bundle_assets:
        records = bundle_referenced_assets(
            records=records,
            output_dir=html_dir,
            assets_dirname="assets",
        )
    html_path = output_dir / "compare_models.html"
    html_path.write_text(build_html(records), encoding="utf-8")
    return html_path


def export_year_csvs(df: pd.DataFrame, run: BenchmarkRun, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    year_col = KEY_YEAR if KEY_YEAR in df.columns else "year"
    for year, year_df in df.groupby(year_col):
        out_name = f"{run.dataset}_h{run.horizon}_year_{int(year)}.csv"
        year_df.to_csv(dest_dir / out_name, index=False, float_format="%.6f")


def run_visualize(
    preds_dir: Path,
    model_col: str,
    *,
    plot_output_dir: Path,
    min_years: int,
) -> None:
    script = Path(__file__).resolve().parent.parent / "viz" / "visualize_results_aggregated.py"
    plot_output_dir.mkdir(parents=True, exist_ok=True)
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
        str(plot_output_dir / "evaluation_plots.pdf"),
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
        "--dashboard",
        action="store_true",
        help="Write compare_models.html — all models side-by-side (use with or after --plot)",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Only rebuild compare_models.html from an existing --output-dir",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=3,
        help="Minimum years required for plotting (passed to visualize script)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dashboard_only:
        summary_path = output_dir / "walk_forward_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"--dashboard-only requires {summary_path}. Run collect first."
            )
        summary_rows = pd.read_csv(summary_path).to_dict(orient="records")
        html_path = write_model_comparison_dashboard(output_dir, summary_rows)
        print(f"[DONE] Multi-model dashboard: {html_path}")
        return

    baselines_dir = args.baselines_dir.resolve()
    preds_root = output_dir / "preds"

    runs = discover_benchmark_runs(
        baselines_dir, phase="walk_forward", latest_only=not args.all_runs
    )
    if not runs:
        print(f"[WARN] No walk_forward runs found in {baselines_dir}")
        return

    summary_rows: list[dict[str, Any]] = []
    models_to_plot: dict[str, Path] = {}

    for run in runs:
        try:
            df, model_col = load_pooled_predictions(run.path, model_slug=run.model)
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
            "dataset": run.dataset,
            "timestamp": run.timestamp,
            "run_dir": str(run.path),
            **flat,
        }
        summary_rows.append(row)

        model_preds_dir = preds_root / f"{run.model}_{run.horizon}"
        export_year_csvs(df, run, model_preds_dir)
        models_to_plot[model_col] = model_preds_dir

        metrics_path = output_dir / "metrics" / f"{run.dataset}_{run.model}.yaml"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)

        print(
            f"[OK] {run.dataset} | {run.model} | "
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
            run_visualize(
                preds_dir,
                model_col,
                plot_output_dir=output_dir / "plots" / model_col,
                min_years=args.min_years,
            )

    if args.dashboard:
        html_path = write_model_comparison_dashboard(output_dir, summary_rows)
        print(f"[DONE] Multi-model dashboard: {html_path}")


if __name__ == "__main__":
    main()
