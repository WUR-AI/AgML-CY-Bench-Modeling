#!/usr/bin/env python3
"""Collect walk-forward Hydra runs into paper-ready metrics and prediction CSVs.

Walk-forward writes one split folder per forecast year. For the paper we pool
those years and compute the same report metrics / plots as screening, but each
year uses its own refit model (true walk-forward).

When a run has multiple seed repetitions (``experiment.n_repetitions`` > 1),
metrics are computed per seed and summarized as mean ± sample std in
``walk_forward_summary.csv`` (primary columns are means; ``*_std`` columns hold
spread). Per-seed rows go to ``walk_forward_by_seed.csv``. Prediction CSVs and
plots use the lowest seed only (illustrative, not an ensemble).

Rows flagged in ``yield_quality_*`` sidecars are dropped at collect time using
``target.filter_samples`` from ``cybench/conf/dataset/target/yield.yaml`` (same
flags as training). Re-run collect after updating sidecars — no benchmark rerun
required for QC-only changes.

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
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from omegaconf import OmegaConf

from cybench.config import KEY_COUNTRY, KEY_LOC, KEY_TARGET, KEY_YEAR, PATH_DATA_DIR, REPO_DIR
from cybench.datasets.yield_quality import apply_yield_quality_filter, filter_samples_from_target
from cybench.evaluation.aggregated_metrics import compute_report_metrics
from cybench.runs.analysis.benchmark_run_catalog import (
    METRIC_KEYS,
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


COUNT_METRIC_KEYS: tuple[str, ...] = ("n_regions", "n_years", "n_samples")


def discover_run_seeds(run_dir: Path) -> list[int]:
    """Seed subfolders under walk-forward year splits (e.g. 2016/42/test_preds.csv)."""
    seeds: set[int] = set()
    for split_dir in run_dir.iterdir():
        if not split_dir.is_dir() or not re.fullmatch(r"\d{4}", split_dir.name):
            continue
        for child in split_dir.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            if (child / "test_preds.csv").exists():
                seeds.add(int(child.name))
    return sorted(seeds)


def aggregate_flat_metrics_across_seeds(
    per_seed_flat: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Mean and sample std per metric; primary keys (r, r2, …) hold the mean."""
    out: dict[str, Any] = {"n_seeds": len(seeds), "seeds": seeds}
    if not per_seed_flat:
        return out
    for key in COUNT_METRIC_KEYS:
        out[key] = per_seed_flat[0].get(key)
    for key in METRIC_KEYS:
        values = [
            float(row[key])
            for row in per_seed_flat
            if row.get(key) is not None and not pd.isna(row[key])
        ]
        if not values:
            out[key] = None
            out[f"{key}_std"] = None
            continue
        series = pd.Series(values, dtype=float)
        out[key] = float(series.mean())
        out[f"{key}_std"] = (
            float(series.std(ddof=1)) if len(values) > 1 else float("nan")
        )
    return out


def format_aggregated_metrics(summary: dict[str, Any]) -> str:
    """Human-readable line for logs: mean ± std when multiple seeds."""
    n_seeds = int(summary.get("n_seeds") or 1)

    def _fmt(key: str, *, decimals: int = 3) -> str | None:
        mean = summary.get(key)
        if mean is None or (isinstance(mean, float) and pd.isna(mean)):
            return None
        std = summary.get(f"{key}_std")
        if n_seeds > 1 and std is not None and not (isinstance(std, float) and pd.isna(std)):
            return f"{key}={float(mean):.{decimals}f}±{float(std):.{decimals}f}"
        return f"{key}={float(mean):.{decimals}f}"

    parts = [_fmt("r"), _fmt("r2"), _fmt("nrmse")]
    parts = [p for p in parts if p]
    if n_seeds > 1:
        parts.append(f"n_seeds={n_seeds}")
    return " | ".join(parts)


def _load_split_preds(
    run_dir: Path, model_col: str, *, seed: int
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for split_dir in sorted(run_dir.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.isdigit():
            continue
        pred_path = split_dir / str(seed) / "test_preds.csv"
        if not pred_path.exists():
            continue
        raw = pd.read_csv(pred_path)
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
        frames.append(cast(pd.DataFrame, formatted.loc[:, keep]))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_pooled_predictions(
    run_dir: Path, *, model_slug: str, seed: int | None = None
) -> tuple[pd.DataFrame, str]:
    model_col = resolve_model_column(run_dir, model_slug)
    seeds = discover_run_seeds(run_dir)
    if not seeds:
        raise ValueError(f"No walk-forward predictions found in {run_dir}")
    use_seed = seed if seed is not None else seeds[0]
    if use_seed not in seeds:
        raise ValueError(
            f"Seed {use_seed} not found in {run_dir}; available seeds: {seeds}"
        )
    df = _load_split_preds(run_dir, model_col, seed=use_seed)
    if df is None:
        raise ValueError(
            f"No walk-forward predictions for seed {use_seed} in {run_dir}"
        )
    return df, model_col


def _apply_quality_filter(
    df: pd.DataFrame,
    run: BenchmarkRun,
    *,
    data_dir: Path,
    quality_flags: list[str] | None,
    apply_qc: bool,
) -> pd.DataFrame:
    if not apply_qc or not quality_flags:
        return df
    n_before = len(df)
    filtered, n_removed = apply_yield_quality_filter(
        df,
        run.crop,
        run.country,
        data_dir=data_dir,
        quality_flags=quality_flags,
    )
    if n_removed:
        print(
            f"[QC] {run.dataset} | {run.model}: removed {n_removed}/{n_before} "
            f"flagged sample(s)"
        )
    return filtered


def collect_walk_forward_run(
    run: BenchmarkRun,
    *,
    data_dir: Path,
    quality_flags: list[str] | None,
    apply_qc: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, str, int] | None:
    """Per-run collect: summary (mean±std), per-seed rows, plot preds, model_col, plot_seed."""
    model_col = resolve_model_column(run.path, run.model)
    seeds = discover_run_seeds(run.path)
    if not seeds:
        return None

    per_seed_flat: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    plot_df: pd.DataFrame | None = None
    plot_seed = seeds[0]

    for seed in seeds:
        df = _load_split_preds(run.path, model_col, seed=seed)
        if df is None:
            print(f"[SKIP] {run.path.name}: missing predictions for seed {seed}")
            return None
        df = _apply_quality_filter(
            df, run, data_dir=data_dir, quality_flags=quality_flags, apply_qc=apply_qc
        )
        metrics = compute_report_metrics(df, target_col=KEY_TARGET, model_col=model_col)
        flat = flatten_report_metrics(metrics)
        per_seed_flat.append(flat)
        per_seed_rows.append(
            {
                "crop": run.crop,
                "country": run.country,
                "model": run.model,
                "horizon": run.horizon,
                "dataset": run.dataset,
                "timestamp": run.timestamp,
                "run_dir": str(run.path),
                "seed": seed,
                **flat,
            }
        )
        if seed == plot_seed:
            plot_df = df

    assert plot_df is not None
    summary = aggregate_flat_metrics_across_seeds(per_seed_flat, seeds)
    return summary, per_seed_rows, plot_df, model_col, plot_seed


def load_walk_forward_summary_metrics(
    run: BenchmarkRun,
    *,
    data_dir: Path | None = None,
    apply_qc: bool = True,
) -> dict[str, Any] | None:
    """Pooled walk-forward metrics; mean across seeds when multiple repetitions exist."""
    quality_flags = None if not apply_qc else filter_samples_from_target()
    if quality_flags == []:
        quality_flags = None
    result = collect_walk_forward_run(
        run,
        data_dir=data_dir or Path(PATH_DATA_DIR),
        quality_flags=quality_flags,
        apply_qc=apply_qc,
    )
    if result is None:
        return None
    summary, _, _, _, _ = result
    return summary


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
        ("spatial", "r2", "r2_spatial"),
        ("temporal", "r2", "r2_temporal"),
        ("anomaly", "r2", "r2_anomaly"),
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
            "n_samples": row.get("n_samples"),
            "images": images,
        }
        for view, metric, key in metric_map:
            raw = row.get(key)
            value: float | None
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                value = None
            else:
                value = float(raw)
            std_raw = row.get(f"{key}_std")
            value_std: float | None
            if std_raw is None or (isinstance(std_raw, float) and pd.isna(std_raw)):
                value_std = None
            else:
                value_std = float(std_raw)
            records.append(
                {
                    **common,
                    "view": view,
                    "metric": metric,
                    "value": value,
                    "value_std": value_std,
                }
            )
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
        out_name = f"{run.dataset}_h{run.horizon}_year_{int(cast(Any, year))}.csv"
        year_df.to_csv(dest_dir / out_name, index=False, float_format="%.6f")


def _run_matches_horizon(run: BenchmarkRun, horizon: str | None) -> bool:
    if not horizon:
        return True
    key = horizon.strip().lower().replace("-", "_")
    if key == "eos":
        return run.horizon == "eos"
    if key in {"mid", "mid_season", "middle_of_season"}:
        return run.horizon in {"mid_season", "mid"}
    return run.horizon == horizon


def _filter_runs(
    runs: list[BenchmarkRun],
    *,
    country: str | None,
    horizon: str | None,
) -> list[BenchmarkRun]:
    out: list[BenchmarkRun] = []
    cc = country.upper() if country else None
    for run in runs:
        if cc and run.country.upper() != cc:
            continue
        if not _run_matches_horizon(run, horizon):
            continue
        out.append(run)
    return out


def run_visualize(
    preds_dir: Path,
    model_col: str,
    *,
    model_label: str,
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
        "--dashboard-assets",
    ]
    print(f"[INFO] Plotting {model_label} ({model_col}) from {preds_dir}")
    proc = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(f"[WARN] Plotting failed for {model_label}", file=sys.stderr)
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
    parser.add_argument(
        "--country",
        help="Optional ISO-2 filter when baselines-dir contains many countries",
    )
    parser.add_argument(
        "--horizon",
        choices=["eos", "mid"],
        help="Optional horizon filter (eos or mid / mid_season runs)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(PATH_DATA_DIR),
        help="Root data directory with yield_quality sidecars (default: cybench/data)",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Do not drop rows flagged in yield_quality sidecars",
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
    runs = _filter_runs(runs, country=args.country, horizon=args.horizon)
    if not runs:
        print(f"[WARN] No walk_forward runs found in {baselines_dir}")
        if args.country or args.horizon:
            print(f"[WARN] After filters country={args.country!r} horizon={args.horizon!r}")
        return

    summary_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    # Key by model slug — torch models all use prediction column "TorchTrainer".
    models_to_plot: dict[str, tuple[Path, str]] = {}
    apply_qc = not args.no_quality_filter
    quality_flags = None if apply_qc else []
    if apply_qc:
        quality_flags = filter_samples_from_target()
        if quality_flags:
            print(f"[QC] Dropping rows flagged by: {', '.join(quality_flags)}")
        else:
            print("[QC] No target.filter_samples configured — keeping all rows")

    for run in runs:
        collected = collect_walk_forward_run(
            run,
            data_dir=args.data_dir,
            quality_flags=quality_flags,
            apply_qc=apply_qc,
        )
        if collected is None:
            print(f"[SKIP] {run.path.name}: no walk-forward predictions")
            continue

        summary, run_seed_rows, plot_df, model_col, plot_seed = collected
        per_seed_rows.extend(run_seed_rows)

        row = {
            "crop": run.crop,
            "country": run.country,
            "model": run.model,
            "horizon": run.horizon,
            "model_col": model_col,
            "dataset": run.dataset,
            "timestamp": run.timestamp,
            "run_dir": str(run.path),
            "plot_seed": plot_seed,
            **summary,
        }
        summary_rows.append(row)

        model_preds_dir = preds_root / f"{run.model}_{run.horizon}"
        export_year_csvs(plot_df, run, model_preds_dir)
        plot_key = f"{run.model}_{run.horizon}"
        models_to_plot[plot_key] = (model_preds_dir, model_col)

        metrics_path = output_dir / "metrics" / f"{run.dataset}_{run.model}.yaml"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            yaml.safe_dump(
                {
                    "summary": summary,
                    "per_seed": {str(r["seed"]): r for r in run_seed_rows},
                },
                f,
                sort_keys=False,
            )

        print(f"[OK] {run.dataset} | {run.model} | {format_aggregated_metrics(summary)}")
        if int(summary.get("n_seeds") or 1) > 1:
            print(f"     plot preds from seed {plot_seed} (representative, not an ensemble)")

    summary_path = output_dir / "walk_forward_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, float_format="%.4f")
    print(f"\n[DONE] Summary: {summary_path} ({len(summary_rows)} rows)")
    if per_seed_rows:
        by_seed_path = output_dir / "walk_forward_by_seed.csv"
        pd.DataFrame(per_seed_rows).to_csv(by_seed_path, index=False, float_format="%.4f")
        print(f"[DONE] Per-seed metrics: {by_seed_path} ({len(per_seed_rows)} rows)")
    print(f"[DONE] Pooled year CSVs: {preds_root}")

    manifest = {
        "baselines_dir": str(baselines_dir),
        "n_runs": len(summary_rows),
        "models": sorted({r["model"] for r in summary_rows}),
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    if args.plot:
        for plot_key, (preds_dir, model_col) in sorted(models_to_plot.items()):
            run_visualize(
                preds_dir,
                model_col,
                model_label=plot_key,
                plot_output_dir=output_dir / "plots" / plot_key,
                min_years=args.min_years,
            )

    if args.dashboard:
        html_path = write_model_comparison_dashboard(output_dir, summary_rows)
        print(f"[DONE] Multi-model dashboard: {html_path}")


if __name__ == "__main__":
    main()
