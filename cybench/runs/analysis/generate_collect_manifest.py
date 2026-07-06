#!/usr/bin/env python3
"""Write a SLURM manifest for parallel walk-forward result collection.

Each line: ``COUNTRY HORIZON VERSION PLOT`` where PLOT is ``yes`` or ``no``.

Uses a fast lustre scan (directory names only). Full readiness checks run inside
each SLURM collect task, not here.

Example::

    poetry run python cybench/runs/analysis/generate_collect_manifest.py \\
        --mode all-available --horizon eos --version 2 \\
        -o cybench/runs/slurm/collect_jobs.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cybench.runs.analysis.publish_pipeline_lib import (
    PipelineDefaults,
    assess_readiness,
    discover_baselines_batches_fast,
    filter_publish_targets,
    filter_ready_targets,
    load_pipeline_defaults,
)
from cybench.runs.slurm.benchmark_submit_lib import resolve_batch_dir

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "dashboard_targets.yaml"


def _resolve_targets_fast(
    *,
    mode: str,
    config_path: Path | None,
    defaults: PipelineDefaults,
    countries: list[str] | None,
    horizons: list[str] | None,
    version: int | None,
) -> list:
    if config_path and config_path.is_file():
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        defaults = load_pipeline_defaults(config_path, overrides=defaults)
        entries = raw.get("include") or raw.get("targets") or []
        if entries and mode == "planned":
            from cybench.runs.analysis.publish_pipeline_lib import _explicit_targets_from_config

            targets = _explicit_targets_from_config(raw, defaults=defaults)
        else:
            targets = discover_baselines_batches_fast(defaults.output_root, defaults=defaults)
    else:
        targets = discover_baselines_batches_fast(defaults.output_root, defaults=defaults)

    return filter_publish_targets(
        targets, countries=countries, horizons=horizons, version=version
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="dashboard_targets.yaml (paths + min_run_fraction)",
    )
    parser.add_argument(
        "--mode",
        choices=["planned", "ready", "all-available"],
        default="all-available",
        help="Target discovery mode (default: all-available)",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        metavar="CC",
        help="Limit to country code(s)",
    )
    parser.add_argument(
        "--horizon",
        action="append",
        dest="horizons",
        help="Limit to horizon(s): eos, early, mid, qtr, early-season, quarter-of-season",
    )
    parser.add_argument(
        "--version",
        type=int,
        metavar="N",
        help="Limit to batch version suffix (e.g. 3 for baselines_DE_eos_v3)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Request plotting in each collect task (slow)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Metrics + preds only (default unless --plot)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cybench/runs/slurm/collect_jobs.txt"),
        help="Manifest path (default: cybench/runs/slurm/collect_jobs.txt)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print manifest lines to stdout instead of writing a file",
    )
    args = parser.parse_args()

    plot = "yes" if args.plot and not args.no_plot else "no"
    config_path = args.config if args.config.is_file() else None
    defaults = load_pipeline_defaults(config_path)
    print(
        "[INFO] generate_collect_manifest: fast scan (no CSV/Hydra reads unless --mode ready)",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[INFO] Scanning batch folders on lustre...",
        file=sys.stderr,
        flush=True,
    )
    targets = _resolve_targets_fast(
        mode=args.mode,
        config_path=config_path,
        defaults=defaults,
        countries=args.countries,
        horizons=args.horizons,
        version=args.version,
    )
    print(
        f"[INFO] Found {len(targets)} batch folder(s) under {defaults.output_root}",
        file=sys.stderr,
        flush=True,
    )
    if args.mode == "ready":
        print(
            "[INFO] --mode ready: checking walk-forward completeness (may take a few minutes)...",
            file=sys.stderr,
            flush=True,
        )
        targets = [t for t, _ in filter_ready_targets(targets)]

    lines: list[str] = []
    for target in targets:
        if args.mode == "ready":
            report = assess_readiness(target)
            if not report.ready or report.complete_runs <= 0:
                continue
        else:
            baselines_dir, _ = resolve_batch_dir(target.output_root, target.batch_name)
            if not baselines_dir.is_dir():
                continue
        lines.append(f"{target.country_upper} {target.batch_horizon} {target.version} {plot}")

    if args.list:
        for line in lines:
            print(line)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = "# COUNTRY HORIZON VERSION PLOT (yes|no)\n"
    args.output.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[DONE] Wrote {len(lines)} collect job(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
