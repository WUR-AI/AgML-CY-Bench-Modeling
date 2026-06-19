#!/usr/bin/env python3
"""Write a SLURM manifest for parallel walk-forward result collection.

Each line: ``COUNTRY HORIZON PLOT`` where PLOT is ``yes`` or ``no``.

Example::

    poetry run python cybench/runs/analysis/generate_collect_manifest.py \\
        --mode all-available --horizon eos --horizon mid \\
        -o cybench/runs/slurm/collect_jobs.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cybench.runs.analysis.publish_pipeline_lib import (
    PipelineDefaults,
    assess_readiness,
    filter_ready_targets,
    load_pipeline_defaults,
    resolve_targets,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "dashboard_targets.yaml"


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
        help="Limit to horizon(s): eos, mid",
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
    targets = resolve_targets(
        mode=args.mode,
        config_path=config_path,
        defaults=defaults,
        countries=args.countries,
        horizons=args.horizons,
    )
    if args.mode == "ready":
        targets = [t for t, _ in filter_ready_targets(targets)]

    lines: list[str] = []
    for target in targets:
        report = assess_readiness(target)
        if args.mode == "ready" and not report.ready:
            continue
        if report.complete_runs <= 0:
            continue
        lines.append(f"{target.country_upper} {target.batch_horizon} {plot}")

    if args.list:
        for line in lines:
            print(line)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = "# COUNTRY HORIZON PLOT (yes|no)\n"
    args.output.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[DONE] Wrote {len(lines)} collect job(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
