#!/usr/bin/env python3
"""Orchestrate collect → publish → index → git for CY-Bench walk-forward dashboards.

Designed to run entirely on the WUR lustre HPC (anunna): baselines, collect output,
and the GitHub Pages clone can all live on lustre or $HOME.

Example (on anunna, from the repo root)::

    poetry run python cybench/runs/analysis/orchestrate_dashboard_publish.py \\
        --config cybench/runs/analysis/dashboard_targets.yaml \\
        --mode ready --dry-run

    poetry run python cybench/runs/analysis/orchestrate_dashboard_publish.py \\
        --mode ready --commit

    poetry run python cybench/runs/analysis/orchestrate_dashboard_publish.py \\
        --country DE --horizon eos --force publish,index,commit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cybench.runs.analysis.publish_pipeline_lib import (
    PipelineDefaults,
    StageName,
    assess_readiness,
    filter_ready_targets,
    load_pipeline_defaults,
    resolve_targets,
    run_collect_stage,
    run_commit_stage,
    run_index_stage,
    run_publish_stage,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "dashboard_targets.yaml"


def _parse_stages(raw: str | None) -> set[StageName]:
    if not raw:
        return {"collect", "publish", "index"}
    stages: set[StageName] = set()
    for part in raw.split(","):
        key = part.strip().lower()
        if key not in {"collect", "publish", "index", "commit"}:
            raise ValueError(f"Unknown stage: {part!r}")
        stages.add(key)  # type: ignore[arg-type]
    return stages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"YAML target list (default: {_DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--mode",
        choices=["planned", "ready", "all-available"],
        default="ready",
        help=(
            "ready: lustre baselines_* batches with complete walk-forward (default); "
            "all-available: all discovered batches; "
            "planned: only explicit include: list in config"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override defaults.output_root (Hydra ../output on lustre)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Override defaults.repo_root (cybench clone used for poetry collect)",
    )
    parser.add_argument(
        "--publish-root",
        type=Path,
        help="Override defaults.publish_root (git clone for GitHub Pages)",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        metavar="CC",
        help="Limit to country code(s), e.g. DE or FR (repeatable)",
    )
    parser.add_argument(
        "--horizon",
        action="append",
        dest="horizons",
        help="Limit to horizon(s): eos, mid, middle-of-season, ...",
    )
    parser.add_argument(
        "--stages",
        help="Comma-separated stages (default: collect,publish,index). Add commit via --commit",
    )
    parser.add_argument(
        "--force",
        help="Comma-separated stages to force even if up to date",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip per-model plots during collect (faster; dashboard panels may be sparse)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Git commit publish-root changes for each processed target",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Git push after commit (implies network access on the login/submit node)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without running collect/publish/commit",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List candidate targets and readiness, then exit",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_file() else None
    defaults = PipelineDefaults(
        output_root=args.output_root or PipelineDefaults().output_root,
        repo_root=args.repo_root or PipelineDefaults().repo_root,
        publish_root=(args.publish_root or PipelineDefaults().publish_root).expanduser(),
    )
    defaults = load_pipeline_defaults(config_path, overrides=defaults)
    try:
        targets = resolve_targets(
            mode="all-available" if args.list else args.mode,
            config_path=config_path,
            defaults=defaults,
            countries=args.countries,
            horizons=args.horizons,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not targets:
        print("[WARN] No targets matched filters")
        return 1

    readiness = {t.batch_name: assess_readiness(t) for t in targets}

    if args.list:
        print(f"{'batch':<28} {'ready':<5} complete  reason")
        print("-" * 72)
        for target in targets:
            report = readiness[target.batch_name]
            mark = "yes" if report.ready else "no"
            exp = report.expected_runs or "?"
            print(
                f"{target.batch_name:<28} {mark:<5} {report.complete_runs:>3}/{exp:<3}  {report.reason}"
            )
        return 0

    if args.mode == "ready":
        skipped = [t for t in targets if not readiness[t.batch_name].ready]
        for target in skipped:
            print(f"[SKIP] {target.batch_name}: {readiness[target.batch_name].reason}")
        targets = [t for t, _ in filter_ready_targets(targets)]

    if not targets:
        print("[DONE] Nothing ready to publish")
        return 0

    stages = _parse_stages(args.stages)
    if args.commit:
        stages.add("commit")
    force = _parse_stages(args.force)
    exit_code = 0

    for target in targets:
        report = readiness.get(target.batch_name) or assess_readiness(target)
        print(f"\n=== {target.batch_name} | {report.complete_runs} runs | slug={target.publish_slug} ===")
        try:
            if "collect" in stages:
                status = run_collect_stage(
                    target,
                    plot=not args.no_plot,
                    force="collect" in force,
                    dry_run=args.dry_run,
                )
                print(f"[{'SKIP' if status.skipped else 'OK'}] collect: {status.message}")

            if "publish" in stages:
                status = run_publish_stage(
                    target,
                    force="publish" in force,
                    dry_run=args.dry_run,
                )
                print(f"[{'SKIP' if status.skipped else 'OK'}] publish: {status.message}")

            if "commit" in stages:
                status = run_commit_stage(
                    target,
                    dry_run=args.dry_run,
                    push=args.push,
                )
                print(f"[{'SKIP' if status.skipped else 'OK'}] commit: {status.message}")
        except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"[ERROR] {target.batch_name}: {exc}", file=sys.stderr)
            exit_code = 1

    if "index" in stages:
        if args.dry_run:
            print(f"\n[DRY-RUN] would rebuild index under {targets[0].publish_root}")
        else:
            status = run_index_stage(targets[0], dry_run=False)
            print(f"\n[OK] index: {status.message}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
