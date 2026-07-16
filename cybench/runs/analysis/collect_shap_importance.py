#!/usr/bin/env python3
"""Collect per-origin SHAP artifacts into model summaries and cross-model tables.

Use after parallel SHAP jobs (one origin per task) that wrote only
``origin_<year>/shap_importance.yaml`` files.

Single crop×country::

    poetry run python cybench/runs/analysis/collect_shap_importance.py \\
        --output-dir /lustre/backup/SHARED/AIN/agml/output/shap_importance/maize_NL_eos \\
        --crop maize --country NL \\
        --baselines-dir /lustre/backup/SHARED/AIN/agml/output/baselines_NL_eos_v4

Auto-discover all cases under the SHAP root (recommended after array jobs)::

    poetry run python cybench/runs/analysis/collect_shap_importance.py \\
        --shap-root /lustre/backup/SHARED/AIN/agml/output/shap_importance \\
        --output-root /lustre/backup/SHARED/AIN/agml/output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cybench.runs.analysis.shap_importance_lib import (
    MODEL_MANIFEST,
    ShapCaseSummary,
    ShapCollectCase,
    collect_shap_output_dir,
    configure_shap_job_logging,
    discover_shap_collect_cases,
)
from cybench.runs.slurm.benchmark_submit_lib import (
    batch_name,
    batch_suffix_to_horizon,
    resolve_batch_dir,
)

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("/lustre/backup/SHARED/AIN/agml/output")


def _parse_models(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [m for m in models if m not in MODEL_MANIFEST]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Supported: {sorted(MODEL_MANIFEST)}"
        )
    return models


def _resolve_baselines_dir(
    case: ShapCollectCase,
    *,
    output_root: Path,
    version: int,
) -> Path | None:
    horizon = batch_suffix_to_horizon(case.horizon_tag)
    batch = batch_name(case.country, horizon, version)
    baselines_dir, _note = resolve_batch_dir(output_root, batch)
    if baselines_dir.is_dir():
        return baselines_dir
    log.warning(
        "Skipping baselines metadata for %s/%s: missing %s",
        case.crop,
        case.country,
        baselines_dir,
    )
    return None


def _collect_case(
    case: ShapCollectCase,
    *,
    baselines_dir: Path | None,
    models: list[str] | None,
) -> list[ShapCaseSummary]:
    return collect_shap_output_dir(
        case.output_dir,
        crop=case.crop,
        country=case.country,
        baselines_dir=baselines_dir,
        models=models,
    )


def _collect_discovered(
    cases: list[ShapCollectCase],
    *,
    output_root: Path,
    baselines_version: int,
    models: list[str] | None,
    dry_run: bool,
) -> list[ShapCaseSummary]:
    all_summaries: list[ShapCaseSummary] = []
    for case in cases:
        baselines_dir = _resolve_baselines_dir(
            case, output_root=output_root, version=baselines_version
        )
        log.info(
            "Collect %s/%s | horizon=%s | models=%s | origins≤%d | %s",
            case.crop,
            case.country,
            case.horizon_tag,
            ",".join(case.models),
            case.n_origins,
            case.output_dir,
        )
        if dry_run:
            continue
        summaries = _collect_case(case, baselines_dir=baselines_dir, models=models)
        if not summaries:
            log.warning("No summaries written for %s/%s", case.crop, case.country)
            continue
        for summary in summaries:
            log.info(
                "  %s | origins=%d | years=%s",
                summary["model"],
                summary["n_origins"],
                ",".join(str(origin["test_years"][0]) for origin in summary["origins"]),
            )
        all_summaries.extend(summaries)
        agg_path = (
            case.output_dir
            / f"{case.crop}_{case.country}"
            / "shap_aggregate_all_models.csv"
        )
        if agg_path.is_file():
            log.info("  wrote %s", agg_path)
    return all_summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shap-root",
        type=Path,
        help="Auto-discover cases under this folder (e.g. .../output/shap_importance)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Baselines parent dir for auto baselines_* resolution (with --shap-root)",
    )
    parser.add_argument(
        "--baselines-version",
        type=int,
        default=4,
        help="Baselines batch version when resolving baselines_* dirs automatically",
    )
    parser.add_argument("--output-dir", type=Path, help="Single case SHAP folder")
    parser.add_argument("--crop", help="Crop slug (single-case mode)")
    parser.add_argument("--country", help="Country ISO2 (single-case mode)")
    parser.add_argument(
        "--horizon",
        default="eos",
        help="Horizon folder tag when filtering discovery (default: eos)",
    )
    parser.add_argument(
        "--crops",
        nargs="*",
        help="Limit discovery to these crops",
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        help="Limit discovery to these countries",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        help="Optional: fill frozen/walk-forward paths in shap_summary.yaml",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model slugs (default: all model dirs with origins)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered cases without writing summaries",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_shap_job_logging(verbose=args.verbose)
    models = _parse_models(args.models)

    if args.shap_root is not None:
        cases = discover_shap_collect_cases(
            args.shap_root,
            crops=args.crops,
            countries=args.countries,
            horizon_tag=args.horizon,
        )
        if not cases:
            log.error("No SHAP cases with origin artifacts under %s", args.shap_root)
            return 1
        log.info("Discovered %d case(s) under %s", len(cases), args.shap_root)
        summaries = _collect_discovered(
            cases,
            output_root=args.output_root,
            baselines_version=args.baselines_version,
            models=models,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            log.info("Dry-run only; no files written")
            return 0
        if not summaries:
            return 1
        return 0

    if not args.output_dir or not args.crop or not args.country:
        parser.error(
            "Provide --shap-root for auto-discovery, or --output-dir with --crop and --country"
        )

    if args.dry_run:
        log.info(
            "Dry-run single case %s/%s under %s",
            args.crop,
            args.country,
            args.output_dir,
        )
        return 0

    summaries = collect_shap_output_dir(
        args.output_dir,
        crop=args.crop,
        country=args.country,
        baselines_dir=args.baselines_dir,
        models=models,
    )
    if not summaries:
        log.error("No origin artifacts found under %s", args.output_dir)
        return 1

    for summary in summaries:
        log.info(
            "Collected %s | origins=%d | years=%s",
            summary["model"],
            summary["n_origins"],
            ",".join(str(origin["test_years"][0]) for origin in summary["origins"]),
        )

    agg_path = args.output_dir / f"{args.crop}_{args.country}" / "shap_aggregate_all_models.csv"
    if agg_path.is_file():
        log.info("Wrote %s", agg_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
