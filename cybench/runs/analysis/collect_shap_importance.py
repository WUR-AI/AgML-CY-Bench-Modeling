#!/usr/bin/env python3
"""Collect per-origin SHAP artifacts into model summaries and cross-model tables.

Use after parallel SHAP jobs (one origin per task) that wrote only
``origin_<year>/shap_importance.yaml`` files.

Example::

    poetry run python cybench/runs/analysis/collect_shap_importance.py \\
        --output-dir /lustre/backup/SHARED/AIN/agml/output/shap_importance/maize_NL_eos \\
        --crop maize --country NL \\
        --baselines-dir /lustre/backup/SHARED/AIN/agml/output/baselines_NL_eos_v4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cybench.runs.analysis.shap_importance_lib import (
    MODEL_MANIFEST,
    collect_shap_output_dir,
    configure_shap_job_logging,
)

log = logging.getLogger(__name__)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        help="Optional: fill frozen/walk-forward paths in shap_summary.yaml",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model slugs (default: all model dirs with origins)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_shap_job_logging(verbose=args.verbose)
    models = _parse_models(args.models)

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
