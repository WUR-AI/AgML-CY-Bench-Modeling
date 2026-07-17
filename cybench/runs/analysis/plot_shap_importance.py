#!/usr/bin/env python3
"""Plot paper-ready SHAP importance summaries from ``compute_shap_importance`` outputs.

Discovers ``shap_summary.yaml`` files under an input root (single or multi-country),
aggregates |SHAP| within meta-groups and EOS-anchored windows, and writes:

- ``shap_family_meta_groups.png`` — bar chart by model family (RF / Transformer by default)
- ``shap_timing_heatmaps.png`` — variable × window heatmaps (tabular models);
  x-axis is chronological windows before EOS (``0, -1, -2, …``)
- CSV tables for supplementary material

Example (single country on cluster)::

    poetry run python cybench/runs/analysis/plot_shap_importance.py \\
        --input-dir /lustre/backup/SHARED/AIN/agml/output/shap_importance/maize_NL_eos \\
        --output-dir figures/shap/maize_NL_eos

Multi-country (scan all case-study folders under a root)::

    poetry run python cybench/runs/analysis/plot_shap_importance.py \\
        --input-dir /lustre/backup/SHARED/AIN/agml/output/shap_importance \\
        --output-dir figures/shap/maize_eos \\
        --crop maize --horizon eos
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cybench.runs.analysis.shap_importance_lib import DEFAULT_MAIZE_FAMILY_MODELS
from cybench.runs.analysis.shap_plot_lib import (
    discover_shap_summaries,
    export_tables,
    load_feature_table,
    meta_group_consistency,
    meta_group_shares,
    plot_meta_group_families,
    plot_timing_heatmaps,
    timing_table,
)

log = logging.getLogger(__name__)


def _parse_models(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder containing SHAP outputs (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for figures and CSV tables",
    )
    parser.add_argument("--crop", help="Filter to one crop (inferred if omitted)")
    parser.add_argument("--horizon", default="eos", help="Prediction horizon tag")
    parser.add_argument(
        "--models",
        help=f"Comma-separated model slugs (default: {','.join(DEFAULT_MAIZE_FAMILY_MODELS)})",
    )
    parser.add_argument("--top-groups", type=int, default=8, help="Meta-groups per panel")
    parser.add_argument(
        "--top-variables",
        type=int,
        default=12,
        help="Variable groups in timing heatmaps",
    )
    parser.add_argument(
        "--top-k-consistency",
        type=int,
        default=10,
        help="Top-k threshold for cross-country consistency table",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip timing heatmaps when only torch summaries are available",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    summary_paths = discover_shap_summaries(args.input_dir)
    if not summary_paths:
        log.error("No shap_summary.yaml files under %s", args.input_dir)
        return 1
    log.info("Found %d shap_summary.yaml file(s)", len(summary_paths))

    feature_table = load_feature_table(summary_paths)
    if feature_table.empty:
        log.error("No feature rows loaded from summaries")
        return 1

    crop = args.crop or str(feature_table["crop"].mode().iloc[0])
    horizon = args.horizon
    models = _parse_models(args.models) or list(DEFAULT_MAIZE_FAMILY_MODELS)
    feature_table = feature_table[
        (feature_table["crop"] == crop) & (feature_table["horizon"] == horizon)
    ].copy()
    if feature_table.empty:
        log.error("No rows after filtering crop=%s horizon=%s", crop, horizon)
        return 1

    countries = sorted(feature_table["country"].unique())
    log.info(
        "Plotting crop=%s horizon=%s | countries=%s | models=%s",
        crop,
        horizon,
        ",".join(countries),
        ",".join(models),
    )

    shares = meta_group_shares(feature_table)
    timing = timing_table(feature_table)
    consistency = meta_group_consistency(
        feature_table, top_k=args.top_k_consistency
    )
    export_tables(
        feature_table=feature_table,
        shares=shares,
        timing=timing,
        consistency=consistency,
        output_dir=args.output_dir,
    )

    plot_meta_group_families(
        shares,
        crop=crop,
        horizon=horizon,
        models=models,
        top_n=args.top_groups,
        output_path=args.output_dir / "shap_family_meta_groups.png",
    )
    log.info("Wrote %s", args.output_dir / "shap_family_meta_groups.png")

    if not args.skip_timing and not timing.empty:
        try:
            plot_timing_heatmaps(
                timing,
                crop=crop,
                horizon=horizon,
                models=[m for m in models if m != "transformer_lf"],
                top_variables=args.top_variables,
                output_path=args.output_dir / "shap_timing_heatmaps.png",
            )
            log.info("Wrote %s", args.output_dir / "shap_timing_heatmaps.png")
        except ValueError as exc:
            log.warning("Skipping timing heatmaps: %s", exc)
    else:
        log.info("Timing heatmaps skipped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
