#!/usr/bin/env python3
"""Retrain walk-forward models and compute SHAP feature importance.

Designed for maize NL family representatives (RF, Transformer, TabPFN) but
general enough for any model listed in ``MODEL_MANIFEST``.

Example (cluster)::

    poetry run python cybench/runs/analysis/compute_shap_importance.py \\
        --crop maize --country NL \\
        --models random_forest,transformer_lf,tabpfn \\
        --baselines-dir /lustre/backup/SHARED/AIN/agml/output/baselines_NL_eos_v2 \\
        --output-dir /lustre/backup/SHARED/AIN/agml/output/shap_importance/maize_NL_eos \\
        --origins 2020

Quick local pilot (last origin only)::

    poetry run python cybench/runs/analysis/compute_shap_importance.py \\
        --crop maize --country NL --models random_forest \\
        --baselines-dir ../output/baselines_NL_eos_v2 \\
        --output-dir ../output/shap_importance/maize_NL_eos \\
        --origins 2020 --force-cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cybench.runs.analysis.shap_importance_lib import (
    DEFAULT_MAIZE_FAMILY_MODELS,
    MODEL_MANIFEST,
    ShapRunSpec,
    aggregate_feature_importance,
    configure_shap_job_logging,
    run_shap_case,
)

log = logging.getLogger(__name__)


def _parse_models(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_MAIZE_FAMILY_MODELS)
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [m for m in models if m not in MODEL_MANIFEST]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Supported: {sorted(MODEL_MANIFEST)}"
        )
    return models


def _parse_origins(raw: str | None, *, last_only: bool) -> tuple[int, ...] | None:
    if last_only:
        return None  # resolved later from dataset years
    if not raw:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--country", default="NL")
    parser.add_argument(
        "--models",
        help=f"Comma-separated slugs (default: {','.join(DEFAULT_MAIZE_FAMILY_MODELS)})",
    )
    parser.add_argument("--horizon", default="eos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        required=True,
        help="Batch folder with screening + walk-forward Hydra runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-model SHAP YAML/CSV outputs",
    )
    parser.add_argument(
        "--origins",
        help="Comma-separated forecast years (default: all walk-forward origins)",
    )
    parser.add_argument(
        "--last-origin-only",
        action="store_true",
        help="Only compute SHAP for the latest walk-forward test year",
    )
    parser.add_argument("--max-background", type=int, default=50)
    parser.add_argument("--max-eval-samples", type=int, default=80)
    parser.add_argument(
        "--shapiq-budget",
        type=int,
        default=64,
        help="Shapley budget per sample for TabPFN (shapiq TabPFNExplainer)",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=5,
        help="sklearn permutation_importance repeats for TabICL/TabDPT",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Override frozen CUDA configs (useful on login nodes)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_shap_job_logging(verbose=args.verbose)

    models = _parse_models(args.models)
    test_years = _parse_origins(args.origins, last_only=args.last_origin_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.last_origin_only and test_years is None:
        from cybench.datasets.data_factory import DataFactory
        from cybench.util.config_utils import reload_config_with_overrides
        from cybench.runs.analysis.shap_importance_lib import (
            CONF_DIR,
            compose_dataset_overrides,
            iter_walk_forward_origins,
        )

        probe_spec = ShapRunSpec(
            crop=args.crop,
            country=args.country,
            model=models[0],
            horizon=args.horizon,
            seed=args.seed,
            baselines_dir=args.baselines_dir,
        )
        meta = MODEL_MANIFEST[models[0]]
        overrides = compose_dataset_overrides(
            probe_spec,
            framework=str(meta["framework"]),
            feature_design=bool(meta["feature_design"]),
        )
        cfg = reload_config_with_overrides(
            CONF_DIR, "config", overrides=[f"model={models[0]}", *overrides]
        )
        years = DataFactory.peek_dataset_years(cfg.dataset)
        last_year = max(
            int(test[0])
            for _train, test in iter_walk_forward_origins(years, seed=args.seed)
        )
        test_years = (last_year,)
        log.info("Last-origin-only: using test year %s", last_year)

    summaries: list[dict] = []
    for model in models:
        spec = ShapRunSpec(
            crop=args.crop,
            country=args.country,
            model=model,
            horizon=args.horizon,
            seed=args.seed,
            baselines_dir=args.baselines_dir,
            test_years=test_years,
            max_background=args.max_background,
            max_eval_samples=args.max_eval_samples,
            shapiq_budget=args.shapiq_budget,
            permutation_repeats=args.permutation_repeats,
            force_cpu=args.force_cpu,
        )
        model_out = args.output_dir / f"{args.crop}_{args.country}" / model
        model_out.mkdir(parents=True, exist_ok=True)
        summary = run_shap_case(spec, output_dir=model_out)
        summaries.append(summary)
        repro = summary["origins"][0]["reproduction"] if summary["origins"] else {}
        log.info(
            "[%s] origins=%d | reproduction corr=%s max_diff=%s",
            model,
            summary["n_origins"],
            repro.get("corr_saved_preds"),
            repro.get("max_abs_pred_diff"),
        )

    all_records: list[dict] = []
    for summary in summaries:
        all_records.extend(summary["origins"])
    agg = aggregate_feature_importance(all_records)
    if not agg.empty:
        agg_path = args.output_dir / f"{args.crop}_{args.country}" / "shap_aggregate_all_models.csv"
        agg.to_csv(agg_path, index=False)
        log.info("Wrote aggregate table to %s", agg_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
