#!/usr/bin/env python3
"""Verify local retraining reproduces cluster walk-forward predictions.

Example (cluster, maize NL Transformer, eos v3)::

    poetry run python cybench/runs/analysis/verify_walk_forward_reproduction.py \\
        --model transformer_lf \\
        --baselines-dir /lustre/backup/SHARED/AIN/agml/output/baselines_NL_eos_v3 \\
        --origins 2020
"""

from __future__ import annotations

import os

# BLAS thread pools must be configured before NumPy/PyTorch import for CPU determinism.
_thread_count = os.environ.get("CYBENCH_TORCH_THREADS", "1")
for _blas_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_var, _thread_count)

import argparse
import json
import logging
import sys
from pathlib import Path

from cybench.datasets.data_factory import DataFactory
from cybench.runs.analysis.shap_importance_lib import (
    CONF_DIR,
    MODEL_MANIFEST,
    ShapRunSpec,
    compose_dataset_overrides,
    find_screening_split_dir,
    find_walk_forward_run_dir,
    iter_walk_forward_origins,
    reproduce_walk_forward_origin,
)
from cybench.util.config_utils import reload_config_with_overrides

log = logging.getLogger(__name__)

REPRO_TOLERANCE_MAX_ABS_DIFF = 1e-2
REPRO_TOLERANCE_MIN_CORR = 0.999


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--country", default="NL")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_MANIFEST))
    parser.add_argument("--horizon", default="eos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        help="Auto-resolve screening + walk-forward dirs (alternative to explicit paths)",
    )
    parser.add_argument("--screening-split-dir", type=Path)
    parser.add_argument("--walk-forward-run-dir", type=Path)
    parser.add_argument("--origins", required=True, help="Comma-separated test years")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    test_years = tuple(int(x.strip()) for x in args.origins.split(",") if x.strip())

    screening_dir = args.screening_split_dir
    walk_forward_dir = args.walk_forward_run_dir
    if args.baselines_dir is not None:
        screening_dir = find_screening_split_dir(
            args.baselines_dir,
            crop=args.crop,
            country=args.country,
            model_slug=args.model,
            horizon=args.horizon,
        )
        walk_forward_dir = find_walk_forward_run_dir(
            args.baselines_dir,
            crop=args.crop,
            country=args.country,
            model_slug=args.model,
            horizon=args.horizon,
        )
    if screening_dir is None or walk_forward_dir is None:
        parser.error(
            "Provide --baselines-dir or both --screening-split-dir and "
            "--walk-forward-run-dir"
        )
    log.info("screening_split_dir=%s", screening_dir)
    log.info("walk_forward_run_dir=%s", walk_forward_dir)

    spec = ShapRunSpec(
        crop=args.crop,
        country=args.country,
        model=args.model,
        horizon=args.horizon,
        seed=args.seed,
        force_cpu=args.force_cpu,
    )
    meta = MODEL_MANIFEST[args.model]
    overrides = compose_dataset_overrides(
        spec,
        framework=str(meta["framework"]),
        feature_design=bool(meta["feature_design"]),
    )
    cfg = reload_config_with_overrides(
        CONF_DIR,
        "config",
        overrides=[f"model={args.model}", *overrides],
    )
    dataset_years = DataFactory.peek_dataset_years(cfg.dataset)

    results: list[dict] = []
    ok = True
    for train_years, origin_test_years in iter_walk_forward_origins(
        dataset_years, seed=args.seed, only_years=test_years
    ):
        record = reproduce_walk_forward_origin(
            spec,
            train_years=train_years,
            test_years=origin_test_years,
            frozen_dir=screening_dir,
            walk_forward_run_dir=walk_forward_dir,
        )
        repro = record["reproduction"]
        max_diff = repro.get("max_abs_pred_diff")
        corr = repro.get("corr_saved_preds")
        origin = int(origin_test_years[0])
        passed = (
            max_diff is not None
            and corr is not None
            and float(max_diff) <= REPRO_TOLERANCE_MAX_ABS_DIFF
            and float(corr) >= REPRO_TOLERANCE_MIN_CORR
        )
        ok = ok and passed
        log.info(
            "origin=%s | n_train=%s n_test=%s | corr=%s max_abs_diff=%s | %s",
            origin,
            record["n_train"],
            record["n_test"],
            corr,
            max_diff,
            "PASS" if passed else "FAIL",
        )
        results.append(record)

    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
