#!/usr/bin/env python3
"""Evaluate one walk-forward origin checkpoint across multiple forecast years.

Walk-forward trains a new model per test year. This script loads a single
checkpoint (default: first origin year) and predicts every evaluation year with
it, then compares report metrics to the pooled walk-forward predictions.

Example::

    poetry run python cybench/runs/analysis/single_origin_multi_year_eval.py \\
        --baselines-dir /home/michiel/WUR/output/baselines_DE_mid_v4 \\
        --crop maize --country DE --model lstm_lf --horizon mid \\
        --origin-year 2017 --seed 42 --force-cpu
"""

from __future__ import annotations

import os

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.torch_dataset import TorchDataset
from cybench.evaluation.aggregated_metrics import compute_report_metrics
from cybench.runs.analysis.collect_walk_forward_results import load_pooled_predictions
from cybench.runs.analysis.shap_importance_lib import (
    ShapRunSpec,
    build_dataset,
    find_saved_model_artifact,
    find_screening_split_dir,
    find_walk_forward_run_dir,
    load_saved_walk_forward_model,
    _prepare_model_cfg,
)
from cybench.util.config_utils import set_seed

log = logging.getLogger(__name__)


def _normalize_horizon(horizon: str) -> str:
    """Map short CLI tags to dataset ``end_of_sequence`` values."""
    mapping = {
        "mid": "middle-of-season",
        "mid-season": "middle-of-season",
        "mid_season": "middle-of-season",
        "middle-of-season": "middle-of-season",
        "eos": "eos",
        "end-of-season": "eos",
        "qtr": "quarter-of-season",
        "quarter-season": "quarter-of-season",
        "quarter_season": "quarter-of-season",
        "quarter-of-season": "quarter-of-season",
        "early": "early-season",
        "early-season": "early-season",
        "early_season": "early-season",
    }
    if horizon not in mapping:
        raise ValueError(f"Unknown horizon {horizon!r}; expected one of {sorted(mapping)}")
    return mapping[horizon]


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_year.r": metrics["region_year"]["r"],
        "region_year.r2": metrics["region_year"]["r2"],
        "region_year.nrmse": metrics["region_year"]["nrmse"],
        "spatial.r_typical_year": metrics["spatial"]["r_typical_year"],
        "temporal.r_typical_region": metrics["temporal"]["r_typical_region"],
        "temporal.r2_typical_region": metrics["temporal"]["r2_typical_region"],
        "temporal.n_slices_regions": metrics["temporal"]["n_slices_regions"],
        "anomaly.r_pooled": metrics["anomaly"]["r_pooled"],
        "anomaly.r_typical_region": metrics["anomaly"]["r_typical_region"],
        "n_samples": metrics["n_samples"],
        "n_regions": metrics["n_regions"],
        "n_years": metrics["n_years"],
    }


def _preds_dataframe(
    dataset: TorchDataset,
    preds: np.ndarray,
    *,
    model_col: str,
) -> pd.DataFrame:
    indices = dataset.indices.reset_index(drop=True)
    return pd.DataFrame(
        {
            KEY_LOC: indices[KEY_LOC].astype(str).values,
            KEY_YEAR: indices[KEY_YEAR].astype(int).values,
            KEY_TARGET: np.asarray(dataset.targets, dtype=float),
            model_col: np.asarray(preds, dtype=float),
        }
    )


def evaluate_single_origin(
    *,
    baselines_dir: Path,
    crop: str,
    country: str,
    model: str,
    horizon: str,
    origin_year: int,
    seed: int,
    predict_years: list[int] | None,
    force_cpu: bool,
) -> dict[str, Any]:
    eos = _normalize_horizon(horizon)
    screening_dir = find_screening_split_dir(
        baselines_dir,
        crop=crop,
        country=country,
        model_slug=model,
        horizon=eos,
    )
    walk_forward_dir = find_walk_forward_run_dir(
        baselines_dir,
        crop=crop,
        country=country,
        model_slug=model,
        horizon=eos,
    )
    artifact = find_saved_model_artifact(
        walk_forward_dir,
        test_year=origin_year,
        seed=seed,
        model_name=model,
    )
    if artifact is None:
        raise FileNotFoundError(
            f"No checkpoint at {walk_forward_dir}/{origin_year}/{seed}/{model}.pt"
        )

    spec = ShapRunSpec(
        crop=crop,
        country=country,
        model=model,
        horizon=eos,
        seed=seed,
        force_cpu=force_cpu,
    )
    set_seed(seed)
    dataset = build_dataset(spec, framework="torch", feature_design=False)
    assert isinstance(dataset, TorchDataset)

    years = sorted(int(y) for y in dataset.indices[KEY_YEAR].unique())
    if predict_years is None:
        # Match the walk-forward evaluation panel (origins present in the run dir).
        predict_years = sorted(
            int(p.name)
            for p in walk_forward_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
    missing = [y for y in predict_years if y not in years]
    if missing:
        raise ValueError(f"Predict years not in dataset: {missing}; available={years}")

    model_cfg, _fs_cfg, _E_star = _prepare_model_cfg(
        spec,
        frozen_dir=screening_dir,
        dataset=dataset,
        framework="torch",
    )
    # Train years for the origin are only needed to size a shell dataloader; use
    # years strictly before the origin (same as walk-forward).
    train_years = [y for y in years if y < origin_year]
    if not train_years:
        raise ValueError(f"No train years before origin {origin_year}")
    train_dataset, _ = dataset.split_on_years((train_years, [origin_year]))
    log.info("Loading checkpoint %s", artifact)
    loaded = load_saved_walk_forward_model(
        model_cfg, artifact, torch_dataset=train_dataset
    )

    model_col = model
    frames: list[pd.DataFrame] = []
    per_year: dict[str, Any] = {}
    for year in predict_years:
        _, year_ds = dataset.split_on_years((train_years, [year]))
        preds, _ = loaded.predict(year_ds)
        year_df = _preds_dataframe(year_ds, np.asarray(preds, dtype=float), model_col=model_col)
        frames.append(year_df)
        year_metrics = compute_report_metrics(
            year_df, target_col=KEY_TARGET, model_col=model_col
        )
        per_year[str(year)] = _flatten_metrics(year_metrics)
        log.info(
            "year=%s n=%s region_year.r=%.4f",
            year,
            len(year_df),
            year_metrics["region_year"]["r"],
        )

    single_df = pd.concat(frames, ignore_index=True)
    single_metrics = compute_report_metrics(
        single_df, target_col=KEY_TARGET, model_col=model_col
    )

    # Sanity: origin-year predictions vs saved walk-forward test_preds.
    saved_path = walk_forward_dir / str(origin_year) / str(seed) / "test_preds.csv"
    origin_sanity: dict[str, Any] | None = None
    if saved_path.is_file():
        saved = pd.read_csv(saved_path)
        origin_df = single_df[single_df[KEY_YEAR] == origin_year].copy()
        merged = origin_df.merge(
            saved.rename(columns={"preds": "saved_preds", "targets": "saved_targets"}),
            on=[KEY_LOC, KEY_YEAR],
            how="inner",
        )
        if len(merged):
            pred_diff = np.abs(merged[model_col].values - merged["saved_preds"].values)
            origin_sanity = {
                "n_matched": int(len(merged)),
                "max_abs_pred_diff": float(np.max(pred_diff)),
                "mean_abs_pred_diff": float(np.mean(pred_diff)),
                "corr_vs_saved": float(np.corrcoef(merged[model_col], merged["saved_preds"])[0, 1]),
            }

    wf_df, wf_col = load_pooled_predictions(walk_forward_dir, model_slug=model, seed=seed)
    wf_df = wf_df[wf_df[KEY_YEAR].isin(predict_years)].copy()
    wf_metrics = compute_report_metrics(wf_df, target_col=KEY_TARGET, model_col=wf_col)

    return {
        "crop": crop,
        "country": country,
        "model": model,
        "horizon": horizon,
        "origin_year": origin_year,
        "seed": seed,
        "predict_years": predict_years,
        "checkpoint": str(artifact),
        "screening_split_dir": str(screening_dir),
        "walk_forward_run_dir": str(walk_forward_dir),
        "origin_sanity_vs_saved": origin_sanity,
        "single_origin_model": _flatten_metrics(single_metrics),
        "walk_forward_pooled": _flatten_metrics(wf_metrics),
        "per_year_single_origin": per_year,
        "n_train_at_origin": len(train_dataset),
    }


def _print_comparison(result: dict[str, Any]) -> None:
    single = result["single_origin_model"]
    wf = result["walk_forward_pooled"]
    keys = [
        "temporal.r_typical_region",
        "temporal.r2_typical_region",
        "anomaly.r_pooled",
        "anomaly.r_typical_region",
        "region_year.r",
        "region_year.r2",
        "region_year.nrmse",
        "spatial.r_typical_year",
        "n_samples",
        "n_regions",
        "n_years",
    ]
    print()
    print(
        f"maize_{result['country']} {result['model']} | origin={result['origin_year']} "
        f"seed={result['seed']} | years={result['predict_years']}"
    )
    print(f"checkpoint: {result['checkpoint']}")
    if result.get("origin_sanity_vs_saved"):
        s = result["origin_sanity_vs_saved"]
        print(
            f"origin sanity vs saved {result['origin_year']} preds: "
            f"max|Δ|={s['max_abs_pred_diff']:.6g} corr={s['corr_vs_saved']:.6f}"
        )
    print()
    print(f"{'metric':<32} {'single_origin':>14} {'walk_forward':>14} {'delta':>12}")
    print("-" * 74)
    for key in keys:
        a = single[key]
        b = wf[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if isinstance(a, float) or isinstance(b, float):
                delta = float(a) - float(b)
                print(f"{key:<32} {float(a):14.4f} {float(b):14.4f} {delta:12.4f}")
            else:
                print(f"{key:<32} {a:14d} {b:14d} {a - b:12d}")
        else:
            print(f"{key:<32} {a!s:>14} {b!s:>14}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-dir", type=Path, required=True)
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--country", default="DE")
    parser.add_argument("--model", default="lstm_lf")
    parser.add_argument("--horizon", default="mid")
    parser.add_argument("--origin-year", type=int, default=2017)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predict-years",
        default=None,
        help="Comma-separated years (default: all walk-forward origin years)",
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    predict_years = None
    if args.predict_years:
        predict_years = [int(x.strip()) for x in args.predict_years.split(",") if x.strip()]

    result = evaluate_single_origin(
        baselines_dir=args.baselines_dir.resolve(),
        crop=args.crop,
        country=args.country,
        model=args.model,
        horizon=args.horizon,
        origin_year=args.origin_year,
        seed=args.seed,
        predict_years=predict_years,
        force_cpu=args.force_cpu,
    )
    _print_comparison(result)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        log.info("Wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
