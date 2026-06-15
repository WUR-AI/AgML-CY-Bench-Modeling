#!/usr/bin/env python3
"""Compare pooled test metrics across arbitrary Hydra baseline run groups.

Match runs on (crop, country, model) and align metrics side-by-side.
Use for screening vs walk-forward, eos vs mid_season, or any labeled pair.

Examples::

    # Screening vs walk-forward (same horizon)
    poetry run python cybench/runs/analysis/compare_benchmark_runs.py \\
        --baselines-dir ../output/baselines \\
        --group wf=walk_forward/eos \\
        --group scr=screening/eos \\
        --output ../output/compare_wf_vs_screen_eos.csv

    # End-of-season vs mid-season walk-forward
    poetry run python cybench/runs/analysis/compare_benchmark_runs.py \\
        --baselines-dir ../output/baselines \\
        --group eos=walk_forward/eos \\
        --group mid=walk_forward/mid_season \\
        --output ../output/compare_horizons.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from cybench.runs.analysis.benchmark_run_catalog import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRIC_KEYS,
    discover_benchmark_runs,
    load_run_metrics,
    parse_group_spec,
)


def _build_group_index(
    baselines_dir: Path,
    label: str,
    phase: str,
    horizon: str | None,
    *,
    seed: int,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in discover_benchmark_runs(
        baselines_dir, phase=phase, horizon=horizon, latest_only=True
    ):
        metrics = load_run_metrics(run, seed=seed)
        if metrics is None:
            continue
        rows[run.dataset_key] = {
            "crop": run.crop,
            "country": run.country,
            "model": run.model,
            "dataset": run.dataset,
            f"{label}__horizon": run.horizon,
            f"{label}__phase": run.phase,
            f"{label}__run_dir": str(run.path),
            **{f"{label}__{k}": metrics.get(k) for k in METRIC_KEYS},
            f"{label}__n_regions": metrics.get("n_regions"),
            f"{label}__n_years": metrics.get("n_years"),
        }
    return rows


def _diagnose_empty_comparison(
    baselines_dir: Path,
    groups: list[tuple[str, str, str | None]],
    indexes: list[dict[tuple[str, str, str], dict[str, Any]]],
) -> None:
    labels = [g[0] for g in groups]
    for label, (_, phase, horizon), idx in zip(labels, groups, indexes):
        n_metrics = len(idx)
        n_dirs = len(
            discover_benchmark_runs(
                baselines_dir, phase=phase, horizon=horizon, latest_only=True
            )
        )
        horizon_s = horizon or "any"
        print(
            f"  [{label}] phase={phase} horizon={horizon_s}: "
            f"{n_dirs} run dirs, {n_metrics} with loadable metrics"
        )
    if all(indexes):
        keys = [set(idx) for idx in indexes]
        overlap = keys[0].intersection(*keys[1:])
        if not overlap:
            print(
                "[HINT] Groups have runs but no shared (crop, country, model). "
                "Check model names and that both phases finished."
            )
    elif any(not idx for idx in indexes):
        empty = [labels[i] for i, idx in enumerate(indexes) if not idx]
        print(f"[HINT] Empty groups: {', '.join(empty)}")


def compare_groups(
    baselines_dir: Path,
    groups: list[tuple[str, str, str | None]],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if len(groups) < 2:
        raise ValueError("Provide at least two --group specs to compare.")

    indexes = [
        _build_group_index(baselines_dir, label, phase, horizon, seed=seed)
        for label, phase, horizon in groups
    ]
    common_keys = set(indexes[0])
    for idx in indexes[1:]:
        common_keys &= set(idx)
    if not common_keys:
        print("[WARN] No matched runs across all groups.")
        _diagnose_empty_comparison(baselines_dir, groups, indexes)
        return pd.DataFrame()

    labels = [g[0] for g in groups]
    rows: list[dict[str, Any]] = []
    for key in sorted(common_keys):
        merged: dict[str, Any] = {}
        for idx in indexes:
            merged.update(idx[key])
        base = indexes[0][key]
        merged.setdefault("crop", base["crop"])
        merged.setdefault("country", base["country"])
        merged.setdefault("model", base["model"])
        merged.setdefault("dataset", base["dataset"])

        for metric in METRIC_KEYS:
            a_label, b_label = labels[0], labels[1]
            a_col, b_col = f"{a_label}__{metric}", f"{b_label}__{metric}"
            if a_col in merged and b_col in merged:
                av, bv = merged.get(a_col), merged.get(b_col)
                if av is not None and bv is not None and not (
                    pd.isna(av) or pd.isna(bv)
                ):
                    merged[f"delta__{metric}"] = float(bv) - float(av)

        rows.append(merged)
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame, labels: list[str]) -> None:
    if len(labels) < 2:
        return
    a, b = labels[0], labels[1]
    print(f"[INFO] Compared groups '{a}' vs '{b}' on {len(df)} matched (crop, country, model) rows.")
    for metric in METRIC_KEYS:
        delta_col = f"delta__{metric}"
        if delta_col not in df.columns:
            continue
        deltas = df[delta_col].dropna()
        if deltas.empty:
            continue
        if metric in HIGHER_IS_BETTER:
            wins = int((deltas > 0).sum())
            direction = f"higher {metric} better → '{b}' wins"
        elif metric in LOWER_IS_BETTER:
            wins = int((deltas < 0).sum())
            direction = f"lower {metric} better → '{b}' wins"
        else:
            wins = int((deltas > 0).sum())
            direction = metric
        print(f"  {metric}: {wins}/{len(deltas)} rows favor '{b}' ({direction})")


def _print_preview(df: pd.DataFrame, labels: list[str]) -> None:
    if df.empty:
        return
    show = ["dataset", "model"]
    for label in labels:
        hc = f"{label}__horizon"
        if hc in df.columns:
            show.append(hc)
    for label in labels:
        for metric in METRIC_KEYS:
            show.append(f"{label}__{metric}")
    for metric in METRIC_KEYS:
        dc = f"delta__{metric}"
        if dc in df.columns:
            show.append(dc)
    show = [c for c in show if c in df.columns]
    print(df.loc[:, show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-dir", type=Path, default=Path("../output/baselines"))
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="LABEL=PHASE[/HORIZON]",
        help="Run group, e.g. wf=walk_forward/eos (repeat for each group)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = [parse_group_spec(spec) for spec in args.group]
    labels = [g[0] for g in groups]

    df = compare_groups(args.baselines_dir.resolve(), groups, seed=args.seed)
    if df.empty:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"[DONE] Wrote {len(df)} rows to {args.output}")
    _print_summary(df, labels)
    print()
    _print_preview(df, labels)


if __name__ == "__main__":
    main()
