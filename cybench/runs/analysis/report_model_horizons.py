#!/usr/bin/env python3
"""Print paired horizon report for a single model (fair country intersection).

Example::

    poetry run python cybench/runs/analysis/report_model_horizons.py \\
        --model tabdpt \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --version 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cybench.runs.analysis.global_insights_lib import (
    HORIZON_DISPLAY_LABELS,
    discover_summary_tables,
    load_summary_frame,
    report_model_horizon_pairs,
)


def _print_report(report: dict) -> None:
    if report.get("error"):
        print(f"[ERROR] {report['error']}")
        return

    model = report["model"]
    crop = report.get("crop", "all")
    horizons = report.get("horizons", [])
    print(f"\n=== {model} | crop={crop} | paired horizons: {', '.join(horizons)} ===")
    print(report.get("interpretation", ""))
    print(f"\nPaired countries: {report['n_paired_countries']}")
    if report.get("excluded_countries"):
        ex = report["excluded_countries"]
        preview = ", ".join(ex[:12])
        suffix = f" (+{len(ex) - 12} more)" if len(ex) > 12 else ""
        print(f"Excluded (incomplete horizons): {preview}{suffix}")

    summary = report.get("summary", {})
    print("\nAggregate (median across paired countries):")
    for hz in horizons:
        key = f"median_nrmse_{hz}"
        label = HORIZON_DISPLAY_LABELS.get(hz, hz)
        if key in summary:
            print(f"  {label}: {summary[key]:.4f}")

    if "eos" in horizons:
        print("\nVs end-of-season (positive delta ⇒ EOS better):")
        for hz in horizons:
            if hz == "eos":
                continue
            rate_key = f"eos_better_than_{hz}_rate"
            med_key = f"median_delta_{hz}_minus_eos"
            if rate_key in summary:
                label = HORIZON_DISPLAY_LABELS.get(hz, hz)
                print(
                    f"  EOS wins vs {label}: {100 * summary[rate_key]:.1f}% "
                    f"(median Δ={summary.get(med_key, float('nan')):+.4f})"
                )

    best_counts: dict[str, int] = {}
    for row in report.get("detail", []):
        bh = row.get("best_horizon")
        if bh:
            best_counts[bh] = best_counts.get(bh, 0) + 1
    if best_counts:
        print("\nBest horizon per country (lowest NRMSE):")
        for hz in horizons:
            n = best_counts.get(hz, 0)
            label = HORIZON_DISPLAY_LABELS.get(hz, hz)
            print(f"  {label}: {n}/{report['n_paired_countries']}")

    print("\nPer-country detail:")
    hdr = ["country"] + [f"nrmse_{hz}" for hz in horizons] + ["best"]
    print("  " + "  ".join(f"{h:>12}" for h in hdr))
    for row in report.get("detail", []):
        parts = [f"{row['country']:>12}"]
        for hz in horizons:
            parts.append(f"{row.get(f'nrmse_{hz}', float('nan')):>12.4f}")
        parts.append(f"{row.get('best_horizon', ''):>12}")
        print("  " + "  ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model slug (e.g. tabdpt)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/output"),
    )
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument("--crop", help="Optional crop filter (maize, wheat, ...)")
    parser.add_argument(
        "--min-sample-ratio",
        type=float,
        default=None,
        help="Drop countries where min/max n_samples across horizons is below this",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    paths = discover_summary_tables(args.output_root.resolve(), version=args.version)
    df = load_summary_frame(paths)
    crops = [args.crop] if args.crop else ["all", *sorted({str(c) for c in df.get("crop", pd.Series()).dropna().unique()})]

    reports: dict[str, dict] = {}
    for crop_key in crops:
        crop_filter = None if crop_key == "all" else crop_key
        report = report_model_horizon_pairs(
            df,
            args.model,
            crop=crop_filter,
            min_sample_ratio=args.min_sample_ratio,
        )
        reports[crop_key] = report
        if not args.json:
            _print_report(report)

    if args.json:
        print(json.dumps(reports, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
