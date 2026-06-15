#!/usr/bin/env python3
"""Backward-compatible wrapper: screening vs walk-forward at the same horizon.

For general comparisons (horizons, phases, custom labels), use
``compare_benchmark_runs.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cybench.runs.compare_benchmark_runs import (
    _print_preview,
    _print_summary,
    compare_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare screening vs walk-forward (use compare_benchmark_runs.py for general cases)."
    )
    parser.add_argument("--baselines-dir", type=Path, default=Path("../output/baselines"))
    parser.add_argument("--output", type=Path, default=Path("../output/screening_vs_walk_forward.csv"))
    parser.add_argument("--horizon", default="eos", help="Horizon tag (default: eos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    horizon = None if args.horizon in ("", "any", "*") else args.horizon
    groups = [
        ("screening", "screening", horizon),
        ("walk_forward", "walk_forward", horizon),
    ]
    df = compare_groups(args.baselines_dir.resolve(), groups, seed=args.seed)
    if df.empty:
        print("[WARN] No paired screening + walk-forward runs found.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"[DONE] Wrote {len(df)} paired rows to {args.output}")
    _print_summary(df, ["screening", "walk_forward"])
    print()
    _print_preview(df, ["screening", "walk_forward"])


if __name__ == "__main__":
    main()
