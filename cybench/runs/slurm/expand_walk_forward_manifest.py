#!/usr/bin/env python3
"""Expand walk-forward GPU manifest rows to one SLURM task per seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from cybench.runs.slurm.benchmark_completion_lib import (
    default_output_root,
    expand_walk_forward_manifest_lines,
    read_manifest,
    resolve_batch_dir,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", required=True, help="Hydra experiment.name / baselines folder")
    parser.add_argument("--horizon", default="eos")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="Expand needs_gpu=yes rows to one line per seed (default for gpu manifests)",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--per-year-large",
        action="store_true",
        help="For countries with many regions, one SLURM task per seed and forecast year",
    )
    parser.add_argument(
        "--region-threshold",
        type=int,
        default=350,
        help="Region count at which per-year walk-forward expansion applies",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_root = args.output_root or default_output_root(repo_root)
    baselines_dir, _ = resolve_batch_dir(output_root, args.batch)
    jobs = read_manifest(args.input.resolve())
    lines = expand_walk_forward_manifest_lines(
        jobs,
        baselines_dir=baselines_dir,
        horizon=args.horizon,
        repo_root=repo_root,
        base_seed=args.base_seed,
        total_repetitions=args.repetitions,
        resume=args.resume,
        per_seed_for_gpu=args.per_seed,
        per_year_for_large_countries=args.per_year_large,
        region_threshold=args.region_threshold,
    )
    header = (
        "# crop country model framework hp_search feature_design needs_gpu [seed] [origin_year]\n"
        "# GPU rows: one SLURM task per seed (× forecast year for large countries)."
    )
    args.output.write_text(
        header + ("\n" + "\n".join(lines) if lines else "") + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Expanded {len(jobs)} job(s) -> {len(lines)} SLURM task(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
