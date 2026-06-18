#!/usr/bin/env python3
"""List incomplete benchmark jobs and optionally submit retries.

Unlike ``orchestrate_benchmark_submit.sh`` (new country batches), this inspects
Hydra output under ``../output/<batch>/`` and builds a **partial manifest** so
SLURM only reruns screening and/or walk-forward jobs that are missing or failed.

Preflight checks flag crop/country/model rows that cannot succeed (e.g. too few
yield years for the fixed screening split).

Examples (from repo root on anunna)::

    # Status table for one batch
    poetry run python cybench/runs/slurm/orchestrate_benchmark_complete.py \\
        --batch baselines_DE_eos_v1 --horizon eos --list

    # Write retry manifest only
    poetry run python cybench/runs/slurm/orchestrate_benchmark_complete.py \\
        --batch baselines_DE_eos_v1 --horizon eos --phase all \\
        -o cybench/runs/slurm/manifests/baselines_DE_eos_v1/benchmark_jobs_retry.txt

    # Submit incomplete jobs (screening + walk-forward with afterok)
    cybench/runs/slurm/orchestrate_benchmark_complete.sh \\
        --batch baselines_DE_eos_v1 --horizon eos --submit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cybench.runs.slurm.benchmark_completion_lib import (
    assess_manifest,
    jobs_for_phase,
    read_manifest,
    resolve_paths,
    split_manifest_groups,
    write_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SLURM_DIR = _REPO_ROOT / "cybench" / "runs" / "slurm"


def _print_report(assessments: list, *, phase: str) -> None:
    blocked = [a for a in assessments if a.blocked]
    scr_done = [a for a in assessments if a.screening_ok and not a.blocked]
    wf_done = [a for a in assessments if a.walk_forward_ok]
    need_scr = [a for a in assessments if a.needs_screening]
    need_wf = [a for a in assessments if a.needs_walk_forward]
    retry = jobs_for_phase(assessments, phase)

    print(
        f"{'crop':<6} {'cc':<4} {'model':<16} {'yrs':>4}  "
        f"{'screen':<8} {'wf':<8}  note"
    )
    print("-" * 88)
    for item in sorted(
        assessments,
        key=lambda a: (a.job.crop, a.job.country, a.job.model),
    ):
        scr = "ok" if item.screening_ok else ("BLOCK" if item.blocked else "MISS")
        wf = "ok" if item.walk_forward_ok else ("n/a" if item.blocked else "MISS")
        note = item.block_reason or item.screening_reason
        if item.screening_ok and not item.walk_forward_ok:
            note = item.walk_forward_reason
        print(
            f"{item.job.crop:<6} {item.job.country:<4} {item.job.model:<16} "
            f"{item.n_years:>4}  {scr:<8} {wf:<8}  {note}"
        )

    print()
    print(
        f"Total {len(assessments)} | complete wf {len(wf_done)} | "
        f"need screening {len(need_scr)} | need walk-forward {len(need_wf)} | "
        f"blocked {len(blocked)} | retry ({phase}) {len(retry)}"
    )


def _submit_retry(
    *,
    batch: str,
    horizon: str,
    phase: str,
    manifest_root: Path,
    assessments: list,
    dry_run: bool,
    force_cpu: bool,
) -> int:
    submit = _SLURM_DIR / "submit_benchmark.sh"
    if not submit.is_file():
        raise FileNotFoundError(submit)

    def _write_and_submit(jobs: list, submit_phase: str, label: str) -> int:
        if not jobs:
            return 0
        retry_path = manifest_root / f"benchmark_jobs_retry_{label}.txt"
        write_manifest(retry_path, jobs)
        target = manifest_root / "benchmark_jobs.txt"
        if dry_run:
            print(f"[DRY-RUN] cp {retry_path} -> {target}")
        else:
            manifest_root.mkdir(parents=True, exist_ok=True)
            target.write_text(retry_path.read_text(encoding="utf-8"), encoding="utf-8")

        cmd = [str(submit), submit_phase, "--horizon", horizon, "--batch", batch]
        if force_cpu:
            cmd.append("--cpu")
        if dry_run:
            cmd.append("--dry-run")
        print(f"[INFO] {' '.join(cmd)}  ({len(jobs)} jobs)")
        if dry_run:
            return 0
        return subprocess.run(cmd, check=False).returncode

    need_scr = [a.job for a in assessments if a.needs_screening]
    need_wf = [a.job for a in assessments if a.needs_walk_forward]

    if phase == "screening":
        return _write_and_submit(need_scr, "screening", "screening")
    if phase == "walk_forward":
        return _write_and_submit(need_wf, "walk_forward", "walk_forward")

    # phase == all: screening retries first, then walk-forward-only rows.
    code = _write_and_submit(need_scr, "screening", "screening")
    if code != 0:
        return code
    return _write_and_submit(need_wf, "walk_forward", "walk_forward")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        required=True,
        help="Hydra experiment.name / output folder (e.g. baselines_DE_eos_v1)",
    )
    parser.add_argument(
        "--horizon",
        default="eos",
        help="Prediction horizon passed to SLURM (default: eos)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Expected job list (default: cybench/runs/slurm/manifests/<batch>/benchmark_jobs.txt)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        help="Hydra output dir (default: ../output/<batch> relative to repo root)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override cybench/data for yield-year preflight",
    )
    parser.add_argument(
        "--phase",
        choices=["screening", "walk_forward", "all"],
        default="all",
        help="Which incomplete jobs to include in retry manifest (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print per-job status and exit",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write retry manifest to this path",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Copy retry manifest into batch dir and call submit_benchmark.sh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --submit: print commands without sbatch",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Route GPU manifest group to CPU (--cpu on submit_benchmark.sh)",
    )
    args = parser.parse_args(argv)

    manifest_path, baselines_dir, manifest_root = resolve_paths(
        batch=args.batch,
        repo_root=_REPO_ROOT,
        baselines_dir=args.baselines_dir,
        manifest_path=args.manifest,
    )
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not baselines_dir.is_dir():
        print(
            f"[WARN] Baselines dir missing (all jobs treated as incomplete): {baselines_dir}",
            file=sys.stderr,
        )

    jobs = read_manifest(manifest_path)
    assessments = assess_manifest(
        jobs,
        baselines_dir=baselines_dir,
        horizon=args.horizon,
        repo_root=_REPO_ROOT,
        data_dir=args.data_dir,
    )
    retry_jobs = jobs_for_phase(assessments, args.phase)

    if args.list or (not args.output and not args.submit):
        _print_report(assessments, phase=args.phase)

    if not retry_jobs and not args.submit:
        return 0
    if not retry_jobs and args.submit:
        print("[DONE] Nothing to retry")
        return 0

    if args.output:
        write_manifest(args.output, retry_jobs)
        groups = split_manifest_groups(retry_jobs)
        print(f"Wrote {len(retry_jobs)} retry jobs to {args.output}")
        for name, rows in groups.items():
            if rows:
                print(f"  {name}: {len(rows)}")

    if args.submit:
        return _submit_retry(
            batch=args.batch,
            horizon=args.horizon,
            phase=args.phase,
            manifest_root=manifest_root,
            assessments=assessments,
            dry_run=args.dry_run,
            force_cpu=args.cpu,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
