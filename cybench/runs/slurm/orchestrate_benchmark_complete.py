#!/usr/bin/env python3
"""List incomplete benchmark jobs and optionally submit retries.

Unlike ``orchestrate_benchmark_submit.sh`` (new country batches), this inspects
Hydra output under ``../output/<batch>/`` and builds a **partial manifest** so
SLURM only reruns screening and/or walk-forward jobs that are missing or failed.

If ``manifests/<batch>/benchmark_jobs.txt`` is missing, the manifest is built by
filtering the shared ``benchmark_jobs.txt`` for the batch country, or by calling
``generate_job_manifest.py`` for that country.

Examples (from repo root on anunna)::

    # One horizon (manifest auto-resolved)
    cybench/runs/slurm/orchestrate_benchmark_complete.sh \\
        --batch baselines_DE_eos_v1 --horizon eos --list

    # Both horizons for Germany (expands to baselines_DE_eos_v1 + baselines_DE_mid_v1)
    cybench/runs/slurm/orchestrate_benchmark_complete.sh \\
        --country DE --horizons eos mid --list

    # Submit retries
    cybench/runs/slurm/orchestrate_benchmark_complete.sh \\
        --country DE --horizons eos mid --submit --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cybench.runs.slurm.benchmark_completion_lib import (
    assess_manifest,
    expand_target_batches,
    jobs_for_phase,
    resolve_paths,
    split_manifest_groups,
    write_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SLURM_DIR = _REPO_ROOT / "cybench" / "runs" / "slurm"


def _print_report(
    assessments: list,
    *,
    batch: str,
    horizon: str,
    manifest_source: str,
    phase: str,
) -> None:
    blocked = [a for a in assessments if a.blocked]
    wf_done = [a for a in assessments if a.walk_forward_ok]
    need_scr = [a for a in assessments if a.needs_screening]
    need_wf = [a for a in assessments if a.needs_walk_forward]
    retry = jobs_for_phase(assessments, phase)

    print(f"\n=== {batch} | horizon={horizon} | manifest: {manifest_source} ===")
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

    code = _write_and_submit(need_scr, "screening", "screening")
    if code != 0:
        return code
    return _write_and_submit(need_wf, "walk_forward", "walk_forward")


def _process_batch(
    *,
    batch: str,
    horizon: str,
    args: argparse.Namespace,
) -> int:
    try:
        manifest_path, baselines_dir, manifest_root, _, jobs, manifest_source = resolve_paths(
            batch=batch,
            repo_root=_REPO_ROOT,
            baselines_dir=args.baselines_dir,
            manifest_path=args.manifest,
            output_root=args.output_root,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not baselines_dir.is_dir():
        print(
            f"[WARN] Baselines dir missing (all jobs treated as incomplete): {baselines_dir}",
            file=sys.stderr,
        )

    assessments = assess_manifest(
        jobs,
        baselines_dir=baselines_dir,
        horizon=horizon,
        repo_root=_REPO_ROOT,
        data_dir=args.data_dir,
    )
    retry_jobs = jobs_for_phase(assessments, args.phase)

    if args.list or (not args.output and not args.submit):
        _print_report(
            assessments,
            batch=batch,
            horizon=horizon,
            manifest_source=manifest_source,
            phase=args.phase,
        )

    if args.output:
        out = args.output
        if len(args._targets) > 1:
            out = args.output.parent / f"{args.output.stem}_{batch}{args.output.suffix}"
        write_manifest(out, retry_jobs)
        groups = split_manifest_groups(retry_jobs)
        print(f"Wrote {len(retry_jobs)} retry jobs to {out}")
        for name, rows in groups.items():
            if rows:
                print(f"  {name}: {len(rows)}")

    if not retry_jobs:
        if args.submit:
            print(f"[DONE] {batch}: nothing to retry")
        return 0

    if args.submit:
        return _submit_retry(
            batch=batch,
            horizon=horizon,
            phase=args.phase,
            manifest_root=manifest_root,
            assessments=assessments,
            dry_run=args.dry_run,
            force_cpu=args.cpu,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        help="Hydra experiment.name / output folder (e.g. baselines_DE_eos_v1)",
    )
    parser.add_argument(
        "--country",
        help="Country code (alternative to --batch; use with --horizons)",
    )
    parser.add_argument(
        "--horizon",
        action="append",
        dest="horizons",
        help="Prediction horizon (repeatable; default: eos). Alias: use --horizons.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        metavar="H",
        help="One or more horizons: eos, mid, middle-of-season (default: eos)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Batch version when expanding from --country (default: 1)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Explicit job list (skips auto-resolve)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        help="Hydra output dir (default: $CYBENCH_OUTPUT_ROOT/<batch> or lustre output)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Parent of baselines_* folders (default: lustre output or ../output)",
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

    horizons = args.horizons or ["eos"]
    if not args.batch and not args.country:
        parser.error("Provide --batch or --country")

    try:
        targets = expand_target_batches(
            batch=args.batch,
            country=args.country,
            horizons=horizons,
            version=args.version,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    args._targets = targets  # type: ignore[attr-defined]
    exit_code = 0
    for batch, horizon in targets:
        code = _process_batch(batch=batch, horizon=horizon, args=args)
        if code != 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
