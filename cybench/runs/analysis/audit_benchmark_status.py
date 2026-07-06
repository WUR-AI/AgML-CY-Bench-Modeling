#!/usr/bin/env python3
"""Audit SLURM benchmark output vs dashboard integration for one or all countries.

Answers: did jobs fail, are results missing, or do results exist but the dashboard
pipeline has not picked them up yet?

Run on anunna (from repo root)::

    poetry run python cybench/runs/analysis/audit_benchmark_status.py --country AO
    poetry run python cybench/runs/analysis/audit_benchmark_status.py --country AO --horizon eos
    poetry run python cybench/runs/analysis/audit_benchmark_status.py --all-countries --horizon eos
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from cybench.runs.analysis.benchmark_run_catalog import discover_benchmark_runs
from cybench.runs.analysis.collect_walk_forward_results import load_pooled_predictions
from cybench.runs.analysis.publish_pipeline_lib import (
    PipelineDefaults,
    PublishTarget,
    assess_readiness,
    horizon_to_batch_suffix,
)
from cybench.runs.slurm.benchmark_completion_lib import (
    default_output_root,
    resolve_country_list,
)
from cybench.runs.slurm.benchmark_submit_lib import batch_name, normalize_horizon, resolve_batch_dir

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MONOLITHIC = "baselines"


@dataclass
class CountryAudit:
    country: str
    horizon: str
    batch_folder: str
    per_country_dir: Path
    per_country_exists: bool
    monolithic_dir: Path
    monolithic_screening: int
    monolithic_walk_forward: int
    monolithic_wf_complete: int
    per_country_screening: int
    per_country_walk_forward: int
    per_country_wf_complete: int
    dashboard_ready: bool
    dashboard_reason: str
    published_html: bool
    collect_html: bool


def _count_runs(
    baselines_dir: Path,
    *,
    country: str,
    horizon_tag: str,
) -> tuple[int, int, int]:
    if not baselines_dir.is_dir():
        return 0, 0, 0
    cc = country.casefold()
    screening = walk_forward = wf_complete = 0
    for phase in ("screening", "walk_forward"):
        runs = discover_benchmark_runs(
            baselines_dir,
            phase=phase,
            horizon=horizon_tag,
            latest_only=True,
            allow_missing=True,
        )
        for run in runs:
            if run.country.casefold() != cc:
                continue
            if phase == "screening":
                if any(run.path.rglob("optimal_model.yaml")):
                    screening += 1
            else:
                walk_forward += 1
                try:
                    load_pooled_predictions(run.path, model_slug=run.model)
                    wf_complete += 1
                except ValueError:
                    pass
    return screening, walk_forward, wf_complete


def audit_country(
    country: str,
    *,
    horizon: str,
    output_root: Path,
    repo_root: Path,
    publish_root: Path,
    version: int,
) -> CountryAudit:
    cc = country.upper()
    slurm_hz = normalize_horizon(horizon)
    batch_hz = horizon_to_batch_suffix(slurm_hz)
    batch_folder = batch_name(cc, slurm_hz, version)
    per_country_dir, _ = resolve_batch_dir(output_root, batch_folder)
    mono_dir = output_root / _MONOLITHIC

    hz_tag = "eos" if batch_hz == "eos" else "mid_season"
    mono_scr, mono_wf, mono_ok = _count_runs(mono_dir, country=cc, horizon_tag=hz_tag)
    pc_scr, pc_wf, pc_ok = _count_runs(per_country_dir, country=cc, horizon_tag=hz_tag)

    target = PublishTarget(
        country=cc,
        batch_horizon=batch_hz,
        version=version,
        output_root=output_root,
        repo_root=repo_root,
        publish_root=publish_root,
    )
    report = assess_readiness(target)
    publish_html = (target.publish_dir / "dashboard.html").is_file()
    collect_html = (target.collect_dir / "compare_models.html").is_file()

    return CountryAudit(
        country=cc,
        horizon=batch_hz,
        batch_folder=per_country_dir.name if per_country_dir.is_dir() else batch_folder,
        per_country_dir=per_country_dir,
        per_country_exists=per_country_dir.is_dir(),
        monolithic_dir=mono_dir,
        monolithic_screening=mono_scr,
        monolithic_walk_forward=mono_wf,
        monolithic_wf_complete=mono_ok,
        per_country_screening=pc_scr,
        per_country_walk_forward=pc_wf,
        per_country_wf_complete=pc_ok,
        dashboard_ready=report.ready,
        dashboard_reason=report.reason,
        published_html=publish_html,
        collect_html=collect_html,
    )


def _verdict(a: CountryAudit) -> str:
    if a.dashboard_ready and a.published_html:
        return "OK published"
    if a.dashboard_ready and a.collect_html:
        return "READY not published (run orchestrate_dashboard_publish)"
    if a.dashboard_ready:
        return "READY not collected"
    if a.per_country_wf_complete > 0 or a.monolithic_wf_complete > 0:
        if not a.per_country_exists and a.monolithic_wf_complete > 0:
            return "RESULTS in output/baselines/ only (dashboard ignores monolithic batch)"
        if a.per_country_wf_complete > 0 and not a.dashboard_ready:
            return f"PARTIAL per-country WF ({a.dashboard_reason})"
        if a.monolithic_wf_complete > 0:
            return "PARTIAL in monolithic baselines/"
    if a.per_country_screening > 0 or a.monolithic_screening > 0:
        return "SCREENING only (walk-forward missing or failed)"
    if a.per_country_exists or a.monolithic_dir.is_dir():
        return "NO successful runs found (check SLURM err logs)"
    return "NEVER SUBMITTED (no output dir)"


def _print_audit(a: CountryAudit) -> None:
    print(f"\n=== {a.country} | {a.horizon} ===")
    print(f"  Per-country dir:  {a.per_country_dir}  ({'exists' if a.per_country_exists else 'missing'})")
    print(f"  Monolithic dir:   {a.monolithic_dir}")
    print(
        f"  Monolithic runs:  screening={a.monolithic_screening}  "
        f"walk_forward={a.monolithic_walk_forward}  wf_complete={a.monolithic_wf_complete}"
    )
    print(
        f"  Per-country runs: screening={a.per_country_screening}  "
        f"walk_forward={a.per_country_walk_forward}  wf_complete={a.per_country_wf_complete}"
    )
    print(f"  Dashboard:        ready={a.dashboard_ready}  ({a.dashboard_reason})")
    print(f"  Collect output:   {a.collect_html}")
    print(f"  Published HTML:   {a.published_html}")
    print(f"  >> {_verdict(a)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country")
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--all-countries", action="store_true")
    parser.add_argument("--horizon", default="eos")
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--publish-root", type=Path)
    args = parser.parse_args(argv)

    if not args.country and not args.countries and not args.all_countries:
        parser.error("Provide --country, --countries, or --all-countries")

    output_root = args.output_root or default_output_root(args.repo_root)
    publish_root = args.publish_root or PipelineDefaults().publish_root

    if args.country:
        countries = [args.country.upper()]
    elif args.countries:
        countries = sorted({c.upper() for c in args.countries})
    else:
        countries = resolve_country_list(
            all_countries=True,
            countries=None,
            country=None,
            repo_root=args.repo_root,
            data_dir=None,
            output_root=output_root,
        )

    if not output_root.is_dir():
        print(f"[WARN] Output root not found: {output_root}", file=sys.stderr)

    for cc in countries:
        _print_audit(
            audit_country(
                cc,
                horizon=args.horizon,
                output_root=output_root,
                repo_root=args.repo_root,
                publish_root=publish_root,
                version=args.version,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
