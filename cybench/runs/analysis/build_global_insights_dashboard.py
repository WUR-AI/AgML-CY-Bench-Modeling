#!/usr/bin/env python3
"""Build a cross-country insights dashboard from collected walk-forward summaries.

Scans ``paper_walk_forward_*`` directories under an output root and writes
``insights.html`` for GitHub Pages (model leaderboard + eos vs mid-season).

Example::

    poetry run python cybench/runs/analysis/build_global_insights_dashboard.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --dest /lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard/insights.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cybench.runs.analysis.global_insights_lib import build_insights_payload


def build_insights_html(payload: dict) -> str:
    template_path = (
        Path(__file__).resolve().parent.parent / "viz" / "global_insights_template.html"
    )
    template = template_path.read_text(encoding="utf-8")
    data_json = json.dumps(payload)
    return template.replace("__DATA_JSON__", data_json)


def write_insights_dashboard(
    *,
    output_root: Path,
    dest: Path,
    version: int = 1,
) -> Path:
    payload = build_insights_payload(output_root, version=version)
    if payload["n_rows"] == 0:
        raise RuntimeError(f"No walk_forward_summary.csv files found under {output_root}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_insights_html(payload), encoding="utf-8")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/output"),
        help="Root containing paper_walk_forward_* collect directories",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Output HTML path (default: <publish-root>/insights.html)",
    )
    parser.add_argument(
        "--publish-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard"),
        help="GitHub Pages clone root (used when --dest is omitted)",
    )
    parser.add_argument("--version", type=int, default=1, help="Batch version tag (default: 1)")
    args = parser.parse_args()

    dest = args.dest or (args.publish_root / "insights.html")
    path = write_insights_dashboard(
        output_root=args.output_root.resolve(),
        dest=dest.resolve(),
        version=args.version,
    )
    payload = build_insights_payload(args.output_root.resolve(), version=args.version)
    print(f"[DONE] Insights dashboard: {path}")
    n_eos = len(payload["leaderboards"].get("eos") or [])
    n_mid = len(payload["leaderboards"].get("mid") or [])
    print(
        f"[INFO] {payload['n_countries']} countries, "
        f"{n_eos} models (eos leaderboard), {n_mid} models (mid leaderboard)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
