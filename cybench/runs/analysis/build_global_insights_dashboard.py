#!/usr/bin/env python3
"""Build a cross-country insights dashboard from collected walk-forward summaries.

Scans ``paper_walk_forward_*`` directories under an output root and writes
``insights.html`` plus related section pages for GitHub Pages.

Example::

    poetry run python cybench/runs/analysis/build_global_insights_dashboard.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --dest /lustre/backup/SHARED/AIN/agml/AgML-CY-Bench-dashboard/insights.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cybench.runs.analysis.global_insights_lib import build_insights_payload
from cybench.runs.analysis.index_map_lib import ensure_world_geojson

INSIGHTS_PAGES: tuple[tuple[str, str], ...] = (
    ("performance", "insights.html"),
    ("horizon", "insights-horizon.html"),
    ("crops", "insights-crops.html"),
    ("sample_size", "insights-sample-size.html"),
)


def build_insights_html(payload: dict, *, page: str = "performance") -> str:
    template_path = (
        Path(__file__).resolve().parent.parent / "viz" / "global_insights_template.html"
    )
    viz_dir = template_path.parent
    template = template_path.read_text(encoding="utf-8")
    for placeholder, filename in (
        ("__FAMILY_PANEL_SCRIPT__", "family_rq1_panel.js"),
        ("__FAMILY_RQ4_PANEL_SCRIPT__", "family_rq4_panel.js"),
    ):
        panel_path = viz_dir / filename
        panel_js = panel_path.read_text(encoding="utf-8") if panel_path.is_file() else ""
        template = template.replace(placeholder, panel_js)
    data_json = json.dumps(payload)
    return (
        template.replace("__PAGE__", page).replace("__DATA_JSON__", data_json)
    )


def write_insights_dashboard(
    *,
    output_root: Path,
    dest: Path,
    version: int = 4,
) -> Path:
    payload = build_insights_payload(output_root, version=version)
    if payload["n_rows"] == 0:
        raise RuntimeError(f"No walk_forward_summary.csv files found under {output_root}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload["geojson_href"] = ensure_world_geojson(dest.parent)
    primary = None
    for page, filename in INSIGHTS_PAGES:
        out = dest if filename == "insights.html" else dest.parent / filename
        out.write_text(build_insights_html(payload, page=page), encoding="utf-8")
        if filename == "insights.html":
            primary = out
    return primary or dest


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
        default=Path("/lustre/backup/SHARED/AIN/agml/AgML-CY-Bench-dashboard"),
        help="GitHub Pages clone root (used when --dest is omitted)",
    )
    parser.add_argument("--version", type=int, default=4, help="Batch version tag (default: 4)")
    args = parser.parse_args()

    dest = args.dest or (args.publish_root / "insights.html")
    path = write_insights_dashboard(
        output_root=args.output_root.resolve(),
        dest=dest.resolve(),
        version=args.version,
    )
    payload = build_insights_payload(args.output_root.resolve(), version=args.version)
    print(f"[DONE] Insights dashboard: {path}")
    for _page, filename in INSIGHTS_PAGES:
        if filename != "insights.html":
            print(f"[DONE]   also wrote {path.parent / filename}")
    n_eos = len(payload["leaderboards"].get("eos", {}).get("all") or [])
    n_mid = len(payload["leaderboards"].get("mid", {}).get("all") or [])
    print(
        f"[INFO] {payload['n_countries']} countries, "
        f"{n_eos} models (eos leaderboard), {n_mid} models (mid leaderboard)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
