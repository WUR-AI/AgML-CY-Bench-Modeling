#!/usr/bin/env python3
"""Build a model-family radar dashboard from collected walk-forward summaries.

Scans ``paper_walk_forward_*`` directories under an output root and writes
``model_families.html`` for GitHub Pages (relative performance across evaluation views).

Example::

    poetry run python cybench/runs/analysis/build_model_family_radar_dashboard.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --dest /lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard/model_families.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cybench.runs.analysis.model_family_radar_lib import build_radar_payload
from cybench.runs.analysis.index_map_lib import ensure_world_geojson


def build_radar_html(payload: dict) -> str:
    template_path = (
        Path(__file__).resolve().parent.parent / "viz" / "model_family_radar_template.html"
    )
    template = template_path.read_text(encoding="utf-8")
    data_json = json.dumps(payload)
    return template.replace("__DATA_JSON__", data_json)


def write_model_family_radar_dashboard(
    *,
    output_root: Path,
    dest: Path,
    version: int = 2,
) -> Path:
    payload = build_radar_payload(output_root, version=version)
    if payload["n_rows"] == 0:
        raise RuntimeError(f"No walk_forward_summary.csv files found under {output_root}")
    if not payload["by_horizon"]:
        raise RuntimeError("No horizon slices in collected summaries")
    payload["geojson_href"] = ensure_world_geojson(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_radar_html(payload), encoding="utf-8")
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
        help="Output HTML path (default: <publish-root>/model_families.html)",
    )
    parser.add_argument(
        "--publish-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard"),
        help="GitHub Pages clone root (used when --dest is omitted)",
    )
    parser.add_argument("--version", type=int, default=2, help="Batch version tag (default: 2)")
    args = parser.parse_args()

    dest = args.dest or (args.publish_root / "model_families.html")
    path = write_model_family_radar_dashboard(
        output_root=args.output_root.resolve(),
        dest=dest.resolve(),
        version=args.version,
    )
    payload = build_radar_payload(args.output_root.resolve(), version=args.version)
    n_families = len(payload["by_horizon"].get("eos", {}).get("all", {}).get("families", []))
    print(f"[DONE] Model-family radar: {path}")
    print(
        f"[INFO] {payload['n_countries']} countries, "
        f"{n_families} family representatives (eos / all crops)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
