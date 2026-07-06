#!/usr/bin/env python3
"""Rebuild compare_models.html from existing summaries and republish country dashboards.

Use after updating dashboard_template.html (no replotting).

Example::

    poetry run python cybench/runs/analysis/refresh_dashboard_views.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --publish-root /lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from cybench.runs.analysis.collect_walk_forward_results import write_model_comparison_dashboard
from cybench.runs.analysis.global_insights_lib import parse_paper_dir_name
from cybench.runs.analysis.publish_dashboard_bundle import publish_bundle

_PAPER_RE = re.compile(
    r"^paper_walk_forward_(?P<country>[a-z]{2})_(?P<horizon>eos|mid|qtr)_v(?P<version>\d+)$"
)


def paper_dirs(output_root: Path, *, version: int | None = None) -> list[Path]:
    """Collected paper dirs; keeps latest vN per country×horizon unless ``version`` is set."""
    by_key: dict[tuple[str, str], tuple[int, Path]] = {}
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_paper_dir_name(entry.name)
        if parsed is None:
            continue
        country, hz, ver = parsed
        if version is not None and ver != version:
            continue
        if not (entry / "walk_forward_summary.csv").is_file():
            continue
        key = (country, hz)
        prev = by_key.get(key)
        if prev is None or ver > prev[0]:
            by_key[key] = (ver, entry)
    return [path for _, path in sorted(by_key.values(), key=lambda x: x[1].name)]


def publish_slug(paper_dir: Path) -> str:
    parsed = parse_paper_dir_name(paper_dir.name)
    if parsed is None:
        raise ValueError(f"Not a paper_walk_forward dir: {paper_dir.name}")
    country, hz, ver = parsed
    return f"{country.lower()}_walk_forward_{hz}_v{ver}"


def refresh_one(paper_dir: Path, publish_root: Path, *, dry_run: bool = False, pages_lite: bool = False) -> None:
    summary_path = paper_dir / "walk_forward_summary.csv"
    rows = pd.read_csv(summary_path).to_dict(orient="records")
    slug = publish_slug(paper_dir)
    if dry_run:
        print(f"[DRY-RUN] refresh {paper_dir.name} -> {publish_root / slug}")
        return
    write_model_comparison_dashboard(paper_dir, rows, bundle_assets=True)
    publish_bundle(
        source_dir=paper_dir,
        dest_dir=publish_root / slug,
        title=None,
        pages_lite=pages_lite,
    )
    print(f"[OK] {paper_dir.name} -> {slug}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--version", type=int, default=3, help="Batch version (default: 3)")
    parser.add_argument("--country", action="append", dest="countries", metavar="CC")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--full-assets",
        action="store_true",
        help="Publish all plot PNGs including maps (may exceed GitHub Pages 1 GB limit)",
    )
    args = parser.parse_args()
    pages_lite = not args.full_assets

    output_root = args.output_root.resolve()
    publish_root = args.publish_root.resolve()
    countries = {c.upper() for c in args.countries} if args.countries else None

    dirs = paper_dirs(output_root, version=args.version)
    if countries:
        dirs = [d for d in dirs if parse_paper_dir_name(d.name)[0] in countries]

    if not dirs:
        print("[WARN] No paper_walk_forward_* dirs with walk_forward_summary.csv")
        return 1

    for paper_dir in dirs:
        refresh_one(paper_dir, publish_root, dry_run=args.dry_run, pages_lite=pages_lite)

    if not args.dry_run:
        from cybench.runs.analysis.build_global_insights_dashboard import write_insights_dashboard
        from cybench.runs.analysis.build_model_family_radar_dashboard import (
            write_model_family_radar_dashboard,
        )
        from cybench.runs.analysis.publish_dashboard_bundle import (
            apply_pages_lite_to_publish_root,
            discover_index_entries,
            prune_obsolete_dashboard_dirs,
            report_publish_bundle_size,
            update_index,
        )

        write_insights_dashboard(
            output_root=output_root,
            dest=publish_root / "insights.html",
            version=args.version,
        )
        write_model_family_radar_dashboard(
            output_root=output_root,
            dest=publish_root / "model_families.html",
            version=args.version,
        )
        prune_obsolete_dashboard_dirs(publish_root)
        if pages_lite:
            apply_pages_lite_to_publish_root(publish_root)
        update_index(publish_root, discover_index_entries(publish_root))
        report_publish_bundle_size(publish_root)
        print(f"[OK] insights.html, model_families.html, index.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
