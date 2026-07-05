#!/usr/bin/env python3
"""Stage a self-contained dashboard bundle for GitHub Pages.

Takes output from ``collect_walk_forward_results.py`` (``compare_models.html`` +
``assets/``) or a standalone ``dashboard.html`` + ``assets/`` tree and copies
them into a publish directory (e.g. a clone of CY-Bench-dashboard).

Example::

    poetry run python cybench/runs/analysis/publish_dashboard_bundle.py \\
        --source-dir ../output/paper_walk_forward_de_mid_v1 \\
        --publish-root ~/CY-Bench-dashboard \\
        --slug de_walk_forward_mid_v1 \\
        --update-index
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COUNTRY_NAMES: dict[str, str] = {
    "ao": "Angola",
    "ar": "Argentina",
    "at": "Austria",
    "au": "Australia",
    "be": "Belgium",
    "bf": "Burkina Faso",
    "bg": "Bulgaria",
    "br": "Brazil",
    "cn": "China",
    "cz": "Czechia",
    "de": "Germany",
    "dk": "Denmark",
    "ee": "Estonia",
    "el": "Greece",
    "es": "Spain",
    "et": "Ethiopia",
    "fi": "Finland",
    "fr": "France",
    "hr": "Croatia",
    "hu": "Hungary",
    "ie": "Ireland",
    "in": "India",
    "it": "Italy",
    "ls": "Lesotho",
    "lt": "Lithuania",
    "lv": "Latvia",
    "mg": "Madagascar",
    "ml": "Mali",
    "mw": "Malawi",
    "mx": "Mexico",
    "mz": "Mozambique",
    "ne": "Niger",
    "nl": "Netherlands",
    "pl": "Poland",
    "pt": "Portugal",
    "ro": "Romania",
    "se": "Sweden",
    "sk": "Slovakia",
    "sn": "Senegal",
    "td": "Chad",
    "us": "United States",
    "za": "South Africa",
    "zm": "Zambia",
}

_HORIZON_LABELS: dict[str, str] = {
    "eos": "End of season",
    "mid": "Mid-season",
    "mid_season": "Mid-season",
    "qtr": "Late season (75% observed)",
    "quarter_season": "Late season (75% observed)",
}


@dataclass(frozen=True)
class IndexEntry:
    href: str
    slug: str
    title: str
    subtitle: str
    country_code: str | None
    kind: str  # walk_forward | screening


def _resolve_html_and_assets(source_dir: Path) -> tuple[Path, Path | None]:
    compare = source_dir / "compare_models.html"
    dashboard = source_dir / "dashboard.html"
    if compare.is_file():
        return compare, source_dir / "assets"
    if dashboard.is_file():
        return dashboard, source_dir / "assets"
    raise FileNotFoundError(
        f"No compare_models.html or dashboard.html in {source_dir}"
    )


_SLUG_RE = re.compile(
    r"^(?P<country>[a-z]{2})_walk_forward_(?P<horizon>eos|mid|qtr)_v(?P<version>\d+)$"
)


def parse_publish_slug(name: str) -> tuple[str, str, int] | None:
    """Parse ``de_walk_forward_mid_v2`` → (``DE``, ``mid``, 2)."""
    match = _SLUG_RE.match(name)
    if not match:
        return None
    return match.group("country").upper(), match.group("horizon"), int(match.group("version"))


def prune_obsolete_dashboard_dirs(
    publish_root: Path,
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove published folders superseded by a newer ``vN`` (same country × horizon).

    GitHub Pages artifacts are limited to 1 GB; keeping only the latest version per
    country×horizon avoids duplicate v1+v2 asset trees (~30% of the bundle).
    """
    publish_root = publish_root.resolve()
    by_key: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    n_dirs = 0
    n_matched = 0
    for child in publish_root.iterdir():
        if not child.is_dir() or child.name == "assets":
            continue
        n_dirs += 1
        parsed = parse_publish_slug(child.name)
        if parsed is None:
            continue
        n_matched += 1
        cc, hz, ver = parsed
        by_key.setdefault((cc, hz), []).append((ver, child))

    removed: list[Path] = []
    for entries in by_key.values():
        if len(entries) < 2:
            continue
        max_ver = max(ver for ver, _ in entries)
        for ver, path in entries:
            if ver < max_ver:
                removed.append(path)
                if dry_run:
                    print(f"[DRY-RUN] prune {path.name}")
                else:
                    shutil.rmtree(path)
                    print(f"[OK] pruned {path.name}")

    if not removed:
        print(
            f"[INFO] publish-root={publish_root} · "
            f"{n_dirs} subdirs scanned · {n_matched} walk-forward slug(s) · "
            f"{len(by_key)} country×horizon keys"
        )
        if n_matched == 0:
            print(
                "[WARN] No folders matching "
                "'{cc}_walk_forward_{eos|mid|qtr}_vN'. "
                "Use the CY-Bench-dashboard git clone as --publish-root "
                "(e.g. /lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard), "
                "not the AgML-CY-Bench-AAAI source tree."
            )
        elif all(len(e) == 1 for e in by_key.values()):
            print("[INFO] No obsolete versions: at most one vN per country×horizon.")
    return sorted(removed, key=lambda p: p.name)


# GitHub Pages artifact limit (deploy fails above this).
GITHUB_PAGES_MAX_BYTES = 1_073_741_824

# Map PNGs are ~45% of each dashboard; downscale for Pages to stay under 1 GB.
PAGES_MAP_MARKERS = ("map_actual", "map_pred")
# Linear scale (0.70 → ~49% pixels / ~half the map bytes; keeps maps readable).
PAGES_MAP_SCALE = 0.70
PAGES_MAP_PNG_OPTIMIZE = True
# Source collect: lower DPI for map panels only (scatter/temporal stay sharp).
MAP_PANEL_DPI = 100
DEFAULT_PANEL_DPI = 160


def _is_map_asset(filename: str) -> bool:
    return any(marker in filename for marker in PAGES_MAP_MARKERS)


def downgrade_map_png(
    src: Path,
    dest: Path,
    *,
    scale: float = PAGES_MAP_SCALE,
) -> int:
    """Resize a map PNG for GitHub Pages; return output size in bytes."""
    from PIL import Image

    with Image.open(src) as img:
        width, height = img.size
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        if new_size != (width, height):
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="PNG", optimize=PAGES_MAP_PNG_OPTIMIZE)
    return dest.stat().st_size


def estimate_publish_bundle_size(publish_root: Path) -> dict[str, Any]:
    """Total bytes under publish_root (excluding .git)."""
    publish_root = publish_root.resolve()
    total = 0
    n_files = 0
    n_dashboard_dirs = 0
    by_horizon: dict[str, int] = {}
    for path in publish_root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        size = path.stat().st_size
        total += size
        n_files += 1
    for child in publish_root.iterdir():
        if not child.is_dir() or child.name in {"assets", ".git"}:
            continue
        if not (child / "dashboard.html").is_file():
            continue
        n_dashboard_dirs += 1
        hz = "other"
        for token in ("_eos_", "_mid_", "_qtr_"):
            if token in child.name:
                hz = token.strip("_")
                break
        dir_bytes = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        by_horizon[hz] = by_horizon.get(hz, 0) + dir_bytes
    return {
        "total_bytes": total,
        "total_mb": round(total / (1024 * 1024), 1),
        "n_files": n_files,
        "n_dashboard_dirs": n_dashboard_dirs,
        "by_horizon_bytes": by_horizon,
        "over_pages_limit": total > GITHUB_PAGES_MAX_BYTES,
        "pages_limit_mb": round(GITHUB_PAGES_MAX_BYTES / (1024 * 1024), 1),
    }


def apply_pages_lite_to_publish_root(
    publish_root: Path,
    *,
    dry_run: bool = False,
    scale: float = PAGES_MAP_SCALE,
) -> tuple[int, int]:
    """Downscale map PNGs in every published dashboard (scatter/temporal unchanged).

    Returns (maps_processed, bytes_saved).
    """
    publish_root = publish_root.resolve()
    processed = 0
    saved = 0
    for child in publish_root.iterdir():
        if not child.is_dir() or child.name in {"assets", ".git"}:
            continue
        assets = child / "assets"
        if not assets.is_dir():
            continue
        for asset in assets.glob("*.png"):
            if not _is_map_asset(asset.name):
                continue
            before = asset.stat().st_size
            if dry_run:
                print(f"[DRY-RUN] downgrade {child.name}/assets/{asset.name}")
                processed += 1
                continue
            tmp = asset.with_name(asset.name + ".tmp")
            after = downgrade_map_png(asset, tmp, scale=scale)
            tmp.replace(asset)
            processed += 1
            saved += max(0, before - after)
    if processed:
        action = "would downgrade" if dry_run else "downgraded"
        print(
            f"[OK] {action} {processed} map PNG(s)"
            + (f", saved {saved / (1024 * 1024):.1f} MB" if saved else "")
        )
    return processed, saved


def report_publish_bundle_size(publish_root: Path) -> None:
    stats = estimate_publish_bundle_size(publish_root)
    print(
        f"[SIZE] publish-root={publish_root.resolve()} · "
        f"{stats['total_mb']} MB · {stats['n_dashboard_dirs']} dashboards · "
        f"{stats['n_files']} files"
    )
    for hz, nbytes in sorted(stats["by_horizon_bytes"].items()):
        print(f"  {hz}: {nbytes / (1024 * 1024):.1f} MB")
    if stats["over_pages_limit"]:
        over_mb = stats["total_bytes"] / (1024 * 1024) - stats["pages_limit_mb"]
        print(
            f"[WARN] Over GitHub Pages artifact limit ({stats['pages_limit_mb']} MB) "
            f"by ~{over_mb:.0f} MB. Run --downgrade-maps-only or republish with --pages-lite."
        )
    else:
        headroom = stats["pages_limit_mb"] - stats["total_mb"]
        print(f"[OK] Within GitHub Pages limit ({headroom:.0f} MB headroom)")


def _copy_assets(
    assets_src: Path,
    dest_assets: Path,
    *,
    pages_lite: bool,
) -> int:
    """Copy assets/; return number of files copied."""
    if dest_assets.exists():
        shutil.rmtree(dest_assets)
    dest_assets.mkdir(parents=True)
    n_copied = 0
    for src_file in sorted(assets_src.iterdir()):
        if not src_file.is_file():
            continue
        dest_file = dest_assets / src_file.name
        if pages_lite and _is_map_asset(src_file.name):
            downgrade_map_png(src_file, dest_file)
        else:
            shutil.copy2(src_file, dest_file)
        n_copied += 1
    return n_copied


def publish_bundle(
    *,
    source_dir: Path,
    dest_dir: Path,
    title: str | None = None,
    pages_lite: bool = False,
) -> Path:
    """Copy HTML + assets into dest_dir/dashboard.html (+ assets/)."""
    html_src, assets_src = _resolve_html_and_assets(source_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_html = dest_dir / "dashboard.html"
    shutil.copy2(html_src, dest_html)

    dest_assets = dest_dir / "assets"
    if assets_src and assets_src.is_dir():
        n_assets = _copy_assets(assets_src, dest_assets, pages_lite=pages_lite)
        if pages_lite:
            print(f"[INFO] pages-lite: copied {n_assets} asset(s) (maps downscaled to {PAGES_MAP_SCALE:.0%})")
    elif dest_assets.exists():
        shutil.rmtree(dest_assets)

    if title:
        readme = dest_dir / "README.txt"
        readme.write_text(f"{title}\nSource: {source_dir.resolve()}\n", encoding="utf-8")

    return dest_html


def _readme_title(readme_path: Path) -> str | None:
    if not readme_path.is_file():
        return None
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.lower().startswith("source:"):
            return line
    return None


def _label_from_slug(slug: str, readme_path: Path | None = None) -> tuple[str, str, str | None]:
    """Return (title, subtitle, country_code)."""
    if readme_path:
        title = _readme_title(readme_path)
        if title:
            # Still parse country from slug for badge when possible.
            m = re.match(
                r"^([a-z]{2})_walk_forward_(.+?)(?:_v\d+)?$", slug, flags=re.IGNORECASE
            )
            cc = m.group(1).upper() if m else None
            return title, _subtitle_from_slug(slug), cc

    m = re.match(r"^([a-z]{2})_walk_forward_(.+?)(?:_v\d+)?$", slug, flags=re.IGNORECASE)
    if m:
        cc = m.group(1).lower()
        country = _COUNTRY_NAMES.get(cc, cc.upper())
        horizon_key = m.group(2).lower()
        horizon = _HORIZON_LABELS.get(horizon_key, horizon_key.replace("_", " ").title())
        return f"{country} walk-forward", f"{horizon} · v{_version_from_slug(slug)}", cc.upper()

    title = slug.replace("_", " ").strip().title()
    return title, "Benchmark dashboard", None


def _version_from_slug(slug: str) -> str:
    m = re.search(r"_v(\d+)$", slug, flags=re.IGNORECASE)
    return m.group(1) if m else "1"


def _subtitle_from_slug(slug: str) -> str:
    m = re.match(r"^[a-z]{2}_walk_forward_(.+?)(?:_v\d+)?$", slug, flags=re.IGNORECASE)
    if not m:
        return "Walk-forward evaluation"
    horizon_key = m.group(1).lower()
    horizon = _HORIZON_LABELS.get(horizon_key, horizon_key.replace("_", " ").title())
    ver = _version_from_slug(slug)
    return f"{horizon} · walk-forward · v{ver}"


def discover_index_entries(publish_root: Path) -> list[IndexEntry]:
    """Discover published dashboards under publish_root."""
    entries: list[IndexEntry] = []
    if (publish_root / "dashboard.html").is_file():
        entries.append(
            IndexEntry(
                href="dashboard.html",
                slug="screening",
                title="Global screening",
                subtitle="All countries · baseline models",
                country_code=None,
                kind="screening",
            )
        )
    for child in sorted(publish_root.iterdir()):
        if not child.is_dir() or child.name in {"assets", ".git"}:
            continue
        if not (child / "dashboard.html").is_file():
            continue
        slug = child.name
        title, subtitle, cc = _label_from_slug(slug, child / "README.txt")
        entries.append(
            IndexEntry(
                href=f"{slug}/dashboard.html",
                slug=slug,
                title=title,
                subtitle=subtitle,
                country_code=cc,
                kind="walk_forward",
            )
        )
    return entries


def build_index_html(entries: list[IndexEntry], *, publish_root: Path | None = None) -> str:
    """Landing page with clickable world map."""
    from cybench.runs.analysis.index_map_lib import (
        build_index_map_html,
        build_index_map_payload,
        ensure_world_geojson,
    )

    if publish_root is None:
        raise ValueError("publish_root is required to build the map index")

    payload = build_index_map_payload(entries, publish_root=publish_root)
    geojson_href = ensure_world_geojson(publish_root)
    return build_index_map_html(payload, geojson_href=geojson_href)


def build_index_html_cards(entries: list[IndexEntry], *, publish_root: Path | None = None) -> str:
    """Legacy card-grid index (kept for reference / fallback)."""
    sections: dict[str, list[IndexEntry]] = {"walk_forward": [], "screening": []}
    for entry in entries:
        sections.setdefault(entry.kind, []).append(entry)

    def render_cards(items: list[IndexEntry]) -> str:
        if not items:
            return '<p class="muted">No dashboards in this section yet.</p>'
        cards = []
        for e in items:
            badge = (
                f'<span class="badge">{html.escape(e.country_code)}</span>'
                if e.country_code
                else '<span class="badge badge-neutral">ALL</span>'
            )
            cards.append(
                f"""<a class="link-card" href="{html.escape(e.href)}">
  <div class="link-card-top">{badge}<span class="arrow" aria-hidden="true">→</span></div>
  <h3>{html.escape(e.title)}</h3>
  <p class="muted">{html.escape(e.subtitle)}</p>
</a>"""
            )
        return f'<div class="grid">{"".join(cards)}</div>'

    walk_forward_html = render_cards(sections.get("walk_forward", []))
    screening_html = render_cards(sections.get("screening", []))
    insights_card = ""
    if publish_root and (publish_root / "insights.html").is_file():
        insights_card = """
  <section class="block">
    <h2>Global benchmarks</h2>
    <div class="grid">
      <a class="link-card" href="insights.html">
        <div class="link-card-top"><span class="badge badge-neutral">ALL</span><span class="arrow" aria-hidden="true">→</span></div>
        <h3>Global insights</h3>
        <p class="muted">All models — leaderboard, model×country heatmap, end-of-season vs mid-season</p>
      </a>"""
        if (publish_root / "model_families.html").is_file():
            insights_card += """
      <a class="link-card" href="model_families.html">
        <div class="link-card-top"><span class="badge badge-neutral">ALL</span><span class="arrow" aria-hidden="true">→</span></div>
        <h3>Model families (paper summary)</h3>
        <p class="muted">Five paradigms — radar chart, median metrics, performance vs training size</p>
      </a>"""
        insights_card += """
    </div>
  </section>"""
    screening_section = ""
    if sections.get("screening"):
        screening_section = f"""
  <section class="block">
    <h2>Screening</h2>
    {screening_html}
  </section>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CY-Bench dashboards</title>
  <style>
    :root {{
      --bg: #f6f8fa;
      --card: #fff;
      --border: #d8dee4;
      --text: #1f2328;
      --muted: #656d76;
      --accent: #0969da;
      --accent-soft: #ddf4ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      margin: 0;
      padding: 1.25rem;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .page {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ margin: 0; font-size: 1.5rem; font-weight: 600; }}
    h2 {{ margin: 0 0 0.85rem; font-size: 1.05rem; font-weight: 600; }}
    .lead {{ margin: 0.35rem 0 1.25rem; color: var(--muted); }}
    .block + .block {{ margin-top: 1.5rem; }}
    .muted {{ color: var(--muted); font-size: 0.9rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 0.85rem;
    }}
    .link-card {{
      display: block;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.95rem 1rem;
      text-decoration: none;
      color: inherit;
      box-shadow: 0 1px 2px rgba(31, 35, 40, 0.06);
      transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
    }}
    .link-card:hover {{
      border-color: var(--accent);
      box-shadow: 0 4px 12px rgba(9, 105, 218, 0.12);
      transform: translateY(-1px);
    }}
    .link-card h3 {{
      margin: 0.35rem 0 0.25rem;
      font-size: 1rem;
      font-weight: 600;
      color: var(--text);
    }}
    .link-card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 0.15rem 0.45rem;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .badge-neutral {{
      background: #f3f4f6;
      color: var(--muted);
    }}
    .arrow {{ color: var(--muted); font-size: 1.1rem; }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>CY-Bench dashboards</h1>
      <p class="lead">Published walk-forward evaluation dashboards (interactive metrics + maps).</p>
    </header>{insights_card}
    <section class="block">
      <h2>Walk-forward</h2>
      {walk_forward_html}
    </section>{screening_section}
  </div>
</body>
</html>
"""


def update_index(publish_root: Path, entries: list[IndexEntry]) -> Path:
    index_path = publish_root / "index.html"
    index_path.write_text(
        build_index_html(entries, publish_root=publish_root), encoding="utf-8"
    )
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory with compare_models.html (or dashboard.html) + assets/",
    )
    parser.add_argument(
        "--publish-root",
        type=Path,
        required=True,
        help="Root of the GitHub Pages repo clone",
    )
    parser.add_argument(
        "--slug",
        type=Path,
        help="Subfolder under publish-root (e.g. de_walk_forward_mid_v1)",
    )
    parser.add_argument(
        "--title",
        help="Optional label stored in dest README.txt",
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Regenerate publish-root/index.html from published folders",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only rebuild index.html (no bundle copy)",
    )
    parser.add_argument(
        "--prune-obsolete",
        action="store_true",
        help="Remove older vN dashboard folders for the same country×horizon before updating index",
    )
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only prune obsolete dashboard folders (no bundle copy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prune actions without deleting",
    )
    parser.add_argument(
        "--pages-lite",
        action="store_true",
        help="Downscale map PNGs when copying assets (keeps all panels; fits GitHub Pages 1 GB limit)",
    )
    parser.add_argument(
        "--downgrade-maps-only",
        action="store_true",
        help="Downscale map PNGs in existing published dashboards, then report size",
    )
    parser.add_argument(
        "--strip-maps-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--report-size",
        action="store_true",
        help="Print publish-root size breakdown and exit",
    )
    args = parser.parse_args()

    publish_root = args.publish_root.resolve()
    publish_root.mkdir(parents=True, exist_ok=True)

    if args.report_size:
        report_publish_bundle_size(publish_root)
        return

    if args.downgrade_maps_only or args.strip_maps_only:
        apply_pages_lite_to_publish_root(publish_root, dry_run=args.dry_run)
        if not args.dry_run:
            report_publish_bundle_size(publish_root)
        return

    if args.prune_only:
        removed = prune_obsolete_dashboard_dirs(publish_root, dry_run=args.dry_run)
        print(f"[DONE] Pruned {len(removed)} obsolete folder(s)")
        if not args.dry_run:
            index_path = update_index(publish_root, discover_index_entries(publish_root))
            print(f"[DONE] Index: {index_path}")
        return

    if args.index_only:
        if not args.update_index:
            args.update_index = True
    elif not args.source_dir or not args.slug:
        parser.error("--source-dir and --slug are required unless --index-only or --prune-only")

    if not args.index_only:
        source_dir = args.source_dir.resolve()
        dest_dir = publish_root / args.slug
        html_path = publish_bundle(
            source_dir=source_dir,
            dest_dir=dest_dir,
            title=args.title,
            pages_lite=args.pages_lite,
        )
        print(f"[DONE] Bundle: {html_path}")
        if (dest_dir / "assets").is_dir():
            n_assets = sum(1 for _ in (dest_dir / "assets").glob("*"))
            print(f"[DONE] Assets: {dest_dir / 'assets'} ({n_assets} files)")

    if args.prune_obsolete:
        removed = prune_obsolete_dashboard_dirs(publish_root, dry_run=args.dry_run)
        if removed:
            print(f"[INFO] Pruned {len(removed)} obsolete folder(s)")

    if args.update_index:
        index_path = update_index(publish_root, discover_index_entries(publish_root))
        print(f"[DONE] Index: {index_path}")


if __name__ == "__main__":
    main()
