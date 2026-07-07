"""Tests for dashboard bundle publish helpers."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.analysis.publish_dashboard_bundle import (
    estimate_publish_bundle_size,
    parse_publish_slug,
    prune_obsolete_dashboard_dirs,
    publish_bundle,
)


def test_parse_publish_slug():
    assert parse_publish_slug("de_walk_forward_mid_v2") == ("DE", "mid", 2)
    assert parse_publish_slug("us_walk_forward_qtr_v2") == ("US", "qtr", 2)
    assert parse_publish_slug("insights.html") is None


def test_prune_obsolete_dashboard_dirs(tmp_path: Path):
    root = tmp_path / "publish"
    root.mkdir()
    for name in ("de_walk_forward_eos_v1", "de_walk_forward_eos_v2", "de_walk_forward_mid_v2"):
        d = root / name
        d.mkdir()
        (d / "dashboard.html").write_text("<html></html>", encoding="utf-8")
    removed = prune_obsolete_dashboard_dirs(root)
    assert [p.name for p in removed] == ["de_walk_forward_eos_v1"]
    assert (root / "de_walk_forward_eos_v2").is_dir()
    assert not (root / "de_walk_forward_eos_v1").exists()


def _write_test_png(path: Path, *, size: tuple[int, int]) -> None:
    from PIL import Image

    Image.new("RGB", size, color=(120, 80, 40)).save(path, format="PNG")


def test_publish_bundle_copies_assets(tmp_path: Path):
    source = tmp_path / "source"
    assets = source / "assets"
    assets.mkdir(parents=True)
    (source / "compare_models.html").write_text("<html></html>", encoding="utf-8")
    _write_test_png(assets / "maize_scatter.png", size=(400, 300))
    (assets / "regions_DE.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    dest = tmp_path / "publish" / "de_walk_forward_eos_v2"
    publish_bundle(source_dir=source, dest_dir=dest)
    copied = {p.name for p in (dest / "assets").iterdir()}
    assert copied == {"maize_scatter.png", "regions_DE.geojson"}
    assert (dest / "dashboard.html").is_file()


def test_estimate_publish_bundle_size(tmp_path: Path):
    root = tmp_path / "publish"
    d = root / "de_walk_forward_eos_v2"
    (d / "assets").mkdir(parents=True)
    (d / "dashboard.html").write_text("x" * 1000, encoding="utf-8")
    stats = estimate_publish_bundle_size(root)
    assert stats["n_dashboard_dirs"] == 1
    assert stats["total_bytes"] >= 1000
