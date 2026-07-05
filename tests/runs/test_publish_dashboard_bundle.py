"""Tests for dashboard bundle publish helpers."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.analysis.publish_dashboard_bundle import (
    parse_publish_slug,
    prune_obsolete_dashboard_dirs,
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
