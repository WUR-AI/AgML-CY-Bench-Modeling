"""Tests for dashboard publish pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.analysis.publish_pipeline_lib import (
    PublishTarget,
    _write_json,
    collect_state_path,
    discover_baselines_batches,
    horizon_to_batch_suffix,
    load_targets_from_config,
    needs_collect,
    needs_publish,
    parse_batch_dir_name,
)


def test_horizon_to_batch_suffix():
    assert horizon_to_batch_suffix("eos") == "eos"
    assert horizon_to_batch_suffix("middle-of-season") == "mid"
    assert horizon_to_batch_suffix("mid_season") == "mid"


def test_parse_batch_dir_name():
    assert parse_batch_dir_name("baselines_de_mid_v1") == ("de", "mid", 1)
    assert parse_batch_dir_name("baselines_FR_eos_v1") == ("FR", "eos", 1)
    assert parse_batch_dir_name("baselines") is None


def test_publish_target_names():
    target = PublishTarget(
        country="DE",
        batch_horizon="mid",
        version=1,
        output_root=Path("/tmp/output"),
        publish_root=Path("/tmp/publish"),
    )
    assert target.batch_name == "baselines_DE_mid_v1"
    assert target.collect_dir == Path("/tmp/output/paper_walk_forward_de_mid_v1")
    assert target.publish_slug == "de_walk_forward_mid_v1"
    assert "Germany" in target.default_title()


def test_needs_collect_skips_when_state_matches(tmp_path: Path):
    target = PublishTarget(
        country="DE",
        batch_horizon="eos",
        version=1,
        output_root=tmp_path,
    )
    collect_dir = target.collect_dir
    collect_dir.mkdir(parents=True)
    (collect_dir / "compare_models.html").write_text("<html></html>", encoding="utf-8")
    fingerprint = {
        "baselines_dir": "/x",
        "n_runs": 2,
        "runs": [
            {
                "dataset": "maize_DE",
                "model": "ridge",
                "horizon": "eos",
                "timestamp": "t1",
            }
        ],
    }
    _write_json(
        collect_state_path(target),
        {**fingerprint, "collected_at": "2026-01-01T00:00:00+00:00"},
    )
    should_run, reason = needs_collect(target, fingerprint=fingerprint)
    assert should_run is False
    assert "up to date" in reason


def test_needs_publish_detects_missing_dashboard(tmp_path: Path):
    target = PublishTarget(
        country="FR",
        batch_horizon="eos",
        version=1,
        output_root=tmp_path,
        publish_root=tmp_path / "publish",
    )
    should_run, reason = needs_publish(target)
    assert should_run is True
    assert "missing" in reason


def test_load_targets_discovers_batches_when_no_include(tmp_path: Path):
    (tmp_path / "baselines_DE_eos_v1").mkdir()
    (tmp_path / "baselines_nl_mid_v1").mkdir()
    cfg = tmp_path / "dashboard_targets.yaml"
    cfg.write_text(
        f"defaults:\n  output_root: {tmp_path}\n  publish_root: {tmp_path}/pub\n",
        encoding="utf-8",
    )
    targets = load_targets_from_config(cfg)
    names = {t.batch_name for t in targets}
    assert names == {"baselines_DE_eos_v1", "baselines_NL_mid_v1"}


def test_discover_baselines_batches(tmp_path: Path):
    (tmp_path / "baselines_AR_eos_v1").mkdir()
    targets = discover_baselines_batches(tmp_path)
    assert len(targets) == 1
    assert targets[0].country_upper == "AR"
