"""Tests for dashboard publish pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.analysis.publish_pipeline_lib import (
    PublishTarget,
    _write_json,
    collect_state_path,
    discover_baselines_batches,
    filter_publish_targets,
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
    assert horizon_to_batch_suffix("quarter-of-season") == "qtr"
    assert horizon_to_batch_suffix("qtr") == "qtr"


def test_parse_batch_dir_name():
    assert parse_batch_dir_name("baselines_de_mid_v1") == ("de", "mid", 1)
    assert parse_batch_dir_name("baselines_FR_eos_v1") == ("FR", "eos", 1)
    assert parse_batch_dir_name("baselines_SK_qtr_v2") == ("SK", "qtr", 2)
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

    qtr = PublishTarget(
        country="DE",
        batch_horizon="qtr",
        version=2,
        output_root=Path("/tmp/output"),
        publish_root=Path("/tmp/publish"),
    )
    assert qtr.batch_name == "baselines_DE_qtr_v2"
    assert qtr.collect_dir == Path("/tmp/output/paper_walk_forward_de_qtr_v2")
    assert "quarter-season" in qtr.default_title().lower()


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
    (tmp_path / "baselines_DE_qtr_v2").mkdir()
    targets = discover_baselines_batches(tmp_path)
    assert len(targets) == 2
    assert {t.country_upper for t in targets} == {"AR", "DE"}
    assert {t.batch_horizon for t in targets} == {"eos", "qtr"}


def test_filter_publish_targets_by_version(tmp_path: Path):
    targets = [
        PublishTarget(country="DE", batch_horizon="eos", version=1, output_root=tmp_path),
        PublishTarget(country="DE", batch_horizon="eos", version=2, output_root=tmp_path),
        PublishTarget(country="FR", batch_horizon="eos", version=2, output_root=tmp_path),
    ]
    filtered = filter_publish_targets(targets, countries=["DE"], version=2)
    assert len(filtered) == 1
    assert filtered[0].batch_name == "baselines_DE_eos_v2"


def test_filter_publish_targets_keeps_latest_version_by_default(tmp_path: Path):
    targets = [
        PublishTarget(country="DE", batch_horizon="eos", version=1, output_root=tmp_path),
        PublishTarget(country="DE", batch_horizon="eos", version=2, output_root=tmp_path),
        PublishTarget(country="DE", batch_horizon="mid", version=1, output_root=tmp_path),
        PublishTarget(country="DE", batch_horizon="mid", version=2, output_root=tmp_path),
    ]
    filtered = filter_publish_targets(targets, keep_latest_version=True)
    assert {t.batch_name for t in filtered} == {
        "baselines_DE_eos_v2",
        "baselines_DE_mid_v2",
    }
    all_versions = filter_publish_targets(targets, keep_latest_version=False)
    assert len(all_versions) == 4


def test_discover_baselines_batches_from_monolithic(tmp_path: Path):
    import pandas as pd

    from cybench.runs.analysis.publish_pipeline_lib import (
        PublishTarget,
        resolve_collect_baselines_dir,
    )

    mono = tmp_path / "baselines"
    run_dir = mono / "maize_AO_ridge_walk_forward_eos_20260101_120000" / "2016" / "42"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "adm_id": ["AO-1"],
            "year": [2016],
            "targets": [10.0],
            "preds": [9.5],
        }
    ).to_csv(run_dir / "test_preds.csv", index=False)

    targets = discover_baselines_batches(tmp_path)
    assert any(t.country_upper == "AO" and t.batch_horizon == "eos" for t in targets)

    target = PublishTarget(
        country="AO",
        batch_horizon="eos",
        version=1,
        output_root=tmp_path,
    )
    baselines_dir, note = resolve_collect_baselines_dir(target)
    assert baselines_dir == mono
    assert note is not None
