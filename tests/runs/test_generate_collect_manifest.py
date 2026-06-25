"""Tests for collect manifest generation."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.analysis.generate_collect_manifest import _resolve_targets_fast
from cybench.runs.analysis.publish_pipeline_lib import PipelineDefaults


def test_resolve_targets_fast_filters_version(tmp_path: Path):
    (tmp_path / "baselines_DE_eos_v1").mkdir()
    (tmp_path / "baselines_DE_eos_v2").mkdir()
    defaults = PipelineDefaults(output_root=tmp_path)

    all_targets = _resolve_targets_fast(
        mode="all-available",
        config_path=None,
        defaults=defaults,
        countries=["DE"],
        horizons=["eos"],
        version=None,
    )
    assert {t.version for t in all_targets} == {1, 2}

    v2_only = _resolve_targets_fast(
        mode="all-available",
        config_path=None,
        defaults=defaults,
        countries=["DE"],
        horizons=["eos"],
        version=2,
    )
    assert len(v2_only) == 1
    assert v2_only[0].batch_name == "baselines_DE_eos_v2"
