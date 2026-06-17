"""Tests for benchmark submit planning."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.slurm.benchmark_submit_lib import (
    batch_name,
    build_submit_plans,
    filed_batches,
    horizon_batch_suffix,
    normalize_horizon,
)


def test_normalize_horizon():
    assert normalize_horizon("eos") == "eos"
    assert normalize_horizon("mid") == "middle-of-season"
    assert normalize_horizon("middle-of-season") == "middle-of-season"


def test_batch_name():
    assert batch_name("de", "eos") == "baselines_DE_eos_v1"
    assert batch_name("pl", "mid") == "baselines_PL_mid_v1"
    assert horizon_batch_suffix("middle-of-season") == "mid"


def test_filed_batches(tmp_path: Path):
    (tmp_path / "baselines_DE_eos_v1").mkdir()
    (tmp_path / "baselines_FR_mid_v1").mkdir()
    (tmp_path / "other").mkdir()
    assert filed_batches(tmp_path) == {
        ("DE", "eos", 1),
        ("FR", "mid", 1),
    }


def test_build_submit_plans_skip_filed(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    manifest = tmp_path / "manifests"
    (manifest / "baselines_AT_eos_v1").mkdir(parents=True)
    (data / "maize" / "BE").mkdir(parents=True)
    yield_path = data / "maize" / "BE" / "yield_maize_BE.csv"
    yield_path.write_text(
        "crop_name,country_code,adm_id,harvest_year,yield\n"
        "maize,BE,BE1,2020,10.0\n"
        "maize,BE,BE2,2020,11.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH_DATA_DIR", str(data))
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    plans = build_submit_plans(
        countries=["BE", "AT"],
        horizons=["eos"],
        manifest_root=manifest,
        data_dir=data,
        region_threshold=100,
    )
    by_cc = {p.country_upper: p for p in plans}
    assert by_cc["AT"].skip is True
    assert by_cc["BE"].skip is False
    assert by_cc["BE"].gpu_partition is False  # 2 regions < 100
