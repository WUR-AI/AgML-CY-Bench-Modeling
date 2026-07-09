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
    assert normalize_horizon("early") == "early-season"
    assert horizon_batch_suffix("early") == "early"


def test_batch_name():
    assert batch_name("de", "eos") == "baselines_DE_eos_v3"
    assert batch_name("pl", "mid") == "baselines_PL_mid_v3"
    assert batch_name("de", "eos", version=1) == "baselines_DE_eos_v1"
    assert horizon_batch_suffix("middle-of-season") == "mid"
    assert horizon_batch_suffix("quarter-of-season") == "qtr"
    assert horizon_batch_suffix("qtr") == "qtr"


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


def test_resolve_batch_dir_and_parse_batch_name(tmp_path: Path):
    from cybench.runs.slurm.benchmark_submit_lib import parse_batch_name, resolve_batch_dir

    assert parse_batch_name("baselines_de_eos_v1") == ("DE", "eos", 1)
    assert parse_batch_name("baselines_SK_qtr_v2") == ("SK", "qtr", 2)
    assert parse_batch_name("baselines_SK_qtr_v2") == ("SK", "qtr", 2)
    output = tmp_path / "out"
    (output / "baselines_fr_mid_v1").mkdir(parents=True)
    path, note = resolve_batch_dir(output, "baselines_FR_mid_v1")
    assert path.name == "baselines_fr_mid_v1"
    assert note is not None


def test_gpu_region_threshold_default_routes_fr_to_cpu(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm import benchmark_submit_lib as lib

    assert lib.DEFAULT_GPU_REGION_THRESHOLD == 600

    def _fake_count(country: str, data_dir=None) -> int:
        return {"FR": 100, "DE": 397, "IN": 557, "US": 2509}[country.upper()]

    monkeypatch.setattr(lib, "count_regions", _fake_count)
    assert lib.gpu_partition_for_country("FR")[0] is False
    assert lib.gpu_partition_for_country("DE")[0] is False
    assert lib.gpu_partition_for_country("IN")[0] is False
    assert lib.gpu_partition_for_country("US")[0] is True


def test_expand_all_country_targets():
    from cybench.runs.slurm.benchmark_completion_lib import expand_all_country_targets

    targets = expand_all_country_targets(
        countries=["DE", "FR"],
        horizons=["eos", "mid"],
        version=1,
    )
    names = [b for b, _ in targets]
    assert "baselines_DE_eos_v1" in names
    assert "baselines_FR_mid_v1" in names
    assert len(targets) == 4
