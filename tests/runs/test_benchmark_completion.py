"""Tests for benchmark completion / retry planning."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.slurm.benchmark_completion_lib import (
    JobRow,
    assess_job,
    check_screening_years,
    jobs_for_phase,
    read_manifest,
    screening_complete,
    write_manifest,
)


def _write_yield(path: Path, crop: str, country: str, years: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["crop_name,country_code,adm_id,harvest_year,yield"]
    for year in years:
        lines.append(f"{crop},{country},R1,{year},10.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_check_screening_years_blocks_short_series():
    ok, reason = check_screening_years({2018, 2019, 2020})
    assert not ok
    assert reason


def test_check_screening_years_accepts_long_series():
    years = set(range(2000, 2012))
    ok, reason = check_screening_years(years)
    assert ok
    assert "train=" in reason


def test_assess_job_blocked_when_too_few_years(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    _write_yield(data / "maize" / "BE" / "yield_maize_BE.csv", "maize", "BE", [2020, 2021])
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))
    job = JobRow("maize", "BE", "ridge", "pandas", "yes", "yes", "no")
    assessment = assess_job(
        job,
        baselines_dir=tmp_path / "output",
        horizon="eos",
        repo_root=Path(__file__).resolve().parents[2],
        data_dir=data,
    )
    assert assessment.blocked
    assert not assessment.needs_screening
    assert not assessment.needs_walk_forward
    assert assessment.block_reason


def test_screening_complete_detects_optimal_model(tmp_path: Path):
    baselines = tmp_path / "baselines"
    run_dir = baselines / "maize_BE_ridge_screening_eos_20260101_120000" / "2019_2020"
    run_dir.mkdir(parents=True)
    (run_dir / "optimal_model.yaml").write_text("model: x\n", encoding="utf-8")
    job = JobRow("maize", "BE", "ridge", "pandas", "yes", "yes", "no")
    ok, reason = screening_complete(
        baselines, job, horizon_tag_value="eos", repo_root=Path(__file__).resolve().parents[2]
    )
    assert ok
    assert "screening_eos" in reason


def test_jobs_for_phase_walk_forward_only(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    years = list(range(2000, 2012))
    _write_yield(data / "maize" / "BE" / "yield_maize_BE.csv", "maize", "BE", years)
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    baselines = tmp_path / "baselines"
    scr_dir = baselines / "maize_BE_ridge_screening_eos_20260101_120000" / "2019_2020"
    scr_dir.mkdir(parents=True)
    (scr_dir / "optimal_model.yaml").write_text("x: 1\n", encoding="utf-8")

    job = JobRow("maize", "BE", "ridge", "pandas", "yes", "yes", "no")
    assessment = assess_job(
        job,
        baselines_dir=baselines,
        horizon="eos",
        repo_root=Path(__file__).resolve().parents[2],
        data_dir=data,
    )
    assert assessment.screening_ok
    assert not assessment.walk_forward_ok
    assert assessment.needs_walk_forward
    assert not assessment.needs_screening

    wf_jobs = jobs_for_phase([assessment], "walk_forward")
    assert wf_jobs == [job]


def test_read_write_manifest_roundtrip(tmp_path: Path):
    path = tmp_path / "jobs.txt"
    job = JobRow("maize", "US", "ridge", "pandas", "yes", "yes", "no")
    write_manifest(path, [job])
    assert read_manifest(path) == [job]


def test_parse_batch_name_and_expand():
    from cybench.runs.slurm.benchmark_completion_lib import expand_target_batches
    from cybench.runs.slurm.benchmark_submit_lib import parse_batch_name

    assert parse_batch_name("baselines_DE_eos_v1") == ("DE", "eos", 1)
    assert parse_batch_name("baselines") is None

    one = expand_target_batches(
        batch="baselines_DE_eos_v1", country=None, horizons=["eos"], version=1
    )
    assert one == [("baselines_DE_eos_v1", "eos")]

    both = expand_target_batches(
        batch="baselines_DE_eos_v1", country=None, horizons=["eos", "mid"], version=1
    )
    assert [b for b, _ in both] == ["baselines_DE_eos_v1", "baselines_DE_mid_v1"]

    by_cc = expand_target_batches(
        batch=None, country="DE", horizons=["eos", "mid"], version=1
    )
    assert [b for b, _ in by_cc] == ["baselines_DE_eos_v1", "baselines_DE_mid_v1"]


def test_ensure_manifest_filters_shared(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm.benchmark_completion_lib import ensure_manifest

    repo = tmp_path / "repo"
    slurm = repo / "cybench" / "runs" / "slurm"
    slurm.mkdir(parents=True)
    shared = slurm / "benchmark_jobs.txt"
    shared.write_text(
        "# header\n"
        "maize DE ridge pandas yes yes no\n"
        "maize FR ridge pandas yes yes no\n",
        encoding="utf-8",
    )
    path, jobs, source = ensure_manifest(
        batch="baselines_DE_eos_v1",
        repo_root=repo,
        manifest_path=None,
    )
    assert len(jobs) == 1
    assert jobs[0].country == "DE"
    assert "filtered" in source
    assert path.is_file()


def test_gpu_partition_for_batch_small_country(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm.benchmark_submit_lib import gpu_partition_for_batch

    data = tmp_path / "data"
    _write_yield(data / "maize" / "AT" / "yield_maize_AT.csv", "maize", "AT", list(range(2000, 2012)))
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    use_gpu, n_regions, country = gpu_partition_for_batch(
        "baselines_AT_eos_v1",
        region_threshold=100,
        data_dir=data,
    )
    assert country == "AT"
    assert n_regions == 1
    assert use_gpu is False


def test_resolve_force_cpu_auto_routes_small_country(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm.orchestrate_benchmark_complete import _resolve_force_cpu

    data = tmp_path / "data"
    _write_yield(data / "maize" / "AT" / "yield_maize_AT.csv", "maize", "AT", list(range(2000, 2012)))
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    force_cpu, reason = _resolve_force_cpu(
        batch="baselines_AT_eos_v1",
        explicit_cpu=False,
        force_gpu=False,
        region_threshold=100,
        data_dir=data,
    )
    assert force_cpu is True
    assert reason is not None
    assert "AT" in reason


def test_resolve_batch_dir_case_insensitive(tmp_path: Path):
    from cybench.runs.slurm.benchmark_submit_lib import resolve_batch_dir

    output = tmp_path / "output"
    (output / "baselines_de_eos_v1").mkdir(parents=True)
    resolved, note = resolve_batch_dir(output, "baselines_DE_eos_v1")
    assert resolved.name == "baselines_de_eos_v1"
    assert note is not None
