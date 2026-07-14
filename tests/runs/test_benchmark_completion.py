"""Tests for benchmark completion / retry planning."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.slurm.benchmark_completion_lib import (
    JobRow,
    assess_job,
    check_screening_years,
    filter_jobs_by_models,
    filter_jobs_cpu_only,
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
    assert jobs_for_phase([assessment], "walk_forward", force_rerun=True) == [job]


def test_filter_jobs_by_models():
    jobs = [
        JobRow("maize", "US", "ridge", "pandas", "yes", "yes", "no"),
        JobRow("maize", "US", "lpjml_bc", "pandas", "no", "no", "no"),
        JobRow("wheat", "DE", "lpjml_bc", "pandas", "no", "no", "no"),
    ]
    assert filter_jobs_by_models(jobs, ["lpjml_bc"]) == jobs[1:]
    assert filter_jobs_by_models(jobs, None) == jobs


def test_filter_jobs_cpu_only():
    jobs = [
        JobRow("maize", "DE", "xgboost", "pandas", "yes", "yes", "no"),
        JobRow("maize", "DE", "lpjml_bc", "pandas", "no", "no", "no"),
        JobRow("maize", "DE", "tst_lf", "torch", "yes", "no", "yes"),
        JobRow("wheat", "DE", "tabdpt", "pandas", "yes", "yes", "yes"),
    ]
    cpu = filter_jobs_cpu_only(jobs)
    assert cpu == jobs[:2]


def test_jobs_for_phase_force_rerun_includes_complete():
    job = JobRow("maize", "BE", "lpjml_bc", "pandas", "no", "no", "no")
    from cybench.runs.slurm.benchmark_completion_lib import JobAssessment

    assessment = JobAssessment(
        job=job,
        n_years=12,
        screening_ok=True,
        screening_reason="ok",
        walk_forward_ok=True,
        walk_forward_reason="done",
        blocked=False,
        block_reason="",
    )
    assert jobs_for_phase([assessment], "walk_forward") == []
    assert jobs_for_phase([assessment], "walk_forward", force_rerun=True) == [job]


def test_read_write_manifest_roundtrip(tmp_path: Path):
    path = tmp_path / "jobs.txt"
    job = JobRow("maize", "US", "ridge", "pandas", "yes", "yes", "no")
    write_manifest(path, [job])
    assert read_manifest(path) == [job]


def test_merge_jobs_supplements_stale_manifest(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm.benchmark_completion_lib import JobRow, merge_jobs_for_models

    repo = tmp_path / "repo"
    data = repo / "cybench" / "data"
    years = list(range(2000, 2012))
    _write_yield(data / "wheat" / "US" / "yield_wheat_US.csv", "wheat", "US", years)
    (data / "wheat" / "US" / "twso_wheat_US.csv").write_text(
        "crop_name,adm_id,date,twso\nwheat,US-01,20010301,1.0\n",
        encoding="utf-8",
    )
    slurm = repo / "cybench" / "runs" / "slurm"
    slurm.mkdir(parents=True)
    (slurm / "models.txt").write_text("twso_bc pandas no no no\n", encoding="utf-8")

    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    stale = [JobRow("maize", "US", "ridge", "pandas", "yes", "yes", "no")]
    jobs, supplemented = merge_jobs_for_models(
        stale,
        country="US",
        models=["twso_bc"],
        repo_root=repo,
        data_dir=data,
        models_path=slurm / "models.txt",
    )
    assert supplemented
    assert len(jobs) == 1
    assert jobs[0].model == "twso_bc"
    assert jobs[0].crop == "wheat"
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


def test_expand_walk_forward_gpu_per_seed(tmp_path: Path):
    from cybench.runs.analysis.collect_walk_forward_results import discover_run_seeds
    from cybench.runs.slurm.benchmark_completion_lib import (
        JobRow,
        expand_walk_forward_manifest_lines,
    )

    repo = tmp_path / "repo"
    baselines = tmp_path / "output" / "baselines_DE_eos_v2"
    run_dir = baselines / "maize_DE_lstm_lf_walk_forward_eos_20260101_120000"
    (run_dir / "2016" / "42").mkdir(parents=True)
    (run_dir / "2016" / "42" / "test_preds.csv").write_text(
        "adm_id,year,targets,preds\nDE-01,2016,10,9\n", encoding="utf-8"
    )
    assert discover_run_seeds(run_dir) == [42]

    gpu_job = JobRow("maize", "DE", "lstm_lf", "torch", "yes", "no", "yes")
    cpu_job = JobRow("maize", "DE", "ridge", "pandas", "yes", "yes", "no")

    lines = expand_walk_forward_manifest_lines(
        [gpu_job, cpu_job],
        baselines_dir=baselines,
        horizon="eos",
        repo_root=repo,
        base_seed=42,
        total_repetitions=5,
        resume=True,
        per_seed_for_gpu=True,
    )
    assert "maize DE lstm_lf torch yes no yes 43" in lines
    assert "maize DE lstm_lf torch yes no yes 46" in lines
    assert not any(line.endswith(" 42") for line in lines)
    assert sum(1 for line in lines if "lstm_lf" in line) == 4
    assert sum(1 for line in lines if "ridge" in line) == 1
    assert all("ridge" not in line or line.count(" ") == 6 for line in lines if "ridge" in line)


def test_walk_forward_complete_requires_all_repetitions(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm.benchmark_completion_lib import (
        JobRow,
        walk_forward_complete,
    )
    from cybench.util.validation import expected_walk_forward_test_years

    data = tmp_path / "data"
    years = list(range(2000, 2025))
    _write_yield(data / "maize" / "EL" / "yield_maize_EL.csv", "maize", "EL", years)
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    repo = tmp_path / "repo"
    baselines = tmp_path / "output" / "baselines_EL_early_v2"
    run_dir = baselines / "maize_EL_tst_lf_walk_forward_early_season_20260709_120000"
    pred = "adm_id,year,targets,preds\nEL-01,2019,10,9\n"
    expected_years = expected_walk_forward_test_years(set(years))
    for seed in (42, 43):
        for year in expected_years:
            d = run_dir / str(year) / str(seed)
            d.mkdir(parents=True)
            (d / "test_preds.csv").write_text(pred, encoding="utf-8")

    job = JobRow("maize", "EL", "tst_lf", "torch", "yes", "no", "yes")
    ok_any, _ = walk_forward_complete(
        baselines,
        job,
        horizon_tag_value="early_season",
        repo_root=repo,
        data_dir=data,
    )
    assert ok_any is True

    ok_all, reason = walk_forward_complete(
        baselines,
        job,
        horizon_tag_value="early_season",
        repo_root=repo,
        total_repetitions=5,
        data_dir=data,
    )
    assert ok_all is False
    assert "missing walk-forward seeds" in reason
    assert "[44, 45, 46]" in reason


def test_walk_forward_complete_requires_all_test_years(tmp_path: Path, monkeypatch):
    from cybench.runs.slurm.benchmark_completion_lib import (
        JobRow,
        walk_forward_complete,
    )
    from cybench.util.validation import expected_walk_forward_test_years

    data = tmp_path / "data"
    years = list(range(2000, 2025))
    _write_yield(data / "maize" / "EL" / "yield_maize_EL.csv", "maize", "EL", years)
    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    repo = tmp_path / "repo"
    baselines = tmp_path / "output" / "baselines_EL_early_v2"
    run_dir = baselines / "maize_EL_tst_lf_walk_forward_early_season_20260709_120000"
    pred = "adm_id,year,targets,preds\nEL-01,2019,10,9\n"
    expected_years = expected_walk_forward_test_years(set(years))
    only_year = expected_years[0]
    for seed in (42, 43, 44, 45, 46):
        d = run_dir / str(only_year) / str(seed)
        d.mkdir(parents=True)
        (d / "test_preds.csv").write_text(pred, encoding="utf-8")

    job = JobRow("maize", "EL", "tst_lf", "torch", "yes", "no", "yes")
    ok, reason = walk_forward_complete(
        baselines,
        job,
        horizon_tag_value="early_season",
        repo_root=repo,
        total_repetitions=5,
        data_dir=data,
    )
    assert ok is False
    assert "incomplete walk-forward years" in reason
    assert f"seed 42 missing {expected_years[1:]}" in reason
