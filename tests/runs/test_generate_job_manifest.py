"""Tests for generate_job_manifest.py."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.slurm.generate_job_manifest import generate


def _write_yield(path: Path, crop: str, country: str, years: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["crop_name,country_code,adm_id,harvest_year,yield"]
    for year in years:
        lines.append(f"{crop},{country},R1,{year},10.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_countries_filter_skips_crop_without_country(
    tmp_path: Path, monkeypatch
):
    data = tmp_path / "data"
    _write_yield(
        data / "maize" / "AO" / "yield_maize_AO.csv",
        "maize",
        "AO",
        list(range(2000, 2012)),
    )
    models = tmp_path / "models.txt"
    models.write_text("ridge pandas yes yes no\n", encoding="utf-8")
    out = tmp_path / "jobs.txt"

    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    n = generate(
        crops=["maize", "wheat"],
        countries=["AO"],
        models_path=models,
        output=out,
    )
    assert n == 1
    text = out.read_text(encoding="utf-8")
    assert "maize AO ridge" in text
    assert "wheat" not in text


def test_skips_crop_country_with_too_few_yield_years(
    tmp_path: Path, monkeypatch
):
    data = tmp_path / "data"
    _write_yield(
        data / "maize" / "SK" / "yield_maize_SK.csv",
        "maize",
        "SK",
        list(range(2000, 2012)),
    )
    _write_yield(
        data / "wheat" / "SK" / "yield_wheat_SK.csv",
        "wheat",
        "SK",
        [2017, 2018],
    )
    models = tmp_path / "models.txt"
    models.write_text("ridge pandas yes yes no\n", encoding="utf-8")
    out = tmp_path / "jobs.txt"

    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    n = generate(
        crops=["maize", "wheat"],
        countries=["SK"],
        models_path=models,
        output=out,
    )
    assert n == 1
    text = out.read_text(encoding="utf-8")
    assert "maize SK ridge" in text
    assert "wheat SK" not in text


def test_skips_lpjml_and_twso_without_predictor_csv(tmp_path: Path, monkeypatch, capsys):
    data = tmp_path / "data"
    years = list(range(2000, 2012))
    _write_yield(data / "maize" / "US" / "yield_maize_US.csv", "maize", "US", years)
    _write_yield(data / "wheat" / "US" / "yield_wheat_US.csv", "wheat", "US", years)
    (data / "wheat" / "US" / "twso_wheat_US.csv").write_text(
        "crop_name,adm_id,date,twso\nwheat,US-01,20010301,1.0\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.txt"
    models.write_text(
        "lpjml_bc pandas no no no\n"
        "twso_bc pandas no no no\n",
        encoding="utf-8",
    )
    out = tmp_path / "jobs.txt"

    import cybench.config as cfg

    monkeypatch.setattr(cfg, "PATH_DATA_DIR", str(data))

    n = generate(
        crops=["maize", "wheat"],
        countries=["US"],
        models_path=models,
        output=out,
    )
    assert n == 1
    text = out.read_text(encoding="utf-8")
    assert "wheat US twso_bc" in text
    assert "lpjml_bc" not in text
    assert "maize US" not in text

    captured = capsys.readouterr().out
    assert "lpjml" in captured
    assert "twso" in captured
