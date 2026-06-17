"""Tests for generate_job_manifest.py."""

from __future__ import annotations

from pathlib import Path

from cybench.runs.slurm.generate_job_manifest import generate


def test_countries_filter_skips_crop_without_country(
    tmp_path: Path, monkeypatch
):
    data = tmp_path / "data"
    (data / "maize" / "AO").mkdir(parents=True)
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
