"""Tests for Zenodo dataset fetch helper."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from data_preparation.fetch_zenodo_data import (
    DEFAULT_DOI,
    extract_geometry_zip,
    fetch,
)


def _write_geometry_zip(dest: Path, folder_name: str, country: str = "DE") -> None:
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr(f"{folder_name}/{country}/readme.txt", "stub shapefile tree")


def test_extract_geometry_zip(tmp_path: Path):
    zip_path = tmp_path / "polygons.zip"
    data_dir = tmp_path / "data"
    _write_geometry_zip(zip_path, "polygons", "DE")
    extract_geometry_zip(zip_path, data_dir, "polygons")
    assert (data_dir / "polygons" / "DE" / "readme.txt").is_file()


def test_fetch_geometries_only_skip_download(tmp_path: Path):
    staging = tmp_path / "staging"
    data_dir = tmp_path / "data"
    staging.mkdir()
    _write_geometry_zip(staging / "centroids.zip", "centroids", "NL")
    _write_geometry_zip(staging / "polygons.zip", "polygons", "NL")

    fetch(
        data_dir,
        staging_dir=staging,
        include_geometries=True,
        geometries_only=True,
        skip_download=True,
    )
    assert (data_dir / "polygons" / "NL" / "readme.txt").is_file()
    assert (data_dir / "centroids" / "NL" / "readme.txt").is_file()


def test_geometries_only_requires_geometry_zips_when_skip_download(tmp_path: Path):
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(FileNotFoundError, match="geometry zips missing"):
        fetch(
            tmp_path / "data",
            staging_dir=staging,
            include_geometries=True,
            geometries_only=True,
            skip_download=True,
        )


def test_default_doi_is_concept_record():
    assert DEFAULT_DOI == "10.5281/zenodo.11502142"
