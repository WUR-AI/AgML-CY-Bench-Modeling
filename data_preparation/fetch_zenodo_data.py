"""Download and unpack the CY-Bench dataset from Zenodo.

Uses the ``zenodo-get`` package (recommended in the Zenodo record):
https://doi.org/10.5281/zenodo.11502142
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from cybench.config import PATH_DATA_DIR

# Concept DOI — always resolves to the latest dataset version on Zenodo.
DEFAULT_DOI = "10.5281/zenodo.11502142"

CYBENCH_DATA_ZIP = "cybench-data.zip"
GEOMETRY_ZIPS = {
    "centroids": "centroids.zip",
    "polygons": "polygons.zip",
}


def _require_zenodo_get():
    try:
        from zenodo_get import zenodo_get
    except ImportError as exc:
        raise SystemExit(
            "zenodo-get is required for downloading. Install with:\n"
            "  poetry install          # Python 3.10+\n"
            "  pip install zenodo-get  # standalone\n"
        ) from exc
    return zenodo_get


def _zenodo_download_file(doi: str, staging_dir: Path, file_glob: str) -> None:
    """Download one file (or glob match) from a Zenodo record via zenodo-get."""
    zenodo_get = _require_zenodo_get()
    zenodo_get(["-d", doi, "-o", str(staging_dir), "-g", file_glob])


def _merge_tree(src: Path, dst: Path) -> None:
    """Copy ``src`` into ``dst``, overwriting existing files."""
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _find_child_dir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = [p for p in root.rglob(name) if p.is_dir() and p.name == name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the shallowest match (top-level folder inside the zip).
        return min(matches, key=lambda p: len(p.parts))
    return None


def extract_cybench_data(zip_path: Path, data_dir: Path) -> None:
    """Unpack ``cybench-data.zip`` into ``data_dir`` (``maize/``, ``wheat/``)."""
    with tempfile.TemporaryDirectory(prefix="cybench-data-") as tmp_name:
        tmp = Path(tmp_name)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)

        for crop in ("maize", "wheat"):
            crop_src = _find_child_dir(tmp, crop)
            if crop_src is None:
                continue
            _merge_tree(crop_src, data_dir / crop)
            print(f"Merged {crop_src} -> {data_dir / crop}")


def extract_geometry_zip(zip_path: Path, data_dir: Path, folder_name: str) -> None:
    """Unpack ``centroids.zip`` or ``polygons.zip`` into ``data_dir/<folder>/``."""
    with tempfile.TemporaryDirectory(prefix=f"cybench-{folder_name}-") as tmp_name:
        tmp = Path(tmp_name)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)

        src = _find_child_dir(tmp, folder_name)
        if src is None:
            raise FileNotFoundError(
                f"Could not find '{folder_name}/' inside {zip_path.name}"
            )
        _merge_tree(src, data_dir / folder_name)
        print(f"Merged {src} -> {data_dir / folder_name}")


def download_files(
    doi: str,
    staging_dir: Path,
    *,
    include_geometries: bool,
    geometries_only: bool = False,
) -> list[Path]:
    globs: list[str] = []
    if geometries_only:
        if not include_geometries:
            raise ValueError("geometries_only requires include_geometries")
        globs.extend(GEOMETRY_ZIPS.values())
    else:
        globs.append(CYBENCH_DATA_ZIP)
        if include_geometries:
            globs.extend(GEOMETRY_ZIPS.values())

    staging_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {doi} into {staging_dir} ...")
    for glob_pattern in globs:
        print(f"  -> {glob_pattern}")
        _zenodo_download_file(doi, staging_dir, glob_pattern)

    paths = [staging_dir / name for name in globs]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Download finished but these files are missing: "
            + ", ".join(p.name for p in missing)
        )
    return paths


def fetch(
    data_dir: Path,
    *,
    doi: str = DEFAULT_DOI,
    staging_dir: Path | None = None,
    include_geometries: bool = False,
    geometries_only: bool = False,
    skip_download: bool = False,
    skip_extract: bool = False,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    staging = staging_dir or (data_dir / ".zenodo_download")
    staging.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        download_files(
            doi,
            staging,
            include_geometries=include_geometries,
            geometries_only=geometries_only,
        )
    elif geometries_only:
        missing = [
            staging / name
            for name in GEOMETRY_ZIPS.values()
            if not (staging / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "--skip-download set but geometry zips missing: "
                + ", ".join(p.name for p in missing)
            )
    elif not (staging / CYBENCH_DATA_ZIP).is_file():
        raise FileNotFoundError(
            f"--skip-download set but {staging / CYBENCH_DATA_ZIP} not found"
        )

    if skip_extract:
        return

    if not geometries_only:
        cybench_zip = staging / CYBENCH_DATA_ZIP
        extract_cybench_data(cybench_zip, data_dir)

    if include_geometries:
        for folder, zip_name in GEOMETRY_ZIPS.items():
            extract_geometry_zip(staging / zip_name, data_dir, folder)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download CY-Bench from Zenodo and unpack into cybench/data. "
            f"Default DOI: {DEFAULT_DOI} (latest version)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(PATH_DATA_DIR),
        help=f"Target data directory (default: {PATH_DATA_DIR}).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Directory for downloaded zip files (default: <data-dir>/.zenodo_download).",
    )
    parser.add_argument(
        "--doi",
        default=DEFAULT_DOI,
        help=f"Zenodo record or concept DOI (default: {DEFAULT_DOI}).",
    )
    parser.add_argument(
        "--geometries",
        action="store_true",
        help="Also download and unpack centroids.zip and polygons.zip (~105 MB).",
    )
    parser.add_argument(
        "--geometries-only",
        action="store_true",
        help="Download/unpack only centroids.zip and polygons.zip (skip cybench-data.zip).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract zips already present in --staging-dir.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download zips; do not unpack.",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 10) and not args.skip_download:
        raise SystemExit(
            "Downloading requires Python 3.10+ (zenodo-get). "
            "Use Python 3.10+ or pass --skip-download with pre-fetched zips."
        )

    fetch(
        args.data_dir,
        doi=args.doi,
        staging_dir=args.staging_dir,
        include_geometries=args.geometries or args.geometries_only,
        geometries_only=args.geometries_only,
        skip_download=args.skip_download,
        skip_extract=args.skip_extract,
    )
    print(f"Done. Data is under {args.data_dir}")


if __name__ == "__main__":
    main()
