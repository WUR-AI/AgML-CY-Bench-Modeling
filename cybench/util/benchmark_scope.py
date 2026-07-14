"""Crop×country pairs included in benchmark evaluation outputs.

Pairs must support the full screening layout (5 test + 2 val + ≥1 train years).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import cybench.config as config
from cybench.util.validation import check_full_benchmark_screening_years

DEFAULT_MIN_YEAR = 2000
DEFAULT_MAX_YEAR = 2024


def _load_yield_years(
    crop: str,
    country: str,
    *,
    data_dir: Path,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
) -> set[int]:
    path = data_dir / crop / country / f"yield_{crop}_{country}.csv"
    if not path.is_file():
        return set()
    years: set[int] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        year_col = "harvest_year" if "harvest_year" in (reader.fieldnames or []) else "year"
        for row in reader:
            raw = row.get(year_col)
            if raw is None or raw == "":
                continue
            year = int(float(raw))
            if min_year <= year <= max_year:
                years.add(year)
    return years


def benchmark_crop_country_key(crop: str, country: str) -> tuple[str, str]:
    return (crop.casefold(), country.upper())


@lru_cache(maxsize=256)
def _full_screening_ok_cached(crop: str, country: str, data_dir: str) -> bool:
    years = _load_yield_years(crop, country, data_dir=Path(data_dir))
    ok, _ = check_full_benchmark_screening_years(years)
    return ok


def is_benchmark_evaluation_crop_country(
    crop: str,
    country: str,
    *,
    data_dir: Path | str | None = None,
    years: set[int] | None = None,
) -> bool:
    """Return True when crop×country has the full benchmark screening test window."""
    if years is not None:
        ok, _ = check_full_benchmark_screening_years(years)
        return ok
    root = str(Path(data_dir or config.PATH_DATA_DIR).resolve())
    crop_key, country_key = benchmark_crop_country_key(crop, country)
    return _full_screening_ok_cached(crop_key, country_key, root)


def is_benchmark_evaluation_country(
    country: str,
    *,
    data_dir: Path | str | None = None,
) -> bool:
    """True when a country has at least one included crop×country pair."""
    cc = country.upper()
    for crop, countries in config.DATASETS.items():
        if cc in countries and is_benchmark_evaluation_crop_country(
            crop, cc, data_dir=data_dir
        ):
            return True
    return False


def benchmark_evaluation_exclusion_reason(
    crop: str,
    country: str,
    *,
    data_dir: Path | str | None = None,
    years: set[int] | None = None,
) -> str | None:
    """Human-readable reason when a pair is excluded; None if included."""
    if years is None:
        years = _load_yield_years(
            crop, country, data_dir=Path(data_dir or config.PATH_DATA_DIR)
        )
    ok, reason = check_full_benchmark_screening_years(years)
    return None if ok else reason
