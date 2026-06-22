"""Yield data quality assessment for CY-Bench.

Reads raw yield files (``yield_{crop}_{country}.csv``) and produces
``yield_quality_{crop}_{country}.csv`` with boolean flag columns appended.
Rows are never removed — downstream filtering is configured in
``cybench/conf/dataset/target/yield.yaml``.

Flags
-----
flag_consecutive_yield
    Admin units whose yield series shows stagnant or suspiciously linear
    trends (near-zero second differences over consecutive years).
flag_area_outlier
    Samples whose planted/harvest area deviates beyond *threshold* standard
    deviations from a polynomial trend (both directions).
flag_yield_outlier
    Samples whose yield is unusually *high* relative to a polynomial trend.
    Low yields are kept — they may reflect genuine shortfalls.

Inspired by HarvestStat Africa (https://github.com/HarvestStat/HarvestStat-Africa),
adapted for statistical crop yield forecasting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cybench.config import DATASETS, KEY_LOC, KEY_TARGET, PATH_DATA_DIR

try:
    from numpy.exceptions import RankWarning  # type: ignore[attr-defined]
except ImportError:  # NumPy < 2
    RankWarning = np.RankWarning  # type: ignore[attr-defined,misc]

FLAG_CONSECUTIVE = "flag_consecutive_yield"
FLAG_AREA = "flag_area_outlier"
FLAG_YIELD = "flag_yield_outlier"
FLAG_COLUMNS = (FLAG_CONSECUTIVE, FLAG_AREA, FLAG_YIELD)

HARVEST_YEAR = "harvest_year"
AREA_COLUMNS = ("planted_area", "harvest_area")


def _count_true(mask: object) -> int:
    """Count True entries in a boolean mask (explicit for type checkers)."""
    return int(np.asarray(mask, dtype=bool).sum())


@dataclass(frozen=True)
class YieldQualitySummary:
    crop: str
    country_code: str
    n_samples: int
    n_flagged: int
    n_unflagged: int
    n_usable: int
    n_consecutive: int
    n_area_outlier: int
    n_yield_outlier: int


def _require_year_sorted(group: pd.DataFrame, *, caller: str) -> None:
    if not group[HARVEST_YEAR].is_monotonic_increasing:
        raise ValueError(f"{caller}: group must be sorted by {HARVEST_YEAR!r}")


def flag_consecutive_values(
    group: pd.DataFrame,
    *,
    column: str = KEY_TARGET,
    threshold_factor: float = 0.005,
    min_consecutive: int = 5,
) -> pd.Series:
    """Flag stagnant or suspiciously linear yield trends within one admin unit."""
    if len(group) < 3:
        return pd.Series(False, index=group.index, dtype=bool)

    _require_year_sorted(group, caller="flag_consecutive_values")
    series = group[column]
    threshold = float(series.median() * threshold_factor)
    second_diff = series.diff().diff().abs().to_numpy()

    is_stagnant = second_diff < threshold
    stagnant_indices = np.zeros(len(group), dtype=bool)
    for ix, flag in enumerate(is_stagnant):
        if flag:
            stagnant_indices[ix] = True
            if ix > 0:
                stagnant_indices[ix - 1] = True
            if ix > 1:
                stagnant_indices[ix - 2] = True

    if int(stagnant_indices.sum()) >= min_consecutive:
        return pd.Series(True, index=group.index, dtype=bool)
    return pd.Series(stagnant_indices, index=group.index, dtype=bool)


def detect_outliers_with_polyfit(
    group: pd.DataFrame,
    column: str,
    *,
    degree: int = 2,
    threshold: float = 3.0,
    direction: str = "both",
) -> pd.Series:
    """Flag outliers relative to a polynomial trend fit over time."""
    outlier_mask = pd.Series(False, index=group.index, dtype=bool)
    if len(group) < 5:
        return outlier_mask

    _require_year_sorted(group, caller="detect_outliers_with_polyfit")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RankWarning)
            coefficients = np.polyfit(group[HARVEST_YEAR], group[column], degree)
        polynomial = np.poly1d(coefficients)
        residuals = group[column] - polynomial(group[HARVEST_YEAR])
        std = float(np.std(residuals))
        if std < 1e-6:
            return outlier_mask

        z_scores = residuals / std
        if direction == "both":
            hits = abs(z_scores) > threshold
        elif direction == "high":
            hits = z_scores > threshold
        elif direction == "low":
            hits = -z_scores > threshold
        else:
            raise ValueError("direction must be 'high', 'low', or 'both'")
        outlier_mask.loc[group.index[hits.to_numpy()]] = True
    except RankWarning:
        pass

    return outlier_mask


def _apply_by_admin(
    df: pd.DataFrame,
    func,
    *,
    column: str,
    **kwargs,
) -> pd.Series:
    flags = pd.Series(False, index=df.index, dtype=bool)
    for _, group in df.groupby(KEY_LOC, sort=False):
        hit = func(group, column=column, **kwargs)
        flags.loc[group.index] = hit.to_numpy(dtype=bool)
    return flags


def _flag_invalid_yield(df: pd.DataFrame) -> pd.Series:
    """Rows with missing or non-positive yield should always be excluded."""
    invalid = df[KEY_TARGET].isna() | df[KEY_TARGET].le(0)
    return invalid.fillna(True)


def _area_column(df: pd.DataFrame) -> str | None:
    for name in AREA_COLUMNS:
        if name in df.columns:
            return name
    return None


def assess_yield_dataframe(
    df: pd.DataFrame,
    *,
    min_usable_year: int | None = None,
) -> tuple[pd.DataFrame, YieldQualitySummary | None]:
    """Append quality flag columns to a yield dataframe (in place on a copy).

    Returns the annotated dataframe and an optional summary when crop/country
    metadata columns are present.
    """
    out = df.copy()
    if out.empty:
        for col in FLAG_COLUMNS:
            out[col] = pd.Series(dtype=bool)
        return out, None

    out = out.sort_values([KEY_LOC, HARVEST_YEAR]).reset_index(drop=True)

    invalid = _flag_invalid_yield(out)
    out[FLAG_CONSECUTIVE] = _apply_by_admin(
        out.loc[~invalid],
        flag_consecutive_values,
        column=KEY_TARGET,
    ).reindex(out.index, fill_value=False)
    out.loc[invalid, FLAG_CONSECUTIVE] = True

    area_col = _area_column(out)
    if area_col is not None:
        area_flags = _apply_by_admin(
            out.loc[~invalid],
            detect_outliers_with_polyfit,
            column=area_col,
            direction="both",
        ).reindex(out.index, fill_value=False)
        out[FLAG_AREA] = area_flags | invalid
    else:
        out[FLAG_AREA] = invalid

    yield_flags = _apply_by_admin(
        out.loc[~invalid],
        detect_outliers_with_polyfit,
        column=KEY_TARGET,
        direction="high",
    ).reindex(out.index, fill_value=False)
    out[FLAG_YIELD] = yield_flags | invalid

    flagged: pd.Series = (
        out[FLAG_CONSECUTIVE].astype(bool)
        | out[FLAG_AREA].astype(bool)
        | out[FLAG_YIELD].astype(bool)
    )
    crop = str(out["crop_name"].iloc[0]) if "crop_name" in out.columns else ""
    country = ""
    if "country_code" in out.columns:
        country = str(out["country_code"].iloc[0])

    unflagged = flagged.eq(False)
    n_usable = _count_true(unflagged)
    if min_usable_year is not None:
        year_ok = out[HARVEST_YEAR].ge(min_usable_year)
        n_usable = _count_true(year_ok & unflagged)

    summary = YieldQualitySummary(
        crop=crop,
        country_code=country,
        n_samples=len(out),
        n_flagged=_count_true(flagged),
        n_unflagged=_count_true(unflagged),
        n_usable=n_usable,
        n_consecutive=_count_true(out[FLAG_CONSECUTIVE].astype(bool)),
        n_area_outlier=_count_true(out[FLAG_AREA].astype(bool)),
        n_yield_outlier=_count_true(out[FLAG_YIELD].astype(bool)),
    )

    return out, summary


def build_yield_quality_file(
    yield_file: str | Path,
    output_file: str | Path,
    *,
    min_usable_year: int | None = None,
) -> YieldQualitySummary | None:
    """Read a yield CSV, annotate flags, and write ``yield_quality_*`` output."""
    df = pd.read_csv(yield_file, header=0)
    annotated, summary = assess_yield_dataframe(df, min_usable_year=min_usable_year)
    annotated.to_csv(output_file, index=False)
    return summary


def iter_yield_files(
    data_dir: str | Path,
    crops: list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """Yield ``(crop, country_code, yield_csv_path)`` for known datasets."""
    root = Path(data_dir)
    crop_list = crops or list(DATASETS)
    paths: list[tuple[str, str, Path]] = []
    for crop in crop_list:
        countries = DATASETS.get(crop, [])
        if not countries and (root / crop).is_dir():
            countries = sorted(
                p.name for p in (root / crop).iterdir() if p.is_dir()
            )
        for country_code in countries:
            csv_path = root / crop / country_code / f"yield_{crop}_{country_code}.csv"
            if csv_path.is_file():
                paths.append((crop, country_code, csv_path))
    return paths


def process_yield_quality_files(
    data_dir: str | Path,
    crops: list[str] | None = None,
    *,
    min_usable_year: int = 2000,
) -> dict[str, dict[str, int]]:
    """Generate quality files for all crops/countries under ``data_dir``."""
    totals = {
        crop: {"org_samples": 0, "qual_samples": 0, "usable_samples": 0}
        for crop in (crops or list(DATASETS))
    }

    for crop, country_code, csv_path in iter_yield_files(data_dir, crops=crops):
        if crop not in totals:
            totals[crop] = {"org_samples": 0, "qual_samples": 0, "usable_samples": 0}

        out_path = csv_path.with_name(f"yield_quality_{crop}_{country_code}.csv")
        summary = build_yield_quality_file(
            csv_path,
            out_path,
            min_usable_year=min_usable_year,
        )
        if summary is None:
            continue

        totals[crop]["org_samples"] += summary.n_samples
        totals[crop]["qual_samples"] += summary.n_unflagged
        totals[crop]["usable_samples"] += summary.n_usable

        pct = 100.0 * summary.n_flagged / summary.n_samples if summary.n_samples else 0.0
        print(
            f"{country_code} | {crop}: {summary.n_samples} samples | "
            f"flagged {pct:.2f}% "
            f"({summary.n_consecutive} consecutive | "
            f"{summary.n_area_outlier} area | "
            f"{summary.n_yield_outlier} yield) | "
            f"unflagged {summary.n_unflagged} | usable (>={min_usable_year}) {summary.n_usable}"
        )

    for crop, stats in totals.items():
        print(f"\nResults for {crop}:")
        print(f"  Total samples: {stats['org_samples']}")
        print(f"  Unflagged samples: {stats['qual_samples']}")
        print(f"  Usable samples (>={min_usable_year}): {stats['usable_samples']}")

    return totals


def default_data_dir() -> Path:
    return Path(PATH_DATA_DIR)
