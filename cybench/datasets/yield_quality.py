"""Yield data quality assessment for CY-Bench.

Reads raw yield files (``yield_{crop}_{country}.csv``) and produces
``yield_quality_{crop}_{country}.csv`` sidecars with row keys and boolean
flag columns only (no duplicated yield/area columns). Rows are never removed
— downstream filtering is configured in
``cybench/conf/dataset/target/yield.yaml`` (``quality``, ``filter_samples``).

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
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from cybench.config import CONF_DIR, DATASETS, KEY_LOC, KEY_TARGET, KEY_YEAR, PATH_DATA_DIR

try:
    from numpy.exceptions import RankWarning  # type: ignore[attr-defined]
except ImportError:  # NumPy < 2
    RankWarning = np.RankWarning  # type: ignore[attr-defined,misc]

FLAG_CONSECUTIVE = "flag_consecutive_yield"
FLAG_AREA = "flag_area_outlier"
FLAG_YIELD = "flag_yield_outlier"
FLAG_COLUMNS = (FLAG_CONSECUTIVE, FLAG_AREA, FLAG_YIELD)

HARVEST_YEAR = "harvest_year"
QUALITY_KEY_COLUMNS = ("crop_name", "country_code", KEY_LOC, HARVEST_YEAR)
AREA_COLUMNS = ("planted_area", "harvest_area")
DEFAULT_OUTLIER_THRESHOLD = 3.0
DEFAULT_POLYFIT_DEGREE = 2
DEFAULT_CONSECUTIVE_THRESHOLD_FACTOR = 0.005
DEFAULT_CONSECUTIVE_MIN_YEARS = 5
DEFAULT_MIN_USABLE_YEAR = 2000
YIELD_TARGET_CONFIG = Path(CONF_DIR) / "dataset" / "target" / "yield.yaml"


@lru_cache(maxsize=1)
def load_yield_target_config() -> dict:
    """Load ``cybench/conf/dataset/target/yield.yaml``."""
    if not YIELD_TARGET_CONFIG.is_file():
        return {}
    with YIELD_TARGET_CONFIG.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _quality_config_section() -> dict:
    cfg = load_yield_target_config()
    quality = cfg.get("quality", {})
    if isinstance(quality, dict):
        return quality
    return {}


@dataclass(frozen=True)
class YieldQualitySettings:
    """Parameters for yield quality flag generation (from yield.yaml ``quality``)."""

    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
    polyfit_degree: int = DEFAULT_POLYFIT_DEGREE
    consecutive_threshold_factor: float = DEFAULT_CONSECUTIVE_THRESHOLD_FACTOR
    consecutive_min_years: int = DEFAULT_CONSECUTIVE_MIN_YEARS
    min_usable_year: int = DEFAULT_MIN_USABLE_YEAR


def _config_float(section: dict, root: dict, key: str, default: float) -> float:
    if key in section:
        return float(section[key])
    if key in root:
        return float(root[key])
    return default


def _config_int(section: dict, root: dict, key: str, default: int) -> int:
    if key in section:
        return int(section[key])
    if key in root:
        return int(root[key])
    return default


def _settings_from_quality_mapping(quality: object) -> YieldQualitySettings:
    section: dict = {}
    if isinstance(quality, DictConfig):
        section = OmegaConf.to_container(quality, resolve=True)  # type: ignore[assignment]
    elif isinstance(quality, dict):
        section = quality

    return YieldQualitySettings(
        outlier_threshold=_config_float(section, {}, "outlier_threshold", DEFAULT_OUTLIER_THRESHOLD),
        polyfit_degree=_config_int(section, {}, "polyfit_degree", DEFAULT_POLYFIT_DEGREE),
        consecutive_threshold_factor=_config_float(
            section, {}, "consecutive_threshold_factor", DEFAULT_CONSECUTIVE_THRESHOLD_FACTOR
        ),
        consecutive_min_years=_config_int(
            section, {}, "consecutive_min_years", DEFAULT_CONSECUTIVE_MIN_YEARS
        ),
        min_usable_year=_config_int(section, {}, "min_usable_year", DEFAULT_MIN_USABLE_YEAR),
    )


def yield_quality_settings_from_target(cfg: DictConfig) -> YieldQualitySettings:
    """Build settings from a Hydra config node with ``target.quality`` (or ``quality``)."""
    quality = OmegaConf.select(cfg, "target.quality")
    if quality is None:
        quality = OmegaConf.select(cfg, "quality")
    return _settings_from_quality_mapping(quality)


def filter_samples_from_target(cfg: DictConfig | None = None) -> list[str] | None:
    """Flag columns from ``target.filter_samples`` (training + visualization)."""
    if cfg is not None:
        samples = OmegaConf.select(cfg, "target.filter_samples")
        if samples is not None:
            return list(samples) if samples else None
    fs = load_yield_target_config().get("filter_samples")
    if isinstance(fs, list) and fs:
        return list(fs)
    return None


def viz_flag_columns(cfg: DictConfig | None = None) -> list[str]:
    """Which quality flag columns to include in PNG diagnostics."""
    selected = filter_samples_from_target(cfg)
    if selected:
        return selected
    return list(FLAG_COLUMNS)


@lru_cache(maxsize=1)
def configured_yield_quality_settings() -> YieldQualitySettings:
    """Load flag-generation settings from yield target config (YAML file)."""
    return _settings_from_quality_mapping(_quality_config_section())


def configured_outlier_threshold() -> float:
    """Outlier z-score threshold from yield target config (YAML)."""
    return configured_yield_quality_settings().outlier_threshold


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
    n_yield_poly_outlier: int
    n_yield_invalid: int


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
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
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
    settings: YieldQualitySettings | None = None,
    outlier_threshold: float | None = None,
) -> tuple[pd.DataFrame, YieldQualitySummary | None]:
    """Append quality flag columns to a yield dataframe (in place on a copy).

    Returns the annotated dataframe and an optional summary when crop/country
    metadata columns are present.
    """
    quality = settings or configured_yield_quality_settings()
    if outlier_threshold is not None:
        quality = replace(quality, outlier_threshold=outlier_threshold)

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
        threshold_factor=quality.consecutive_threshold_factor,
        min_consecutive=quality.consecutive_min_years,
    ).reindex(out.index, fill_value=False)
    out.loc[invalid, FLAG_CONSECUTIVE] = True

    area_col = _area_column(out)
    if area_col is not None:
        area_flags = _apply_by_admin(
            out.loc[~invalid],
            detect_outliers_with_polyfit,
            column=area_col,
            direction="both",
            degree=quality.polyfit_degree,
            threshold=quality.outlier_threshold,
        ).reindex(out.index, fill_value=False)
        out[FLAG_AREA] = area_flags | invalid
    else:
        out[FLAG_AREA] = invalid

    yield_flags = _apply_by_admin(
        out.loc[~invalid],
        detect_outliers_with_polyfit,
        column=KEY_TARGET,
        direction="high",
        degree=quality.polyfit_degree,
        threshold=quality.outlier_threshold,
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
        n_yield_poly_outlier=_count_true(yield_flags),
        n_yield_invalid=_count_true(invalid),
    )

    return out, summary


def quality_merge_keys(df: pd.DataFrame) -> list[str]:
    """Columns used to join yield data with quality sidecars."""
    return [col for col in QUALITY_KEY_COLUMNS if col in df.columns]


def _parse_flag_columns(df: pd.DataFrame, flag_cols: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = flag_cols or [col for col in FLAG_COLUMNS if col in out.columns]
    for col in cols:
        if out[col].dtype == bool:
            continue
        out[col] = out[col].astype(str).str.lower().eq("true")
    return out


def slim_quality_dataframe(annotated: pd.DataFrame) -> pd.DataFrame:
    """Keep only row keys and flag columns for ``yield_quality_*`` output."""
    keys = quality_merge_keys(annotated)
    if KEY_LOC not in keys or HARVEST_YEAR not in keys:
        keys = [col for col in (KEY_LOC, HARVEST_YEAR) if col in annotated.columns]
    return annotated[keys + list(FLAG_COLUMNS)]


def merge_yield_with_quality(
    yield_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join yield values with quality flags (supports legacy full quality CSVs)."""
    if KEY_TARGET in quality_df.columns:
        return quality_df.copy()

    keys = quality_merge_keys(quality_df)
    if KEY_LOC not in keys or HARVEST_YEAR not in keys:
        keys = [
            col
            for col in (KEY_LOC, HARVEST_YEAR)
            if col in yield_df.columns and col in quality_df.columns
        ]
    flag_cols = [col for col in FLAG_COLUMNS if col in quality_df.columns]
    flags = _parse_flag_columns(quality_df, flag_cols)

    left = yield_df.copy()
    left["_orig_idx"] = np.arange(len(left))
    sort_cols = [col for col in (KEY_LOC, HARVEST_YEAR) if col in left.columns]
    left = left.sort_values(sort_cols)
    right = flags.sort_values(sort_cols)

    left["_join_ord"] = left.groupby(keys, sort=False).cumcount()
    right["_join_ord"] = right.groupby(keys, sort=False).cumcount()
    merge_on = keys + ["_join_ord"]

    merged = left.merge(
        right[merge_on + flag_cols],
        on=merge_on,
        how="left",
    )
    merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx", "_join_ord"])
    for col in flag_cols:
        merged[col] = merged[col].fillna(False).astype(bool)
    return merged.reset_index(drop=True)


def _attach_quality_flags(
    df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach quality flag columns to rows keyed by admin unit and harvest year."""
    flag_cols = [col for col in FLAG_COLUMNS if col in quality_df.columns]
    flags = _parse_flag_columns(quality_df, flag_cols)

    left = df.copy()
    left["_orig_idx"] = np.arange(len(left))
    year_col = KEY_YEAR if KEY_YEAR in left.columns else HARVEST_YEAR
    if year_col != HARVEST_YEAR:
        left = left.rename(columns={year_col: HARVEST_YEAR})

    keys = [col for col in quality_merge_keys(flags) if col in left.columns]
    if KEY_LOC not in keys or HARVEST_YEAR not in keys:
        keys = [
            col
            for col in (KEY_LOC, HARVEST_YEAR)
            if col in left.columns
        ]

    sort_cols = [KEY_LOC, HARVEST_YEAR]
    left = left.sort_values(sort_cols)
    right = flags.sort_values(sort_cols)

    left["_join_ord"] = left.groupby(keys, sort=False).cumcount()
    right["_join_ord"] = right.groupby(keys, sort=False).cumcount()
    merge_on = keys + ["_join_ord"]

    merged = left.merge(
        right[merge_on + flag_cols],
        on=merge_on,
        how="left",
    )
    merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx", "_join_ord"])
    for col in flag_cols:
        merged[col] = merged[col].fillna(False).astype(bool)
    if year_col != HARVEST_YEAR:
        merged = merged.rename(columns={HARVEST_YEAR: year_col})
    return merged.reset_index(drop=True)


def apply_yield_quality_filter(
    df: pd.DataFrame,
    crop: str,
    country_code: str,
    *,
    data_dir: str | Path | None = None,
    quality_flags: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Drop rows flagged in ``yield_quality_*`` sidecars (runtime QC for analysis)."""
    flag_cols = quality_flags if quality_flags is not None else filter_samples_from_target()
    if not flag_cols or df.empty:
        return df.copy(), 0

    root = Path(data_dir or PATH_DATA_DIR)
    quality_path = root / crop / country_code / f"yield_quality_{crop}_{country_code}.csv"
    if not quality_path.is_file():
        return df.copy(), 0

    annotated = _attach_quality_flags(df, pd.read_csv(quality_path))
    present = [col for col in flag_cols if col in annotated.columns]
    if not present:
        return df.copy(), 0

    flagged = annotated[present].any(axis=1)
    n_removed = _count_true(flagged)
    keep = annotated.loc[~flagged].drop(
        columns=[col for col in FLAG_COLUMNS if col in annotated.columns]
    )
    return keep.reset_index(drop=True), n_removed


def build_yield_quality_file(
    yield_file: str | Path,
    output_file: str | Path,
    *,
    min_usable_year: int | None = None,
    settings: YieldQualitySettings | None = None,
) -> YieldQualitySummary | None:
    """Read a yield CSV, annotate flags, and write a slim ``yield_quality_*`` sidecar."""
    quality = settings or configured_yield_quality_settings()
    stats_year = quality.min_usable_year if min_usable_year is None else min_usable_year
    df = pd.read_csv(yield_file, header=0)
    annotated, summary = assess_yield_dataframe(
        df,
        min_usable_year=stats_year,
        settings=quality,
    )
    slim_quality_dataframe(annotated).to_csv(output_file, index=False)
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
    settings: YieldQualitySettings | None = None,
) -> dict[str, dict[str, int]]:
    """Generate quality files for all crops/countries under ``data_dir``."""
    quality = settings or configured_yield_quality_settings()
    min_usable_year = quality.min_usable_year
    totals = {
        crop: {
            "org_samples": 0,
            "qual_samples": 0,
            "usable_samples": 0,
            "yield_poly": 0,
            "yield_invalid": 0,
        }
        for crop in (crops or list(DATASETS))
    }

    for crop, country_code, csv_path in iter_yield_files(data_dir, crops=crops):
        if crop not in totals:
            totals[crop] = {
                "org_samples": 0,
                "qual_samples": 0,
                "usable_samples": 0,
                "yield_poly": 0,
                "yield_invalid": 0,
            }

        out_path = csv_path.with_name(f"yield_quality_{crop}_{country_code}.csv")
        summary = build_yield_quality_file(
            csv_path,
            out_path,
            settings=quality,
        )
        if summary is None:
            continue

        totals[crop]["org_samples"] += summary.n_samples
        totals[crop]["qual_samples"] += summary.n_unflagged
        totals[crop]["usable_samples"] += summary.n_usable
        totals[crop]["yield_poly"] += summary.n_yield_poly_outlier
        totals[crop]["yield_invalid"] += summary.n_yield_invalid

        pct = 100.0 * summary.n_flagged / summary.n_samples if summary.n_samples else 0.0
        print(
            f"{country_code} | {crop}: {summary.n_samples} samples | "
            f"flagged {pct:.2f}% "
            f"({summary.n_consecutive} consecutive | "
            f"{summary.n_area_outlier} area | "
            f"{summary.n_yield_poly_outlier} yield polyfit | "
            f"{summary.n_yield_invalid} yield invalid ≤0) | "
            f"unflagged {summary.n_unflagged} | usable (>={min_usable_year}) {summary.n_usable}"
        )

    for crop, stats in totals.items():
        if stats["org_samples"] == 0:
            continue
        print(f"\nResults for {crop}:")
        print(f"  Total samples: {stats['org_samples']}")
        print(f"  Unflagged samples: {stats['qual_samples']}")
        print(f"  Usable samples (>={min_usable_year}): {stats['usable_samples']}")
        print(
            f"  Yield flags: {stats['yield_poly']} polyfit high | "
            f"{stats['yield_invalid']} invalid ≤0 "
            f"({stats['yield_poly'] + stats['yield_invalid']} in flag_yield_outlier)"
        )

    return totals


def default_data_dir() -> Path:
    return Path(PATH_DATA_DIR)
