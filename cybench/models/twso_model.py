from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cybench.config import CROP_CALENDAR_DATES, KEY_LOC, KEY_TARGET, KEY_YEAR, PATH_DATA_DIR
from cybench.datasets.alignment import compute_crop_season_window
from cybench.datasets.dataset import PandasDataset
from cybench.models.baseline_csv_paths import twso_csv_path
from cybench.models.lpjml_model import (
    _LocationCalibration,
    _MomentCalibration,
    _needs_global_calibration,
    bias_correct_lpj_yield,
)
from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle
from cybench.util.validation import default_screening_validation_cfg, get_screening_partitions

log = logging.getLogger(__name__)


class TwsoNotApplicableError(Exception):
    """TWSO baseline cannot be calibrated (no overlapping predictor and yields)."""


TWSO_COL = "twso"
TWSO_SCALE = 0.001
DEFAULT_START_OF_SEQUENCE = "sos-60"
DEFAULT_END_OF_SEQUENCE = "eos"
DEFAULT_MIN_YEAR = 2000
DEFAULT_MAX_YEAR = 2024
# Skip loc-years whose last in-season TWSO is more than this many days before EOS
# (truncated files, e.g. US 2023 ending weeks before harvest).
DEFAULT_MAX_DAYS_BEFORE_EOS = 14


def crop_calendar_csv_path(
    crop: str, country: str, data_dir: str | Path = PATH_DATA_DIR
) -> Path:
    return Path(data_dir) / crop / country / f"crop_calendar_{crop}_{country}.csv"


def _aggregate_max_twso(df: pd.DataFrame, *, scale: float) -> pd.Series:
    grouped = df.groupby([KEY_LOC, KEY_YEAR], observed=True)[TWSO_COL].max()
    return (grouped * scale).sort_index()


def _twso_loc_year_complete_mask(
    aligned: pd.DataFrame,
    crop_season_df: pd.DataFrame,
    *,
    max_days_before_eos: int,
) -> pd.Series:
    """True when the last in-season TWSO date is within ``max_days_before_eos`` of EOS."""
    if aligned.empty:
        return pd.Series(dtype=bool)

    last = aligned.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].max()
    eos = crop_season_df.set_index([KEY_LOC, KEY_YEAR])["end_of_sequence_date"]
    complete: dict[tuple[str, int], bool] = {}
    for key, last_dt in last.items():
        eos_dt = eos.get(key)
        if eos_dt is None or pd.isna(eos_dt):
            complete[key] = False
            continue
        gap_days = int((pd.Timestamp(eos_dt) - pd.Timestamp(last_dt)).days)
        complete[key] = gap_days <= max_days_before_eos
    return pd.Series(complete, dtype=bool)


def twso_coverage_table(
    crop: str,
    country: str,
    *,
    data_dir: str | Path = PATH_DATA_DIR,
    start_of_sequence: str = DEFAULT_START_OF_SEQUENCE,
    end_of_sequence: str = DEFAULT_END_OF_SEQUENCE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    max_days_before_eos: int = DEFAULT_MAX_DAYS_BEFORE_EOS,
) -> pd.DataFrame:
    """Per location-year TWSO season coverage relative to crop-calendar EOS."""
    path = twso_csv_path(crop, country, data_dir=data_dir)
    cal_path = crop_calendar_csv_path(crop, country, data_dir=data_dir)
    if not path.is_file() or not cal_path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df[KEY_YEAR] = df["date"].dt.year
    df = df[[KEY_LOC, KEY_YEAR, "date", TWSO_COL]].copy()

    crop_cal = pd.read_csv(cal_path)
    crop_season_df = compute_crop_season_window(
        crop_cal,
        min_year=min_year,
        max_year=max_year,
        start_of_sequence=start_of_sequence,
        end_of_sequence=end_of_sequence,
    )
    aligned = _align_twso_to_season(df, crop_season_df)
    if aligned.empty:
        return pd.DataFrame()

    last = aligned.groupby([KEY_LOC, KEY_YEAR], observed=True)["date"].max().rename("last_twso_date")
    eos = crop_season_df.set_index([KEY_LOC, KEY_YEAR])["end_of_sequence_date"].rename("eos_date")
    out = last.to_frame().join(eos, how="left")
    out["days_before_eos"] = (out["eos_date"] - out["last_twso_date"]).dt.days
    out["complete"] = out["days_before_eos"] <= max_days_before_eos
    return out.reset_index()


def _align_twso_to_season(df: pd.DataFrame, crop_season_df: pd.DataFrame) -> pd.DataFrame:
    """Keep TWSO observations that fall inside each location-year season window.

    Uses the same year reassignment as ``align_to_crop_season_window`` but skips
    the sparse-data bracket check (TWSO is often sampled sparsely within a season).
    """
    aligned = df.merge(
        crop_season_df[[KEY_LOC, KEY_YEAR] + CROP_CALENDAR_DATES],
        on=[KEY_LOC, KEY_YEAR],
    )
    aligned[KEY_YEAR] = np.where(
        aligned["date"] > aligned["eos_date"],
        aligned[KEY_YEAR] + 1,
        aligned[KEY_YEAR],
    )
    aligned = aligned.drop(columns=CROP_CALENDAR_DATES).merge(
        crop_season_df, on=[KEY_LOC, KEY_YEAR]
    )
    return aligned[
        (aligned["date"] >= aligned["start_of_sequence_date"])
        & (aligned["date"] <= aligned["end_of_sequence_date"])
    ]


def load_twso_yields(
    crop: str,
    country: str,
    *,
    data_dir: str | Path = PATH_DATA_DIR,
    start_of_sequence: str = DEFAULT_START_OF_SEQUENCE,
    end_of_sequence: str = DEFAULT_END_OF_SEQUENCE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    scale: float = TWSO_SCALE,
    max_days_before_eos: int = DEFAULT_MAX_DAYS_BEFORE_EOS,
) -> pd.Series:
    """Load season-aligned max TWSO as a Series indexed by (adm_id, year).

    Daily TWSO values are aligned to the crop-season window (same defaults as
    the benchmark dataset config), then aggregated with ``max`` per location-year
    and scaled to t/ha-compatible units (× ``scale``, default 0.001).

    Location-years whose last in-season observation falls more than
    ``max_days_before_eos`` days before EOS are returned as NaN (skipped).
    """
    path = twso_csv_path(crop, country, data_dir=data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"TWSO predictor file not found: {path}")

    cal_path = crop_calendar_csv_path(crop, country, data_dir=data_dir)
    if not cal_path.is_file():
        raise FileNotFoundError(f"Crop calendar required for TWSO alignment: {cal_path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df[KEY_YEAR] = df["date"].dt.year
    df = df[[KEY_LOC, KEY_YEAR, "date", TWSO_COL]].copy()

    crop_cal = pd.read_csv(cal_path)
    crop_season_df = compute_crop_season_window(
        crop_cal,
        min_year=min_year,
        max_year=max_year,
        start_of_sequence=start_of_sequence,
        end_of_sequence=end_of_sequence,
    )
    aligned = _align_twso_to_season(df, crop_season_df)
    yields = _aggregate_max_twso(aligned, scale=scale)
    complete = _twso_loc_year_complete_mask(
        aligned, crop_season_df, max_days_before_eos=max_days_before_eos
    )
    n_incomplete = 0
    for key in yields.index:
        if key not in complete.index or not bool(complete[key]):
            yields[key] = np.nan
            n_incomplete += 1
    if n_incomplete:
        log.info(
            "TWSO %s/%s: skipped %d location-year(s) with last observation "
            "more than %d day(s) before EOS",
            crop,
            country,
            n_incomplete,
            max_days_before_eos,
        )
    return yields


def _load_yield_years(
    crop: str,
    country: str,
    *,
    data_dir: str | Path,
    min_year: int,
    max_year: int,
) -> set[int]:
    path = Path(data_dir) / crop / country / f"yield_{crop}_{country}.csv"
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


def _load_yield_rows_for_years(
    crop: str,
    country: str,
    years: set[int],
    *,
    data_dir: str | Path,
) -> pd.DataFrame:
    path = Path(data_dir) / crop / country / f"yield_{crop}_{country}.csv"
    df = pd.read_csv(path)
    year_col = "harvest_year" if "harvest_year" in df.columns else "year"
    df = df[df[year_col].isin(years)]
    return df[[KEY_LOC, year_col, KEY_TARGET]].rename(columns={year_col: KEY_YEAR})


def twso_training_overlap_ok(
    y: pd.DataFrame,
    twso_yields: pd.Series,
) -> bool:
    keys = list(zip(y[KEY_LOC], y[KEY_YEAR].astype(int), strict=True))
    mod_vals = [twso_yields.get(key, np.nan) for key in keys]
    return any(np.isfinite(val) for val in mod_vals)


def twso_screening_viable(
    crop: str,
    country: str,
    *,
    data_dir: str | Path = PATH_DATA_DIR,
    end_of_sequence: str = DEFAULT_END_OF_SEQUENCE,
    min_year: int = DEFAULT_MIN_YEAR,
    max_year: int = DEFAULT_MAX_YEAR,
    seed: int = 42,
    max_days_before_eos: int = DEFAULT_MAX_DAYS_BEFORE_EOS,
) -> tuple[bool, str]:
    """Whether twso_bc screening can fit on train+val years for this horizon."""
    root = Path(data_dir)
    if not twso_csv_path(crop, country, data_dir=root).is_file():
        return False, "missing twso csv"
    if not crop_calendar_csv_path(crop, country, data_dir=root).is_file():
        return False, "missing crop calendar"

    years = _load_yield_years(
        crop, country, data_dir=root, min_year=min_year, max_year=max_year
    )
    if not years:
        return False, "no yield years in dataset window"
    try:
        train_years, val_years, test_years = get_screening_partitions(
            default_screening_validation_cfg(), years, seed=seed
        )
    except (AssertionError, ValueError) as exc:
        return False, str(exc)

    fit_years = set(train_years) | set(val_years)
    y = _load_yield_rows_for_years(crop, country, fit_years, data_dir=root)
    if y.empty:
        return False, "no yield rows for screening train+val years"

    try:
        twso_yields = load_twso_yields(
            crop,
            country,
            data_dir=root,
            end_of_sequence=end_of_sequence,
            min_year=min_year,
            max_year=max_year,
            max_days_before_eos=max_days_before_eos,
        )
    except FileNotFoundError as exc:
        return False, str(exc)

    if not twso_training_overlap_ok(y, twso_yields):
        return (
            False,
            "no overlapping TWSO and observed yields for screening train+val years "
            f"(horizon={end_of_sequence}, train+val={sorted(fit_years)}, test={test_years})",
        )
    return True, "ok"


class TwsoBiasCorrectedModel(BaseModel):
    """Standalone TWSO baseline with per-location variance-matching bias correction.

    Uses the in-season maximum of TWSO (scaled to t/ha units) as the process
    model output, then applies the same location-level moment matching used for
    LPJmL (global fallback when local calibration is unreliable).
    """

    def __init__(
        self,
        name: str = "twso_bc",
        data_dir: str | None = None,
        start_of_sequence: str = DEFAULT_START_OF_SEQUENCE,
        end_of_sequence: str = DEFAULT_END_OF_SEQUENCE,
        min_year: int = DEFAULT_MIN_YEAR,
        max_year: int = DEFAULT_MAX_YEAR,
        scale: float = TWSO_SCALE,
        min_location_years: int = 5,
        min_std_mod: float = 0.1,
        max_days_before_eos: int = DEFAULT_MAX_DAYS_BEFORE_EOS,
        **_ignored,
    ):
        self.name = name
        self._data_dir = data_dir or PATH_DATA_DIR
        self._start_of_sequence = start_of_sequence
        self._end_of_sequence = end_of_sequence
        self._min_year = int(min_year)
        self._max_year = int(max_year)
        self._scale = float(scale)
        self._min_location_years = int(min_location_years)
        self._min_std_mod = float(min_std_mod)
        self._max_days_before_eos = int(max_days_before_eos)
        self._crop: str | None = None
        self._country: str | None = None
        self._twso_yields: pd.Series | None = None
        self._calibration: dict[str, _LocationCalibration] = {}
        self._global_calibration: _MomentCalibration | None = None

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        crop, country = self._resolve_crop_country(dataset)
        self._crop = crop
        self._country = country
        self._twso_yields = load_twso_yields(
            crop,
            country,
            data_dir=self._data_dir,
            start_of_sequence=self._start_of_sequence,
            end_of_sequence=self._end_of_sequence,
            min_year=self._min_year,
            max_year=self._max_year,
            scale=self._scale,
            max_days_before_eos=self._max_days_before_eos,
        )

        y = dataset.y.reset_index()
        self._calibration = {}
        self._global_calibration = self._fit_global_calibration(y)

        for loc, obs_sub in y.groupby(KEY_LOC):
            mod_sub = self._twso_series_for_rows(obs_sub[[KEY_LOC, KEY_YEAR]])
            if mod_sub.empty:
                continue
            mean_obs = float(obs_sub[KEY_TARGET].mean())
            std_obs = float(obs_sub[KEY_TARGET].std(ddof=0))
            mean_mod = float(mod_sub.mean())
            std_mod = float(mod_sub.std(ddof=0))
            if np.isnan(std_obs):
                std_obs = 0.0
            if np.isnan(std_mod):
                std_mod = 0.0
            self._calibration[str(loc)] = _LocationCalibration(
                mean_obs=mean_obs,
                std_obs=std_obs,
                mean_mod=mean_mod,
                std_mod=std_mod,
                n_years=len(mod_sub),
            )

        n_global = sum(
            1
            for loc in self._calibration
            if _needs_global_calibration(
                self._calibration[loc],
                min_location_years=self._min_location_years,
                min_std_mod=self._min_std_mod,
            )
        )
        log.info(
            "TwsoBiasCorrectedModel fitted for %s/%s: %d locations calibrated "
            "(%d use global fallback, N_min=%d, sigma_P_min=%.3f)",
            crop,
            country,
            len(self._calibration),
            n_global,
            self._min_location_years,
            self._min_std_mod,
        )
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        if self._twso_yields is None or self._global_calibration is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")

        y = dataset.y
        locs = y.index.get_level_values(KEY_LOC)
        years = y.index.get_level_values(KEY_YEAR)
        predictions = np.empty(len(y), dtype=float)

        for i, (loc, year) in enumerate(zip(locs, years, strict=True)):
            loc_s = str(loc)
            twso = self._twso_yields.get((loc, year), np.nan)
            if pd.isna(twso):
                predictions[i] = np.nan
                continue
            local = self._calibration.get(loc_s)
            cal = self._effective_calibration(local)
            predictions[i] = bias_correct_lpj_yield(float(twso), cal)

        return predictions, {}

    def _effective_calibration(
        self, local: _LocationCalibration | None
    ) -> _MomentCalibration:
        assert self._global_calibration is not None
        if _needs_global_calibration(
            local,
            min_location_years=self._min_location_years,
            min_std_mod=self._min_std_mod,
        ):
            return self._global_calibration
        assert local is not None
        return local

    def _fit_global_calibration(self, y: pd.DataFrame) -> _MomentCalibration:
        assert self._twso_yields is not None
        keys = list(zip(y[KEY_LOC], y[KEY_YEAR].astype(int), strict=True))
        mod_vals = [self._twso_yields.get(key, np.nan) for key in keys]
        mod_arr = np.asarray(mod_vals, dtype=float)
        obs_arr = y[KEY_TARGET].to_numpy(dtype=float)
        mask = ~np.isnan(mod_arr)
        obs_arr = obs_arr[mask]
        mod_arr = mod_arr[mask]

        if obs_arr.size == 0:
            raise TwsoNotApplicableError(
                "No overlapping TWSO and observed yields in training set"
            )

        std_obs = float(obs_arr.std(ddof=0))
        std_mod = float(mod_arr.std(ddof=0))
        if np.isnan(std_obs):
            std_obs = 0.0
        if np.isnan(std_mod):
            std_mod = 0.0
        return _MomentCalibration(
            mean_obs=float(obs_arr.mean()),
            std_obs=std_obs,
            mean_mod=float(mod_arr.mean()),
            std_mod=std_mod,
        )

    def _twso_series_for_rows(self, rows: pd.DataFrame) -> pd.Series:
        assert self._twso_yields is not None
        idx = list(zip(rows[KEY_LOC], rows[KEY_YEAR], strict=True))
        values = [self._twso_yields.get(key, np.nan) for key in idx]
        return pd.Series(values, dtype=float).dropna()

    @staticmethod
    def _resolve_crop_country(dataset: PandasDataset) -> tuple[str, str]:
        cfg = dataset.cfg
        if hasattr(cfg, "crop"):
            crop = cfg.crop.name if hasattr(cfg.crop, "name") else cfg.crop["name"]
            country = cfg.country
        else:
            crop = cfg["crop"]["name"]
            country = cfg["country"]
        if not crop or not country:
            raise ValueError("Dataset config must define crop.name and country for TWSO model")
        if isinstance(country, list):
            raise ValueError("TwsoBiasCorrectedModel supports one country per dataset")
        return str(crop), str(country)

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str = "twso_bc") -> TwsoBiasCorrectedModel:
        return load_pickle(model_path, name)
