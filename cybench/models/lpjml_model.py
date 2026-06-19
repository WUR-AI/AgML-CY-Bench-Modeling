from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR, PATH_DATA_DIR
from cybench.datasets.dataset import PandasDataset
from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle

log = logging.getLogger(__name__)

LPJML_FILE_STEM = "lpjml"
LPJML_COL_RAINFED = "lpj_yield_rainfed"
LPJML_COL_IRRIGATED = "lpj_yield_irrigated"
LPJML_VARIANTS = ("rainfed", "irrigated")


@dataclass(frozen=True)
class _LocationCalibration:
    mean_obs: float
    std_obs: float
    mean_mod: float
    std_mod: float


def lpjml_csv_path(crop: str, country: str, data_dir: str | Path = PATH_DATA_DIR) -> Path:
    return Path(data_dir) / crop / country / f"{LPJML_FILE_STEM}_{crop}_{country}.csv"


def _resolve_lpj_yield_series(df: pd.DataFrame, variant: str) -> pd.Series:
    """Pick rainfed, irrigated, or rainfed-with-irrigated fallback."""
    if variant not in LPJML_VARIANTS:
        raise ValueError(f"variant must be one of {LPJML_VARIANTS}, got {variant!r}")

    if LPJML_COL_RAINFED in df.columns and LPJML_COL_IRRIGATED in df.columns:
        rainfed = pd.Series(df[LPJML_COL_RAINFED])
        irrigated = pd.Series(df[LPJML_COL_IRRIGATED])
        if variant == "rainfed":
            return rainfed
        return irrigated

    raise ValueError(
        f"Expected columns {LPJML_COL_RAINFED} and {LPJML_COL_IRRIGATED} in LPJmL CSV"
    )


def load_lpjml_yields(
    crop: str,
    country: str,
    *,
    data_dir: str | Path = PATH_DATA_DIR,
    variant: str = "rainfed",
) -> pd.Series:
    """Load LPJmL yields as a Series indexed by (adm_id, year)."""
    path = lpjml_csv_path(crop, country, data_dir=data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"LPJmL predictor file not found: {path}")

    df = pd.read_csv(path)
    if "year" not in df.columns:
        df["year"] = df["date"].astype(str).str[:4].astype(int)

    yields = _resolve_lpj_yield_series(df, variant)
    out = df.rename(columns={"adm_id": KEY_LOC, "year": KEY_YEAR}).assign(_lpj=yields)
    return out.set_index([KEY_LOC, KEY_YEAR])["_lpj"].sort_index()


def bias_correct_lpj_yield(
    lpj_yield: float,
    calibration: _LocationCalibration,
) -> float:
    """Per-location mean/std rescaling (notebook-style bias correction)."""
    if calibration.std_mod == 0 or np.isnan(calibration.std_mod):
        return float(calibration.mean_obs)
    return float(
        ((lpj_yield - calibration.mean_mod) / calibration.std_mod) * calibration.std_obs
        + calibration.mean_obs
    )


class LpjmlBiasCorrectedModel(BaseModel):
    """Standalone LPJmL baseline with per-location bias correction.

    Fits location-wise mean/std mapping from training observed yields to LPJmL
    yields, then applies it at prediction time. Uses only LPJmL outputs — no
    weather or other CY-Bench predictors.
    """

    def __init__(
        self,
        name: str = "lpjml_bc",
        data_dir: str | None = None,
        variant: str = "rainfed",
        **_ignored,
    ):
        self.name = name
        self._data_dir = data_dir or PATH_DATA_DIR
        if variant not in LPJML_VARIANTS:
            raise ValueError(f"variant must be one of {LPJML_VARIANTS}, got {variant!r}")
        self._variant = variant
        self._crop: str | None = None
        self._country: str | None = None
        self._lpj_yields: pd.Series | None = None
        self._calibration: dict[str, _LocationCalibration] = {}
        self._global_mean_obs: float | None = None

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **fit_params
    ) -> tuple[Any, dict[str, Any]]:
        crop, country = self._resolve_crop_country(dataset)
        self._crop = crop
        self._country = country
        self._lpj_yields = load_lpjml_yields(
            crop, country, data_dir=self._data_dir, variant=self._variant
        )

        y = dataset.y.reset_index()
        self._global_mean_obs = float(y[KEY_TARGET].mean())
        self._calibration = {}

        for loc, obs_sub in y.groupby(KEY_LOC):
            mod_sub = self._lpj_series_for_rows(obs_sub[[KEY_LOC, KEY_YEAR]])
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
            )

        log.info(
            "LpjmlBiasCorrectedModel fitted for %s/%s: %d locations calibrated",
            crop,
            country,
            len(self._calibration),
        )
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dataset: PandasDataset, **predict_params
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        if self._lpj_yields is None or self._global_mean_obs is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")

        y = dataset.y
        locs = y.index.get_level_values(KEY_LOC)
        years = y.index.get_level_values(KEY_YEAR)
        predictions = np.empty(len(y), dtype=float)

        for i, (loc, year) in enumerate(zip(locs, years, strict=True)):
            loc_s = str(loc)
            lpj = self._lpj_yields.get((loc, year), np.nan)
            if pd.isna(lpj):
                predictions[i] = np.nan
                continue
            cal = self._calibration.get(loc_s)
            if cal is None:
                predictions[i] = float(lpj)
                log.debug("No calibration for %s; using raw LPJmL", loc_s)
                continue
            predictions[i] = bias_correct_lpj_yield(float(lpj), cal)

        return predictions, {}

    def _lpj_series_for_rows(self, rows: pd.DataFrame) -> pd.Series:
        assert self._lpj_yields is not None
        idx = list(zip(rows[KEY_LOC], rows[KEY_YEAR], strict=True))
        values = [self._lpj_yields.get(key, np.nan) for key in idx]
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
            raise ValueError("Dataset config must define crop.name and country for LPJmL model")
        if isinstance(country, list):
            raise ValueError("LpjmlBiasCorrectedModel supports one country per dataset")
        return str(crop), str(country)

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str = "lpjml_bc") -> LpjmlBiasCorrectedModel:
        return load_pickle(model_path, name)
