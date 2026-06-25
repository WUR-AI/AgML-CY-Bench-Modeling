"""Lightweight path helpers for LPJmL / TWSO baseline CSVs (no torch import)."""

from __future__ import annotations

from pathlib import Path

from cybench.config import PATH_DATA_DIR

LPJML_FILE_STEM = "lpjml"
TWSO_FILE_STEM = "twso"


def lpjml_csv_path(crop: str, country: str, data_dir: str | Path = PATH_DATA_DIR) -> Path:
    return Path(data_dir) / crop / country / f"{LPJML_FILE_STEM}_{crop}_{country}.csv"


def twso_csv_path(crop: str, country: str, data_dir: str | Path = PATH_DATA_DIR) -> Path:
    return Path(data_dir) / crop / country / f"{TWSO_FILE_STEM}_{crop}_{country}.csv"
