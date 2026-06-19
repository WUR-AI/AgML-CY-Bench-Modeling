#!/usr/bin/env python3
"""Install LPJmL region-aggregated CSVs into CY-Bench data layout.

Creates one file per crop/country::

    cybench/data/{crop}/{country}/lpjml_{crop}_{country}.csv

Columns: crop_name, adm_id, date, lpj_yield_rainfed, lpj_yield_irrigated

Example::

    .venv/bin/python data_preparation/prepare_lpjml_data.py \\
        --lpj-root /path/to/LPJmL/region_aggregated \\
        --data-dir cybench/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cybench.config import DATASETS, PATH_DATA_DIR
from cybench.models.lpjml_model import (
    LPJML_COL_IRRIGATED,
    LPJML_COL_RAINFED,
    LPJML_FILE_STEM,
)


def _find_variant_csv(lpj_root: Path, crop: str, country: str, variant: str) -> Path | None:
    base = lpj_root / crop / country
    if not base.is_dir():
        return None
    for sub in base.iterdir():
        if not sub.is_dir() or variant not in sub.name.lower():
            continue
        for csv in sub.glob(f"*_{crop}_{country}.csv"):
            return csv
    return None


def _load_variant(path: Path, yield_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    val_col = [c for c in df.columns if c not in {"crop_name", "adm_id", "date"}][0]
    out = df.rename(columns={val_col: yield_col})
    return out.loc[:, ["crop_name", "adm_id", "date", yield_col]].copy()


def prepare_country(
    lpj_root: Path,
    data_dir: Path,
    crop: str,
    country: str,
    *,
    overwrite: bool,
) -> bool:
    rainfed = _find_variant_csv(lpj_root, crop, country, "rainfed")
    if rainfed is None:
        return False

    out_dir = data_dir / crop / country
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{LPJML_FILE_STEM}_{crop}_{country}.csv"
    if out_path.exists() and not overwrite:
        return True

    df = _load_variant(rainfed, LPJML_COL_RAINFED)
    irrigated = _find_variant_csv(lpj_root, crop, country, "irrigated")
    if irrigated is not None:
        irr = _load_variant(irrigated, LPJML_COL_IRRIGATED)
        df = df.merge(irr, on=["crop_name", "adm_id", "date"], how="outer")
    else:
        df[LPJML_COL_IRRIGATED] = pd.NA

    df = df.loc[:, ["crop_name", "adm_id", "date", LPJML_COL_RAINFED, LPJML_COL_IRRIGATED]]
    df.to_csv(out_path, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lpj-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(PATH_DATA_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crop", action="append", choices=sorted(DATASETS))
    args = parser.parse_args()

    if not args.lpj_root.is_dir():
        raise SystemExit(f"LPJ root not found: {args.lpj_root}")

    crops = args.crop or list(DATASETS)
    written = 0
    missing = []
    for crop in crops:
        for country in DATASETS[crop]:
            ok = prepare_country(
                args.lpj_root,
                args.data_dir,
                crop,
                country,
                overwrite=args.overwrite,
            )
            if ok:
                written += 1
            else:
                missing.append(f"{crop}/{country}")

    print(f"Wrote or kept {written} lpjml_*.csv files under {args.data_dir}")
    if missing:
        print(f"No LPJ export for {len(missing)} crop/country pairs:")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
