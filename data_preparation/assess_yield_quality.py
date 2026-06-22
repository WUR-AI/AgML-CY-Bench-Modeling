"""CLI wrapper for yield quality assessment."""

from __future__ import annotations

import argparse

from cybench.config import PATH_DATA_DIR
from cybench.datasets.yield_quality import process_yield_quality_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate yield_quality_* CSV files with flag columns for CY-Bench. "
            "Filtering during training is configured in cybench/conf/dataset/target/yield.yaml."
        )
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=PATH_DATA_DIR,
        help=f"Root data directory (default: {PATH_DATA_DIR}).",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=["maize", "wheat"],
        help="Crop names to process (default: maize wheat).",
    )
    parser.add_argument(
        "--min-usable-year",
        type=int,
        default=2000,
        help="Year threshold for usable-sample statistics (default: 2000).",
    )
    args = parser.parse_args()

    print("Start crop yield data quality assessment...")
    process_yield_quality_files(
        args.directory,
        args.crops,
        min_usable_year=args.min_usable_year,
    )


if __name__ == "__main__":
    main()
