import argparse
import os

import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, PATH_DATA_DIR


def build_yield_quality_file(yield_file: str, output_file: str) -> None:
    """Create quality flags expected by dataset/target/yield.yaml."""
    df = pd.read_csv(yield_file, header=0)

    # Keep DataFactory and yield.yaml unchanged by populating expected flag columns.
    # Flag any sample that should be excluded from training/evaluation (non-positive yield).
    bad_yield_flag = df[KEY_TARGET].le(0).fillna(True)
    df["flag_consecutive_yield"] = bad_yield_flag
    df["flag_area_outlier"] = bad_yield_flag
    df["flag_yield_outlier"] = bad_yield_flag
    flagged_entries = int(bad_yield_flag.sum())
    df.to_csv(output_file, index=False)
    print(f"Wrote {output_file} (flagged {flagged_entries} rows).")


def process_files(input_dir: str, crops: list[str]) -> None:
    for crop in crops:
        crop_dir = os.path.join(input_dir, crop)
        if not os.path.isdir(crop_dir):
            continue

        for country_code in os.listdir(crop_dir):
            path_data_cn = os.path.join(crop_dir, country_code)
            if not os.path.isdir(path_data_cn):
                continue

            yield_file = os.path.join(path_data_cn, f"yield_{crop}_{country_code}.csv")
            if not os.path.exists(yield_file):
                continue

            quality_file = os.path.join(
                path_data_cn, "_".join(["yield_quality", crop, country_code]) + ".csv"
            )
            build_yield_quality_file(yield_file, quality_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate yield quality files with the expected flag_* columns "
            "and mark rows where yield is missing or not positive."
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

    args = parser.parse_args()
    process_files(args.directory, args.crops)


if __name__ == "__main__":
    main()
