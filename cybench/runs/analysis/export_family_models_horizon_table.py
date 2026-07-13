#!/usr/bin/env python3
"""Export per-family horizon table with every model (median across countries).

Unlike ``export_family_table_latex.py`` (one EOS representative per family), this
lists all models in each family with median [IQR] per forecast horizon. Horizons
are scored independently, so incomplete early-season coverage shows as blank cells
rather than dropping the model.

Example::

    poetry run python cybench/runs/analysis/export_family_models_horizon_table.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --version 2 \\
        --metric nrmse \\
        -o family_models_horizon.csv

    poetry run python cybench/runs/analysis/export_family_models_horizon_table.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --version 2 \\
        --format markdown \\
        --crop maize \\
        -o family_models_maize.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cybench.runs.analysis.global_insights_lib import (
    HORIZON_DISPLAY_LABELS,
    build_family_models_horizon_table,
    discover_summary_tables,
    horizons_in_data,
    load_summary_frame,
)


def _format_cell(median: object, q25: object, q75: object, *, digits: int = 3) -> str:
    if median is None or (isinstance(median, float) and pd.isna(median)):
        return "—"
    med_s = f"{float(median):.{digits}f}"
    if q25 is None or q75 is None or pd.isna(q25) or pd.isna(q75):
        return med_s
    return f"{med_s} [{float(q25):.{digits}f}, {float(q75):.{digits}f}]"


def build_markdown_table(
    table: pd.DataFrame,
    *,
    metric: str,
    horizons: tuple[str, ...],
    crop_label: str,
) -> str:
    if table.empty:
        return f"# By family (median across countries) — {crop_label}\n\n_No data._\n"

    lines = [
        f"# By family (median across countries) — {crop_label}",
        "",
        "Median per country at each horizon (each country weighs equally). "
        "Brackets = interquartile range across countries. "
        "`*` = EOS family representative. "
        "Blank = no runs at that horizon.",
        "",
    ]
    for family, fam_df in table.groupby("family", sort=False):
        lines.append(f"## {family}")
        lines.append("")
        header = ["Model", *[HORIZON_DISPLAY_LABELS.get(hz, hz) for hz in horizons], "Rep."]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for _, row in fam_df.iterrows():
            name = str(row["display_name"])
            if row.get("is_representative"):
                name = f"{name}*"
            cells = [name]
            for hz in horizons:
                med = row.get(f"{hz}_{metric}_median")
                q25 = row.get(f"{hz}_{metric}_q25")
                q75 = row.get(f"{hz}_{metric}_q75")
                n_cc = row.get(f"n_countries_{hz}", 0)
                cell = _format_cell(med, q25, q75)
                if cell != "—" and n_cc not in (None, 0):
                    cell = f"{cell} (n={int(n_cc)})"
                cells.append(cell)
            cells.append("✓" if row.get("is_representative") else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/output"),
        help="Root containing paper_walk_forward_* collect directories",
    )
    parser.add_argument("--version", type=int, default=2, help="Collect batch version tag")
    parser.add_argument(
        "--crop",
        choices=["all", "maize", "wheat"],
        default="all",
        help="Crop filter (default: all)",
    )
    parser.add_argument(
        "--metric",
        default="nrmse",
        help="Primary metric for markdown view (default: nrmse)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["nrmse", "r2", "r_spatial", "r_temporal", "r_res"],
        help="Metrics in CSV export (default: paper family table metrics)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Write output to this file (default: stdout)",
    )
    args = parser.parse_args()

    paths = discover_summary_tables(args.output_root.resolve(), version=args.version)
    if not paths:
        raise SystemExit(
            f"No walk_forward_summary.csv under {args.output_root} (version={args.version})"
        )

    df = load_summary_frame(paths)
    for col in args.metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    crop_filter = None if args.crop == "all" else args.crop
    horizons = horizons_in_data(df)
    table = build_family_models_horizon_table(
        df,
        crop=crop_filter,
        horizons=horizons,
        metrics=tuple(args.metrics),
    )
    if table.empty:
        raise SystemExit("No rows in family×model horizon table (check metrics / data).")

    crop_label = "all crops" if crop_filter is None else crop_filter
    if args.format == "markdown":
        if args.metric not in args.metrics:
            raise SystemExit(f"--metric {args.metric!r} must be included in --metrics")
        text = build_markdown_table(
            table, metric=args.metric, horizons=horizons, crop_label=crop_label
        )
    else:
        text = table.to_csv(index=False)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"[DONE] Wrote {args.out} ({len(table)} models)")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
