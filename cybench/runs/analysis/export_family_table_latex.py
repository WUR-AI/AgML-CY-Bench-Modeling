#!/usr/bin/env python3
"""Export crop-split model-family table as LaTeX (NRMSE, R², spatial/temporal/anomaly r).

Example::

    poetry run python cybench/runs/analysis/export_family_table_latex.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --horizon eos \\
        --version 2 \\
        -o family_table.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cybench.runs.analysis.global_insights_lib import discover_summary_tables, load_summary_frame
from cybench.runs.analysis.model_family_radar_lib import (
    PAPER_FAMILY_TABLE_METRICS,
    build_paper_family_table_latex,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/output"),
        help="Root containing paper_walk_forward_* collect directories",
    )
    parser.add_argument("--version", type=int, default=2, help="Collect batch version tag")
    parser.add_argument("--horizon", default="eos", help="Batch horizon slug (default: eos)")
    parser.add_argument(
        "--crops",
        nargs="+",
        default=["maize", "wheat"],
        help="Crop sections in table order (default: maize wheat)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Write LaTeX to this file (default: stdout)",
    )
    args = parser.parse_args()

    paths = discover_summary_tables(args.output_root.resolve(), version=args.version)
    if not paths:
        raise SystemExit(f"No walk_forward_summary.csv under {args.output_root} (version={args.version})")

    df = load_summary_frame(paths)
    for metric in PAPER_FAMILY_TABLE_METRICS:
        if metric in df.columns:
            df[metric] = __import__("pandas").to_numeric(df[metric], errors="coerce")

    latex = build_paper_family_table_latex(
        df,
        batch_horizon=args.horizon,
        crops=tuple(args.crops),
    )
    if args.out:
        args.out.write_text(latex, encoding="utf-8")
        print(f"[DONE] Wrote {args.out}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
