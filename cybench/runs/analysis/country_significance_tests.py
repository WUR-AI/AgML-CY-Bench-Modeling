#!/usr/bin/env python3
"""Country-level bootstrap inference for best-AI vs best-traditional NRMSE (paper §5.2).

Example::

    poetry run python cybench/runs/analysis/country_significance_tests.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --version 2 \\
        --horizon eos \\
        --latex-table ai_bootstrap_table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cybench.runs.analysis.country_significance_lib import (
    DEFAULT_N_BOOTSTRAP,
    analyze_all_crops,
    format_results_latex_sentence,
    format_results_latex_table,
    format_results_markdown_table,
)
from cybench.runs.analysis.global_insights_lib import discover_summary_tables, load_summary_frame


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
        help="Crops to analyze (default: maize wheat)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Country bootstrap replicates (default: {DEFAULT_N_BOOTSTRAP})",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap")
    parser.add_argument(
        "-o",
        "--out-json",
        type=Path,
        help="Write full results JSON (country rows omitted unless --include-countries)",
    )
    parser.add_argument(
        "--include-countries",
        action="store_true",
        help="Include per-country rows in JSON output",
    )
    parser.add_argument(
        "--latex-table",
        type=Path,
        help="Write LaTeX summary table for the paper",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        help="Write LaTeX prose sentences (one per crop)",
    )
    args = parser.parse_args()

    paths = discover_summary_tables(args.output_root.resolve(), version=args.version)
    if not paths:
        raise SystemExit(f"No walk_forward_summary.csv under {args.output_root} (version={args.version})")

    df = load_summary_frame(paths)
    if "nrmse" in df.columns:
        df["nrmse"] = pd.to_numeric(df["nrmse"], errors="coerce")

    results = analyze_all_crops(
        df,
        batch_horizon=args.horizon,
        crops=tuple(args.crops),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    print(format_results_markdown_table(results))
    print()
    for crop in args.crops:
        if crop in results:
            print(format_results_latex_sentence(results[crop]))
            print()

    if args.latex_table:
        args.latex_table.write_text(format_results_latex_table(results), encoding="utf-8")
        print(f"[DONE] Wrote {args.latex_table}")

    if args.out_json:
        payload: dict = {}
        for crop, res in results.items():
            entry = {k: v for k, v in res.items() if k != "countries"}
            if args.include_countries:
                countries = res.get("countries")
                if isinstance(countries, pd.DataFrame):
                    entry["countries"] = countries.to_dict(orient="records")
            payload[crop] = entry
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote {args.out_json}")

    if args.latex:
        lines = [format_results_latex_sentence(results[c]) for c in args.crops if c in results]
        args.latex.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        print(f"[DONE] Wrote {args.latex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
