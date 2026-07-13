#!/usr/bin/env python3
"""Export wide region×year prediction tables from collected walk-forward output.

Reads ``paper_walk_forward_*`` directories produced by ``collect_walk_forward_results.py``
and writes one CSV per country (or per crop×country) with model predictions as columns.

Example::

    poetry run python cybench/runs/analysis/export_wide_predictions.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --dest ../output/wide_predictions_v2_eos \\
        --horizon eos --version 2 --zip

    poetry run python cybench/runs/analysis/export_wide_predictions.py \\
        --output-root /lustre/backup/SHARED/AIN/agml/output \\
        --dest ../output/wide_predictions_DE \\
        --country DE --horizon eos --version 2
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.runs.analysis.collect_walk_forward_results import load_pooled_predictions
from cybench.runs.analysis.global_insights_lib import parse_paper_dir_name
from cybench.runs.viz.region_map_lib import (
    dataset_crop,
    infer_pred_column,
    load_dataset_year_csvs,
    preds_dir_for_row,
)

SplitMode = Literal["country", "crop"]
SourceMode = Literal["collect", "baselines"]

_INDEX_COLS = ("adm_id", "year", "crop", "yield")


@dataclass(frozen=True)
class CollectBundle:
    country: str
    horizon: str
    version: int
    path: Path


def discover_collect_bundles(
    output_root: Path,
    *,
    version: int,
    countries: set[str] | None = None,
    horizons: set[str] | None = None,
) -> list[CollectBundle]:
    if not output_root.is_dir():
        return []
    bundles: list[CollectBundle] = []
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_paper_dir_name(entry.name)
        if parsed is None:
            continue
        country, horizon, ver = parsed
        if ver != version:
            continue
        if countries and country not in countries:
            continue
        if horizons and horizon not in horizons:
            continue
        summary = entry / "walk_forward_summary.csv"
        if not summary.is_file():
            continue
        bundles.append(
            CollectBundle(country=country, horizon=horizon, version=ver, path=entry)
        )
    return bundles


def _loc_column(df: pd.DataFrame) -> str:
    if KEY_LOC in df.columns:
        return KEY_LOC
    if "adm_id" in df.columns:
        return "adm_id"
    raise ValueError(f"No location column in prediction frame (columns={list(df.columns)})")


def _year_column(df: pd.DataFrame) -> str:
    if KEY_YEAR in df.columns:
        return KEY_YEAR
    if "year" in df.columns:
        return "year"
    if "harvest_year" in df.columns:
        return "harvest_year"
    raise ValueError(f"No year column in prediction frame (columns={list(df.columns)})")


def _target_column(df: pd.DataFrame) -> str:
    if KEY_TARGET in df.columns:
        return KEY_TARGET
    if "yield" in df.columns:
        return "yield"
    if "targets" in df.columns:
        return "targets"
    raise ValueError(f"No yield column in prediction frame (columns={list(df.columns)})")


def normalize_prediction_frame(
    df: pd.DataFrame,
    *,
    model_slug: str,
    model_col: str | None,
    dataset: str,
) -> pd.DataFrame:
    """Return adm_id, year, crop, yield, <model_slug> from a pooled prediction frame."""
    loc_col = _loc_column(df)
    year_col = _year_column(df)
    target_col = _target_column(df)
    pred_col = infer_pred_column(df, model_col=model_col)
    if pred_col is None:
        raise ValueError(
            f"Could not infer prediction column for model={model_slug!r} "
            f"dataset={dataset!r}"
        )

    out = pd.DataFrame(
        {
            "adm_id": df[loc_col].astype(str),
            "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
            "crop": dataset_crop(dataset),
            "yield": pd.to_numeric(df[target_col], errors="coerce"),
            model_slug: pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["adm_id", "year", "yield", model_slug])


def dedupe_prediction_rows(
    df: pd.DataFrame,
    *,
    model_slug: str,
) -> tuple[pd.DataFrame, int]:
    """Collapse duplicate (adm_id, year, crop) rows by averaging yield and predictions."""
    keys = ["adm_id", "year", "crop"]
    if not df.duplicated(subset=keys, keep=False).any():
        return df, 0
    n_before = len(df)
    deduped = (
        df.groupby(keys, as_index=False)[["yield", model_slug]]
        .mean()
        .astype({"year": "Int64"})
    )
    return deduped, n_before - len(deduped)


def _year_csv_files(preds_dir: Path, dataset: str) -> list[Path]:
    return sorted(preds_dir.glob(f"{dataset}_h*_year_*.csv"))


def diagnose_model_predictions(
    row: dict[str, Any],
    collect_dir: Path,
    *,
    source: SourceMode,
    seed: int | None,
) -> dict[str, Any]:
    """Report why (adm_id, year, crop) keys are duplicated for one model."""
    model = str(row["model"])
    dataset = str(row["dataset"])
    crop = dataset_crop(dataset)
    report: dict[str, Any] = {"model": model, "dataset": dataset, "crop": crop}

    if source == "baselines":
        frame = load_model_predictions(row, collect_dir, source=source, seed=seed)
        if frame is None or frame.empty:
            report["error"] = "no predictions loaded"
            return report
        dup_mask = frame.duplicated(subset=["adm_id", "year", "crop"], keep=False)
        report["n_rows"] = len(frame)
        report["n_duplicate_rows"] = int(dup_mask.sum())
        if dup_mask.any():
            sample = (
                frame.loc[dup_mask, ["adm_id", "year", "crop"]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            report["sample_duplicate_keys"] = sample
            report["note"] = "Duplicates come from pooled walk-forward test_preds.csv files"
        return report

    preds_dir = preds_dir_for_row(collect_dir, row)
    if preds_dir is None:
        report["error"] = "missing preds dir"
        return report
    files = _year_csv_files(preds_dir, dataset)
    report["preds_dir"] = str(preds_dir)
    report["n_year_csv_files"] = len(files)

    horizon_tags = sorted(
        {
            m.group(1)
            for fp in files
            if (m := re.search(rf"{re.escape(dataset)}_h(.+)_year_\d+\.csv$", fp.name))
        }
    )
    report["horizon_tags_in_filenames"] = horizon_tags

    files_by_year: dict[str, list[str]] = {}
    within_file_dupes: list[dict[str, Any]] = []
    for fp in files:
        year_match = re.search(r"_year_(\d+)\.csv$", fp.name)
        year = year_match.group(1) if year_match else "?"
        files_by_year.setdefault(year, []).append(fp.name)
        try:
            raw = pd.read_csv(fp)
        except OSError as exc:
            within_file_dupes.append({"file": fp.name, "error": str(exc)})
            continue
        if raw.empty:
            continue
        loc_col = _loc_column(raw)
        year_col = _year_column(raw)
        dup_mask = raw.duplicated(subset=[loc_col, year_col], keep=False)
        if dup_mask.any():
            within_file_dupes.append(
                {
                    "file": fp.name,
                    "duplicate_rows": int(dup_mask.sum()),
                    "sample_keys": (
                        raw.loc[dup_mask, [loc_col, year_col]]
                        .drop_duplicates()
                        .head(3)
                        .rename(columns={loc_col: "adm_id", year_col: "year"})
                        .to_dict(orient="records")
                    ),
                }
            )

    years_with_multiple_files = {
        year: names for year, names in sorted(files_by_year.items()) if len(names) > 1
    }
    report["years_with_multiple_files"] = years_with_multiple_files
    report["within_file_duplicates"] = within_file_dupes

    frame = load_model_predictions(row, collect_dir, source=source, seed=seed)
    if frame is None or frame.empty:
        report["error"] = "no predictions loaded after concat"
        return report
    dup_mask = frame.duplicated(subset=["adm_id", "year", "crop"], keep=False)
    report["n_rows_after_concat"] = len(frame)
    report["n_duplicate_rows_after_concat"] = int(dup_mask.sum())
    if dup_mask.any():
        report["sample_duplicate_keys"] = (
            frame.loc[dup_mask, ["adm_id", "year", "crop"]]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
    if years_with_multiple_files:
        report["likely_cause"] = "multiple year CSV files for the same harvest year"
    elif within_file_dupes:
        report["likely_cause"] = "duplicate adm_id+year rows inside one or more year CSV files"
    elif int(dup_mask.sum()) > 0:
        report["likely_cause"] = "duplicate keys after concat (inspect year CSV set)"
    else:
        report["likely_cause"] = "no duplicates in this model frame"
    return report


def print_collect_diagnosis(bundle: CollectBundle, *, source: SourceMode, seed: int | None) -> None:
    summary_rows = pd.read_csv(bundle.path / "walk_forward_summary.csv").to_dict(
        orient="records"
    )
    print(f"\n=== diagnose {bundle.path.name} ===")
    for row in summary_rows:
        report = diagnose_model_predictions(
            row, bundle.path, source=source, seed=seed
        )
        model = report["model"]
        print(f"\n[{model}] dataset={report.get('dataset')}")
        if report.get("error"):
            print(f"  error: {report['error']}")
            continue
        if source == "collect":
            print(f"  preds_dir: {report.get('preds_dir')}")
            print(f"  year CSV files: {report.get('n_year_csv_files')}")
            tags = report.get("horizon_tags_in_filenames") or []
            if tags:
                print(f"  horizon tags in filenames: {', '.join(tags)}")
            multi = report.get("years_with_multiple_files") or {}
            if multi:
                print("  years with >1 CSV file:")
                for year, names in multi.items():
                    print(f"    {year}: {names}")
            within = report.get("within_file_duplicates") or []
            if within:
                print("  within-file duplicate adm_id+year:")
                for item in within:
                    print(f"    {item}")
        n_dup = report.get("n_duplicate_rows_after_concat", report.get("n_duplicate_rows"))
        print(f"  rows after load: {report.get('n_rows_after_concat', report.get('n_rows'))}")
        print(f"  duplicate rows: {n_dup}")
        if report.get("likely_cause"):
            print(f"  likely cause: {report['likely_cause']}")
        if report.get("sample_duplicate_keys"):
            print(f"  sample keys: {report['sample_duplicate_keys']}")


def load_model_predictions(
    row: dict[str, Any],
    collect_dir: Path,
    *,
    source: SourceMode,
    seed: int | None,
) -> pd.DataFrame | None:
    model = str(row["model"])
    dataset = str(row["dataset"])
    model_col = str(row.get("model_col") or "") or None

    if source == "baselines":
        run_dir = Path(str(row["run_dir"]))
        use_seed = seed if seed is not None else int(row.get("plot_seed") or 42)
        try:
            df, resolved_col = load_pooled_predictions(
                run_dir, model_slug=model, seed=use_seed
            )
        except (ValueError, OSError):
            return None
        return normalize_prediction_frame(
            df,
            model_slug=model,
            model_col=resolved_col,
            dataset=dataset,
        )

    preds_dir = preds_dir_for_row(collect_dir, row)
    if preds_dir is None:
        return None
    df = load_dataset_year_csvs(preds_dir, dataset)
    if df is None or df.empty:
        return None
    return normalize_prediction_frame(
        df,
        model_slug=model,
        model_col=model_col,
        dataset=dataset,
    )


def _finalize_model_frame(
    frame: pd.DataFrame,
    *,
    model: str,
) -> tuple[pd.DataFrame, int]:
    frame, n_dupes = dedupe_prediction_rows(frame, model_slug=model)
    return frame, n_dupes


def build_wide_table(
    summary_rows: list[dict[str, Any]],
    collect_dir: Path,
    *,
    source: SourceMode = "collect",
    seed: int | None = None,
    crops: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge per-model long frames into one wide region×year table."""
    model_order: list[str] = []
    merged: pd.DataFrame | None = None
    seeds_seen: set[int] = set()
    skipped_models: list[str] = []
    duplicate_rows_collapsed: dict[str, int] = {}

    for row in summary_rows:
        crop = dataset_crop(str(row["dataset"]))
        if crops and crop not in crops:
            continue
        model = str(row["model"])
        frame = load_model_predictions(row, collect_dir, source=source, seed=seed)
        if frame is None or frame.empty:
            skipped_models.append(model)
            continue
        frame, n_dupes = _finalize_model_frame(frame, model=model)
        if n_dupes:
            duplicate_rows_collapsed[model] = n_dupes
        if seed is not None:
            seeds_seen.add(seed)
        elif row.get("plot_seed") is not None and not pd.isna(row.get("plot_seed")):
            seeds_seen.add(int(row["plot_seed"]))

        model_only = frame[["adm_id", "year", "crop", model]].copy()
        if merged is None:
            merged = frame[list(_INDEX_COLS)].copy()
            model_order.append(model)
            merged[model] = frame[model].values
            continue

        merged = merged.merge(
            model_only,
            on=["adm_id", "year", "crop"],
            how="outer",
            validate="one_to_one",
        )
        if model not in model_order:
            model_order.append(model)

    if merged is None or merged.empty:
        raise ValueError(f"No prediction rows found under {collect_dir}")

    merged = merged.sort_values(["crop", "adm_id", "year"]).reset_index(drop=True)
    meta = {
        "source_dir": str(collect_dir.resolve()),
        "source": source,
        "seed": sorted(seeds_seen) if seeds_seen else None,
        "models": model_order,
        "skipped_models": skipped_models,
        "duplicate_rows_collapsed": duplicate_rows_collapsed,
        "n_rows": int(len(merged)),
        "crops": sorted(merged["crop"].dropna().unique().tolist()),
    }
    return merged, meta


def output_basename(
    bundle: CollectBundle,
    *,
    split: SplitMode,
    crop: str | None = None,
) -> str:
    cc = bundle.country.lower()
    if split == "crop":
        if not crop:
            raise ValueError("crop is required when split='crop'")
        return f"{crop}_{cc}_{bundle.horizon}_v{bundle.version}_preds"
    return f"{cc}_{bundle.horizon}_v{bundle.version}_preds"


def write_wide_csv(df: pd.DataFrame, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, float_format="%.6f")
    return dest


def export_bundle(
    bundle: CollectBundle,
    dest_root: Path,
    *,
    split: SplitMode,
    source: SourceMode,
    seed: int | None,
    crops: set[str] | None,
) -> list[dict[str, Any]]:
    summary_path = bundle.path / "walk_forward_summary.csv"
    summary_rows = pd.read_csv(summary_path).to_dict(orient="records")
    written: list[dict[str, Any]] = []

    if split == "country":
        wide, meta = build_wide_table(
            summary_rows,
            bundle.path,
            source=source,
            seed=seed,
            crops=crops,
        )
        out_path = dest_root / f"{output_basename(bundle, split=split)}.csv"
        write_wide_csv(wide, out_path)
        written.append(
            {
                "country": bundle.country,
                "horizon": bundle.horizon,
                "version": bundle.version,
                "crop": None,
                "path": str(out_path),
                **meta,
            }
        )
        return written

    crop_values = sorted(
        {
            dataset_crop(str(row["dataset"]))
            for row in summary_rows
            if dataset_crop(str(row["dataset"]))
        }
    )
    if crops:
        crop_values = [c for c in crop_values if c in crops]
    for crop in crop_values:
        wide, meta = build_wide_table(
            summary_rows,
            bundle.path,
            source=source,
            seed=seed,
            crops={crop},
        )
        out_path = dest_root / f"{output_basename(bundle, split=split, crop=crop)}.csv"
        write_wide_csv(wide, out_path)
        written.append(
            {
                "country": bundle.country,
                "horizon": bundle.horizon,
                "version": bundle.version,
                "crop": crop,
                "path": str(out_path),
                **meta,
            }
        )
    return written


def write_readme(dest_root: Path, *, records: list[dict[str, Any]], version: int) -> Path:
    lines = [
        "CY-Bench walk-forward predictions (wide format)",
        "",
        "One row per region × harvest year × crop.",
        "Columns: adm_id, year, crop, yield, then one column per model (seed repetitions pooled to a single seed at collect time).",
        f"Batch version: v{version}",
        "",
        "Files:",
    ]
    for rec in records:
        name = Path(rec["path"]).name
        seed_repr = rec.get("seed")
        models = rec.get("models") or []
        lines.append(
            f"  - {name}: {rec['country']} {rec.get('crop') or 'maize+wheat'} "
            f"{rec['horizon']} | {rec['n_rows']} rows | {len(models)} models | seed={seed_repr}"
        )
    readme = dest_root / "README.txt"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def zip_exports(dest_root: Path, *, zip_name: str | None = None) -> Path:
    archive = dest_root / (zip_name or f"{dest_root.name}.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dest_root.rglob("*")):
            if not path.is_file() or path == archive:
                continue
            zf.write(path, arcname=path.relative_to(dest_root).as_posix())
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/lustre/backup/SHARED/AIN/agml/output"),
        help="Root containing paper_walk_forward_* collect directories",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Directory for exported CSV files (and optional zip/README)",
    )
    parser.add_argument("--version", type=int, default=2, help="Collect batch version tag")
    parser.add_argument(
        "--horizon",
        action="append",
        dest="horizons",
        help="Limit to horizon(s): eos, mid, qtr, early (repeatable)",
    )
    parser.add_argument(
        "--country",
        action="append",
        dest="countries",
        metavar="CC",
        help="Limit to ISO-2 country code(s) (repeatable)",
    )
    parser.add_argument(
        "--crop",
        action="append",
        dest="crops",
        choices=["maize", "wheat"],
        help="Limit to crop(s) (repeatable)",
    )
    parser.add_argument(
        "--split",
        choices=["country", "crop"],
        default="country",
        help="One CSV per country (both crops) or per crop×country (default: country)",
    )
    parser.add_argument(
        "--source",
        choices=["collect", "baselines"],
        default="collect",
        help="Read pooled preds from collect output (default) or re-load from run_dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed when --source baselines (default: plot_seed from summary, usually 42)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a zip archive of all exported files in --dest",
    )
    parser.add_argument(
        "--zip-name",
        help="Zip filename inside --dest (default: <dest.name>.zip)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print duplicate-key analysis for collect preds and exit (no export)",
    )
    args = parser.parse_args(argv)

    if not args.diagnose and args.dest is None:
        parser.error("--dest is required unless --diagnose is set")

    countries = {c.upper() for c in args.countries} if args.countries else None
    horizons = set(args.horizons) if args.horizons else None
    crops = set(args.crops) if args.crops else None

    bundles = discover_collect_bundles(
        args.output_root.resolve(),
        version=args.version,
        countries=countries,
        horizons=horizons,
    )
    if not bundles:
        raise SystemExit(
            f"No paper_walk_forward_* collect dirs with walk_forward_summary.csv "
            f"under {args.output_root} (version={args.version})"
        )

    if args.diagnose:
        for bundle in bundles:
            print_collect_diagnosis(bundle, source=args.source, seed=args.seed)
        return 0

    dest_root = args.dest.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for bundle in bundles:
        try:
            records = export_bundle(
                bundle,
                dest_root,
                split=args.split,
                source=args.source,
                seed=args.seed,
                crops=crops,
            )
        except ValueError as exc:
            print(f"[SKIP] {bundle.path.name}: {exc}")
            continue
        for rec in records:
            rel = Path(rec["path"]).name
            seed_repr = rec.get("seed")
            print(
                f"[OK] {rel} | {rec['n_rows']} rows | "
                f"{len(rec.get('models') or [])} models | seed={seed_repr}"
            )
            dupes = rec.get("duplicate_rows_collapsed") or {}
            if dupes:
                total = sum(dupes.values())
                models = ", ".join(f"{m}({n})" for m, n in sorted(dupes.items()))
                print(
                    f"     [WARN] collapsed {total} duplicate region×year row(s) "
                    f"(likely overlapping year CSVs in preds/): {models}"
                )
            if rec.get("skipped_models"):
                print(f"     skipped models: {', '.join(rec['skipped_models'])}")
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No prediction files were written")

    manifest_path = dest_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(all_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(dest_root, records=all_records, version=args.version)
    print(f"[DONE] manifest: {manifest_path}")

    if args.zip:
        archive = zip_exports(dest_root, zip_name=args.zip_name)
        print(f"[DONE] zip: {archive} ({archive.stat().st_size / (1024 * 1024):.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
