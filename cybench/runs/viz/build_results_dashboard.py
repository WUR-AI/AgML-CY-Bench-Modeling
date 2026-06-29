#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cybench.evaluation.aggregated_metrics import compute_report_metrics


@dataclass
class SourceConfig:
    model_name: str
    results_dir: str


FIXED_COLS = {"country_code", "adm_id", "year", "yield", "crop"}


def parse_source(raw: str) -> SourceConfig:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"Invalid source '{raw}'. Expected format MODEL_NAME:/abs/or/rel/path"
        )
    model_name, results_dir = raw.split(":", 1)
    model_name = model_name.strip()
    results_dir = results_dir.strip()
    if not model_name:
        raise argparse.ArgumentTypeError(f"Invalid source '{raw}': empty model name.")
    if not results_dir:
        raise argparse.ArgumentTypeError(f"Invalid source '{raw}': empty path.")
    return SourceConfig(model_name=model_name, results_dir=results_dir)


def discover_stats_json(results_dir: str) -> List[str]:
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        [
            os.path.join(results_dir, fn)
            for fn in os.listdir(results_dir)
            if fn.endswith("_stats.json")
        ]
    )


def discover_run_dirs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        return []
    dirs = []
    for name in sorted(os.listdir(runs_root)):
        fp = os.path.join(runs_root, name)
        if not os.path.isdir(fp):
            continue
        if any(tag in name for tag in ("_rolling_", "_screening_", "_walk_forward_")):
            dirs.append(fp)
    return dirs


def parse_run_dir_name(run_dir: str) -> Dict[str, str]:
    name = os.path.basename(run_dir)
    # <crop>_<country>_<model>_{rolling|screening|walk_forward}_<timestamp>
    out = {"dataset": name, "model": "unknown_model"}
    for phase in ("walk_forward", "screening", "rolling"):
        token = f"_{phase}_"
        if token in name:
            prefix, _ = name.split(token, 1)
            parts = prefix.split("_")
            if len(parts) >= 3:
                out["dataset"] = f"{parts[0]}_{parts[1]}"
                out["model"] = "_".join(parts[2:])
            return out
    parts = name.split("_")
    if len(parts) >= 5:
        out["dataset"] = f"{parts[0]}_{parts[1]}"
        out["model"] = "_".join(parts[2:-3]) if len(parts) > 5 else parts[2]
    return out


def calc_r_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    if len(y_true) < 2:
        return np.nan, np.nan
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else np.nan
    else:
        r2 = float(1.0 - ss_res / ss_tot)
    return r, r2


def calc_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.mean(np.abs(y_true)))
    return np.nan if denom == 0 else float(rmse / denom)


def get_prediction_column(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
    if override:
        return override if override in df.columns else None
    candidates = [c for c in df.columns if c not in FIXED_COLS]
    return candidates[0] if len(candidates) == 1 else None


def load_records_from_runs_root(
    runs_root: str, output_dir: str, prediction_col: Optional[str]
) -> List[Dict]:
    records: List[Dict] = []
    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        print(f"[WARN] No run dirs found in: {runs_root}")
        return records

    for run_dir in run_dirs:
        parsed = parse_run_dir_name(run_dir)
        dataset = parsed["dataset"]
        model_name = parsed["model"]

        # Prefer run-level stats produced by visualize_results_aggregated.py
        # so dashboard values match summary_table outputs exactly.
        stats_fp = os.path.join(run_dir, f"{dataset}_stats.json")
        if os.path.exists(stats_fp):
            with open(stats_fp, "r") as f:
                s = json.load(f)

            panel_dir = os.path.join(run_dir, "report_assets")
            images = {}
            for p in ["map_actual", "map_pred", "scatter", "temporal"]:
                fp = os.path.join(panel_dir, f"{dataset}_{p}.png")
                if os.path.exists(fp):
                    images[p] = os.path.relpath(fp, output_dir).replace(os.sep, "/")

            common = {
                "model": model_name,
                "dataset": dataset,
                "n_regions": s.get("n_regions"),
                "n_years": s.get("n_years"),
                "n_samples": s.get("n_samples"),
                "images": images,
            }
            for view, metric, value in _view_metric_values_from_stats(s):
                records.append(
                    {**common, "view": view, "metric": metric, "value": value}
                )
            continue

        year_csvs = sorted(
            [
                os.path.join(run_dir, fn)
                for fn in os.listdir(run_dir)
                if fn.endswith(".csv") and "_year_" in fn
            ]
        )
        if not year_csvs:
            continue

        df = pd.concat([pd.read_csv(fp) for fp in year_csvs], ignore_index=True)
        pred_col = get_prediction_column(df, prediction_col)
        if pred_col is None:
            print(
                f"[WARN] Could not infer prediction column in {run_dir}; "
                f"use --prediction_col. Columns: {list(df.columns)}"
            )
            continue

        y_true = pd.to_numeric(df["yield"], errors="coerce")
        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
        clean = pd.DataFrame(
            {
                "yield": y_true,
                "pred": y_pred,
                "adm_id": df["adm_id"],
                "year": df["year"],
            }
        ).dropna()
        if clean.empty:
            continue

        report = compute_report_metrics(
            clean.rename(columns={"yield": "yield", "pred": "pred"}),
            "yield",
            "pred",
            loc_col="adm_id",
            year_col="year",
        )

        panel_dir = os.path.join(run_dir, "report_assets")
        images = {}
        for p in ["map_actual", "map_pred", "scatter", "temporal"]:
            fp = os.path.join(panel_dir, f"{dataset}_{p}.png")
            if os.path.exists(fp):
                images[p] = os.path.relpath(fp, output_dir).replace(os.sep, "/")

        common = {
            "model": model_name,
            "dataset": dataset,
            "n_regions": report["n_regions"],
            "n_years": report["n_years"],
            "n_samples": report["n_samples"],
            "images": images,
        }
        for view, metric, value in _view_metric_values_from_report(report):
            records.append({**common, "view": view, "metric": metric, "value": value})

    return records


def infer_prediction_col_for_run(
    run_dir: str, prediction_col_override: Optional[str]
) -> Optional[str]:
    year_csvs = sorted(
        [
            os.path.join(run_dir, fn)
            for fn in os.listdir(run_dir)
            if fn.endswith(".csv") and "_year_" in fn
        ]
    )
    if not year_csvs:
        return None
    sample_df = pd.read_csv(year_csvs[0])
    return get_prediction_column(sample_df, prediction_col_override)


def generate_assets_for_runs(
    run_dirs: List[str], prediction_col_override: Optional[str]
) -> None:
    visualize_script = os.path.join(os.path.dirname(__file__), "visualize_results_aggregated.py")
    for run_dir in run_dirs:
        pred_col = infer_prediction_col_for_run(run_dir, prediction_col_override)
        if pred_col is None:
            print(
                f"[WARN] Skip asset generation for {run_dir}: could not infer prediction column."
            )
            continue
        cmd = [
            sys.executable,
            visualize_script,
            "--results_dir",
            run_dir,
            "--model",
            pred_col,
            "--save_individual",
        ]
        print(f"[INFO] Generating assets for: {run_dir} (model={pred_col})")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[WARN] Asset generation failed for {run_dir}")
            if proc.stderr:
                print(proc.stderr.strip())
        else:
            print(f"[OK] Assets generated for: {run_dir}")


def _view_metric_values_from_report(report: dict) -> list[tuple[str, str, Optional[float]]]:
    ry = report.get("region_year", {})
    sp = report.get("spatial", {})
    tm = report.get("temporal", {})
    an = report.get("anomaly", {})
    return [
        ("region_year", "r", safe_float(ry.get("r"))),
        ("region_year", "r2", safe_float(ry.get("r2"))),
        ("region_year", "nrmse", safe_float(ry.get("nrmse"))),
        ("spatial", "r2", safe_float(sp.get("r2_typical_year"))),
        ("temporal", "r2", safe_float(tm.get("r2_typical_region"))),
        ("anomaly", "r2", safe_float(an.get("r2_typical_region"))),
    ]


def _view_metric_values_from_stats(s: dict) -> list[tuple[str, str, Optional[float]]]:
    m = s.get("metrics_model", {})
    sp = s.get("metrics_spatial", {})
    tm = s.get("metrics_temporal", {})
    an = s.get("metrics_anomaly", {})
    if tm and an:
        return _view_metric_values_from_report(
            {
                "region_year": m,
                "spatial": sp,
                "temporal": tm,
                "anomaly": an,
            }
        )
    return [
        ("region_year", "r", safe_float(m.get("r"))),
        ("region_year", "r2", safe_float(m.get("r2"))),
        ("region_year", "nrmse", safe_float(m.get("nrmse"))),
        (
            "spatial",
            "r2",
            safe_float(sp.get("r2_typical_year", sp.get("r2"))),
        ),
        (
            "temporal",
            "r2",
            safe_float(
                tm.get("r2_typical_region", s.get("r2_time_model"))
                if tm
                else s.get("r2_time_model")
            ),
        ),
        (
            "anomaly",
            "r2",
            safe_float(
                an.get("r2_typical_region", m.get("r2_res"))
                if an
                else m.get("r2_res")
            ),
        ),
    ]

def safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        val = float(v)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None


def load_records(sources: List[SourceConfig], output_dir: str) -> List[Dict]:
    records: List[Dict] = []
    for source in sources:
        stat_files = discover_stats_json(source.results_dir)
        if not stat_files:
            print(f"[WARN] No *_stats.json found in: {source.results_dir}")
            continue

        panel_dir = os.path.join(source.results_dir, "report_assets")
        for stat_fp in stat_files:
            with open(stat_fp, "r") as f:
                s = json.load(f)

            dataset = s.get("dataset")
            if not dataset:
                continue

            panel_paths_abs = {
                "map_actual": os.path.join(panel_dir, f"{dataset}_map_actual.png"),
                "map_pred": os.path.join(panel_dir, f"{dataset}_map_pred.png"),
                "scatter": os.path.join(panel_dir, f"{dataset}_scatter.png"),
                "temporal": os.path.join(panel_dir, f"{dataset}_temporal.png"),
            }
            panel_paths_rel = {
                k: os.path.relpath(v, output_dir).replace(os.sep, "/")
                for k, v in panel_paths_abs.items()
                if os.path.exists(v)
            }

            common = {
                "model": source.model_name,
                "dataset": dataset,
                "n_regions": s.get("n_regions"),
                "n_years": s.get("n_years"),
                "n_samples": s.get("n_samples"),
                "images": panel_paths_rel,
            }

            view_metric_values = _view_metric_values_from_stats(s)

            for view, metric, value in view_metric_values:
                records.append(
                    {
                        **common,
                        "view": view,
                        "metric": metric,
                        "value": value,
                    }
                )
    return records


def build_html(records: List[Dict]) -> str:
    data_json = json.dumps(records)
    template_path = Path(__file__).with_name("dashboard_template.html")
    template = template_path.read_text(encoding="utf-8")
    return template.replace("__DATA_JSON__", data_json)


def bundle_referenced_assets(
    records: List[Dict], output_dir: str, assets_dirname: str = "assets"
) -> List[Dict]:
    """Copy only image assets referenced in records into output_dir/assets_dirname."""
    assets_dir = os.path.join(output_dir, assets_dirname)
    # Start clean to avoid stale files from previous dashboard builds.
    if os.path.isdir(assets_dir):
        shutil.rmtree(assets_dir)
    os.makedirs(assets_dir, exist_ok=True)

    copied_by_source: Dict[str, str] = {}
    copied_by_hash: Dict[str, str] = {}
    used_names = set()
    image_keys = ["map_actual", "map_pred", "scatter", "temporal"]

    def file_sha256(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def unique_dest_name(base_name: str, content_hash: str) -> str:
        root, ext = os.path.splitext(base_name)
        candidate = base_name
        if candidate in used_names:
            candidate = f"{root}_{content_hash[:8]}{ext}"
        return candidate

    for rec in records:
        images = rec.get("images", {})
        if not isinstance(images, dict):
            continue

        for key in image_keys:
            rel_path = images.get(key)
            if not rel_path:
                continue

            abs_src = os.path.abspath(os.path.join(output_dir, rel_path))
            if not os.path.exists(abs_src):
                continue

            if abs_src in copied_by_source:
                images[key] = copied_by_source[abs_src]
                continue

            content_hash = file_sha256(abs_src)
            if content_hash in copied_by_hash:
                rel_dst = copied_by_hash[content_hash]
                copied_by_source[abs_src] = rel_dst
                images[key] = rel_dst
                continue

            src_name = os.path.basename(abs_src)
            dest_name = unique_dest_name(src_name, content_hash)
            used_names.add(dest_name)
            rel_dst = f"{assets_dirname}/{dest_name}"
            abs_dst = os.path.join(assets_dir, dest_name)
            shutil.copy2(abs_src, abs_dst)
            copied_by_hash[content_hash] = rel_dst
            copied_by_source[abs_src] = rel_dst
            images[key] = rel_dst

    print(
        f"[INFO] Bundled unique assets: {len(copied_by_hash)} into {assets_dir}"
    )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Build a local interactive multi-model dashboard."
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        default=[],
        help="Model results source in format MODEL_NAME:/path/to/results_dir (repeatable).",
    )
    parser.add_argument(
        "--runs_root",
        help=(
            "Alternative input mode: root directory with Hydra run folders "
            "(names containing _rolling_, _screening_, or _walk_forward_)."
        ),
    )
    parser.add_argument(
        "--prediction_col",
        help="Prediction column in yearly CSVs (if omitted, auto-detected when unique).",
    )
    parser.add_argument(
        "--generate_assets",
        action="store_true",
        help=(
            "When used with --runs_root, first run visualize_results_aggregated.py "
            "for each run folder to create report_assets PNGs."
        ),
    )
    parser.add_argument(
        "--output_html",
        required=True,
        help="Output HTML file path (e.g., /tmp/dashboard.html).",
    )
    parser.add_argument(
        "--bundle_assets",
        action="store_true",
        help=(
            "Copy only image assets referenced by dashboard rows into a local "
            "folder next to --output_html and rewrite paths to that folder."
        ),
    )
    parser.add_argument(
        "--assets_dirname",
        default="assets",
        help="Folder name for bundled assets under the output HTML directory.",
    )
    args = parser.parse_args()

    output_html = os.path.abspath(args.output_html)
    output_dir = os.path.dirname(output_html)
    os.makedirs(output_dir, exist_ok=True)

    records: List[Dict] = []
    if args.source:
        records.extend(load_records(args.source, output_dir))
    if args.runs_root:
        if args.generate_assets:
            run_dirs = discover_run_dirs(args.runs_root)
            generate_assets_for_runs(
                run_dirs=run_dirs, prediction_col_override=args.prediction_col
            )
        records.extend(
            load_records_from_runs_root(
                runs_root=args.runs_root,
                output_dir=output_dir,
                prediction_col=args.prediction_col,
            )
        )

    if not records:
        raise RuntimeError(
            "No records found. Check --source paths and/or --runs_root inputs."
        )

    if args.bundle_assets:
        records = bundle_referenced_assets(
            records=records,
            output_dir=output_dir,
            assets_dirname=args.assets_dirname,
        )

    html = build_html(records)
    with open(output_html, "w") as f:
        f.write(html)

    print(f"[DONE] Dashboard written to: {output_html}")
    print(f"[INFO] Records loaded: {len(records)}")


if __name__ == "__main__":
    main()
