"""Build compact SHAP payloads for the country walk-forward dashboard."""

# pyright: reportCallIssue=false

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from cybench.runs.analysis.shap_plot_lib import (
    MODEL_LABELS,
    load_feature_table,
    load_shap_summary,
    meta_group_shares,
)


class ShapFeatureEntry(TypedDict):
    name: str
    mean_abs_shap: float
    rank: int


class ShapMetaGroupEntry(TypedDict):
    name: str
    share_pct: float


class ShapDashboardEntry(TypedDict):
    model: str
    model_label: str
    dataset: str
    crop: str
    country: str
    horizon: str
    explainer: str
    n_origins: int
    meta_groups: list[ShapMetaGroupEntry]
    top_features: list[ShapFeatureEntry]


class ShapDashboardPayload(TypedDict):
    available: bool
    shap_dir: str | None
    by_key: dict[str, ShapDashboardEntry]


def _dashboard_key(dataset: str, model: str) -> str:
    return f"{dataset}|||{model}"


def _batch_dir_name(*, crop: str, country: str, horizon: str) -> str:
    return f"{crop}_{country.upper()}_{horizon}"


def resolve_shap_input_dirs(
    *,
    shap_dir: Path | None,
    output_root: Path | None,
    summary_rows: list[dict[str, Any]],
) -> list[Path]:
    """Return SHAP batch dirs for every crop×country×horizon in ``summary_rows``.

    Layout on disk is one directory per crop::

        <output_root>/shap_importance/<crop>_<CC>_<horizon>/<crop>_<CC>/<model>/shap_summary.yaml

    An explicit ``shap_dir`` (single batch or parent) is kept as-is when it exists.
    Auto-discovery scans *all* crops in the summary — not only the first row.
    """
    if shap_dir is not None:
        resolved = shap_dir.resolve()
        return [resolved] if resolved.is_dir() else []
    if output_root is None or not summary_rows:
        return []

    shap_root = output_root.resolve() / "shap_importance"
    found: list[Path] = []
    seen: set[str] = set()
    for row in summary_rows:
        crop = str(row.get("crop", ""))
        country = str(row.get("country", "")).upper()
        horizon = str(row.get("horizon", "eos"))
        if not crop or not country:
            continue
        batch_name = _batch_dir_name(crop=crop, country=country, horizon=horizon)
        if batch_name in seen:
            continue
        seen.add(batch_name)
        candidate = shap_root / batch_name
        if candidate.is_dir():
            found.append(candidate)
    return found


def resolve_shap_input_dir(
    *,
    shap_dir: Path | None,
    output_root: Path | None,
    summary_rows: list[dict[str, Any]],
) -> Path | None:
    """Return the first resolved SHAP batch dir (compat wrapper)."""
    dirs = resolve_shap_input_dirs(
        shap_dir=shap_dir,
        output_root=output_root,
        summary_rows=summary_rows,
    )
    return dirs[0] if dirs else None


def _summary_path(shap_dir: Path, *, crop: str, country: str, model: str) -> Path:
    return shap_dir / f"{crop}_{country.upper()}" / model / "shap_summary.yaml"


def _find_summary_path(
    shap_dirs: Sequence[Path],
    *,
    crop: str,
    country: str,
    model: str,
    horizon: str,
) -> Path | None:
    """Locate ``shap_summary.yaml`` for one summary row across batch dirs."""
    country_u = country.upper()
    batch_name = _batch_dir_name(crop=crop, country=country_u, horizon=horizon)
    # Prefer the matching crop batch dir when several are provided.
    ordered = sorted(
        shap_dirs,
        key=lambda path: (0 if path.name == batch_name else 1, path.name),
    )
    for shap_dir in ordered:
        # Explicit parent ``shap_importance`` root: dig into the crop batch.
        if shap_dir.name == "shap_importance":
            candidate = (
                shap_dir
                / batch_name
                / f"{crop}_{country_u}"
                / model
                / "shap_summary.yaml"
            )
            if candidate.is_file():
                return candidate
            continue
        candidate = _summary_path(shap_dir, crop=crop, country=country_u, model=model)
        if candidate.is_file():
            return candidate
    return None


def _aggregate_top_features(summary: dict[str, Any], *, top_n: int = 15) -> list[ShapFeatureEntry]:
    rows: list[dict[str, Any]] = []
    for origin in summary.get("origins", []):
        if not isinstance(origin, dict):
            continue
        for feat in origin.get("features", []):
            if not isinstance(feat, dict):
                continue
            rows.append(
                {
                    "name": str(feat["name"]),
                    "mean_abs_shap": float(feat["mean_abs_shap"]),
                    "rank": int(feat.get("rank", 0)),
                }
            )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    agg = frame.groupby("name", as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "median"),
        mean_rank=("rank", "mean"),
    )
    sorted_rows = sorted(
        agg.to_dict(orient="records"),
        key=lambda row: (-float(row["mean_abs_shap"]), float(row["mean_rank"])),
    )[:top_n]
    out: list[ShapFeatureEntry] = []
    for rank, row in enumerate(sorted_rows, start=1):
        out.append(
            ShapFeatureEntry(
                name=str(row["name"]),
                mean_abs_shap=round(float(row["mean_abs_shap"]), 8),
                rank=rank,
            )
        )
    return out


def _explainer_from_summary(summary: dict[str, Any]) -> str:
    origins = summary.get("origins") or []
    if not origins:
        return "unknown"
    first = origins[0]
    if isinstance(first, dict) and first.get("explainer"):
        return str(first["explainer"])
    return "unknown"


def build_shap_dashboard_payload(
    shap_dir: Path | Sequence[Path],
    summary_rows: list[dict[str, Any]],
    *,
    top_features: int = 15,
) -> ShapDashboardPayload:
    """Build dashboard JSON keyed by ``dataset|||model`` from SHAP summaries.

    ``shap_dir`` may be one batch directory or several (one per crop). Summary
    paths are resolved per row so maize and wheat for the same country both load.
    """
    shap_dirs = (
        [shap_dir]
        if isinstance(shap_dir, Path)
        else [Path(path) for path in shap_dir]
    )
    shap_dirs = [path for path in shap_dirs if path.is_dir()]
    shap_dir_label = (
        None
        if not shap_dirs
        else (
            str(shap_dirs[0])
            if len(shap_dirs) == 1
            else ",".join(str(path) for path in shap_dirs)
        )
    )
    empty: ShapDashboardPayload = {
        "available": False,
        "shap_dir": shap_dir_label,
        "by_key": {},
    }
    if not shap_dirs or not summary_rows:
        return empty

    summary_paths: list[Path] = []
    by_key: dict[str, ShapDashboardEntry] = {}
    for row in summary_rows:
        crop = str(row.get("crop", ""))
        country = str(row.get("country", "")).upper()
        model = str(row.get("model", ""))
        dataset = str(row.get("dataset", ""))
        horizon = str(row.get("horizon", "eos"))
        if not crop or not country or not model or not dataset:
            continue
        summary_path = _find_summary_path(
            shap_dirs,
            crop=crop,
            country=country,
            model=model,
            horizon=horizon,
        )
        if summary_path is None:
            continue
        summary_paths.append(summary_path)
        summary = load_shap_summary(summary_path)
        by_key[_dashboard_key(dataset, model)] = ShapDashboardEntry(
            model=model,
            model_label=MODEL_LABELS.get(model, model),
            dataset=dataset,
            crop=crop,
            country=country,
            horizon=horizon,
            explainer=_explainer_from_summary(summary),
            n_origins=int(summary.get("n_origins") or len(summary.get("origins") or [])),
            meta_groups=[],
            top_features=_aggregate_top_features(summary, top_n=top_features),
        )

    if not by_key:
        return empty

    feature_table = load_feature_table(summary_paths)
    if not feature_table.empty:
        shares = meta_group_shares(feature_table)
        if not shares.empty:
            for key, entry in by_key.items():
                crop = entry["crop"]
                model = entry["model"]
                horizon = entry["horizon"]
                subset = shares[
                    (shares["crop"] == crop)
                    & (shares["model"] == model)
                    & (shares["horizon"] == horizon)
                ]
                sorted_groups = sorted(
                    subset.to_dict(orient="records"),
                    key=lambda row: -float(row["median_share_pct"]),
                )
                entry["meta_groups"] = [
                    ShapMetaGroupEntry(
                        name=str(row["meta_group"]),
                        share_pct=round(float(row["median_share_pct"]), 2),
                    )
                    for row in sorted_groups
                ]

    return ShapDashboardPayload(
        available=True,
        shap_dir=shap_dir_label,
        by_key=by_key,
    )


def write_shap_sidecar(output_dir: Path, payload: ShapDashboardPayload) -> Path | None:
    """Persist SHAP dashboard payload next to collect outputs for debugging."""
    if not payload.get("available"):
        return None
    path = output_dir / "shap_dashboard_data.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
