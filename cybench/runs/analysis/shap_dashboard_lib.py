"""Build compact SHAP payloads for the country walk-forward dashboard."""

# pyright: reportCallIssue=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, cast

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


def resolve_shap_input_dir(
    *,
    shap_dir: Path | None,
    output_root: Path | None,
    summary_rows: list[dict[str, Any]],
) -> Path | None:
    """Return a SHAP output root, preferring an explicit path then auto-discovery."""
    if shap_dir is not None:
        resolved = shap_dir.resolve()
        return resolved if resolved.is_dir() else None
    if output_root is None or not summary_rows:
        return None
    row = summary_rows[0]
    crop = str(row.get("crop", ""))
    country = str(row.get("country", "")).upper()
    horizon = str(row.get("horizon", "eos"))
    if not crop or not country:
        return None
    candidate = output_root.resolve() / "shap_importance" / f"{crop}_{country}_{horizon}"
    return candidate if candidate.is_dir() else None


def _summary_path(shap_dir: Path, *, crop: str, country: str, model: str) -> Path:
    return shap_dir / f"{crop}_{country.upper()}" / model / "shap_summary.yaml"


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
    shap_dir: Path,
    summary_rows: list[dict[str, Any]],
    *,
    top_features: int = 15,
) -> ShapDashboardPayload:
    """Build dashboard JSON keyed by ``dataset|||model`` from SHAP summaries."""
    empty: ShapDashboardPayload = {
        "available": False,
        "shap_dir": str(shap_dir),
        "by_key": {},
    }
    if not shap_dir.is_dir() or not summary_rows:
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
        summary_path = _summary_path(shap_dir, crop=crop, country=country, model=model)
        if not summary_path.is_file():
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
        shap_dir=str(shap_dir),
        by_key=by_key,
    )


def write_shap_sidecar(output_dir: Path, payload: ShapDashboardPayload) -> Path | None:
    """Persist SHAP dashboard payload next to collect outputs for debugging."""
    if not payload.get("available"):
        return None
    path = output_dir / "shap_dashboard_data.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
