"""Parse SHAP outputs and build paper-ready importance summaries and figures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from cybench.runs.analysis.shap_importance_lib import (
    DEFAULT_MAIZE_FAMILY_MODELS,
    coalesce_onehot_feature_name,
)

STAT_SUFFIXES = ("min", "max", "mean", "sum")
_TABULAR_TEMPORAL = re.compile(
    rf"^(.+)_({'|'.join(STAT_SUFFIXES)})_(\d+)$"
)
_CTX = re.compile(r"^ctx:(.+)$")
_TS = re.compile(r"^ts:(.+)$")

META_GROUPS: dict[str, tuple[str, ...]] = {
    "Vegetation": ("ndvi", "cum_ndvi", "fpar", "cum_fpar"),
    "Temperature": ("tmin", "tmax", "tavg", "gdd", "cum_gdd"),
    "Water": ("prec", "cum_prec", "cwb", "cum_cwb", "et0", "vpd"),
    "Radiation": ("rad",),
    "Soil moisture": ("ssm", "rsm"),
    "Static": (
        "latitude",
        "longitude",
        "elevation",
        "awc",
        "bulk_density",
        "drainage_class",
        "cec",
        "clay",
        "ph",
        "sand",
        "silt",
        "soc",
    ),
}

BASE_TO_META: dict[str, str] = {
    base: meta for meta, bases in META_GROUPS.items() for base in bases
}

MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "tabpfn": "TabPFN",
    "tabicl": "TabICL",
    "tabdpt": "TabDPT",
    "transformer_lf": "Transformer",
}


@dataclass(frozen=True)
class ParsedFeature:
    raw: str
    variable_group: str
    statistic: str | None
    window: int | None
    channel: str | None
    meta_group: str


def parse_feature_name(name: str) -> ParsedFeature:
    """Parse tabular ``var_stat_window`` or torch ``ctx:/ts:`` feature names."""
    # Coalesce one-hot dummies first so drainage_class_4 → drainage_class.
    name = coalesce_onehot_feature_name(name)

    ctx_match = _CTX.match(name)
    if ctx_match:
        base = ctx_match.group(1)
        return ParsedFeature(
            raw=name,
            variable_group=base,
            statistic=None,
            window=None,
            channel="ctx",
            meta_group=BASE_TO_META.get(base, "Static"),
        )

    ts_match = _TS.match(name)
    if ts_match:
        base = ts_match.group(1)
        return ParsedFeature(
            raw=name,
            variable_group=base,
            statistic=None,
            window=None,
            channel="ts",
            meta_group=BASE_TO_META.get(base, "Other"),
        )

    tab_match = _TABULAR_TEMPORAL.match(name)
    if tab_match:
        base, statistic, window_s = tab_match.groups()
        return ParsedFeature(
            raw=name,
            variable_group=base,
            statistic=statistic,
            window=int(window_s),
            channel="tabular",
            meta_group=BASE_TO_META.get(base, "Other"),
        )

    return ParsedFeature(
        raw=name,
        variable_group=name,
        statistic=None,
        window=None,
        channel=None,
        meta_group=BASE_TO_META.get(name, "Other"),
    )


def discover_shap_summaries(input_root: Path) -> list[Path]:
    """Return all ``shap_summary.yaml`` files under *input_root*."""
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    return sorted(input_root.rglob("shap_summary.yaml"))


def load_shap_summary(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def feature_rows_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-origin feature lists from one ``shap_summary.yaml``.

    One-hot dummies (e.g. ``drainage_class_4``) are coalesced by summing |SHAP|
    into the parent stem before emitting rows.
    """
    rows: list[dict[str, Any]] = []
    crop = str(summary.get("crop", ""))
    country = str(summary.get("country", ""))
    model = str(summary.get("model", ""))
    horizon = str(summary.get("horizon", ""))
    for origin in summary.get("origins", []):
        if not isinstance(origin, dict):
            continue
        test_years = origin.get("test_years") or []
        origin_year = int(test_years[0]) if test_years else None
        # Sum |SHAP| across one-hot levels within this origin.
        coalesced: dict[str, float] = {}
        for feat in origin.get("features", []):
            if not isinstance(feat, dict):
                continue
            name = coalesce_onehot_feature_name(str(feat["name"]))
            coalesced[name] = coalesced.get(name, 0.0) + float(feat["mean_abs_shap"])
        ranked = sorted(coalesced.items(), key=lambda item: (-item[1], item[0]))
        for rank, (name, value) in enumerate(ranked, start=1):
            if not np.isfinite(value) or value <= 0:
                continue
            parsed = parse_feature_name(name)
            rows.append(
                {
                    "crop": crop,
                    "country": country,
                    "model": model,
                    "horizon": horizon,
                    "origin": origin_year,
                    "feature": parsed.raw,
                    "variable_group": parsed.variable_group,
                    "statistic": parsed.statistic,
                    "window": parsed.window,
                    "channel": parsed.channel,
                    "meta_group": parsed.meta_group,
                    "mean_abs_shap": float(value),
                    "rank": rank,
                }
            )
    return rows


def load_feature_table(summary_paths: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        rows.extend(feature_rows_from_summary(load_shap_summary(path)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _median_by_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(keys, as_index=False)["share_pct"]
        .median()
        .rename(columns={"share_pct": "median_share_pct"})
    )
    return grouped


def meta_group_shares(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Sum |SHAP| within meta-group, normalize to 100% per slice."""
    if feature_table.empty:
        return pd.DataFrame()

    keys = ["crop", "country", "model", "horizon", "origin"]
    grouped = (
        feature_table.groupby(keys + ["meta_group"], as_index=False)["mean_abs_shap"]
        .sum()
        .rename(columns={"mean_abs_shap": "group_abs_shap"})
    )
    totals = grouped.groupby(keys, as_index=False)["group_abs_shap"].transform("sum")
    grouped["share_pct"] = np.where(
        totals > 0, 100.0 * grouped["group_abs_shap"] / totals, 0.0
    )

    per_origin = _median_by_keys(
        grouped,
        ["crop", "country", "model", "horizon", "meta_group"],
    )
    per_country = (
        per_origin.groupby(
            ["crop", "model", "horizon", "meta_group"], as_index=False
        )
        .agg(
            median_share_pct=("median_share_pct", "median"),
            n_countries=("country", "nunique"),
        )
        .sort_values(["model", "median_share_pct"], ascending=[True, False])
    )
    return per_country


def meta_group_consistency(feature_table: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    """Fraction of countries where a meta-group ranks in the top-*k* variable groups."""
    if feature_table.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    slice_keys = ["crop", "country", "model", "horizon", "origin"]
    for key_vals, chunk in feature_table.groupby(slice_keys):
        meta_scores = (
            chunk.groupby("meta_group", as_index=False)["mean_abs_shap"]
            .sum()
            .sort_values("mean_abs_shap", ascending=False)
        )
        top_meta = set(meta_scores.head(top_k)["meta_group"])
        row = dict(zip(slice_keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        for meta in meta_scores["meta_group"]:
            row_copy = dict(row)
            row_copy["meta_group"] = meta
            row_copy["in_top_k"] = meta in top_meta
            rows.append(row_copy)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["crop", "model", "horizon", "meta_group"], as_index=False)
        .agg(
            top_k_country_fraction=("in_top_k", "mean"),
            n_country_origins=("country", "nunique"),
        )
        .sort_values(["model", "top_k_country_fraction"], ascending=[True, False])
    )


def timing_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Window-level shares for tabular models (RF / TabPFN)."""
    if feature_table.empty:
        return pd.DataFrame()

    tabular = feature_table[feature_table["window"].notna()].copy()
    if tabular.empty:
        return pd.DataFrame()

    keys = ["crop", "country", "model", "horizon", "origin"]
    grouped = (
        tabular.groupby(keys + ["variable_group", "window"], as_index=False)[
            "mean_abs_shap"
        ]
        .sum()
        .rename(columns={"mean_abs_shap": "group_abs_shap"})
    )
    totals = grouped.groupby(keys, as_index=False)["group_abs_shap"].transform("sum")
    grouped["share_pct"] = np.where(
        totals > 0, 100.0 * grouped["group_abs_shap"] / totals, 0.0
    )

    per_origin = _median_by_keys(
        grouped,
        ["crop", "country", "model", "horizon", "variable_group", "window"],
    )
    return (
        per_origin.groupby(
            ["crop", "model", "horizon", "variable_group", "window"], as_index=False
        )["median_share_pct"]
        .median()
        .rename(columns={"median_share_pct": "median_share_pct"})
    )


def _model_panel_order(models: Iterable[str]) -> list[str]:
    preferred = [m for m in DEFAULT_MAIZE_FAMILY_MODELS if m in set(models)]
    extras = sorted({m for m in models if m not in preferred})
    return preferred + extras


def window_relative_to_eos(window: int | float) -> int:
    """Map stored EOS-anchored window index to a chronological label.

    Feature design stores ``0`` = EOS window, ``1`` = one aggregate before EOS,
    ``2`` = two before, … Display as ``0, -1, -2, …`` so time runs left→right
    toward harvest.
    """
    return -int(window)


def chrono_window_columns(windows: Iterable[int | float]) -> list[int]:
    """Sort stored window indices early→late (largest index first → EOS last)."""
    return sorted({int(w) for w in windows}, reverse=True)


def plot_meta_group_families(
    shares: pd.DataFrame,
    *,
    crop: str,
    horizon: str,
    models: Sequence[str] | None = None,
    top_n: int = 8,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Three-panel horizontal bar chart of meta-group shares per model family."""
    frame = shares[
        (shares["crop"] == crop) & (shares["horizon"] == horizon)
    ].copy()
    if frame.empty:
        raise ValueError(f"No meta-group shares for crop={crop!r}, horizon={horizon!r}")

    model_list = _model_panel_order(models or frame["model"].unique())
    frame = frame[frame["model"].isin(model_list)]
    top_meta = (
        frame.groupby("meta_group")["median_share_pct"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    frame = frame[frame["meta_group"].isin(top_meta)]
    order = (
        frame.groupby("meta_group")["median_share_pct"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )

    n_models = len(model_list)
    # Do not share y-axes: empty sibling ticklabels under sharey can clear the
    # left-panel category names in some Matplotlib versions.
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 5.2), sharey=False)
    if n_models == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=len(order))
    color_map = dict(zip(order, palette))

    for ax, model in zip(axes, model_list):
        chunk = frame[frame["model"] == model].set_index("meta_group").reindex(order)
        y = np.arange(len(order))
        ax.barh(
            y,
            chunk["median_share_pct"].fillna(0.0).to_numpy(),
            color=[color_map[m] for m in order],
            edgecolor="white",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        is_left = ax is axes[0]
        ax.tick_params(axis="y", labelleft=is_left, length=0 if not is_left else 3)
        ax.set_ylabel("Feature group" if is_left else "")
        ax.set_xlabel("Median share of |SHAP| (%)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        title
        or f"Feature-group importance by model family ({crop}, {horizon})",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_timing_heatmaps(
    timing: pd.DataFrame,
    *,
    crop: str,
    horizon: str,
    models: Sequence[str] | None = None,
    top_variables: int = 12,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Heatmaps of variable group × EOS-anchored window (tabular models only).

    Columns are chronological: early season (e.g. −8) → EOS (0). Stored window
    indices (0 = EOS, 1 = previous aggregate, …) are shown as 0, −1, −2, …
    """
    frame = timing[(timing["crop"] == crop) & (timing["horizon"] == horizon)].copy()
    if frame.empty:
        raise ValueError("No tabular timing data available for heatmaps.")

    model_list = [
        m
        for m in _model_panel_order(models or frame["model"].unique())
        if m in set(frame["model"]) and m != "transformer_lf"
    ]
    if not model_list:
        raise ValueError("No tabular models with window-level SHAP found.")

    top_vars = (
        frame.groupby("variable_group")["median_share_pct"]
        .median()
        .sort_values(ascending=False)
        .head(top_variables)
        .index.tolist()
    )
    frame = frame[frame["variable_group"].isin(top_vars)]

    n_models = len(model_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, 6.0))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, model_list):
        chunk = frame[frame["model"] == model]
        pivot = chunk.pivot_table(
            index="variable_group",
            columns="window",
            values="median_share_pct",
            aggfunc="median",
        )
        var_order = (
            pivot.max(axis=1).sort_values(ascending=False).index.tolist()
        )
        pivot = pivot.reindex(var_order)
        col_order = chrono_window_columns(pivot.columns)
        pivot = pivot.reindex(col_order, axis=1)
        pivot.columns = [window_relative_to_eos(w) for w in pivot.columns]
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            cbar_kws={"label": "Median share (%)"},
            linewidths=0.2,
            linecolor="white",
        )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Windows before EOS (0 = EOS)")
        ax.set_ylabel("Variable group" if ax is axes[0] else "")

    fig.suptitle(
        title
        or f"Seasonal timing of feature importance ({crop}, {horizon})",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def export_tables(
    *,
    feature_table: pd.DataFrame,
    shares: pd.DataFrame,
    timing: pd.DataFrame,
    consistency: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not feature_table.empty:
        feature_table.to_csv(output_dir / "shap_features_long.csv", index=False)
    if not shares.empty:
        shares.to_csv(output_dir / "shap_meta_group_shares.csv", index=False)
    if not timing.empty:
        timing.to_csv(output_dir / "shap_timing_shares.csv", index=False)
    if not consistency.empty:
        consistency.to_csv(output_dir / "shap_meta_group_consistency.csv", index=False)
