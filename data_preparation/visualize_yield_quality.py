"""Visualize yield quality flags for manual inspection.

Produces one PNG report per crop/country (summary, panels, residuals stacked).

    Hydra overrides examples::

    crop_name=maize country_code=BR target.quality.outlier_threshold=4.5
    directory=cybench/testdata output_dir=output/yield_quality_viz/maize_BR
    target.filter_samples=[flag_yield_outlier]
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from cybench.config import KEY_LOC, KEY_TARGET, PATH_DATA_DIR
from cybench.datasets.yield_quality import (
    FLAG_AREA,
    FLAG_COLUMNS,
    FLAG_CONSECUTIVE,
    FLAG_YIELD,
    HARVEST_YEAR,
    YieldQualitySettings,
    assess_yield_dataframe,
    configured_yield_quality_settings,
    iter_yield_files,
    merge_yield_with_quality,
    viz_flag_columns,
    yield_quality_settings_from_target,
)

try:
    from numpy.exceptions import RankWarning  # type: ignore[attr-defined]
except ImportError:  # NumPy < 2
    RankWarning = np.RankWarning  # type: ignore[attr-defined,misc]

FLAG_LABELS = {
    FLAG_CONSECUTIVE: "Consecutive / stagnant yield",
    FLAG_AREA: "Area outlier",
    FLAG_YIELD: "High yield outlier (polyfit)",
    "invalid_yield": "Invalid yield (≤ 0 or missing)",
}

FLAG_COLORS = {
    FLAG_CONSECUTIVE: "#e67e22",
    FLAG_AREA: "#9b59b6",
    FLAG_YIELD: "#e74c3c",
}


def _poly_trend(
    years: np.ndarray,
    values: np.ndarray,
    *,
    degree: int = 2,
) -> np.ndarray | None:
    if len(years) < 5:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RankWarning)
            coeffs = np.polyfit(years, values, degree)
    except RankWarning:
        return None
    return np.poly1d(coeffs)(years)


def _load_annotated(
    crop: str,
    country_code: str,
    data_dir: Path,
    *,
    settings: YieldQualitySettings,
) -> pd.DataFrame:
    root = data_dir / crop / country_code
    quality_path = root / f"yield_quality_{crop}_{country_code}.csv"
    yield_path = root / f"yield_{crop}_{country_code}.csv"
    if not yield_path.is_file():
        raise FileNotFoundError(f"Missing yield file: {yield_path}")

    if quality_path.is_file():
        yield_df = pd.read_csv(yield_path)
        quality_df = pd.read_csv(quality_path)
        df = merge_yield_with_quality(yield_df, quality_df)
        return df.sort_values([KEY_LOC, HARVEST_YEAR]).reset_index(drop=True)

    annotated, _ = assess_yield_dataframe(
        pd.read_csv(yield_path),
        settings=settings,
    )
    return annotated


def _quality_csv_has_flags(
    quality_path: Path,
    flag_columns: list[str] | None = None,
) -> bool:
    if not quality_path.is_file():
        return False
    df = pd.read_csv(quality_path)
    cols = flag_columns or list(FLAG_COLUMNS)
    if not set(cols).issubset(df.columns):
        return False
    return bool(np.asarray(df[cols].astype(bool).any(axis=1)).any())


def run_yield_quality_visualizations(
    data_dir: Path,
    crops: list[str],
    *,
    settings: YieldQualitySettings,
    output_root: Path,
    max_panels: int = 9,
    flags: list[str] | None = None,
    countries: list[str] | None = None,
    only_if_flagged: bool = True,
) -> list[Path]:
    """Generate PNG diagnostics for each crop/country under ``data_dir``."""
    saved: list[Path] = []
    jobs = list(iter_yield_files(data_dir, crops=crops))
    for crop, country_code, csv_path in jobs:
        if countries is not None and country_code not in countries:
            continue
        quality_path = csv_path.with_name(f"yield_quality_{crop}_{country_code}.csv")
        check_flags = flags or list(FLAG_COLUMNS)
        if only_if_flagged and not _quality_csv_has_flags(quality_path, check_flags):
            continue
        out_dir = output_root
        try:
            paths = visualize_yield_quality(
                crop,
                country_code,
                data_dir=data_dir,
                output_dir=out_dir,
                max_panels=max_panels,
                flags=flags,
                settings=settings,
            )
        except FileNotFoundError as exc:
            print(f"Skip viz {crop}/{country_code}: {exc}")
            continue
        saved.extend(paths)
        print(f"Viz {crop}/{country_code}: → {paths[0]}")
    return saved


def _area_column(df: pd.DataFrame) -> str | None:
    for name in ("harvest_area", "planted_area"):
        if name in df.columns:
            return name
    return None


def _invalid_yield(df: pd.DataFrame) -> pd.Series:
    invalid = df[KEY_TARGET].isna() | df[KEY_TARGET].le(0)
    return invalid.fillna(True)


def _yield_outlier_flags(df: pd.DataFrame) -> pd.Series:
    """Polyfit yield-outlier flags, excluding invalid-yield rows for display."""
    return df[FLAG_YIELD].astype(bool) & ~_invalid_yield(df)


def _flagged_invalid_yield(df: pd.DataFrame) -> pd.Series:
    return df[FLAG_YIELD].astype(bool) & _invalid_yield(df)


def _summary_flag_counts(
    df: pd.DataFrame,
    flags: list[str] | None = None,
) -> dict[str, int]:
    """Counts for summary plots (yield split into polyfit vs invalid)."""
    all_counts = {
        FLAG_LABELS[FLAG_CONSECUTIVE]: int(np.asarray(df[FLAG_CONSECUTIVE], dtype=bool).sum()),
        FLAG_LABELS[FLAG_AREA]: int(np.asarray(df[FLAG_AREA], dtype=bool).sum()),
        FLAG_LABELS[FLAG_YIELD]: int(_yield_outlier_flags(df).sum()),
        FLAG_LABELS["invalid_yield"]: int(_flagged_invalid_yield(df).sum()),
    }
    if flags is None:
        return all_counts
    selected: dict[str, int] = {}
    if FLAG_CONSECUTIVE in flags:
        selected[FLAG_LABELS[FLAG_CONSECUTIVE]] = all_counts[FLAG_LABELS[FLAG_CONSECUTIVE]]
    if FLAG_AREA in flags:
        selected[FLAG_LABELS[FLAG_AREA]] = all_counts[FLAG_LABELS[FLAG_AREA]]
    if FLAG_YIELD in flags:
        selected[FLAG_LABELS[FLAG_YIELD]] = all_counts[FLAG_LABELS[FLAG_YIELD]]
        selected[FLAG_LABELS["invalid_yield"]] = all_counts[FLAG_LABELS["invalid_yield"]]
    return selected


def _draw_summary_counts(
    ax_bar: plt.Axes,
    ax_pie: plt.Axes,
    df: pd.DataFrame,
    *,
    crop: str,
    country_code: str,
    flags: list[str] | None = None,
) -> None:
    counts = _summary_flag_counts(df, flags=flags)
    check_cols = list(flags) if flags else list(FLAG_COLUMNS)
    any_flagged = int(np.asarray(df[check_cols].astype(bool).any(axis=1), dtype=bool).sum())
    poly_count = counts.get(FLAG_LABELS[FLAG_YIELD], 0)
    invalid_count = counts.get(FLAG_LABELS["invalid_yield"], 0)

    labels = list(counts.keys())
    values = list(counts.values())
    bar_colors = []
    for label in labels:
        if label == FLAG_LABELS[FLAG_CONSECUTIVE]:
            bar_colors.append(FLAG_COLORS[FLAG_CONSECUTIVE])
        elif label == FLAG_LABELS[FLAG_AREA]:
            bar_colors.append(FLAG_COLORS[FLAG_AREA])
        elif label == FLAG_LABELS[FLAG_YIELD]:
            bar_colors.append(FLAG_COLORS[FLAG_YIELD])
        else:
            bar_colors.append("#7f8c8d")

    ax_bar.barh(labels, values, color=bar_colors)
    ax_bar.set_xlabel("Flagged rows")
    ax_bar.set_title("Flag counts (non-exclusive)", fontsize=11, pad=12)
    if values and max(values) > 0:
        xmax = max(values)
        for idx, value in enumerate(values):
            ax_bar.text(value + xmax * 0.02, idx, str(value), va="center", fontsize=10)

    sizes = [any_flagged, len(df) - any_flagged]
    pie_colors = ["#c0392b", "#27ae60"]
    wedges, _ = ax_pie.pie(
        sizes,
        colors=pie_colors,
        startangle=90,
        counterclock=False,
    )
    ax_pie.legend(
        wedges,
        [
            f"Flagged ({any_flagged:,})",
            f"Clean ({len(df) - any_flagged:,})",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    ax_pie.set_title("Sample split", fontsize=11, pad=12)
    _ = poly_count, invalid_count, crop, country_code


def _plot_admin_series(
    ax: plt.Axes,
    group: pd.DataFrame,
    *,
    highlight_flags: tuple[str, ...],
    area_col: str | None,
    poly_yield_flags: pd.Series | None = None,
    polyfit_degree: int = 2,
) -> None:
    group = group.sort_values(HARVEST_YEAR)
    years = group[HARVEST_YEAR].to_numpy(dtype=float)
    yields = group[KEY_TARGET].to_numpy(dtype=float)
    valid = yields > 0

    if valid.any():
        ax.plot(
            years[valid],
            yields[valid],
            color="#2c3e50",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Yield",
        )
        trend = _poly_trend(years[valid], yields[valid], degree=polyfit_degree)
        if trend is not None:
            ax.plot(
                years[valid],
                trend,
                color="#3498db",
                linestyle="--",
                linewidth=1.2,
                label="Poly trend (valid yrs)",
            )

    invalid = cast(pd.DataFrame, group[_invalid_yield(group)])
    if not invalid.empty:
        invalid_yields = np.nan_to_num(
            np.asarray(invalid[KEY_TARGET], dtype=float),
            nan=0.0,
        )
        ax.scatter(
            invalid[HARVEST_YEAR],
            invalid_yields,
            s=70,
            marker="x",
            color="#7f8c8d",
            linewidths=2,
            zorder=5,
            label=FLAG_LABELS["invalid_yield"],
        )

    for flag in highlight_flags:
        if flag == FLAG_YIELD and poly_yield_flags is not None:
            flagged = cast(
                pd.DataFrame,
                group.loc[poly_yield_flags.loc[group.index].astype(bool)],
            )
        else:
            flagged = cast(pd.DataFrame, group.loc[group[flag].astype(bool)])
        if flagged.empty:
            continue
        flagged = flagged[~_invalid_yield(flagged)]
        if flagged.empty:
            continue
        ax.scatter(
            flagged[HARVEST_YEAR],
            flagged[KEY_TARGET],
            s=70,
            color=FLAG_COLORS[flag],
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
            label=FLAG_LABELS[flag],
        )

    ax.set_xlabel("Harvest year")
    ax.set_ylabel("Yield (t/ha)")
    ax.grid(alpha=0.25)

    if area_col is not None and area_col in group.columns:
        ax2 = ax.twinx()
        ax2.plot(
            years,
            group[area_col],
            color="#95a5a6",
            linewidth=1.0,
            alpha=0.7,
            label=area_col.replace("_", " "),
        )
        ax2.set_ylabel(area_col.replace("_", " ").title())
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    else:
        ax.legend(fontsize=7, loc="upper left")


def _select_admin_units(
    df: pd.DataFrame,
    flag: str,
    *,
    max_panels: int,
    prefer_exclusive: bool,
    poly_yield_flags: pd.Series | None = None,
) -> list[str]:
    if flag == FLAG_YIELD and poly_yield_flags is not None:
        mask = poly_yield_flags.astype(bool)
    else:
        mask = df[flag].astype(bool)

    if prefer_exclusive:
        other_flags = [col for col in FLAG_COLUMNS if col != flag]
        if flag == FLAG_YIELD and poly_yield_flags is not None:
            exclusive = mask & ~df[other_flags].any(axis=1)
        else:
            exclusive = mask & ~df[other_flags].any(axis=1)
        if exclusive.any():
            mask = exclusive

    units = (
        df.loc[mask, KEY_LOC]
        .value_counts()
        .head(max_panels)
        .index.tolist()
    )
    if units:
        return units

    return df.loc[mask, KEY_LOC].drop_duplicates().head(max_panels).tolist()


def _draw_flag_panels(
    axes: np.ndarray,
    df: pd.DataFrame,
    flag: str,
    units: list[str],
    *,
    poly_yield_flags: pd.Series | None = None,
    polyfit_degree: int = 2,
) -> None:
    area_col = _area_column(df)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, loc in zip(axes_flat, units):
        group = cast(pd.DataFrame, df.loc[df[KEY_LOC] == loc])
        _plot_admin_series(
            ax,
            group,
            highlight_flags=(flag,),
            area_col=area_col if flag == FLAG_AREA else None,
            poly_yield_flags=poly_yield_flags,
            polyfit_degree=polyfit_degree,
        )
        if flag == FLAG_YIELD and poly_yield_flags is not None:
            n_flagged = int(poly_yield_flags.loc[group.index].astype(bool).sum())
            ax.set_title(f"{loc} — {n_flagged} polyfit outlier yr", fontsize=10)
        else:
            n_flagged = int(group[flag].astype(bool).sum())
            ax.set_title(f"{loc} ({n_flagged} flagged yr)", fontsize=10)

    for ax in axes_flat[len(units) :]:
        ax.axis("off")


def _residual_scatter_rows(
    df: pd.DataFrame,
    yield_outlier_flags: pd.Series,
    settings: YieldQualitySettings,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, group in df.groupby(KEY_LOC, sort=False):
        group = group.sort_values(HARVEST_YEAR)
        valid = cast(pd.DataFrame, group.loc[~_invalid_yield(group)])
        if len(valid) < 5:
            continue
        years = np.asarray(valid[HARVEST_YEAR], dtype=float)
        values = np.asarray(valid[KEY_TARGET], dtype=float)
        trend = _poly_trend(years, values, degree=settings.polyfit_degree)
        if trend is None:
            continue
        residuals = values - trend
        std = float(np.std(residuals))
        if std < 1e-6:
            continue
        z = residuals / std
        for year, yld, z_score, flagged in zip(
            years,
            values,
            z,
            yield_outlier_flags.loc[valid.index].astype(bool),
            strict=True,
        ):
            rows.append(
                {
                    KEY_LOC: group[KEY_LOC].iloc[0],
                    HARVEST_YEAR: year,
                    KEY_TARGET: yld,
                    "z_score": z_score,
                    FLAG_YIELD: flagged,
                }
            )
    return rows


def _draw_yield_residuals(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    crop: str,
    country_code: str,
    yield_outlier_flags: pd.Series,
    settings: YieldQualitySettings,
) -> None:
    """Scatter of polyfit residual z-scores on a single axes."""
    rows = _residual_scatter_rows(df, yield_outlier_flags, settings)
    total_yield_col = int(df[FLAG_YIELD].astype(bool).sum())
    invalid_flagged = int(_flagged_invalid_yield(df).sum())

    if not rows:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for residual plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    resid_df = pd.DataFrame(rows)
    flagged = resid_df[resid_df[FLAG_YIELD]]
    n_poly = len(flagged)

    if flagged.empty:
        ax.text(
            0.5,
            0.5,
            (
                f"No polyfit high-yield outliers (z > {settings.outlier_threshold:g}).\n"
                f"flag_yield_outlier has {total_yield_col} rows "
                f"({invalid_flagged} invalid ≤0 only)."
            ),
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    clean = resid_df[~resid_df[FLAG_YIELD]]
    ax.scatter(
        clean[HARVEST_YEAR],
        clean["z_score"],
        s=12,
        alpha=0.15,
        color="#2c3e50",
        label="Not flagged",
    )
    ax.scatter(
        flagged[HARVEST_YEAR],
        flagged["z_score"],
        s=80,
        color=FLAG_COLORS[FLAG_YIELD],
        edgecolors="black",
        linewidths=0.5,
        label=f"Polyfit high yield ({n_poly})",
        zorder=5,
    )
    ax.axhline(
        settings.outlier_threshold,
        color=FLAG_COLORS[FLAG_YIELD],
        linestyle="--",
        linewidth=1.0,
        label=f"z = {settings.outlier_threshold:g} threshold",
    )
    ax.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Harvest year")
    ax.set_ylabel("Yield residual z-score (poly trend)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)


def _report_header_text(
    crop: str,
    country_code: str,
    *,
    n_samples: int,
    n_poly: int,
    n_invalid: int,
    flag_list: list[str],
) -> tuple[str, str]:
    title = f"Yield quality — {crop} / {country_code}"
    parts = [f"{n_samples:,} samples"]
    if FLAG_YIELD in flag_list:
        parts.append(f"{n_poly} polyfit high-yield")
        parts.append(f"{n_invalid} invalid ≤0")
    return title, " · ".join(parts)


def build_yield_quality_report_figure(
    df: pd.DataFrame,
    crop: str,
    country_code: str,
    *,
    max_panels: int = 9,
    flags: list[str] | None = None,
    settings: YieldQualitySettings,
) -> plt.Figure:
    """Single combined figure: summary, panel grids, and optional residual plot."""
    flag_list = flags if flags is not None else list(FLAG_COLUMNS)
    yield_outlier_flags = _yield_outlier_flags(df)
    n_poly = int(yield_outlier_flags.sum())
    n_invalid = int(_flagged_invalid_yield(df).sum())

    panel_plans: list[tuple[str, list[str], int, int]] = []
    for flag in flag_list:
        if flag not in FLAG_COLUMNS:
            raise ValueError(f"Unknown flag {flag!r}; expected one of {FLAG_COLUMNS}")
        units = _select_admin_units(
            df,
            flag,
            max_panels=max_panels,
            prefer_exclusive=flag != FLAG_YIELD,
            poly_yield_flags=yield_outlier_flags if flag == FLAG_YIELD else None,
        )
        if units:
            n = len(units)
            ncols = min(3, n)
            nrows = math.ceil(n / ncols)
            panel_plans.append((flag, units, nrows, ncols))

    show_residuals = FLAG_YIELD in flag_list and n_poly > 0

    header_ratio = 0.55
    content_ratios: list[float] = [2.2]
    for _flag, _units, nrows, _ncols in panel_plans:
        content_ratios.append(max(nrows * 2.6, 2.4))
    if show_residuals:
        content_ratios.append(2.4)

    summary_only = len(content_ratios) == 1
    fig_height = 0.9 * sum(content_ratios) + (1.1 if summary_only else 1.8)
    fig_height = max(fig_height, 4.8 if summary_only else 7.0)

    height_ratios = [header_ratio, *content_ratios]
    fig = plt.figure(figsize=(12, fig_height))
    gs = fig.add_gridspec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=0.42,
    )

    title, subtitle = _report_header_text(
        crop,
        country_code,
        n_samples=len(df),
        n_poly=n_poly,
        n_invalid=n_invalid,
        flag_list=flag_list,
    )
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_ax.text(
        0.5,
        0.72,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        transform=header_ax.transAxes,
    )
    header_ax.text(
        0.5,
        0.22,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        transform=header_ax.transAxes,
    )

    summary_gs = gs[1].subgridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.28)
    ax_bar = fig.add_subplot(summary_gs[0, 0])
    ax_pie = fig.add_subplot(summary_gs[0, 1])
    _draw_summary_counts(
        ax_bar,
        ax_pie,
        df,
        crop=crop,
        country_code=country_code,
        flags=flag_list,
    )

    section_idx = 2
    for flag, units, nrows, ncols in panel_plans:
        panel_gs = gs[section_idx].subgridspec(
            nrows + 1,
            ncols,
            height_ratios=[0.14] + [1.0] * nrows,
            hspace=0.45,
            wspace=0.32,
        )
        title_ax = fig.add_subplot(panel_gs[0, :])
        title_ax.axis("off")
        if flag == FLAG_YIELD:
            section_title = (
                f"Example admin units — polyfit high-yield outliers "
                f"({n_poly} total, showing {len(units)})"
            )
            title_color = FLAG_COLORS[FLAG_YIELD]
        else:
            section_title = FLAG_LABELS[flag]
            title_color = FLAG_COLORS.get(flag, "#2c3e50")
        title_ax.text(
            0.0,
            0.5,
            section_title,
            fontsize=11,
            fontweight="bold",
            color=title_color,
            va="center",
            ha="left",
        )
        panel_axes = np.empty((nrows, ncols), dtype=object)
        for row in range(nrows):
            for col in range(ncols):
                panel_axes[row, col] = fig.add_subplot(panel_gs[row + 1, col])
        _draw_flag_panels(
            panel_axes,
            df,
            flag,
            units,
            poly_yield_flags=yield_outlier_flags if flag == FLAG_YIELD else None,
            polyfit_degree=settings.polyfit_degree,
        )
        section_idx += 1

    if show_residuals:
        resid_gs = gs[section_idx].subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.2)
        resid_title_ax = fig.add_subplot(resid_gs[0])
        resid_title_ax.axis("off")
        resid_title_ax.text(
            0.0,
            0.5,
            "Residual z-scores (polyfit outliers in red)",
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax_resid = fig.add_subplot(resid_gs[1])
        _draw_yield_residuals(
            ax_resid,
            df,
            crop=crop,
            country_code=country_code,
            yield_outlier_flags=yield_outlier_flags,
            settings=settings,
        )

    fig.subplots_adjust(top=0.97, bottom=0.08, left=0.08, right=0.96)
    return fig


def visualize_yield_quality(
    crop: str,
    country_code: str,
    *,
    data_dir: Path,
    output_dir: Path,
    max_panels: int = 9,
    flags: list[str] | None = None,
    settings: YieldQualitySettings | None = None,
) -> list[Path]:
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = settings or configured_yield_quality_settings()
    df = _load_annotated(crop, country_code, data_dir, settings=quality)
    flag_list = flags if flags is not None else list(FLAG_COLUMNS)

    report_fig = build_yield_quality_report_figure(
        df,
        crop,
        country_code,
        max_panels=max_panels,
        flags=flag_list,
        settings=quality,
    )
    report_path = output_dir / f"yield_quality_{crop}_{country_code}.png"
    report_fig.savefig(report_path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(report_fig)
    return [report_path]


def main(cfg: DictConfig) -> None:
    settings = yield_quality_settings_from_target(cfg)
    data_dir = Path(cfg.directory or PATH_DATA_DIR)
    output_dir = Path(cfg.output_dir)
    flags = viz_flag_columns(cfg)

    paths = visualize_yield_quality(
        str(cfg.crop_name),
        str(cfg.country_code),
        data_dir=data_dir,
        output_dir=output_dir,
        max_panels=int(cfg.max_panels),
        flags=flags,
        settings=settings,
    )
    print(f"Wrote {len(paths)} figure(s) to {output_dir.resolve()}:")
    for path in paths:
        print(f"  {path}")


@hydra.main(
    version_base=None,
    config_path="../cybench/conf/dataset",
    config_name="visualize_yield_quality",
)
def hydra_main(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_main()
