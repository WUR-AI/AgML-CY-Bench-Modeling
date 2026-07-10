"""Country-level bootstrap inference for best-AI vs best-traditional NRMSE gains.

Protocol (paper Results §5.2):
- Average walk-forward metrics over five random seeds first.
- Country is the unit of inference (paired comparisons within crop).
- For each country: best traditional (Average, Trend, LPJmL) vs best AI
  (feature-engineered, sequence, foundation).
- Bootstrap countries with replacement (default 10,000); keep pairings intact.
- Percentile 95% CIs for median absolute improvement, median % improvement,
  and AI win rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cybench.runs.analysis.model_family_radar_lib import (
    _ai_benefit_map_slice,
    ai_error_reduction_pct,
)
from cybench.runs.analysis.global_insights_lib import horizons_in_data

DEFAULT_N_BOOTSTRAP = 10_000


def _seed_averaged_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per crop×country×model; mean metrics across seed repetitions when present."""
    if df.empty:
        return df.copy()
    work = df.copy()
    metric_cols = [
        c
        for c in (
            "nrmse",
            "r2",
            "r_spatial",
            "r_temporal",
            "r_res",
            "r_spatial_agg",
            "r_temporal_agg",
            "r2_res",
            "r2_anomaly",
        )
        if c in work.columns
    ]
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    key_cols = [c for c in ("crop", "country", "model", "batch_horizon") if c in work.columns]
    if "seed" in work.columns or "repetition" in work.columns:
        seed_col = "seed" if "seed" in work.columns else "repetition"
        grouped = work.groupby(key_cols, as_index=False)[metric_cols].mean()
        meta_cols = [c for c in work.columns if c not in {*key_cols, *metric_cols, seed_col}]
        if meta_cols:
            first = work.groupby(key_cols, as_index=False)[meta_cols].first()
            grouped = grouped.merge(first, on=key_cols, how="left")
        return grouped
    return work.drop_duplicates(subset=key_cols)


def country_ai_benefit_frame(
    df: pd.DataFrame,
    *,
    batch_horizon: str,
    crop: str,
    representatives: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-country paired best-traditional vs best-AI NRMSE (matches Figure ai_gain_maps)."""
    work = _seed_averaged_country_metrics(df)
    payload = _ai_benefit_map_slice(
        work,
        batch_horizon=batch_horizon,
        crop=crop,
        representatives=representatives,
    )
    rows = payload.get("countries") or []
    if not rows:
        return pd.DataFrame(
            columns=[
                "country",
                "nrmse_trad",
                "nrmse_ai",
                "delta_abs",
                "delta_pct",
                "ai_wins",
                "traditional_model",
                "ai_model",
            ]
        )
    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "nrmse_traditional": "nrmse_trad",
            "nrmse_ai": "nrmse_ai",
        }
    )
    frame["delta_abs"] = frame["nrmse_trad"] - frame["nrmse_ai"]
    if "benefit_pct" in frame.columns:
        frame["delta_pct"] = frame["benefit_pct"]
    else:
        frame["delta_pct"] = frame.apply(
            lambda r: ai_error_reduction_pct(r["nrmse_ai"], r["nrmse_trad"]),
            axis=1,
        )
    frame["ai_wins"] = frame["delta_abs"] > 0
    return frame.sort_values("country").reset_index(drop=True)


def _percentile_ci(samples: np.ndarray, *, ci: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def bootstrap_country_ai_metrics(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float | None]:
    """Bootstrap countries; report CIs for median Δ_abs, median Δ%, and win rate."""
    if frame.empty:
        return {
            "median_nrmse_trad": None,
            "median_nrmse_ai": None,
            "median_delta_abs": None,
            "delta_abs_ci_lo": None,
            "delta_abs_ci_hi": None,
            "median_delta_pct": None,
            "delta_pct_ci_lo": None,
            "delta_pct_ci_hi": None,
            "win_rate": None,
            "win_rate_ci_lo": None,
            "win_rate_ci_hi": None,
        }

    delta_abs = frame["delta_abs"].to_numpy(dtype=float)
    delta_pct = frame["delta_pct"].to_numpy(dtype=float)
    ai_wins = frame["ai_wins"].to_numpy(dtype=bool)
    n = delta_abs.size
    nrmse_trad = (
        frame["nrmse_trad"].to_numpy(dtype=float)
        if "nrmse_trad" in frame.columns
        else np.full(n, np.nan)
    )
    nrmse_ai = (
        frame["nrmse_ai"].to_numpy(dtype=float)
        if "nrmse_ai" in frame.columns
        else np.full(n, np.nan)
    )
    med_trad = float(np.median(nrmse_trad)) if np.isfinite(nrmse_trad).any() else None
    med_ai = float(np.median(nrmse_ai)) if np.isfinite(nrmse_ai).any() else None

    if n == 1:
        return {
            "median_nrmse_trad": med_trad,
            "median_nrmse_ai": med_ai,
            "median_delta_abs": float(delta_abs[0]),
            "delta_abs_ci_lo": float(delta_abs[0]),
            "delta_abs_ci_hi": float(delta_abs[0]),
            "median_delta_pct": float(delta_pct[0]),
            "delta_pct_ci_lo": float(delta_pct[0]),
            "delta_pct_ci_hi": float(delta_pct[0]),
            "win_rate": float(ai_wins[0]),
            "win_rate_ci_lo": float(ai_wins[0]),
            "win_rate_ci_hi": float(ai_wins[0]),
        }

    rng = np.random.default_rng(seed)
    boot_med_abs = np.empty(n_bootstrap, dtype=float)
    boot_med_pct = np.empty(n_bootstrap, dtype=float)
    boot_wr = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_med_abs[i] = float(np.median(delta_abs[idx]))
        boot_med_pct[i] = float(np.median(delta_pct[idx]))
        boot_wr[i] = float(np.mean(ai_wins[idx]))

    abs_lo, abs_hi = _percentile_ci(boot_med_abs, ci=ci)
    pct_lo, pct_hi = _percentile_ci(boot_med_pct, ci=ci)
    wr_lo, wr_hi = _percentile_ci(boot_wr, ci=ci)

    return {
        "median_nrmse_trad": med_trad,
        "median_nrmse_ai": med_ai,
        "median_delta_abs": float(np.median(delta_abs)),
        "delta_abs_ci_lo": abs_lo,
        "delta_abs_ci_hi": abs_hi,
        "median_delta_pct": float(np.median(delta_pct)),
        "delta_pct_ci_lo": pct_lo,
        "delta_pct_ci_hi": pct_hi,
        "win_rate": float(np.mean(ai_wins)),
        "win_rate_ci_lo": wr_lo,
        "win_rate_ci_hi": wr_hi,
    }


def wilcoxon_two_sided_pvalue(deltas: np.ndarray | pd.Series) -> float | None:
    """Optional two-sided Wilcoxon signed-rank test on paired NRMSE differences."""
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    if np.allclose(arr, 0.0):
        return 1.0
    try:
        result = stats.wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue)
    except ValueError:
        return None


def analyze_country_ai_benefit(
    df: pd.DataFrame,
    *,
    batch_horizon: str = "eos",
    crop: str,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    representatives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize country-level best-AI vs best-traditional comparison for one crop."""
    frame = country_ai_benefit_frame(
        df,
        batch_horizon=batch_horizon,
        crop=crop,
        representatives=representatives,
    )
    n_countries = int(len(frame))
    boot = bootstrap_country_ai_metrics(
        frame,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    if n_countries == 0:
        return {
            "crop": crop,
            "batch_horizon": batch_horizon,
            "n_countries": 0,
            "n_ai_wins": 0,
            "wilcoxon_pvalue": None,
            "countries": frame,
            **boot,
        }

    return {
        "crop": crop,
        "batch_horizon": batch_horizon,
        "n_countries": n_countries,
        "n_ai_wins": int(frame["ai_wins"].sum()),
        "wilcoxon_pvalue": wilcoxon_two_sided_pvalue(frame["delta_abs"]),
        "countries": frame,
        **boot,
    }


def analyze_all_crops(
    df: pd.DataFrame,
    *,
    batch_horizon: str = "eos",
    crops: tuple[str, ...] = ("maize", "wheat"),
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    representatives: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        crop: analyze_country_ai_benefit(
            df,
            batch_horizon=batch_horizon,
            crop=crop,
            n_bootstrap=n_bootstrap,
            seed=seed,
            representatives=representatives,
        )
        for crop in crops
    }


def bootstrap_stats_json(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable bootstrap summary (omit per-country frame)."""
    return {k: v for k, v in result.items() if k != "countries"}


def _resolve_metric_column(work: pd.DataFrame, metric: str) -> str | None:
    """Map dashboard metric id to a column present in *work*."""
    if metric in work.columns:
        return metric
    fallbacks: dict[str, tuple[str, ...]] = {
        "r_spatial": ("r_spatial_agg",),
        "r_temporal": ("r_temporal_agg",),
        "r_res": ("r2_res", "r_anomaly"),
    }
    for alt in fallbacks.get(metric, ()):
        if alt in work.columns:
            return alt
    return None


def prepare_work_for_family_vs_naive(work: pd.DataFrame) -> pd.DataFrame:
    """Seed-average and keep rows even when only non-NRMSE metrics are present."""
    return _seed_averaged_country_metrics(work)


def _country_metric_median(
    model_grp: pd.DataFrame,
    country: str,
    metric: str,
) -> float | None:
    from cybench.runs.analysis.global_insights_lib import _median_in_group

    if model_grp.empty or "country" not in model_grp.columns or metric not in model_grp.columns:
        return None
    sub = model_grp[model_grp["country"].astype(str) == str(country)]
    return _median_in_group(sub, metric)


def family_vs_naive_country_deltas(
    work: pd.DataFrame,
    *,
    family_model: str,
    naive_model: str,
    metric: str,
    higher_is_better: bool,
) -> np.ndarray:
    """Per-country paired improvement; positive => family better than naive.

    Uses the global family and naive representatives only (same models as the table).
    """
    from cybench.runs.analysis.model_family_radar_lib import _country_median_metric

    column = _resolve_metric_column(work, metric)
    if (
        work.empty
        or "country" not in work.columns
        or column is None
        or not naive_model
    ):
        return np.array([], dtype=float)

    deltas: list[float] = []
    for _, cc_grp in work.groupby("country", sort=True):
        fam_val = _country_median_metric(cc_grp, family_model, column)
        naive_val = _country_median_metric(cc_grp, naive_model, column)
        if fam_val is None or naive_val is None:
            continue
        deltas.append((fam_val - naive_val) if higher_is_better else (naive_val - fam_val))
    return np.asarray(deltas, dtype=float)


def bootstrap_family_vs_naive_stats(
    deltas: np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | bool | int | None]:
    """Bootstrap country medians; one-sided p = share of bootstrap draws with median <= 0."""
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {
            "n_countries": 0,
            "median_delta": None,
            "ci_lo": None,
            "ci_hi": None,
            "p_one_sided": None,
            "significant": False,
        }
    median_obs = float(np.median(arr))
    if n < 2:
        return {
            "n_countries": int(n),
            "median_delta": round(median_obs, 4),
            "ci_lo": None,
            "ci_hi": None,
            "p_one_sided": None,
            "significant": False,
        }
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boots[i] = float(np.median(arr[rng.integers(0, n, size=n)]))
    ci_lo, ci_hi = _percentile_ci(boots, ci=0.95)
    # One-sided bootstrap p-value (family better than naive).
    p_one_sided = float((np.sum(boots <= 0.0) + 1) / (n_bootstrap + 1))
    significant = float(np.quantile(boots, alpha)) > 0.0
    return {
        "n_countries": int(n),
        "median_delta": round(median_obs, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "p_one_sided": round(p_one_sided, 4),
        "significant": significant,
    }


def bootstrap_one_sided_significant(
    deltas: np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> bool:
    """True when one-sided 95% rule holds: 5th percentile of bootstrap medians > 0."""
    return bool(
        bootstrap_family_vs_naive_stats(
            deltas,
            n_bootstrap=n_bootstrap,
            seed=seed,
            alpha=alpha,
        )["significant"]
    )


def build_family_vs_naive_significance(
    work: pd.DataFrame,
    representatives: dict[str, str],
    *,
    metrics: tuple[str, ...] | list[str] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float | bool | int | None]]]:
    """One-sided country bootstrap: family representative vs naive representative."""
    from cybench.runs.analysis.model_family_radar_lib import (
        FAMILY_ORDER,
        VIEW_METRICS,
        _metric_higher_is_better,
    )

    metric_list = tuple(metrics or VIEW_METRICS)
    naive_model = representatives.get("Naive baselines")
    if not naive_model or work.empty:
        return {}
    out: dict[str, dict[str, dict[str, float | bool | int | None]]] = {}
    for fi, family in enumerate(FAMILY_ORDER):
        if family == "Naive baselines" or family not in representatives:
            continue
        family_model = representatives[family]
        stats_by_metric: dict[str, dict[str, float | bool | int | None]] = {}
        for mi, metric in enumerate(metric_list):
            column = _resolve_metric_column(work, metric)
            if column is None:
                stats_by_metric[metric] = {
                    "n_countries": 0,
                    "median_delta": None,
                    "ci_lo": None,
                    "ci_hi": None,
                    "p_one_sided": None,
                    "significant": False,
                }
                continue
            deltas = family_vs_naive_country_deltas(
                work,
                family_model=family_model,
                naive_model=naive_model,
                metric=metric,
                higher_is_better=_metric_higher_is_better(metric),
            )
            stats_by_metric[metric] = bootstrap_family_vs_naive_stats(
                deltas,
                n_bootstrap=n_bootstrap,
                seed=seed + fi * 100 + mi,
            )
        out[family] = stats_by_metric
    return out


FAMILY_VS_NAIVE_SIG_NOTE = (
    "* One-sided country bootstrap vs naive family representative (B=10,000). "
    "Hover: median per-country Δ (bootstrap target) and table-median gap "
    "(difference of the two table cells). Bold = best family for that metric."
)


def empty_family_vs_naive_stats() -> dict[str, dict[str, float | bool | int | None]]:
    from cybench.runs.analysis.model_family_radar_lib import VIEW_METRICS

    empty = {
        "n_countries": 0,
        "median_delta": None,
        "ci_lo": None,
        "ci_hi": None,
        "p_one_sided": None,
        "significant": False,
    }
    return {m: dict(empty) for m in VIEW_METRICS}


COUNTRY_BOOTSTRAP_NOTE = (
    "Per country: best traditional NRMSE = min(Average, Trend, LPJmL); best data-driven NRMSE = "
    "min(feature-engineered, sequence, foundation). NRMSE columns are medians of those "
    "country-level values. Median improvement (%) is the median of "
    "100×(traditional−data-driven)/traditional per country. Win rate: fraction of countries "
    "where data-driven NRMSE is lower. Seed-averaged walk-forward NRMSE; bootstrap resamples "
    "countries (95% percentile CIs on improvement and win rate)."
)


def build_country_bootstrap_payload(
    df: pd.DataFrame,
    *,
    crops: tuple[str, ...] | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    representatives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Dashboard payload: bootstrap stats indexed by horizon and crop."""
    if df.empty:
        return {
            "by_horizon": {},
            "n_bootstrap": n_bootstrap,
            "note": COUNTRY_BOOTSTRAP_NOTE,
        }
    crop_list = crops
    if crop_list is None:
        crop_list = tuple(sorted({str(c) for c in df["crop"].dropna().unique()}))
    present_crops = set(df["crop"].astype(str)) if "crop" in df.columns else set()
    by_horizon: dict[str, dict[str, Any]] = {}
    for hz in horizons_in_data(df):
        crop_stats: dict[str, Any] = {}
        for crop in crop_list:
            if crop not in present_crops:
                continue
            res = analyze_country_ai_benefit(
                df,
                batch_horizon=hz,
                crop=crop,
                n_bootstrap=n_bootstrap,
                seed=seed,
                representatives=representatives,
            )
            crop_stats[crop] = bootstrap_stats_json(res)
        if crop_stats:
            by_horizon[hz] = crop_stats
    return {
        "by_horizon": by_horizon,
        "n_bootstrap": n_bootstrap,
        "note": COUNTRY_BOOTSTRAP_NOTE,
    }


def _fmt_num(value: float | None, *, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "---"
    return f"{value:.{digits}f}"


def _fmt_pct_rate(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "---"
    return f"{100.0 * value:.0f}\\%"


def _fmt_ci(lo: float | None, hi: float | None, *, digits: int = 1, percent: bool = False) -> str:
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "[---, ---]"
    if percent:
        return f"[{100.0 * lo:.0f}, {100.0 * hi:.0f}]"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_nrmse_pct(value: float | None) -> str:
    """Format pooled NRMSE fraction as percent (matches Table 1)."""
    if value is None or not np.isfinite(value):
        return "---"
    return f"{100.0 * float(value):.1f}"


def format_results_markdown_table(results: dict[str, dict[str, Any]]) -> str:
    """Paper-style summary table (markdown)."""
    lines = [
        "| Crop | Trad. NRMSE (%) | AI NRMSE (%) | Median improvement (%) | 95% CI | AI win rate | 95% CI |",
        "|------|-----------------|--------------|------------------------|--------|-------------|--------|",
    ]
    for crop, res in results.items():
        med = res.get("median_delta_pct")
        lo, hi = res.get("delta_pct_ci_lo"), res.get("delta_pct_ci_hi")
        wr = res.get("win_rate")
        wr_lo, wr_hi = res.get("win_rate_ci_lo"), res.get("win_rate_ci_hi")
        lines.append(
            "| "
            + " | ".join(
                [
                    crop.capitalize(),
                    _fmt_nrmse_pct(res.get("median_nrmse_trad")),
                    _fmt_nrmse_pct(res.get("median_nrmse_ai")),
                    _fmt_num(med),
                    _fmt_ci(lo, hi).strip("[]") if lo is not None else "---",
                    f"{100 * wr:.0f}%" if wr is not None else "---",
                    _fmt_ci(wr_lo, wr_hi, percent=True).strip("[]")
                    if wr_lo is not None
                    else "---",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_results_latex_table(
    results: dict[str, dict[str, Any]],
    *,
    caption: str | None = None,
    label: str = "tab:ai_country_bootstrap",
) -> str:
    """LaTeX table for main paper or supplement."""
    if caption is None:
        caption = (
            "Country-level bootstrap summary of AI advantage over traditional baselines. "
            "Traditional and data-driven NRMSE are medians across countries of the best "
            "traditional (Average, Trend, LPJmL) and best data-driven model per country. "
            "Median improvement (\\%) is the median of "
            "$100\\times(\\mathrm{NRMSE}_{\\mathrm{trad}}-\\mathrm{NRMSE}_{\\mathrm{AI}})"
            "/\\mathrm{NRMSE}_{\\mathrm{trad}}$ per country. Win rate: fraction of countries "
            "where data-driven NRMSE is lower. Bootstrap resamples countries ($B=10{,}000$); "
            "intervals are percentile 95\\% CIs."
        )
    body_rows: list[str] = []
    for crop, res in results.items():
        trad = _fmt_nrmse_pct(res.get("median_nrmse_trad"))
        ai = _fmt_nrmse_pct(res.get("median_nrmse_ai"))
        med = _fmt_num(res.get("median_delta_pct"))
        pct_ci = _fmt_ci(res.get("delta_pct_ci_lo"), res.get("delta_pct_ci_hi"))
        wr = _fmt_pct_rate(res.get("win_rate"))
        wr_ci = _fmt_ci(res.get("win_rate_ci_lo"), res.get("win_rate_ci_hi"), percent=True)
        body_rows.append(
            f"{crop.capitalize()} & {trad} & {ai} & {med} & {pct_ci} & {wr} & {wr_ci} \\\\"
        )
    body = "\n".join(body_rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Crop & Trad.\\ NRMSE (\\%) & AI NRMSE (\\%) & Median impr. (\\%) & 95\\% CI & AI win rate & 95\\% CI \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def format_results_latex_sentence(result: dict[str, Any]) -> str:
    """One LaTeX-ready prose sentence for a single crop (bootstrap-focused)."""
    crop = str(result["crop"]).capitalize()
    n = result["n_countries"]
    wins = result["n_ai_wins"]
    if n == 0:
        return f"No country-level comparisons available for {crop.lower()}."
    med = _fmt_num(result.get("median_delta_pct"))
    pct_ci = _fmt_ci(result.get("delta_pct_ci_lo"), result.get("delta_pct_ci_hi"))
    wr = _fmt_pct_rate(result.get("win_rate"))
    wr_ci = _fmt_ci(result.get("win_rate_ci_lo"), result.get("win_rate_ci_hi"), percent=True)
    return (
        f"\\textbf{{{crop}}}: the best data-driven model achieved lower NRMSE than the "
        f"best traditional baseline in {wins}/{n} countries (median improvement {med}\\%; "
        f"95\\% bootstrap CI {pct_ci}; win rate {wr}, 95\\% CI {wr_ci})."
    )
