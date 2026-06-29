"""Discover Hydra baseline runs and load pooled test metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from cybench.config import KEY_TARGET
from cybench.evaluation.aggregated_metrics import compute_report_metrics
from cybench.util.prediction_horizon import parse_run_name_suffix

PHASES = ("walk_forward", "screening", "rolling")

# Flat metric keys produced by flatten_report_metrics / load_run_metrics.
METRIC_KEYS: tuple[str, ...] = (
    "r",
    "r2",
    "nrmse",
    "r2_spatial",
    "r_spatial_agg",
    "r2_spatial_agg",
    "r2_temporal",
    "r_temporal_agg",
    "r2_temporal_agg",
    "r2_anomaly",
    "r_res",
    "r2_res",
)

# For deltas: higher is better vs lower is better.
HIGHER_IS_BETTER = frozenset(
    {
        "r",
        "r2",
        "r2_spatial",
        "r_spatial_agg",
        "r2_spatial_agg",
        "r2_temporal",
        "r_temporal_agg",
        "r2_temporal_agg",
        "r2_anomaly",
        "r_res",
        "r2_res",
    }
)
LOWER_IS_BETTER = frozenset({"nrmse"})


def flatten_report_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    ry = metrics.get("region_year", {})
    sp = metrics.get("spatial", {})
    tm = metrics.get("temporal", {})
    an = metrics.get("anomaly", {})
    r2_spatial = sp.get("r2_typical_year")
    if r2_spatial is None:
        r2_spatial = ry.get("median_r2")
    r2_temporal = tm.get("r2_typical_region")
    r2_anomaly = an.get("r2_typical_region")
    if r2_anomaly is None:
        r2_anomaly = ry.get("r2_res")
    return {
        "n_regions": metrics.get("n_regions"),
        "n_years": metrics.get("n_years"),
        "n_samples": metrics.get("n_samples"),
        "r": ry.get("r"),
        "r2": ry.get("r2"),
        "nrmse": ry.get("nrmse"),
        "r_res": ry.get("r_res"),
        "r2_res": ry.get("r2_res"),
        "r2_spatial": r2_spatial,
        "r_spatial_agg": sp.get("r_aggregate", sp.get("r_climatology", sp.get("r"))),
        "r2_spatial_agg": sp.get("r2_aggregate", sp.get("r2_climatology", sp.get("r2"))),
        "r2_temporal": r2_temporal,
        "r_temporal_agg": tm.get("r_aggregate", tm.get("r")),
        "r2_temporal_agg": tm.get("r2_aggregate", tm.get("r2")),
        "r2_anomaly": r2_anomaly,
    }


@dataclass(frozen=True)
class BenchmarkRun:
    crop: str
    country: str
    model: str
    phase: str
    horizon: str
    timestamp: str
    path: Path

    @property
    def dataset(self) -> str:
        return f"{self.crop}_{self.country}"

    @property
    def match_key(self) -> tuple[str, str, str, str]:
        return (self.crop, self.country, self.model, self.horizon)

    @property
    def dataset_key(self) -> tuple[str, str, str]:
        """Match key for cross-group comparisons (horizon may differ per group)."""
        return (self.crop, self.country, self.model)


def parse_benchmark_run_dir(name: str, path: Path) -> BenchmarkRun | None:
    for phase in PHASES:
        token = f"_{phase}_"
        if token not in name:
            continue
        prefix, suffix = name.split(token, 1)
        parts = prefix.split("_", 2)
        if len(parts) < 3:
            return None
        crop, country, model = parts
        try:
            horizon, timestamp = parse_run_name_suffix(suffix)
        except ValueError:
            return None
        return BenchmarkRun(
            crop=crop,
            country=country,
            model=model,
            phase=phase,
            horizon=horizon,
            timestamp=timestamp,
            path=path,
        )
    return None


def discover_benchmark_runs(
    baselines_dir: Path,
    *,
    phase: str | None = None,
    horizon: str | None = None,
    latest_only: bool = True,
    allow_missing: bool = False,
) -> list[BenchmarkRun]:
    if not baselines_dir.is_dir():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Baselines directory not found: {baselines_dir}")

    by_key: dict[tuple[str, str, str, str, str], BenchmarkRun] = {}
    for entry in sorted(baselines_dir.iterdir()):
        if not entry.is_dir():
            continue
        run = parse_benchmark_run_dir(entry.name, entry)
        if run is None:
            continue
        if phase is not None and run.phase != phase:
            continue
        if horizon is not None and run.horizon != horizon:
            continue
        key = (run.crop, run.country, run.model, run.phase, run.horizon)
        if key not in by_key or run.timestamp > by_key[key].timestamp:
            by_key[key] = run

    if not latest_only:
        runs: list[BenchmarkRun] = []
        for entry in sorted(baselines_dir.iterdir()):
            if not entry.is_dir():
                continue
            run = parse_benchmark_run_dir(entry.name, entry)
            if run is None:
                continue
            if phase is not None and run.phase != phase:
                continue
            if horizon is not None and run.horizon != horizon:
                continue
            runs.append(run)
        return sorted(
            runs,
            key=lambda r: (r.crop, r.country, r.model, r.phase, r.horizon, r.timestamp),
        )

    return sorted(
        by_key.values(),
        key=lambda r: (r.crop, r.country, r.model, r.phase, r.horizon),
    )


def _screening_metrics_path(run_dir: Path, seed: int = 42) -> Path | None:
    """Pooled test split: multi-year folder (e.g. 2016_2017_...), not a single year."""
    for path in sorted(run_dir.glob(f"*/{seed}/report_metrics.yaml")):
        if re.fullmatch(r"\d{4}", path.parent.parent.name):
            continue
        return path
    return None


def _metrics_from_report_yaml(path: Path) -> dict[str, Any]:
    raw = OmegaConf.to_container(OmegaConf.load(path))
    if not isinstance(raw, dict):
        return {}
    return flatten_report_metrics(
        {
            "n_regions": raw.get("n_regions"),
            "n_years": raw.get("n_years"),
            "n_samples": raw.get("n_samples"),
            "region_year": raw.get("region_year") or {},
            "spatial": raw.get("spatial") or {},
            "temporal": raw.get("temporal") or {},
            "anomaly": raw.get("anomaly") or {},
        }
    )


def load_run_metrics(run: BenchmarkRun, *, seed: int = 42) -> dict[str, Any] | None:
    """Pooled test metrics for a single Hydra run directory."""
    if run.phase == "walk_forward":
        from cybench.runs.analysis.collect_walk_forward_results import (
            load_walk_forward_summary_metrics,
        )

        summary = load_walk_forward_summary_metrics(run)
        if summary is None:
            return None
        return {
            k: summary.get(k)
            for k in ("n_regions", "n_years", "n_samples") + METRIC_KEYS
        }

    metrics_path = _screening_metrics_path(run.path, seed=seed)
    if metrics_path is None:
        return None
    flat = _metrics_from_report_yaml(metrics_path)
    return flat or None


def parse_group_spec(spec: str) -> tuple[str, str | None, str | None]:
    """
    Parse ``label=phase`` or ``label=phase/horizon``.

    Examples: ``wf=walk_forward/eos``, ``screen=screening``, ``a=screening/mid_season``
    """
    if "=" not in spec:
        raise ValueError(f"Invalid group spec {spec!r}. Use LABEL=PHASE or LABEL=PHASE/HORIZON")
    label, rhs = spec.split("=", 1)
    label = label.strip()
    rhs = rhs.strip()
    if not label or not rhs:
        raise ValueError(f"Invalid group spec {spec!r}")
    if "/" in rhs:
        phase, horizon = rhs.split("/", 1)
        return label, phase.strip(), horizon.strip() or None
    return label, rhs, None
