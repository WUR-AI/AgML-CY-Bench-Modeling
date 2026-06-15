"""Discover Hydra baseline runs and load pooled test metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from cybench.config import KEY_TARGET
from cybench.evaluation.aggregated_metrics import compute_report_metrics
from cybench.runs.collect_walk_forward_results import (
    flatten_report_metrics,
    load_pooled_predictions,
)
from cybench.util.prediction_horizon import parse_run_name_suffix

PHASES = ("walk_forward", "screening", "rolling")

# Flat metric keys produced by flatten_report_metrics / load_run_metrics.
METRIC_KEYS: tuple[str, ...] = (
    "r",
    "r2",
    "nrmse",
    "r_spatial",
    "r2_spatial",
    "r_temporal",
    "r2_temporal",
    "r_res",
    "r2_res",
)

# For deltas: higher is better vs lower is better.
HIGHER_IS_BETTER = frozenset({"r", "r2", "r_spatial", "r2_spatial", "r_temporal", "r2_temporal", "r_res", "r2_res"})
LOWER_IS_BETTER = frozenset({"nrmse"})


@dataclass(frozen=True)
class BenchmarkRun:
    crop: str
    country: str
    model: str
    phase: str
    horizon: str | None
    timestamp: str
    path: Path

    @property
    def dataset(self) -> str:
        return f"{self.crop}_{self.country}"

    @property
    def match_key(self) -> tuple[str, str, str, str | None]:
        return (self.crop, self.country, self.model, self.horizon)


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
        horizon, timestamp = parse_run_name_suffix(suffix)
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
) -> list[BenchmarkRun]:
    if not baselines_dir.is_dir():
        raise FileNotFoundError(f"Baselines directory not found: {baselines_dir}")

    by_key: dict[tuple[str, str, str, str, str | None], BenchmarkRun] = {}
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
            key=lambda r: (r.crop, r.country, r.model, r.phase, r.horizon or "", r.timestamp),
        )

    return sorted(
        by_key.values(),
        key=lambda r: (r.crop, r.country, r.model, r.phase, r.horizon or ""),
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
        }
    )


def load_run_metrics(run: BenchmarkRun, *, seed: int = 42) -> dict[str, Any] | None:
    """Pooled test metrics for a single Hydra run directory."""
    if run.phase == "walk_forward":
        try:
            df, model_col = load_pooled_predictions(run.path, model_slug=run.model)
        except ValueError:
            return None
        return flatten_report_metrics(
            compute_report_metrics(df, target_col=KEY_TARGET, model_col=model_col)
        )

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
