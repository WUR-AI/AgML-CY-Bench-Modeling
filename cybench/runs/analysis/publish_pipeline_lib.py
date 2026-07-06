"""Shared logic for the dashboard publish pipeline (collect → publish → git)."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from cybench.runs.analysis.benchmark_run_catalog import discover_benchmark_runs
from cybench.runs.analysis.collect_walk_forward_results import load_pooled_predictions
from cybench.runs.analysis.publish_dashboard_bundle import (
    _COUNTRY_NAMES,
    apply_pages_lite_to_publish_root,
    discover_index_entries,
    prune_obsolete_dashboard_dirs,
    publish_bundle,
    report_publish_bundle_size,
    update_index,
)
from cybench.runs.slurm.benchmark_submit_lib import (
    horizon_batch_suffix,
    resolve_batch_dir,
    resolve_case_insensitive_child,
)

Mode = Literal["planned", "ready", "all-available"]
StageName = Literal["collect", "publish", "index", "commit"]

MONOLITHIC_BASELINES_DIR = "baselines"

_BATCH_RE = re.compile(
    r"^baselines_(?P<country>[A-Za-z]{2})_(?P<batch_hz>eos|mid|qtr|early)_v(?P<version>\d+)$"
)
_PAPER_COLLECT_RE = re.compile(
    r"^paper_walk_forward_(?P<country>[a-z]{2})_(?P<batch_hz>eos|mid|qtr|early)_v(?P<version>\d+)$"
)

# Batch folder suffix → horizon tags accepted under ../output/<batch>/.
HORIZON_TAGS_BY_BATCH_SUFFIX: dict[str, tuple[str, ...]] = {
    "early": ("early_season", "early"),
    "eos": ("eos",),
    "mid": ("mid_season", "mid"),
    "qtr": ("quarter_season", "qtr"),
}

_BATCH_HORIZON_LABELS: dict[str, str] = {
    "early": "early season (25% observed)",
    "eos": "end-of-season",
    "mid": "mid-season",
    "qtr": "late season (75% observed, 25% left)",
}


@dataclass(frozen=True)
class PipelineDefaults:
    version: int = 3
    output_root: Path = Path("/lustre/backup/SHARED/AIN/agml/output")
    repo_root: Path = Path("/lustre/backup/SHARED/AIN/agml/AgML-CY-Bench-AAAI")
    publish_root: Path = Path("/lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard")
    min_run_fraction: float = 1.0
    pages_lite: bool = True


@dataclass
class PublishTarget:
    country: str
    batch_horizon: str  # eos | mid | qtr (batch folder suffix)
    version: int = 3
    output_root: Path = field(default_factory=lambda: PipelineDefaults().output_root)
    repo_root: Path = field(default_factory=lambda: PipelineDefaults().repo_root)
    publish_root: Path = field(default_factory=lambda: PipelineDefaults().publish_root)
    min_run_fraction: float = 1.0
    pages_lite: bool = True
    title: str | None = None

    @property
    def country_upper(self) -> str:
        return self.country.upper()

    @property
    def country_lower(self) -> str:
        return self.country.lower()

    @property
    def batch_name(self) -> str:
        return f"baselines_{self.country_upper}_{self.batch_horizon}_v{self.version}"

    @property
    def baselines_dir(self) -> Path:
        resolved, _ = resolve_batch_dir(self.output_root, self.batch_name)
        return resolved

    @property
    def collect_dir(self) -> Path:
        return self.output_root / f"paper_walk_forward_{self.country_lower}_{self.batch_horizon}_v{self.version}"

    @property
    def publish_slug(self) -> str:
        return f"{self.country_lower}_walk_forward_{self.batch_horizon}_v{self.version}"

    @property
    def publish_dir(self) -> Path:
        return self.publish_root / self.publish_slug

    @property
    def horizon_tags(self) -> tuple[str, ...]:
        return HORIZON_TAGS_BY_BATCH_SUFFIX[self.batch_horizon]

    def default_title(self) -> str:
        if self.title:
            return self.title
        country = _COUNTRY_NAMES.get(self.country_lower, self.country_upper)
        hz_label = _BATCH_HORIZON_LABELS.get(self.batch_horizon, self.batch_horizon)
        return f"{country} walk-forward, {hz_label} ({self.country_upper} maize/wheat)"


@dataclass
class ReadinessReport:
    target: PublishTarget
    expected_runs: int
    complete_runs: int
    ready: bool
    reason: str


@dataclass
class StageStatus:
    stage: StageName
    skipped: bool
    message: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_batch_dir_name(name: str) -> tuple[str, str, int] | None:
    match = _BATCH_RE.match(name)
    if not match:
        return None
    return match.group("country"), match.group("batch_hz"), int(match.group("version"))


def _complete_walk_forward_runs(
    baselines_dir: Path,
    *,
    country: str | None,
    horizon_tags: tuple[str, ...],
) -> list:
    if not baselines_dir.is_dir():
        return []
    cc = country.upper() if country else None
    seen: set[tuple[str, str, str]] = set()
    complete: list = []
    for horizon_tag in horizon_tags:
        for run in discover_benchmark_runs(
            baselines_dir,
            phase="walk_forward",
            horizon=horizon_tag,
            latest_only=True,
            allow_missing=True,
        ):
            if cc and run.country.upper() != cc:
                continue
            key = (run.crop, run.country, run.model)
            if key in seen:
                continue
            try:
                load_pooled_predictions(run.path, model_slug=run.model)
            except ValueError:
                continue
            seen.add(key)
            complete.append(run)
    complete.sort(key=lambda r: (r.crop, r.country, r.model))
    return complete


def resolve_collect_baselines_dir(target: PublishTarget) -> tuple[Path, str | None]:
    """Prefer per-country batch output; fall back to monolithic ``output/baselines/``."""
    per_country, alias = resolve_batch_dir(target.output_root, target.batch_name)
    if alias:
        note_prefix = alias
    else:
        note_prefix = None

    if _complete_walk_forward_runs(
        per_country,
        country=target.country_upper,
        horizon_tags=target.horizon_tags,
    ):
        return per_country, note_prefix

    if per_country.is_dir():
        # Batch folder exists (e.g. baselines_AO_mid_v1) — never redirect to monolithic.
        return per_country, note_prefix

    monolithic = target.output_root / MONOLITHIC_BASELINES_DIR
    if _complete_walk_forward_runs(
        monolithic,
        country=target.country_upper,
        horizon_tags=target.horizon_tags,
    ):
        note = (
            f"collecting from {monolithic} "
            f"(no complete runs under {per_country.name})"
        )
        return monolithic, note

    return per_country, note_prefix


def discover_baselines_batches(
    output_root: Path,
    *,
    defaults: PipelineDefaults | None = None,
) -> list[PublishTarget]:
    defaults = defaults or PipelineDefaults()
    targets: list[PublishTarget] = []
    seen: set[tuple[str, str, int]] = set()
    if not output_root.is_dir():
        return targets
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_batch_dir_name(entry.name)
        if parsed is None:
            continue
        country, batch_hz, version = parsed
        key = (country.upper(), batch_hz, version)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            PublishTarget(
                country=country.upper(),
                batch_horizon=batch_hz,
                version=version,
                output_root=output_root,
                repo_root=defaults.repo_root,
                publish_root=defaults.publish_root,
                min_run_fraction=defaults.min_run_fraction,
                pages_lite=defaults.pages_lite,
            )
        )

    mono = output_root / MONOLITHIC_BASELINES_DIR
    if mono.is_dir():
        for batch_hz, horizon_tags in HORIZON_TAGS_BY_BATCH_SUFFIX.items():
            by_country: set[str] = set()
            for run in _complete_walk_forward_runs(
                mono, country=None, horizon_tags=horizon_tags
            ):
                by_country.add(run.country.upper())
            for cc in sorted(by_country):
                key = (cc, batch_hz, defaults.version)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(
                    PublishTarget(
                        country=cc,
                        batch_horizon=batch_hz,
                        version=defaults.version,
                        output_root=output_root,
                        repo_root=defaults.repo_root,
                        publish_root=defaults.publish_root,
                        min_run_fraction=defaults.min_run_fraction,
                pages_lite=defaults.pages_lite,
                    )
                )
    return targets


def discover_baselines_batches_fast(
    output_root: Path,
    *,
    defaults: PipelineDefaults | None = None,
) -> list[PublishTarget]:
    """List ``baselines_{CC}_{eos|mid|qtr}_vN`` folders only (no monolithic scan)."""
    defaults = defaults or PipelineDefaults()
    targets: list[PublishTarget] = []
    seen: set[tuple[str, str, int]] = set()
    if not output_root.is_dir():
        return targets
    for entry in sorted(output_root.glob("baselines_*")):
        if not entry.is_dir():
            continue
        parsed = parse_batch_dir_name(entry.name)
        if parsed is None:
            continue
        country, batch_hz, version = parsed
        key = (country.upper(), batch_hz, version)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            PublishTarget(
                country=country.upper(),
                batch_horizon=batch_hz,
                version=version,
                output_root=output_root,
                repo_root=defaults.repo_root,
                publish_root=defaults.publish_root,
                min_run_fraction=defaults.min_run_fraction,
                pages_lite=defaults.pages_lite,
            )
        )
    return targets


def has_walk_forward_runs_fast(target: PublishTarget) -> bool:
    """True if the batch folder contains any walk-forward run dirs for this country."""
    baselines_dir, _ = resolve_batch_dir(target.output_root, target.batch_name)
    if not baselines_dir.is_dir():
        return False
    cc = target.country_upper
    for _ in baselines_dir.glob(f"*_{cc}_*_walk_forward_*"):
        return True
    return False


def discover_paper_walk_forward_targets(
    output_root: Path,
    *,
    defaults: PipelineDefaults | None = None,
    countries: list[str] | None = None,
    horizons: list[str] | None = None,
    version: int | None = None,
) -> list[PublishTarget]:
    """Discover targets from existing ``paper_walk_forward_*`` collect output (fast)."""
    defaults = defaults or PipelineDefaults()
    targets: list[PublishTarget] = []
    seen: set[tuple[str, str, int]] = set()
    if not output_root.is_dir():
        return targets
    for entry in sorted(output_root.glob("paper_walk_forward_*")):
        if not entry.is_dir():
            continue
        match = _PAPER_COLLECT_RE.match(entry.name)
        if match is None:
            continue
        country = match.group("country").upper()
        batch_hz = match.group("batch_hz")
        ver = int(match.group("version"))
        key = (country, batch_hz, ver)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            PublishTarget(
                country=country,
                batch_horizon=batch_hz,
                version=ver,
                output_root=output_root,
                repo_root=defaults.repo_root,
                publish_root=defaults.publish_root,
                min_run_fraction=defaults.min_run_fraction,
                pages_lite=defaults.pages_lite,
            )
        )
    return filter_publish_targets(
        targets, countries=countries, horizons=horizons, version=version
    )


def _summary_row_count(collect_dir: Path) -> int:
    summary = collect_dir / "walk_forward_summary.csv"
    if not summary.is_file():
        return 0
    try:
        text = summary.read_text(encoding="utf-8")
    except OSError:
        return 0
    lines = [line for line in text.splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def assess_publish_readiness(target: PublishTarget) -> ReadinessReport:
    """Fast readiness from collected paper output (no baselines scan)."""
    collect_dir = target.collect_dir
    html = collect_dir / "compare_models.html"
    n_rows = _summary_row_count(collect_dir)
    if html.is_file() and n_rows > 0:
        return ReadinessReport(
            target=target,
            expected_runs=0,
            complete_runs=n_rows,
            ready=True,
            reason=f"collected: {n_rows} summary rows in {collect_dir.name}",
        )
    if html.is_file():
        return ReadinessReport(
            target=target,
            expected_runs=0,
            complete_runs=0,
            ready=False,
            reason=f"empty summary in {collect_dir.name}",
        )
    return ReadinessReport(
        target=target,
        expected_runs=0,
        complete_runs=0,
        ready=False,
        reason=f"missing {html}",
    )


def load_pipeline_defaults(
    config_path: Path | None,
    *,
    overrides: PipelineDefaults | None = None,
) -> PipelineDefaults:
    """Merge optional YAML defaults with CLI overrides."""
    base = overrides or PipelineDefaults()
    if not config_path or not config_path.is_file():
        return base
    import yaml

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = raw.get("defaults") or {}
    return PipelineDefaults(
        version=int(cfg.get("version", base.version)),
        output_root=Path(cfg.get("output_root", base.output_root)),
        repo_root=Path(cfg.get("repo_root", base.repo_root)),
        publish_root=Path(cfg.get("publish_root", base.publish_root)).expanduser(),
        min_run_fraction=float(cfg.get("min_run_fraction", base.min_run_fraction)),
        pages_lite=bool(cfg.get("pages_lite", base.pages_lite)),
    )


def _explicit_targets_from_config(
    raw: dict,
    *,
    defaults: PipelineDefaults,
) -> list[PublishTarget]:
    """Build targets from ``include`` or legacy ``targets`` list (may pin not-yet-existing batches)."""
    entries = raw.get("include") or raw.get("targets") or []
    targets: list[PublishTarget] = []
    for item in entries:
        country = str(item["country"])
        horizons = item.get("horizons")
        if horizons is None:
            horizons = [item["horizon"]]
        for horizon in horizons:
            batch_hz = horizon_to_batch_suffix(str(horizon))
            targets.append(
                PublishTarget(
                    country=country,
                    batch_horizon=batch_hz,
                    version=int(item.get("version", defaults.version)),
                    output_root=defaults.output_root,
                    repo_root=defaults.repo_root,
                    publish_root=defaults.publish_root,
                    min_run_fraction=float(item.get("min_run_fraction", defaults.min_run_fraction)),
                    pages_lite=bool(item.get("pages_lite", defaults.pages_lite)),
                    title=item.get("title"),
                )
            )
    return targets


def load_targets_from_config(
    config_path: Path,
    *,
    defaults: PipelineDefaults | None = None,
    discover: bool = True,
) -> list[PublishTarget]:
    import yaml

    defaults = load_pipeline_defaults(config_path, overrides=defaults)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    explicit = _explicit_targets_from_config(raw, defaults=defaults)
    if explicit:
        return explicit
    if discover:
        return discover_baselines_batches(defaults.output_root, defaults=defaults)
    return []


def horizon_to_batch_suffix(horizon: str) -> str:
    """Map SLURM / CLI horizon to batch folder suffix (``eos`` / ``mid`` / ``qtr``)."""
    return horizon_batch_suffix(horizon)


def filter_publish_targets(
    targets: list[PublishTarget],
    *,
    countries: list[str] | None = None,
    horizons: list[str] | None = None,
    version: int | None = None,
    keep_latest_version: bool = False,
) -> list[PublishTarget]:
    """Narrow targets by country, horizon (eos|mid|qtr), and optional batch version.

    When ``keep_latest_version`` is true, only the highest ``vN`` per (country, horizon)
    is kept — use for GitHub Pages publish to avoid duplicate v1+v2 bundles.
    """
    if countries:
        wanted = {c.upper() for c in countries}
        targets = [t for t in targets if t.country_upper in wanted]
    if horizons:
        wanted_hz = {horizon_to_batch_suffix(h) for h in horizons}
        targets = [t for t in targets if t.batch_horizon in wanted_hz]
    if version is not None:
        targets = [t for t in targets if t.version == version]
    elif keep_latest_version:
        latest: dict[tuple[str, str], PublishTarget] = {}
        for target in targets:
            key = (target.country_upper, target.batch_horizon)
            prev = latest.get(key)
            if prev is None or target.version > prev.version:
                latest[key] = target
        targets = sorted(
            latest.values(),
            key=lambda t: (t.country_upper, t.batch_horizon, t.version),
        )
    return targets


def expected_job_count(target: PublishTarget) -> int:
    manifests_root = target.repo_root / "cybench/runs/slurm/manifests"
    manifest_root = resolve_case_insensitive_child(manifests_root, target.batch_name)
    if manifest_root is None:
        manifest_root = manifests_root / target.batch_name
    manifest = manifest_root / "benchmark_jobs.txt"
    if not manifest.is_file():
        shared = target.repo_root / "cybench/runs/slurm/benchmark_jobs.txt"
        if shared.is_file():
            from cybench.runs.slurm.benchmark_completion_lib import filter_jobs_by_country, read_manifest

            jobs = filter_jobs_by_country(read_manifest(shared), target.country_upper)
            return len(jobs)
        return 0
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


def count_complete_walk_forward_runs(target: PublishTarget) -> int:
    baselines_dir, _ = resolve_collect_baselines_dir(target)
    return len(
        _complete_walk_forward_runs(
            baselines_dir,
            country=target.country_upper,
            horizon_tags=target.horizon_tags,
        )
    )


def assess_readiness(target: PublishTarget) -> ReadinessReport:
    expected = expected_job_count(target)
    complete = count_complete_walk_forward_runs(target)
    baselines_dir, source_note = resolve_collect_baselines_dir(target)
    if complete <= 0 and not baselines_dir.is_dir():
        reason = f"missing baselines dir {target.baselines_dir}"
        if source_note:
            reason = f"{reason}; {source_note}"
        return ReadinessReport(
            target=target,
            expected_runs=expected,
            complete_runs=complete,
            ready=False,
            reason=reason,
        )
    if expected <= 0:
        return ReadinessReport(
            target=target,
            expected_runs=expected,
            complete_runs=complete,
            ready=complete > 0,
            reason="no manifest job count; using any complete runs",
        )
    threshold = max(1, int(expected * target.min_run_fraction))
    ready = complete >= threshold
    reason = (
        f"{complete}/{expected} walk-forward runs complete (need >={threshold})"
        if ready
        else f"only {complete}/{expected} walk-forward runs (need >={threshold})"
    )
    if source_note:
        reason = f"{reason}; {source_note}"
    return ReadinessReport(
        target=target,
        expected_runs=expected,
        complete_runs=complete,
        ready=ready,
        reason=reason,
    )


def baselines_fingerprint(target: PublishTarget) -> dict[str, Any]:
    baselines_dir, source_note = resolve_collect_baselines_dir(target)
    runs_meta: list[dict[str, str]] = []
    for run in _complete_walk_forward_runs(
        baselines_dir,
        country=target.country_upper,
        horizon_tags=target.horizon_tags,
    ):
        runs_meta.append(
            {
                "dataset": run.dataset,
                "model": run.model,
                "horizon": run.horizon,
                "timestamp": run.timestamp,
            }
        )
    payload: dict[str, Any] = {
        "baselines_dir": str(baselines_dir.resolve()),
        "n_runs": len(runs_meta),
        "runs": runs_meta,
    }
    if source_note:
        payload["source_note"] = source_note
    return payload


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_state_path(target: PublishTarget) -> Path:
    return target.collect_dir / ".publish_pipeline_state.json"


def publish_state_path(target: PublishTarget) -> Path:
    return target.publish_dir / ".publish_meta.json"


def needs_collect(
    target: PublishTarget,
    *,
    force: bool = False,
    fingerprint: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if force:
        return True, "forced"
    compare_html = target.collect_dir / "compare_models.html"
    if not compare_html.is_file():
        return True, "missing compare_models.html"
    state = _read_json(collect_state_path(target))
    fingerprint = fingerprint or baselines_fingerprint(target)
    if state is None:
        return True, "no pipeline state file"
    if state.get("n_runs") != fingerprint.get("n_runs"):
        return True, "run count changed"
    if state.get("runs") != fingerprint.get("runs"):
        return True, "baselines runs changed"
    return False, "collect output up to date"


def needs_publish(
    target: PublishTarget,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    if force:
        return True, "forced"
    dashboard = target.publish_dir / "dashboard.html"
    if not dashboard.is_file():
        return True, "missing published dashboard.html"
    meta = _read_json(publish_state_path(target))
    if meta is None:
        return True, "no publish metadata"
    source = str(target.collect_dir.resolve())
    if meta.get("source_dir") != source:
        return True, "collect source changed"
    collect_state = _read_json(collect_state_path(target))
    if collect_state and meta.get("collected_at") != collect_state.get("collected_at"):
        return True, "collect regenerated"
    return False, "publish bundle up to date"


def run_collect_subprocess(
    target: PublishTarget,
    *,
    plot: bool = True,
    dry_run: bool = False,
) -> None:
    script = target.repo_root / "cybench/runs/analysis/collect_walk_forward_results.py"
    baselines_dir, source_note = resolve_collect_baselines_dir(target)
    cmd = [
        "poetry",
        "run",
        "python",
        str(script),
        "--baselines-dir",
        str(baselines_dir),
        "--output-dir",
        str(target.collect_dir),
        "--country",
        target.country_upper,
        "--horizon",
        target.batch_horizon,
        "--dashboard",
    ]
    if plot:
        cmd.append("--plot")
    if dry_run:
        if source_note:
            print(f"[DRY-RUN] {source_note}")
        print(f"[DRY-RUN] collect: {' '.join(cmd)}")
        return
    if source_note:
        print(f"[INFO] {source_note}")
    proc = subprocess.run(cmd, cwd=target.repo_root, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"collect failed for {target.batch_name} (exit {proc.returncode})")


def run_collect_stage(
    target: PublishTarget,
    *,
    plot: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> StageStatus:
    fingerprint = baselines_fingerprint(target)
    if fingerprint["n_runs"] < 1:
        raise RuntimeError(f"No complete walk-forward runs in {target.baselines_dir}")
    should_run, reason = needs_collect(target, force=force, fingerprint=fingerprint)
    if not should_run:
        return StageStatus("collect", True, reason)
    if dry_run:
        run_collect_subprocess(target, plot=plot, dry_run=True)
        return StageStatus("collect", False, f"would run: {reason}")
    run_collect_subprocess(target, plot=plot, dry_run=False)
    collected_at = _utc_now_iso()
    state = {**fingerprint, "collected_at": collected_at}
    _write_json(collect_state_path(target), state)
    manifest = target.collect_dir / "manifest.json"
    if manifest.is_file():
        manifest_data = _read_json(manifest) or {}
        manifest_data["pipeline_collected_at"] = collected_at
        _write_json(manifest, manifest_data)
    return StageStatus("collect", False, f"collected {fingerprint['n_runs']} runs")


def run_publish_stage(
    target: PublishTarget,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> StageStatus:
    should_run, reason = needs_publish(target, force=force)
    if not should_run:
        return StageStatus("publish", True, reason)
    if dry_run:
        print(
            f"[DRY-RUN] publish {target.collect_dir} -> {target.publish_dir} "
            f"(slug={target.publish_slug})"
        )
        return StageStatus("publish", False, f"would run: {reason}")
    target.publish_root.mkdir(parents=True, exist_ok=True)
    publish_bundle(
        source_dir=target.collect_dir,
        dest_dir=target.publish_dir,
        title=target.default_title(),
        pages_lite=target.pages_lite,
    )
    collect_state = _read_json(collect_state_path(target)) or {}
    _write_json(
        publish_state_path(target),
        {
            "slug": target.publish_slug,
            "source_dir": str(target.collect_dir.resolve()),
            "published_at": _utc_now_iso(),
            "collected_at": collect_state.get("collected_at"),
            "n_runs": collect_state.get("n_runs"),
        },
    )
    return StageStatus("publish", False, f"published to {target.publish_dir}")


def run_index_stage(
    target: PublishTarget,
    *,
    dry_run: bool = False,
    insights_version: int | None = None,
) -> StageStatus:
    if dry_run:
        print(f"[DRY-RUN] rebuild index under {target.publish_root}")
        return StageStatus("index", False, "would rebuild index.html")
    prune_obsolete_dashboard_dirs(target.publish_root)
    if target.pages_lite:
        apply_pages_lite_to_publish_root(target.publish_root)
    entries = discover_index_entries(target.publish_root)
    index_path = update_index(target.publish_root, entries)
    version = insights_version if insights_version is not None else target.version
    extras: list[str] = []
    try:
        from cybench.runs.analysis.build_global_insights_dashboard import write_insights_dashboard

        insights_path = write_insights_dashboard(
            output_root=target.output_root,
            dest=target.publish_root / "insights.html",
            version=version,
        )
        extras.append(insights_path.name)
    except RuntimeError as exc:
        extras.append(f"insights skipped ({exc})")

    try:
        from cybench.runs.analysis.build_model_family_radar_dashboard import (
            write_model_family_radar_dashboard,
        )

        radar_path = write_model_family_radar_dashboard(
            output_root=target.output_root,
            dest=target.publish_root / "model_families.html",
            version=version,
        )
        extras.append(radar_path.name)
    except RuntimeError as exc:
        extras.append(f"model_families skipped ({exc})")

    msg = f"updated {index_path}" + (f" and {', '.join(extras)}" if extras else "")
    report_publish_bundle_size(target.publish_root)
    return StageStatus("index", False, msg)


def run_commit_stage(
    target: PublishTarget,
    *,
    dry_run: bool = False,
    push: bool = False,
) -> StageStatus:
    publish_root = target.publish_root
    if not (publish_root / ".git").is_dir():
        return StageStatus("commit", True, f"not a git repo: {publish_root}")

    rel_paths = [target.publish_slug, "index.html"]
    for global_page in ("insights.html", "model_families.html"):
        if (publish_root / global_page).is_file():
            rel_paths.append(global_page)

    message = f"Update {target.publish_slug} dashboard"
    return git_commit_paths(
        publish_root,
        rel_paths,
        message=message,
        push=push,
        dry_run=dry_run,
    )


def git_commit_paths(
    publish_root: Path,
    paths: list[str],
    *,
    message: str,
    push: bool = False,
    dry_run: bool = False,
) -> StageStatus:
    """Stage and commit changes under ``paths`` (plus deletions and new files)."""
    if not (publish_root / ".git").is_dir():
        return StageStatus("commit", True, f"not a git repo: {publish_root}")

    status = subprocess.run(
        ["git", "-C", str(publish_root), "status", "--porcelain", "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    if not status.stdout.strip():
        return StageStatus("commit", True, "no changes to commit")

    if dry_run:
        print(f"[DRY-RUN] git add + commit in {publish_root}: {message}")
        if push:
            print("[DRY-RUN] git push")
        return StageStatus("commit", False, f"would commit: {message}")

    subprocess.run(
        ["git", "-C", str(publish_root), "add", "-A", "--", *paths],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(publish_root), "commit", "-m", message],
        check=True,
    )
    if push:
        subprocess.run(["git", "-C", str(publish_root), "push"], check=True)
    return StageStatus("commit", False, message)


def git_commit_all(
    publish_root: Path,
    *,
    message: str,
    push: bool = False,
    dry_run: bool = False,
) -> StageStatus:
    """``git add -A``, commit if anything staged, optionally push."""
    if not (publish_root / ".git").is_dir():
        return StageStatus("commit", True, f"not a git repo: {publish_root}")

    if dry_run:
        print(f"[DRY-RUN] git add -A && git commit in {publish_root}: {message}")
        if push:
            print("[DRY-RUN] git push")
        return StageStatus("commit", False, f"would commit: {message}")

    subprocess.run(["git", "-C", str(publish_root), "add", "-A"], check=True)
    staged = subprocess.run(
        ["git", "-C", str(publish_root), "diff", "--cached", "--quiet"],
        check=False,
    )
    if staged.returncode == 0:
        dirty = subprocess.run(
            ["git", "-C", str(publish_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty.stdout.strip():
            print(
                "[WARN] Working tree has changes but nothing was staged — "
                "check .gitignore or file permissions",
                file=__import__("sys").stderr,
            )
        return StageStatus("commit", True, "no changes to commit")

    subprocess.run(
        ["git", "-C", str(publish_root), "commit", "-m", message],
        check=True,
    )
    if push:
        subprocess.run(["git", "-C", str(publish_root), "push"], check=True)
    return StageStatus("commit", False, message)


def resolve_targets(
    *,
    mode: Mode,
    config_path: Path | None,
    defaults: PipelineDefaults,
    countries: list[str] | None = None,
    horizons: list[str] | None = None,
    version: int | None = None,
) -> list[PublishTarget]:
    if config_path and config_path.is_file():
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        has_include = bool(raw.get("include") or raw.get("targets"))
        defaults = load_pipeline_defaults(config_path, overrides=defaults)
        if mode == "planned" and has_include:
            targets = _explicit_targets_from_config(raw, defaults=defaults)
        else:
            targets = load_targets_from_config(config_path, defaults=defaults, discover=True)
    else:
        targets = discover_baselines_batches(defaults.output_root, defaults=defaults)

    targets = filter_publish_targets(
        targets,
        countries=countries,
        horizons=horizons,
        version=version,
        keep_latest_version=version is None,
    )

    if mode == "all-available":
        return targets

    if mode == "planned" and not targets:
        raise ValueError(
            "--mode planned with an empty target set: add include: to the config "
            "or use --mode ready / all-available for lustre discovery"
        )

    if mode in {"planned", "ready"}:
        return targets

    raise ValueError(f"Unknown mode: {mode}")


def filter_ready_targets(targets: list[PublishTarget]) -> list[tuple[PublishTarget, ReadinessReport]]:
    out: list[tuple[PublishTarget, ReadinessReport]] = []
    for target in targets:
        report = assess_readiness(target)
        if report.ready:
            out.append((target, report))
    return out
