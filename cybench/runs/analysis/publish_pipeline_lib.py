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
    discover_index_entries,
    publish_bundle,
    update_index,
)

Mode = Literal["planned", "ready", "all-available"]
StageName = Literal["collect", "publish", "index", "commit"]

_BATCH_RE = re.compile(
    r"^baselines_(?P<country>[A-Za-z]{2})_(?P<batch_hz>eos|mid)_v(?P<version>\d+)$"
)

# Batch folder suffix → horizon tags accepted under ../output/<batch>/.
HORIZON_TAGS_BY_BATCH_SUFFIX: dict[str, tuple[str, ...]] = {
    "eos": ("eos",),
    "mid": ("mid_season", "mid"),
}


@dataclass(frozen=True)
class PipelineDefaults:
    version: int = 1
    output_root: Path = Path("/lustre/backup/SHARED/AIN/agml/output")
    repo_root: Path = Path("/lustre/backup/SHARED/AIN/agml/AgML-CY-Bench-AAAI")
    publish_root: Path = Path("/lustre/backup/SHARED/AIN/agml/CY-Bench-dashboard")
    min_run_fraction: float = 1.0


@dataclass
class PublishTarget:
    country: str
    batch_horizon: str  # eos | mid (batch folder suffix)
    version: int = 1
    output_root: Path = field(default_factory=lambda: PipelineDefaults().output_root)
    repo_root: Path = field(default_factory=lambda: PipelineDefaults().repo_root)
    publish_root: Path = field(default_factory=lambda: PipelineDefaults().publish_root)
    min_run_fraction: float = 1.0
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
        return self.output_root / self.batch_name

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
        hz_label = "end-of-season" if self.batch_horizon == "eos" else "mid-season"
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


def discover_baselines_batches(
    output_root: Path,
    *,
    defaults: PipelineDefaults | None = None,
) -> list[PublishTarget]:
    defaults = defaults or PipelineDefaults()
    targets: list[PublishTarget] = []
    if not output_root.is_dir():
        return targets
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_batch_dir_name(entry.name)
        if parsed is None:
            continue
        country, batch_hz, version = parsed
        targets.append(
            PublishTarget(
                country=country,
                batch_horizon=batch_hz,
                version=version,
                output_root=output_root,
                repo_root=defaults.repo_root,
                publish_root=defaults.publish_root,
                min_run_fraction=defaults.min_run_fraction,
            )
        )
    return targets


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
    key = horizon.strip().lower().replace("-", "_")
    if key in {"eos"}:
        return "eos"
    if key in {"mid", "mid_season", "middle_of_season", "middle-of-season", "mid-season"}:
        return "mid"
    raise ValueError(f"Unsupported horizon for batch naming: {horizon!r}")


def expected_job_count(target: PublishTarget) -> int:
    manifest = (
        target.repo_root
        / "cybench/runs/slurm/manifests"
        / target.batch_name
        / "benchmark_jobs.txt"
    )
    if not manifest.is_file():
        return 0
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


def count_complete_walk_forward_runs(target: PublishTarget) -> int:
    if not target.baselines_dir.is_dir():
        return 0
    seen: set[tuple[str, str, str]] = set()
    for horizon_tag in target.horizon_tags:
        runs = discover_benchmark_runs(
            target.baselines_dir,
            phase="walk_forward",
            horizon=horizon_tag,
            latest_only=True,
        )
        for run in runs:
            key = (run.crop, run.country, run.model)
            if key in seen:
                continue
            try:
                load_pooled_predictions(run.path, model_slug=run.model)
            except ValueError:
                continue
            seen.add(key)
    return len(seen)


def assess_readiness(target: PublishTarget) -> ReadinessReport:
    expected = expected_job_count(target)
    complete = count_complete_walk_forward_runs(target)
    if not target.baselines_dir.is_dir():
        return ReadinessReport(
            target=target,
            expected_runs=expected,
            complete_runs=complete,
            ready=False,
            reason=f"missing baselines dir {target.baselines_dir}",
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
    return ReadinessReport(
        target=target,
        expected_runs=expected,
        complete_runs=complete,
        ready=ready,
        reason=reason,
    )


def baselines_fingerprint(target: PublishTarget) -> dict[str, Any]:
    runs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for horizon_tag in target.horizon_tags:
        for run in discover_benchmark_runs(
            target.baselines_dir,
            phase="walk_forward",
            horizon=horizon_tag,
            latest_only=True,
        ):
            key = (run.crop, run.country, run.model)
            if key in seen:
                continue
            try:
                load_pooled_predictions(run.path, model_slug=run.model)
            except ValueError:
                continue
            seen.add(key)
            runs.append(
                {
                    "dataset": run.dataset,
                    "model": run.model,
                    "horizon": run.horizon,
                    "timestamp": run.timestamp,
                }
            )
    runs.sort(key=lambda r: (r["dataset"], r["model"], r["horizon"]))
    return {
        "baselines_dir": str(target.baselines_dir.resolve()),
        "n_runs": len(runs),
        "runs": runs,
    }


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
    cmd = [
        "poetry",
        "run",
        "python",
        str(script),
        "--baselines-dir",
        str(target.baselines_dir),
        "--output-dir",
        str(target.collect_dir),
        "--dashboard",
    ]
    if plot:
        cmd.append("--plot")
    if dry_run:
        print(f"[DRY-RUN] collect: {' '.join(cmd)}")
        return
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
) -> StageStatus:
    if dry_run:
        print(f"[DRY-RUN] rebuild index under {target.publish_root}")
        return StageStatus("index", False, "would rebuild index.html")
    entries = discover_index_entries(target.publish_root)
    index_path = update_index(target.publish_root, entries)
    try:
        from cybench.runs.analysis.build_global_insights_dashboard import write_insights_dashboard

        insights_path = write_insights_dashboard(
            output_root=target.output_root,
            dest=target.publish_root / "insights.html",
            version=target.version,
        )
        msg = f"updated {index_path} and {insights_path.name}"
    except RuntimeError as exc:
        msg = f"updated {index_path} (insights skipped: {exc})"
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
    insights = publish_root / "insights.html"
    if insights.is_file():
        rel_paths.append("insights.html")
    status = subprocess.run(
        ["git", "-C", str(publish_root), "status", "--porcelain", "--", *rel_paths],
        capture_output=True,
        text=True,
        check=False,
    )
    if not status.stdout.strip():
        return StageStatus("commit", True, "no changes to commit")

    message = f"Update {target.publish_slug} dashboard"
    if dry_run:
        print(f"[DRY-RUN] git commit in {publish_root}: {message}")
        if push:
            print("[DRY-RUN] git push")
        return StageStatus("commit", False, f"would commit: {message}")

    subprocess.run(
        ["git", "-C", str(publish_root), "add", *rel_paths],
        check=True,
    )
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

    if countries:
        wanted = {c.upper() for c in countries}
        targets = [t for t in targets if t.country_upper in wanted]
    if horizons:
        wanted_hz = {horizon_to_batch_suffix(h) for h in horizons}
        targets = [t for t in targets if t.batch_horizon in wanted_hz]

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
