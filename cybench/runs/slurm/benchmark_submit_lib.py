"""Plan SLURM benchmark submissions by country (region counts, filed batches)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from cybench.config import DATASETS, PATH_DATA_DIR

# Countries with at least this many regions use the gpu SLURM partition for torch/TabPFN.
DEFAULT_GPU_REGION_THRESHOLD = 105

_BATCH_RE = re.compile(
    r"^baselines_(?P<country>[A-Za-z]{2})_(?P<batch_hz>eos|mid|qtr|early)_v(?P<version>\d+)$"
)


def normalize_horizon(horizon: str) -> str:
    """SLURM / Hydra value for ``PREDICTION_HORIZON`` (never bare ``mid``)."""
    key = horizon.strip().lower().replace("-", "_")
    if key == "eos":
        return "eos"
    if key in {"mid", "mid_season", "middle_of_season", "midseason"}:
        return "middle-of-season"
    if key in {"quarter", "quarter_season", "quarter_of_season", "qtr"}:
        return "quarter-of-season"
    if key in {"early", "early_season", "early_season"}:
        return "early-season"
    return horizon.strip()


def horizon_batch_suffix(horizon: str) -> str:
    """Short tag used in batch folder names (``eos`` / ``mid``)."""
    norm = normalize_horizon(horizon)
    if norm == "eos":
        return "eos"
    if norm == "middle-of-season":
        return "mid"
    if norm.startswith("quarter") or norm == "qtr":
        return "qtr"
    if norm == "early-season" or norm.startswith("early"):
        return "early"
    return re.sub(r"[^a-z0-9]+", "_", norm.lower()).strip("_")[:8] or "custom"


def batch_suffix_to_horizon(batch_hz: str) -> str:
    """Map batch folder horizon tag back to ``PREDICTION_HORIZON`` / Hydra value."""
    key = batch_hz.strip().lower()
    if key == "eos":
        return "eos"
    if key == "mid":
        return "middle-of-season"
    if key == "qtr":
        return "quarter-of-season"
    if key == "early":
        return "early-season"
    return batch_hz


def batch_name(country: str, horizon: str, version: int = 3) -> str:
    cc = country.upper()
    hz = horizon_batch_suffix(horizon)
    return f"baselines_{cc}_{hz}_v{version}"


def parse_batch_name(name: str) -> tuple[str, str, int] | None:
    match = _BATCH_RE.match(name)
    if not match:
        return None
    return match.group("country").upper(), match.group("batch_hz"), int(match.group("version"))


def resolve_case_insensitive_child(parent: Path, name: str) -> Path | None:
    """Return an existing child of ``parent`` whose name matches ``name`` case-insensitively."""
    direct = parent / name
    if direct.exists():
        return direct
    if not parent.is_dir():
        return None
    key = name.casefold()
    for entry in parent.iterdir():
        if entry.name.casefold() == key:
            return entry
    return None


def resolve_batch_dir(parent: Path, batch: str) -> tuple[Path, str | None]:
    """Resolve a baselines/manifest batch folder; tolerate upper/lower country codes."""
    resolved = resolve_case_insensitive_child(parent, batch)
    if resolved is not None:
        if resolved.name == batch:
            return resolved, None
        return resolved, f"using {resolved.name} (case alias for {batch})"

    parsed = parse_batch_name(batch)
    if parsed and parent.is_dir():
        cc, hz, ver = parsed
        for cc_variant in (cc.upper(), cc.lower()):
            candidate = f"baselines_{cc_variant}_{hz}_v{ver}"
            resolved = resolve_case_insensitive_child(parent, candidate)
            if resolved is not None:
                note = (
                    None
                    if resolved.name == batch
                    else f"using {resolved.name} (case alias for {batch})"
                )
                return resolved, note

    return parent / batch, None


def gpu_partition_for_country(
    country: str,
    *,
    region_threshold: int = DEFAULT_GPU_REGION_THRESHOLD,
    data_dir: Path | None = None,
) -> tuple[bool, int]:
    """Return whether the torch/TabPFN group should use the gpu partition."""
    n_regions = count_regions(country, data_dir)
    return n_regions >= region_threshold, n_regions


def gpu_partition_for_batch(
    batch: str,
    *,
    region_threshold: int = DEFAULT_GPU_REGION_THRESHOLD,
    data_dir: Path | None = None,
) -> tuple[bool | None, int, str | None]:
    """Return (use_gpu_partition, n_regions, country) for a baselines_* batch name."""
    parsed = parse_batch_name(batch)
    if parsed is None:
        return None, 0, None
    country, _, _ = parsed
    use_gpu, n_regions = gpu_partition_for_country(
        country,
        region_threshold=region_threshold,
        data_dir=data_dir,
    )
    return use_gpu, n_regions, country


def count_regions(country: str, data_dir: Path | None = None) -> int:
    """Max unique ``adm_id`` across maize/wheat yield files for a country."""
    data_dir = Path(data_dir or PATH_DATA_DIR)
    cc = country.upper()
    max_regions = 0
    for crop in ("maize", "wheat"):
        if cc not in DATASETS.get(crop, ()):
            continue
        path = data_dir / crop / cc / f"yield_{crop}_{cc}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            ids = {row["adm_id"] for row in reader if row.get("adm_id")}
        max_regions = max(max_regions, len(ids))
    return max_regions


def countries_with_data(data_dir: Path | None = None) -> list[str]:
    data_dir = Path(data_dir or PATH_DATA_DIR)
    found: set[str] = set()
    for crop in ("maize", "wheat"):
        crop_dir = data_dir / crop
        if not crop_dir.is_dir():
            continue
        for entry in crop_dir.iterdir():
            if entry.is_dir() and len(entry.name) == 2:
                found.add(entry.name.upper())
    return sorted(found)


def filed_batches(manifest_root: Path) -> set[tuple[str, str, int]]:
    """Return set of (country_upper, batch_hz, version) with manifest dirs."""
    filed: set[tuple[str, str, int]] = set()
    if not manifest_root.is_dir():
        return filed
    for entry in manifest_root.iterdir():
        if not entry.is_dir():
            continue
        match = _BATCH_RE.match(entry.name)
        if match:
            filed.add(
                (
                    match.group("country").upper(),
                    match.group("batch_hz"),
                    int(match.group("version")),
                )
            )
    return filed


@dataclass(frozen=True)
class SubmitPlan:
    country: str
    horizon: str
    batch: str
    n_regions: int
    gpu_partition: bool  # True → gpu queue; False → --cpu for torch/TabPFN group only
    skip: bool
    skip_reason: str = ""

    @property
    def country_upper(self) -> str:
        return self.country.upper()


def build_submit_plans(
    *,
    countries: list[str] | None = None,
    horizons: list[str] | None = None,
    version: int = 3,
    region_threshold: int = DEFAULT_GPU_REGION_THRESHOLD,
    manifest_root: Path,
    data_dir: Path | None = None,
    pending_only: bool = True,
    force: bool = False,
) -> list[SubmitPlan]:
    data_dir = Path(data_dir or PATH_DATA_DIR)
    manifest_root = Path(manifest_root)
    hz_list = [normalize_horizon(h) for h in (horizons or ["eos", "middle-of-season"])]
    filed = filed_batches(manifest_root)

    cc_list = [c.upper() for c in (countries or countries_with_data(data_dir))]
    plans: list[SubmitPlan] = []
    for country in sorted(set(cc_list)):
        n_regions = count_regions(country, data_dir)
        for horizon in hz_list:
            hz_tag = horizon_batch_suffix(horizon)
            batch = batch_name(country, horizon, version)
            key = (country.upper(), hz_tag, version)
            gpu_partition = n_regions >= region_threshold

            if not force and key in filed:
                plans.append(
                    SubmitPlan(
                        country=country,
                        horizon=horizon,
                        batch=batch,
                        n_regions=n_regions,
                        gpu_partition=gpu_partition,
                        skip=True,
                        skip_reason=f"manifest exists: {manifest_root / batch}",
                    )
                )
                continue

            if n_regions < 1:
                plans.append(
                    SubmitPlan(
                        country=country,
                        horizon=horizon,
                        batch=batch,
                        n_regions=0,
                        gpu_partition=False,
                        skip=True,
                        skip_reason="no yield data found",
                    )
                )
                continue

            plans.append(
                SubmitPlan(
                    country=country,
                    horizon=horizon,
                    batch=batch,
                    n_regions=n_regions,
                    gpu_partition=gpu_partition,
                    skip=False,
                )
            )
    return plans


def _cmd_list(args: argparse.Namespace) -> int:
    plans = build_submit_plans(
        countries=args.countries,
        horizons=args.horizons,
        version=args.version,
        region_threshold=args.region_threshold,
        manifest_root=Path(args.manifest_root),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        pending_only=not args.all,
        force=args.force,
    )
    if args.json:
        print(json.dumps([asdict(p) for p in plans], indent=2))
        return 0
    if args.plan_tsv:
        for p in plans:
            if p.skip:
                continue
            gpu = "gpu" if p.gpu_partition else "cpu"
            print(f"{p.country_upper}\t{p.horizon}\t{p.batch}\t{gpu}\t{p.n_regions}")
        return 0
    print(
        f"{'country':<6} {'horizon':<18} {'regions':>7}  {'gpu':<5}  {'action':<6}  batch / note"
    )
    print("-" * 90)
    for p in plans:
        gpu = "yes" if p.gpu_partition else "cpu"
        action = "skip" if p.skip else "submit"
        note = p.skip_reason if p.skip else p.batch
        print(
            f"{p.country_upper:<6} {p.horizon:<18} {p.n_regions:>7}  {gpu:<5}  {action:<6}  {note}"
        )
    n_submit = sum(1 for p in plans if not p.skip)
    print(f"\n{len(plans)} planned, {n_submit} to submit (threshold={args.region_threshold} regions)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="Default: eos middle-of-season",
    )
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument(
        "--region-threshold",
        type=int,
        default=DEFAULT_GPU_REGION_THRESHOLD,
        help=(
            "Countries with >= N regions use gpu partition for the torch/TabPFN group "
            f"(default: {DEFAULT_GPU_REGION_THRESHOLD})"
        ),
    )
    parser.add_argument("--all", action="store_true", help="Include already-filed batches in listing")
    parser.add_argument("--force", action="store_true", help="Plan submit even if manifest exists")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--plan-tsv",
        action="store_true",
        help="Machine-readable lines: country horizon batch gpu|cpu n_regions (submit rows only)",
    )
    args = parser.parse_args()
    raise SystemExit(_cmd_list(args))


if __name__ == "__main__":
    main()
