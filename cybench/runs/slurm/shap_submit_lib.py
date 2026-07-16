"""Plan parallel SHAP array submissions across benchmark crop×country pairs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from cybench.config import DATASETS, PATH_DATA_DIR
from cybench.runs.analysis.shap_importance_lib import (
    discover_origin_record_paths,
    find_walk_forward_run_dir,
)
from cybench.runs.slurm.benchmark_completion_lib import load_yield_years
from cybench.runs.slurm.benchmark_submit_lib import (
    batch_name,
    count_regions,
    horizon_batch_suffix,
    normalize_horizon,
    resolve_batch_dir,
    slurm_memory_for_country,
)
from cybench.util.benchmark_scope import is_benchmark_evaluation_crop_country
from cybench.util.validation import expected_walk_forward_test_years

DEFAULT_MODELS = ("random_forest", "transformer_lf")
DEFAULT_OUTPUT_ROOT = Path("/lustre/backup/SHARED/AIN/agml/output")
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ShapSubmitPlan:
    crop: str
    country: str
    model: str
    horizon: str
    batch: str
    baselines_dir: str
    output_dir: str
    origins: tuple[int, ...]
    origins_list: str
    array_spec: str
    n_regions: int
    n_pending: int
    skip: bool
    skip_reason: str = ""
    slurm_mem: str | None = None

    @property
    def country_upper(self) -> str:
        return self.country.upper()


def benchmark_crop_country_pairs(
    *,
    crops: list[str] | None = None,
    countries: list[str] | None = None,
    data_dir: Path | None = None,
) -> list[tuple[str, str]]:
    root = Path(data_dir or PATH_DATA_DIR)
    crop_list = crops or list(DATASETS)
    pairs: list[tuple[str, str]] = []
    country_filter = {c.upper() for c in countries} if countries else None
    for crop in crop_list:
        if crop not in DATASETS:
            raise ValueError(f"Unknown crop {crop!r}. Choose from {list(DATASETS)}")
        for country in DATASETS[crop]:
            if country_filter is not None and country.upper() not in country_filter:
                continue
            if not (root / crop / country).is_dir():
                continue
            if not is_benchmark_evaluation_crop_country(crop, country, data_dir=root):
                continue
            pairs.append((crop, country))
    return sorted(pairs)


def _completed_origin_years(model_dir: Path) -> set[int]:
    years: set[int] = set()
    for path in discover_origin_record_paths(model_dir):
        years.add(int(path.parent.name.removeprefix("origin_")))
    return years


def build_shap_submit_plans(
    *,
    models: list[str] | None = None,
    crops: list[str] | None = None,
    countries: list[str] | None = None,
    horizon: str = "eos",
    version: int = 4,
    seed: int = DEFAULT_SEED,
    output_root: Path | None = None,
    data_dir: Path | None = None,
    pending_only: bool = True,
    force: bool = False,
) -> list[ShapSubmitPlan]:
    model_list = list(models or DEFAULT_MODELS)
    root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    data_dir = Path(data_dir or PATH_DATA_DIR)
    horizon_norm = normalize_horizon(horizon)
    hz_tag = horizon_batch_suffix(horizon_norm)

    plans: list[ShapSubmitPlan] = []
    for crop, country in benchmark_crop_country_pairs(
        crops=crops, countries=countries, data_dir=data_dir
    ):
        cc = country.upper()
        batch = batch_name(cc, horizon_norm, version)
        baselines_dir, _note = resolve_batch_dir(root, batch)
        n_regions = count_regions(cc, data_dir)
        slurm_mem = slurm_memory_for_country(cc, data_dir=data_dir)
        out_dir = root / "shap_importance" / f"{crop}_{cc}_{hz_tag}"

        if not baselines_dir.is_dir():
            for model in model_list:
                plans.append(
                    ShapSubmitPlan(
                        crop=crop,
                        country=cc,
                        model=model,
                        horizon=horizon_norm,
                        batch=batch,
                        baselines_dir=str(baselines_dir),
                        output_dir=str(out_dir),
                        origins=(),
                        origins_list="",
                        array_spec="",
                        n_regions=n_regions,
                        n_pending=0,
                        skip=True,
                        skip_reason=f"missing baselines dir: {baselines_dir}",
                        slurm_mem=slurm_mem,
                    )
                )
            continue

        years = load_yield_years(crop, cc, data_dir=data_dir)
        expected_origins = tuple(
            expected_walk_forward_test_years(years, seed=seed)
        )
        if not expected_origins:
            for model in model_list:
                plans.append(
                    ShapSubmitPlan(
                        crop=crop,
                        country=cc,
                        model=model,
                        horizon=horizon_norm,
                        batch=batch,
                        baselines_dir=str(baselines_dir),
                        output_dir=str(out_dir),
                        origins=(),
                        origins_list="",
                        array_spec="",
                        n_regions=n_regions,
                        n_pending=0,
                        skip=True,
                        skip_reason="no walk-forward origins resolved from yield years",
                        slurm_mem=slurm_mem,
                    )
                )
            continue

        for model in model_list:
            try:
                find_walk_forward_run_dir(
                    baselines_dir,
                    crop=crop,
                    country=cc,
                    model_slug=model,
                    horizon=horizon_norm,
                )
            except FileNotFoundError as exc:
                plans.append(
                    ShapSubmitPlan(
                        crop=crop,
                        country=cc,
                        model=model,
                        horizon=horizon_norm,
                        batch=batch,
                        baselines_dir=str(baselines_dir),
                        output_dir=str(out_dir),
                        origins=expected_origins,
                        origins_list=" ".join(str(y) for y in expected_origins),
                        array_spec=f"0-{len(expected_origins) - 1}",
                        n_regions=n_regions,
                        n_pending=len(expected_origins),
                        skip=True,
                        skip_reason=str(exc),
                        slurm_mem=slurm_mem,
                    )
                )
                continue

            model_dir = out_dir / f"{crop}_{cc}" / model
            done = _completed_origin_years(model_dir)
            if force:
                pending = expected_origins
            else:
                pending = tuple(y for y in expected_origins if y not in done)

            if pending_only and not pending:
                plans.append(
                    ShapSubmitPlan(
                        crop=crop,
                        country=cc,
                        model=model,
                        horizon=horizon_norm,
                        batch=batch,
                        baselines_dir=str(baselines_dir),
                        output_dir=str(out_dir),
                        origins=expected_origins,
                        origins_list=" ".join(str(y) for y in expected_origins),
                        array_spec=f"0-{len(expected_origins) - 1}",
                        n_regions=n_regions,
                        n_pending=0,
                        skip=True,
                        skip_reason=f"all {len(expected_origins)} origins complete",
                        slurm_mem=slurm_mem,
                    )
                )
                continue

            origins_list = " ".join(str(y) for y in pending)
            array_spec = f"0-{len(pending) - 1}"
            plans.append(
                ShapSubmitPlan(
                    crop=crop,
                    country=cc,
                    model=model,
                    horizon=horizon_norm,
                    batch=batch,
                    baselines_dir=str(baselines_dir),
                    output_dir=str(out_dir),
                    origins=pending,
                    origins_list=origins_list,
                    array_spec=array_spec,
                    n_regions=n_regions,
                    n_pending=len(pending),
                    skip=False,
                    slurm_mem=slurm_mem,
                )
            )
    return plans


def _cmd_list(args: argparse.Namespace) -> int:
    plans = build_shap_submit_plans(
        models=args.models,
        crops=args.crops,
        countries=args.countries,
        horizon=args.horizon,
        version=args.version,
        seed=args.seed,
        output_root=Path(args.output_root) if args.output_root else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        pending_only=not args.all,
        force=args.force,
    )
    if args.json:
        print(json.dumps([asdict(p) for p in plans], indent=2))
        return 0
    if args.plan_tsv:
        for plan in plans:
            if plan.skip:
                continue
            mem = plan.slurm_mem or ""
            print(
                f"{plan.crop}\t{plan.country_upper}\t{plan.model}\t{plan.horizon}\t"
                f"{plan.batch}\t{plan.origins_list}\t{plan.array_spec}\t"
                f"{plan.n_regions}\t{plan.n_pending}\t{plan.baselines_dir}\t"
                f"{plan.output_dir}\t{mem}"
            )
        return 0

    print(
        f"{'crop':<6} {'cc':<4} {'model':<16} {'regions':>7}  {'pending':>7}  "
        f"{'action':<6}  note"
    )
    print("-" * 100)
    for plan in plans:
        action = "skip" if plan.skip else "submit"
        note = plan.skip_reason if plan.skip else f"{plan.array_spec} | {plan.origins_list}"
        print(
            f"{plan.crop:<6} {plan.country_upper:<4} {plan.model:<16} "
            f"{plan.n_regions:>7}  {plan.n_pending:>7}  {action:<6}  {note}"
        )
    n_submit = sum(1 for p in plans if not p.skip)
    n_tasks = sum(p.n_pending for p in plans if not p.skip)
    print(f"\n{len(plans)} planned, {n_submit} jobs, {n_tasks} array tasks")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--crops", nargs="*", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"Default: {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument("--horizon", default="eos")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--all", action="store_true", help="Include already-complete jobs")
    parser.add_argument("--force", action="store_true", help="Resubmit all origins")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--plan-tsv",
        action="store_true",
        help=(
            "Machine-readable submit rows: crop country model horizon batch "
            "origins_list array_spec n_regions n_pending baselines_dir output_dir slurm_mem"
        ),
    )
    args = parser.parse_args()
    raise SystemExit(_cmd_list(args))


if __name__ == "__main__":
    main()
