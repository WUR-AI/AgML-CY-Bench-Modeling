from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

log = logging.getLogger(__name__)

_SCREENING_VALIDATION_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "validation" / "screening.yaml"
)
_WALK_FORWARD_VALIDATION_PATH = (
    Path(__file__).resolve().parents[1] / "conf" / "validation" / "walk_forward.yaml"
)


def _as_python_years(years) -> list[int]:
    """OmegaConf and YAML serializers require native int, not numpy int64."""
    return [int(y) for y in years]


def _sorted_years(dataset_years) -> list[int]:
    return sorted(int(y) for y in dataset_years)


def _val_window_size(val_years) -> int:
    if isinstance(val_years, ListConfig):
        return len(list(val_years))
    if isinstance(val_years, str) and val_years.endswith("-last"):
        return int(val_years.split("-")[0])
    if isinstance(val_years, (list, tuple)):
        return len(val_years)
    raise ValueError(f"Cannot infer validation window size from val_years={val_years!r}")


def _resolve_hold_out_years(
    split_years,
    years_list: list[Any],
    seed: int,
    which: str,
    cfg,
) -> list[Any]:
    """Resolve a hold-out year list from config (shared by all validation modes)."""
    if isinstance(split_years, ListConfig):
        split_years = list(split_years)
        assert all(
            year in years_list for year in split_years
        ), f"Selected years ({split_years}) are not in dataset: {years_list}"
        return split_years

    if isinstance(split_years, str):
        if split_years == "loyocv":
            return years_list

        if split_years.endswith("-last"):
            k = int(split_years.split("-")[0])

            if which == "test":
                available_test_years = len(years_list) - 1
                reserve_val = (
                    getattr(cfg, "name", None) != "walk_forward"
                    and isinstance(cfg.val_years, str)
                    and cfg.val_years.endswith("-last")
                )
                if reserve_val:
                    available_test_years -= int(cfg.val_years.split("-")[0])

                assert available_test_years, (
                    "Your validation configuration doesnt fit to the number of available years. "
                    f"Only {len(years_list)} available"
                )
                if k > available_test_years:
                    log.warning(
                        f"Validation doesnt happen on {k} last years but only on the "
                        f"{available_test_years} available"
                    )
                    k = available_test_years
            return years_list[-k:]

        if split_years.endswith("%-split"):
            percentage = int(split_years.split("%")[0])
            assert 0 < percentage < 100, f"Invalid percentage: {percentage}"
            n_hold_out = max(1, int(len(years_list) * percentage / 100))
            rng = np.random.RandomState(seed)
            return sorted(rng.choice(years_list, size=n_hold_out, replace=False).tolist())

        raise ValueError(f"Unknown split_years format: {split_years}")

    raise TypeError(f"split_years must be list or str, got {type(split_years)}")


def get_screening_partitions(
    cfg,
    dataset_years: set[Any],
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Fixed screening split: earliest years -> train, then val window, then test hold-out.

    Returns (train_years, val_years, test_years).
    """
    years_list = _sorted_years(dataset_years)
    test_years = _as_python_years(
        _resolve_hold_out_years(
            cfg.test_years, years_list, seed, which="test", cfg=cfg
        )
    )
    pre_test = [y for y in years_list if y not in test_years]

    if isinstance(cfg.val_years, ListConfig):
        val_years = _as_python_years(cfg.val_years)
    elif isinstance(cfg.val_years, (list, tuple)):
        val_years = _as_python_years(cfg.val_years)
    elif isinstance(cfg.val_years, str) and cfg.val_years.endswith("-last"):
        k = _val_window_size(cfg.val_years)
        assert len(pre_test) > k, (
            f"Not enough pre-test years ({len(pre_test)}) for a {k}-year validation window"
        )
        val_years = _as_python_years(pre_test[-k:])
    else:
        raise ValueError(f"Unsupported val_years for screening: {cfg.val_years!r}")

    assert all(y in pre_test for y in val_years), (
        f"Validation years {val_years} must lie before the test block {test_years}"
    )
    assert not set(val_years) & set(test_years), "Validation and test years must not overlap"

    train_years = _as_python_years(y for y in pre_test if y not in val_years)
    assert train_years, (
        f"No train years left. train={train_years}, val={val_years}, test={test_years}"
    )
    return train_years, val_years, test_years


def get_screening_pre_test_years(
    dataset_years: set[Any],
    *,
    seed: int = 42,
    cfg=None,
) -> list[int]:
    """Screening train ∪ val years — used to fit normalization (excludes test block)."""
    train_years, val_years, _test_years = get_screening_partitions(
        cfg, dataset_years, seed=seed
    )
    return _as_python_years(list(train_years) + list(val_years))


@lru_cache(maxsize=1)
def default_screening_validation_cfg() -> DictConfig:
    """Load ``cybench/conf/validation/screening.yaml`` (benchmark screening split)."""
    return cast(
        DictConfig,
        OmegaConf.create(
            yaml.safe_load(_SCREENING_VALIDATION_PATH.read_text(encoding="utf-8"))
        ),
    )


@lru_cache(maxsize=1)
def default_walk_forward_validation_cfg() -> DictConfig:
    """Load ``cybench/conf/validation/walk_forward.yaml`` (benchmark walk-forward split)."""
    return cast(
        DictConfig,
        OmegaConf.create(
            yaml.safe_load(_WALK_FORWARD_VALIDATION_PATH.read_text(encoding="utf-8"))
        ),
    )


def expected_walk_forward_test_years(
    dataset_years: set[Any],
    *,
    seed: int = 42,
    cfg: DictConfig | None = None,
) -> list[int]:
    """Forecast-origin test years for walk-forward (one rolling split per year)."""
    cfg = cfg or default_walk_forward_validation_cfg()
    years: list[int] = []
    for _train, test in get_splits(
        cfg, which="test", dataset_years=dataset_years, seed=seed
    ):
        years.extend(int(y) for y in test)
    return sorted(set(years))


def get_splits(
        cfg,
        which: str,
        dataset_years: set[Any],
        seed: int = 42,
) -> Iterator[tuple[list[Any], list[Any]]]:
    """
    Builds an iterator over test or val years based on a validation config file, see /conf/validation/...
    Yields (train_years, val_years) tuples

    Params:
        cfg:
        which: either 'test' or 'val'
        dataset_years: the set of available years in the dataset to be split
        seed: random seed for all splitting requiring randomness

    Usage:
        for train, val in get_train_val_splits(cfg.validation, dataset.years):
            train_ds, val_ds = dataset.split_on_years((train, val))
    """
    years_list = _sorted_years(dataset_years)

    if cfg.name == "screening":
        if which == "val":
            # HPO must use the fixed screening train/val window only; test years
            # are resolved from the full dataset and must never enter HPO splits.
            train_years, val_years, _test_years = get_screening_partitions(
                cfg, dataset_years, seed=seed
            )
            yield train_years, val_years
            return

        if which == "test":
            train_years, val_years, test_years = get_screening_partitions(
                cfg, dataset_years, seed=seed
            )
            yield train_years + val_years, test_years
            return

        raise ValueError(f"screening mode only supports which='val' or which='test', got {which!r}")

    split_years = cfg.test_years if which == "test" else cfg.val_years
    hold_out_years = _as_python_years(
        _resolve_hold_out_years(
            split_years, years_list, seed, which=which, cfg=cfg
        )
    )

    rolling_modes = ("rolling", "walk_forward")
    split_mode = cfg.name

    #### 2. Step: Select the training-set based on the split methode
    if split_mode == 'single':
        # returning a single set of train- and hold-out- years
        train_years = _as_python_years(y for y in years_list if y not in hold_out_years)
        assert train_years, f"No train years left. Hold-out-years: {hold_out_years} | Available years: {years_list}"
        yield train_years, hold_out_years

    elif split_mode in rolling_modes:
        # returning a set of PAST train-years for each hold-out-year
        for hold_out_year in sorted(hold_out_years):
            train_years = _as_python_years(y for y in years_list if y < hold_out_year)
            assert train_years, f"No train years left. Hold-out-year: {hold_out_year} | Available years: {years_list}"
            yield train_years, [hold_out_year]

    elif split_mode == 'loyocv':
        # returning a set of train-years for each hold-out-year
        for hold_out_year in sorted(hold_out_years):
            train_years = _as_python_years(y for y in years_list if y != hold_out_year)
            assert train_years, f"No train years left. Hold-out-year: {hold_out_year} | Available years: {years_list}"
            yield train_years, [hold_out_year]
    else:
        raise ValueError('Unknown split:', split_mode)
