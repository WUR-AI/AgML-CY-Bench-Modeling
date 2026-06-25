"""Tests for temporal validation splits."""

from hydra import compose, initialize

from cybench.util.validation import (
    default_screening_validation_cfg,
    get_screening_partitions,
    get_splits,
)


def _screening_cfg():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        return compose(
            config_name="config",
            overrides=[
                "validation=screening",
                "validation.test_years=5-last",
                "validation.val_years=2-last",
            ],
        ).validation


def test_default_screening_validation_cfg_matches_yaml():
    cfg = default_screening_validation_cfg()
    assert cfg.name == "screening"
    assert cfg.test_years == "5-last"
    assert cfg.val_years == "2-last"


def test_screening_partitions_fixed_layout():
    cfg = _screening_cfg()
    years = set(range(2000, 2025))  # 2000..2024

    train, val, test = get_screening_partitions(cfg, years)

    assert test == list(range(2020, 2025))
    assert val == [2018, 2019]
    assert train == list(range(2000, 2018))


def test_screening_partitions_accepts_numpy_int64_years():
    import numpy as np
    from omegaconf import OmegaConf

    cfg = _screening_cfg()
    years = {np.int64(y) for y in range(2000, 2025)}

    train, val, test = get_screening_partitions(cfg, years)

    conf = OmegaConf.create(
        {
            "train_years": train,
            "val_years": val,
            "test_years": test,
            "final_fit_years": train + val,
        }
    )
    assert conf.train_years[0] == 2000


def test_screening_hpo_uses_train_val_only():
    cfg = _screening_cfg()
    years = set(range(2000, 2025))
    train, val, test = get_screening_partitions(cfg, years)

    splits = list(get_splits(cfg=cfg, which="val", dataset_years=years, seed=42))
    assert len(splits) == 1
    hpo_train, hpo_val = splits[0]
    assert hpo_train == train
    assert hpo_val == val
    assert not set(hpo_train) & set(test)
    assert not set(hpo_val) & set(test)
    # Regression: must not use trailing calendar years (old bug used 2023/2024 as val).
    assert hpo_val != years_list_tail(years, 2)


def years_list_tail(years, k):
    return sorted(years)[-k:]


def test_screening_hpo_maize_us_like_years():
    """Regression for maize/US: HPO must not train on or validate on test years."""
    cfg = _screening_cfg()
    years = set(range(2003, 2024))  # 2003..2023

    train, val, test = get_screening_partitions(cfg, years)
    assert test == list(range(2019, 2024))
    assert val == [2017, 2018]
    assert train == list(range(2003, 2017))

    hpo_train, hpo_val = next(get_splits(cfg=cfg, which="val", dataset_years=years))
    assert hpo_train == train
    assert hpo_val == val
    assert not set(hpo_train) & set(test)
    assert not set(hpo_val) & set(test)


def test_screening_hpo_truncated_years_gives_wrong_val_window():
    """Document regression: pre-test years alone must not drive HPO val selection."""
    cfg = _screening_cfg()
    years = set(range(2003, 2024))
    train, val, test = get_screening_partitions(cfg, years)
    pre_test = set(train) | set(val)

    wrong_train, wrong_val = next(
        get_splits(cfg=cfg, which="val", dataset_years=pre_test, seed=42)
    )
    assert wrong_val == [2012, 2013]
    assert wrong_val != val


def test_screening_pre_test_years_for_normalizer():
    from cybench.util.validation import get_screening_pre_test_years

    cfg = _screening_cfg()
    years = set(range(2003, 2024))
    train, val, test = get_screening_partitions(cfg, years)
    fit_years = get_screening_pre_test_years(years, seed=42, cfg=cfg)
    assert fit_years == train + val
    assert not set(fit_years) & set(test)


def test_screening_final_fit_pool_excludes_test():
    cfg = _screening_cfg()
    years = set(range(2000, 2025))

    splits = list(get_splits(cfg=cfg, which="test", dataset_years=years, seed=42))
    assert len(splits) == 1
    final_fit_years, test_years = splits[0]

    assert test_years == list(range(2020, 2025))
    assert final_fit_years == list(range(2000, 2020))
    assert not set(final_fit_years) & set(test_years)
