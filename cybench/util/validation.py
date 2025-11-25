from typing import Optional, List, Dict

import numpy as np
from omegaconf import DictConfig


def get_train_test_splits(
        cfg: Dict,
        years: set
):
    """
    Builds an iterator over test years based on a validation config file, see /conf/validation/...
    Yields (train_years, test_years) tuples

    Usage:
        for train, test in get_splits(cfg.validation, dataset.years):
            train_ds, test_ds = dataset.split_on_years((train, test))
    """
    years = sorted(years)

    if cfg.name == 'single':
        test = [y for y in cfg.test_years if y in years]
        if cfg.train_years:
            train = [y for y in (cfg.train_years + cfg.val_years) if y in years]
        else:
            train = [y for y in years if y not in  cfg.test_years]
        assert train, "No train years found. Please specify at least one year."
        assert test, "No test years found. Please specify at least one year."
        yield train, test

    elif cfg.name == 'rolling':
        if "-last" in cfg.test_years:
            k = int(cfg.test_years.split("-")[0])
            test_years = years[-k:]
        else:
            test_years = [y for y in cfg.test_years if y in years]

        for test_year in sorted(test_years):
            train = [y for y in years if y < test_year]
            assert train, f"Not enough data for test year '{test_year}'. There are {len(years)} years in your dataset"
            yield train, [test_year]

    elif cfg.name == 'loyocv':
        # check if a subset of test years is provided and otherwise take all years as test
        test_years = cfg.get('test_years', years)

        for test_year in sorted(test_years):
            if test_year not in years:
                continue
            train = [y for y in years if y != test_year]
            yield train, [test_year]
    else:
        raise ValueError('Unknown split:', cfg.name)

def get_train_val_splits(
        cfg: DictConfig,
        years: set
):
    pass
