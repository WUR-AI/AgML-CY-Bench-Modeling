"""Tests for mRMR feature selection."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.datasets.dataset import PandasDataset
from cybench.util.feature_selection import (
    apply_mrmr_at_origin,
    fit_mrmr_on_years,
    resolved_feature_selection_cfg,
    select_mrmr_features,
)


def _fs_cfg(k: int = 4) -> DictConfig:
    return OmegaConf.create(
        {
            "name": "mrmr",
            "mrmr_method": "FCD",
            "k": k,
            "max_column_nan_rate": 0.05,
        }
    )


def _make_dataset(n_samples: int = 200, n_features: int = 10) -> PandasDataset:
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(2010, 2020), n_samples // 10)
    locs = np.tile(np.arange(n_samples // 10), 10)
    index = pd.MultiIndex.from_arrays(
        [locs.astype(str), years],
        names=[KEY_LOC, KEY_YEAR],
    )

    signal = rng.normal(size=n_samples)
    x = pd.DataFrame(
        {
            f"signal_{i}": signal + rng.normal(scale=0.1, size=n_samples)
            for i in range(3)
        }
        | {
            f"noise_{i}": rng.normal(size=n_samples)
            for i in range(n_features - 3)
        },
        index=index,
    )
    y = pd.DataFrame({KEY_TARGET: signal + rng.normal(scale=0.05, size=n_samples)}, index=index)
    return PandasDataset(cfg=None, y=y, x=x)


def test_mrmr_prefers_informative_features():
    ds = _make_dataset()
    selected = select_mrmr_features(ds.x, ds.y[KEY_TARGET], k=3)
    assert len(selected) == 3
    assert sum(col.startswith("signal_") for col in selected) >= 2


def test_mrmr_k_capped_at_n_features():
    ds = _make_dataset(n_features=5)
    selected = select_mrmr_features(ds.x, ds.y[KEY_TARGET], k=100)
    assert len(selected) == 5


def test_select_features_subsets_columns():
    ds = _make_dataset()
    subset = ds.select_features(["signal_0", "noise_0"])
    assert list(subset.x.columns) == ["signal_0", "noise_0"]
    assert len(subset) == len(ds)


def test_fit_mrmr_uses_train_years_only():
    ds = _make_dataset()
    selected_early = fit_mrmr_on_years(ds, [2010, 2011, 2012], fs_cfg=_fs_cfg())
    selected_late = fit_mrmr_on_years(ds, [2017, 2018, 2019], fs_cfg=_fs_cfg())
    assert len(selected_early) == 4
    assert len(selected_late) == 4


def test_apply_mrmr_at_origin_subsets_both_splits():
    ds = _make_dataset()
    train_ds, test_ds = ds.split_on_years(([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017], [2018, 2019]))
    train_out, test_out, selected = apply_mrmr_at_origin(
        source_dataset=ds,
        train_years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
        fs_cfg=_fs_cfg(),
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    assert len(selected) == 4
    assert list(train_out.x.columns) == selected
    assert list(test_out.x.columns) == selected


def test_resolved_feature_selection_cfg_strips_search():
    cfg = OmegaConf.create(
        {
            "feature_selection": {
                "name": "mrmr",
                "mrmr_method": "FCD",
                "k": 50,
                "_search_": {"k": {"type": "int", "low": 10, "high": 100}},
            }
        }
    )
    resolved = resolved_feature_selection_cfg(cfg)
    assert resolved is not None
    assert resolved.k == 50
    assert "_search_" not in resolved


def test_mrmr_drops_sparse_columns_without_imputation():
    ds = _make_dataset()
    x = ds.x.copy()
    x["sparse_8"] = np.nan
    x.loc[x.index[:50], "sparse_8"] = 1.0
    selected = select_mrmr_features(x, ds.y[KEY_TARGET], k=3, max_nan_rate=0.05)
    assert len(selected) == 3
    assert "sparse_8" not in selected


def test_mrmr_uses_complete_cases_only():
    ds = _make_dataset()
    x = ds.x.copy()
    x.loc[x.index[:5], "noise_0"] = np.nan
    selected = select_mrmr_features(x, ds.y[KEY_TARGET], k=3)
    assert len(selected) == 3


def test_mrmr_drops_zero_variance_columns():
    ds = _make_dataset()
    x = ds.x.copy()
    x["dead_feature"] = 0.0
    x["dead_feature_2"] = 1.0
    selected = select_mrmr_features(x, ds.y[KEY_TARGET], k=3)
    assert len(selected) == 3
    assert "dead_feature" not in selected
    assert "dead_feature_2" not in selected


def test_mrmr_rejects_empty_matrix():
    index = pd.MultiIndex.from_tuples([], names=[KEY_LOC, KEY_YEAR])
    x = pd.DataFrame(index=index)
    y = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="empty"):
        select_mrmr_features(x, y, k=1)
