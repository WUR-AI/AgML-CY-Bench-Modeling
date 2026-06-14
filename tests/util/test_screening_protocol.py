"""Tests for screening artifact I/O and walk-forward splits."""

from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

from cybench.util.screening_artifacts import load_optimal_epochs, save_optimal_epochs
from cybench.util.validation import get_splits


def test_save_and_load_optimal_epochs(tmp_path: Path):
    out = tmp_path / "optimal_epochs.yaml"
    save_optimal_epochs(out, 47, max_epochs_budget=100, best_val_loss=3.14)
    assert load_optimal_epochs(out) == 47


def test_walk_forward_rolling_splits_match_rolling():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        wf = compose(
            config_name="config",
            overrides=[
                "validation=walk_forward",
                "validation.test_years=5-last",
            ],
        ).validation
        rolling = compose(
            config_name="config",
            overrides=[
                "validation=rolling",
                "validation.test_years=5-last",
            ],
        ).validation

    years = set(range(2000, 2025))
    wf_splits = list(get_splits(cfg=wf, which="test", dataset_years=years, seed=42))
    rolling_splits = list(get_splits(cfg=rolling, which="test", dataset_years=years, seed=42))
    assert wf_splits == rolling_splits
