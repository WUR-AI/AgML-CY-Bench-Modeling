"""Feature selection utilities (mRMR at each forecast origin)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd
from feature_engine.selection import MRMR
from omegaconf import DictConfig, OmegaConf

from cybench.config import KEY_TARGET, KEY_YEAR
from cybench.datasets.dataset import PandasDataset
from cybench.util.config_utils import remove_search_keys

log = logging.getLogger(__name__)


def _target_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Target must be a single column.")
        return y.iloc[:, 0]
    return y


def _filter_columns_by_coverage(x: pd.DataFrame, max_nan_rate: float) -> pd.DataFrame:
    """Keep columns with at most ``max_nan_rate`` missing values in this training slice."""
    if max_nan_rate < 0 or max_nan_rate > 1:
        raise ValueError(f"max_nan_rate must be in [0, 1], got {max_nan_rate}")

    keep = [
        str(col)
        for col in x.columns
        if float(x[col].isna().mean()) <= max_nan_rate
    ]
    if not keep:
        raise ValueError(
            f"No features left after coverage filter (max_nan_rate={max_nan_rate})."
        )
    return x.loc[:, keep]


def _drop_zero_variance_columns(x: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance on the given training rows."""
    std = x.std(ddof=0)
    dropped = std[std == 0].index.tolist()
    if dropped:
        log.info(
            "Dropped %d zero-variance feature(s) before mRMR (e.g. %s).",
            len(dropped),
            ", ".join(str(c) for c in dropped[:5]),
        )
        x = x.drop(columns=dropped)
    if x.empty:
        raise ValueError("No features left after dropping zero-variance columns.")
    return x


def select_mrmr_features(
    x: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    *,
    k: int,
    method: str = "FCD",
    max_nan_rate: float = 0.05,
) -> list[str]:
    """Run feature-engine MRMR and return selected column names in selection order."""
    if x.empty:
        raise ValueError("Cannot run mRMR on an empty feature matrix.")

    y = _target_series(y)
    x = _filter_columns_by_coverage(x, max_nan_rate=max_nan_rate)

    # Complete cases only — no imputation of structural missingness.
    valid = x.notna().all(axis=1) & y.notna()
    x = x.loc[valid]
    y_fit: pd.Series = y.loc[valid]
    if x.empty:
        raise ValueError("No complete cases left for mRMR after dropping NaN rows.")

    x = _drop_zero_variance_columns(x)

    k = min(int(k), x.shape[1])
    if k <= 0:
        raise ValueError(f"mRMR requires k >= 1, got {k}.")

    selector = MRMR(method=method, max_features=k, regression=True)
    selector.fit(x, y_fit)
    return [str(name) for name in selector.get_feature_names_out()]


def fit_mrmr_on_years(
    dataset: PandasDataset,
    train_years: list[int],
    fs_cfg: DictConfig,
) -> list[str]:
    """Fit mRMR using only rows from ``train_years`` (supervised selection)."""
    mask = dataset.y.index.get_level_values(KEY_YEAR).isin(train_years)
    x = dataset.x.loc[mask]
    y = dataset.y.loc[mask, KEY_TARGET]
    return select_mrmr_features(
        x,
        y,
        k=int(fs_cfg.k),
        method=str(fs_cfg.get("mrmr_method", "FCD")),
        max_nan_rate=float(fs_cfg.get("max_column_nan_rate", 0.05)),
    )


def apply_mrmr_at_origin(
    *,
    source_dataset: PandasDataset,
    train_years: list[int],
    fs_cfg: DictConfig,
    train_dataset: PandasDataset,
    eval_dataset: PandasDataset,
) -> tuple[PandasDataset, PandasDataset, list[str]]:
    """Fit mRMR on the current training window; subset train and eval pools."""
    log.debug(
        "mRMR at origin | train_years=%s | k=%d | method=%s | candidates=%d",
        train_years,
        int(fs_cfg.k),
        fs_cfg.get("mrmr_method", "FCD"),
        source_dataset.x.shape[1],
    )
    selected = fit_mrmr_on_years(source_dataset, train_years, fs_cfg=fs_cfg)
    return (
        train_dataset.select_features(selected),
        eval_dataset.select_features(selected),
        selected,
    )


def save_selected_features(
    path: Path,
    *,
    selected: list[str],
    fs_cfg: DictConfig,
    train_years: list[int],
) -> None:
    OmegaConf.save(
        OmegaConf.create(
            {
                "method": fs_cfg.name,
                "mrmr_method": str(fs_cfg.get("mrmr_method", "FCD")),
                "k": int(fs_cfg.k),
                "max_column_nan_rate": float(fs_cfg.get("max_column_nan_rate", 0.05)),
                "train_years": list(train_years),
                "selected_features": selected,
            }
        ),
        f=path,
    )


def feature_selection_enabled(cfg) -> bool:
    """Whether mRMR should run for this experiment.

    Missing ``validation.apply_feature_selection`` keeps the historical default:
    apply a feature-selection config or a frozen screening artifact when one exists.
    Set the flag false to keep screened model hyperparameters and use the full
    feature table (the ``nofeatsel`` ablation).
    """
    value = OmegaConf.select(cfg, "validation.apply_feature_selection", default=None)
    if value is None:
        return True
    return bool(value)


def resolved_feature_selection_cfg(cfg) -> DictConfig | None:
    """Return feature-selection config with any ``_search_`` keys stripped."""
    if not feature_selection_enabled(cfg):
        return None
    if "feature_selection" not in cfg:
        return None
    return remove_search_keys(cfg.feature_selection)


def resolve_run_feature_selection(
    cfg,
    *,
    is_walk_forward: bool,
    frozen_fs_cfg: DictConfig | None,
) -> DictConfig | None:
    """Feature-selection config for one split, or None to keep every column.

    Walk-forward normally freezes ``optimal_feature_selection.yaml`` from screening.
    ``validation.apply_feature_selection=false`` ignores that artifact and does not
    read a Hydra ``feature_selection`` group. Model hyperparameters stay on
    ``optimal_model.yaml``.
    """
    if not feature_selection_enabled(cfg):
        return None
    if is_walk_forward:
        if frozen_fs_cfg is None:
            return None
        return cast(DictConfig, OmegaConf.create(OmegaConf.to_container(frozen_fs_cfg)))
    return resolved_feature_selection_cfg(cfg)
