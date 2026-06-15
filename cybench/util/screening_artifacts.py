"""Load and save frozen artifacts from the screening phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def save_optimal_epochs(
    path: Path | str,
    E_star: int,
    *,
    max_epochs_budget: int | None = None,
    best_val_loss: float | None = None,
) -> None:
    payload: dict[str, Any] = {"E_star": int(E_star)}
    if max_epochs_budget is not None:
        payload["max_epochs_budget"] = int(max_epochs_budget)
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    OmegaConf.save(OmegaConf.create(payload), str(path))


def load_optimal_epochs(path: Path | str) -> int | None:
    fp = Path(path)
    if not fp.exists():
        return None
    data = OmegaConf.load(fp)
    if "E_star" not in data:
        return None
    return int(data.E_star)


def load_frozen_screening_artifacts(screening_split_dir: Path | str) -> tuple[DictConfig, DictConfig | None, int | None]:
    """
    Load optimal model / feature-selection / E* configs from a screening split folder.
    """
    root = Path(screening_split_dir)
    model_path = root / "optimal_model.yaml"
    if not model_path.exists():
        raise FileNotFoundError(f"No optimal_model.yaml in {root}")
    model_cfg = cast(DictConfig, OmegaConf.load(model_path))

    fs_cfg: DictConfig | None = None
    fs_path = root / "optimal_feature_selection.yaml"
    if fs_path.exists():
        fs_cfg = cast(DictConfig, OmegaConf.load(fs_path))

    E_star = load_optimal_epochs(root / "optimal_epochs.yaml")
    return model_cfg, fs_cfg, E_star
