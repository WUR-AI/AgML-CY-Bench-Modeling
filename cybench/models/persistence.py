"""Symmetric pickle save/load helpers for tabular CY-Bench models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def pickle_path(model_path: str | Path, name: str) -> Path:
    """Resolve the pickle file path for save/load.

    A path ending in ``.pkl`` is treated as a file. Otherwise ``model_path`` is
    treated as a directory and ``{name}.pkl`` is appended.
    """
    path = Path(model_path)
    if path.suffix == ".pkl":
        return path
    return path / f"{name}.pkl"


def save_pickle(obj: Any, model_path: str | Path, name: str) -> None:
    path = pickle_path(model_path, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(model_path: str | Path, name: str) -> Any:
    with open(pickle_path(model_path, name), "rb") as f:
        return pickle.load(f)
