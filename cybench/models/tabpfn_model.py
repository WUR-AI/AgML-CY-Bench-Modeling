"""Backward-compatible re-exports; implementations live in tabular_foundation_model."""

from cybench.models.tabular_foundation_model import (
    PreprocessMode,
    SubsampleMode,
    TabPFNModel,
    TabularRegressor,
    _is_cuda_oom_error,
    _predict_in_batches,
    _subsample_indices,
)

__all__ = [
    "PreprocessMode",
    "SubsampleMode",
    "TabPFNModel",
    "TabularRegressor",
    "_is_cuda_oom_error",
    "_predict_in_batches",
    "_subsample_indices",
]
