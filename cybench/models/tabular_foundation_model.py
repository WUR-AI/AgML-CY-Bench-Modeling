from __future__ import annotations

import importlib
import logging
from abc import abstractmethod
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cybench.datasets.dataset import PandasDataset
from cybench.models.model import BaseModel
from cybench.models.persistence import load_pickle, save_pickle

log = logging.getLogger(__name__)

PreprocessMode = Literal["none", "sklearn"]
SubsampleMode = Literal["random", "quantile"]


@runtime_checkable
class TabularRegressor(Protocol):
    """Minimal regressor surface used by tabular foundation model wrappers."""

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> Any: ...

    def predict(self, X: npt.NDArray[Any], *args: Any, **kwargs: Any) -> npt.NDArray[Any]: ...


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Best-effort CUDA OOM detection across torch / driver error variants."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass

    msg = str(exc).lower()
    name = exc.__class__.__name__.lower()
    oom_markers = (
        "out of memory",
        "outofmemory",
        "cuda error",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "can't allocate",
        "cannot allocate",
        "failed to allocate",
    )
    if any(marker in msg for marker in oom_markers):
        return True
    return "oom" in name or "outofmemory" in name


def _subsample_indices(
    y: npt.NDArray[Any],
    n_samples: int,
    rng: np.random.Generator,
    *,
    method: SubsampleMode = "random",
    n_bins: int = 10,
) -> npt.NDArray[np.intp]:
    """Select training rows for tabular foundation model context limits."""
    n = len(y)
    if n <= n_samples:
        return np.arange(n, dtype=np.intp)

    if method == "random":
        return rng.choice(n, size=n_samples, replace=False)

    if method == "quantile":
        order = np.argsort(y)
        chunks = np.array_split(order, max(2, n_bins))
        per_chunk = max(1, n_samples // len(chunks))
        selected: list[int] = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            take = min(per_chunk, len(chunk))
            selected.extend(rng.choice(chunk, size=take, replace=False).tolist())

        if len(selected) < n_samples:
            remaining = np.setdiff1d(order, np.asarray(selected, dtype=np.intp))
            extra = min(n_samples - len(selected), len(remaining))
            if extra > 0:
                selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        if len(selected) > n_samples:
            selected = rng.choice(
                np.asarray(selected, dtype=np.intp),
                size=n_samples,
                replace=False,
            ).tolist()
        return np.asarray(selected, dtype=np.intp)

    raise ValueError(f"Unknown subsample method: {method!r}")


def _build_sklearn_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _require_module(module: str, install_hint: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(install_hint) from exc


def _import_symbol(module: str, attr: str, install_hint: str) -> Any:
    try:
        return getattr(importlib.import_module(module), attr)
    except ImportError as exc:
        raise ImportError(install_hint) from exc


class TabularFoundationModel(BaseModel):
    """Shared CY-Bench wrapper for tabular foundation regressors (TabPFN, TabICL, TabDPT)."""

    def __init__(
        self,
        *,
        name: str,
        verbose: bool = False,
        framework: str | None = None,
        device: str = "auto",
        predict_batch_size: int = 256,
        allow_cpu_fallback: bool = False,
        max_train_samples: int | None = None,
        subsample: SubsampleMode = "random",
        subsample_bins: int = 10,
        preprocess: PreprocessMode = "none",
        random_state: int = 42,
        estimator_kwargs: dict[str, Any] | None = None,
    ):
        self._check_import()
        self.name = name
        self.verbose = verbose
        self.device = _resolve_device(device)
        self.predict_batch_size = max(1, int(predict_batch_size))
        self.allow_cpu_fallback = allow_cpu_fallback
        self.max_train_samples = max_train_samples
        self.subsample: SubsampleMode = subsample
        self.subsample_bins = max(2, int(subsample_bins))
        self.preprocess: PreprocessMode = preprocess
        self.random_state = random_state
        self.estimator_kwargs = dict(estimator_kwargs or {})

        self.preprocessor: Pipeline | None = (
            _build_sklearn_preprocessor() if preprocess == "sklearn" else None
        )
        self.estimator: TabularRegressor | None = None
        self._train_X: npt.NDArray[Any] | None = None
        self._train_y: npt.NDArray[Any] | None = None
        log.info(
            "Initialized %s (device=%s, preprocess=%s)",
            self.name,
            self.device,
            self.preprocess,
        )

    @classmethod
    @abstractmethod
    def _check_import(cls) -> None:
        """Raise ImportError when the backing package is missing."""

    @abstractmethod
    def _make_estimator(self, device: str | None = None) -> TabularRegressor:
        """Construct a fresh estimator for the requested device."""

    def _prepare_training_data(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        return X, y

    def _call_predict(
        self,
        estimator: TabularRegressor,
        X: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        return cast(npt.NDArray[Any], np.asarray(estimator.predict(X)))

    def _prepare_features(
        self,
        X_df: pd.DataFrame,
        *,
        fit: bool,
    ) -> npt.NDArray[Any]:
        if self.preprocess == "sklearn":
            assert self.preprocessor is not None
            if fit:
                return cast(npt.NDArray[Any], self.preprocessor.fit_transform(X_df))
            return cast(npt.NDArray[Any], self.preprocessor.transform(X_df))
        return cast(npt.NDArray[Any], X_df.to_numpy(dtype=np.float64, copy=False))

    def _maybe_subsample(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        if self.max_train_samples is None or len(X) <= self.max_train_samples:
            return X, y

        rng = np.random.default_rng(self.random_state)
        idx = _subsample_indices(
            y,
            self.max_train_samples,
            rng,
            method=self.subsample,
            n_bins=self.subsample_bins,
        )
        log.warning(
            "Subsampled %s training data from %d to %d samples "
            "(max_train_samples=%d, subsample=%s)",
            self.name,
            len(X),
            len(idx),
            self.max_train_samples,
            self.subsample,
        )
        return X[idx], y[idx]

    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dataset: PandasDataset,
        **fit_params: Any,
    ) -> tuple[Any, dict[str, Any]]:
        del fit_params
        X_df, y_df = dataset.xy
        X = self._prepare_features(X_df, fit=True)
        y = y_df.values.ravel()
        X, y = self._maybe_subsample(X, y)
        X, y = self._prepare_training_data(X, y)

        self._train_X = X
        self._train_y = y
        self.estimator = self._make_estimator()
        assert self.estimator is not None

        if self.verbose:
            log.info(
                "%s fit | n=%d features=%d device=%s preprocess=%s",
                self.name,
                len(X),
                X.shape[1],
                self.device,
                self.preprocess,
            )
        self.estimator.fit(X, y)
        return self, {}

    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dataset: PandasDataset,
        **predict_params: Any,
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        del predict_params
        if self.estimator is None:
            raise RuntimeError(f"{self.__class__.__name__}.predict called before fit.")

        X_df, _ = dataset.xy
        X = self._prepare_features(X_df, fit=False)
        if self.verbose:
            log.info("%s predict | n=%d features=%d", self.name, len(X), X.shape[1])
        return self._predict_with_fallback(X), {}

    def _predict_in_batches(
        self,
        X: npt.NDArray[Any],
        batch_size: int,
        *,
        estimator: TabularRegressor | None = None,
    ) -> npt.NDArray[Any]:
        pred_estimator = estimator if estimator is not None else self.estimator
        assert pred_estimator is not None
        preds: list[npt.NDArray[Any]] = []
        batch_starts = range(0, len(X), batch_size)
        show_bar = self.verbose and len(X) > batch_size
        for start in tqdm(
            batch_starts,
            desc=f"{self.name} predict",
            unit="batch",
            disable=not show_bar,
            leave=False,
        ):
            batch = X[start : start + batch_size]
            preds.append(self._call_predict(pred_estimator, batch))
        return np.concatenate(preds, axis=0)

    def _predict_with_fallback(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        assert self.estimator is not None

        try:
            self._maybe_empty_cuda_cache()
            return self._call_predict(self.estimator, X)
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            log.warning(
                "%s GPU OOM during full-batch predict (n=%d); retrying in batches",
                self.name,
                len(X),
            )

        batch_size = self.predict_batch_size
        while True:
            try:
                self._maybe_empty_cuda_cache()
                return self._predict_in_batches(X, batch_size)
            except Exception as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                if batch_size == 1:
                    if not self.allow_cpu_fallback:
                        raise RuntimeError(
                            f"{self.name} GPU OOM persisted down to batch_size=1. "
                            "Reduce features/samples or set allow_cpu_fallback=true."
                        ) from exc
                    return self._predict_on_cpu(X)
                batch_size = max(1, batch_size // 2)
                log.warning(
                    "%s GPU OOM; retrying predict with batch_size=%d",
                    self.name,
                    batch_size,
                )

    def _predict_on_cpu(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self._train_X is None or self._train_y is None:
            raise RuntimeError(f"{self.name} CPU fallback requires fitted training data.")

        log.warning("%s falling back to CPU fit/predict", self.name)
        cpu_estimator = self._make_estimator(device="cpu")
        cpu_estimator.fit(self._train_X, self._train_y)
        return self._predict_in_batches(
            X,
            self.predict_batch_size,
            estimator=cpu_estimator,
        )

    @staticmethod
    def _maybe_empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def save(self, model_path: str) -> None:
        save_pickle(self, model_path, self.name)

    @classmethod
    def load(cls, model_path: str, name: str) -> TabularFoundationModel:
        return load_pickle(model_path, name)


def _predict_in_batches(
    model: TabularRegressor,
    X: npt.NDArray[Any],
    batch_size: int,
    *,
    verbose: bool = False,
) -> npt.NDArray[Any]:
    """Backward-compatible helper for tests that pass a bare estimator."""
    preds: list[npt.NDArray[Any]] = []
    batch_starts = range(0, len(X), batch_size)
    show_bar = verbose and len(X) > batch_size
    for start in tqdm(
        batch_starts,
        desc="Tabular foundation predict",
        unit="batch",
        disable=not show_bar,
        leave=False,
    ):
        batch = X[start : start + batch_size]
        preds.append(np.asarray(model.predict(batch)))
    return np.concatenate(preds, axis=0)


def _pad_train_rows(
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    min_rows: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Repeat training rows deterministically until at least ``min_rows`` samples."""
    n_cur = len(X)
    if n_cur >= min_rows:
        return X, y

    reps = min_rows // n_cur
    rem = min_rows % n_cur
    X_rep = np.repeat(X, repeats=reps, axis=0)
    y_rep = np.repeat(y, repeats=reps, axis=0)
    if rem > 0:
        idx = np.arange(rem) % n_cur
        X_rep = np.vstack([X_rep, X[idx]])
        y_rep = np.concatenate([y_rep, y[idx]])
    return X_rep, y_rep


def _with_device(
    kwargs: dict[str, Any],
    device: str | None,
    resolved_device: str,
) -> dict[str, Any]:
    out = dict(kwargs)
    if device is not None and "device" not in out:
        out["device"] = device
    elif resolved_device != "auto" and "device" not in out:
        out["device"] = resolved_device
    return out


class TabPFNModel(TabularFoundationModel):
    """TabPFN regressor for tabular CY-Bench features (PandasDataset).

    TabPFN applies its own internal preprocessing (scaling, outlier handling,
    missing values). By default ``preprocess=none`` passes features through so
    you avoid double scaling. Use ``preprocess=sklearn`` for median imputation +
    standard scaling when benchmarking against sklearn-style pipelines.
    """

    def __init__(
        self,
        name: str = "tabpfn",
        verbose: bool = False,
        framework: str | None = None,
        device: str = "auto",
        predict_batch_size: int = 256,
        allow_cpu_fallback: bool = False,
        max_train_samples: int | None = None,
        subsample: SubsampleMode = "random",
        subsample_bins: int = 10,
        preprocess: PreprocessMode = "none",
        random_state: int = 42,
        **tabpfn_kwargs: Any,
    ):
        self.tabpfn_kwargs = tabpfn_kwargs
        super().__init__(
            name=name,
            verbose=verbose,
            framework=framework,
            device=device,
            predict_batch_size=predict_batch_size,
            allow_cpu_fallback=allow_cpu_fallback,
            max_train_samples=max_train_samples,
            subsample=subsample,
            subsample_bins=subsample_bins,
            preprocess=preprocess,
            random_state=random_state,
            estimator_kwargs=tabpfn_kwargs,
        )

    @classmethod
    def _check_import(cls) -> None:
        _require_module("tabpfn", "TabPFN is not installed. Add it with: poetry add tabpfn")

    def _make_estimator(self, device: str | None = None) -> TabularRegressor:
        TabPFNRegressor = _import_symbol(
            "tabpfn",
            "TabPFNRegressor",
            "TabPFN is not installed. Add it with: poetry add tabpfn",
        )

        kwargs = dict(self.tabpfn_kwargs)
        if "random_state" not in kwargs:
            kwargs["random_state"] = self.random_state
        return cast(
            TabularRegressor,
            TabPFNRegressor(device=device or self.device, **kwargs),
        )

    @classmethod
    def load(cls, model_path: str, name: str = "tabpfn") -> TabPFNModel:
        return cast(TabPFNModel, super().load(model_path, name))


class TabICLModel(TabularFoundationModel):
    """TabICL regressor for tabular CY-Bench features (PandasDataset)."""

    def __init__(
        self,
        name: str = "tabicl",
        verbose: bool = False,
        framework: str | None = None,
        device: str = "auto",
        predict_batch_size: int = 256,
        allow_cpu_fallback: bool = False,
        max_train_samples: int | None = None,
        subsample: SubsampleMode = "random",
        subsample_bins: int = 10,
        preprocess: PreprocessMode = "none",
        random_state: int = 42,
        **tabicl_kwargs: Any,
    ):
        self.tabicl_kwargs = tabicl_kwargs
        super().__init__(
            name=name,
            verbose=verbose,
            framework=framework,
            device=device,
            predict_batch_size=predict_batch_size,
            allow_cpu_fallback=allow_cpu_fallback,
            max_train_samples=max_train_samples,
            subsample=subsample,
            subsample_bins=subsample_bins,
            preprocess=preprocess,
            random_state=random_state,
            estimator_kwargs=tabicl_kwargs,
        )

    @classmethod
    def _check_import(cls) -> None:
        _require_module("tabicl", "TabICL is not installed. Add it with: poetry add tabicl")

    def _make_estimator(self, device: str | None = None) -> TabularRegressor:
        TabICLRegressor = _import_symbol(
            "tabicl",
            "TabICLRegressor",
            "TabICL is not installed. Add it with: poetry add tabicl",
        )

        kwargs = _with_device(dict(self.tabicl_kwargs), device, self.device)
        if "random_state" not in kwargs:
            kwargs["random_state"] = self.random_state
        return cast(TabularRegressor, TabICLRegressor(**kwargs))

    @classmethod
    def load(cls, model_path: str, name: str = "tabicl") -> TabICLModel:
        return cast(TabICLModel, super().load(model_path, name))


class TabDPTModel(TabularFoundationModel):
    """TabDPT regressor for tabular CY-Bench features (PandasDataset)."""

    def __init__(
        self,
        name: str = "tabdpt",
        verbose: bool = False,
        framework: str | None = None,
        device: str = "auto",
        predict_batch_size: int = 256,
        allow_cpu_fallback: bool = False,
        max_train_samples: int | None = None,
        subsample: SubsampleMode = "random",
        subsample_bins: int = 10,
        preprocess: PreprocessMode = "none",
        random_state: int = 42,
        min_train_samples: int = 100,
        **tabdpt_kwargs: Any,
    ):
        self.tabdpt_kwargs = tabdpt_kwargs
        self.min_train_samples = max(1, int(min_train_samples))
        self._context_size: int | None = None
        super().__init__(
            name=name,
            verbose=verbose,
            framework=framework,
            device=device,
            predict_batch_size=predict_batch_size,
            allow_cpu_fallback=allow_cpu_fallback,
            max_train_samples=max_train_samples,
            subsample=subsample,
            subsample_bins=subsample_bins,
            preprocess=preprocess,
            random_state=random_state,
            estimator_kwargs=tabdpt_kwargs,
        )

    @classmethod
    def _check_import(cls) -> None:
        _require_module("tabdpt", "TabDPT is not installed. Add it with: poetry add tabdpt")

    def _make_estimator(self, device: str | None = None) -> TabularRegressor:
        TabDPTRegressor = _import_symbol(
            "tabdpt",
            "TabDPTRegressor",
            "TabDPT is not installed. Add it with: poetry add tabdpt",
        )

        kwargs = {
            key: value
            for key, value in self.tabdpt_kwargs.items()
            if key != "n_ensembles"
        }
        kwargs = _with_device(kwargs, device, self.device)
        return cast(TabularRegressor, TabDPTRegressor(**kwargs))

    def _prepare_training_data(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Pad small training folds so TabDPT can run; duplicates rows deterministically."""
        if len(X) < self.min_train_samples:
            n_orig = len(X)
            X, y = _pad_train_rows(X, y, self.min_train_samples)
            log.warning(
                "TabDPT padded training data from %d to %d rows (min_train_samples=%d)",
                n_orig,
                len(X),
                self.min_train_samples,
            )
        self._context_size = len(y)
        return X, y

    def _call_predict(
        self,
        estimator: TabularRegressor,
        X: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        if self._context_size is None:
            raise RuntimeError("TabDPTModel.predict called before fit.")
        n_ensembles = int(self.tabdpt_kwargs.get("n_ensembles", 8))
        preds = estimator.predict(
            X,
            n_ensembles=n_ensembles,
            context_size=self._context_size,
            seed=self.random_state,
        )
        return cast(npt.NDArray[Any], np.asarray(preds).reshape(-1))

    @classmethod
    def load(cls, model_path: str, name: str = "tabdpt") -> TabDPTModel:
        return cast(TabDPTModel, super().load(model_path, name))
