"""Load or retrain walk-forward models and compute SHAP-based feature importance."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from cybench.config import KEY_TARGET
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import BaseDataset, PandasDataset
from cybench.datasets.torch_dataset import TorchDataset
from cybench.models.model import BaseModel
from cybench.models.sklearn_models import RandomForest
from cybench.models.tabular_foundation_model import TabPFNModel, TabularFoundationModel
from cybench.util.config_utils import (
    adjust_model_cfg_to_dataset,
    apply_force_cpu_to_frozen_model_cfg,
    is_cybench_force_cpu,
    reload_config_with_overrides,
    remove_search_keys,
    set_seed,
)
from cybench.util.feature_selection import apply_mrmr_at_origin
from cybench.util.prediction_horizon import prediction_horizon_tag
from cybench.util.screening_artifacts import load_frozen_screening_artifacts
from cybench.util.validation import (
    default_screening_validation_cfg,
    get_screening_pre_test_years,
    get_splits,
)

log = logging.getLogger(__name__)

_NOISY_LOGGERS = (
    "numba",
    "numba.core",
    "shap",
    "shapiq",
    "matplotlib",
    "PIL",
    "filelock",
)


def configure_shap_job_logging(*, verbose: bool = False) -> None:
    """Keep Slurm stderr readable: cybench INFO, third-party WARNING+ only."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("cybench").setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "cybench" / "conf"

# Maize family representatives discussed for NL case study.
DEFAULT_MAIZE_FAMILY_MODELS: tuple[str, ...] = (
    "random_forest",
    "transformer_lf",
    "tabpfn",
)

class ModelManifestEntry(TypedDict):
    framework: str
    feature_design: bool
    needs_gpu: bool


class InterpretabilitySpec(TypedDict):
    method: str
    explainer_label: str


# Best interpretability method per model family (dashboard shows explainer_label).
INTERPRETABILITY_BY_MODEL: dict[str, InterpretabilitySpec] = {
    "random_forest": {
        "method": "tree_shap",
        "explainer_label": "TreeSHAP",
    },
    "tabpfn": {
        "method": "tabpfn_shapley",
        "explainer_label": "TabPFNShapley",
    },
    "tabicl": {
        "method": "sklearn_permutation",
        "explainer_label": "PermutationImportance",
    },
    "tabdpt": {
        "method": "sklearn_permutation",
        "explainer_label": "PermutationImportance",
    },
    "transformer_lf": {
        "method": "gradient_shap",
        "explainer_label": "GradientSHAP",
    },
}

ICL_TABULAR_MODELS = frozenset({"tabpfn", "tabicl", "tabdpt"})

# Subsample at most this many train (background) / test (eval) rows; use all rows when n < cap.
DEFAULT_MAX_BACKGROUND = 500
DEFAULT_MAX_EVAL_SAMPLES = 500
ICL_MAX_BACKGROUND = 25
ICL_MAX_EVAL_SAMPLES = 20


class FeatureRank(TypedDict):
    name: str
    mean_abs_shap: float
    rank: int


class ReproductionStats(TypedDict):
    corr_saved_preds: float | None
    max_abs_pred_diff: float | None


class WithinRunStats(TypedDict):
    repeats: float
    max_abs_pred_diff: float
    mean_abs_pred_diff: float


class PandasShapPayload(TypedDict):
    explainer: str
    features: list[FeatureRank]
    native_importance: NotRequired[list[FeatureRank]]
    model: NotRequired[str]


class TorchShapPayload(TypedDict):
    explainer: str
    features: list[FeatureRank]
    temporal_aggregation: str


class ShapOriginRecord(TypedDict):
    crop: str
    country: str
    model: str
    horizon: str
    seed: int
    train_years: list[int]
    test_years: list[int]
    n_train: int
    n_test: int
    reproduction: ReproductionStats
    explainer: str
    features: list[FeatureRank]
    temporal_aggregation: NotRequired[str]
    native_importance: NotRequired[list[FeatureRank]]


class ReproduceOriginRecord(TypedDict):
    crop: str
    country: str
    model: str
    horizon: str
    seed: int
    from_scratch: bool
    train_years: list[int]
    test_years: list[int]
    n_train: int
    n_test: int
    reproduction: ReproductionStats
    within_run: WithinRunStats | None


class ShapCaseSummary(TypedDict):
    crop: str
    country: str
    model: str
    horizon: str
    seed: int
    frozen_screening_dir: str
    walk_forward_run_dir: str | None
    n_origins: int
    origins: list[ShapOriginRecord]


# Mirrors cybench/runs/slurm/models.txt columns: framework, feature_design, needs_gpu.
MODEL_MANIFEST: dict[str, ModelManifestEntry] = {
    "random_forest": {
        "framework": "pandas",
        "feature_design": True,
        "needs_gpu": False,
    },
    "tabpfn": {
        "framework": "pandas",
        "feature_design": True,
        "needs_gpu": True,
    },
    "tabicl": {
        "framework": "pandas",
        "feature_design": True,
        "needs_gpu": True,
    },
    "tabdpt": {
        "framework": "pandas",
        "feature_design": True,
        "needs_gpu": True,
    },
    "transformer_lf": {
        "framework": "torch",
        "feature_design": False,
        "needs_gpu": True,
    },
}


@dataclass(frozen=True)
class ShapRunSpec:
    crop: str
    country: str
    model: str
    horizon: str = "eos"
    seed: int = 42
    baselines_dir: Path | None = None
    walk_forward_run_dir: Path | None = None
    screening_split_dir: Path | None = None
    test_years: tuple[int, ...] | None = None
    max_background: int = DEFAULT_MAX_BACKGROUND
    max_eval_samples: int = DEFAULT_MAX_EVAL_SAMPLES
    shapiq_budget: int = 64
    permutation_repeats: int = 5
    force_cpu: bool = False
    use_cache: bool = False


def model_run_name(model_slug: str) -> str:
    cfg_path = CONF_DIR / "model" / f"{model_slug}.yaml"
    if cfg_path.exists():
        return cast(
            str,
            OmegaConf.select(OmegaConf.load(cfg_path), "name", default=model_slug),
        )
    return model_slug


def _dataset_row_indices(dataset: BaseDataset) -> pd.DataFrame:
    if isinstance(dataset, (PandasDataset, TorchDataset)):
        return dataset.indices.copy()
    raise TypeError(
        f"verify_predictions requires PandasDataset or TorchDataset, got {type(dataset).__name__}"
    )


def _horizon_tag(horizon: str) -> str:
    return prediction_horizon_tag(horizon)


def find_screening_split_dir(
    baselines_dir: Path,
    *,
    crop: str,
    country: str,
    model_slug: str,
    horizon: str,
) -> Path:
    """Return screening split folder containing optimal_model.yaml."""
    model_name = model_run_name(model_slug)
    htag = _horizon_tag(horizon)
    pattern = f"{crop}_{country}_{model_name}_screening_{htag}_*"
    candidates = sorted(
        baselines_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No screening run matching {pattern!r} under {baselines_dir}"
        )
    for run_dir in candidates:
        hits = sorted(run_dir.glob("*/optimal_model.yaml"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        f"No optimal_model.yaml under screening runs for {crop}/{country}/{model_slug}"
    )


def find_walk_forward_run_dir(
    baselines_dir: Path,
    *,
    crop: str,
    country: str,
    model_slug: str,
    horizon: str,
) -> Path:
    model_name = model_run_name(model_slug)
    htag = _horizon_tag(horizon)
    pattern = f"{crop}_{country}_{model_name}_walk_forward_{htag}_*"
    candidates = sorted(
        baselines_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No walk-forward run matching {pattern!r} under {baselines_dir}"
        )
    return candidates[0]


def compose_dataset_overrides(
    spec: ShapRunSpec,
    *,
    framework: str,
    feature_design: bool,
) -> list[str]:
    overrides = [
        f"dataset/crop={spec.crop}",
        f"dataset.country={spec.country}",
        f"dataset.use_cache={str(spec.use_cache).lower()}",
        f"dataset.temporal.season.end_of_sequence={spec.horizon}",
        f"dataset.framework={framework}",
    ]
    if framework == "pandas" and feature_design:
        overrides.append("dataset/temporal=feature_design")
    return overrides


def build_dataset(
    spec: ShapRunSpec,
    *,
    framework: str,
    feature_design: bool,
) -> BaseDataset:
    overrides = compose_dataset_overrides(
        spec, framework=framework, feature_design=feature_design
    )
    cfg = reload_config_with_overrides(
        CONF_DIR,
        "config",
        overrides=[f"model={spec.model}", *overrides],
    )
    dataset_cfg = cast(DictConfig, cfg.dataset)
    partition_cfg = default_screening_validation_cfg()
    years = DataFactory.peek_dataset_years(dataset_cfg)
    normalizer_fit_years = get_screening_pre_test_years(
        years, seed=spec.seed, cfg=partition_cfg
    )
    dataset = DataFactory(dataset_cfg).build(normalizer_fit_years=normalizer_fit_years)
    if framework == "torch" and OmegaConf.select(cfg, "process"):
        assert isinstance(dataset, TorchDataset)
        process_cfg = cast(DictConfig, cfg.process)
        dataset.process(process_cfg)
    return dataset


def _prepare_model_cfg(
    spec: ShapRunSpec,
    *,
    frozen_dir: Path,
    dataset: BaseDataset,
    framework: str,
) -> tuple[DictConfig, DictConfig | None, int | None]:
    frozen_model_cfg, frozen_fs_cfg, frozen_E_star = load_frozen_screening_artifacts(
        frozen_dir
    )
    model_cfg = remove_search_keys(
        cast(DictConfig, OmegaConf.create(OmegaConf.to_container(frozen_model_cfg)))
    )
    if spec.force_cpu or is_cybench_force_cpu():
        model_cfg = apply_force_cpu_to_frozen_model_cfg(model_cfg)
    if framework == "torch":
        assert isinstance(dataset, TorchDataset)
        model_cfg = adjust_model_cfg_to_dataset(model_cfg, dataset)
        if frozen_E_star is not None:
            with open_dict(model_cfg):
                model_cfg.epochs = int(frozen_E_star)
                model_cfg.early_stopping_monitor = "train"
    fs_cfg = (
        cast(DictConfig, OmegaConf.create(OmegaConf.to_container(frozen_fs_cfg)))
        if frozen_fs_cfg is not None
        else None
    )
    return model_cfg, fs_cfg, frozen_E_star


def _is_torch_model(model_cfg: DictConfig) -> bool:
    framework = cast(str | None, OmegaConf.select(model_cfg, "framework"))
    return framework == "torch"


def find_saved_model_artifact(
    walk_forward_run_dir: Path | None,
    *,
    test_year: int,
    seed: int,
    model_name: str,
) -> Path | None:
    """Return a saved checkpoint under ``<run_dir>/<year>/<seed>/`` if present."""
    if walk_forward_run_dir is None:
        return None
    repetition_dir = walk_forward_run_dir / str(test_year) / str(seed)
    for suffix in (".pt", ".pkl"):
        candidate = repetition_dir / f"{model_name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_saved_walk_forward_model(
    model_cfg: DictConfig,
    artifact_path: Path,
    *,
    torch_dataset: TorchDataset | None = None,
) -> BaseModel:
    """Load a walk-forward checkpoint written by ``run_experiments`` (``store.model=true``)."""
    from cybench.models.torch.trainer import TorchTrainer

    cfg = cast(
        DictConfig,
        OmegaConf.create(OmegaConf.to_container(remove_search_keys(model_cfg))),
    )
    if _is_torch_model(cfg):
        if torch_dataset is None:
            raise ValueError("torch_dataset is required to load a torch checkpoint")
        cfg = adjust_model_cfg_to_dataset(cfg, torch_dataset)
        shell = cast(TorchTrainer, instantiate(cfg))
        return TorchTrainer.load(
            str(artifact_path),
            model=shell.model,
            optimizer=shell.optimizer,
            name=shell.name,
            device=str(shell.device),
            dataloader=shell.dataloader,
            scheduler=shell.scheduler,
        )

    loader = get_class(cast(str, cfg._target_))
    return cast(
        BaseModel,
        loader.load(str(artifact_path.parent), name=cast(str, cfg.name)),
    )


def resolve_model_at_origin(
    *,
    model_cfg: DictConfig,
    fs_cfg: DictConfig | None,
    source_dataset: BaseDataset,
    train_dataset: BaseDataset,
    test_dataset: BaseDataset,
    train_years: list[int],
    walk_forward_nn: bool,
    walk_forward_run_dir: Path | None = None,
    seed: int | None = None,
    test_years: Sequence[int] | None = None,
    load_checkpoint: bool = True,
) -> tuple[BaseModel, BaseDataset, BaseDataset]:
    """Fit at a walk-forward origin, or load a saved checkpoint when available."""
    if fs_cfg is not None:
        if not isinstance(source_dataset, PandasDataset):
            raise ValueError("mRMR feature selection requires a PandasDataset.")
        train_dataset, test_dataset, _selected = apply_mrmr_at_origin(
            source_dataset=source_dataset,
            train_years=train_years,
            fs_cfg=fs_cfg,
            train_dataset=cast(PandasDataset, train_dataset),
            eval_dataset=cast(PandasDataset, test_dataset),
        )

    artifact: Path | None = None
    if (
        load_checkpoint
        and walk_forward_run_dir is not None
        and seed is not None
        and test_years is not None
    ):
        artifact = find_saved_model_artifact(
            walk_forward_run_dir,
            test_year=int(test_years[0]),
            seed=int(seed),
            model_name=cast(str, OmegaConf.select(model_cfg, "name")),
        )
    if artifact is not None:
        log.info("Loading saved walk-forward model from %s", artifact)
        torch_ds = train_dataset if isinstance(train_dataset, TorchDataset) else None
        model = load_saved_walk_forward_model(
            model_cfg, artifact, torch_dataset=torch_ds
        )
        return model, train_dataset, test_dataset

    model_seed = int(cast(int, OmegaConf.select(model_cfg, "seed") or 42))
    set_seed(model_seed)
    model = cast(BaseModel, instantiate(model_cfg))
    if walk_forward_nn:
        _ = model.fit(train_dataset, early_stopping_monitor="train")
    else:
        _ = model.fit(train_dataset)
    return model, train_dataset, test_dataset


def fit_at_origin(
    *,
    model_cfg: DictConfig,
    fs_cfg: DictConfig | None,
    source_dataset: BaseDataset,
    train_dataset: BaseDataset,
    test_dataset: BaseDataset,
    train_years: list[int],
    walk_forward_nn: bool,
    walk_forward_run_dir: Path | None = None,
    seed: int | None = None,
    test_years: Sequence[int] | None = None,
    load_checkpoint: bool = True,
) -> tuple[BaseModel, BaseDataset, BaseDataset]:
    return resolve_model_at_origin(
        model_cfg=model_cfg,
        fs_cfg=fs_cfg,
        source_dataset=source_dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_years=train_years,
        walk_forward_nn=walk_forward_nn,
        walk_forward_run_dir=walk_forward_run_dir,
        seed=seed,
        test_years=test_years,
        load_checkpoint=load_checkpoint,
    )


def verify_predictions(
    *,
    walk_forward_run_dir: Path,
    test_year: int,
    seed: int,
    new_preds: npt.NDArray[np.float64],
    test_dataset: BaseDataset | None = None,
) -> ReproductionStats:
    pred_path = walk_forward_run_dir / str(test_year) / str(seed) / "test_preds.csv"
    if not pred_path.exists():
        return ReproductionStats(corr_saved_preds=None, max_abs_pred_diff=None)
    saved_df = pd.read_csv(pred_path)
    new_preds = np.asarray(new_preds, dtype=np.float64)
    if len(saved_df) != len(new_preds):
        return ReproductionStats(corr_saved_preds=None, max_abs_pred_diff=None)

    saved: npt.NDArray[np.float64]
    # Align on adm_id/year: TorchDataset row order is not stable across builds.
    if test_dataset is not None:
        indices_df = _dataset_row_indices(test_dataset).reset_index(drop=True)
        new_df = pd.concat(
            [indices_df, pd.DataFrame({"preds": new_preds})],
            axis=1,
        )
        key_cols = [c for c in ("adm_id", "year") if c in saved_df.columns and c in new_df.columns]
        if key_cols:
            merged = saved_df.merge(new_df, on=key_cols, suffixes=("_saved", "_new"))
            if len(merged) == len(saved_df):
                saved = cast(
                    npt.NDArray[np.float64],
                    merged["preds_saved"].to_numpy(dtype=np.float64),
                )
                new_preds = cast(
                    npt.NDArray[np.float64],
                    merged["preds_new"].to_numpy(dtype=np.float64),
                )
            else:
                saved = cast(
                    npt.NDArray[np.float64],
                    saved_df["preds"].to_numpy(dtype=np.float64),
                )
        else:
            saved = cast(
                npt.NDArray[np.float64],
                saved_df["preds"].to_numpy(dtype=np.float64),
            )
    else:
        saved = cast(
            npt.NDArray[np.float64],
            saved_df["preds"].to_numpy(dtype=np.float64),
        )

    corr_matrix = np.corrcoef(saved, new_preds)
    corr = float(cast(float, corr_matrix[0, 1])) if len(saved) > 1 else 1.0
    return ReproductionStats(
        corr_saved_preds=corr,
        max_abs_pred_diff=float(cast(float, np.max(np.abs(saved - new_preds)))),
    )


def interpretability_for_model(model_slug: str) -> InterpretabilitySpec:
    """Return the configured interpretability method for a model slug."""
    if model_slug in INTERPRETABILITY_BY_MODEL:
        return INTERPRETABILITY_BY_MODEL[model_slug]
    return InterpretabilitySpec(
        method="permutation_shap",
        explainer_label="PermutationSHAP",
    )


# TabPFN (and similar ICL tabular models) use family-specific explainers below.


def resolve_shap_sample_limits(
    model_slug: str,
    *,
    max_background: int,
    max_eval_samples: int,
    n_train: int | None = None,
    n_test: int | None = None,
) -> tuple[int, int]:
    """``min(dataset size, cap)``; tighter caps for expensive ICL explainers."""
    bg = min(max_background, n_train) if n_train is not None else max_background
    ev = min(max_eval_samples, n_test) if n_test is not None else max_eval_samples
    if model_slug in ICL_TABULAR_MODELS:
        return min(bg, ICL_MAX_BACKGROUND), min(ev, ICL_MAX_EVAL_SAMPLES)
    return bg, ev


def _require_tabular_train_arrays(
    model: TabularFoundationModel,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if model._train_X is None or model._train_y is None:
        raise RuntimeError(
            f"{model.__class__.__name__} must be fitted before computing importance."
        )
    return (
        np.asarray(model._train_X, dtype=np.float64),
        np.asarray(model._train_y, dtype=np.float64).reshape(-1),
    )


def _aggregate_shapiq_first_order(
    explanations: Sequence[object],
    *,
    n_features: int,
) -> npt.NDArray[np.float64]:
    if not explanations:
        raise ValueError("TabPFN Shapley explainer returned no explanations.")
    stacked = np.zeros((len(explanations), n_features), dtype=np.float64)
    for row, explanation in enumerate(explanations):
        order1 = np.asarray(
            cast(Any, explanation).get_n_order_values(1),
            dtype=np.float64,
        ).reshape(-1)
        if order1.shape[0] != n_features:
            raise ValueError(
                f"Shapley vector length {order1.shape[0]} does not match "
                f"{n_features} features."
            )
        stacked[row] = np.abs(order1)
    return np.mean(stacked, axis=0)


def _compute_tabpfn_shapley_importance(
    model: TabPFNModel,
    *,
    X_eval: pd.DataFrame,
    feature_names: list[str],
    shapiq_budget: int,
    seed: int,
) -> list[FeatureRank]:
    from shapiq.explainer import TabPFNExplainer  # pyright: ignore[reportMissingImports]

    if model.estimator is None:
        raise RuntimeError("TabPFNModel.predict called before fit.")
    X_train, y_train = _require_tabular_train_arrays(model)
    X_eval_np = np.asarray(
        model._prepare_features(X_eval, fit=False),
        dtype=np.float64,
    )
    explainer = TabPFNExplainer(
        cast(Any, model.estimator),
        X_train,
        y_train,
        verbose=False,
    )
    explanations = explainer.explain_X(
        X_eval_np,
        budget=int(shapiq_budget),
        random_state=int(seed),
        verbose=False,
    )
    mean_abs = _aggregate_shapiq_first_order(
        explanations,
        n_features=len(feature_names),
    )
    return _rank_features(feature_names, mean_abs)


def _compute_sklearn_permutation_importance(
    model: TabularFoundationModel,
    *,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    feature_names: list[str],
    seed: int,
    n_repeats: int,
) -> list[FeatureRank]:
    from sklearn.inspection import permutation_importance

    if model.estimator is None:
        raise RuntimeError(
            f"{model.__class__.__name__}.predict called before fit."
        )
    X = np.asarray(model._prepare_features(X_eval, fit=False), dtype=np.float64)
    y = np.asarray(y_eval.values.ravel(), dtype=np.float64)
    result = permutation_importance(
        model.estimator,
        X,
        y,
        n_repeats=int(n_repeats),
        random_state=int(seed),
        n_jobs=1,
    )
    importances = np.asarray(cast(Any, result)["importances_mean"], dtype=np.float64)
    return _rank_features(feature_names, importances)


def _subsample_indices(n: int, k: int, rng: np.random.Generator) -> npt.NDArray[np.int_]:
    k = min(int(k), int(n))
    if k <= 0:
        raise ValueError("Need at least one sample for SHAP.")
    if k >= n:
        return np.arange(n, dtype=int)
    return np.sort(rng.choice(n, size=k, replace=False))


def _mean_abs_feature_importance(
    values: npt.NDArray[np.float64] | np.ndarray,
    *,
    n_features: int,
) -> npt.NDArray[np.float64]:
    """Collapse sample/time axes and return one mean |SHAP| per feature."""
    arr = np.abs(np.asarray(values, dtype=np.float64))
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("SHAP values must not be scalar.")
    if arr.ndim == 1:
        if arr.shape[0] != n_features:
            raise ValueError(
                f"SHAP vector length {arr.shape[0]} does not match {n_features} features."
            )
        return arr.astype(np.float64, copy=False)
    if arr.shape[-1] != n_features:
        raise ValueError(
            f"SHAP trailing dimension {arr.shape[-1]} does not match {n_features} features "
            f"(full shape {arr.shape})."
        )
    lead_axes = tuple(range(arr.ndim - 1))
    return np.mean(arr, axis=lead_axes)


def _rank_features(
    names: Sequence[str],
    mean_abs: npt.NDArray[np.float64],
) -> list[FeatureRank]:
    values = np.asarray(mean_abs, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(names):
        raise ValueError(
            f"SHAP importance length {values.shape[0]} does not match "
            f"{len(names)} feature names."
        )
    order = np.argsort(-values)
    rows: list[FeatureRank] = []
    for rank, index in enumerate(order, start=1):
        idx = int(index)
        val = float(values[idx])
        if not np.isfinite(val) or val <= 0:
            continue
        rows.append(
            FeatureRank(
                name=str(names[idx]),
                mean_abs_shap=round(val, 8),
                rank=rank,
            )
        )
    return rows


def _native_rf_importance(
    model: RandomForest, feature_names: list[str]
) -> list[FeatureRank]:
    pipe = model.model
    estimator = cast(RandomForestRegressor, pipe.named_steps["estimator"])
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    return _rank_features(feature_names, importances)


def compute_shap_pandas(
    model: BaseModel,
    *,
    train_dataset: PandasDataset,
    test_dataset: PandasDataset,
    model_slug: str,
    max_background: int,
    max_eval_samples: int,
    shapiq_budget: int,
    permutation_repeats: int,
    seed: int,
) -> PandasShapPayload:
    import shap  # pyright: ignore[reportMissingImports]
    from shap.maskers import Independent  # pyright: ignore[reportMissingImports]

    X_train, _ = train_dataset.xy
    X_test, y_test = test_dataset.xy
    feature_names: list[str] = [str(c) for c in X_train.columns]
    rng = np.random.default_rng(seed)
    bg_idx = _subsample_indices(len(X_train), max_background, rng)
    eval_idx = _subsample_indices(len(X_test), max_eval_samples, rng)
    X_bg = X_train.iloc[bg_idx]
    X_eval = X_test.iloc[eval_idx]
    interpretability = interpretability_for_model(model_slug)
    method = interpretability["method"]
    explainer_label = interpretability["explainer_label"]

    if method == "tree_shap":
        if not isinstance(model, RandomForest):
            raise TypeError(
                f"tree_shap interpretability requires RandomForest, got {type(model).__name__}"
            )
        pipe = model.model
        imputer = cast(SimpleImputer, pipe.named_steps["imputer"])
        scaler = cast(StandardScaler, pipe.named_steps["scaler"])
        estimator = cast(RandomForestRegressor, pipe.named_steps["estimator"])
        X_eval_t = scaler.transform(imputer.transform(X_eval))
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_eval_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = _mean_abs_feature_importance(
            np.asarray(shap_values, dtype=float),
            n_features=len(feature_names),
        )
        return PandasShapPayload(
            explainer=explainer_label,
            features=_rank_features(feature_names, mean_abs),
            native_importance=_native_rf_importance(model, feature_names),
        )

    if method == "tabpfn_shapley":
        if not isinstance(model, TabPFNModel):
            raise TypeError(
                f"tabpfn_shapley requires TabPFNModel, got {type(model).__name__}"
            )
        return PandasShapPayload(
            explainer=explainer_label,
            features=_compute_tabpfn_shapley_importance(
                model,
                X_eval=X_eval,
                feature_names=feature_names,
                shapiq_budget=shapiq_budget,
                seed=seed,
            ),
            model=model_slug,
        )

    if method == "sklearn_permutation":
        if not isinstance(model, TabularFoundationModel):
            raise TypeError(
                "sklearn_permutation requires TabularFoundationModel, "
                f"got {type(model).__name__}"
            )
        y_eval = y_test.iloc[eval_idx][KEY_TARGET]
        return PandasShapPayload(
            explainer=explainer_label,
            features=_compute_sklearn_permutation_importance(
                model,
                X_eval=X_eval,
                y_eval=y_eval,
                feature_names=feature_names,
                seed=seed,
                n_repeats=permutation_repeats,
            ),
            model=model_slug,
        )

    def predict_matrix(x_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        frame = pd.DataFrame(x_matrix, columns=pd.Index(feature_names))
        y_dummy = pd.DataFrame({KEY_TARGET: np.zeros(len(frame), dtype=np.float32)})
        ds = PandasDataset(
            cfg=test_dataset.cfg,
            y=y_dummy,
            x=frame,
            normalizer=test_dataset.normalizer,
        )
        preds, _ = model.predict(ds)
        return np.asarray(preds, dtype=float).reshape(-1)

    background = X_bg.to_numpy(dtype=float)
    eval_matrix = X_eval.to_numpy(dtype=float)
    masker = Independent(background)
    explainer = shap.Explainer(predict_matrix, masker, algorithm="permutation")
    explanation = explainer(eval_matrix)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    mean_abs = _mean_abs_feature_importance(values, n_features=values.shape[-1])
    return PandasShapPayload(
        explainer=explainer_label,
        features=_rank_features(feature_names, mean_abs),
        model=model_slug,
    )


def compute_shap_torch(
    model: BaseModel,
    *,
    train_dataset: TorchDataset,
    test_dataset: TorchDataset,
    max_background: int,
    max_eval_samples: int,
    seed: int,
) -> TorchShapPayload:
    import shap  # pyright: ignore[reportMissingImports]
    import torch
    import torch.nn as nn
    from cybench.models.torch.trainer import TorchTrainer

    if not isinstance(model, TorchTrainer):
        raise TypeError(
            f"compute_shap_torch expects TorchTrainer, got {type(model).__name__}"
        )
    device = model.device
    rng = np.random.default_rng(seed)
    bg_idx = _subsample_indices(len(train_dataset), max_background, rng)
    eval_idx = _subsample_indices(len(test_dataset), max_eval_samples, rng)

    def stack_batch(ds: TorchDataset, indices: Iterable[int]) -> tuple[torch.Tensor, ...]:
        ys, ctxs, tss, doys = zip(*(ds[int(i)] for i in indices))
        return (
            torch.stack(ys).to(device),
            torch.stack(ctxs).to(device),
            torch.stack(tss).to(device),
            torch.stack(doys).to(device),
        )

    _, bg_ctx, bg_ts, bg_doy = stack_batch(train_dataset, bg_idx)
    _, eval_ctx, eval_ts, eval_doy = stack_batch(test_dataset, eval_idx)

    class _TorchPredictor(nn.Module):
        core: nn.Module

        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.core = core

        def forward(
            self,
            x_ctx: torch.Tensor,
            x_ts: torch.Tensor,
            doy: torch.Tensor,
        ) -> torch.Tensor:
            pred = cast(torch.Tensor, self.core(x_ctx, x_ts, doy))
            if pred.ndim > 1:
                pred = pred.squeeze(-1)
            return pred.unsqueeze(-1)

    predictor = _TorchPredictor(model.model).eval()
    background = [bg_ctx, bg_ts, bg_doy]
    explainer = shap.GradientExplainer(predictor, background)
    shap_values = explainer.shap_values([eval_ctx, eval_ts, eval_doy])
    if not isinstance(shap_values, list) or len(shap_values) != 3:
        raise RuntimeError("Expected SHAP values for context, temporal, and doy inputs.")

    ctx_names = list(train_dataset.x_context_columns)
    ts_names = list(train_dataset.x_ts_columns)
    ctx_mean = _mean_abs_feature_importance(
        np.asarray(shap_values[0], dtype=float),
        n_features=len(ctx_names),
    )
    ts_mean = _mean_abs_feature_importance(
        np.asarray(shap_values[1], dtype=float),
        n_features=len(ts_names),
    )

    feature_rows = _rank_features(
        [f"ctx:{name}" for name in ctx_names],
        ctx_mean,
    )
    feature_rows.extend(
        _rank_features(
            [f"ts:{name}" for name in ts_names],
            ts_mean,
        )
    )
    feature_rows.sort(key=lambda row: row["rank"])
    for i, row in enumerate(feature_rows, start=1):
        row["rank"] = i
    interpretability = interpretability_for_model("transformer_lf")
    return TorchShapPayload(
        explainer=interpretability["explainer_label"],
        features=feature_rows,
        temporal_aggregation="mean_abs_over_time",
    )


def _fit_and_predict_at_origin(
    spec: ShapRunSpec,
    *,
    train_years: list[int],
    test_years: list[int],
    frozen_dir: Path,
    walk_forward_run_dir: Path | None,
    from_scratch: bool,
) -> tuple[npt.NDArray[np.float64], BaseDataset, int, int]:
    meta = MODEL_MANIFEST[spec.model]
    framework = meta["framework"]
    feature_design = meta["feature_design"]
    set_seed(spec.seed)

    dataset = build_dataset(
        spec, framework=framework, feature_design=feature_design
    )
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    model_cfg, fs_cfg, _E_star = _prepare_model_cfg(
        spec,
        frozen_dir=frozen_dir,
        dataset=dataset,
        framework=framework,
    )
    fit_run_dir = None if from_scratch else walk_forward_run_dir
    model, train_dataset, test_dataset = fit_at_origin(
        model_cfg=model_cfg,
        fs_cfg=fs_cfg,
        source_dataset=dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_years=train_years,
        walk_forward_nn=_is_torch_model(model_cfg),
        walk_forward_run_dir=fit_run_dir,
        seed=spec.seed,
        test_years=test_years,
        load_checkpoint=not from_scratch,
    )
    preds, _ = model.predict(test_dataset)
    return (
        np.asarray(preds, dtype=np.float64),
        test_dataset,
        len(train_dataset),
        len(test_dataset),
    )


def reproduce_walk_forward_origin(
    spec: ShapRunSpec,
    *,
    train_years: list[int],
    test_years: list[int],
    frozen_dir: Path,
    walk_forward_run_dir: Path | None,
    from_scratch: bool = False,
    within_run_repeats: int = 1,
) -> ReproduceOriginRecord:
    """Refit one walk-forward origin and compare predictions to saved test_preds.csv."""
    if within_run_repeats < 1:
        raise ValueError("within_run_repeats must be >= 1")

    preds, test_dataset, n_train, n_test = _fit_and_predict_at_origin(
        spec,
        train_years=train_years,
        test_years=test_years,
        frozen_dir=frozen_dir,
        walk_forward_run_dir=walk_forward_run_dir,
        from_scratch=from_scratch,
    )
    within_run: WithinRunStats | None = None
    if within_run_repeats > 1:
        repeat_preds = [preds]
        for _ in range(within_run_repeats - 1):
            repeat_preds.append(
                _fit_and_predict_at_origin(
                    spec,
                    train_years=train_years,
                    test_years=test_years,
                    frozen_dir=frozen_dir,
                    walk_forward_run_dir=walk_forward_run_dir,
                    from_scratch=from_scratch,
                )[0]
            )
        stacked = np.stack(repeat_preds, axis=0)
        ref = cast(npt.NDArray[np.float64], stacked[0])
        diffs = cast(
            npt.NDArray[np.float64],
            np.max(np.abs(stacked[1:] - ref), axis=1),
        )
        within_run = WithinRunStats(
            repeats=float(within_run_repeats),
            max_abs_pred_diff=float(np.max(diffs)),
            mean_abs_pred_diff=float(np.mean(diffs)),
        )

    reproduction: ReproductionStats = (
        verify_predictions(
            walk_forward_run_dir=walk_forward_run_dir,
            test_year=int(test_years[0]),
            seed=spec.seed,
            new_preds=preds,
            test_dataset=test_dataset,
        )
        if walk_forward_run_dir is not None
        else ReproductionStats(corr_saved_preds=None, max_abs_pred_diff=None)
    )
    return ReproduceOriginRecord(
        crop=spec.crop,
        country=spec.country,
        model=spec.model,
        horizon=spec.horizon,
        seed=spec.seed,
        from_scratch=from_scratch,
        train_years=train_years,
        test_years=test_years,
        n_train=n_train,
        n_test=n_test,
        reproduction=reproduction,
        within_run=within_run,
    )


def run_origin_shap(
    spec: ShapRunSpec,
    *,
    train_years: list[int],
    test_years: list[int],
    frozen_dir: Path,
    walk_forward_run_dir: Path | None,
) -> ShapOriginRecord:
    meta = MODEL_MANIFEST[spec.model]
    framework = meta["framework"]
    feature_design = meta["feature_design"]
    set_seed(spec.seed)

    dataset = build_dataset(
        spec, framework=framework, feature_design=feature_design
    )
    train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
    model_cfg, fs_cfg, _E_star = _prepare_model_cfg(
        spec,
        frozen_dir=frozen_dir,
        dataset=dataset,
        framework=framework,
    )
    model, train_dataset, test_dataset = fit_at_origin(
        model_cfg=model_cfg,
        fs_cfg=fs_cfg,
        source_dataset=dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_years=train_years,
        walk_forward_nn=_is_torch_model(model_cfg),
        walk_forward_run_dir=walk_forward_run_dir,
        seed=spec.seed,
        test_years=test_years,
    )
    preds, _ = model.predict(test_dataset)
    reproduction: ReproductionStats = (
        verify_predictions(
            walk_forward_run_dir=walk_forward_run_dir,
            test_year=int(test_years[0]),
            seed=spec.seed,
            new_preds=np.asarray(preds, dtype=np.float64),
            test_dataset=test_dataset,
        )
        if walk_forward_run_dir is not None
        else ReproductionStats(corr_saved_preds=None, max_abs_pred_diff=None)
    )

    max_background, max_eval_samples = resolve_shap_sample_limits(
        spec.model,
        max_background=spec.max_background,
        max_eval_samples=spec.max_eval_samples,
        n_train=len(train_dataset),
        n_test=len(test_dataset),
    )
    log.info(
        "SHAP sample limits | %s/%s | background=%d/%d | eval=%d/%d",
        spec.crop,
        spec.country,
        max_background,
        len(train_dataset),
        max_eval_samples,
        len(test_dataset),
    )

    shap_payload: PandasShapPayload | TorchShapPayload
    if isinstance(train_dataset, PandasDataset) and isinstance(test_dataset, PandasDataset):
        shap_payload = compute_shap_pandas(
            model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_slug=spec.model,
            max_background=max_background,
            max_eval_samples=max_eval_samples,
            shapiq_budget=spec.shapiq_budget,
            permutation_repeats=spec.permutation_repeats,
            seed=spec.seed,
        )
    elif isinstance(train_dataset, TorchDataset) and isinstance(test_dataset, TorchDataset):
        shap_payload = compute_shap_torch(
            model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            max_background=max_background,
            max_eval_samples=max_eval_samples,
            seed=spec.seed,
        )
    else:
        raise TypeError("Train/test dataset types do not match after feature selection.")

    record: ShapOriginRecord = {
        "crop": spec.crop,
        "country": spec.country,
        "model": spec.model,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "train_years": train_years,
        "test_years": test_years,
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "reproduction": reproduction,
        "explainer": shap_payload["explainer"],
        "features": shap_payload["features"],
    }
    if "native_importance" in shap_payload:
        record["native_importance"] = shap_payload["native_importance"]
    if "temporal_aggregation" in shap_payload:
        record["temporal_aggregation"] = shap_payload["temporal_aggregation"]
    return record


def iter_walk_forward_origins(
    dataset_years: set[int],
    *,
    seed: int,
    only_years: Sequence[int] | None = None,
) -> Iterable[tuple[list[int], list[int]]]:
    cfg = OmegaConf.create({"name": "walk_forward", "test_years": "5-last"})
    for train_years, test_years in get_splits(
        cfg=cfg, which="test", dataset_years=dataset_years, seed=seed
    ):
        if only_years is not None and cast(int, test_years[0]) not in only_years:
            continue
        yield list(train_years), list(test_years)


def aggregate_feature_importance(records: Sequence[ShapOriginRecord]) -> pd.DataFrame:
    """Median mean_abs_shap per feature across walk-forward origins."""
    rows: list[dict[str, object]] = []
    for record in records:
        origin = int(record["test_years"][0])
        for feat in record.get("features", []):
            rows.append(
                {
                    "model": record["model"],
                    "origin": origin,
                    "feature": feat["name"],
                    "mean_abs_shap": feat["mean_abs_shap"],
                    "rank": feat["rank"],
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    agg = cast(
        pd.DataFrame,
        frame.groupby(["model", "feature"], as_index=False).agg(
            median_mean_abs_shap=("mean_abs_shap", "median"),
            mean_rank=("rank", "mean"),
            n_origins=("origin", "nunique"),
        ),
    ).sort_values(["model", "median_mean_abs_shap"], ascending=[True, False])
    agg["aggregate_rank"] = (
        agg.groupby("model")["median_mean_abs_shap"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    return agg


def discover_origin_record_paths(model_dir: Path) -> list[Path]:
    """Return ``origin_<year>/shap_importance.yaml`` paths sorted by year."""
    paths = sorted(model_dir.glob("origin_*/shap_importance.yaml"))
    return sorted(paths, key=lambda path: int(path.parent.name.removeprefix("origin_")))


def load_origin_record(path: Path) -> ShapOriginRecord:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return cast(ShapOriginRecord, payload)


def gather_origin_records(model_dir: Path) -> list[ShapOriginRecord]:
    """Load all per-origin SHAP records written under *model_dir*."""
    records: list[ShapOriginRecord] = []
    for path in discover_origin_record_paths(model_dir):
        records.append(load_origin_record(path))
    return records


def write_model_shap_summary(
    *,
    model_dir: Path,
    records: Sequence[ShapOriginRecord],
    frozen_screening_dir: str,
    walk_forward_run_dir: str | None,
) -> ShapCaseSummary:
    """Write ``shap_summary.yaml`` and ``shap_aggregate.csv`` for one model."""
    if not records:
        raise ValueError(f"No origin records to summarize under {model_dir}")
    first = records[0]
    summary = ShapCaseSummary(
        crop=str(first["crop"]),
        country=str(first["country"]),
        model=str(first["model"]),
        horizon=str(first["horizon"]),
        seed=int(first["seed"]),
        frozen_screening_dir=frozen_screening_dir,
        walk_forward_run_dir=walk_forward_run_dir,
        n_origins=len(records),
        origins=list(records),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(dict(summary)), model_dir / "shap_summary.yaml")
    agg = aggregate_feature_importance(records)
    if not agg.empty:
        agg.to_csv(model_dir / "shap_aggregate.csv", index=False)
    return summary


def collect_shap_case_dir(
    model_dir: Path,
    *,
    baselines_dir: Path | None = None,
) -> ShapCaseSummary:
    """Rebuild model-level SHAP summary/aggregate from ``origin_*/`` artifacts."""
    records = gather_origin_records(model_dir)
    if not records:
        raise FileNotFoundError(f"No origin_*/shap_importance.yaml under {model_dir}")

    frozen_dir = ""
    walk_forward_run_dir: str | None = None
    if baselines_dir is not None:
        spec = ShapRunSpec(
            crop=str(records[0]["crop"]),
            country=str(records[0]["country"]),
            model=str(records[0]["model"]),
            horizon=str(records[0]["horizon"]),
            seed=int(records[0]["seed"]),
            baselines_dir=baselines_dir,
        )
        frozen_path, wf_path, _ = resolve_shap_paths(spec)
        frozen_dir = str(frozen_path)
        walk_forward_run_dir = str(wf_path) if wf_path is not None else None

    return write_model_shap_summary(
        model_dir=model_dir,
        records=records,
        frozen_screening_dir=frozen_dir,
        walk_forward_run_dir=walk_forward_run_dir,
    )


_SHAP_OUTPUT_DIR_RE = re.compile(
    r"^(?P<crop>maize|wheat)_(?P<country>[A-Za-z]{2})_(?P<hz>eos|mid|qtr|early)$"
)


@dataclass(frozen=True)
class ShapCollectCase:
    """One crop×country SHAP output folder with at least one origin artifact."""

    crop: str
    country: str
    horizon_tag: str
    output_dir: Path
    models: tuple[str, ...]
    n_origins: int


def discover_shap_collect_cases(
    shap_root: Path,
    *,
    crops: Sequence[str] | None = None,
    countries: Sequence[str] | None = None,
    horizon_tag: str | None = None,
) -> list[ShapCollectCase]:
    """Discover ``shap_importance/{crop}_{CC}_{hz}/`` dirs with origin YAML artifacts."""
    if not shap_root.is_dir():
        raise FileNotFoundError(f"SHAP root not found: {shap_root}")

    crop_filter = {c.casefold() for c in crops} if crops else None
    country_filter = {c.upper() for c in countries} if countries else None
    cases: list[ShapCollectCase] = []

    for output_dir in sorted(shap_root.iterdir()):
        if not output_dir.is_dir():
            continue
        match = _SHAP_OUTPUT_DIR_RE.match(output_dir.name)
        if match is None:
            continue
        crop = match.group("crop")
        country = match.group("country").upper()
        hz = match.group("hz")
        if crop_filter is not None and crop.casefold() not in crop_filter:
            continue
        if country_filter is not None and country not in country_filter:
            continue
        if horizon_tag is not None and hz != horizon_tag:
            continue

        case_dir = output_dir / f"{crop}_{country}"
        if not case_dir.is_dir():
            continue

        model_names: list[str] = []
        origin_count = 0
        for model_dir in sorted(case_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            paths = discover_origin_record_paths(model_dir)
            if not paths:
                continue
            model_names.append(model_dir.name)
            origin_count = max(origin_count, len(paths))

        if not model_names:
            continue

        cases.append(
            ShapCollectCase(
                crop=crop,
                country=country,
                horizon_tag=hz,
                output_dir=output_dir,
                models=tuple(model_names),
                n_origins=origin_count,
            )
        )
    return cases


def collect_shap_output_dir(
    output_dir: Path,
    *,
    crop: str,
    country: str,
    baselines_dir: Path | None = None,
    models: Sequence[str] | None = None,
) -> list[ShapCaseSummary]:
    """Collect per-model summaries and write cross-model aggregate CSV."""
    case_dir = output_dir / f"{crop}_{country}"
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    model_dirs = sorted(
        path for path in case_dir.iterdir() if path.is_dir() and path.name != "__pycache__"
    )
    if models is not None:
        allowed = set(models)
        model_dirs = [path for path in model_dirs if path.name in allowed]

    summaries: list[ShapCaseSummary] = []
    all_records: list[ShapOriginRecord] = []
    for model_dir in model_dirs:
        if not discover_origin_record_paths(model_dir):
            continue
        summary = collect_shap_case_dir(model_dir, baselines_dir=baselines_dir)
        summaries.append(summary)
        all_records.extend(summary["origins"])

    agg = aggregate_feature_importance(all_records)
    if not agg.empty:
        agg.to_csv(case_dir / "shap_aggregate_all_models.csv", index=False)
    return summaries


def resolve_shap_paths(spec: ShapRunSpec) -> tuple[Path, Path | None, Path]:
    baselines_dir = spec.baselines_dir
    if baselines_dir is None:
        raise ValueError("baselines_dir is required when screening/walk-forward dirs are not set.")
    frozen_dir = spec.screening_split_dir or find_screening_split_dir(
        baselines_dir,
        crop=spec.crop,
        country=spec.country,
        model_slug=spec.model,
        horizon=spec.horizon,
    )
    walk_forward_run_dir = spec.walk_forward_run_dir
    if walk_forward_run_dir is None:
        try:
            walk_forward_run_dir = find_walk_forward_run_dir(
                baselines_dir,
                crop=spec.crop,
                country=spec.country,
                model_slug=spec.model,
                horizon=spec.horizon,
            )
        except FileNotFoundError:
            walk_forward_run_dir = None
    return frozen_dir, walk_forward_run_dir, baselines_dir


def run_shap_case(
    spec: ShapRunSpec,
    *,
    output_dir: Path,
    write_summary: bool = True,
) -> ShapCaseSummary:
    if spec.model not in MODEL_MANIFEST:
        raise ValueError(
            f"Unsupported model {spec.model!r}. Supported: {sorted(MODEL_MANIFEST)}"
        )
    frozen_dir, walk_forward_run_dir, _baselines_dir = resolve_shap_paths(spec)
    meta = MODEL_MANIFEST[spec.model]
    dataset = build_dataset(
        spec,
        framework=meta["framework"],
        feature_design=meta["feature_design"],
    )
    records: list[ShapOriginRecord] = []
    for train_years, test_years in iter_walk_forward_origins(
        dataset.years,
        seed=spec.seed,
        only_years=spec.test_years,
    ):
        log.info(
            "SHAP | %s/%s | %s | origin=%s | train=%d | test=%d",
            spec.crop,
            spec.country,
            spec.model,
            test_years[0],
            len(train_years),
            len(test_years),
        )
        record = run_origin_shap(
            spec,
            train_years=train_years,
            test_years=test_years,
            frozen_dir=frozen_dir,
            walk_forward_run_dir=walk_forward_run_dir,
        )
        records.append(record)
        origin_dir = output_dir / f"origin_{test_years[0]}"
        origin_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create(dict(record)),
            origin_dir / "shap_importance.yaml",
        )

    summary = ShapCaseSummary(
        crop=spec.crop,
        country=spec.country,
        model=spec.model,
        horizon=spec.horizon,
        seed=spec.seed,
        frozen_screening_dir=str(frozen_dir),
        walk_forward_run_dir=str(walk_forward_run_dir) if walk_forward_run_dir else None,
        n_origins=len(records),
        origins=records,
    )
    if write_summary:
        OmegaConf.save(
            OmegaConf.create(dict(summary)),
            output_dir / "shap_summary.yaml",
        )
        agg = aggregate_feature_importance(records)
        if not agg.empty:
            agg.to_csv(output_dir / "shap_aggregate.csv", index=False)
    return summary
