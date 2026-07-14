"""Load or retrain walk-forward models and compute SHAP-based feature importance."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast

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
    max_background: int = 50
    max_eval_samples: int = 80
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
    seed: int,
) -> PandasShapPayload:
    import shap  # pyright: ignore[reportMissingImports]
    from shap.maskers import Independent  # pyright: ignore[reportMissingImports]

    X_train, _ = train_dataset.xy
    X_test, _ = test_dataset.xy
    feature_names: list[str] = [str(c) for c in X_train.columns]
    rng = np.random.default_rng(seed)
    bg_idx = _subsample_indices(len(X_train), max_background, rng)
    eval_idx = _subsample_indices(len(X_test), max_eval_samples, rng)
    X_bg = X_train.iloc[bg_idx]
    X_eval = X_test.iloc[eval_idx]

    if isinstance(model, RandomForest):
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
            explainer="TreeExplainer",
            features=_rank_features(feature_names, mean_abs),
            native_importance=_native_rf_importance(model, feature_names),
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
        explainer="PermutationExplainer",
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
    return TorchShapPayload(
        explainer="GradientExplainer",
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

    shap_payload: PandasShapPayload | TorchShapPayload
    if isinstance(train_dataset, PandasDataset) and isinstance(test_dataset, PandasDataset):
        shap_payload = compute_shap_pandas(
            model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_slug=spec.model,
            max_background=spec.max_background,
            max_eval_samples=spec.max_eval_samples,
            seed=spec.seed,
        )
    elif isinstance(train_dataset, TorchDataset) and isinstance(test_dataset, TorchDataset):
        shap_payload = compute_shap_torch(
            model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            max_background=spec.max_background,
            max_eval_samples=spec.max_eval_samples,
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


def run_shap_case(spec: ShapRunSpec, *, output_dir: Path) -> ShapCaseSummary:
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
    OmegaConf.save(
        OmegaConf.create(dict(summary)),
        output_dir / "shap_summary.yaml",
    )
    agg = aggregate_feature_importance(records)
    if not agg.empty:
        agg.to_csv(output_dir / "shap_aggregate.csv", index=False)
    return summary
