from __future__ import annotations

import os

# Optional CPU determinism for torch training (set CYBENCH_TORCH_THREADS, default 1).
_thread_count = os.environ.get("CYBENCH_TORCH_THREADS")
if _thread_count is not None:
    for _blas_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_blas_var, _thread_count)

import sys
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import logging
from codecarbon import track_emissions
from omegaconf import DictConfig, OmegaConf, open_dict

from cybench.util.prediction_horizon import prediction_horizon_tag

OmegaConf.register_new_resolver("prediction_horizon_tag", prediction_horizon_tag)

from cybench.datasets.data_factory import DataFactory
from cybench.datasets.dataset import PandasDataset
from cybench.evaluation.eval import evaluate_predictions
from cybench.evaluation.aggregated_metrics import compute_report_metrics, format_report_metrics
from cybench.util.config_utils import (
    adjust_model_cfg_to_dataset,
    apply_force_cpu_to_frozen_model_cfg,
    remove_search_keys,
    set_seed,
    walk_forward_force_cpu,
)
from cybench.util.optuna_hyper_opt import OptunaOptimizer
from cybench.util.store_and_cache import make_folder, save_preds, save_meta_dict
from cybench.util.feature_selection import (
    apply_mrmr_at_origin,
    resolved_feature_selection_cfg,
    save_selected_features,
)
from cybench.util.screening_artifacts import (
    load_frozen_screening_artifacts,
    load_optimal_epochs,
)
from cybench.util.validation import (
    default_screening_validation_cfg,
    get_screening_partitions,
    get_screening_pre_test_years,
    get_splits,
)
from cybench.config import KEY_COUNTRY, KEY_LOC, KEY_YEAR, KEY_TARGET
from cybench.models.twso_model import TwsoNotApplicableError

# init logger
log = logging.getLogger(__name__)


def _resolve_normalizer_fit_years(cfg) -> list[int] | None:
    """Fit normalization on screening train ∪ val only (exclude test block)."""
    if cfg.validation.name == "screening":
        partition_cfg = cfg.validation
    elif cfg.validation.name == "walk_forward":
        partition_cfg = default_screening_validation_cfg()
    else:
        return None
    years = DataFactory.peek_dataset_years(cfg.dataset)
    fit_years = get_screening_pre_test_years(
        years, seed=cfg.experiment.seed, cfg=partition_cfg
    )
    log.info("Normalizer fit years (screening train+val): %s", fit_years)
    return fit_years


def _model_target(cfg) -> str:
    target = OmegaConf.select(cfg, "model._target_") or ""
    return target if isinstance(target, str) else ""


def _maybe_drop_temporal_for_standalone_models(cfg) -> None:
    """Standalone baselines (average yield, LPJmL) do not use time-series predictors."""
    target = _model_target(cfg)
    standalone_markers = (
        "AverageYieldModel",
        "TrendModel",
        "LpjmlBiasCorrectedModel",
        "TwsoBiasCorrectedModel",
    )
    if not any(marker in target for marker in standalone_markers):
        return
    if "temporal" not in cfg.dataset or "sources" not in cfg.dataset.temporal:
        return
    with open_dict(cfg.dataset.temporal):
        cfg.dataset.temporal.sources.clear()
    log.info(
        "%s: dropped all temporal sources (model does not use time series).",
        target.rsplit(".", 1)[-1],
    )


def _is_torch_model(model_cfg) -> bool:
    return OmegaConf.select(model_cfg, "framework") == "torch"


def _prepare_screening_final_nn_cfg(model_cfg: DictConfig, E_star: int) -> DictConfig:
    """Screening final fit: train on train+val for exactly E* epochs (no val early stopping)."""
    cfg_out = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(model_cfg)))
    with open_dict(cfg_out):
        cfg_out.epochs = int(E_star)
        if "early_stopping" in cfg_out:
            del cfg_out.early_stopping
    return cfg_out


def _prepare_walk_forward_nn_cfg(model_cfg: DictConfig, E_star: int) -> DictConfig:
    """Walk-forward: at most E* epochs, early stop on training loss if converged."""
    cfg_out = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(model_cfg)))
    with open_dict(cfg_out):
        cfg_out.epochs = int(E_star)
        cfg_out.early_stopping_monitor = "train"
    return cfg_out


def _fit_model(model, train_dataset, *, walk_forward_nn: bool = False):
    if walk_forward_nn:
        return model.fit(train_dataset, early_stopping_monitor="train")
    return model.fit(train_dataset)


def _store_enabled(cfg, key: str, *, default: bool = False) -> bool:
    return bool(OmegaConf.select(cfg, f"store.{key}", default=default))


@track_emissions(log_level="WARNING")
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    #print("=== Final Composed Config ===")
    #print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.seed)

    _maybe_drop_temporal_for_standalone_models(cfg)

    log.info("=== Create Datasets ===")
    normalizer_fit_years = _resolve_normalizer_fit_years(cfg)
    dataset = DataFactory(cfg.dataset).build(normalizer_fit_years=normalizer_fit_years)
    log.info(f"Dataset years: {sorted(dataset.years)}")
    horizon = OmegaConf.select(cfg, "dataset.temporal.season.end_of_sequence")
    log.info("Prediction horizon (end_of_sequence): %s → tag %s", horizon, prediction_horizon_tag(str(horizon or "eos")))
    if "process" in cfg and cfg.dataset.framework == "torch":
        dataset.process(cfg.process)
    # TODO extra function for testing config compatibility
    if cfg.dataset.framework == "pandas": assert "torch_model" not in cfg.model, "You selected a torch model but no torch dataset. Switch to torch dataset by dataset.framework=torch or select a model that operates on tabular data (PandasDataset)."
    if cfg.dataset.framework == "torch":
        assert "torch_model" in cfg.model, "Your model config is missing the key 'torch_model'. Select a model operating on torch datasets or select another framework, such as dataset.framework=pandas for creating a PandasDataset suiting models for tabular data."
        # adjust and save model config to match the input datas dimension
        cfg.model = adjust_model_cfg_to_dataset(cfg.model, dataset)


    # split data in train- and test-set based on the validation strategy
    run_output_dir = HydraConfig.get().runtime.output_dir
    is_walk_forward = cfg.validation.name == "walk_forward"
    frozen_model_cfg: DictConfig | None = None
    frozen_fs_cfg: DictConfig | None = None
    frozen_E_star: int | None = None

    if is_walk_forward:
        frozen_dir = cfg.validation.get("frozen_screening_dir")
        if not frozen_dir:
            raise ValueError(
                "validation=walk_forward requires validation.frozen_screening_dir "
                "pointing to the screening split folder with optimal_*.yaml files."
            )
        frozen_model_cfg, frozen_fs_cfg, frozen_E_star = load_frozen_screening_artifacts(
            frozen_dir
        )
        frozen_model_cfg = remove_search_keys(frozen_model_cfg)
        log.info(
            "Walk-forward phase | frozen artifacts from %s | E*=%s",
            frozen_dir,
            frozen_E_star,
        )

    if cfg.validation.name == "screening":
        train_years, val_years, test_years = get_screening_partitions(
            cfg=cfg.validation,
            dataset_years=dataset.years,
            seed=cfg.experiment.seed,
        )
        log.info(
            "Screening split | train=%s | val=%s (HPO) | test=%s (held out)",
            train_years,
            val_years,
            test_years,
        )

    for train_test_split in get_splits(cfg=cfg.validation,
                                       which="test",
                                       dataset_years=dataset.years,
                                       seed=cfg.experiment.seed
                                       ):
        train_years, test_years = train_test_split
        log.info(f"== Split Test: {test_years} ==")
        train_dataset, test_dataset = dataset.split_on_years(years_split=train_test_split)
        # create a folder for each split
        split_path = make_folder(dir=run_output_dir, name=test_years)

        if cfg.validation.name == "screening" and _store_enabled(
            cfg, "save_screening_partitions", default=True
        ):
            screen_train, screen_val, screen_test = get_screening_partitions(
                cfg=cfg.validation,
                dataset_years=dataset.years,
                seed=cfg.experiment.seed,
            )
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "train_years": screen_train,
                        "val_years": screen_val,
                        "test_years": screen_test,
                        "final_fit_years": train_years,
                    }
                ),
                f=split_path / "screening_partitions.yaml",
            )

        # check whether hyperparameter tuning is equipped:
        fs_cfg: DictConfig | None = None
        if is_walk_forward:
            model_cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(frozen_model_cfg)))
            if walk_forward_force_cpu(cfg):
                prev_device = OmegaConf.select(model_cfg, "device")
                model_cfg = apply_force_cpu_to_frozen_model_cfg(model_cfg)
                if prev_device and prev_device != "cpu":
                    log.info(
                        "Walk-forward force CPU: frozen model device %s → cpu",
                        prev_device,
                    )
            fs_cfg = (
                cast(DictConfig, OmegaConf.create(OmegaConf.to_container(frozen_fs_cfg)))
                if frozen_fs_cfg is not None
                else None
            )
            if _is_torch_model(model_cfg) and frozen_E_star is not None:
                model_cfg = _prepare_walk_forward_nn_cfg(model_cfg, frozen_E_star)
        elif "hp_search" in cfg:
            hp_optimizer = OptunaOptimizer(
                cfg=cfg,
                dataset=train_dataset,
                path=split_path,
                study_name=f"{split_path.parent.name}_{split_path.name}",
                split_dataset_years=dataset.years,
            )
            model_cfg = cast(DictConfig, hp_optimizer.optimize())
        else:
            # _search_ keys for hyperparameter tuning have to be removed before model instantiation
            model_cfg = remove_search_keys(cfg.model)
        if _store_enabled(cfg, "save_split_configs"):
            OmegaConf.save(config=model_cfg, f=split_path / "model_config.yaml")

        if (
            cfg.validation.name == "screening"
            and _is_torch_model(model_cfg)
            and (split_path / "optimal_epochs.yaml").exists()
        ):
            E_star = load_optimal_epochs(split_path / "optimal_epochs.yaml")
            if E_star is not None:
                model_cfg = _prepare_screening_final_nn_cfg(model_cfg, E_star)
                log.info("Screening final NN fit | E*=%d epochs on train+val", E_star)

        if cfg.validation.name == "screening":
            # Always save for walk-forward chaining (incl. naive models without HPO).
            OmegaConf.save(config=model_cfg, f=split_path / "optimal_model.yaml")

        if not is_walk_forward:
            fs_cfg = resolved_feature_selection_cfg(cfg)
        if fs_cfg is not None:
            if not isinstance(train_dataset, PandasDataset):
                raise ValueError(
                    "feature_selection is only supported for PandasDataset "
                    "(dataset.framework=pandas)."
                )
            optimal_fs_path = split_path / "optimal_feature_selection.yaml"
            if optimal_fs_path.exists():
                fs_cfg = cast(DictConfig, OmegaConf.load(optimal_fs_path))
            elif is_walk_forward and frozen_fs_cfg is not None:
                fs_cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(frozen_fs_cfg)))
            train_dataset, test_dataset, selected = apply_mrmr_at_origin(
                source_dataset=cast(PandasDataset, dataset),
                train_years=list(train_years),
                fs_cfg=fs_cfg,
                train_dataset=train_dataset,
                eval_dataset=cast(PandasDataset, test_dataset),
            )
            if _store_enabled(cfg, "save_split_configs"):
                save_selected_features(
                    split_path / "selected_features.yaml",
                    selected=selected,
                    fs_cfg=fs_cfg,
                    train_years=list(train_years),
                )
            log.info(
                "Final mRMR at origin | k=%d | selected %d features | train years %s",
                int(fs_cfg.k),
                len(selected),
                train_years,
            )

        log.info(
            "Train final model on %d datapoints (years %s)",
            len(train_dataset),
            train_years,
        )
        metric_ls = []
        for i in range(cfg.experiment.n_repetitions):
            meta_dict = {}

            # set new seed for each repetition
            seed = cfg.experiment.seed + i
            set_seed(seed)
            repetition_path = make_folder(dir=split_path, name=seed)

            # create, fit final model and predict test
            model = instantiate(model_cfg)
            try:
                fit_info = _fit_model(
                    model,
                    train_dataset,
                    walk_forward_nn=is_walk_forward and _is_torch_model(model_cfg),
                )
            except TwsoNotApplicableError as exc:
                if cfg.validation.name == "screening":
                    log.warning(
                        "[SKIP] TWSO screening not applicable for %s/%s: %s",
                        cfg.dataset.crop.name,
                        cfg.dataset.country,
                        exc,
                    )
                    sys.exit(0)
                raise
            test_preds, pred_info = model.predict(test_dataset)

            # save preds, model, ...
            year_preds_df = save_preds(path=repetition_path, dataset=test_dataset, preds=test_preds, file_name=f'test_preds')

            # Also export split predictions directly to the run root:
            # <crop>_<country>_year_<yyyy>.csv for easy downstream consumption.
            # If running multiple repetitions, append _seed_<seed> to avoid overwrite.
            test_year_tag = "_".join(str(y) for y in test_years)
            country_cfg = cfg.dataset.country
            country_tag = country_cfg if isinstance(country_cfg, str) else "-".join(country_cfg)
            horizon_tag = prediction_horizon_tag(
                str(OmegaConf.select(cfg, "dataset.temporal.season.end_of_sequence") or "eos")
            )
            root_file = f"{cfg.dataset.crop.name}_{country_tag}_h{horizon_tag}_year_{test_year_tag}"
            if cfg.experiment.n_repetitions > 1:
                root_file = f"{root_file}_seed_{seed}"

            # Format to wide, model-named schema expected by downstream scripts, e.g.
            # country_code,adm_id,year,yield,AverageYieldModel
            model_col = cfg.model._target_.split(".")[-1] if "_target_" in cfg.model else str(cfg.model.name)
            formatted_df = year_preds_df.rename(columns={"targets": KEY_TARGET, "preds": model_col})
            if KEY_COUNTRY not in formatted_df.columns:
                if isinstance(country_cfg, str):
                    formatted_df[KEY_COUNTRY] = country_cfg
                else:
                    # Multi-country fallback: infer from adm_id prefix (e.g. US-01-003 -> US)
                    formatted_df[KEY_COUNTRY] = formatted_df[KEY_LOC].astype(str).str.split("-").str[0]

            output_cols = [KEY_COUNTRY, KEY_LOC, KEY_YEAR, KEY_TARGET, model_col]
            formatted_df = formatted_df[output_cols]
            if _store_enabled(cfg, "export_root_csv"):
                formatted_df.to_csv(
                    f"{run_output_dir}/{root_file}.csv",
                    index=False,
                    float_format="%.6f",
                )
            if cfg.store.model:
                model.save(str(repetition_path))
            if cfg.store.meta:
                save_meta_dict(path=repetition_path, dict=meta_dict)

            # evaluate
            eval_metric = evaluate_predictions(y_true=test_dataset.targets, y_pred=test_preds, cfg=cfg.evaluation)
            report_metrics = compute_report_metrics(
                cast(pd.DataFrame, formatted_df),
                target_col=KEY_TARGET,
                model_col=model_col,
            )
            report_metrics["n_train"] = len(train_dataset)
            metric_ls.append(eval_metric)
            log.info(f"Split {train_test_split[-1]} (seed {seed}) finished with metrics: {eval_metric}")
            log.info(
                "Split %s (seed %s) report metrics: %s",
                train_test_split[-1],
                seed,
                format_report_metrics(report_metrics),
            )
            OmegaConf.save(
                OmegaConf.create(report_metrics),
                f=repetition_path / "report_metrics.yaml",
            )
        if cfg.experiment.n_repetitions > 1:
            for metric in metric_ls[0].keys():
                print(f"Average {metric}: {np.mean([metrics[metric] for metrics in metric_ls]):.3} (+- {np.std([metrics[metric] for metrics in metric_ls]):.3})")

if __name__ == "__main__":
    main()