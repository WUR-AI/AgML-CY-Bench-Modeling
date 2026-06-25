from __future__ import annotations

import logging
import os
import re
import copy
import threading
from pathlib import Path
from typing import Any, Optional, cast

import hydra
import numpy as np
import optuna
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
from tqdm import tqdm

from cybench.util.config_utils import set_seed, remove_search_keys
from cybench.datasets.dataset import BaseDataset, PandasDataset
from cybench.util.feature_selection import apply_mrmr_at_origin, resolved_feature_selection_cfg
from cybench.util.screening_artifacts import save_optimal_epochs
from cybench.util.validation import get_splits

log = logging.getLogger(__name__)

#### Custom Multi Pruner

class SequentialPruner(optuna.pruners.BasePruner):
    """Apply pruners in order; first one that wants to prune wins."""
    def __init__(self, pruners):
        self._pruners = list(pruners)

    def prune(self, study, trial) -> bool:
        return any(p.prune(study, trial) for p in self._pruners)

def build_pruner(cfg_pruner):
    if cfg_pruner is None:
        return optuna.pruners.NopPruner()

    # list of pruners
    if isinstance(cfg_pruner, (list, ListConfig)):
        pruners = [instantiate(p) for p in cfg_pruner]
        if len(pruners) == 1:
            return pruners[0]
        return SequentialPruner(pruners)

    # single pruner
    return instantiate(cfg_pruner)


def _extract_search_space(cfg: DictConfig | Any, prefix: str = "") -> dict[str, Any]:
    """
    Recursively extract all _search_: definitions from a config, keyed by their
    dotted parameter path (e.g. "epochs", "dataloader.batch_size").
    Returns a flat dict suitable for saving as search_space.yaml.
    """
    result = {}
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if key == "_search_":
                for param_name, param_details in value.items():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    result[full_name] = OmegaConf.to_container(param_details, resolve=True)
            elif isinstance(value, DictConfig):
                key_str = str(key)
                new_prefix = f"{prefix}.{key_str}" if prefix else key_str
                result.update(_extract_search_space(value, new_prefix))
    return result


def load_previous_best_trials(
    storage: str,
    current_study_name: str,
    n_best: Optional[int] = None,
) -> list[optuna.trial.FrozenTrial] | None:
    """
    Load a previously saved study (t-1) from storage if it exists, filtering for the best trials.
    Regex logic: Looks strictly at the LAST 4 digits as the year, treating everything before as the prefix.
    """
    # 1. Flexible Regex: Capture everything (Group 1) up until the last 4 digits (Group 2)
    # ^(.*)    -> Group 1: The Prefix (Greedy, includes separators like _ or /)
    # (\d{4})$ -> Group 2: The Year (Last 4 digits)
    match = re.search(r"^(.*)(\d{4})$", current_study_name)

    if not match:
        log.warning(f"Could not parse year from study name: {current_study_name}. Warm start disabled.")
        return None

    # FIX: Unpack explicitly. Group 2 is the year.
    prefix = match.group(1)
    year_str = match.group(2)

    try:
        current_year = int(year_str)
    except ValueError:
        log.warning(f"Failed to convert year '{year_str}' to int.")
        return None

    target_year = current_year - 1

    # 2. Reconstruct the Target Name
    # We simply append the (year - 1) to the exact prefix we found.
    # e.g. "path/to/2020" -> prefix="path/to/" -> target="path/to/2019"
    # e.g. "exp_2020"     -> prefix="exp_"     -> target="exp_2019"
    target_study_name = f"{prefix}{target_year}"

    # 3. Check if this study exists in storage
    summaries = optuna.get_all_study_summaries(storage=storage)
    study_exists = any(s.study_name == target_study_name for s in summaries)

    if not study_exists:
        log.info(f"No previous study found: {target_study_name} (derived from {current_study_name})")
        return None

    log.info(f"Loading source trials from previous study: {target_study_name}")
    try:
        previous_study = optuna.load_study(study_name=target_study_name, storage=storage)
    except Exception as e:
        log.warning(f"Failed to load previous study {target_study_name}: {e}")
        return None

    # 4. Filter for COMPLETE trials only
    valid_trials = [t for t in previous_study.trials if
                    t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    if not valid_trials:
        log.warning(f"Previous study {target_study_name} has no complete trials. Warm start skipped.")
        return None

    # 5. Filter n_best
    if n_best is not None and n_best > 0:
        # Sort based on direction (Assuming minimization for Loss)
        valid_trials.sort(
            key=lambda t: t.value if t.value is not None else float("inf"),
            reverse=False,
        )

        original_count = len(valid_trials)
        valid_trials = valid_trials[:n_best]
        log.info(f"Filtered top {len(valid_trials)}/{original_count} trials from {target_year} for warm start.")

    return valid_trials


####

class OptunaOptimizer:
    """
    Hyperparameter Optimizer using Optuna.

    This class handles the creation of an Optuna study, the definition of the
    objective function based on a Hydra configuration with recursive `_search_`
    keys, and the storage of the optimal configuration.

    Args:
        cfg: Configuration containing hp_config (Optuna settings), val_cfg (for val-splitting) &
                                model_cfg (model plus `_search_` keys to be optimized)
        dataset (BaseDataset): The dataset instance to be used for optimization.
        path (str): Directory path where the `optimal_model.yaml` will be stored.
        study_name (str): Unique identifier for the Optuna study.
    """

    def __init__(
            self,
            cfg,
            dataset: BaseDataset,
            path: str | Path,
            study_name: str,
            split_dataset_years: set[Any] | list[Any] | None = None,
    ):
        self.multi_gpu = False
        if cfg.experiment.n_jobs > 1:
            if cfg.experiment.device == "cuda":
                import torch

                # check whether there are enough GPUs for parallel trials
                self._n_gpus = torch.cuda.device_count()
                assert self._n_gpus >= 1, f"No GPU found. Check cuda installation or fallback to CPU: in config.yaml change experiment.device=cpu."
                assert self._n_gpus == cfg.experiment.n_jobs, f"Number of available GPUs ({self._n_gpus}) is not equal to number of parallel jobs ({cfg.experiment.n_jobs}). Match those numbers to achieve optimal runtime and utilization."
                # assign multi-GPU scheduling state
                self._gpu_active_trials = [0] * self._n_gpus
                self._gpu_lock = threading.Lock()
                self.multi_gpu = True

        self.cfg = cfg
        self.hp_config = cfg.hp_search
        self.val_cfg = cfg.validation
        self.dataset = dataset
        self.path = Path(path)
        self.study_name = study_name
        self.split_dataset_years = (
            set(int(y) for y in split_dataset_years)
            if split_dataset_years is not None
            else None
        )

        # Ensure output directory exists
        self.path.mkdir(parents=True, exist_ok=True)

    def optimize(self) -> DictConfig:
        """
        Runs the Optuna optimization process.

        Returns:
            DictConfig: The best model configuration found, with `_search_` keys
                        resolved to their optimal values.
        """
        log.info(f"Starting Optuna optimization: {self.study_name}")

        # 1. Setup Optuna Storage, Sampler & Pruner
        storage = self.hp_config.storage.url if self.hp_config.get("storage") else None

        # 2. Create Pruner
        if "pruner" in self.hp_config and self.hp_config.pruner is not None:
            pruner = build_pruner(self.hp_config.pruner)
        else:
            pruner = None

        # 3. Get transfer parameters for sampler
        n_warmup = self.hp_config.get("n_warmup_best", 0)
        n_enqueue = self.hp_config.get("enqueue_n_best", 0)
        n_warmup_trials = None
        n_enqueue_trials = None
        if n_warmup and storage:
            n_warmup_trials = load_previous_best_trials(storage, self.study_name, n_warmup)
        if n_enqueue and storage:
            n_enqueue_trials = load_previous_best_trials(storage, self.study_name, n_enqueue)
        # further the finetuning might shrink the amount of necessary trials
        n_finetune_trials = self.hp_config.get("n_finetune_trials", 0)
        # Check if we successfully loaded ANY previous knowledge
        has_prior_knowledge = (n_warmup_trials is not None and len(n_warmup_trials) > 0) or \
                              (n_enqueue_trials is not None and len(n_enqueue_trials) > 0)

        if n_finetune_trials and has_prior_knowledge:
            log.info(f"Transfer successful. Reducing trials to n_finetune_trials={n_finetune_trials}")
            n_trials = n_finetune_trials
            # Ensure we have at least 1 startup trial to register search space if using pure warm start
            self.hp_config.sampler.n_startup_trials = 1
        else:
            n_trials = self.hp_config.get("n_trials", 100)

        # 4. Create Sampler
        if "sampler" in self.hp_config:
            # We explicitly check the target to decide how to instantiate: CmaEsSampler
            is_cma = "CmaEsSampler" in self.hp_config.sampler.get("_target_", "")

            if is_cma and n_warmup > 0 and n_warmup_trials:
                # CMA-ES: Pass source_trials into __init__
                sampler = instantiate(self.hp_config.sampler, source_trials=n_warmup_trials)
                log.info(f"Initialized CMA-ES with {len(n_warmup_trials)} source trials.")
            else:
                # Standard Instantiation (TPE or CMA without history)
                sampler = instantiate(self.hp_config.sampler)
        else:
            log.info("No sampler specified: falling back to `optuna.samplers.TPESampler`. Specifying sampler is recommended.")
            sampler = optuna.samplers.TPESampler(seed=self.hp_config.seed)

        # 5. Create Study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=self.hp_config.storage.get("load_if_exists", True),
            direction="minimize"
        )

        # 6. Enqueue the best previous trails
        if n_enqueue:
            if n_enqueue_trials:
                log.info(f"Enqueueing {len(n_enqueue_trials)} best trials for TPE warm start.")
                for t in n_enqueue_trials:
                    study.enqueue_trial(t.params)

        # 7. Optimize
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=self.hp_config.get("timeout", None),
            n_jobs=self.hp_config.get("n_jobs", 1),
            show_progress_bar=self.hp_config.logging.get("show_progress_bar", True)
        )

        log.info(f"Optimization finished. Best trial: {study.best_trial.params}")

        # 8. Reconstruct and Save Best Config
        # We use a FixedTrial with the best params to reconstruct the full config structure
        best_trial = optuna.trial.FixedTrial(study.best_params)
        best_config = self.cfg.copy()

        # Resolve the search space with best values
        self._resolve_search_space(best_config, best_trial)
        best_model_config = best_config.model

        # Clean any remaining artifacts (though _resolve_search_space handles _search_ removal)
        best_model_config = remove_search_keys(best_model_config)

        # Save to disk
        output_file = self.path / "optimal_model.yaml"
        with open(output_file, "w") as f:
            OmegaConf.save(best_model_config, f)

        if "feature_selection" in best_config:
            fs_cfg = remove_search_keys(best_config.feature_selection)
            fs_file = self.path / "optimal_feature_selection.yaml"
            OmegaConf.save(fs_cfg, fs_file)
            log.info(f"Optimal feature selection saved to: {fs_file}")

        # Save search space metadata for post-hoc boundary analysis
        search_space = _extract_search_space(self.cfg.model)
        if "feature_selection" in self.cfg:
            search_space.update(_extract_search_space(self.cfg.feature_selection, "feature_selection"))
        if search_space and bool(OmegaConf.select(self.cfg, "store.save_hpo_search_space", default=False)):
            ss_file = self.path / "search_space.yaml"
            OmegaConf.save(OmegaConf.create(search_space), ss_file)
            log.info(f"Search space metadata saved to: {ss_file}")

        log.info(f"Optimal model config saved to: {output_file}")

        E_star = study.best_trial.user_attrs.get("E_star")
        if E_star is not None:
            max_budget = study.best_params.get("epochs")
            save_optimal_epochs(
                self.path / "optimal_epochs.yaml",
                int(E_star),
                max_epochs_budget=int(max_budget) if max_budget is not None else None,
            )
            log.info("Optimal early-stopping epoch E*=%s saved to optimal_epochs.yaml", E_star)

        return best_model_config

    def _objective(self, trial: optuna.Trial) -> float:
        """
        The objective function for Optuna.
        1. Clones the base config.
        2. Samples parameters using the trial.
        3. Instantiates the model.
        4. Performs validation (CV or Hold-out).
        5. Returns the mean validation metric.
        """
        # Deep copy to avoid modifying the original config for other trials
        cfg_copy = copy.deepcopy(self.cfg)

        # Apply suggested parameters to the config
        self._resolve_search_space(cfg_copy, trial)
        trial_model_cfg = cfg_copy.model

        # Instantiate model with specific trial configuration
        trial_model_cfg = remove_search_keys(trial_model_cfg)

        fs_cfg = None
        if "feature_selection" in cfg_copy:
            if not isinstance(self.dataset, PandasDataset):
                raise ValueError(
                    "feature_selection requires a PandasDataset "
                    "(dataset.framework=pandas)."
                )
            fs_cfg = resolved_feature_selection_cfg(cfg_copy)

        if self.multi_gpu:
            device = self._assign_gpu()
            log.debug(
                f"[Trial {trial.number}] | device={device} | "
                f"thread={threading.current_thread().name} | "
                f"gpu_load={self._gpu_active_trials}"  # e.g. [2, 1, 1, 1]
            )
            trial_model_cfg["device"] = device

        show_progress = bool(
            self.hp_config.get("logging", {}).get("show_progress_bar", True)
        )
        if show_progress:
            trial_model_cfg["verbose"] = True

        val_metrics = []
        device = "cpu"
        try:
            for i in range(self.hp_config.repetitions):
                set_seed(self.hp_config.seed + i)

                # Screening HPO must resolve train/val from the full timeline; ``dataset``
                # is already truncated to pre-test years for the final fit.
                years_for_splits = (
                    self.split_dataset_years
                    if self.split_dataset_years is not None
                    else set(self.dataset.years)
                )
                splits = list(
                    get_splits(
                        cfg=self.val_cfg,
                        which="val",
                        dataset_years=years_for_splits,
                        seed=self.hp_config.seed,
                    )
                )

                split_iter = tqdm(
                    splits,
                    desc=f"Trial {trial.number} splits",
                    unit="split",
                    disable=not show_progress or len(splits) <= 1,
                    leave=False,
                )
                for train_years, val_years in split_iter:
                    if show_progress and len(splits) > 1:
                        split_iter.set_postfix(
                            train=f"{min(train_years)}-{max(train_years)}",
                            val=f"{min(val_years)}-{max(val_years)}",
                        )
                    train_dataset, val_dataset = self.dataset.split_on_years((train_years, val_years))

                    if fs_cfg is not None:
                        train_dataset, val_dataset, _ = apply_mrmr_at_origin(
                            source_dataset=self.dataset,
                            train_years=list(train_years),
                            fs_cfg=fs_cfg,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                        )

                    if "pretrained_from" in trial_model_cfg:
                        # append the test-year to the model config, so that it loads the model that only trained on years before the test-year
                        trial_model_cfg["test_years"] = [int(year) for year in val_years]

                    max_epochs = int(trial_model_cfg.get("epochs", 100))
                    log_interval = 5
                    if self.hp_config.get("logging") is not None:
                        log_interval = int(self.hp_config.logging.get("log_interval", 5))
                    log.info(
                        "Trial %d: training %s for up to %d epochs (train %s, val %s)",
                        trial.number,
                        trial_model_cfg.get("name", "model"),
                        max_epochs,
                        train_years,
                        val_years,
                    )
                    fit_kwargs: dict[str, Any] = {"epoch_log_interval": log_interval}

                    # Instantiate and fit
                    model = instantiate(trial_model_cfg, verbose=False)
                    fit_stages = tqdm(
                        total=2,
                        desc=f"Trial {trial.number} fit/predict",
                        unit="stage",
                        disable=not show_progress,
                        leave=False,
                    )
                    try:
                        if "early_stopping" in trial_model_cfg:
                            _, history = model.fit(
                                train_dataset, val_dataset=val_dataset, **fit_kwargs
                            )
                        else:
                            _, history = model.fit(train_dataset, **fit_kwargs)
                        fit_stages.update(1)
                        fit_stages.set_postfix(stage="predict")

                        # Predict and Evaluate
                        preds, _ = model.predict(val_dataset)
                        fit_stages.update(1)
                    finally:
                        fit_stages.close()

                    best_epoch = None
                    if getattr(model, "early_stopping", None) is not None:
                        best_epoch = model.early_stopping.best_epoch
                    elif isinstance(history, dict):
                        best_epoch = history.get("best_epoch")
                    if best_epoch is not None:
                        trial.set_user_attr("E_star", int(best_epoch))

                    assert preds.ndim == val_dataset.targets.ndim, (
                        f"The model output shape {preds.shape} does not match "
                        f"target shape {val_dataset.targets.shape}"
                    )
                    val_metric = np.mean((val_dataset.targets - preds) ** 2)

                    val_metrics.append(val_metric)
                    log.info(
                        "Trial %d finished | val MSE=%.4f | E*=%s",
                        trial.number,
                        float(val_metric),
                        trial.user_attrs.get("E_star"),
                    )
                    log.debug(f"Validation metric ({i} / {val_years}): {val_metric}")

                    # Optional: Pruning based on intermediate folds
                    trial.report(float(np.mean(val_metrics)), step=len(val_metrics))
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            return float(np.mean(val_metrics))
        finally:
            if self.multi_gpu:
                self._release_gpu(device)

    def _resolve_search_space(
            self,
            cfg: DictConfig | Any,
            trial: optuna.trial.Trial | optuna.trial.FixedTrial,
            prefix: str = "",
    ) -> None:
        """
        Recursively traverses the config, finds `_search_` keys, samples values
        from Optuna, updates the config in-place, and removes the `_search_` key.
        """
        if isinstance(cfg, DictConfig):
            # Iterate over a copy of keys to allow modification
            for key in list(cfg.keys()):
                value = cfg[key]

                # Check if this node has a search definition
                if key == "_search_":
                    search_params = value

                    # Iterate over parameters defined in _search_
                    for param_name, param_details in search_params.items():
                        # Construct unique name for Optuna
                        full_param_name = f"{prefix}.{param_name}" if prefix else param_name

                        # Extract type and constraints
                        p_type = param_details["type"]
                        param_container = cast(
                            dict[str, Any],
                            OmegaConf.to_container(param_details, resolve=True),
                        )
                        kwargs = {
                            k: v for k, v in param_container.items() if k != "type"
                        }

                        suggestion = None

                        if p_type == "int":
                            suggestion = trial.suggest_int(full_param_name, **kwargs)
                        elif p_type == "float":
                            suggestion = trial.suggest_float(full_param_name, **kwargs)
                        elif p_type == "categorical":
                            # 'choices' is required for categorical
                            choices = kwargs.get("choices")
                            if not choices:
                                raise ValueError(f"Categorical param {full_param_name} missing 'choices'")
                            suggestion = trial.suggest_categorical(full_param_name, choices)
                        else:
                            raise ValueError(f"Unknown search type '{p_type}' for {full_param_name}.")

                        cfg[param_name] = suggestion

                    # Remove the _search_ key after processing
                    del cfg._search_

                elif isinstance(value, (DictConfig, dict)):
                    # Recurse
                    key_str = str(key)
                    new_prefix = f"{prefix}.{key_str}" if prefix else key_str
                    self._resolve_search_space(value, trial, new_prefix)

    def _assign_gpu(self) -> str:
        """Pick the GPU with the fewest active trials."""
        if self._n_gpus == 0:
            return "cpu"
        with self._gpu_lock:
            gpu_id = int(np.argmin(self._gpu_active_trials))
            self._gpu_active_trials[gpu_id] += 1
            return f"cuda:{gpu_id}"

    def _release_gpu(self, device: str) -> None:
        """Decrement counter when trial finishes or fails."""
        if device == "cpu":
            return
        gpu_id = int(device.split(":")[-1])
        with self._gpu_lock:
            self._gpu_active_trials[gpu_id] = max(0, self._gpu_active_trials[gpu_id] - 1)