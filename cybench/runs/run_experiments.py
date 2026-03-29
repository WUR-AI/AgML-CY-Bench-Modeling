import time
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import logging
from codecarbon import track_emissions
from omegaconf import OmegaConf
from pathlib import Path

from cybench.datasets.data_factory import DataFactory
from cybench.evaluation.eval import evaluate_predictions
from cybench.util.config_utils import adjust_model_cfg_to_dataset, set_seed, remove_search_keys
from cybench.util.optuna_hyper_opt import OptunaOptimizer
from cybench.util.store_and_cache import make_folder, save_preds, save_meta_dict
from cybench.util.validation import get_splits

# init logger
log = logging.getLogger(__name__)



@track_emissions(log_level="WARNING")
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    #print("=== Final Composed Config ===")
    #print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.seed)

    log.info("=== Create Datasets ===")
    dataset = DataFactory(cfg.dataset).build()
    if "process" in cfg: dataset.process(cfg.process)
    # TODO extra function for testing config compatibility
    if cfg.dataset.framework == "pandas": assert "torch_model" not in cfg.model, "You selected a torch model but no torch dataset. Switch to torch dataset by dataset.framework=torch or select a model that operates on tabular data (PandasDataset)."
    if cfg.dataset.framework == "torch":
        assert "torch_model" in cfg.model, "Your model config is missing the key 'torch_model'. Select a model operating on torch datasets or select another framework, such as dataset.framework=pandas for creating a PandasDataset suiting models for tabular data."
        # adjust and save model config to match the input datas dimension
        cfg.model = adjust_model_cfg_to_dataset(cfg.model, dataset)


    # split data in train- and test-set based on the validation strategy
    for train_test_split in get_splits(cfg=cfg.validation,
                                       which="test",
                                       dataset_years=dataset.years,
                                       seed=cfg.experiment.seed
                                       ):
        train_years, test_years = train_test_split
        log.info(f"== Split Test: {test_years} ==")
        train_dataset, test_dataset = dataset.split_on_years(years_split=train_test_split)
        # create a folder for each split
        split_path = make_folder(dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, name=test_years)

        # check whether hyperparameter tuning is equipped:
        if "hp_search" in cfg:
            hp_optimizer = OptunaOptimizer(
                cfg=cfg,
                dataset=train_dataset,
                path=split_path,
                study_name=f"{split_path.parent.name}_{split_path.name}"
            )
            model_cfg = hp_optimizer.optimize()
        else:
            # _search_ keys for hyperparameter tuning have to be removed before model instantiation
            model_cfg = remove_search_keys(cfg.model)
        # save final model config
        OmegaConf.save(config=model_cfg, f=split_path / "model_config.yaml")

        log.info(f"Train final model on {len(train_dataset.y)} datapoints")
        metric_ls = []
        for i in range(cfg.experiment.n_repetitions):
            meta_dict = {}

            # set new seed for each repetition
            seed = cfg.experiment.seed + i
            set_seed(seed)
            repetition_path = make_folder(dir=split_path, name=seed)

            # create, fit final model and predict test
            model = instantiate(model_cfg)
            fit_info = model.fit(train_dataset, val_dataset=test_dataset)
            test_preds, pred_info = model.predict(test_dataset)

            # save preds, model, ...
            save_preds(path=repetition_path, dataset=test_dataset, preds=test_preds, file_name=f'test_preds')
            if cfg.store.model: model.save(path=repetition_path)
            if cfg.store.meta: save_meta_dict(path=repetition_path, dict=meta_dict)

            # evaluate
            eval_metric = evaluate_predictions(y_true=test_dataset.targets, y_pred=test_preds, cfg=cfg.evaluation)
            metric_ls.append(eval_metric)
            log.info(f"Split {train_test_split[-1]} (seed {seed}) finished with metrics: {eval_metric}")
        if cfg.experiment.n_repetitions > 1:
            for metric in metric_ls[0].keys():
                print(f"Average {metric}: {np.mean([metrics[metric] for metrics in metric_ls]):.3} (+- {np.std([metrics[metric] for metrics in metric_ls]):.3})")

if __name__ == "__main__":
    main()