import hydra
import matplotlib.pyplot as plt
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging

from cybench.config import ExperimentConfig
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.torch_dataset import TorchDataset
from cybench.util.config_utils import adjust_model_cfg_to_dataset
from cybench.util.store_and_cache import make_split_folder, save_preds
from cybench.util.validation import get_train_test_splits

# init logger
log = logging.getLogger(__name__)

# init config store to use custom config dataclass (see config.py)
conf_store = ConfigStore.instance()
conf_store.store(name="exp_config", node=ExperimentConfig)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig):
    print("=== Final Composed Config ===")
    print(OmegaConf.to_yaml(cfg))
    log.info("=== Create Datasets ===")
    dataset = DataFactory(cfg.dataset).build()
    cfg.model = adjust_model_cfg_to_dataset(cfg.model, dataset)

    # split data in train- and test-set based on the validation strategy
    for years_split in get_train_test_splits(cfg=cfg.validation, years=dataset.years):
        train_dataset, test_dataset = dataset.split_on_years(years_split=years_split)
        # create a folder for each split
        split_path = make_split_folder(run_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, split_name=years_split[-1])
        # check whether hyperparameter tuning is equipped:
        if cfg.hp_search:
            model_cfg = cfg.model # "TODO: implement Optuna hyperparameter search"
        else:
            model_cfg = cfg.model
        # create final model
        model = instantiate(model_cfg)
        _, fit_info = model.fit(train_dataset, val_dataset=test_dataset)
        test_preds, pred_info = model.predict(test_dataset)
        # save preds, model, ...
        save_preds(path=split_path, dataset=test_dataset, preds=test_preds, pred_info=pred_info)
        print("wow")

if __name__ == "__main__":
    main()