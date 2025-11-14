import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging

from cybench.config import ExperimentConfig
from cybench.datasets.data_factory import DataFactory
from cybench.datasets.torch_dataset import TorchDataset
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

    # split data in train- and test-set based on the validation strategy
    for years_split in get_train_test_splits(cfg=cfg.validation, years=dataset.years):
        train_dataset, test_dataset = dataset.split_on_years(years_split=years_split)
        # check whether hyperparameter tuning is equipped:
        if cfg.hp_search:
            model_cfg = "TODO: implement Optuna hyperparameter search"
        else:
            model_cfg = cfg.model
        # create final model
        model = instantiate(model_cfg)
        model.fit_predict(train_dataset, test_dataset)

if __name__ == "__main__":
    main()