"""Smoke test for the Hydra experiment pipeline (replaces run_benchmark integration test)."""

from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.datasets.data_factory import DataFactory
from cybench.evaluation.eval import evaluate_predictions
from cybench.util.config_utils import remove_search_keys, set_seed
from cybench.util.validation import get_splits


def test_run_experiments_smoke():
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=maize",
                "dataset.country=NL",
                "dataset.framework=pandas",
                "dataset.target.filter_samples=null",
                "model=average",
                "validation=single",
                "validation.test_years=[2010,2011]",
                "experiment.n_repetitions=1",
                "experiment.seed=42",
            ],
        )

    set_seed(cfg.experiment.seed)
    dataset = DataFactory(cfg.dataset).build()
    model_cfg = remove_search_keys(cfg.model)

    for train_test_split in get_splits(
        cfg=cfg.validation,
        which="test",
        dataset_years=dataset.years,
        seed=cfg.experiment.seed,
    ):
        train_dataset, test_dataset = dataset.split_on_years(years_split=train_test_split)
        model = instantiate(model_cfg)
        model.fit(train_dataset, val_dataset=test_dataset)
        test_preds, _ = model.predict(test_dataset)

        assert test_preds.shape[0] == len(test_dataset)
        metrics = evaluate_predictions(
            y_true=test_dataset.targets,
            y_pred=test_preds,
            cfg=cfg.evaluation,
        )
        assert "normalized_rmse" in metrics
        assert metrics["normalized_rmse"] >= 0
