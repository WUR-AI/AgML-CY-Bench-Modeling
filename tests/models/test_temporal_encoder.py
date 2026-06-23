import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.datasets.data_factory import DataFactory
from cybench.evaluation.eval import evaluate_predictions
from cybench.models.torch.model_components.temporal_encoder import (
    AvgPoolTokenizer,
    LSTMProcessor,
    LastStepPooling,
    TemporalEncoder,
)
from cybench.util.config_utils import adjust_model_cfg_to_dataset, remove_keys


def test_avg_pool_tokenizer_shape():
    batch_size, seq_len, in_dim, embed_dim, patch_size = 2, 25, 8, 8, 10
    tokenizer = AvgPoolTokenizer(
        in_dim=in_dim, embed_dim=embed_dim, patch_size=patch_size
    )
    x = torch.randn(batch_size, seq_len, in_dim)
    doys = torch.randint(1, 366, (batch_size, seq_len))

    out = tokenizer(x, doys)
    assert out.shape == (batch_size, 2, embed_dim)


def test_avg_pool_tokenizer_channel_means_eos_anchored():
    patch_size = 10
    tokenizer = AvgPoolTokenizer(in_dim=1, embed_dim=1, patch_size=patch_size)
    x = torch.arange(25, dtype=torch.float32).view(1, 25, 1)
    doys = torch.zeros(1, 25, dtype=torch.int16)

    out = tokenizer(x, doys)

    assert out.shape == (1, 2, 1)
    assert out[0, 0, 0].item() == pytest.approx(torch.arange(5, 15).float().mean().item())
    assert out[0, 1, 0].item() == pytest.approx(torch.arange(15, 25).float().mean().item())


def test_avg_pool_tokenizer_has_no_trainable_parameters():
    tokenizer = AvgPoolTokenizer(in_dim=4, embed_dim=8, patch_size=10)
    assert sum(p.numel() for p in tokenizer.parameters()) == 0


def test_avg_pool_tokenizer_pads_channels_to_embed_dim():
    tokenizer = AvgPoolTokenizer(in_dim=3, embed_dim=5, patch_size=5)
    x = torch.ones(1, 10, 3)
    out = tokenizer(x, torch.zeros(1, 10, dtype=torch.int16))

    assert out.shape == (1, 2, 5)
    assert torch.all(out[..., 3:] == 0)


def test_avg_pool_tokenizer_rejects_short_sequence():
    tokenizer = AvgPoolTokenizer(in_dim=2, embed_dim=2, patch_size=10)
    x = torch.randn(1, 8, 2)
    with pytest.raises(ValueError, match="shorter than patch_size"):
        tokenizer(x, torch.zeros(1, 8, dtype=torch.int16))


def test_avg_pool_temporal_encoder_end_to_end():
    batch_size, seq_len, in_dim, embed_dim = 2, 30, 6, 6
    encoder = TemporalEncoder(
        tokenizer=AvgPoolTokenizer(in_dim=in_dim, embed_dim=embed_dim, patch_size=10),
        processor=LSTMProcessor(embed_dim=embed_dim, num_layers=1, dropout=0.0),
        pooling=LastStepPooling(),
        embed_dim=embed_dim,
    )
    temporal = torch.randn(batch_size, seq_len, in_dim)
    doys = torch.randint(1, 366, (batch_size, seq_len))

    out = encoder(temporal, doys)
    assert out.shape == (batch_size, embed_dim)


def test_lstm_lf_with_avg_pool_tokenizer_runs_on_dataset():
    overrides = [
        "dataset/crop=wheat",
        "dataset.country=NL",
        "dataset.framework=torch",
        "dataset.use_cache=false",
        "model=lstm_lf",
        "experiment.device=cpu",
        "model.epochs=1",
        "model/torch_model/temporal_encoder/tokenizer=avg_pool",
        "model.torch_model.embed_dim=8",
        "dataset/temporal=no_aggregate",
        "~dataset.temporal.sources.ndvi",
        "~dataset.temporal.sources.soil_moisture",
        "dataset.temporal.sources.meteo.select=[tmin,tmax,tavg,prec,rad,et0,vpd]",
    ]
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(config_name="config", overrides=overrides)

    dataset = DataFactory(cfg.dataset).build()
    cfg.model = adjust_model_cfg_to_dataset(cfg.model, dataset)
    model_cfg = remove_keys(cfg.model, "_search_")
    model = instantiate(model_cfg)

    even_years = {year for year in dataset.years if year % 2 == 0}
    odd_years = dataset.years - even_years
    train_dataset, test_dataset = dataset.split_on_years((even_years, odd_years))

    model.fit(train_dataset)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    targets = test_dataset.targets
    evaluation_result = evaluate_predictions(targets, test_preds, cfg.evaluation)
    assert "normalized_rmse" in evaluation_result
