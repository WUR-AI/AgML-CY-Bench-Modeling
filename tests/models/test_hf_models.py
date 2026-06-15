import copy

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from hydra import compose, initialize
from hydra.utils import instantiate

from cybench.datasets.data_factory import DataFactory
from cybench.evaluation.eval import evaluate_predictions
from cybench.models.torch.architectures.late_fusion_network import LateFusionNetwork
from cybench.models.torch.model_components.context_encoder import ContextEncoder
from cybench.models.torch.model_components.fusion import GatedFusion
from cybench.models.torch.model_components.head import RegressionHead
from cybench.models.torch.model_components.hf_temporal_encoder import (
    AutoformerTemporalEncoder,
    InformerTemporalEncoder,
    PatchTSTTemporalEncoder,
    TSTTemporalEncoder,
)
from cybench.util.config_utils import adjust_model_cfg_to_dataset, remove_keys

HF_ENCODER_CASES = [
    pytest.param(
        PatchTSTTemporalEncoder,
        {"patch_length": 8, "patch_stride": 4, "d_model": 32, "ffn_dim": 64, "num_layers": 2},
        id="patchtst",
    ),
    pytest.param(
        InformerTemporalEncoder,
        {"d_model": 32, "ffn_dim": 64, "num_layers": 2},
        id="informer",
    ),
    pytest.param(
        TSTTemporalEncoder,
        {"d_model": 32, "ffn_dim": 64, "num_layers": 2},
        id="tst",
    ),
    pytest.param(
        AutoformerTemporalEncoder,
        {"d_model": 32, "ffn_dim": 64, "num_layers": 2},
        id="autoformer",
    ),
]

HF_MODEL_CONFIGS = [
    pytest.param("patchtst_lf", id="patchtst_lf"),
    pytest.param("informer_lf", id="informer_lf"),
    pytest.param("tst_lf", id="tst_lf"),
    pytest.param("autoformer_lf", id="autoformer_lf"),
]


@pytest.mark.parametrize("encoder_cls,encoder_kwargs", HF_ENCODER_CASES)
def test_hf_temporal_encoder_shape(encoder_cls, encoder_kwargs):
    batch_size, seq_len, n_ts, embed_dim = 2, 64, 6, 128
    encoder = encoder_cls(
        in_dim=n_ts,
        embed_dim=embed_dim,
        seq_len=seq_len,
        num_attention_heads=4,
        **encoder_kwargs,
    )
    temporal = torch.randn(batch_size, seq_len, n_ts)
    doys = torch.randint(1, 366, (batch_size, seq_len))

    out = encoder(temporal, doys)
    assert out.shape == (batch_size, embed_dim)


@pytest.mark.parametrize("encoder_cls,encoder_kwargs", HF_ENCODER_CASES)
def test_hf_temporal_encoder_rejects_wrong_sequence_length(encoder_cls, encoder_kwargs):
    seq_len, n_ts, embed_dim = 64, 6, 128
    encoder = encoder_cls(
        in_dim=n_ts,
        embed_dim=embed_dim,
        seq_len=seq_len,
        num_attention_heads=4,
        **encoder_kwargs,
    )
    temporal = torch.randn(2, seq_len + 10, n_ts)
    doys = torch.randint(1, 366, (2, seq_len + 10))

    with pytest.raises(ValueError, match="Expected sequence length"):
        encoder(temporal, doys)


@pytest.mark.parametrize("encoder_cls,encoder_kwargs", HF_ENCODER_CASES)
def test_hf_late_fusion_forward_shape(encoder_cls, encoder_kwargs):
    batch_size, seq_len, n_ts, n_ctx, embed_dim = 2, 64, 6, 10, 128
    temporal_encoder = encoder_cls(
        in_dim=n_ts,
        embed_dim=embed_dim,
        seq_len=seq_len,
        num_attention_heads=4,
        **encoder_kwargs,
    )
    model = LateFusionNetwork(
        context_encoder=ContextEncoder(
            in_dim=n_ctx, hidden_dims=[embed_dim], embed_dim=embed_dim
        ),
        temporal_encoder=temporal_encoder,
        fusion=GatedFusion(
            temporal_dim=embed_dim,
            static_dim=embed_dim,
            out_dim=embed_dim,
        ),
        regression_head=RegressionHead(hidden_dims=[embed_dim, embed_dim], dropout=0.0),
        context_in_dim=n_ctx,
        temporal_in_dim=n_ts,
        embed_dim=embed_dim,
    )

    context = torch.randn(batch_size, n_ctx)
    temporal = torch.randn(batch_size, seq_len, n_ts)
    doys = torch.randint(1, 366, (batch_size, seq_len))

    out = model(context, temporal, doys)
    assert out.shape == (batch_size,)


@pytest.mark.parametrize("model_name", HF_MODEL_CONFIGS)
def test_hf_lf_fit_predict(model_name):
    with initialize(version_base=None, config_path="../../cybench/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset/crop=wheat",
                "dataset.country=NL",
                "dataset.framework=torch",
                "dataset.target.filter_samples=null",
                f"model={model_name}",
                "experiment.device=cpu",
            ],
        )

    test_cfg = copy.deepcopy(cfg)
    dataset = DataFactory(test_cfg.dataset).build()
    adjust_model_cfg_to_dataset(test_cfg.model, dataset)

    even_years = {year for year in dataset.years if year % 2 == 0}
    odd_years = dataset.years - even_years
    train_dataset, test_dataset = dataset.split_on_years(
        years_split=(even_years, odd_years)
    )

    model_cfg = remove_keys(test_cfg.model, "_search_")
    model = instantiate(model_cfg)

    model.fit(train_dataset, val_dataset=test_dataset, epochs=2)
    test_preds, _ = model.predict(test_dataset)
    assert test_preds.shape[0] == len(test_dataset)

    evaluation_result = evaluate_predictions(
        y_true=test_dataset.targets,
        y_pred=test_preds,
        cfg=test_cfg.evaluation,
    )
    for metric in ("normalized_rmse", "mape"):
        assert metric in evaluation_result
        assert not np.isnan(evaluation_result[metric])
