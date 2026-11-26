"""Tests for CNN-only mixed precision."""

import pytest
import torch as T

from rl_lib.networks.factory import Network
from rl_lib.networks.mixed_precision import resolve_cnn_autocast_dtype


def test_resolve_cnn_autocast_dtype_defaults_to_fp32():
    assert resolve_cnn_autocast_dtype("none", "cpu") is None


def test_bf16_cnn_autocast_requires_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        resolve_cnn_autocast_dtype("bf16", "cpu")


@pytest.mark.skipif(
    not T.cuda.is_available() or not T.cuda.is_bf16_supported(),
    reason="requires CUDA BF16 support",
)
def test_bf16_cnn_features_are_cast_to_fp32_before_network_heads(
    network: Network,
):
    assert resolve_cnn_autocast_dtype("bf16", "cuda") == T.bfloat16
    fp32_network = network.cuda().eval()
    bf16_network = Network(
        observation_dim=network.observation_dim,
        action_dim=network.action_dim,
        stack_size=network.stack_size,
        config=network.cfg,
        cnn_autocast_dtype=T.bfloat16,
    ).cuda().eval()
    T.nn.Module.load_state_dict(bf16_network, fp32_network.state_dict())
    observations = T.randn(4, 1, 96, 96, device="cuda")

    with T.no_grad():
        features = bf16_network.feature_extract(observations)
        temporal = bf16_network.temporal_encode(features.reshape(2, 2, -1))
        action_dist, action_mean, values = bf16_network.heads(temporal, 1.0)
        log_probs = action_dist.log_prob(action_mean)

    assert features.dtype == T.float32
    assert action_mean.dtype == T.float32
    assert values.dtype == T.float32
    assert action_dist.base_dist.concentration1.dtype == T.float32
    assert log_probs.dtype == T.float32
    assert T.isfinite(log_probs).all()
