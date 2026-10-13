"""Measure how much value information the frozen policy features retain.

Collects complete stochastic-policy episodes using the saved training
configuration, records CNN and temporal features, and compares the checkpoint's
critic with ridge and MLP probes on discounted reward-to-go. Episode-level
splits prevent adjacent frames from the same trajectory leaking across splits.

Example
-------
uv run --extra gpu python scripts/benchmarks/representation_probe.py \\
  --config logs/artifacts/happy-grizzly-20261010T060841Z-e77714/config/run_config.yaml \\
  --checkpoint logs/artifacts/happy-grizzly-20261010T060841Z-e77714/final_state_dict.pt \\
  --episodes 200
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch as T
from torch import nn
from tqdm import tqdm
import yaml

from rl_lib.agent import Agent
from rl_lib.checkpoint_loaders import LocalCheckpointLoader
from rl_lib.device import resolve_device
from rl_lib.envs.make_env import make_env
from rl_lib.networks.factory import Network
from rl_lib.networks.mixed_precision import resolve_cnn_autocast_dtype
from rl_lib.run_config import RunConfig


@dataclass(slots=True)
class EpisodeData:
    """Feature and target samples from one complete environment episode.

    Attributes:
        cnn_features: Per-frame CNN features, shape (episode_length, cnn_dim).
        temporal_features: Encoded histories, shape
            (episode_length, temporal_dim).
        critic_values: Checkpoint critic estimates, shape (episode_length,).
        targets: Discounted reward-to-go, shape (episode_length,).
        episode_return: Sum of collected (possibly normalized) rewards.
        length: Number of environment transitions in the episode.
    """

    cnn_features: NDArray[np.float32]
    temporal_features: NDArray[np.float32]
    critic_values: NDArray[np.float32]
    targets: NDArray[np.float32]
    episode_return: float
    length: int


def discounted_returns(
    rewards: Sequence[float], gamma: float, bootstrap_value: float = 0.0
) -> NDArray[np.float32]:
    """Compute discounted reward-to-go for a complete trajectory.

    Args:
        rewards: Collected rewards in chronological order.
        gamma: Discount factor in [0, 1].
        bootstrap_value: Value used after the last reward, such as the critic
            estimate for a time-limit truncation; use zero for termination.

    Returns:
        Float32 array of discounted returns with shape (episode_length,).
    """
    returns = np.empty(len(rewards), dtype=np.float32)
    running_return = float(bootstrap_value)
    for index in range(len(rewards) - 1, -1, -1):
        running_return = float(rewards[index]) + gamma * running_return
        returns[index] = running_return
    return returns


def split_episode_indices(
    episode_count: int,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Split whole episodes into train, validation, and test partitions.

    Args:
        episode_count: Number of complete episodes in the dataset.
        seed: Seed for deterministic episode shuffling.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.

    Returns:
        Three arrays of episode indices; no episode appears in multiple splits.

    Raises:
        ValueError: If the split fractions or episode count are invalid.
    """
    if episode_count < 4:
        raise ValueError("at least four complete episodes are required")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    shuffled = np.random.default_rng(seed).permutation(episode_count)
    train_count = min(max(1, int(episode_count * train_fraction)), episode_count - 2)
    validation_count = min(
        max(1, int(episode_count * validation_fraction)),
        episode_count - train_count - 1,
    )
    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + validation_count],
        shuffled[train_count + validation_count :],
    )


def explained_variance(
    targets: NDArray[np.floating[Any]], predictions: NDArray[np.floating[Any]]
) -> float:
    """Compute explained variance using the rollout metric convention.

    Args:
        targets: Reference values with shape (samples,).
        predictions: Predictions with shape (samples,).

    Returns:
        ``1 - Var(targets - predictions) / Var(targets)``.
    """
    target_variance = float(np.var(targets))
    if target_variance <= np.finfo(np.float64).eps:
        return float("nan")
    return float(1.0 - np.var(targets - predictions) / target_variance)


def _prediction_metrics(
    targets: NDArray[np.float32], predictions: NDArray[np.float32]
) -> dict[str, float]:
    """Summarize prediction error in target units."""
    errors = targets - predictions
    return {
        "explained_variance": explained_variance(targets, predictions),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_error": float(np.mean(errors)),
    }


def _fit_affine_critic_calibration(
    train_values: NDArray[np.float32],
    train_targets: NDArray[np.float32],
    test_values: NDArray[np.float32],
    test_targets: NDArray[np.float32],
) -> dict[str, Any]:
    """Fit a scale and offset on training episodes; score once on test episodes."""
    design = np.column_stack(
        (train_values.astype(np.float64, copy=False), np.ones(train_values.size))
    )
    scale, offset = np.linalg.lstsq(
        design, train_targets.astype(np.float64, copy=False), rcond=None
    )[0]
    train_predictions = (scale * train_values + offset).astype(np.float32)
    test_predictions = (scale * test_values + offset).astype(np.float32)
    return {
        "scale": float(scale),
        "offset": float(offset),
        "fit_split": "training episodes only",
        "train": _prediction_metrics(train_targets, train_predictions),
        "test": _prediction_metrics(test_targets, test_predictions),
    }


def _load_artifact_config(path: str | Path) -> RunConfig:
    """Load a saved run config after removing resolved runtime metadata."""
    with Path(path).open("rt", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    raw.pop("runtime", None)
    return RunConfig.model_validate(raw)


def _episode_targets(
    rewards: list[float], gamma: float, bootstrap_value: float
) -> NDArray[np.float32]:
    """Build discounted targets for a collected episode."""
    return discounted_returns(rewards, gamma, bootstrap_value)


def _make_episode_data(
    cnn_features: list[NDArray[np.float32]],
    temporal_features: list[NDArray[np.float32]],
    critic_values: list[float],
    rewards: list[float],
    gamma: float,
    bootstrap_value: float,
) -> EpisodeData:
    """Materialize one completed trajectory as compact arrays."""
    if not rewards:
        raise ValueError("cannot create a dataset episode with no transitions")
    return EpisodeData(
        cnn_features=np.stack(cnn_features).astype(np.float32, copy=False),
        temporal_features=np.stack(temporal_features).astype(np.float32, copy=False),
        critic_values=np.asarray(critic_values, dtype=np.float32),
        targets=_episode_targets(rewards, gamma, bootstrap_value),
        episode_return=float(np.sum(rewards, dtype=np.float64)),
        length=len(rewards),
    )


def _flatten_episodes(
    episodes: list[EpisodeData], indices: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Concatenate selected episodes into CNN, temporal, value, and target arrays."""
    selected = [episodes[int(index)] for index in indices]
    return (
        np.concatenate([episode.cnn_features for episode in selected]),
        np.concatenate([episode.temporal_features for episode in selected]),
        np.concatenate([episode.critic_values for episode in selected]),
        np.concatenate([episode.targets for episode in selected]),
    )


def _ridge_probe(
    train_x: NDArray[np.float32],
    train_y: NDArray[np.float32],
    validation_x: NDArray[np.float32],
    validation_y: NDArray[np.float32],
    test_x: NDArray[np.float32],
    test_y: NDArray[np.float32],
    device: T.device,
) -> dict[str, float | int]:
    """Fit a standardized ridge probe, choosing regularization on validation."""
    x_train = T.as_tensor(train_x, device=device, dtype=T.float32)
    y_train = T.as_tensor(train_y, device=device, dtype=T.float32)
    x_mean = x_train.mean(dim=0)
    x_scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_scale = y_train.std(unbiased=False).clamp_min(1e-6)

    x_train = (x_train - x_mean) / x_scale
    y_train = (y_train - y_mean) / y_scale
    gram = x_train.T @ x_train / x_train.shape[0]
    cross = x_train.T @ y_train / x_train.shape[0]
    identity = T.eye(gram.shape[0], device=device, dtype=gram.dtype)
    x_validation = (T.as_tensor(validation_x, device=device) - x_mean) / x_scale
    x_test = (T.as_tensor(test_x, device=device) - x_mean) / x_scale
    y_validation_np = validation_y
    y_test_np = test_y

    candidates = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    best_alpha = candidates[0]
    best_validation_ev = -float("inf")
    best_weights: T.Tensor | None = None
    for alpha in candidates:
        weights = T.linalg.solve(gram + alpha * identity, cross)
        validation_predictions = (
            x_validation @ weights * y_scale + y_mean
        ).detach().cpu().numpy()
        validation_ev = explained_variance(y_validation_np, validation_predictions)
        if validation_ev > best_validation_ev:
            best_alpha = alpha
            best_validation_ev = validation_ev
            best_weights = weights

    assert best_weights is not None
    train_predictions = (x_train @ best_weights * y_scale + y_mean).detach().cpu().numpy()
    test_predictions = (x_test @ best_weights * y_scale + y_mean).detach().cpu().numpy()
    return {
        "alpha": best_alpha,
        "train_explained_variance": explained_variance(train_y, train_predictions),
        "validation_explained_variance": best_validation_ev,
        "test_explained_variance": explained_variance(y_test_np, test_predictions),
    }


def _mlp_probe(
    train_x: NDArray[np.float32],
    train_y: NDArray[np.float32],
    validation_x: NDArray[np.float32],
    validation_y: NDArray[np.float32],
    test_x: NDArray[np.float32],
    test_y: NDArray[np.float32],
    device: T.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> dict[str, float | int]:
    """Fit a two-layer MLP with validation-based early stopping."""
    T.manual_seed(seed)
    x_train = T.as_tensor(train_x, device=device, dtype=T.float32)
    y_train = T.as_tensor(train_y, device=device, dtype=T.float32).unsqueeze(-1)
    x_validation = T.as_tensor(validation_x, device=device, dtype=T.float32)
    y_validation = T.as_tensor(validation_y, device=device, dtype=T.float32).unsqueeze(-1)
    x_test = T.as_tensor(test_x, device=device, dtype=T.float32)
    y_test = T.as_tensor(test_y, device=device, dtype=T.float32).unsqueeze(-1)

    x_mean = x_train.mean(dim=0)
    x_scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean()
    y_scale = y_train.std(unbiased=False).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_scale
    x_validation = (x_validation - x_mean) / x_scale
    x_test = (x_test - x_mean) / x_scale
    y_train = (y_train - y_mean) / y_scale
    y_validation_normalized = (y_validation - y_mean) / y_scale

    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 256),
        nn.GELU(),
        nn.Linear(256, 1),
    ).to(device)
    optimizer = T.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_validation_loss = float("inf")
    best_state: dict[str, T.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        permutation = T.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            predictions = model(x_train[batch_indices])
            loss = T.nn.functional.mse_loss(predictions, y_train[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with T.no_grad():
            validation_loss = float(
                T.nn.functional.mse_loss(
                    model(x_validation), y_validation_normalized
                ).item()
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP probe did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with T.no_grad():
        train_predictions = (model(x_train).squeeze(-1) * y_scale + y_mean).cpu().numpy()
        validation_predictions = (
            model(x_validation).squeeze(-1) * y_scale + y_mean
        ).cpu().numpy()
        test_predictions = (model(x_test).squeeze(-1) * y_scale + y_mean).cpu().numpy()

    return {
        "best_epoch": best_epoch,
        "train_explained_variance": explained_variance(
            train_y, train_predictions
        ),
        "validation_explained_variance": explained_variance(
            validation_y, validation_predictions
        ),
        "test_explained_variance": explained_variance(test_y, test_predictions),
    }


def _collect_episodes(
    agent: Agent,
    network: Network,
    environment: Any,
    episodes_requested: int,
    seed: int,
    gamma: float,
    max_vector_steps: int,
) -> list[EpisodeData]:
    """Collect complete episodes and record the features used by the policy."""
    captured: dict[str, T.Tensor] = {}

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[T.Tensor, ...], output: T.Tensor) -> None:
            captured[name] = output

        return hook

    hooks = [
        network.cnn.register_forward_hook(capture("cnn")),
        network.sequence_encoder.register_forward_hook(capture("temporal")),
    ]
    num_envs = environment.num_envs
    pending_cnn: list[list[NDArray[np.float32]]] = [[] for _ in range(num_envs)]
    pending_temporal: list[list[NDArray[np.float32]]] = [[] for _ in range(num_envs)]
    pending_values: list[list[float]] = [[] for _ in range(num_envs)]
    pending_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    bootstrap_values = np.zeros(num_envs, dtype=np.float32)
    completed: list[EpisodeData] = []

    try:
        state, _ = environment.reset(seed=seed)
        done = T.zeros(num_envs, dtype=T.bool, device=agent.device)
        for _step in tqdm(
            range(max_vector_steps),
            desc="Collecting probe episodes",
            mininterval=5.0,
        ):
            captured.clear()
            (
                next_state,
                _state_tensor,
                _action,
                _log_probs,
                values,
                rewards,
                terminated,
                truncated,
                done,
                info,
            ) = agent.step_env(
                environment,
                state,
                done,
                temperature=agent.cfg.action_temperature,
            )

            if "cnn" not in captured or "temporal" not in captured:
                raise RuntimeError("feature hooks did not observe the policy forward pass")
            # One compact transfer keeps collection from synchronizing once per feature.
            feature_payload = T.cat(
                (
                    captured["cnn"].detach(),
                    captured["temporal"].detach(),
                    values.detach().unsqueeze(-1),
                ),
                dim=-1,
            ).float().cpu().numpy()
            cnn_dim = captured["cnn"].shape[-1]
            temporal_dim = captured["temporal"].shape[-1]
            values_np = feature_payload[:, cnn_dim + temporal_dim]
            terminated_np = terminated.detach().cpu().numpy().astype(np.bool_)
            truncated_np = truncated.detach().cpu().numpy().astype(np.bool_)

            truncation_indices = np.flatnonzero(truncated_np)
            bootstrap_values.fill(0.0)
            if truncation_indices.size:
                final_observations = info.get("final_obs", info.get("final_observation"))
                if final_observations is None:
                    raise RuntimeError(
                        "truncated episode did not provide final observations for bootstrap"
                    )
                final_mask = info.get("_final_obs", info.get("_final_observation"))
                if final_mask is not None and not np.asarray(final_mask, dtype=np.bool_)[
                    truncation_indices
                ].all():
                    raise RuntimeError(
                        "vector environment omitted a final observation for a truncated episode"
                    )
                bootstrap_observations = next_state.copy()
                for env_index in truncation_indices:
                    # Gymnasium 1.3 SAME_STEP returns final_obs as a batched
                    # object array, so assign each image to preserve NHWC dtype.
                    bootstrap_observations[env_index] = np.asarray(
                        final_observations[env_index]
                    )
                bootstrap_tensor = T.from_numpy(bootstrap_observations).to(agent.device)
                with T.no_grad():
                    truncation_values = agent.bootstrap_value(bootstrap_tensor)
                bootstrap_values[truncation_indices] = truncation_values[
                    truncation_indices
                ].detach().cpu().numpy()

            for env_index in range(num_envs):
                pending_cnn[env_index].append(
                    feature_payload[env_index, :cnn_dim].copy()
                )
                pending_temporal[env_index].append(
                    feature_payload[env_index, cnn_dim : cnn_dim + temporal_dim].copy()
                )
                pending_values[env_index].append(float(values_np[env_index]))
                pending_rewards[env_index].append(float(rewards[env_index]))

                if not (terminated_np[env_index] or truncated_np[env_index]):
                    continue
                bootstrap = (
                    0.0
                    if terminated_np[env_index]
                    else float(bootstrap_values[env_index])
                )
                completed.append(
                    _make_episode_data(
                        pending_cnn[env_index],
                        pending_temporal[env_index],
                        pending_values[env_index],
                        pending_rewards[env_index],
                        gamma,
                        bootstrap,
                    )
                )
                pending_cnn[env_index] = []
                pending_temporal[env_index] = []
                pending_values[env_index] = []
                pending_rewards[env_index] = []
                if len(completed) >= episodes_requested:
                    break

            if len(completed) >= episodes_requested:
                break
            state = next_state
        else:
            raise RuntimeError(
                f"collected only {len(completed)} complete episodes in "
                f"{max_vector_steps} vector steps; increase --max-vector-steps"
            )
    finally:
        for hook in hooks:
            hook.remove()

    return completed[:episodes_requested]


def run_probe(
    config_path: str | Path,
    checkpoint_path: str | Path,
    episodes_requested: int = 200,
    num_envs: int | None = None,
    seed: int | None = None,
    device_override: str | None = None,
    output_path: str | Path | None = None,
    mlp_epochs: int = 100,
    mlp_patience: int = 12,
    batch_size: int = 1024,
    max_vector_steps: int = 500_000,
) -> dict[str, Any]:
    """Collect episodes and fit frozen-feature value probes.

    Args:
        config_path: Saved run config YAML; resolved ``runtime`` data is ignored.
        checkpoint_path: Local network state-dict checkpoint.
        episodes_requested: Number of complete episodes to collect.
        num_envs: Vector environment count, defaulting to the saved run value.
        seed: Collection seed, defaulting to the saved run seed plus 10,000.
        device_override: Optional PyTorch device override.
        output_path: Optional JSON file to write the report to.
        mlp_epochs: Maximum MLP training epochs.
        mlp_patience: Early-stopping epochs without validation improvement.
        batch_size: MLP minibatch size.
        max_vector_steps: Safety limit on vector environment iterations.

    Returns:
        JSON-serializable metadata and train/validation/test explained variance.

    Raises:
        ValueError: If requested counts or training settings are invalid.
    """
    if episodes_requested < 10:
        raise ValueError("at least 10 episodes are required for episode-level splits")
    if mlp_epochs < 1 or mlp_patience < 1 or batch_size < 1:
        raise ValueError("MLP epochs, patience, and batch size must be positive")
    if max_vector_steps < 1:
        raise ValueError("max_vector_steps must be positive")
    if num_envs is not None and num_envs < 1:
        raise ValueError("num_envs must be positive")

    config = _load_artifact_config(config_path)
    if device_override is not None:
        config = config.model_copy(
            update={"run": config.run.model_copy(update={"device": device_override})}
        )
    device = resolve_device(config)
    collection_seed = config.run.seed + 10_000 if seed is None else seed
    environment_config = config.environment.model_copy(
        update={"num_envs": num_envs or config.environment.num_envs}
    )
    T.manual_seed(collection_seed)
    np.random.seed(collection_seed)

    with closing(make_env(environment_config)) as environment:
        network = Network(
            observation_dim=environment.observation_space.shape[-1],
            action_dim=environment.action_space.shape[-1],
            stack_size=config.agent.stack_size,
            config=config.network,
            cnn_autocast_dtype=resolve_cnn_autocast_dtype(
                config.run.cnn_autocast, device
            ),
        ).to(device)
        if device.type == "cuda":
            T.backends.cudnn.benchmark = config.run.cudnn_benchmark
            if config.run.channels_last:
                network.cnn.to(memory_format=T.channels_last)
        network = LocalCheckpointLoader(checkpoint_path).load(network, device)
        network.eval()
        agent = Agent(network, device, config.agent)
        agent.eval()
        collected = _collect_episodes(
            agent,
            network,
            environment,
            episodes_requested,
            collection_seed,
            config.rollout.gamma,
            max_vector_steps,
        )

    train_indices, validation_indices, test_indices = split_episode_indices(
        len(collected), collection_seed
    )
    train_cnn, train_temporal, train_values, train_targets = _flatten_episodes(
        collected, train_indices
    )
    validation_cnn, validation_temporal, _validation_values, validation_targets = (
        _flatten_episodes(collected, validation_indices)
    )
    test_cnn, test_temporal, test_values, test_targets = _flatten_episodes(
        collected, test_indices
    )

    feature_reports: dict[str, Any] = {}
    for name, train_x, validation_x, test_x in (
        ("cnn", train_cnn, validation_cnn, test_cnn),
        ("temporal", train_temporal, validation_temporal, test_temporal),
    ):
        feature_reports[name] = {
            "dimension": int(train_x.shape[1]),
            "ridge": _ridge_probe(
                train_x,
                train_targets,
                validation_x,
                validation_targets,
                test_x,
                test_targets,
                device,
            ),
            "mlp": _mlp_probe(
                train_x,
                train_targets,
                validation_x,
                validation_targets,
                test_x,
                test_targets,
                device,
                collection_seed,
                mlp_epochs,
                mlp_patience,
                batch_size,
            ),
        }

    calibrated_critic = _fit_affine_critic_calibration(
        train_values, train_targets, test_values, test_targets
    )
    normalizer_note = (
        "Targets use a newly initialized reward normalizer, so their scale can "
        "differ from the training run."
        if environment_config.normalize_rewards
        else "Targets use unnormalized environment rewards."
    )
    report: dict[str, Any] = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "seed": collection_seed,
        "num_envs": environment_config.num_envs,
        "action_temperature": config.agent.action_temperature,
        "gamma": config.rollout.gamma,
        "reward_normalization": environment_config.normalize_rewards,
        "target": (
            "discounted reward-to-go from collected environment rewards; "
            "time-limit truncations bootstrap from the checkpoint critic"
        ),
        "episodes": {
            "total": len(collected),
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
            "mean_length": float(np.mean([episode.length for episode in collected])),
            "mean_collected_return": float(
                np.mean([episode.episode_return for episode in collected])
            ),
            "samples": {
                "train": int(train_targets.size),
                "validation": int(validation_targets.size),
                "test": int(test_targets.size),
            },
        },
        "features": feature_reports,
        "checkpoint_critic": {
            "input": "temporal features",
            "raw_test": _prediction_metrics(test_targets, test_values),
            "affine_calibration": calibrated_critic,
        },
        "interpretation_note": (
            "The critic's affine calibration is fit on training episodes only; "
            "test episodes are untouched. Explained variance is insensitive "
            "to a constant offset, while RMSE and mean_error show offset effects. "
            "Targets are Monte Carlo returns under the frozen policy. "
            f"{normalizer_note} Their noise and scale differ from PPO's rollout GAE targets. "
            "A weak critic score alone is not evidence of weak features."
        ),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Saved training run_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Saved network state-dict checkpoint")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", dest="device_override")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-vector-steps", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    """Run the feature representation probe and print its JSON report."""
    args = _parse_args()
    report = run_probe(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        episodes_requested=args.episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        device_override=args.device_override,
        output_path=args.output_path,
        mlp_epochs=args.mlp_epochs,
        mlp_patience=args.mlp_patience,
        batch_size=args.batch_size,
        max_vector_steps=args.max_vector_steps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
