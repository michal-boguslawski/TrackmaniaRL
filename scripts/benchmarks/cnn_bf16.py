"""Compare FP32 PPO with CNN-only BF16 autocast on a fixed CUDA minibatch.

The benchmark reports action/value/loss/gradient differences, checks a short
series of PPO updates for finite losses and parameters, and measures repeated
minibatch update latency. It uses the packaged CarRacing network without
starting an environment or compiling the model.
"""

from __future__ import annotations

import argparse
from importlib import resources
import json
import math
from pathlib import Path
import sys
from typing import Callable

import torch as T

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl_lib.agent import Agent
from rl_lib.networks.factory import Network
from rl_lib.networks.mixed_precision import resolve_cnn_autocast_dtype
from rl_lib.run_config import AgentSettings, TrainerSettings, load_config
from rl_lib.training.ppo.trainer import PPOTrainer
from scripts.benchmarking import benchmark_callables


def _max_abs_diff(left: T.Tensor, right: T.Tensor) -> float:
    """Return the maximum absolute difference between two tensors."""
    return (left.float() - right.float()).abs().max().item()


def _gradient_summary(
    reference: list[T.Tensor], candidate: list[T.Tensor]
) -> dict[str, float]:
    """Summarize gradient norm difference and cosine similarity."""
    paired_gradients = [
        (reference_gradient, candidate_gradient)
        for reference_gradient, candidate_gradient in zip(reference, candidate, strict=True)
        if reference_gradient is not None and candidate_gradient is not None
    ]
    reference_flat = T.cat(
        [gradient.detach().float().flatten() for gradient, _ in paired_gradients]
    )
    candidate_flat = T.cat(
        [gradient.detach().float().flatten() for _, gradient in paired_gradients]
    )
    return {
        "fp32_l2_norm": reference_flat.norm().item(),
        "bf16_l2_norm": candidate_flat.norm().item(),
        "max_abs_diff": _max_abs_diff(reference_flat, candidate_flat),
        "cosine_similarity": T.nn.functional.cosine_similarity(
            reference_flat, candidate_flat, dim=0
        ).item(),
    }


def run_benchmark(
    batch_size: int = 128,
    warmups: int = 3,
    repetitions: int = 10,
    stability_steps: int = 10,
    seed: int = 123,
) -> dict[str, object]:
    """Run correctness, update-stability, and latency comparisons.

    Args:
        batch_size: Number of transitions in each PPO minibatch.
        warmups: Untimed update calls per precision mode.
        repetitions: Timed update calls per precision mode.
        stability_steps: Consecutive updates checked for finite state.
        seed: Seed for model initialization and synthetic input generation.

    Returns:
        JSON-serializable measurements and numerical comparison summaries.

    Raises:
        RuntimeError: If CUDA or BF16 support is unavailable, or a stability
            check encounters a non-finite loss or parameter.
    """
    if not T.cuda.is_available() or not T.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires a CUDA GPU with BF16 support")
    T.manual_seed(seed)
    device = T.device("cuda")
    config_path = resources.files("rl_lib.config") / "ppo_carracing.yaml"
    run_config = load_config(str(config_path))
    agent_config = AgentSettings(stack_size=run_config.agent.stack_size)

    fp32_network = Network(
        observation_dim=1,
        action_dim=3,
        stack_size=agent_config.stack_size,
        config=run_config.network,
    ).to(device)
    bf16_network = Network(
        observation_dim=1,
        action_dim=3,
        stack_size=agent_config.stack_size,
        config=run_config.network,
        cnn_autocast_dtype=resolve_cnn_autocast_dtype("bf16", device),
    ).to(device)
    T.nn.Module.load_state_dict(bf16_network, fp32_network.state_dict())
    for network in (fp32_network, bf16_network):
        network.cnn.to(memory_format=T.channels_last)

    fp32_agent = Agent(fp32_network, device, agent_config)
    bf16_agent = Agent(bf16_network, device, agent_config)
    fp32_trainer = PPOTrainer(fp32_agent, run_config.trainer, verbosity=0)
    bf16_trainer = PPOTrainer(bf16_agent, run_config.trainer, verbosity=0)

    sample_count = batch_size * agent_config.stack_size
    observations = T.randint(
        0, 256, (sample_count, 96, 96, 1), dtype=T.uint8, device=device
    )
    dones = T.zeros(sample_count, dtype=T.bool, device=device)
    with T.no_grad():
        fp32_log_probs, fp32_values, fp32_dist, fp32_means = fp32_agent.evaluate_actions(
            observations, T.full((batch_size, 3), 0.5, device=device), dones
        )
        actions = fp32_dist.sample()
        fp32_log_probs, fp32_values, fp32_dist, fp32_means = fp32_agent.evaluate_actions(
            observations, actions, dones
        )
        bf16_log_probs, bf16_values, bf16_dist, bf16_means = bf16_agent.evaluate_actions(
            observations, actions, dones
        )

    act_observations = observations[:batch_size]
    act_dones = T.zeros(batch_size, dtype=T.bool, device=device)
    fp32_agent.reset()
    bf16_agent.reset()
    fp32_actions, _, fp32_act_values = fp32_agent.act(
        act_observations, act_dones, temperature=0.0
    )
    bf16_actions, _, bf16_act_values = bf16_agent.act(
        act_observations, act_dones, temperature=0.0
    )

    batch = {
        "observation": observations,
        "action": actions,
        "old_log_probs": fp32_log_probs.detach(),
        "old_values": fp32_values.detach(),
        "returns": fp32_values.detach() + T.randn_like(fp32_values) * 0.25,
        "advantages": T.randn(batch_size, device=device),
        "dones": dones,
    }
    fp32_losses, _ = fp32_trainer.calculate_losses(**batch)
    bf16_losses, _ = bf16_trainer.calculate_losses(**batch)
    fp32_total_loss = (
        fp32_losses[0]
        + run_config.trainer.critic_beta * fp32_losses[1]
        - run_config.trainer.entropy_coef * fp32_losses[2]
    )
    bf16_total_loss = (
        bf16_losses[0]
        + run_config.trainer.critic_beta * bf16_losses[1]
        - run_config.trainer.entropy_coef * bf16_losses[2]
    )
    fp32_total_loss.backward()
    bf16_total_loss.backward()
    fp32_gradients = [parameter.grad for parameter in fp32_network.parameters()]
    bf16_gradients = [parameter.grad for parameter in bf16_network.parameters()]
    gradient_comparison = _gradient_summary(fp32_gradients, bf16_gradients)
    fp32_trainer._optimizer.zero_grad(set_to_none=True)
    bf16_trainer._optimizer.zero_grad(set_to_none=True)

    fp32_stability_losses = []
    bf16_stability_losses = []
    for _ in range(stability_steps):
        for trainer, network, history in (
            (fp32_trainer, fp32_network, fp32_stability_losses),
            (bf16_trainer, bf16_network, bf16_stability_losses),
        ):
            metrics = trainer.train_step(**batch)
            loss_value = (
                metrics.get("metrics/actor_loss", math.nan)
                + run_config.trainer.critic_beta * metrics.get("loss/critic", math.nan)
                - run_config.trainer.entropy_coef * metrics.get("loss/entropy", math.nan)
            )
            if not math.isfinite(loss_value) or not all(
                T.isfinite(parameter).all().item() for parameter in network.parameters()
            ):
                raise RuntimeError(f"Non-finite PPO update in {network.cnn_autocast_dtype}")
            history.append(loss_value)

    # Restart both optimizers and networks from the same state for a fair timing.
    initial_state = {
        key: value.detach().clone() for key, value in fp32_network.state_dict().items()
    }
    timed_trainers = {}
    for name, network, agent in (
        ("fp32", fp32_network, fp32_agent),
        ("cnn_bf16", bf16_network, bf16_agent),
    ):
        T.nn.Module.load_state_dict(network, initial_state)
        timed_trainers[name] = PPOTrainer(agent, run_config.trainer, verbosity=0)

    def update(trainer: PPOTrainer) -> Callable[[], object]:
        """Create a closure for a single fixed-minibatch PPO update."""
        return lambda: trainer.train_step(**batch)

    prepared_observations = fp32_agent._preprocess_observation(observations)

    def cnn_forward(network: Network) -> Callable[[], object]:
        """Create an inference-only CNN closure over preprocessed input."""
        def operation() -> T.Tensor:
            with T.no_grad():
                return network.feature_extract(prepared_observations)

        return operation

    operations = {
        "fp32_cnn": cnn_forward(fp32_network),
        "cnn_bf16_cnn": cnn_forward(bf16_network),
        "fp32_update": update(timed_trainers["fp32"]),
        "cnn_bf16_update": update(timed_trainers["cnn_bf16"]),
    }
    timings = benchmark_callables(
        operations,
        warmups=warmups,
        repetitions=repetitions,
        device=device,
    )
    return {
        "environment": {
            "gpu": T.cuda.get_device_name(device),
            "torch": T.__version__,
            "cuda": T.version.cuda,
            "batch_size": batch_size,
            "stack_size": agent_config.stack_size,
            "observation": "96x96x1 uint8 NHWC",
            "warmups": warmups,
            "repetitions": repetitions,
            "stability_steps": stability_steps,
        },
        "numerics": {
            "action_max_abs_diff": _max_abs_diff(fp32_actions, bf16_actions),
            "act_value_max_abs_diff": _max_abs_diff(fp32_act_values, bf16_act_values),
            "evaluate_action_mean_max_abs_diff": _max_abs_diff(fp32_means, bf16_means),
            "evaluate_value_max_abs_diff": _max_abs_diff(fp32_values, bf16_values),
            "log_prob_max_abs_diff": _max_abs_diff(fp32_log_probs, bf16_log_probs),
            "loss_components_fp32": [loss.item() for loss in fp32_losses],
            "loss_components_bf16": [loss.item() for loss in bf16_losses],
            "total_loss_max_abs_diff": _max_abs_diff(fp32_total_loss, bf16_total_loss),
            "gradient": gradient_comparison,
        },
        "stability": {
            "fp32_loss_finite": all(math.isfinite(value) for value in fp32_stability_losses),
            "cnn_bf16_loss_finite": all(math.isfinite(value) for value in bf16_stability_losses),
            "fp32_losses": fp32_stability_losses,
            "cnn_bf16_losses": bf16_stability_losses,
        },
        "timings": {
            name: result.to_dict() for name, result in timings.items()
        },
        "speedup": {
            "cnn_forward": timings["fp32_cnn"].speedup_over(timings["cnn_bf16_cnn"]),
            "ppo_update": timings["cnn_bf16_update"].speedup_over(timings["fp32_update"]),
        },
    }


def main() -> None:
    """Parse benchmark options and print the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--stability-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    arguments = parser.parse_args()
    print(json.dumps(run_benchmark(**vars(arguments)), indent=2))


if __name__ == "__main__":
    main()
