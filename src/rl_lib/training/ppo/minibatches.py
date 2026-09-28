"""Minibatch generation for temporal PPO rollouts."""

from collections.abc import Iterator

import numpy as np
import torch as T
from torch.profiler import record_function


def get_iid_minibatches(
    batch: dict[str, T.Tensor],
    minibatch_size: int,
    stack_size: int,
    shuffle: bool,
) -> Iterator[dict[str, T.Tensor]]:
    """Yield transition minibatches with their matching temporal windows.

    Samples transitions across environments and time, then reconstructs each
    transition's observation and done windows of length ``stack_size``.

    Args:
        batch: Rollout data. Observations and dones have leading dimensions
            (num_envs, sequence), while transition fields use
            (num_envs, batch).
        minibatch_size: Maximum number of transitions per yielded minibatch.
        stack_size: Temporal observation-window length.
        shuffle: Randomly permute transitions when true; otherwise preserve
            environment-major flattened order.

    Yields:
        Minibatch mappings with flattened observation and done windows and
        transition fields whose leading dimension is the minibatch size.
    """
    num_envs, batch_size, _ = batch["action"].shape
    device = batch["action"].device
    flat_order = (
        np.random.permutation(batch_size * num_envs)
        if shuffle
        else np.arange(batch_size * num_envs)
    )
    with record_function("transfer/minibatch_indices_to_device"):
        indices = T.as_tensor(flat_order, dtype=T.long, device=device)
    window_offsets = T.arange(stack_size, device=device)
    for index in range(0, batch_size * num_envs, minibatch_size):
        flat_indices = indices[index : index + minibatch_size]
        env_indices = flat_indices.remainder(num_envs)
        time_indices = flat_indices.div(num_envs, rounding_mode="floor")
        window_indices = time_indices[:, None] + window_offsets

        # Advanced indexing materializes new tensors: this is a device-side
        # copy of the whole observation stack, not a view, and it is the
        # dominant per-minibatch memory cost of the update loop.
        with record_function("transfer/minibatch_gather"):
            observation_windows = batch["observation"][
                env_indices[:, None], window_indices
            ]
            done_windows = batch["dones"][env_indices[:, None], window_indices]
        yield {
            "observation": observation_windows.reshape(
                -1, *batch["observation"].shape[2:]
            ),
            "action": batch["action"][env_indices, time_indices],
            "old_log_probs": batch["old_log_probs"][env_indices, time_indices],
            "old_values": batch["critic_value"][env_indices, time_indices],
            "returns": batch["returns"][env_indices, time_indices],
            "advantages": batch["advantages"][env_indices, time_indices],
            "dones": done_windows.reshape(-1),
        }
