"""Agent: manages policy evaluation and temporal history for vectorized environments.

The Agent wraps a Network and maintains per-environment observation/done windows
to construct fixed-length temporal sequences. It handles observation preprocessing
(NHWC uint8 -> NCHW float32), temporal encoding with episode-boundary masking,
and action sampling from the Beta-distribution actor.
"""

from collections import deque
from logging import getLogger
from typing import Any

import torch as T
from gymnasium import Env
from numpy.typing import NDArray
from torch import nn
from torch.distributions import Distribution
from torch.nn.parameter import Parameter
from torch.profiler import record_function

from rl_lib.checkpoint_loaders import LocalCheckpointLoader
from rl_lib.networks.actor import Actor
from rl_lib.networks.encoder import Encoder
from rl_lib.networks.factory import Network
from rl_lib.policy.history import History
from rl_lib.policy.masks import episode_mask
from rl_lib.run_config import AgentSettings
from rl_lib.training.transitions import Transition

logger = getLogger(__name__)


class Agent:
    """Policy evaluation with temporal context for vectorized environments.

    Maintains per-environment rolling windows of observations and done flags
    to feed fixed-length sequences into the temporal encoder. Handles the
    NHWC->NCHW conversion, observation normalization, and Beta-distribution
    action sampling with affine transform to environment action space.

    Attributes:
        cfg (AgentSettings): Runtime configuration (stack_size, temperature, etc.)
        network (Network): The underlying policy/value network.
        device (torch.device): Compute device for tensors.
        stack_size (int): Number of frames in the temporal window.
        last_step_had_truncation (bool): Whether the latest environment step
            truncated at least one environment.
    """

    def __init__(
        self,
        encoder: Encoder,
        actor: Actor,
        # network: Network,
        device: T.device | str,
        config: AgentSettings,
    ):
        """Initialize the agent.

        Args:
            # TODO
            device: Target device for tensor operations.
            config: Agent configuration including stack_size and preprocessing params.

        Raises:
            ValueError: If config.stack_size != network.stack_size.
        """
        if config.stack_size != encoder.stack_size:
            raise ValueError(
                f"agent stack_size ({config.stack_size}) must match the encoder stack_size ({encoder.stack_size})"
            )
        self.cfg = config
        self._stack_size = config.stack_size
        self._device = T.device(device)
        self._obs_window: deque[T.Tensor] = deque(maxlen=self._stack_size)
        self._done_window: deque[T.Tensor] = deque(maxlen=self._stack_size)
        self._encoder = encoder
        self._actor = actor
        self._history = History(max_length=self._stack_size)
        # Cached 0-dim tensors so observation normalization is one addcdiv
        # kernel (offset + x / divisor) instead of separate div and add.
        self._observation_divisor = T.tensor(
            config.observation_divisor, device=self._device, dtype=T.float32
        )
        self._observation_offset = T.tensor(
            config.observation_offset, device=self._device, dtype=T.float32
        )

    @property
    def network(self) -> Network:
        """Underlying policy/value network."""
        return self._network

    @property
    def device(self) -> T.device:
        """Compute device for tensor operations."""
        return self._device

    def _preprocess_observation(self, observation: T.Tensor) -> T.Tensor:
        """Convert raw NHWC uint8 observation to normalized NCHW float32.

        Args:
            observation: Tensor of shape (batch, height, width, channel), dtype uint8.

        Returns:
            Tensor of shape (batch, channel, height, width), dtype float32,
            normalized to [-1, 1] via (x / divisor + offset).
        """
        assert observation.dtype == T.uint8
        with record_function("transfer/observation_normalize"):
            observation_tensor = observation.to(self._device, T.float32)
            # addcdiv computes offset + observation / divisor elementwise with
            # the same rounding as the previous separate div/add kernels.
            observation_tensor = T.addcdiv(
                self._observation_offset, observation_tensor, self._observation_divisor
            )
        with record_function("agent/permute_to_nchw"):
            return observation_tensor.permute(0, 3, 1, 2)

    def act(
        self,
        observation: T.Tensor,
        done: T.Tensor,
        temperature: float | None = None,
    ) -> tuple[T.Tensor, dict[str, T.Tensor]]:
        """Sample action, log-prob, and value for a batch of observations.

        Args:
            observation: Tensor of shape (batch, height, width, channel), uint8.
            done: Boolean tensor of shape (batch,) indicating episode termination.
            temperature: Sampling temperature. Defaults to config.action_temperature.
                Use 0.0 for deterministic (mean) action.

        Returns:
            Tuple of:
            - action: Tensor of shape (batch, action_dim) in environment action space.
            - log_probs: Tensor of shape (batch,) log-probability of sampled action.
            - value: Tensor of shape (batch,) state value estimate.
        """
        with T.no_grad():
            temperature = (
                self.cfg.action_temperature if temperature is None else temperature
            )
            features = self._encoder.frame_features(observation)
            self._history.append(features, done)
            windowed_features, windowed_done = self._history.peek()

            temporal = self._encoder.aggregate(
                windowed_features,
                episode_mask(windowed_done),
            )

            _, action, value, log_probs = self.heads(temporal, temperature)

        extras = {
            "log_probs": log_probs,
            "value": value,
        }
        return (action, extras)

    def evaluate_actions(
        self, observation: T.Tensor, action: T.Tensor, dones: T.Tensor | None = None
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor, Distribution]:
        """Evaluate log-probs, values, and distribution for given observations/actions.

        Used during PPO policy update. Constructs temporal windows by unfolding
        the batch dimension (assumes data is already flattened by rollout collector).

        Args:
            observation: Tensor of shape (batch * seq, height, width, channel).
            action: Tensor of shape (batch * seq, action_dim).
            dones: Boolean tensor of shape (batch * seq,) or None.

        Returns:
            Tuple of (log_probs, values, action_dist, action_mean) where:
            - log_probs: (batch * seq,)
            - values: (batch * seq,)
            - action_dist: Beta-based TransformedDistribution
            - action_mean: (batch * seq, action_dim)
        """
        temporal_encoding = self._encoder(
            observation, episode_mask(dones) if dones is not None else None
        )

        action_dist, action_mean, values, log_probs = self.heads(
            temporal_encoding
        )  # TODO

        return log_probs, values, action_dist, action_mean

    @property
    def stack_size(self) -> int:
        """Number of frames in the temporal context window."""
        return self._stack_size

    def eval(self) -> None:
        """Set network to evaluation mode (disables dropout, etc.)."""
        self._network.eval()

    def train(self) -> None:
        """Set network to training mode."""
        self._network.train()

    def network_parameter_groups(self) -> dict[str, list[Parameter]]:
        """Group parameters by network component and weight-decay eligibility.

        Groups:
        - cnn_decay: CNN weights (with weight decay)
        - cnn_no_decay: CNN biases/norms (no weight decay)
        - sequence_encoder_decay: Sequence encoder weights (with weight decay)
        - sequence_encoder_no_decay: Sequence encoder biases/norms (no weight decay)
        - head_decay: Actor/critic weights (with weight decay)
        - head_no_decay: Actor/critic biases + log_std params (no weight decay)

        Returns:
            Dict mapping group name to list of parameters.
        """
        component_groups: dict[str, list[Parameter]] = {}
        for component, module in (
            ("cnn", self._network.cnn),
            ("sequence_encoder", self._network.sequence_encoder),
        ):
            decay: list[Parameter] = []
            no_decay: list[Parameter] = []
            for name, p in module.named_parameters():
                (
                    no_decay if "bias" in name or "norm" in name.lower() else decay
                ).append(p)
            component_groups[f"{component}_decay"] = decay
            component_groups[f"{component}_no_decay"] = no_decay

        head_decay, head_no_decay = [], []
        for module in [self._network.actor, self._network.critic]:
            for name, p in module.named_parameters():
                (
                    head_no_decay
                    if "bias" in name or "_log_std" in name
                    else head_decay
                ).append(p)

        return {
            **component_groups,
            "head_decay": head_decay,
            "head_no_decay": head_no_decay,
        }

    def clip_grad_norm(self, max_norm: float) -> T.Tensor:
        """Clip gradient norm of all network parameters.

        Args:
            max_norm: Maximum allowed gradient norm.

        Returns:
            Total norm of gradients before clipping.
        """
        return nn.utils.clip_grad_norm_(self._network.parameters(), max_norm)

    def clip_grad_norms(
        self,
        backbone_max_norm: float,
        actor_max_norm: float,
        critic_max_norm: float,
    ) -> dict[str, T.Tensor]:
        """Measure module norms once and clip the three PPO parameter groups.

        Per-parameter norms are computed together, then reused to form both the
        diagnostic module norms and the clipping-group norms. The returned
        tensors remain on the model device so callers can batch scalar
        synchronization with other metrics.

        Args:
            backbone_max_norm: Maximum norm for CNN and sequence encoder grads.
            actor_max_norm: Maximum norm for actor grads.
            critic_max_norm: Maximum norm for critic grads.

        Returns:
            Scalar tensors for the unclipped module norms and the three
            pre-clipping parameter-group norms.
        """
        norms, gradients = self._get_gradient_norm_tensors()
        group_norms = {
            "grad_norm/backbone_total": self._combine_norms(
                [norms["grad_norm/cnn"], norms["grad_norm/sequence_encoder"]]
            ),
            "grad_norm/actor_total": norms["grad_norm/actor"],
            "grad_norm/critic_total": norms["grad_norm/critic"],
        }

        backbone_scale = self._clip_gradient_group(
            gradients["backbone"],
            group_norms["grad_norm/backbone_total"],
            backbone_max_norm,
        )
        self._clip_gradient_group(
            gradients["actor"], group_norms["grad_norm/actor_total"], actor_max_norm
        )
        self._clip_gradient_group(
            gradients["critic"], group_norms["grad_norm/critic_total"], critic_max_norm
        )
        return {
            **norms,
            **group_norms,
            "grad_clip/backbone_scale": backbone_scale,
            "grad_clip/backbone_clipped": (backbone_scale < 1.0).to(dtype=T.float32),
        }

    def get_partial_clip_grad_norms(self) -> dict[str, float]:
        """Get per-module gradient norms for monitoring (no clipping).

        Returns:
            Dict with module norms, global norm, and per-layer CNN norms.
        """
        norm_tensors, _ = self._get_gradient_norm_tensors()
        keys = tuple(norm_tensors)
        values = T.stack([norm_tensors[key] for key in keys]).detach().cpu().tolist()
        return dict(zip(keys, values, strict=True))

    def _get_gradient_norm_tensors(
        self,
    ) -> tuple[dict[str, T.Tensor], dict[str, list[T.Tensor]]]:
        """Compute all diagnostic norms in one batched pass over gradients."""
        modules = {
            "cnn": list(self._network.cnn.parameters()),
            "sequence_encoder": list(self._network.sequence_encoder.parameters()),
            "actor": list(self._network.actor.parameters()),
            "critic": list(self._network.critic.parameters()),
        }
        gradients: dict[str, list[T.Tensor]] = {
            name: [
                parameter.grad for parameter in parameters if parameter.grad is not None
            ]
            for name, parameters in modules.items()
        }
        active_parameters = [
            (name, parameter)
            for name, parameters in modules.items()
            for parameter in parameters
            if parameter.grad is not None
        ]
        active_names = [name for name, _ in active_parameters]
        all_gradients = [parameter.grad for _, parameter in active_parameters]

        if all_gradients:
            per_parameter_norms = T._foreach_norm(all_gradients, 2)
            parameter_norms = {
                id(parameter): norm
                for (_, parameter), norm in zip(
                    active_parameters, per_parameter_norms, strict=True
                )
            }
            module_norms: dict[str, list[T.Tensor]] = {name: [] for name in modules}
            for name, norm in zip(active_names, per_parameter_norms, strict=True):
                module_norms[name].append(norm)
            zero = per_parameter_norms[0].new_zeros(())
            norms = {
                f"grad_norm/{name}": self._combine_norms(module_norms[name], zero)
                for name in modules
            }
            norms["grad_norm/max"] = self._combine_norms(
                list(per_parameter_norms), zero
            )
            for layer_name, parameters in self._cnn_layer_parameter_groups().items():
                layer_norms = [
                    parameter_norms[id(parameter)]
                    for parameter in parameters
                    if id(parameter) in parameter_norms
                ]
                norms[f"grad_norm/cnn_{layer_name}"] = self._combine_norms(
                    layer_norms, zero
                )
        else:
            zero = T.zeros((), device=self.device)
            norms = {
                **{f"grad_norm/{name}": zero for name in modules},
                "grad_norm/max": zero,
            }
            norms.update(
                {
                    f"grad_norm/cnn_{layer_name}": zero
                    for layer_name in self._cnn_layer_parameter_groups()
                }
            )

        return norms, {
            "backbone": gradients["cnn"] + gradients["sequence_encoder"],
            "actor": gradients["actor"],
            "critic": gradients["critic"],
        }

    @staticmethod
    def _combine_norms(norms: list[T.Tensor], zero: T.Tensor | None = None) -> T.Tensor:
        """Combine parameter norms into a single L2 norm."""
        if not norms:
            assert zero is not None
            return zero
        return T.linalg.vector_norm(T.stack(norms), ord=2)

    def _cnn_layer_parameter_groups(self) -> dict[str, list[Parameter]]:
        """Group CNN parameters by layer for the per-layer gradient norms."""
        return self._network.cnn.layer_parameter_groups()

    @staticmethod
    def _clip_gradient_group(
        gradients: list[T.Tensor], total_norm: T.Tensor, max_norm: float
    ) -> T.Tensor:
        """Scale gradients in-place and return the applied coefficient."""
        clip_coefficient = T.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        if gradients:
            T._foreach_mul_(gradients, clip_coefficient)
        return clip_coefficient

    def save_state_dict(self, path: str) -> None:
        """Save network state dict to disk."""
        self._network.save_state_dict(path)

    def load_state_dict(self, path: str) -> None:
        """Load network weights from a local model checkpoint.

        Args:
            path: Local model or state-dict checkpoint file.
        """
        LocalCheckpointLoader(path).load(self._network, self.device)

    def reset(self) -> None:
        """Clear observation and done history windows."""
        self._obs_window.clear()
        self._done_window.clear()

    def step_env(
        self,
        env: Env,
        state: NDArray,
        done: T.Tensor,
        temperature: float | None = None,
    ) -> tuple[Transition, dict[str, Any]]:
        """Step the environment using this agent's policy.

        Handles numpy<->tensor conversion and combines terminated/truncated
        into a single done flag so callers don't duplicate logic.

        Args:
            env: Vectorized Gymnasium environment.
            state: Numpy array of shape (num_envs, height, width, channel), uint8.
            done: Boolean tensor of shape (num_envs,) from previous step.
            temperature: Sampling temperature. Defaults to config.action_temperature.

        Returns:
            # TODO
        """
        temperature = (
            self.cfg.action_temperature if temperature is None else temperature
        )
        with record_function("agent/policy"):
            action, extras = self.act(state, done, temperature)

        with record_function("environment/step"):
            next_state, reward, terminated, truncated, info = env.step(action)

        transition = Transition(
            obs=state,
            action=action,
            reward=reward.to(T.float32),
            terminated=terminated,
            truncated=truncated,
            next_obs=next_state,
            extras=extras,
        )

        return (
            transition,
            info,
        )

    def network_config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable network hyperparameters."""
        return self._network.config()
