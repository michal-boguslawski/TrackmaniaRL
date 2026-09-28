"""Agent: manages policy evaluation and temporal history for vectorized environments.

The Agent wraps a Network and maintains per-environment observation/done windows
to construct fixed-length temporal sequences. It handles observation preprocessing
(NHWC uint8 -> NCHW float32), temporal encoding with episode-boundary masking,
and action sampling from the Beta-distribution actor.
"""

from collections import deque
from numpy.typing import NDArray
from logging import getLogger
from gymnasium import Env
import torch as T
from torch import nn
from torch.distributions import Distribution
from torch.profiler import record_function
from torch.nn.parameter import Parameter
from typing import Iterator

from rl_lib.networks.factory import Network
from rl_lib.run_config import AgentSettings


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
    """

    def __init__(
        self,
        network: Network,
        device: T.device | str,
        config: AgentSettings,
    ):
        """Initialize the agent.

        Args:
            network: Policy/value network with matching stack_size.
            device: Target device for tensor operations.
            config: Agent configuration including stack_size and preprocessing params.

        Raises:
            ValueError: If config.stack_size != network.stack_size.
        """
        if config.stack_size != network.stack_size:
            raise ValueError(
                f"agent stack_size ({config.stack_size}) must match the network stack_size ({network.stack_size})"
            )
        self.cfg = config
        self._stack_size = config.stack_size
        self._device = T.device(device)
        self._obs_window: deque[T.Tensor] = deque(maxlen=self._stack_size)
        self._done_window: deque[T.Tensor] = deque(maxlen=self._stack_size)
        self._network = network

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
            observation_tensor = observation.to(self._device, T.float32) / self.cfg.observation_divisor
            observation_tensor = observation_tensor + self.cfg.observation_offset
        with record_function("agent/permute_to_nchw"):
            return observation_tensor.permute(0, 3, 1, 2)

    def feature_extract(self, observation: T.Tensor) -> T.Tensor:
        """Extract CNN features from raw observation.

        Args:
            observation: Tensor of shape (batch, height, width, channel), dtype uint8.

        Returns:
            Tensor of shape (batch, cnn_out_dim) from the CNN encoder.
        """
        obs = self._preprocess_observation(observation)
        return self._network.feature_extract(obs)

    def _get_features_window(self, features: T.Tensor) -> T.Tensor:
        """Build temporal window of features, padding with copies of first frame.

        Args:
            features: Tensor of shape (batch, feature_dim) for current step.

        Returns:
            Tensor of shape (batch, stack_size, feature_dim).
        """
        while len(self._obs_window) < self._stack_size:
            self._obs_window.append(features.clone())
        self._obs_window.append(features)
        return T.stack(list(self._obs_window), dim=1)

    def _get_mask_window(self, done: T.Tensor) -> T.Tensor:
        """Build temporal mask marking valid history positions per environment.

        A position is valid (False) if no episode boundary precedes it within
        the window. The mask is True for positions that should be zeroed out.

        Args:
            done: Boolean tensor of shape (batch,) indicating episode termination
                at the current step.

        Returns:
            Boolean mask of shape (batch, stack_size) where True means
            "mask this position (zero out features)".
        """
        while len(self._done_window) < self._stack_size:
            self._done_window.append(T.zeros_like(done))
        self._done_window.append(done)
        stacked_done_window = T.stack(list(self._done_window), dim=1)
        # True where a done has occurred at or before this position in the window
        mask = stacked_done_window.logical_not() & (stacked_done_window.flip(1).cumsum(dim=1).flip(1) > 0)
        return mask

    def temporal_encode(self, x: T.Tensor, mask: T.Tensor | None = None) -> T.Tensor:
        """Encode a sequence of features through the temporal encoder.

        Args:
            x: Tensor of shape (batch, stack_size, feature_dim).
            mask: Optional boolean mask of shape (batch, stack_size). Positions
                where True are zeroed before encoding.

        Returns:
            Tensor of shape (batch, temporal_out_dim) after temporal encoding
            and squeezing the sequence dimension.
        """
        if mask is None:
            masked_x = x
        else:
            assert mask.shape == x.shape[:2], f"mask: {mask.shape}, x: {x.shape}"
            masked_x = x.masked_fill(mask.unsqueeze(-1), 0.)
        return self._network.temporal_encode(masked_x).squeeze(1)

    def heads(self, temporal: T.Tensor, temperature: float | None = None) -> tuple[Distribution, T.Tensor]:
        """Compute action distribution and value from temporal encoding.

        Args:
            temporal: Tensor of shape (batch, temporal_out_dim).
            temperature: Sampling temperature for the Beta distribution actor.
                Defaults to config.action_temperature.

        Returns:
            Tuple of (action_distribution, action_mean, value).
        """
        temperature = self.cfg.action_temperature if temperature is None else temperature
        return self._network.heads(temporal, temperature)

    def act(
        self,
        observation: T.Tensor,
        done: T.Tensor,
        temperature: float | None = None
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
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
            temperature = self.cfg.action_temperature if temperature is None else temperature
            features = self.feature_extract(observation)

            temporal = self.temporal_encode(
                self._get_features_window(features),
                self._get_mask_window(done),
            )

            action_dist, action, value = self.heads(temporal, temperature)
            
            if temperature != 0.:
                action = action_dist.sample()

            log_probs = action_dist.log_prob(action)

            if not T.isfinite(log_probs).all():
                logger.error(
                    "log_probs are not finite: action=%s, alpha=%s, beta=%s",
                    action,
                    action_dist.base_dist.concentration1,
                    action_dist.base_dist.concentration0,
                )
                raise ValueError("log_probs are not finite")

        return (
            action,
            log_probs,
            value
        )

    def bootstrap_value(self, observation: T.Tensor) -> T.Tensor:
        """Evaluate value of a successor observation without mutating history.

        Used for GAE bootstrap when an episode truncates (time limit). The
        current history already contains the observation that produced the
        action; we append the final observation to a temporary window.

        Args:
            observation: Tensor of shape (batch, height, width, channel), uint8.
                The final observation from a truncated episode.

        Returns:
            Tensor of shape (batch,) bootstrap value estimate.
        """
        with T.no_grad():
            features = self.feature_extract(observation)
            history_size = self._stack_size - 1
            prior_features = list(self._obs_window)[-history_size:] if history_size else []
            windowed_features = T.stack([*prior_features, features], dim=1)

            prior_dones = list(self._done_window)[-history_size:] if history_size else []
            final_done = T.zeros(features.shape[0], dtype=T.bool, device=features.device)
            windowed_dones = T.stack([*prior_dones, final_done], dim=1)
            mask = windowed_dones.logical_not() & (
                windowed_dones.flip(1).cumsum(dim=1).flip(1) > 0
            )

            temporal = self.temporal_encode(windowed_features, mask)
            return self._network.critic(temporal)

    def evaluate_actions(
        self,
        observation: T.Tensor,
        action: T.Tensor,
        dones: T.Tensor | None = None
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
        extracted_features = self.feature_extract(observation)
        windowed_features = extracted_features.unfold(0, self._stack_size, self._stack_size).permute(0, 2, 1)
        
        windowed_dones = dones.unfold(0, self._stack_size, self._stack_size)
        mask = windowed_dones.logical_not() & (windowed_dones.flip(1).cumsum(1).flip(1) > 0)

        temporal_encoding = self.temporal_encode(windowed_features, mask)
        action_dist, action_mean, values = self.heads(temporal_encoding)

        log_probs = action_dist.log_prob(action)
        return log_probs, values, action_dist, action_mean

    def action_transform(self, action: T.Tensor) -> T.Tensor:
        """Apply affine transform from Beta [0,1] to environment action bounds.

        Args:
            action: Tensor in [0, 1] range from Beta distribution.

        Returns:
            Tensor scaled to [action_low, action_high] per dimension.
        """
        return self._network.action_transform(action)

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
        """Group network parameters by role for differential learning rates.

        Groups:
        - backbone_decay: CNN + temporal encoder weights (with weight decay)
        - backbone_no_decay: CNN + temporal encoder biases/norms (no weight decay)
        - head_decay: Actor/critic weights (with weight decay)
        - head_no_decay: Actor/critic biases + log_std params (no weight decay)

        Returns:
            Dict mapping group name to list of parameters.
        """
        decay, no_decay = [], []
        for module in [self._network.cnn, self._network.sequence_encoder]:
            for name, p in module.named_parameters():
                (no_decay if "bias" in name or "norm" in name.lower() else decay).append(p)

        head_decay, head_no_decay = [], []
        for module in [self._network.actor, self._network.critic]:
            for name, p in module.named_parameters():
                (head_no_decay if "bias" in name or "_log_std" in name else head_decay).append(p)

        return {
            "backbone_decay": decay,
            "backbone_no_decay": no_decay,
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

        self._clip_gradient_group(
            gradients["backbone"], group_norms["grad_norm/backbone_total"], backbone_max_norm
        )
        self._clip_gradient_group(
            gradients["actor"], group_norms["grad_norm/actor_total"], actor_max_norm
        )
        self._clip_gradient_group(
            gradients["critic"], group_norms["grad_norm/critic_total"], critic_max_norm
        )
        return {**norms, **group_norms}

    def get_partial_clip_grad_norms(self) -> dict[str, float]:
        """Get per-module gradient norms for monitoring (no clipping).

        Returns:
            Dict with keys: cnn, sequence_encoder, actor, critic, max.
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
            name: [parameter.grad for parameter in parameters if parameter.grad is not None]
            for name, parameters in modules.items()
        }
        active_names = [name for name, grads in gradients.items() for _ in grads]
        all_gradients = [grad for grads in gradients.values() for grad in grads]

        if all_gradients:
            per_parameter_norms = T._foreach_norm(all_gradients, 2)
            module_norms: dict[str, list[T.Tensor]] = {name: [] for name in modules}
            for name, norm in zip(active_names, per_parameter_norms, strict=True):
                module_norms[name].append(norm)
            zero = per_parameter_norms[0].new_zeros(())
            norms = {
                f"grad_norm/{name}": self._combine_norms(module_norms[name], zero)
                for name in modules
            }
            norms["grad_norm/max"] = self._combine_norms(list(per_parameter_norms), zero)
        else:
            zero = T.zeros((), device=self.device)
            norms = {
                **{f"grad_norm/{name}": zero for name in modules},
                "grad_norm/max": zero,
            }

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

    @staticmethod
    def _clip_gradient_group(
        gradients: list[T.Tensor], total_norm: T.Tensor, max_norm: float
    ) -> None:
        """Scale gradients in-place using an already-computed group norm."""
        if gradients:
            clip_coefficient = T.clamp(max_norm / (total_norm + 1e-6), max=1.0)
            T._foreach_mul_(gradients, clip_coefficient)

    def save_state_dict(self, path: str) -> None:
        """Save network state dict to disk."""
        self._network.save_state_dict(path)

    def load_state_dict(self, path: str) -> None:
        """Load network state dict from disk."""
        self._network.load_state_dict(path)

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
    ) -> tuple:
        """Step the environment using this agent's policy.

        Handles numpy<->tensor conversion and combines terminated/truncated
        into a single done flag so callers don't duplicate logic.

        Args:
            env: Vectorized Gymnasium environment.
            state: Numpy array of shape (num_envs, height, width, channel), uint8.
            done: Boolean tensor of shape (num_envs,) from previous step.
            temperature: Sampling temperature. Defaults to config.action_temperature.

        Returns:
            Tuple of (next_state, state_t, action, log_probs, value, reward,
            terminated, truncated, done, info).
        """
        with record_function("transfer/observation_to_device"):
            state_t = T.from_numpy(state).to(self._device)
        temperature = self.cfg.action_temperature if temperature is None else temperature
        with record_function("agent/policy"):
            action, log_probs, value = self.act(state_t, done, temperature)

        with record_function("transfer/action_to_host"):
            action_np = action.cpu().numpy()
        with record_function("environment/step"):
            next_state, reward, terminated, truncated, info = env.step(action_np)

        with record_function("transfer/flags_to_device"):
            terminated_t = T.from_numpy(terminated).to(self._device)
            truncated_t = T.from_numpy(truncated).to(self._device)
        done_t = T.logical_or(terminated_t, truncated_t)

        return next_state, state_t, action, log_probs, value, reward, terminated_t, truncated_t, done_t, info

    def network_config(self) -> dict[str, int | float | str]:
        """Flat, MLflow-loggable network hyperparameters."""
        return self._network.config()
