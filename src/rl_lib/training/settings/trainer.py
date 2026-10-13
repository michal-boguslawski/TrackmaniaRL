"""Top-level trainer settings and compatibility migration."""

from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from rl_lib.training.settings.base import StrictModel
from rl_lib.training.settings.loss import PPOLossSettings
from rl_lib.training.settings.optimizer import OptimizerSettings
from rl_lib.training.settings.scheduler import (
    CosineSchedulerSettings,
    SchedulerSettings,
)


class KLControlSettings(StrictModel):
    """KL-based early stopping controls for PPO updates."""

    target_kl: float = Field(default=0.03, gt=0, description="Target approximate KL per update.")
    warmup_steps: int = Field(default=20_000, ge=0, description="Updates before KL stopping is enabled.")
    stop_multiplier: float = Field(default=3.0, gt=0, description="Multiplier applied to target_kl for hard stopping.")
    hard_stop: bool = Field(default=True, description="Stop PPO updates when the KL threshold is exceeded.")


class GradientSettings(StrictModel):
    """Gradient clipping limits for PPO network components."""

    backbone_max_grad_norm: float = Field(default=0.5, gt=0, description="Maximum gradient norm for the CNN and sequence encoder.")
    actor_max_grad_norm: float = Field(default=0.5, gt=0, description="Maximum gradient norm for the actor.")
    critic_max_grad_norm: float = Field(default=0.5, gt=0, description="Maximum gradient norm for the critic.")


class StabilitySettings(StrictModel):
    """Numerical guards used by PPO loss and metric calculations."""

    advantage_epsilon: float = Field(default=1e-4, gt=0, description="Small constant used when normalizing advantages.")
    explained_variance_epsilon: float = Field(default=1e-8, gt=0, description="Small constant preventing a zero variance denominator.")
    log_ratio_min: FiniteFloat = Field(default=-5.0, description="Lower bound applied to policy log ratios.")
    log_ratio_max: FiniteFloat = Field(default=5.0, description="Upper bound applied to policy log ratios.")
    skip_nonfinite_updates: bool = Field(default=True, description="Skip optimizer updates with non-finite losses or gradients.")
    tanh_saturation_threshold: float = Field(default=3.5, gt=0, description="Absolute action-mean threshold used for saturation diagnostics.")

    @model_validator(mode="after")
    def validate_log_ratio_range(self):
        """Require an increasing interval for policy log-ratio clamping."""
        if self.log_ratio_min >= self.log_ratio_max:
            raise ValueError("trainer.stability.log_ratio_min must be lower than log_ratio_max")
        return self


class TrainerSettings(StrictModel):
    """Trainer identity and grouped algorithm-specific optimization settings.

    Loss settings are algorithm-specific, so a future SAC trainer can provide a
    different loss configuration without adding SAC parameters to the PPO loss
    model. Optimizer and scheduler settings remain independently typed.

    Attributes:
        name: Trainer or algorithm identity.
        loss: PPO objective and entropy settings.
        kl: KL-based update stopping settings.
        gradients: Component-wise gradient clipping settings.
        stability: Numerical guards for losses and diagnostics.
        optimizer: Optimizer and parameter-group settings.
        scheduler: Learning-rate scheduler variant and settings.
        minibatch_indexing_mode: Whether minibatch indices are IID or sequential.
    """

    name: str = Field(
        default="ppo",
        min_length=1,
        description="Trainer/algorithm identity; currently the PPO trainer is implemented.",
    )
    loss: PPOLossSettings = Field(
        default_factory=PPOLossSettings,
        description="Algorithm-specific loss configuration.",
    )
    kl: KLControlSettings = Field(
        default_factory=KLControlSettings,
        description="KL-based stopping behavior.",
    )
    gradients: GradientSettings = Field(
        default_factory=GradientSettings,
        description="Gradient clipping limits.",
    )
    stability: StabilitySettings = Field(
        default_factory=StabilitySettings,
        description="Numerical stability and non-finite update behavior.",
    )
    optimizer: OptimizerSettings = Field(
        default_factory=OptimizerSettings,
        description="Optimizer implementation and parameter-group settings.",
    )
    scheduler: SchedulerSettings = Field(
        default_factory=CosineSchedulerSettings,
        description="Learning-rate scheduler variant and its options.",
    )
    minibatch_indexing_mode: Literal["iid", "sequential"] = Field(
        default="iid",
        description="Minibatch sample ordering strategy.",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, values):
        """Move the former flat trainer fields into their grouped settings."""
        if not isinstance(values, dict):
            return values
        values = dict(values)
        grouped_fields = {
            "loss": {
                "ppo_epsilon": "ppo_epsilon",
                "value_clip_epsilon": "value_clip_epsilon",
                "critic_beta": "critic_beta",
                "mean_reg_coef": "mean_reg_coef",
                "advantage_normalization_strategy": "advantage_normalization_strategy",
            },
            "kl": {
                "target_kl": "target_kl",
                "kl_warmup_steps": "warmup_steps",
                "kl_stop_multiplier": "stop_multiplier",
                "hard_stop_kl": "hard_stop",
            },
            "gradients": {
                "backbone_max_grad_norm": "backbone_max_grad_norm",
                "actor_max_grad_norm": "actor_max_grad_norm",
                "critic_max_grad_norm": "critic_max_grad_norm",
            },
            "stability": {
                "advantage_epsilon": "advantage_epsilon",
                "explained_variance_epsilon": "explained_variance_epsilon",
                "log_ratio_min": "log_ratio_min",
                "log_ratio_max": "log_ratio_max",
                "skip_nonfinite_updates": "skip_nonfinite_updates",
                "tanh_saturation_threshold": "tanh_saturation_threshold",
            },
        }
        for section, field_map in grouped_fields.items():
            legacy_values = {
                old_key: field_map[old_key]
                for old_key in field_map
                if old_key in values
            }
            if not legacy_values:
                continue
            section_values = cls._as_dict(values.get(section, {}))
            for old_key, new_key in legacy_values.items():
                section_values.setdefault(new_key, values.pop(old_key))
            values[section] = section_values

        legacy_entropy = {
            "entropy_coef": "coefficient",
            "entropy_decay": "decay",
            "entropy_coef_min": "min_coefficient",
        }
        if any(key in values for key in legacy_entropy):
            loss_values = cls._as_dict(values.get("loss", {}))
            entropy_values = cls._as_dict(loss_values.get("entropy", {}))
            for old_key, new_key in legacy_entropy.items():
                if old_key in values:
                    entropy_values.setdefault(new_key, values.pop(old_key))
            loss_values["entropy"] = entropy_values
            values["loss"] = loss_values

        optimizer_keys = {
            "backbone_lr": "backbone_lr",
            "cnn_lr": "cnn_lr",
            "sequence_encoder_lr": "sequence_encoder_lr",
            "head_lr": "head_lr",
            "weight_decay": "weight_decay",
            "no_decay_weight_decay": "no_decay_weight_decay",
            "optimizer_eps": "eps",
            "optimizer_beta1": "beta1",
            "optimizer_beta2": "beta2",
            "optimizer_amsgrad": "amsgrad",
            "fused_optimizer": "fused",
        }
        optimizer = cls._as_dict(values.get("optimizer", {}))
        for old_key, new_key in optimizer_keys.items():
            if old_key in values:
                optimizer.setdefault(new_key, values.pop(old_key))
        if isinstance(values.get("optimizer"), str):
            optimizer.setdefault("name", values.pop("optimizer"))
        if optimizer:
            values["optimizer"] = optimizer

        scheduler = values.get("scheduler")
        if isinstance(scheduler, str):
            scheduler_name = values.pop("scheduler")
            scheduler_options = {"name": scheduler_name}
            if "scheduler_min_lr" in values:
                min_lr = values.pop("scheduler_min_lr")
                if scheduler_name == "cosine":
                    scheduler_options["eta_min"] = min_lr
                elif scheduler_name == "linear":
                    scheduler_options["min_lr"] = min_lr
            if "scheduler_start_factor" in values:
                start_factor = values.pop("scheduler_start_factor")
                if scheduler_name == "linear":
                    scheduler_options["start_factor"] = start_factor
            if "scheduler_extra_iters" in values:
                extra_iters = values.pop("scheduler_extra_iters")
                if scheduler_name in ("cosine", "linear"):
                    scheduler_options["extra_iters"] = extra_iters
            values["scheduler"] = scheduler_options
        elif scheduler is None:
            legacy_scheduler_keys = (
                "scheduler_min_lr",
                "scheduler_start_factor",
                "scheduler_extra_iters",
            )
            if any(key in values for key in legacy_scheduler_keys):
                scheduler_options = {"name": "cosine"}
                if "scheduler_min_lr" in values:
                    scheduler_options["eta_min"] = values.pop("scheduler_min_lr")
                values.pop("scheduler_start_factor", None)
                if "scheduler_extra_iters" in values:
                    scheduler_options["extra_iters"] = values.pop("scheduler_extra_iters")
                values["scheduler"] = scheduler_options
        return values

    @staticmethod
    def _as_dict(value):
        """Copy a settings mapping or a Pydantic model into a dictionary."""
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return {}
