import pytest
from pydantic import TypeAdapter, ValidationError

from rl_lib.buffers.rollout_buffer import RolloutBuffer
from rl_lib.profiling_config import ProfilingConfig, load_profiling_config
from rl_lib.envs.wrappers.settings import WrapperSettings, make_wrapper_settings
from rl_lib.run_config import RolloutSettings, RunConfig, load_config
from rl_lib.training.ppo.trainer import PPOTrainer


def test_sample_yaml_loads_full_current_run_defaults():
    from importlib import resources
    config_path = str(resources.files("rl_lib.config") / "ppo_carracing.yaml")
    config = load_config(config_path)

    assert config.environment.id == "CarRacing-v3"
    assert config.environment.num_envs == 8
    assert config.run.algorithm == "ppo"
    assert config.run.total_steps == 1_000_000
    assert config.run.pretrained_checkpoint is None
    assert config.run.cnn_autocast == "none"
    assert config.rollout.gamma == pytest.approx(0.99)
    assert config.rollout.epochs == 3
    assert config.trainer.name == "ppo"
    assert config.trainer.loss.name == "ppo"
    assert config.trainer.loss.ppo_epsilon == pytest.approx(0.1)
    assert config.network.cnn.conv_layers[0].kernel_size == 8
    assert config.callback("checkpoints").interval == 100_000
    assert config.callback("evaluation").interval == 100_000
    assert config.callback("record_video").interval == 100_000
    assert config.callback("metrics").granularity == "batch"
    assert config.callback("evaluation").final_episodes == 1_000
    reward_wrapper = next(
        wrapper for wrapper in config.environment.wrappers if wrapper.name == "reward_on_done"
    )
    assert reward_wrapper.on_truncated == 0.0
    assert all(wrapper.name != "off_track_penalty" for wrapper in config.environment.wrappers)
    assert config.tracking.verbosity == 2


def test_performance_backend_defaults_and_validation():
    from rl_lib.run_config import RunSettings
    from rl_lib.training.settings import TrainerSettings

    settings = RunSettings()
    assert settings.cudnn_benchmark is True
    assert settings.channels_last is True
    assert settings.torch_compile is False
    assert settings.torch_compile_mode == "reduce-overhead"
    assert settings.cnn_autocast == "none"
    assert TrainerSettings().optimizer.fused is True

    with pytest.raises(ValidationError):
        RunSettings(torch_compile_mode="turbo")
    with pytest.raises(ValidationError):
        RunSettings(cnn_autocast="fp16")


def test_packaged_profiling_config_is_separate_and_disabled_by_default():
    from importlib import resources

    path = resources.files("rl_lib.config") / "profiling.yaml"
    config = load_profiling_config(path)

    assert isinstance(config, ProfilingConfig)
    assert config.enabled is False
    assert config.wait_steps is None
    assert (config.warmup_steps, config.active_steps) == (1, 2)
    assert config.resolve_schedule(1024) == (1022, 1)
    assert config.resolve_schedule(1) == (0, 0)


def test_profile_config_matches_the_documented_capture_recipe():
    from importlib import resources

    config = load_config(str(resources.files("rl_lib.config") / "manual_profile.yaml"))

    assert config.rollout.buffer_size == 128
    assert config.rollout.minibatch_size == 64
    assert config.run.total_steps >= 132
    # The network is what sets the shape of the traced forward/backward.
    production = load_config(str(resources.files("rl_lib.config") / "ppo_carracing.yaml"))
    assert config.network == production.network
    # Both callbacks run full episodes outside the capture and would dominate it.
    assert config.callback("evaluation") is None
    assert config.callback("record_video") is None
    # state_dict() during a periodic checkpoint would land in the recorded window.
    assert config.callback("checkpoints") is None


def test_profile_config_schedule_covers_the_first_rollout_update():
    from importlib import resources

    config = load_config(str(resources.files("rl_lib.config") / "manual_profile.yaml"))
    profiling = load_profiling_config(resources.files("rl_lib.config") / "profiling.yaml")

    wait_steps, warmup_steps = profiling.resolve_schedule(config.rollout.buffer_size)

    # RolloutCollector calls profiler.step() once per vector step and runs the
    # update on the step that fills the buffer, so automatic alignment must put
    # the first active step exactly there.
    assert wait_steps + warmup_steps + 1 == config.rollout.buffer_size
    assert config.run.total_steps >= config.rollout.buffer_size + warmup_steps + 1


def test_profiling_schedule_can_be_set_explicitly(tmp_path):
    path = tmp_path / "profiling.yaml"
    path.write_text("wait_steps: 10\nwarmup_steps: 3\n", encoding="utf-8")

    assert load_profiling_config(path).resolve_schedule(1024) == (10, 3)


def test_profiling_config_validates_capture_schedule(tmp_path):
    path = tmp_path / "profiling.yaml"
    path.write_text("enabled: true\nactive_steps: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_profiling_config(path)


def test_tracking_verbosity_selects_how_many_metrics_reach_the_backends(tmp_path):
    path = tmp_path / "verbosity.yaml"
    path.write_text("tracking:\n  verbosity: 1\n", encoding="utf-8")

    assert load_config(str(path)).tracking.verbosity == 1


def test_tracking_verbosity_rejects_an_unknown_level(tmp_path):
    path = tmp_path / "bad_verbosity.yaml"
    path.write_text("tracking:\n  verbosity: 3\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(str(path))


def test_missing_yaml_sections_resolve_from_pydantic_defaults(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text("run:\n  seed: 42\n", encoding="utf-8")

    config = load_config(str(path))

    assert config.run.seed == 42
    assert config.environment.num_envs == 16
    assert config.rollout.gae_lambda == pytest.approx(0.95)
    assert config.network.actor.action_low == [-1.0, 0.0, 0.0]


def test_pretrained_checkpoint_can_be_configured(tmp_path):
    path = tmp_path / "pretrained.yaml"
    path.write_text("run:\n  pretrained_checkpoint: models/policy.pt\n", encoding="utf-8")

    assert load_config(str(path)).run.pretrained_checkpoint == "models/policy.pt"


def test_pretrained_mlflow_checkpoint_can_be_configured(tmp_path):
    path = tmp_path / "pretrained-mlflow.yaml"
    path.write_text(
        "run:\n  pretrained_mlflow_run_id: runs:/run-123/final_model\n"
        "  pretrained_mlflow_checkpoint_step: 500000\n",
        encoding="utf-8",
    )

    config = load_config(str(path))

    assert config.run.pretrained_mlflow_run_id == "runs:/run-123/final_model"
    assert config.run.pretrained_mlflow_checkpoint_step == 500000


@pytest.mark.parametrize(
    "run_settings",
    [
        {
            "pretrained_checkpoint": "local.pt",
            "pretrained_mlflow_run_id": "run-123",
        },
        {"pretrained_mlflow_checkpoint_step": 100},
    ],
)
def test_pretrained_checkpoint_sources_are_validated(run_settings):
    with pytest.raises(ValidationError, match="pretrained"):
        RunConfig.model_validate({"run": run_settings})


def test_run_config_accepts_a_registered_algorithm_name(tmp_path):
    path = tmp_path / "algorithm.yaml"
    path.write_text("run:\n  algorithm: sac\n", encoding="utf-8")

    config = load_config(str(path))

    assert config.run.algorithm == "sac"
    assert config.trainer.name == "sac"


def test_explicit_trainer_name_must_match_run_algorithm():
    with pytest.raises(ValidationError, match="trainer.name must match run.algorithm"):
        RunConfig.model_validate({
            "run": {"algorithm": "sac"},
            "trainer": {"name": "ppo"},
        })


def test_loader_rejects_unknown_fields(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text("trainer:\n  ppo_epsilonn: 0.3\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="ppo_epsilonn"):
        load_config(str(path))


def test_loader_reports_malformed_yaml(tmp_path):
    path = tmp_path / "malformed.yaml"
    path.write_text("run: [unterminated", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(str(path))


@pytest.mark.parametrize(
    "yaml_text",
    [
        "rollout:\n  gamma: 1.5\n",
        "rollout:\n  buffer_size: 2\n  minibatch_size: 100\n",
        "trainer:\n  backbone_lr: 0.00001\n  scheduler_min_lr: 0.1\n",
    ],
)
def test_run_config_rejects_invalid_ranges_and_relationships(tmp_path, yaml_text):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(str(path))


def test_wrapper_settings_validate_name_and_wrapper_specific_options():
    wrapper_adapter = TypeAdapter(WrapperSettings)
    with pytest.raises(ValidationError):
        wrapper_adapter.validate_python({"name": "not_registered"})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        wrapper_adapter.validate_python({"name": "grayscale", "stack_size": 3})

    assert make_wrapper_settings("frame_stack", stack_size=3).stack_size == 3

    config = RunConfig.model_validate({
        "environment": {
            "wrappers": [{"name": "reward_on_done", "on_truncated": -5.0}],
        },
        "callbacks": {"episode_statistics": False, "record_video": False},
    })
    assert config.environment.wrappers[0].on_truncated == -5.0


def test_rollout_and_trainer_settings_reach_runtime_components(agent):
    buffer = RolloutBuffer(
        RolloutSettings(buffer_size=8, gamma=0.8, gae_lambda=0.7),
        stack_size=agent.stack_size,
    )
    trainer = PPOTrainer(agent, RunConfig.model_validate({
        "trainer": {
            "backbone_lr": 0.0002,
            "head_lr": 0.0003,
            "optimizer": "Adam",
            "scheduler": "none",
            "backbone_max_grad_norm": 0.75,
            "entropy_coef_min": 0.00002,
        },
    }).trainer)

    assert buffer.config()["gamma"] == pytest.approx(0.8)
    assert buffer.config()["gae_lambda"] == pytest.approx(0.7)
    assert trainer.cfg.optimizer.backbone_lr == pytest.approx(0.0002)
    assert trainer.cfg.optimizer.head_lr == pytest.approx(0.0003)
    assert [group["lr"] for group in trainer._optimizer.param_groups] == pytest.approx(
        [0.0002, 0.0002, 0.0002, 0.0002, 0.0003, 0.0003]
    )
    assert trainer._optimizer.__class__.__name__ == "Adam"
    assert trainer.cfg.gradients.backbone_max_grad_norm == pytest.approx(0.75)
    assert trainer.cfg.loss.entropy.min_coefficient == pytest.approx(0.00002)


def test_component_learning_rates_override_backbone_rate(agent):
    trainer = PPOTrainer(
        agent,
        RunConfig.model_validate({
            "trainer": {
                "backbone_lr": 0.0002,
                "cnn_lr": 0.0001,
                "sequence_encoder_lr": 0.00005,
                "head_lr": 0.0003,
                "scheduler_min_lr": 1e-8,
            },
        }).trainer,
    )

    assert [group["lr"] for group in trainer._optimizer.param_groups] == pytest.approx(
        [0.0001, 0.0001, 0.00005, 0.00005, 0.0003, 0.0003]
    )


@pytest.mark.parametrize(
    ("scheduler", "expected_type"),
    [
        ({"name": "cosine", "eta_min": 1e-8}, "CosineSchedulerSettings"),
        ({"name": "linear", "min_lr": 1e-8, "start_factor": 0.8}, "LinearSchedulerSettings"),
        ({"name": "exponential", "gamma": 0.97}, "ExponentialSchedulerSettings"),
    ],
)
def test_scheduler_settings_validate_type_specific_options(scheduler, expected_type):
    config = RunConfig.model_validate({"trainer": {"scheduler": scheduler}})

    assert type(config.trainer.scheduler).__name__ == expected_type


def test_scheduler_settings_reject_options_for_other_scheduler_types():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({
            "trainer": {"scheduler": {"name": "exponential", "eta_min": 1e-8}},
        })


def test_environment_wrapper_arguments_survive_validation():
    config = RunConfig.model_validate({
        "environment": {
            "wrappers": [
                {"name": "frame_stack", "stack_size": 5},
                {"name": "max_and_skip", "skip": 3},
            ],
        },
        "callbacks": {"episode_statistics": False, "record_video": False},
    })

    assert config.environment.wrappers[0].stack_size == 5
    assert config.environment.wrappers[1].skip == 3


def test_evaluation_settings_use_training_env_count_and_disable_reward_normalization():
    config = RunConfig.model_validate({
        "environment": {"num_envs": 10, "normalize_rewards": True},
        "callbacks": [
            {
                "name": "evaluation",
                "interval": 25_000,
                "episodes": 10,
                "final_episodes": 1_000,
            },
        ],
    })

    settings = config.evaluation_settings()

    assert settings.environment.num_envs == 10
    assert settings.environment.normalize_rewards is False
    assert settings.environment.record_video is False
    assert all(wrapper.name != "reward_on_done" for wrapper in settings.environment.wrappers)
    assert settings.environment.wrappers[0].name == "record_episode_stats"
    assert (settings.interval, settings.episodes, settings.final_episodes) == (
        25_000,
        10,
        1_000,
    )
    assert config.callback("record_video") is None
