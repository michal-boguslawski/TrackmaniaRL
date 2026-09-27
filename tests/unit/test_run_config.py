import pytest
from pydantic import ValidationError

from rl_lib.buffers.rollout_buffer import RolloutBuffer
from rl_lib.run_config import RolloutSettings, RunConfig, WrapperSettings, load_config
from rl_lib.training.ppo.trainer import PPOTrainer


def test_sample_yaml_loads_full_current_run_defaults():
    from importlib import resources
    config_path = str(resources.files("rl_lib.config") / "ppo_carracing.yaml")
    config = load_config(config_path)

    assert config.environment.id == "CarRacing-v3"
    assert config.environment.num_envs == 16
    assert config.run.algorithm == "ppo"
    assert config.run.total_steps == 3_000_000
    assert config.rollout.gamma == pytest.approx(0.99)
    assert config.trainer.ppo_epsilon == pytest.approx(0.1)
    assert config.network.cnn.conv_layers[0].kernel_size == 8
    assert config.callback("checkpoints").interval == 200
    assert config.callback("evaluation").final_episodes == 1_000
    assert config.tracking.verbosity == 2


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


def test_run_config_accepts_a_registered_algorithm_name(tmp_path):
    path = tmp_path / "algorithm.yaml"
    path.write_text("run:\n  algorithm: sac\n", encoding="utf-8")

    config = load_config(str(path))

    assert config.run.algorithm == "sac"


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
    with pytest.raises(ValidationError):
        WrapperSettings.model_validate({"name": "not_registered"})

    with pytest.raises(ValidationError, match="not supported"):
        WrapperSettings.model_validate({"name": "grayscale", "stack_size": 3})

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
    assert trainer.cfg.backbone_lr == pytest.approx(0.0002)
    assert trainer.cfg.head_lr == pytest.approx(0.0003)
    assert trainer._optimizer.__class__.__name__ == "Adam"
    assert trainer.cfg.backbone_max_grad_norm == pytest.approx(0.75)
    assert trainer.cfg.entropy_coef_min == pytest.approx(0.00002)


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
