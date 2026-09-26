from importlib import resources
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = str(resources.files("rl_lib.config") / "ppo_carracing.yaml")


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default if default is not None else "")


def _train() -> None:
    from rl_lib.train import main as train

    config_path = _ask("Training config", DEFAULT_CONFIG)
    train(config_path)


def _evaluate() -> None:
    from rl_lib.evaluation import evaluate_checkpoint, evaluate_mlflow
    from rl_lib.run_config import load_config
    from rl_lib.tracking.console_logger import ConsoleMetricsLogger

    source = _ask("Model source: 1 local checkpoint, 2 MLflow", "1")
    episodes = _positive_int("Number of episodes", "5")
    num_envs = _positive_int("Number of evaluation environments")
    record_video = _ask("Record video? (y/N)", "N").lower() in {"y", "yes"}
    metrics_loggers = [ConsoleMetricsLogger()]

    if source == "1":
        config_path = _ask("Config", DEFAULT_CONFIG)
        checkpoint_path = _ask("Checkpoint path")
        while not checkpoint_path:
            logger.warning("Please enter a checkpoint path.")
            checkpoint_path = _ask("Checkpoint path")
        evaluate_checkpoint(
            config_path,
            checkpoint_path,
            episodes=episodes,
            record_video=record_video,
            num_envs=num_envs,
            metrics_loggers=metrics_loggers,
        )
        return

    if source == "2":
        experiment_default = "CarRacing-v3"
        try:
            experiment_default = load_config(DEFAULT_CONFIG).experiment_name
        except ValueError:
            logger.debug("Default config unavailable; using the default experiment name")
        experiment_name = _ask("MLflow experiment", experiment_default)
        run_id = _ask("MLflow run ID / URI (blank to choose from available models)") or None
        evaluate_mlflow(
            run_id=run_id,
            experiment_name=experiment_name,
            episodes=episodes,
            num_envs=num_envs,
            record_video=record_video,
            metrics_loggers=metrics_loggers,
        )
        return

    raise ValueError("Choose model source 1 (local checkpoint) or 2 (MLflow)")


def _positive_int(prompt: str, default: str | None = None) -> int:
    while True:
        raw = _ask(prompt, default)
        try:
            value = int(raw)
            if value < 1:
                raise ValueError
            return value
        except ValueError:
            logger.warning("Enter a positive whole number for %s.", prompt.lower())


def main() -> None:
    """Interactive entry point for training and checkpoint evaluation."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG)
    actions = {"1": ("Train PPO", _train), "2": ("Evaluate checkpoint", _evaluate)}
    while True:
        print("\nrl-lib")
        for key, (label, _) in actions.items():
            print(f"  {key}. {label}")
        print("  3. Exit")
        try:
            choice = input("Select an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice == "3":
            return
        if choice not in actions:
            print("Choose 1, 2, or 3.")
            continue

        try:
            actions[choice][1]()
        except KeyboardInterrupt:
            print("\nOperation interrupted.")
        except Exception as exc:
            logger.exception("Operation failed: %s", exc)
