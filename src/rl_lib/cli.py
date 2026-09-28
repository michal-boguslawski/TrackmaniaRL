"""Interactive CLI for training and evaluation.

Provides a text-based menu for:
1. Training a PPO agent from config
2. Evaluating a local checkpoint or MLflow model
"""

from importlib import resources
import logging

from rl_lib.logger_setup import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = str(resources.files("rl_lib.config") / "ppo_carracing.yaml")


def _flush_logs() -> None:
    """Drain the queued log records so output is on screen before a prompt.

    ``config/logging.yaml`` routes records through a QueueListener thread, so
    records logged just before a prompt can be written after it, which reads as
    a prompt answering itself. ``QueueHandler.flush()`` is a no-op in the
    standard library, so the listener is stopped (which drains the queue and
    joins the thread) and then restarted.
    """
    handler = logging.getHandlerByName("queue_handler")
    listener = getattr(handler, "listener", None)
    if listener is None or listener._thread is None:
        return
    listener.stop()
    listener.start()


def _ask(prompt: str, default: str | None = None) -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default if default is not None else "")


def _prompt_for_input(prompt: str) -> str:
    """Read one line for a library prompt, draining queued log output first.

    Used as the ``input_fn`` for MLflow model selection, where the run list is
    logged immediately before the prompt. The prompt string already carries its
    own colon, so this does not go through :func:`_ask`.
    """
    _flush_logs()
    return input(prompt).strip()


def _train() -> None:
    """Run training from config path."""
    from rl_lib.train import main as train

    config_path = _ask("Training config", DEFAULT_CONFIG)
    train(config_path)


def _evaluate() -> None:
    """Run evaluation of local checkpoint or MLflow model."""
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
            input_fn=_prompt_for_input,
        )
        return

    raise ValueError("Choose model source 1 (local checkpoint) or 2 (MLflow)")


def _positive_int(prompt: str, default: str | None = None) -> int:
    """Prompt for positive integer with validation."""
    while True:
        raw = _ask(prompt, default)
        try:
            value = int(raw)
            if value < 1:
                raise ValueError
            return value
        except ValueError:
            logger.warning("Enter a positive whole number for %s.", prompt.lower())


def _configure_logging() -> None:
    """Configure root logging for the interactive session.

    Uses the project's own logging setup so the interactive prompts and the
    training run share one configuration: the packaged ``config/logging.yaml``
    puts a DEBUG console handler on stdout and a rotating file handler on
    ``logs/app.log``, and injects the ``session_id`` its formatter references.

    Reconfiguring is required rather than guarded on "root has no handlers
    yet": importing ``rl_lib`` transitively imports MLflow, whose
    PyTorch-Lightning autologging module calls ``logging.basicConfig`` at
    import time, so root already has a WARNING-only handler by the time this
    runs and the INFO records that list selectable runs would be dropped.
    """
    setup_logging()


def main() -> None:
    """Interactive entry point for training and checkpoint evaluation."""
    _configure_logging()
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
