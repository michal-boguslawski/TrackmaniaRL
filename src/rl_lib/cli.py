from pathlib import Path


DEFAULT_CONFIG = str(Path(__file__).parent / "config" / "ppo_carracing.yaml")


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default if default is not None else "")


def _train() -> None:
    from rl_lib.train import main as train

    config_path = _ask("Training config", DEFAULT_CONFIG)
    train(config_path)


def _evaluate() -> None:
    from rl_lib.evaluate import evaluate_checkpoint

    config_path = _ask("Config", DEFAULT_CONFIG)
    checkpoint_path = _ask("Checkpoint path")
    while not checkpoint_path:
        print("Please enter a checkpoint path.")
        checkpoint_path = _ask("Checkpoint path")

    while True:
        raw_episodes = _ask("Number of episodes", "5")
        try:
            episodes = int(raw_episodes)
            if episodes < 1:
                raise ValueError
            break
        except ValueError:
            print("Enter a positive whole number of episodes.")

    record_video = _ask("Record video? (y/N)", "N").lower() in {"y", "yes"}
    evaluate_checkpoint(config_path, checkpoint_path, episodes, record_video)


def main() -> None:
    """Interactive entry point for training and checkpoint evaluation."""
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
            print(f"Operation failed: {exc}")
