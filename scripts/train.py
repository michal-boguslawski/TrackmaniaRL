"""Training entrypoint script.

Loads configuration from YAML and runs the PPO training pipeline.
Uses package resource to locate default config file.
"""

import argparse
from importlib import resources

from rl_lib.train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO agent using a YAML configuration")
    default_config = str(resources.files("rl_lib.config") / "ppo_carracing.yaml")
    parser.add_argument("--config", default=default_config, help="YAML run configuration")
    main(parser.parse_args().config)
