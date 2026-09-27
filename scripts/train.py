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
    default_profiling_config = str(resources.files("rl_lib.config") / "profiling.yaml")
    parser.add_argument("--config", default=default_config, help="YAML run configuration")
    parser.add_argument(
        "--profiling-config",
        default=default_profiling_config,
        help="Separate YAML configuration for optional PyTorch profiling",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=None,
        help="Enable profiling for this run, overriding profiling config",
    )
    parser.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="Disable profiling for this run, overriding profiling config",
    )
    args = parser.parse_args()
    main(args.config, profiling_config_path=args.profiling_config, profile=args.profile)
