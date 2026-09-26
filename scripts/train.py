import argparse

from rl_lib.train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the PPO agent using a YAML configuration")
    parser.add_argument("--config", default="configs/ppo_carracing.yaml", help="YAML run configuration")
    main(parser.parse_args().config)
