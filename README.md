# Trackmania RL

## Install

Python 3.13 or newer is required. Select exactly one PyTorch build when syncing:

```bash
# NVIDIA CUDA 13.0 build (GPU training)
uv sync --extra gpu

# CPU-only build (no CUDA runtime packages)
uv sync --extra cpu
```

The `cpu` and `gpu` extras are mutually exclusive. When running commands without
syncing first, select the same extra, for example:

```bash
uv run --extra cpu pytest
uv run --extra gpu python scripts/train.py
```

The GPU build uses the official PyTorch CUDA 13.0 wheel index; a compatible
NVIDIA driver and GPU are needed to execute CUDA workloads. The CPU build uses
the official PyTorch CPU wheel index.

## Training configuration

Training is configured with a validated Pydantic model loaded from YAML. The
sample `configs/ppo_carracing.yaml` lists the run, environment and wrappers,
agent, rollout, network, PPO trainer, tracking, and callback settings. Missing
fields use the defaults in `rl_lib.run_config`; unknown fields and invalid
values are rejected.

```bash
uv run --extra gpu python scripts/train.py --config configs/ppo_carracing.yaml
```

`run.total_steps` counts vector-environment steps (one call to `env.step`), not
the aggregate number of transitions across all environments. Copy the sample
YAML to create another run configuration.
