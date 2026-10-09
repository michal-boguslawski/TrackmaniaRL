# Usage and development instructions

This guide contains the detailed commands for installation, training,
evaluation, MLflow, profiling, tests, and benchmarks. Commands assume the
repository root as the working directory.

## Install

Python 3.13 or newer is required. Select exactly one PyTorch build; the CPU and
GPU extras are mutually exclusive. The GPU build uses the PyTorch CUDA 13.0
index and requires a compatible NVIDIA driver and GPU.

```bash
# NVIDIA GPU
uv sync --extra gpu

# CPU only (use this instead of the GPU command)
uv sync --extra cpu
```

Add the `dev` extra when running tests or using development dependencies. Use
the same CPU/GPU extra as the environment for every `uv run` command; switching
extras can re-sync and replace the installed PyTorch build.

## Train

The default entrypoint uses
[`src/rl_lib/config/ppo_carracing.yaml`](../src/rl_lib/config/ppo_carracing.yaml):

```bash
uv run --extra gpu python scripts/train.py
```

Replace `gpu` with `cpu` for a CPU-only installation. The default run is
intentionally long: 3,000,000 vector-environment steps across 16 environments
represent 48,000,000 individual environment transitions. `run.total_steps`
counts calls to vector `env.step`, not transitions multiplied across
environments. Copy the YAML config to customize an experiment; it is validated
by strict Pydantic models, so unknown fields and invalid values are rejected.

```bash
uv run --extra gpu python scripts/train.py \
  --config path/to/your_config.yaml
```

`cnn_autocast: bf16`, `channels_last`, and the fused optimizer are CUDA-only.
The sample config currently sets `cnn_autocast: none`; if enabling BF16, use a
compatible CUDA device. For the full set of command-line options:

```bash
uv run --extra gpu python scripts/train.py --help
```

## Short diagnostic runs and profiling

Two small configurations are provided for diagnostics; neither is a substitute
for a full training result.

- `manual_smoke.yaml` runs 8 vector steps with 2 environments (16 transitions
  and two small PPO updates). It exercises the short-run training and artifact
  workflow. Its callbacks include checkpointing and video recording.
- `manual_profile.yaml` uses the production network with a 128-step rollout,
  minibatches of 64, and no evaluation or periodic video/checkpoint callbacks.
  Its default profiler schedule captures the first PPO update. Profiling adds
  overhead, so use it for bounded diagnostic captures, not normal training.

```bash
# Short diagnostic run
uv run --extra cpu python scripts/train.py \
  --config src/rl_lib/config/manual_smoke.yaml --no-profile

# Profile one PPO update
uv run --extra cpu python scripts/train.py \
  --config src/rl_lib/config/manual_profile.yaml --profile
```

Use `--profile` or `--no-profile` to override the profiling config. The separate
`--profiling-config` option selects a profiling YAML file. Traces are written
under `logs/profiles/<session-id>/`; open them with Perfetto or Chrome tracing.
Named regions cover policy inference, environment stepping, rollout writes,
PPO update phases, and device transfers.

## Evaluate a checkpoint

The standalone evaluator supports MLflow runs and local checkpoints. For an
MLflow checkpoint or the run's `final_model` artifact:

```bash
# Evaluate a particular checkpoint step
uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID \
  --checkpoint-step 2200000 --episodes 10

# Evaluate the MLflow final_model artifact
uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID --episodes 10
```

For a local checkpoint, provide the checkpoint and the YAML config that
produced it:

```bash
uv run --extra gpu python scripts/evaluate.py --source local \
  --config src/rl_lib/config/ppo_carracing.yaml \
  --checkpoint logs/artifacts/SESSION_ID/final_state_dict.pt \
  --episodes 10 --record-video
```

Evaluation runs real episodes and may take substantially longer than a smoke
run. By default the evaluator uses raw environment rewards rather than the
training reward normalizer.

## MLflow tracking

Training writes to the local SQLite tracking store `mlflow.db` by default.
Start the repository's MLflow server for the browser UI and HTTP clients:

```bash
MLFLOW_HOST=127.0.0.1 scripts/mlflow_server.sh start
scripts/mlflow_server.sh status
scripts/mlflow_server.sh logs
scripts/mlflow_server.sh stop
```

Then open `http://127.0.0.1:5000/`. To evaluate through the server, point the
client at the same tracking store:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID --episodes 10
```

The server provides read/write access and has no authentication. Keep it bound
to loopback or behind a firewall and use an SSH tunnel for remote access; do
not expose its port directly to the public internet. A generic local forward
looks like this (replace the placeholders with your own connection details):

```bash
ssh -N -L 5001:127.0.0.1:5000 <user>@<server> -p <ssh-port>
```

Browse to `http://localhost:5001/` while the tunnel is active. MLflow's host
and CORS allowlists compare hostnames including ports; if you customize them,
keep their entries aligned with the address used by the browser. Do not use
wildcard allowlists as a security workaround.

Run outputs are stored under `logs/artifacts/<session-id>/`, including model
checkpoints, the resolved `config/run_config.yaml`, parameters, and final
evaluation results. Application logs and recorded videos are under `logs/`.

## Tests and build

Pytest discovers tests under `tests/` and reads `pytest.toml`. The commands
below use the CPU extra; substitute `gpu` for `cpu` if using the GPU build.
Install the development extra alongside your chosen PyTorch extra, then run:

```bash
uv run --extra cpu --extra dev pytest
```

Useful focused test commands:

```bash
uv run --extra cpu --extra dev pytest tests/unit/test_ppo_trainer.py
uv run --extra cpu --extra dev pytest tests/unit/test_rollout_collector.py
uv run --extra cpu --extra dev pytest tests/unit/test_run_config.py
```

The suite currently has a strict expected failure for reading completed-episode
statistics from `info["final_info"]` under Gymnasium's `SAME_STEP` autoreset
mode. Build the package with:

```bash
uv build
```

## Performance benchmarks

These are focused tools for component-level investigation, not full-training
launchers. Each script documents its options with `--help`:

```bash
uv run --extra cpu python -m scripts.benchmarks.rollout_buffer --help
uv run --extra cpu python -m scripts.benchmarks.metric_verbosity --help
uv run --extra cpu python -m scripts.benchmarks.cnn_bf16 --help
uv run --extra cpu python -m scripts.benchmarks.compare_reports --help
```

`compare_reports.py` compares saved benchmark reports and requires both a
baseline and candidate report. See the [PPO metric reference](metrics.md) for
metric definitions and diagnostic cross-checks.
