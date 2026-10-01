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
sample `src/rl_lib/config/ppo_carracing.yaml` lists the run, environment and
wrappers, agent, rollout, network, PPO trainer, tracking, and callback settings.
Missing fields use the defaults in `rl_lib.run_config`; unknown fields and
invalid values are rejected.

```bash
uv run --extra gpu python scripts/train.py
```

`run.total_steps` means calls to vector `env.step`, not total transitions across
all parallel environments. In the production sample, 3,000,000 vector steps
across 16 environments correspond to 48,000,000 environment transitions. Copy
the sample YAML to create another run configuration.

`cnn_autocast: bf16`, `channels_last`, and `fused_optimizer` are CUDA-only:
`bf16` raises on a CPU device, so set `cnn_autocast: none` for CPU runs.

## Commands

All commands assume the repository root as the working directory. Choose exactly
one PyTorch extra; add `--extra dev` when you need pytest and other development
dependencies.

### Install, test, and build

```bash
uv sync --extra cpu
uv sync --extra gpu

uv run --extra cpu --extra dev pytest
uv run --extra cpu --extra dev pytest tests/unit/test_ppo_trainer.py tests/unit/test_rollout_collector.py
uv run --extra cpu --extra dev pytest tests/unit/test_run_config.py

uv build
```

Pytest discovers tests under `tests/` and reads `pytest.toml`. Use the same
extra as the environment you trained in; a CUDA benchmark needs `--extra gpu`.

### Train and profile

```bash
# Default production config; profiler off by default.
uv run --extra gpu python scripts/train.py --no-profile

# Explicit config and profiler overrides.
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --no-profile
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --profile
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_profile.yaml --profile
uv run --extra cpu python scripts/train.py --help
```

Profiling is disabled in the packaged config; omit `--profile` for a normal run,
or use `--no-profile` to make the choice explicit. `--config` also selects a run
configuration, and `--profiling-config` a separate profiler configuration.

Two short configs ship for diagnostics:

- `manual_smoke.yaml` — 8 vector steps with 2 environments (16 transitions, 2
  small PPO updates). It proves the profiler wiring, not throughput, and it
  enables MLflow plus periodic and final checkpoints while omitting video and
  scheduled evaluation.
- `manual_profile.yaml` — an informative capture: the `ppo_carracing.yaml`
  network verbatim, a 128-step rollout, a minibatch of 64, and no evaluation,
  video, or periodic checkpoint callbacks. The default schedule resolves to 126
  wait, 1 warmup, and 2 active vector steps, so the recorded window is the first
  PPO update (4 minibatches of forward, backward, and optimizer work).

Traces are written to `./logs/profiles/<session-id>/` as `.pt.trace.json`; open
them in Perfetto or Chrome tracing. Named regions cover policy inference,
environment step, buffer add, PPO forward/backward/optimizer, and device
transfers. Profiling adds overhead, so keep it enabled for short captures only.
`scripts/train.py` with the default config is a long run, not a smoke check.

### MLflow and interactive evaluation

```bash
# Browse the local ./mlruns store (no tracking server needed).
uv run --extra cpu mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

# Only if you need a tracking HTTP server; the UI does not require one.
uv run --extra cpu mlflow server --host 127.0.0.1 --port 5000

# Interactive checkpoint evaluation.
uv run --extra cpu rl-lib

# Evaluate a selected checkpoint directly, without creating a script.
uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID \
  --checkpoint-step 2200000 --episodes 10

# Evaluate the MLflow final_model artifact instead of a periodic checkpoint.
uv run --extra gpu python scripts/evaluate.py --run-id RUN_ID --episodes 10
```

Set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` before training or evaluating to
use an HTTP tracking server.

`rl-lib` offers checkpoint evaluation. Choose MLflow as the source and enter the
experiment name and run ID, or choose a local checkpoint and point it at
`logs/artifacts/<session-id>/final_state_dict.pt` together with the YAML config
that produced it. For MLflow evaluation, the interactive prompt accepts a
checkpoint step; leave it blank to evaluate `final_model`. The standalone
`scripts/evaluate.py` command supports both MLflow and local checkpoints; run
`uv run --extra gpu python scripts/evaluate.py --help` for its options. Evaluation
runs real episodes, so it takes longer than a smoke training run.

Locally, `logs/artifacts/<session-id>/` receives the `.pt` model and
state-dictionary artifacts and `logs/app.log` records run startup and training
messages. Small run metadata is saved alongside them:
`config/run_config.yaml`, `config/parameters.json`, and, after final evaluation,
`evaluation/final_results.json` (including a reference to the saved config).
These metadata files are visible to Git; model artifacts, videos, application
logs, and the `mlruns` store remain ignored. The session ID is printed in the
application log.

### Performance benchmark scripts

Focused performance tools, not full-training launchers. Each exposes its full
options via `--help`:

```bash
uv run --extra cpu python -m scripts.benchmarks.rollout_buffer --help
uv run --extra cpu python -m scripts.benchmarks.metric_verbosity --help
uv run --extra cpu python -m scripts.benchmarks.cnn_bf16 --help
uv run --extra cpu python -m scripts.benchmarks.compare_reports --help
```

`scripts/benchmarks/compare_reports.py` compares two saved benchmark reports and
requires paths to a baseline and a candidate report. Unit tests remain the
fastest broad regression check.
