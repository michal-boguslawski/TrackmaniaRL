# Manual training validation and command cheatsheet

Use this checklist after installing one PyTorch extra. Commands below assume the
repository root as the working directory. The short run config is
`src/rl_lib/config/manual_smoke.yaml`; it performs 8 vector-environment steps
with 2 environments, giving 16 environment transitions and 2 small PPO updates.
It retains the normal network and environment wrappers, enables MLflow, and logs
periodic and final checkpoints. It deliberately leaves video and scheduled
evaluation callbacks out of the fast smoke run.

## 1. Prepare MLflow and run a short training job

For a local MLflow file store, use two terminals. The run defaults to MLflow's
local `./mlruns` store; the UI reads the same store:

```bash
# Terminal 1
uv run --extra cpu mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

# Terminal 2: one full run without an attached profiler
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --no-profile
```

Open <http://127.0.0.1:5000>, select the `Trackmania-RL-manual-smoke`
experiment, and verify:

- [x] A new run appears and finishes with status **FINISHED**.
- [x] Parameters include the run configuration and flattened training/network
  settings; `config/run_config.yaml` is present as an artifact.
- [x] Training metrics appear against steps (for example PPO loss/entropy and
  update diagnostics); there are metrics at the rollout update boundaries.
- [x] Artifacts include a periodic checkpoint under `checkpoints/step_...` and
  `final_state_dict`. The model is logged under the run's `final_model` name in
  the MLflow model registry (MLflow 3 stores it under
  `mlruns/<exp>/models/m-.../artifacts`, not inside the run's artifact
  directory).
- [x] Locally, `logs/artifacts/<session-id>/` also contains the `.pt` model and
  state-dictionary artifacts. The session ID is printed in the application log.
- [x] `logs/app.log` contains the run startup and training messages.

The local MLflow UI is for inspecting metrics and downloaded artifacts. For a
remote tracking server, set `MLFLOW_TRACKING_URI` on the training command, for
example:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --no-profile
```

Use the forwarded/server URI in place of `127.0.0.1:5000` as needed. Start a
server with `uv run --extra cpu mlflow server --host 127.0.0.1 --port 5000`
only if you need a tracking HTTP server; local `mlflow ui` does not need one.

## 2. Load/evaluate what the run logged

In a new terminal, use the interactive entry point:

```bash
uv run --extra cpu rl-lib
```

Choose **2. Evaluate checkpoint**, then choose MLflow as the source, enter
`Trackmania-RL-manual-smoke`, select the run ID from the UI (or leave it blank
to list runs and choose its prefix), request **1** episode and **1** evaluation
environment, and answer **N** for video. Check that it loads `config/run_config.yaml`
and `final_model`, completes inference, and reports evaluation metrics/return.
The prompt is also suitable for selecting a local checkpoint instead: choose
source 1 and point it at
`logs/artifacts/<session-id>/final_state_dict.pt`, using the same smoke YAML.

- [x] MLflow model discovery lists the completed run.
- [x] Model and saved run config load without error; inference completes.
- [x] Evaluation returns and `evaluation/mlflow/...` metrics are reported.

This evaluation executes a real CarRacing episode, so it takes longer than the
8-step training smoke run.

## 3. Verify the profiler

The packaged `src/rl_lib/config/profiling.yaml` is the profiling config. It is
disabled by default, uses shape and memory recording, writes to
`./logs/profiles/<session-id>/`, and automatically positions its schedule around
the first rollout update. No profiling-config edit is needed for the smoke YAML.

```bash
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --profile
```

With the smoke rollout size of 4, the default schedule resolves to 2 wait steps,
1 warmup step, and 2 recorded steps. The collector advances that schedule once
per vector-environment step. **8 total vector steps** are sufficient to capture
the first PPO update and write the trace. GPU profiling uses the same command
with `--extra gpu` and also records CUDA activity when CUDA is available.

- [x] Training completes and reports the profile output directory.
- [x] A `.pt.trace.json` trace exists under `logs/profiles/<session-id>/`.
- [x] Open the trace in Perfetto or Chrome tracing and inspect CPU (and CUDA,
  on GPU) activity. Named regions include policy inference, environment step,
  rollout buffer/update, PPO forward/backward/optimizer, and transfers.

This 8-step capture validates profiler wiring, not representative throughput.
For an informative profile, retain the production network, increase the rollout
size (for example 128), set `minibatch_size` to 64, and set `total_steps` to at
least 132. Automatic alignment then records the first rollout update; profiling
adds overhead, so keep it enabled only for short diagnostic captures.

`src/rl_lib/config/manual_profile.yaml` is that config, so the recipe does not
have to be retyped:

```bash
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_profile.yaml --profile
```

It copies `ppo_carracing.yaml`'s network verbatim and sets a 128-step rollout, a
minibatch of 64, and 132 total steps. The schedule resolves to 126 wait, 1
warmup, and 2 active vector steps, so the recorded window is the first PPO
update: 4 minibatches of forward, backward, and optimizer work, versus the single
minibatch in the smoke trace. It omits the evaluation, video, and periodic
checkpoint callbacks, which would otherwise add work the profiler would charge
to the capture.

## 4. Optional artifact checks

- [x] For a separate video test, add `- name: record_video` to the smoke config's
  callbacks and rerun it. Verify video artifacts appear under `videos/` in
  MLflow. The callback records at step 0 and again at rollout end, so this runs
  two full evaluation episodes and can take substantially longer than the smoke
  training job.
- [x] If using a remote tracking server, repeat sections 1–2 with
  `MLFLOW_TRACKING_URI` set and verify the server can serve/download the
  `final_model` and config artifacts.
- [x] Try a second run and confirm it creates a distinct MLflow run while using
  the same experiment.

## Commands

### Install, test, train, profile, and build

Choose exactly one PyTorch extra. Add `--extra dev` when you need pytest and
other development dependencies.

```bash
uv sync --extra cpu
uv sync --extra gpu

uv run --extra cpu --extra dev pytest
uv run --extra cpu --extra dev pytest tests/unit/test_mlflow_logger.py tests/unit/test_mlflow_evaluation.py tests/unit/test_local_artifact_logger.py tests/unit/test_model_checkpoint_callback.py
uv run --extra cpu --extra dev pytest tests/unit/test_run_config.py
uv build

# Default production config (3,000,000 vector-environment steps, 16 envs); profiler is off by default.
uv run --extra gpu python scripts/train.py --no-profile

# Explicit config and profiler overrides.
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --no-profile
uv run --extra cpu python scripts/train.py --config src/rl_lib/config/manual_smoke.yaml --profile
uv run --extra cpu python scripts/train.py --help
```

`run.total_steps` means calls to vector `env.step`, not total transitions across
all parallel environments. In the production sample, 3,000,000 vector steps
across 16 environments correspond to 48,000,000 environment transitions.
Profiling is disabled in the packaged config; omit `--profile` for a normal
training run, or use `--no-profile` to make the choice explicit.

### MLflow UI and interactive evaluation

```bash
uv run --extra cpu mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
uv run --extra cpu mlflow server --host 127.0.0.1 --port 5000
uv run --extra cpu rl-lib
```

Set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` before training/evaluation to
use an HTTP tracking server. The UI and the server are separate commands: use
`mlflow ui` to browse a local store, and `mlflow server` when a tracking API
server is needed.

### Performance benchmark scripts

These scripts expose their full options via `--help`:

```bash
uv run --extra cpu python -m scripts.benchmarks.rollout_buffer --help
uv run --extra cpu python -m scripts.benchmarks.metric_verbosity --help
uv run --extra cpu python -m scripts.benchmarks.compare_reports --help
```

`scripts/benchmarks/compare_reports.py` compares two saved benchmark reports;
it requires paths to a baseline and candidate report. Unit tests remain the
fastest broad regression check; these scripts are focused performance tools,
not full-training launchers.
