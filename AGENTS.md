# Agent instructions

## Build and test
- Python 3.13+ is required (`.python-version`); choose one PyTorch build with `uv sync --extra gpu` or `uv sync --extra cpu`.
- The GPU extra installs PyTorch's CUDA 13.0 build and its CUDA dependencies; the CPU extra uses CPU-only wheels.
- Build the package with `uv build` (produces an sdist and wheel via `uv_build`).
- Run tests with the same extra as the environment, e.g. `uv run --extra cpu pytest` or `uv run --extra gpu pytest`; run one file with `uv run --extra gpu pytest tests/test_actor.py`. Pytest discovers tests under `tests/` and reads `pytest.toml`.
- Tests are unit-level only; there is no end-to-end training or environment test suite. The current rollout-buffer and PPO-trainer tests use call signatures/shapes that do not match the implementations, so reconcile them before treating their results as a source regression.
- No CI workflow or configured lint, formatter, or type-check command is present; don't assume one is required.

## Implementation map
- Training flow: `scripts/train.py` wires `make_env` → `Agent`/`Network` + `RolloutBuffer` → `RolloutCollector` → `PPOTrainer`; collector callbacks handle rollout stats/video/params, and trainer callbacks handle minibatch metrics/checkpoints.
- Model flow: `Agent` converts uint8 NHWC observations to normalized NCHW tensors and maintains per-environment temporal windows; `Network` composes CNN → `TemporalCNN1D` → Beta-distribution actor and value critic. The actor affinely maps Beta samples to steering `[-1, 1]` and gas/brake `[0, 1]`.
- `RolloutBuffer` owns GAE. Preserve Gymnasium's distinction: bootstrap through truncation but stop advantage recursion at either termination or truncation. Update buffer tests when changing this API.
- Environment wrappers are registered in `src/rl_lib/envs/wrappers/registry.py`; `make_env` accepts registry keys and creates vector environments. Add new wrapper names there before using them in a training config/script.
- The script and `mlflow_logger.py` import via `src.rl_lib.*`, while most package code uses `rl_lib.*`. Prefer the installed package path `rl_lib.*` for new package imports; importing the same modules through both names can create duplicate module/class identities.

## Entrypoints and plans
- The installable package is `src/rl_lib`. The `rl-lib` console script currently calls the placeholder `rl_lib.main()`; it does not start training.
- The training assembly is `scripts/train.py`; executing it starts a long CarRacing-v3 PPO run (3,000,000 steps, 16 environments) and writes run artifacts. Don't use it as a build/smoke check.
- `rl-portfolio-project-spec.md` records the intended CarRacing PPO/SAC stage and later TrackMania/TMRL goal; `stage1-architecture.md` proposes a PPO/SAC layout, configs, and design assumptions. These are plans, not proof those pieces exist: current training is CarRacing/PPO, with no SAC/TD3 or TrackMania implementation. Follow source/config over the proposed tree. `README.md` is empty.
