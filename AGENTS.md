# Agent instructions

## Documentation Standards

### Docstring Format
Use **NumPy style** docstrings (compatible with Sphinx/numpydoc) for all public classes, methods, and functions. Private methods (leading underscore) should have a brief one-line docstring.

**Module-level docstring** (top of every .py file):
```python
"""Brief one-line summary.

Extended description if needed, explaining the module's purpose,
key abstractions, and how it fits into the training pipeline.
"""
```

**Class docstring:**
```python
class ClassName:
    """One-line summary of the class.

    Extended description covering responsibilities, key attributes,
    and usage patterns. Include shape conventions for tensor-heavy classes.

    Attributes:
        attr_name (type): Description of attribute.
    """
```

**Method/Function docstring:**
```python
def method_name(self, param1: Type, param2: Type = default) -> ReturnType:
    """One-line summary.

    Extended description if needed. Explain side effects, mutability,
    and any non-obvious behavior.

    Args:
        param1 (Type): Description including expected shape/range.
        param2 (Type): Description with default value.

    Returns:
        ReturnType: Description including shape and semantics.

    Raises:
        ExceptionType: Condition that raises it.
    """
```

**Tensor shape annotations**: Always document tensor shapes in Args/Returns using the convention:
- `(batch, ...)` for batched inputs
- `(batch, seq, feature)` for sequences
- `(batch, channel, height, width)` for images (NCHW)
- `(batch, height, width, channel)` for raw observations (NHWC)

### Inline Comments
- Use sparingly for non-obvious logic, mathematical derivations, or subtle bugs avoided
- Prefer clear variable names and type hints over comments
- Comment "why" not "what"
- Use `#` for single-line, `"""..."""` for multi-line explanations within functions

### Type Hints
- All public APIs must have complete type annotations
- Use `from __future__ import annotations` for forward references
- Prefer `T.Tensor` over `torch.Tensor` (import torch as T)
- Use `NDArray` from `numpy.typing` for numpy arrays

### Naming Conventions
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Private: `_leading_underscore`
- Constants: `UPPER_SNAKE_CASE`
- Type variables: `T_` prefix (e.g., `T_Observation`)

### Config/Validation Classes (Pydantic)
- Document each field with `Field(description="...")` for schema generation
- Add model-level docstring explaining the configuration section

---

## Build and test
- Python 3.13+ is required (`.python-version`); choose one PyTorch build with `uv sync --extra gpu` or `uv sync --extra cpu`.
- The GPU extra installs PyTorch's CUDA 13.0 build and its CUDA dependencies; the CPU extra uses CPU-only wheels.
- Build the package with `uv build` (produces an sdist and wheel via `uv_build`).
- Run tests with the same extra as the environment, e.g. `uv run --extra cpu pytest` or `uv run --extra gpu pytest`; run one file with `uv run --extra gpu pytest tests/unit/test_actor.py`. Pytest discovers tests under `tests/` and reads `pytest.toml`.
- Tests are unit-level only; there is no end-to-end training or environment test suite. The suite is green, with one strict xfail left: `RecordStatisticLoggerCallback` does not read episode stats from `info["final_info"]`, which is where `SAME_STEP` autoreset puts them.
- No CI workflow or configured lint, formatter, or type-check command is present; don't assume one is required.

## Implementation map
- Training flow: `scripts/train.py` wires `make_env` → `Agent`/`Network` + `RolloutBuffer` → `RolloutCollector` → `PPOTrainer`; collector callbacks handle rollout stats/video/params, and trainer callbacks handle minibatch metrics/checkpoints.
- Model flow: `Agent` converts uint8 NHWC observations to normalized NCHW tensors and maintains per-environment temporal windows; `Network` composes CNN → `TemporalCNN1D` → Beta-distribution actor and value critic. The actor affinely maps Beta samples to steering `[-1, 1]` and gas/brake `[0, 1]`.
- `RolloutBuffer` owns GAE. Preserve Gymnasium's distinction: bootstrap through truncation but stop advantage recursion at either termination or truncation. Update buffer tests when changing this API.
- The buffer stores the flags produced by stepping *at* a step, but `get()` returns `dones` shifted one column so they mean "this frame is the first of a new episode" — the convention `Agent._get_mask_window` uses while collecting. Do not "simplify" that shift away: the observation and done windows have to meet on the same column or PPO evaluates a different history than the action was taken with.
- Environment wrappers are registered in `src/rl_lib/envs/wrappers/registry.py`; `make_env` accepts registry keys and creates vector environments. Add new wrapper names there before using them in a training config/script.
- Every module imports via `rl_lib.*`, including `scripts/train.py` and `mlflow_logger.py`. Keep it that way: importing the same modules through both `rl_lib.*` and `src.rl_lib.*` creates duplicate module/class identities.

## Entrypoints and plans
- The installable package is `src/rl_lib`. The `rl-lib` console script currently calls the placeholder `rl_lib.main()`; it does not start training.
- The training assembly is `scripts/train.py`; executing it starts a long CarRacing-v3 PPO run (3,000,000 steps, 16 environments) and writes run artifacts. Don't use it as a build/smoke check.
- `rl-portfolio-project-spec.md` records the intended CarRacing PPO/SAC stage and later TrackMania/TMRL goal; `stage1-architecture.md` proposes a PPO/SAC layout, configs, and design assumptions. These are plans, not proof those pieces exist: current training is CarRacing/PPO, with no SAC/TD3 or TrackMania implementation. Follow source/config over the proposed tree. `README.md` is empty.
