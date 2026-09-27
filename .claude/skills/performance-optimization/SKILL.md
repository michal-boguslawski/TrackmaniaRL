---
name: performance-optimization
description: Use when profiling or optimizing runtime, memory, throughput, or latency in this repository.
---

# Performance optimization workflow

Use this workflow for code performance work. Keep measurements reproducible and
make correctness an explicit gate before accepting a speedup.

## 1. Establish the baseline

- Trace the hot path from its real caller and identify the workload dimensions
  that affect cost (batch size, sequence length, observation shape, device,
  dtype, and thread count).
- Record existing semantics and tests that protect ordering, numerical
  behavior, mutation, device placement, and boundary conditions.
- Measure before editing. Separate setup, steady-state work, and result
  assembly when they have different cost profiles.
- Avoid long training entry points as smoke tests. In this repository,
  `scripts/train.py` launches a long PPO run.

## 2. Change one cost center at a time

- Prefer an implementation that removes repeated allocations, copies, Python
  dispatch, or synchronization without changing public behavior.
- Preserve edge cases documented in `AGENTS.md`, especially temporal rollout
  alignment and the distinction between terminated and truncated episodes.
- Keep a correctness reference for algorithmic changes, such as a simple
  sequential implementation when replacing a recurrence with a scan.

## 3. Validate before interpreting timings

- Compare outputs from baseline and candidate on deterministic inputs before
  measuring performance. Check shapes, dtype/device, ordering, and numeric
  tolerances.
- Run the narrow tests first, then the affected integration tests, and run the
  full suite for cross-cutting changes:

  ```bash
  uv run --extra cpu pytest tests/unit/test_<area>.py
  uv run --extra cpu pytest tests/unit/test_ppo_trainer.py tests/unit/test_rollout_collector.py
  uv run --extra cpu pytest
  ```

- Use the same CPU/GPU extra as the environment. Run a CUDA benchmark only when
  CUDA is available; synchronize around timed GPU work.

## 4. Benchmark reproducibly

- Use `scripts.benchmarking.benchmark_callables` for paired callable comparisons. It
  performs warmups, rotates measurement order, reports individual samples,
  median, mean, p90, min/max, and optionally peak allocated CUDA memory.
- Specify a CUDA `device` to synchronize operations and collect CUDA memory
  deltas. Leave it unset for CPU-only work; CPU memory is not inferred from
  Python allocation tracking.
- Keep benchmark setup outside timed closures when measuring a specific phase.
  Include setup when measuring end-to-end cost. Use identical inputs and
  configuration for baseline and candidate.
- Benchmark more than one workload size when the optimization may change with
  batch or sequence length. Treat small timing differences cautiously and
  repeat measurements if results vary materially.
- Report environment, workload shape, warmups/repetitions, latency statistics,
  memory results, and any correctness tolerance. Do not present one noisy
  microbenchmark as a general speedup.

### Rollout-buffer benchmark

The rollout-buffer runner measures add, get, and end-to-end phases for one
selected project worktree. It does not embed or preserve a copy of an older
implementation. Use that same runner against separate baseline and candidate
worktrees, then compare the JSON reports:

```bash
TOOL=/path/to/optimization-worktree/scripts/benchmarks/rollout_buffer.py
uv run --extra cpu python "$TOOL" --project-root /path/to/master-worktree \
  --label master --size 256 --num-envs 16 --stack-size 4 \
  --repetitions 7 --warmups 2 --json /tmp/master.json
uv run --extra cpu python "$TOOL" --project-root /path/to/optimization-worktree \
  --label candidate --size 256 --num-envs 16 --stack-size 4 \
  --repetitions 7 --warmups 2 --json /tmp/candidate.json
uv run --extra cpu python -m scripts.benchmarks.compare_reports \
  /tmp/master.json /tmp/candidate.json
```

Create the master and candidate worktrees before running these commands, and
invoke the exact same script path both times. Run from the repository's CPU or
GPU environment as appropriate. On a CUDA machine, use `uv run --extra gpu` and
pass `--device cuda` (or a specific device such as `cuda:0`). The runner records
the selected worktree's commit, branch, and dirty status, validates output
shapes and finite GAE values, and measures add/get/end-to-end separately. JSON
output paths require an existing parent directory.

Generic usage from another benchmark script:

```python
from scripts.benchmarking import benchmark_callables

results = benchmark_callables(
    {"baseline": baseline, "candidate": candidate},
    warmups=2,
    repetitions=10,
    device="cuda",  # omit for CPU-only work
)
print(results["candidate"].speedup_over(results["baseline"]))
```

## 5. Finish with an evidence-based summary

State the code path changed, semantics preserved, tests run, benchmark setup,
results for each relevant phase, and whether the measured tradeoff supports the
optimization. If add/get or memory tradeoffs differ, report each rather than
only the combined number.
