# Performance Optimization Handoff

This file captures the profiling review, code changes, measurements, validation,
and next experiments so optimization work can continue in a new chat.

## Profile findings

Reviewed the latest trace:

`logs/profiles/heavenly-shrew-20260928T220143Z-c8d19a/heavenly-shrew_130484.1790632908455017749.pt.trace.json`

The earlier profile review found:

- The trace was from a smoke-sized workload: `num_envs=2`, rollout/buffer size 4,
  minibatch size 4, stack size 4 (16 images per minibatch), not the normal PPO
  config (`num_envs=16`, minibatch 512, stack size 4).
- Profiled span was about 382 ms; `ppo/update` was about 310 ms.
- The trace counted about 2,044 CUDA kernel launches and 1,792
  `cudaFuncGetAttributes` calls. Aggregate GPU kernel durations were about 25 ms
  in that small workload, so fixed dispatch/synchronization costs were
  significant there.
- The kernel list included NCHW/NHWC conversion kernels. Observation
  normalization and conversion were also repeated throughout rollout and PPO
  updates.
- `environment/step` was about 44 ms across two steps in the captured region.

These profile values are specific to that short trace and should not be treated
as steady-state measurements for the production-sized workload.

## Changes made

The following performance controls and changes were added:

- `src/rl_lib/run_config.py`
  - Added `run.cudnn_benchmark` and `run.channels_last` (default enabled).
  - Added opt-in `run.torch_compile` (default disabled) and a validated
    `run.torch_compile_mode` (default `reduce-overhead`).
  - Added `trainer.fused_optimizer` (default enabled; only applied on CUDA).
- `src/rl_lib/train.py`
  - Applies cuDNN autotuning on CUDA when configured.
  - Converts the CNN weights to channels-last format on CUDA when configured.
  - Optionally wraps the network with `torch.compile`.
- `src/rl_lib/training/ppo/trainer.py`
  - Requests PyTorch's fused Adam/AdamW implementation on CUDA when enabled.
- `src/rl_lib/agent.py`
  - Replaced separate observation division and addition with `torch.addcdiv`,
    using cached scalar tensors. Tests and a direct check showed exact output
    equality for the tested normalization inputs.
- `src/rl_lib/config/ppo_carracing.yaml`
  - Enables `cudnn_benchmark`, `channels_last`, and `torch_compile` with
    `reduce-overhead` for the normal CarRacing run config.
- Tests were added for normalization equality, optimizer CPU fallback, and
  config defaults/validation.

Important: `torch_compile` is opt-in by default in the schema but enabled in
`ppo_carracing.yaml`. It can add a significant one-time startup/compilation cost.
Disable it in that YAML if startup latency or compile incompatibility is a
problem.

## Validation

Environment used: NVIDIA GeForce RTX 3060, CUDA-enabled PyTorch `2.13.0+cu130`,
Python 3.13. The environment initially had CPU-only PyTorch; it was changed with
`uv sync --extra gpu --extra dev`.

- Full test suite run under the GPU-enabled environment:
  `uv run --extra gpu --extra dev pytest` → **427 passed, 4 warnings**.
  This means the suite ran with the GPU PyTorch build; it does not imply every
  individual test moved tensors onto CUDA (many tests explicitly exercise CPU).
- Plain CUDA end-to-end smoke training completed using
  `src/rl_lib/config/manual_smoke.yaml`.
- Compiled CUDA smoke training also completed with a temporary smoke config
  enabling `torch_compile: true` and `torch_compile_mode: reduce-overhead`.
- The actual CarRacing config's PPO actor, critic, and entropy losses matched
  between the baseline and tuned implementations in the benchmark (reported
  difference `0.00e+00` for all three).

## Benchmark results

Paired timings used the repository helper `scripts/benchmarking.py`, with CUDA
synchronization and peak allocation tracking. The benchmark drivers were
temporary files under `/tmp/opencode` and were not committed to the repository.
Workload used the real PPO minibatch shape: 512 transitions × 4 frames = 2,048
96×96 grayscale images. GPU train-step benchmark used 3 warmups and 10 measured
repetitions; act used 3 warmups and 20 repetitions. Results are from one RTX
3060 and should be re-measured on the target training setup.

| Operation | Baseline | Candidate | Result |
|---|---:|---:|---:|
| Preprocess (2,048 images) | 1.27 ms | 0.82 ms | 1.53× faster |
| PPO train step, eager | 32.29 ms | 31.58 ms tuned | 1.02× faster |
| PPO train step, compiled | 32.29 ms | 31.66 ms tuned + compiled | 1.02× faster |
| Rollout `act`, batch 16 | 1.406 ms | 1.368 ms compiled/channels-last | 1.03× faster |

The baseline/tuned train-step figures include the layout/optimizer differences;
they are small and should be treated cautiously. A separate comparison with
cuDNN autotuning disabled/enabled reported 36.75 ms vs 32.29 ms for the baseline
path, but this was not a perfectly isolated repeated A/B experiment. Treat the
cuDNN result as promising, not conclusive.

A tuned train-step trace reported 387 kernels and 29.13 ms summed kernel time
over a 35.31 ms kernel span (~82% aggregate GPU busy time for that capture).
This supports that the production-sized PPO minibatch is largely compute-bound,
unlike the smoke profile.

## Recommended next work

Prioritize measurement and keep correctness checks explicit:

1. **Prototype BF16 autocast for the CNN path.** The RTX 3060 is Ampere and
   supports BF16. Measure autocast around the convolutional backbone first,
   retaining FP32 for Beta distribution parameters, log-probabilities, PPO loss,
   and value/loss reductions. Compare actions, values, losses, gradients, and
   short training stability against FP32. BF16 generally does not require a
   `GradScaler`; FP16 may.
2. **Reduce synchronization in collection and PPO metrics.** Investigate the
   boolean branches on CUDA tensors such as `if not T.isfinite(loss)` and
   `if not T.isfinite(log_probs).all()`, since they force host synchronization.
   Preserve non-finite-update protection and error reporting while batching or
   deferring checks. At diagnostic verbosity, vectorize per-action metric
   reductions and avoid transferring every diagnostic scalar every minibatch;
   retain the approximate KL needed for early stopping.
3. **Benchmark end-to-end rollout throughput, not just a minibatch.** Include
   environment stepping, policy inference, logging, and PPO updates. Check
   whether async vector environments and environment stepping dominate for the
   production config.
4. **Test `torch.compile` under a representative full run.** Current smoke
   validation passed, but the measured compiled train-step gain was only about
   1–2%. Measure compile startup cost, graph recaptures, memory, and steady-state
   transitions/second. Compare `default` vs `reduce-overhead`; do not assume
   CUDA graphs improve the real workload.
5. **Consider CNN FLOP reductions only with RL-quality evaluation.** Lower input
   resolution, fewer channels, or an architecture change can save real compute,
   but will change learning behavior and needs controlled training comparisons.

For each candidate, use the same seeds/workload and GPU environment, verify
numerical behavior before timing, and report median/p90 latency, throughput, and
memory. Avoid using `scripts/train.py` with its default 3,000,000-step config as
a smoke test.

## Working-tree caution

At handoff, `git status` also showed modified files not changed as part of the
performance edits summarized above: `TODO.md`, `src/rl_lib/training/ppo/losses.py`,
and `src/rl_lib/training/rollout_collector.py`. Those diffs were not reviewed as
part of this handoff; inspect them before attributing or changing them. Keep
unrelated working-tree changes intact.
