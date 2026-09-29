# Code review TODOs

## Correctness first

- [x] Choose and configure an explicit Gymnasium vector-environment autoreset mode; ensure reset-only steps are not stored as policy transitions.
- [x] Align termination/truncation flags with observation windows so collection and PPO action evaluation use the same temporal masks, including at episode starts.
- [x] Bootstrap truncated episodes from the final observation rather than a reset observation.
- [x] Apply the actor mean regularization in transformed action coordinates; neutral steering should not be penalized toward full-left steering.
- [x] Fix `RecordVideoCallback.on_rollout_end()` to pass a valid step and guarantee its environment closes if recording fails.
- [x] Stop swallowing `BaseException` in the training entrypoint; preserve error and interruption status in the process and MLflow run.
- [x] Guard CUDA memory-stat calls so the advertised CPU fallback can start without CUDA.
- [ ] Read episode statistics from `info["final_info"]` in `RecordStatisticLoggerCallback`. Under `SAME_STEP` autoreset the stats wrapper runs inside the vector env, so the terminal payload never reaches the top level of `info` and every episode is dropped. This is the suite's only xfail; note that Gymnasium 1.3 reports `final_info` as a dict of batched arrays with an `_episode` mask, not as a list of per-env infos, so the xfail test's fixture has to be rewritten with the fix.
- [ ] Make `Agent.evaluate_actions`'s `dones` argument required. It is annotated `T.Tensor | None = None` but dereferenced unconditionally, so the annotation promises a call that raises `AttributeError`.
- [x] Keep `mean_reg` pointed at the no-input action `[steering=0, gas=0, brake=0]`; the `[0.5, 0.5]` gas/brake support midpoint applies both controls, so it is not neutral.

## Tests and reproducibility

- [x] Update tests to match current Actor, Agent, Network, RolloutBuffer, and PPOTrainer APIs.
- [x] Add focused tests for autoreset behavior, temporal-mask alignment, and termination-versus-truncation bootstrapping.
- [x] Centralize training/environment settings and log the resolved configuration, including seed, GAE parameters, wrappers, and the meaning of `training_steps`.
- [x] Use one import path consistently (`rl_lib.*` or `src.rl_lib.*`) to avoid loading duplicate module identities.
- [ ] Make the `rl-lib` console entrypoint launch the intended application, or remove it until it does. `rl_lib.main` still prints the `uv init` placeholder.
- [ ] Fill `RuntimeSettings.video_folder` in the entrypoint. `RunConfig` derives the real per-session path as a property, so the runtime field is logged as `""` and the path actually used never reaches MLflow.
- [ ] Cover the resolved-config dump with a test. The params logger is only checked against a literal dict, so nothing would catch the `training_steps` unit or the GAE parameters going missing.

## Efficiency and maintainability

- [ ] Profile environment stepping, device transfers, inference, PPO updates, and logging before tuning vectorization or device placement. `ProfilingConfig` (`src/rl_lib/profiling_config.py`, `src/rl_lib/config/profiling.yaml`) adds a `torch.profiler` capture toggled by `--profile`/`--no-profile`, with `record_function` regions for policy inference, environment stepping, truncation bootstrap, buffer writes, the PPO forward/backward/optimizer phases, and the host/device copies (`transfer/*`: observation and reward uploads, action download, flag uploads, per-minibatch index upload and observation gather, and the per-minibatch metrics sync). No profile has been run yet, so the cost split is still unknown, and the async vector-env workers and the logging callbacks are not instrumented.
- [x] Replace Python-loop minibatch assembly with batched indexing/gathering for sequence windows. `PPOTrainer._get_iid_minibatches` builds every field with a list comprehension of per-sample `T.stack`/`T.cat` calls.
- [x] Avoid repeated gradient-norm passes and excessive per-minibatch `.item()` synchronization; aggregate diagnostic metrics where practical. Each minibatch runs eight full gradient-norm passes (three for clipping, five for the per-module diagnostics in `Agent.get_parital_clip_grad_norms`), each ending in its own `.item()`.
- [x] Evaluate preallocated rollout storage to reduce per-step tensor cloning and stacking overhead. `RolloutBuffer` keeps a `deque` per field, clones on every append, restacks every field in `get()`, and runs the GAE recursion as a Python loop.
- [x] Make CPU/CUDA installation choices explicit; separate development/notebook dependencies from runtime dependencies.
- [x] Move `pytest`, `matplotlib` and `ipykernel` out of the runtime dependencies into a `dev` extra. The CPU/GPU extras are done; the dependency split is not.
- [x] Consider splitting PPO loss/metrics, minibatch generation, and update-loop responsibilities once correctness tests are in place. `PPOTrainer` is a single 434-line class covering the optimizer, clipping, three loss/metric helpers, the minibatch generator and the update loop.


5. Remove hidden per-step synchronizations:
- trainer.py:222 — if not T.isfinite(loss) forces a device sync before backward, every minibatch (skip_nonfinite_updates: true). The finiteness check could ride along with the metrics sync that already happens, or scale the loss on-device instead of branching.
- agent.py:223 — T.isfinite(log_probs).all() in an if syncs on every env step during rollout collection. Should be debug-only or folded into the batched transfer.
- _actor_loss_metrics / _entropy_loss compute ~20 diagnostic reductions per minibatch at verbosity: 2, including Python for index in ... loops that launch one kernel per action dim per metric. These could be vectorized (one mean(dim=0)/max(dim=0) each) and the KL early-stop needs only one scalar synced per minibatch — the rest could transfer once per epoch.
