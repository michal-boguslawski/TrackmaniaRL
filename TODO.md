# Code review TODOs

## Correctness first

- [x] Choose and configure an explicit Gymnasium vector-environment autoreset mode; ensure reset-only steps are not stored as policy transitions.
- [ ] Align termination/truncation flags with observation windows so collection and PPO action evaluation use the same temporal masks, including at episode starts.
- [x] Bootstrap truncated episodes from the final observation rather than a reset observation.
- [ ] Apply the actor mean regularization in transformed action coordinates; neutral steering should not be penalized toward full-left steering.
- [ ] Fix `RecordVideoCallback.on_rollout_end()` to pass a valid step and guarantee its environment closes if recording fails.
- [x] Stop swallowing `BaseException` in the training entrypoint; preserve error and interruption status in the process and MLflow run.
- [x] Guard CUDA memory-stat calls so the advertised CPU fallback can start without CUDA.

## Tests and reproducibility

- [ ] Update tests to match current Actor, Agent, Network, RolloutBuffer, and PPOTrainer APIs.
- [ ] Add focused tests for autoreset behavior, temporal-mask alignment, and termination-versus-truncation bootstrapping.
- [ ] Centralize training/environment settings and log the resolved configuration, including seed, GAE parameters, wrappers, and the meaning of `training_steps`.
- [ ] Use one import path consistently (`rl_lib.*` or `src.rl_lib.*`) to avoid loading duplicate module identities.
- [ ] Make the `rl-lib` console entrypoint launch the intended application, or remove it until it does.

## Efficiency and maintainability

- [ ] Profile environment stepping, device transfers, inference, PPO updates, and logging before tuning vectorization or device placement.
- [ ] Replace Python-loop minibatch assembly with batched indexing/gathering for sequence windows.
- [ ] Avoid repeated gradient-norm passes and excessive per-minibatch `.item()` synchronization; aggregate diagnostic metrics where practical.
- [ ] Evaluate preallocated rollout storage to reduce per-step tensor cloning and stacking overhead.
- [x] Make CPU/CUDA installation choices explicit; separate development/notebook dependencies from runtime dependencies.
- [ ] Consider splitting PPO loss/metrics, minibatch generation, and update-loop responsibilities once correctness tests are in place.
