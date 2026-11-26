# Training diagnostics and learning-rate validation

## Verify the gradient and optimizer path

- [x] Add a small CPU unit test that runs a representative PPO minibatch and asserts nonzero gradients reach the CNN, temporal encoder, actor, and critic parameter groups.
- [ ] Add a one-step direction test: after an optimizer step, recompute the same minibatch objective and verify it decreases within a reasonable tolerance. Include a deliberately excessive learning rate case to ensure the test catches overshooting.
- [ ] Add a tiny-batch overfit test over a few hundred updates. Verify the policy/value objectives improve and the actor distribution changes from initialization; this is a fast regression check, not a claim that PPO should overfit real rollouts.
- [ ] Keep these checks small and deterministic so they run as unit tests and catch detached graphs, incorrect loss signs, and broken optimizer wiring without a full environment run.

## Log metrics that diagnose optimization

- [ ] Add PPO clip fraction: the fraction of samples whose probability ratio is outside the clipping interval. Interpret alongside KL; sustained high values can indicate overly large updates, while near-zero values with no policy movement can indicate updates that are too small.
- [ ] Add an occasional same-minibatch objective delta measured before and after the optimizer step. Make the extra forward configurable or run it only every N minibatches; a consistently rising objective can flag overshooting or a sign error.
- [ ] Log pre-clipping gradient norms per optimizer parameter group.
- [ ] Log per-group relative parameter update size (`||Δw|| / ||w||`) to compare effective update magnitudes across learning-rate configurations.
- [ ] Fix rollout action standard-deviation logging: calculate a standard deviation rather than repeating the mean, and cover it with a test.
- [ ] Log per-action-dimension Beta entropy (and retain alpha/beta diagnostics) to show whether the policy distribution is changing or remaining near its initialization.
- [ ] Log the effective entropy coefficient during training. Check whether configured entropy decay is actually advanced; either implement the intended schedule and test it, or remove/clarify unused decay settings.

## Establish a cheap learning-rate and improvement check

- [ ] Reproduce the known-good `ppo_carracing.yaml` baseline with a short budget and save its early-training curves as a reference envelope: episode reward/slope, approximate KL, clip fraction, action distribution/entropy, and relative parameter updates.
- [ ] Compare new configurations at fixed environment-step checkpoints against that reference instead of judging only by final return.
- [ ] Run controlled one-factor ablations before committing to a new recipe: change only learning rates to the known-good values, then separately test actor width and PPO clipping settings. Keep seed, environment, and training budget fixed.
- [ ] Add an improvement detector that warns or records a run as stalled when the episode-reward trend is non-positive over a configurable window after a configurable warm-up period.
- [ ] Ensure each run logs the resolved config, seed, code revision, and the metric definitions needed to reproduce and compare these experiments.
