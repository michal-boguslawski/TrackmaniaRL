# Metric reference

Every scalar PPO training sends to a tracking backend, what it means, and what to
cross-check it against. For *how* a metric reaches a backend, and the two
invariants that govern adding one, see the "Metric pipeline" section of
`AGENTS.md`.

## Generated vs. maintained

The trainer metric names below are **generated**, not transcribed:

```bash
uv run --extra cpu python scripts/benchmarks/metric_inventory.py
```

That script runs a real update at `Verbosity.ALL` and prints the names it
observed, so the list cannot silently drift from the code. Regenerate it when
you add a metric. The collector-side metrics in the last section are emitted
directly by callbacks and need a live environment, so they are maintained by
hand.

A name appearing in the inventory proves only that it is emitted, not that its
value is meaningful — the tests are what pin down the latter.

## Reading the diagnostics

Two axes matter when a run misbehaves.

*Policy health* — `metrics/alpha`, `metrics/beta`, `metrics/entropy_*`,
`metrics/beta_mean_*`. These describe how peaked the action distribution is and
where it has settled.

*Update health* — `metrics/approx_kl`, `metrics/ratio_max`,
`metrics/clip_fraction`, `metrics/beta_mean_*`, `grad_norm/max`. These describe
whether the optimizer is moving the policy too far in one step.

Most failures show up in both axes at once, and the pairing is what identifies
the cause:

| Symptom | Likely cause |
| --- | --- |
| `concentration_max` spikes while the averages stay flat | A few states are extremely confident; this is where non-finite log-probs come from |
| `alpha` and `beta` jump together with `ratio_max` | Unstable update, not a distributional change |
| `beta_mean_{i}` pinned near 0 or 1 | Action saturated at a boundary (full gas/brake) |
| `approx_kl` spikes with falling `entropy_*` | Policy collapsing toward deterministic faster than the trust region allows |
| `clip_fraction` high with flat `approx_kl` | Trust region too narrow for the learning rate |

## `loss/`

| Metric | Meaning |
| --- | --- |
| `loss/total` | `actor + critic_beta * critic - entropy_coef * entropy`, the quantity that is backpropagated |
| `loss/critic` | Clipped Huber loss, taking the worse of clipped and unclipped value error |
| `loss/entropy` | Sum of per-dimension Beta entropies; higher means more exploratory |

## `metrics/` — policy

| Metric | Meaning |
| --- | --- |
| `metrics/alpha` | Mean Beta concentration across all action dimensions. A steady rise means the policy is going deterministic; read against `metrics/entropy_*` and `approx_kl` spikes |
| `metrics/beta` | The same for the second concentration parameter. Moving together with `alpha` points at an unstable update |
| `metrics/alpha_{i}` | `alpha` split per action dimension, which localizes drift to steering, gas, or brake |
| `metrics/beta_{i}` | `beta` split per action dimension |
| `metrics/concentration_max` | Largest concentration anywhere in the batch. A large gap to the averages marks the few states where log-probs go non-finite |
| `metrics/beta_mean_{i}` | Mean of the untransformed Beta, in the `[0, 1]` support. Near 0 or 1 is saturation at a boundary |
| `metrics/entropy_{i}` | Beta entropy per action dimension |
| `metrics/log_std_{i}` | Log of the Beta's standard deviation. Retained for Gaussian-policy comparability, but a **poor** dispersion proxy here: the Beta support is fixed and its width is governed by the concentrations above |
| `metrics/mean_abs_max` | Largest absolute mean action in the batch |
| `metrics/mean_abs_max_{i}` | The same per action dimension |
| `metrics/tanh_saturation_frac` | Fraction of mean actions beyond `tanh_saturation_threshold` of the action bounds |
| `metrics/mean_reg` | Mean of the squared mean actions in transformed coordinates, where `[0, 0, 0]` is no input; the term `mean_reg_coef` penalizes drift toward the action-range midpoint |

## `metrics/` — update

| Metric | Meaning |
| --- | --- |
| `metrics/actor_loss` | Total actor objective, including `mean_reg` |
| `metrics/surrogate_loss` | Clipped surrogate alone, without `mean_reg` |
| `metrics/approx_kl` | KL between old and new policy, summed over action dimensions. **Load-bearing:** `update_loop.py` reads it to stop an epoch early, so it cannot be renamed or dropped without changing training control flow |
| `metrics/approx_kl_{i}` | The same per action dimension, which shows which control is driving policy movement |
| `metrics/ratio_max` | Largest importance ratio in the batch |
| `metrics/clip_fraction` | Fraction of samples whose ratio fell outside `[1-eps, 1+eps]` |

## `grad_norm/`

| Metric | Meaning |
| --- | --- |
| `grad_norm/max` | Largest module gradient norm; compare against the configured clip values |
| `grad_norm/cnn` | CNN trunk norm |
| `grad_norm/sequence_encoder` | Temporal encoder norm |
| `grad_norm/actor` | Actor head norm |
| `grad_norm/critic` | Critic head norm |

## `rollout/`

Batch summaries over the collected rollout, logged before the update.

| Metric | Meaning |
| --- | --- |
| `rollout/returns` | Mean GAE return |
| `rollout/advantages_mean` | Mean advantage **as collected**, logged before any normalization, so it is not expected to be zero. A sustained offset points at reward or value bias |
| `rollout/advantages_std` | Advantage spread before normalization |
| `rollout/critic_values` | Mean critic estimate |
| `rollout/explained_variance` | How much of the return variance the critic explains; near 1 is good, at or below 0 the critic is no better than the return mean |
| `rollout/old_log_probs` | Mean log-probability under the behavior policy |
| `rollout/action_mean_{i}` | Mean sampled action per dimension, in environment units |
| `rollout/action_std_{i}` | Spread of sampled actions per dimension; collapse here is premature convergence |

## `training/`

| Metric | Meaning |
| --- | --- |
| `training/entropy_coef` | Live entropy coefficient after this update's decay, floored at `entropy_coef_min` |
| `training/lr_{i}` | Learning rate of optimizer param group `i`; the index depends on how many groups the agent defines |

## Collector-side (maintained by hand)

These are emitted by callbacks rather than the trainer, so the inventory script
does not see them.

| Metric | Meaning |
| --- | --- |
| `episode/returns` | Mean undiscounted return of completed episodes |
| `episode/lengths` | Mean episode length in steps |
| `episode/n` | Episodes accumulated in the reporting window |
| `evaluation/<scope>/episode_return` | Return of one evaluation episode |
| `evaluation/<scope>/episode_length` | Length of one evaluation episode |
| `evaluation/<scope>/return_mean` | Mean return across evaluation episodes |
| `evaluation/<scope>/return_std` | Standard deviation of evaluation returns |
| `evaluation/<scope>/return_min` | Worst evaluation episode |
| `evaluation/<scope>/return_max` | Best evaluation episode |
| `evaluation/<scope>/length_mean` | Mean evaluation episode length |
| `evaluation/<scope>/episodes` | Number of evaluation episodes run |

`RecordStatisticLoggerCallback` does not currently log episode stats under
`SAME_STEP` autoreset, because the payload lands in `info["final_info"]` inside
the vector env and never reaches the top level. See the outstanding item in
`docs/TODO.md`.

## Adding a metric

1. Compute a **0-dim** tensor in the relevant method of
   `src/rl_lib/training/ppo/losses.py`, guarded by
   `self._verbosity.logs_diagnostics`. Per-dimension values must be fanned out
   into `metrics/name_0`, `metrics/name_1`, ... rather than returned as a vector.
2. Add the name to the table above and to the diagnostics assertions in
   `tests/unit/test_ppo_trainer.py`.
3. Regenerate the inventory and confirm the name shows up:

   ```bash
   uv run --extra cpu python scripts/benchmarks/metric_inventory.py
   ```