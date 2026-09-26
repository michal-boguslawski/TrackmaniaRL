# Stage 1 Architecture — CarRacing-v3 (PPO + SAC)

Scoped from the confirmed project spec. Covers directory structure, algorithm
design decisions, and a rough milestone timeline for Stage 1 only. TrackMania
(Stage 2) architecture gets its own doc once Stage 1 is stable — no point
designing that interface before we know what transfers.

**Assumptions I'm making below (flag if wrong):** single repo for both stages
(clear internal separation, not two repos), Nature CNN as the visual encoder
baseline, standard literature hyperparameters as a starting point (not tuned).

---

## 1. Directory Structure

```
rl-portfolio/
├── src/
│   └── rl_lib/
│       ├── algorithms/
│       │   ├── ppo.py              # PPO agent: update step, clipped loss
│       │   ├── sac.py              # SAC agent: update step, twin Q, alpha
│       │   └── td3.py              # stretch — added after ppo/sac are solid
│       ├── networks/
│       │   ├── cnn.py              # shared visual backbone (Nature CNN)
│       │   ├── actor.py            # Gaussian (PPO) / squashed Gaussian (SAC)
│       │   └── critic.py           # value head (PPO) / twin Q heads (SAC)
│       ├── buffers/
│       │   ├── rollout_buffer.py   # on-policy, GAE, cleared each update
│       │   └── replay_buffer.py    # off-policy, circular, terminated/truncated aware
│       ├── envs/
│       │   ├── wrappers.py         # grayscale, resize, frame stack, action repeat
│       │   └── make_env.py
│       ├── training/
│       │   ├── ppo_trainer.py
│       │   ├── sac_trainer.py
│       │   └── evaluator.py        # shared eval loop, deterministic policy
│       ├── tracking/
│       │   └── mlflow_logger.py
│       └── config.py               # pydantic-settings, one model per algorithm
├── configs/
│   ├── ppo_carracing.yaml
│   └── sac_carracing.yaml
├── scripts/
│   ├── train.py                    # entrypoint, algorithm picked via config
│   └── evaluate.py
├── tests/
│   ├── test_gae.py                 # GAE vs. hand-computed reference values
│   ├── test_ppo_loss.py            # clipped surrogate, known ratios/advantages
│   ├── test_replay_buffer.py       # circular overwrite, sampling correctness
│   ├── test_sac_target.py          # target Q with entropy term
│   └── test_wrappers.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml          # training container + mlflow server
├── docs/
│   └── architecture.md
├── pyproject.toml
└── README.md
```

PPO and SAC each own their agent file and trainer — no shared `BaseAgent`
abstraction. They don't share enough update logic to justify one, and forcing
a common interface now would just add indirection for two algorithms that
differ in almost every step (on vs. off-policy, buffer type, loss shape). The
CNN encoder is the one thing genuinely shared — same architecture, but each
algorithm instantiates and trains its own copy, no weight sharing between them.

## 2. Algorithm Design Decisions

**Observation preprocessing** (shared by both):
- Grayscale, resize if needed, frame stack of 4
- Normalize pixel values to `[0, 1]` or standardize — pick one and log it in config

**Action space handling** — this is the part most likely to bite you:
CarRacing's action space is `[steer ∈ [-1, 1], gas ∈ [0, 1], brake ∈ [0, 1]]`,
not symmetric. For SAC's `tanh`-squashed Gaussian, you need per-dimension
rescaling from `[-1, 1]` to each action's true range, not a single global
scale. For PPO's Gaussian policy, clip (not just sample-then-clip blindly —
account for it in the log-prob if you clip post-sampling).

**terminated vs. truncated** — Gymnasium distinguishes these; going off-track
in CarRacing is a `terminated` (bootstrap value = 0), while a time-limit cutoff
is `truncated` (bootstrap using the value function). Get this wrong in the
replay/rollout buffer and both algorithms will learn a subtly wrong value
function. Worth a dedicated unit test.

**PPO defaults to start from** (literature-standard, not tuned):
`clip ε = 0.2`, `GAE λ = 0.95`, `γ = 0.99`, entropy bonus coefficient ~0.01,
separate actor/critic heads off the shared CNN trunk (not fully shared —
reduces gradient interference between policy and value losses).

**SAC defaults to start from**:
`τ = 0.005` (target network soft update), automatic entropy temperature
tuning with target entropy `= -action_dim`, twin Q networks with min-clipping
for the target.

**Open design question (not blocking, revisit after baselines run):** whether
CarRacing's stock reward (small per-frame penalty + tile completion bonus) is
shaped enough to learn efficiently, or whether you'll want to add shaping.
Don't decide this until you see how the vanilla reward behaves first.

## 3. Config & Testing

- One Pydantic-settings model per algorithm, one YAML per run — consistent
  with how you've handled config elsewhere. MLflow logs the resolved config
  alongside every run.
- The four unit tests listed above (GAE, PPO loss, replay buffer, SAC target)
  are the ones with real correctness risk — bugs in these fail silently as
  "the agent just doesn't learn," which is much harder to debug than a failing
  test. Wrapper tests are lower-value but cheap to write.

## 4. Rough Milestone Timeline (10–20 hrs/week, not hard dates)

| Milestone | Rough effort |
|---|---|
| Env wrappers + CNN encoder + smoke test | ~1 week |
| PPO implementation, debugged to a solved run | ~2–3 weeks |
| SAC implementation, debugged to a solved run | ~2–3 weeks |
| Comparison write-up (sample efficiency, stability, final reward) | ~1 week |
| TD3 stretch (optional) | ~1–2 weeks |

Stage 1 total: roughly 6–10 weeks at your stated pace. Stage 2 (TrackMania)
planning starts once PPO/SAC are both solid.

---
*Review and flag anything you'd change — assumptions, defaults, or structure —
before this goes to a coding agent.*
