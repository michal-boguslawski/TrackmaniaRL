# RL Portfolio Project — Specification (Draft for Review)

## 1. Project Goal
Two-stage RL portfolio project for **General ML Engineer / MLE** roles (mixed company types — not narrowed to startup/big tech/research lab). Standalone ML/RL piece — no explicit tie-in to actuarial background. Complements the existing RetailFlow data engineering portfolio piece as a from-scratch deep RL demonstration.

## 2. Scope & Stages

**Stage 1 — CarRacing-v3 (Gymnasium)**
- Continuous action space (steering / gas / brake)
- Two from-scratch PyTorch implementations, no Stable-Baselines3 or similar:
  - **On-policy: PPO** — clean implementation, not reusing prior Atari/BipedalWalker-Hardcore code
  - **Off-policy: SAC**
- Depth/speed priority: **balanced** — solid, correct implementation, no extensive hyperparameter tuning campaigns
- **Stretch goal:** **TD3** as an optional third algorithm, added after the core PPO/SAC pair is working. Not blocking Stage 1 completion.

**Stage 2 — TrackMania**
- **Hard requirement**, core to the portfolio narrative (not optional)
- Interface: **TMRL** (existing open-source Gym-style wrapper for TrackMania)
- Algorithm(s): extend whichever of PPO/SAC transfers better from Stage 1 (to be decided based on Stage 1 results)
- Success criterion: **competitive lap times vs. human/benchmark** — not just "completes laps," a real performance bar

## 3. Timeline & Compute
- Timeline: flexible, no fixed deadline
- Availability: ~10–20 hrs/week
- Compute: cloud/rented GPU (Vast.ai, Lambda, AWS, or similar) — no persistent local GPU assumed

## 4. Infrastructure
- **Experiment tracking:** MLflow
- **Containerization:** Docker (consistent with RetailFlow's approach — multi-stage builds, layer caching discipline)
- **Testing:** unit tests for core RL components — replay/rollout buffers, GAE computation, loss functions. Not full CI rigor, but real test coverage on the algorithmic core, not just smoke tests.

## 5. Deliverables
- GitHub repo + README (**portfolio-grade polish**: architecture diagrams, thorough docs)
- Video demo of the agent racing (CarRacing and/or TrackMania)
- No blog post or slide deck requested

## 6. Milestones (sequence, not dated — timeline is flexible)
1. PPO baseline working on CarRacing-v3
2. SAC baseline working on CarRacing-v3
3. Stage 1 comparison write-up (PPO vs. SAC — sample efficiency, stability, final performance)
4. Stretch: TD3 as optional third algorithm
5. TMRL integration — port trained agent(s) into TrackMania environment
6. TrackMania training to competitive lap times
7. Final polish — README, architecture diagrams, recorded demo video

---
*Spec confirmed — TrackMania-on-Linux path already identified, PPO confirmed as on-policy pick, TD3 added as optional stretch. Ready to hand to a coding agent for directory structure, architecture, and implementation planning.*
