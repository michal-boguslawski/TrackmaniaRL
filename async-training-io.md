# Plan: Asynchronous I/O and evaluation in the training loop

## Goal

Reduce time the PPO collector spends waiting on logging, checkpointing, evaluation, and video encoding, without changing training semantics or silently losing important records.

Use background threads first. MLflow and local artifact writes are I/O-bound; keep the design simpler than a process pool or asyncio implementation. Use separate work queues for short metric writes and long evaluation/video jobs so a slow recording cannot delay metrics.

## Current synchronous work

The main blocking points identified in the codebase are:

| Work | Main code path | Impact |
|---|---|---|
| MLflow metric writes | `tracking/callbacks/metrics_logger.py::flush`, `tracking/sinks/mlflow_sink.py::log_metrics` | Usually once per PPO update |
| Episode-stat metric writes | `tracking/callbacks/episode_statistics.py::flush` | Can occur inside the environment-step callback |
| Evaluation and recording | `tracking/callbacks/evaluate.py`, `tracking/callbacks/record_video.py` | Runs complete evaluation episodes synchronously |
| Video encoding and local file writes | Gymnasium `RecordVideo` wrapper, used by `RecordVideoCallback` | Happens during/at the end of recording |
| Checkpoint serialization and upload | `tracking/callbacks/model_checkpoint.py`, `MLflowMetricsSink::log_state_dict` | Periodic; currently receives live state-dict tensors |
| Final model and artifacts | `ModelCheckpointCallback` and `EvaluationCallback` at rollout end | Can delay shutdown substantially |

Console/file logging already uses Python's `QueueHandler`/`QueueListener`. Keep the existing metric tensor conversion boundary in `tracking/metric_values.py`; it batches device values into one host transfer before logging.

## Invariants and design constraints

- Preserve PPO behavior. In particular, keep the `approx_kl` host synchronization at the epoch boundary because it drives early stopping.
- Convert device metrics to plain Python scalars on the training thread before enqueueing. Do not retain GPU tensors in a background queue.
- Snapshot checkpoint/model weights before enqueueing. Never let a worker serialize mutable parameters while the optimizer may be updating them.
- Use explicit MLflow run identity in worker writes (for example, an `MlflowClient` with the captured run ID) rather than relying on thread-local active-run context. Start and end the run on the owning training thread.
- Drain and stop workers before ending the MLflow run or shutting down logging.
- Bound queues. Under normal load producers should return quickly; if a queue fills, apply explicit backpressure or a documented skip policy and report it. Do not silently drop metrics or checkpoint failures.
- Keep video/evaluation environments worker-owned. A worker must not use or close an environment while another thread is stepping it.
- Do not share live policy modules with inference workers. `Policy.fork()` currently isolates history but shares modules, and inference toggles module train/eval mode.

## Phase 0 — Make run shutdown/lifecycle explicit

1. Trace the current startup and shutdown path in `train.py` and `tracking/sinks/mlflow_sink.py`.
2. Ensure the owning training code closes the MLflow run on success, exceptions, and interruption, with the intended FINISHED/KILLED/FAILED status.
3. Make shutdown ordering explicit: stop accepting work, drain/join workers, end the MLflow run, then stop the logging listener.
4. Make close operations idempotent and ensure a worker failure is surfaced to the caller or recorded in the final run status.

This is a prerequisite for background writes: otherwise queued records could be written after the run has ended.

## Phase 1 — Background metric logging

1. Add an async metrics sink/decorator under `tracking/sinks/`, with one FIFO writer thread for short metric calls.
2. Enqueue immutable payloads of scalar metrics and step values. Keep the existing conversion and aggregation in `MetricsLoggingCallback`; do not move GPU work to the worker.
3. Route both PPO metric flushes and episode-stat flushes through the async sink.
4. Capture the MLflow run ID when the sink is created and use it for worker writes.
5. Add bounded-queue behavior, worker exception reporting, queue health/drop counters, a drain operation, and a synchronous configuration option for debugging.
6. Keep local metric console/file reporting behavior compatible.

**Acceptance criteria:** a deliberately slow metric sink does not stall normal collector flushes; metric payloads retain their step values; drain-on-close writes all queued metrics; sink failures are visible rather than silently swallowed.

## Phase 2 — Asynchronous checkpoints and artifacts

1. At each checkpoint trigger, make a defensive CPU snapshot of the state dict on the training thread, then enqueue the immutable snapshot for serialization/upload.
2. Move local checkpoint writes and MLflow state-dict uploads to an artifact writer, separate from the short metric queue.
3. Review final `log_model` behavior separately: enqueue only after creating a safe model snapshot, and check whether explicit requirements can avoid slow dependency inference.
4. Ensure final artifacts are drained before run closure.

**Acceptance criteria:** mutate training weights immediately after enqueueing and verify the saved checkpoint still contains the captured weights; artifact failures and backlog are reported; metrics are not starved behind a large artifact upload.

## Phase 3 — Deep-forked background evaluation and video

1. Add a deep-fork/snapshot path for inference that owns independent model modules and a copied history/state. Capture weights before returning control to training.
2. Run evaluation and video recording in a dedicated background worker with its own evaluation environment. The worker alone steps and closes that environment.
3. Preserve the existing recording trigger policy, but use single-flight scheduling: if a job is still active at the next trigger, skip or coalesce the trigger and report it instead of building an unbounded queue.
4. Keep model mode changes confined to the copied model. Do not call `eval()`/`train()` on modules used by PPO.
5. Drain or cancel according to a documented policy at shutdown, and wait for the worker before closing its environment.
6. Decide separately whether to enable the currently commented-out video artifact upload; if enabled, send it through the artifact writer.

Apply the same safe snapshot pattern to `EvaluationCallback`, even if evaluation and video use separate scheduling policies.

**Acceptance criteria:** evaluation/video can overlap training without changing the training model's weights, mode, or history; no environment is concurrently stepped/closed; slow jobs do not accumulate without bound; shutdown leaves no active worker or partial recording.

## Phase 4 — Tests, configuration, and measurement

- Unit-test FIFO/step preservation, bounded-queue behavior, worker exceptions, drain/join, idempotent close, and sync fallback.
- Test checkpoint snapshot immutability after the source model changes.
- Test deep-fork isolation: parameter updates and train/eval mode changes in one model do not affect the other.
- Test single-flight video/evaluation scheduling and worker-owned environment cleanup with fakes/mocks.
- Add configuration fields only where needed. If sample YAML or config assertions are changed, update their expectations together.
- Benchmark collector throughput and update latency with synchronous and asynchronous modes, including slow/failing sink simulations. Confirm async overhead is worthwhile and monitor queue depth.
- Preserve existing metric names, verbosity behavior, callback ordering where relevant, and the required KL early-stop synchronization.

## Suggested implementation order

1. Phase 0: lifecycle correctness.
2. Phase 1: async metric sink and tests.
3. Phase 2: checkpoint/artifact writer and snapshot tests.
4. Phase 3: deep-fork inference, background evaluation/video, and lifecycle tests.
5. Phase 4: configuration, documentation, and performance comparison.

## Explicitly out of scope

- Moving PPO's KL early-stop decision or GPU metric conversion off the training thread.
- Changing GAE, rollout-buffer conventions, callback semantics, or metric naming.
- Introducing multiprocessing or asyncio before measurements show background threads are insufficient.
