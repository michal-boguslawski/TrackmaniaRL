"""Reusable timing helpers for paired performance benchmarks.

The helpers run warmups and repeated measurements, report robust latency
statistics, and optionally synchronize CUDA work and record peak allocated
memory. They intentionally do not impose a benchmark framework dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
from statistics import fmean, median
from time import perf_counter

import torch as T


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Latency summary for one benchmark case.

    Attributes:
        name: Label supplied for the benchmark case.
        samples_ms: Individual measured durations in milliseconds.
        median_ms: Median measured duration.
        mean_ms: Arithmetic mean of measured durations.
        p90_ms: Nearest-rank 90th percentile duration.
        min_ms: Fastest measured duration.
        max_ms: Slowest measured duration.
        peak_cuda_bytes: Largest increase in allocated CUDA memory during a
            measured call, or None when CUDA memory was not measured.
    """

    name: str
    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    peak_cuda_bytes: int | None

    def speedup_over(self, other: BenchmarkResult) -> float:
        """Return ``other``'s median latency divided by this result's latency.

        Values above 1 indicate this result is faster than ``other``.

        Args:
            other: Baseline result to compare against.

        Returns:
            Median latency speedup ratio.
        """
        return other.median_ms / self.median_ms

    def to_dict(self) -> dict[str, str | float | int | list[float] | None]:
        """Convert the result to a JSON-serializable mapping."""
        result = asdict(self)
        result["samples_ms"] = list(self.samples_ms)
        return result


def _resolve_device(device: T.device | str | None) -> T.device | None:
    """Resolve and validate an optional synchronization device."""
    if device is None:
        return None
    resolved = T.device(device)
    if resolved.type == "cuda" and not T.cuda.is_available():
        raise RuntimeError(f"CUDA synchronization requested for unavailable device {resolved}")
    return resolved


def _synchronize(device: T.device | None) -> None:
    """Wait for pending work when measuring CUDA operations."""
    if device is not None and device.type == "cuda":
        T.cuda.synchronize(device)


def _run_once(
    operation: Callable[[], object],
    device: T.device | None,
) -> tuple[float, int | None]:
    """Time one invocation and return duration and CUDA allocation delta."""
    _synchronize(device)
    starting_allocated = None
    if device is not None and device.type == "cuda":
        starting_allocated = T.cuda.memory_allocated(device)
        T.cuda.reset_peak_memory_stats(device)

    started = perf_counter()
    output = operation()
    _synchronize(device)
    elapsed_ms = (perf_counter() - started) * 1_000

    peak_cuda_bytes = None
    if starting_allocated is not None:
        peak_cuda_bytes = max(
            0,
            T.cuda.max_memory_allocated(device) - starting_allocated,
        )
    del output
    return elapsed_ms, peak_cuda_bytes


def _summarize(
    name: str,
    samples_ms: list[float],
    peak_cuda_bytes: list[int],
) -> BenchmarkResult:
    """Build a summary from raw measurements."""
    ordered = sorted(samples_ms)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return BenchmarkResult(
        name=name,
        samples_ms=tuple(samples_ms),
        median_ms=median(samples_ms),
        mean_ms=fmean(samples_ms),
        p90_ms=ordered[p90_index],
        min_ms=ordered[0],
        max_ms=ordered[-1],
        peak_cuda_bytes=max(peak_cuda_bytes) if peak_cuda_bytes else None,
    )


def benchmark_callables(
    operations: Mapping[str, Callable[[], object]],
    *,
    warmups: int = 2,
    repetitions: int = 10,
    device: T.device | str | None = None,
) -> dict[str, BenchmarkResult]:
    """Benchmark multiple callables with paired, rotating measurement order.

    Every operation receives the same number of warmups and measurements.
    During measurement, case order rotates each repetition to reduce systematic
    bias from cache state, temperature, or clock drift.

    Args:
        operations: Mapping from unique case labels to zero-argument callables.
        warmups: Number of untimed invocations per case.
        repetitions: Number of measured invocations per case; must be positive.
        device: Optional CUDA device to synchronize around each invocation and
            use for peak allocated-memory measurements. Leave unset for CPU-only
            operations.

    Returns:
        Benchmark summaries keyed by the same labels as ``operations``.

    Raises:
        ValueError: If there are no operations, warmups are negative, or
            repetitions are not positive.
        RuntimeError: If CUDA synchronization is requested but unavailable.
    """
    if not operations:
        raise ValueError("at least one benchmark operation is required")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    resolved_device = _resolve_device(device)
    names = list(operations)
    for _ in range(warmups):
        for name in names:
            _synchronize(resolved_device)
            output = operations[name]()
            _synchronize(resolved_device)
            del output

    samples = {name: [] for name in names}
    peaks: dict[str, list[int]] = {name: [] for name in names}
    for repetition in range(repetitions):
        offset = repetition % len(names)
        ordered_names = names[offset:] + names[:offset]
        for name in ordered_names:
            duration_ms, peak_bytes = _run_once(operations[name], resolved_device)
            samples[name].append(duration_ms)
            if peak_bytes is not None:
                peaks[name].append(peak_bytes)

    return {
        name: _summarize(name, samples[name], peaks[name])
        for name in names
    }


def benchmark_callable(
    operation: Callable[[], object],
    *,
    name: str = "operation",
    warmups: int = 2,
    repetitions: int = 10,
    device: T.device | str | None = None,
) -> BenchmarkResult:
    """Benchmark one callable and return its latency summary.

    Args:
        operation: Zero-argument callable to measure.
        name: Display label for the result.
        warmups: Number of untimed invocations.
        repetitions: Number of measured invocations; must be positive.
        device: Optional CUDA device to synchronize and measure memory on.

    Returns:
        Summary of repeated execution times.
    """
    return benchmark_callables(
        {name: operation},
        warmups=warmups,
        repetitions=repetitions,
        device=device,
    )[name]
