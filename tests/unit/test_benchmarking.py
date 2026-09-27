"""Tests for reusable benchmark reporting and paired execution order."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarking import benchmark_callable, benchmark_callables
from scripts.benchmarks.compare_reports import compare_report_data


def test_benchmark_callables_warm_up_and_rotate_paired_measurements():
    calls: list[str] = []

    def baseline() -> None:
        calls.append("baseline")

    def candidate() -> None:
        calls.append("candidate")

    results = benchmark_callables(
        {"baseline": baseline, "candidate": candidate},
        warmups=2,
        repetitions=3,
    )

    assert calls == [
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert len(results["baseline"].samples_ms) == 3
    assert results["baseline"].min_ms <= results["baseline"].median_ms
    assert results["baseline"].median_ms <= results["baseline"].max_ms
    assert results["candidate"].speedup_over(results["baseline"]) > 0


def test_benchmark_result_serializes_samples_as_json_array():
    result = benchmark_callable(lambda: None, name="noop", warmups=0, repetitions=2)

    serialized = json.dumps(result.to_dict())

    assert json.loads(serialized)["name"] == "noop"
    assert len(json.loads(serialized)["samples_ms"]) == 2


@pytest.mark.parametrize(
    "operations,warmups,repetitions",
    [({}, 0, 1), ({"case": lambda: None}, -1, 1), ({"case": lambda: None}, 0, 0)],
)
def test_benchmark_callables_reject_invalid_settings(operations, warmups, repetitions):
    with pytest.raises(ValueError):
        benchmark_callables(operations, warmups=warmups, repetitions=repetitions)


def test_compare_report_data_checks_workload_and_computes_speedup():
    config = {
        "size": 64,
        "num_envs": 4,
        "stack_size": 4,
        "observation_shape": [32, 32, 1],
        "action_dim": 3,
        "device": "cpu",
        "threads": 1,
        "warmups": 1,
        "repetitions": 5,
        "seed": 123,
        "torch_version": "2.x",
        "python_version": "3.x",
    }
    baseline = {
        "configuration": config,
        "results": {
            "add": {"median_ms": 10.0, "peak_cuda_bytes": None},
        },
    }
    candidate = {
        "configuration": {**config, "label": "candidate"},
        "results": {
            "add": {"median_ms": 5.0, "peak_cuda_bytes": None},
        },
    }

    comparisons = compare_report_data(baseline, candidate)

    assert comparisons[0]["speedup"] == 2.0
    assert comparisons[0]["peak_cuda_bytes_delta"] is None


def test_compare_report_data_rejects_mismatched_workloads():
    baseline = {
        "configuration": {"size": 64},
        "results": {"add": {"median_ms": 10.0, "peak_cuda_bytes": None}},
    }
    candidate = {
        "configuration": {"size": 128},
        "results": {"add": {"median_ms": 5.0, "peak_cuda_bytes": None}},
    }

    with pytest.raises(ValueError, match="size"):
        compare_report_data(baseline, candidate)
