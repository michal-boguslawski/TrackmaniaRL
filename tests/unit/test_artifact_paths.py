"""Unit coverage for zero-padded step names in artifact paths.

Checkpoint and video artifacts are named after the step that produced them.
Padding to a fixed width keeps them in step order for anything that sorts
names as text, which is why the padding is applied when a name is built and
again by the backend on write.
"""

import pytest

from rl_lib.tracking.artifacts import (
    format_step,
    normalize_artifact_path,
    step_artifact_path,
)


# ---------------------------------------------------------------- building


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, "000"), (7, "007"), (42, "042"), (200, "200"), (1234, "1234")],
)
def test_format_step_pads_to_three_digits_without_truncating(step, expected):
    assert format_step(step) == expected


def test_format_step_width_is_configurable():
    assert format_step(7, digits=5) == "00007"


def test_format_step_requires_a_positive_width():
    with pytest.raises(ValueError, match="digits must be positive"):
        format_step(7, digits=0)


@pytest.mark.parametrize(
    ("directory", "step", "expected"),
    [
        ("checkpoints", 3, "checkpoints/step_003"),
        ("checkpoints", 1000, "checkpoints/step_1000"),
        ("videos/", 0, "videos/step_000"),
    ],
)
def test_step_artifact_path_names_the_step_last(directory, step, expected):
    assert step_artifact_path(directory, step) == expected


# ---------------------------------------------------------------- normalizing


@pytest.mark.parametrize(
    ("artifact_path", "expected"),
    [
        ("checkpoints/step_3", "checkpoints/step_003"),
        ("videos/step_0", "videos/step_000"),
        ("checkpoints/step_1234", "checkpoints/step_1234"),
        ("checkpoints/step_003", "checkpoints/step_003"),
        ("checkpoints/step_1/nested", "checkpoints/step_001/nested"),
    ],
)
def test_normalize_artifact_path_pads_every_step_segment(artifact_path, expected):
    assert normalize_artifact_path(artifact_path) == expected


def test_normalize_artifact_path_is_idempotent():
    once = normalize_artifact_path("videos/step_9")

    assert normalize_artifact_path(once) == once


@pytest.mark.parametrize(
    "artifact_path",
    ["final_model", "final_state_dict", "config/run_config.yaml", "checkpoints/latest"],
)
def test_normalize_artifact_path_leaves_other_names_alone(artifact_path):
    assert normalize_artifact_path(artifact_path) == artifact_path


def test_normalize_artifact_path_passes_none_through():
    assert normalize_artifact_path(None) is None
