"""Artifact path naming shared by the tracking backends.

Checkpoint and video artifacts are named after the step that produced them.
Zero-padding the step to a fixed width keeps them in the natural order when a
directory listing or an artifact browser sorts names as text, which is where
unpadded names such as ``step_9`` and ``step_10`` sort incorrectly.

Names are built with ``step_artifact_path`` by the callbacks that log them, and
``normalize_artifact_path`` is applied again by the backends on write, so any
step component reaching a logger ends up padded regardless of its origin.
"""

from __future__ import annotations

import re


STEP_DIGITS = 3
_STEP_SEGMENT = re.compile(r"^(step_)(\d+)$")


def format_step(step: int, digits: int = STEP_DIGITS) -> str:
    """Render a step as a zero-padded string.

    Args:
        step: Step counter to render.
        digits: Minimum width; wider steps keep all their digits.

    Returns:
        Step string padded to at least ``digits`` characters.

    Raises:
        ValueError: If ``digits`` is not positive.
    """
    if digits < 1:
        raise ValueError("digits must be positive")
    return f"{step:0{digits}d}"


def step_artifact_path(directory: str, step: int, digits: int = STEP_DIGITS) -> str:
    """Build an artifact path whose trailing segment is a padded step.

    Args:
        directory: Artifact directory (e.g. ``checkpoints``).
        step: Step counter to embed in the name.
        digits: Minimum width of the padded step.

    Returns:
        Artifact path such as ``checkpoints/step_003``.
    """
    return f"{directory.strip('/')}/step_{format_step(step, digits)}"


def normalize_artifact_path(artifact_path: str | None) -> str | None:
    """Pad every ``step_<digits>`` segment of an artifact path.

    Padding is idempotent, and segments that are not step names (including
    ``final_model``) are left untouched.

    Args:
        artifact_path: Artifact path, or None for backends that infer it.

    Returns:
        The path with padded step segments, or None if given None.
    """
    if artifact_path is None:
        return None
    segments = artifact_path.split("/")
    for index, segment in enumerate(segments):
        match = _STEP_SEGMENT.match(segment)
        if match:
            segments[index] = f"{match.group(1)}{format_step(int(match.group(2)))}"
    return "/".join(segments)
