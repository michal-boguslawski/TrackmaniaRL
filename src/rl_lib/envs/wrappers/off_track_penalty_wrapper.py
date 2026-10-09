"""Dense reward shaping for driving beyond the CarRacing road edge.

The wrapper reads the car pose and generated track centerline from Gymnasium's
CarRacing environment. These simulator attributes are internal APIs and should
be kept isolated here.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH
from numpy.typing import NDArray


def _distance_to_centerline(
    position: NDArray[np.float64], centerline: NDArray[np.float64]
) -> float:
    """Return the shortest distance from a point to a closed centerline."""
    starts = centerline
    ends = np.roll(centerline, shift=-1, axis=0)
    segments = ends - starts
    lengths_squared = np.einsum("ij,ij->i", segments, segments)
    valid = lengths_squared > 0.0
    if not np.any(valid):
        raise ValueError("track centerline must contain at least one nonzero segment")

    starts = starts[valid]
    segments = segments[valid]
    lengths_squared = lengths_squared[valid]
    projections = np.einsum("ij,ij->i", position - starts, segments) / lengths_squared
    projections = np.clip(projections, 0.0, 1.0)
    nearest = starts + projections[:, None] * segments
    distances = np.linalg.norm(nearest - position, axis=1)
    return float(np.min(distances))


class OffTrackPenaltyWrapper(gym.Wrapper):
    """Apply a dense penalty when the car's center leaves the road surface.

    The car position is measured in Box2D world coordinates and projected onto
    the generated track's closed centerline. CarRacing's ``TRACK_WIDTH`` is the
    road half-width. The penalty is zero while the car center remains within that
    width and increases quadratically, up to ``penalty_weight``, beyond the edge.

    Attributes:
        penalty_weight: Maximum magnitude of the per-step off-track penalty.
    """

    def __init__(
        self,
        env: gym.Env,
        penalty_weight: float = 0.05,
    ) -> None:
        """Initialize off-track reward shaping.

        Args:
            env: A CarRacing environment or wrapper around one.
            penalty_weight: Nonnegative maximum magnitude of the penalty per step.

        Raises:
            ValueError: If ``penalty_weight`` is negative or non-finite.
        """
        if not isfinite(penalty_weight) or penalty_weight < 0.0:
            raise ValueError("penalty_weight must be finite and nonnegative")

        super().__init__(env)
        self.penalty_weight = float(penalty_weight)
        self._centerline: NDArray[np.float64] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and cache its newly generated track centerline.

        Args:
            seed: Optional random seed passed to the environment.
            options: Optional reset options passed to the environment.

        Returns:
            The initial observation and reset information.

        Raises:
            TypeError: If the wrapped environment is not Gymnasium CarRacing.
            RuntimeError: If the environment did not generate a usable track.
        """
        observation, info = self.env.reset(seed=seed, options=options)
        base_env = self.unwrapped
        track = getattr(base_env, "track", None)
        if track is None or len(track) < 2 or getattr(base_env, "car", None) is None:
            raise TypeError(
                "OffTrackPenaltyWrapper requires a reset Gymnasium CarRacing environment"
            )

        centerline = np.asarray([(point[2], point[3]) for point in track], dtype=np.float64)
        if centerline.ndim != 2 or centerline.shape[1] != 2 or not np.isfinite(centerline).all():
            raise RuntimeError("CarRacing generated an invalid track centerline")
        self._centerline = centerline
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the environment and penalize the car for leaving the road.

        Args:
            action: Action passed through to CarRacing.

        Returns:
            The observation, shaped reward, termination flags, and info. The info
            dictionary includes ``track_lateral_offset`` and ``off_track_penalty``.

        Raises:
            RuntimeError: If called before reset or if the car pose is unavailable.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._centerline is None:
            raise RuntimeError("OffTrackPenaltyWrapper.step() called before reset()")

        car = getattr(self.unwrapped, "car", None)
        if car is None or getattr(car, "hull", None) is None:
            raise RuntimeError("CarRacing car pose is unavailable")
        position = np.asarray(tuple(car.hull.position), dtype=np.float64)
        distance = _distance_to_centerline(position, self._centerline)
        normalized_offset = min(distance / TRACK_WIDTH, 2.0)
        excess_offset = max(0.0, normalized_offset - 1.0)
        penalty = -self.penalty_weight * excess_offset**2

        info = dict(info)
        info["track_lateral_offset"] = normalized_offset
        info["off_track_penalty"] = penalty
        return observation, float(reward) + penalty, terminated, truncated, info
