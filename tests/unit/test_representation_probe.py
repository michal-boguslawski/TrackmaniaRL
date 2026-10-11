"""Tests for the offline frozen-feature representation probe."""

from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
from pathlib import Path
import pytest
import sys
import torch as T

_PROBE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "representation_probe.py"
)
_PROBE_SPEC = spec_from_file_location("representation_probe", _PROBE_PATH)
assert _PROBE_SPEC is not None and _PROBE_SPEC.loader is not None
_PROBE_MODULE = module_from_spec(_PROBE_SPEC)
sys.modules[_PROBE_SPEC.name] = _PROBE_MODULE
_PROBE_SPEC.loader.exec_module(_PROBE_MODULE)

discounted_returns = _PROBE_MODULE.discounted_returns
explained_variance = _PROBE_MODULE.explained_variance
split_episode_indices = _PROBE_MODULE.split_episode_indices


def test_discounted_returns_bootstraps_truncated_episode() -> None:
    """A truncation bootstrap should propagate backward through the episode."""
    np.testing.assert_allclose(
        discounted_returns([1.0, 2.0], gamma=0.5, bootstrap_value=4.0),
        [3.0, 4.0],
    )


def test_discounted_returns_uses_zero_for_terminated_episode() -> None:
    """A terminated episode should not bootstrap beyond its final reward."""
    np.testing.assert_allclose(
        discounted_returns([1.0, 2.0], gamma=0.5),
        [2.0, 2.0],
    )


def test_split_episode_indices_are_disjoint_and_cover_dataset() -> None:
    """Every episode belongs to exactly one of the three partitions."""
    splits = split_episode_indices(100, seed=17)
    flattened = np.concatenate(splits)

    assert [len(split) for split in splits] == [70, 15, 15]
    assert len(np.unique(flattened)) == 100
    np.testing.assert_array_equal(np.sort(flattened), np.arange(100))


def test_split_episode_indices_requires_nonempty_test_split() -> None:
    """Small datasets cannot silently produce an empty test partition."""
    with pytest.raises(ValueError, match="at least four"):
        split_episode_indices(3, seed=0)


def test_explained_variance_is_one_for_exact_predictions() -> None:
    """Perfect predictions explain all target variance."""
    targets = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert explained_variance(targets, targets.copy()) == pytest.approx(1.0)


def test_critic_affine_calibration_fits_train_only_and_scores_test() -> None:
    """Calibration should recover target scale/offset without using test labels."""
    train_values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    train_targets = 3.0 * train_values + 4.0
    test_values = np.asarray([4.0, 5.0], dtype=np.float32)
    test_targets = 3.0 * test_values + 4.0

    result = _PROBE_MODULE._fit_affine_critic_calibration(
        train_values, train_targets, test_values, test_targets
    )

    assert result["scale"] == pytest.approx(3.0)
    assert result["offset"] == pytest.approx(4.0)
    assert result["fit_split"] == "training episodes only"
    assert result["test"]["explained_variance"] == pytest.approx(1.0)
    assert result["test"]["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_probe_fitting_helpers_return_test_scores() -> None:
    """Both probe implementations score held-out samples without shape errors."""
    rng = np.random.default_rng(11)
    features = rng.normal(size=(120, 4)).astype(np.float32)
    targets = (2.0 * features[:, 0] - features[:, 1]).astype(np.float32)
    ridge = _PROBE_MODULE._ridge_probe(
        features[:70], targets[:70],
        features[70:95], targets[70:95],
        features[95:], targets[95:],
        T.device("cpu"),
    )
    mlp = _PROBE_MODULE._mlp_probe(
        features[:70], targets[:70],
        features[70:95], targets[70:95],
        features[95:], targets[95:],
        T.device("cpu"), seed=3, epochs=2, patience=1, batch_size=16,
    )

    assert ridge["test_explained_variance"] > 0.99
    assert np.isfinite(mlp["test_explained_variance"])
