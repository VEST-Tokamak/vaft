"""TF-current acquisition excursions are repaired, healthy records untouched (#1543)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.machine_mapping.tf import repair_tf_excursions
from vaft.machine_mapping.utils import resolve_vest_diagnostic

TIME = np.arange(0.0, 1.0, 4e-5)


@pytest.fixture(scope="module")
def config():
    return resolve_vest_diagnostic(48245, "tf")["processing"]["excursion_repair"]


def _tf(seed: int = 0, plateau: float = 12000.0, noise: float = 1500.0) -> tuple[np.ndarray, np.ndarray]:
    """A TF current: ramp to the plateau by 0.15 s, slow decay, ~13 % rms raw noise."""
    true = plateau * np.clip(TIME / 0.15, 0, 1) * np.exp(-np.clip(TIME - 0.2, 0, None) / 0.8)
    return true, true + noise * np.random.default_rng(seed).standard_normal(TIME.size)


def test_the_repair_is_enabled_for_vest(config):
    assert config["enabled"] is True
    assert config["window"] == [0.25, 0.40]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_healthy_record_is_left_exactly_as_it_was(config, seed):
    _true, measured = _tf(seed)
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert intervals == []
    np.testing.assert_array_equal(repaired, measured)


@pytest.mark.parametrize("shift", [-25000.0, +13000.0])
def test_the_48245_excursion_is_bridged_to_within_one_percent(config, shift):
    """The trace leaves the plateau from ~0.303 s to ~0.326 s, as on 48238-48269."""
    true, measured = _tf()
    excursion = (TIME > 0.303) & (TIME < 0.326)
    measured[excursion] += shift
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start <= 0.303 and end >= 0.326 and end - start < 0.030
    span = (TIME > 0.30) & (TIME < 0.33)
    assert abs(np.mean(repaired[span] - true[span])) < 0.01 * 12000.0
    outside = ~((TIME >= start) & (TIME <= end))
    np.testing.assert_array_equal(repaired[outside], measured[outside])


def test_a_shot_with_the_tf_off_is_not_touched(config):
    _true, measured = _tf(plateau=10.0, noise=50.0)
    measured[(TIME > 0.30) & (TIME < 0.32)] += 500.0
    repaired, intervals = repair_tf_excursions(TIME, measured, config)
    assert intervals == []
    np.testing.assert_array_equal(repaired, measured)


def test_a_disabled_repair_returns_the_record_unchanged(config):
    _true, measured = _tf()
    measured[(TIME > 0.303) & (TIME < 0.326)] -= 25000.0
    repaired, intervals = repair_tf_excursions(TIME, measured, config | {"enabled": False})
    assert intervals == []
    np.testing.assert_array_equal(repaired, measured)
