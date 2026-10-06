"""A current pulse that is pickup is not a plasma (issue #1733).

48927 was labelled ``Plasma`` from 3.3 kA that started at 0.296 s and never
ended, with a dark H-alpha and a responding gauge: PF pickup on the Rogowski,
which the pulse detector's noise-relative threshold accepts.  A census of
production products put every dark, never-ending pulse in that class.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.omas.shot_class import MIN_PLASMA_CURRENT_A, shot_class

from _plasma_timing_fixtures import current, grid, light, synthetic_ods


def _shot(ip, *, lit: bool, puff: bool = True, seed: int = 1733):
    t = grid()
    rng = np.random.default_rng(seed)
    dark = 0.002 * rng.standard_normal(t.size)
    ods = synthetic_ods(slow=light(t) if lit else dark, ip=ip(t), t=t)
    ods["barometry.ids_properties.homogeneous_time"] = 0
    ods["barometry.gauge.0.pressure.time"] = t
    ods["barometry.gauge.0.pressure.data"] = (
        1.0 + np.exp(-((t - 0.29) / 0.01) ** 2) if puff else 1.0 + 1e-6 * rng.standard_normal(t.size)
    )
    return ods


def _unended(peak: float):
    """48927: a ramp from 0.296 s that is still rising when the record ends."""

    def ip(t):
        rng = np.random.default_rng(48927)
        return peak * np.clip((t - 0.296) / (t[-1] - 0.296), 0.0, 1.0) + 77.0 * rng.standard_normal(t.size)

    return ip


def test_48927_pickup_with_dark_light_is_a_breakdown_failure():
    record = shot_class(_shot(_unended(3300.0), lit=False))
    assert record.ip_pulse  # the detector does find a pulse ...
    assert record.label == "BD failure"  # ... but it is not a discharge
    assert record.decided_by == "pressure_response"
    assert "ip_pulse_dark_unended" in record.flags
    assert "halpha_dark_with_ip_pulse" in record.flags
    assert "never ends" in record.reason and record.reason.startswith("no discharge current")


def test_a_large_dark_drift_that_never_ends_is_not_a_plasma():
    """43690/45780: a drifting Rogowski reads hundreds of kA with no light (#1373)."""
    record = shot_class(_shot(_unended(100e3), lit=False))
    assert record.label != "Plasma"
    assert "ip_pulse_dark_unended" in record.flags


def test_a_lit_pulse_below_the_floor_is_a_breakdown_failure():
    record = shot_class(_shot(lambda t: current(t, peak=1000.0, noise=30.0), lit=True))
    assert record.ip_pulse
    assert record.label == "BD failure"
    assert "ip_pulse_below_floor" in record.flags


def test_a_small_lit_discharge_that_ends_stays_a_plasma():
    """48807-48815: 5-15 kA, light seen, the pulse ends near 0.312 s."""
    record = shot_class(_shot(lambda t: current(t, peak=8000.0), lit=True))
    assert record.label == "Plasma" and record.decided_by == "ip_pulse"
    assert not {"ip_pulse_below_floor", "ip_pulse_dark_unended"} & set(record.flags)


def test_darkness_alone_does_not_refuse_a_discharge_that_ends():
    """73 dark pulses in the census end inside the record; tens of kA of them are plasmas."""
    record = shot_class(_shot(lambda t: current(t, peak=60e3), lit=False))
    assert record.label == "Plasma"
    assert "ip_pulse_dark_unended" not in record.flags


@pytest.mark.parametrize(("floor", "label"), [(500.0, "Plasma"), (MIN_PLASMA_CURRENT_A, "BD failure")])
def test_the_floor_is_a_parameter(floor, label):
    ods = _shot(lambda t: current(t, peak=1000.0, noise=30.0), lit=True)
    assert shot_class(ods, min_plasma_current=floor).label == label


def test_a_dark_pulse_that_ends_by_collapse_is_kept():
    """Pinned on purpose: a dark discharge with a vessel-current tail ends by a
    collapse, not at the span end, and must not be refused for darkness alone."""

    def ip(t):
        rng = np.random.default_rng(4)
        y = 20e3 * np.clip((t - 0.300) / 0.05, 0.0, 1.0)
        y[t >= 0.355] = 5e3
        return y + 77.0 * rng.standard_normal(t.size)

    record = shot_class(_shot(ip, lit=False))
    assert "ip_pulse_dark_unended" not in record.flags
    assert record.label == "Plasma"
