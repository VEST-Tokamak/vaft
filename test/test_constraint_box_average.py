"""EFIT constraints are box averages of the samples inside the window (issue #433).

The legacy writer interpolated the 25 kHz diagnostics onto a 1e-4 s grid of
its own and averaged that: about 60 % of the samples never counted, and the
rest counted with arbitrary weights.  Every sample in the window now counts
once, and a window with no sample is an error rather than a fiction.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.code.efit.legacy import box_average, vfit_equilibrium_form_constraints

DT = 4e-5
TIME = np.arange(0.26, 0.36, DT)


def test_the_box_average_uses_every_sample_once_with_equal_weight():
    data = np.zeros(TIME.size)
    centre = 0.30
    inside = np.abs(TIME - centre) <= 0.0005
    data[inside] = 1.0
    data[np.flatnonzero(inside)[3]] = 26.0  # one spike inside the window
    n = int(inside.sum())
    assert n == 25
    expected = (26.0 + 1.0 * (n - 1)) / n  # the spike counts exactly once, 1/N
    assert box_average(TIME, data, centre, 0.0005) == pytest.approx(expected)


def test_the_window_is_closed_and_no_sample_is_interpolated():
    data = TIME * 1e3  # linear ramp: interpolation and box average agree only if all samples count
    centre, half = 0.30000, 0.0005
    inside = (TIME >= centre - half) & (TIME <= centre + half)
    assert box_average(TIME, data, centre, half) == pytest.approx(float(np.mean(data[inside])))
    # A grid whose only inside sample is the boundary one is still averaged.
    coarse = np.array([0.2990, 0.2995, 0.3005, 0.3020])
    assert box_average(coarse, np.array([1.0, 2.0, 3.0, 4.0]), 0.30, 0.0005) == pytest.approx(2.5)


def test_a_window_with_no_sample_is_an_error_named_after_the_constraint():
    with pytest.raises(ValueError, match=r"ip: no sample inside \[0\.4995, 0\.5005\]"):
        box_average(TIME, np.ones(TIME.size), 0.5, 0.0005, what="ip")
    with pytest.raises(ValueError, match="samples against"):
        box_average(TIME, np.ones(3), 0.3, 0.0005)


def _equilibrium(window: float) -> ODS:
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = TIME
    ods["magnetics.ip.0.time"] = TIME
    ods["magnetics.ip.0.data"] = 1.0e5 * np.sin((TIME - 0.26) / 0.1 * np.pi)
    ods["magnetics.ip.0.data_error_upper"] = np.full(TIME.size, 100.0)
    ods["magnetics.b_field_pol_probe.0.identifier"] = "P0"
    ods["magnetics.b_field_pol_probe.0.field.data"] = 0.05 * np.cos(2 * np.pi * 50 * TIME)
    ods["magnetics.b_field_pol_probe.0.field.data_error_upper"] = np.full(TIME.size, 1e-3)
    ods["magnetics.flux_loop.0.identifier"] = "L0"
    ods["magnetics.flux_loop.0.flux.data"] = 0.01 * TIME
    ods["magnetics.flux_loop.0.flux.data_error_upper"] = np.full(TIME.size, 1e-4)
    EQ = ods["equilibrium"]
    vfit_equilibrium_form_constraints(
        EQ, ods["pf_active"], ods["magnetics"], ods["tf"], [0.30, 0.31],
        ["bpol_probe", "flux_loop", "ip"], window,
    )
    return EQ


def test_every_constraint_family_is_the_box_average_of_its_samples():
    EQ = _equilibrium(0.0005)
    inside = np.abs(TIME - 0.31) <= 0.0005
    ip = 1.0e5 * np.sin((TIME - 0.26) / 0.1 * np.pi)
    assert float(EQ["time_slice.1.constraints.ip.measured"]) == pytest.approx(float(np.mean(ip[inside])))
    assert float(EQ["time_slice.1.constraints.ip.measured_error_upper"]) == pytest.approx(100.0)
    probe = 0.05 * np.cos(2 * np.pi * 50 * TIME)
    assert float(EQ["time_slice.1.constraints.bpol_probe.0.measured"]) == pytest.approx(float(np.mean(probe[inside])))
    assert float(EQ["time_slice.1.constraints.flux_loop.0.measured"]) == pytest.approx(float(np.mean(0.01 * TIME[inside])))


def test_the_window_is_a_control_of_its_own():
    narrow = _equilibrium(0.0001)
    wide = _equilibrium(0.0020)
    probe = 0.05 * np.cos(2 * np.pi * 50 * TIME)
    for EQ, half in ((narrow, 0.0001), (wide, 0.0020)):
        inside = np.abs(TIME - 0.30) <= half
        assert float(EQ["time_slice.0.constraints.bpol_probe.0.measured"]) == pytest.approx(float(np.mean(probe[inside])))
    assert float(narrow["time_slice.0.constraints.bpol_probe.0.measured"]) != float(wide["time_slice.0.constraints.bpol_probe.0.measured"])


def test_a_slice_outside_the_diagnostics_grid_is_refused():
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = TIME
    ods["magnetics.ip.0.time"] = TIME
    ods["magnetics.ip.0.data"] = np.ones(TIME.size)
    ods["magnetics.ip.0.data_error_upper"] = np.ones(TIME.size)
    with pytest.raises(ValueError, match="ip: no sample inside"):
        vfit_equilibrium_form_constraints(ods["equilibrium"], ods["pf_active"], ods["magnetics"], ods["tf"], [0.50], ["ip"], 0.0005)
