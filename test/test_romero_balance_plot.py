"""Romero's voltage and volt-second balance as a registered plot (#781, #1590)."""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

import vaft
import vaft.omas
import vaft.plot
from vaft.omas.process_wrapper import compute_romero_flux_balance_ods
from vaft.omas.resistive_zeff import current_carrying_window
from vaft.plot.models import Panels

NAME = "summary_time_romero_balance"


@pytest.fixture(scope="module")
def ods():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


def _extract(ods, **options):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.plot.extract(NAME, ods, **options)


def _reference(ods, R_p, I_ni=0.0, time_range=None):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = current_carrying_window(ods) if time_range is None else time_range
        return compute_romero_flux_balance_ods(ods, R_p=R_p, I_ni=I_ni, time_range=window)


def _series(panel, label):
    (match,) = [s for s in panel.series if s.label == label]
    return match


def test_without_a_resistance_only_the_implied_resistive_part_is_drawn(ods):
    model = _extract(ods)
    assert isinstance(model, Panels) and len(model.models) == 4
    voltages, flux, inductance, resistance = model.models
    ref = _reference(ods, 0.0)
    np.testing.assert_allclose(_series(voltages, "V_B = -dpsi_B/dt (boundary)").y, ref["V_B"])
    np.testing.assert_allclose(_series(voltages, "V_I (internal inductive)").y, ref["V_I"])
    np.testing.assert_allclose(_series(voltages, "V_B - V_I (implied resistive)").y, ref["V_B"] - ref["V_I"])
    np.testing.assert_allclose(_series(flux, "-(psi_B - psi_B(t0)) direct").y, ref["Phi_B_direct"])
    np.testing.assert_allclose(_series(inductance, "L_i = mu0 R0 li_3 / 2").y, ref["L_i"] * 1e6)
    labels = [s.label for panel in model.models for s in panel.series]
    assert not [label for label in labels if label.startswith(("V_R =", "residual", "Phi_R", "closure", "R_p"))]
    assert "implied, not measured" in voltages.title
    assert "I_ni = 0 (purely Ohmic, assumed)" in model.suptitle


def test_a_given_resistance_adds_v_r_and_the_closure_residuals(ods):
    t = _reference(ods, 0.0)["time"]
    r_p = np.linspace(30e-6, 10e-6, t.size)
    model = _extract(ods, plasma_resistance=r_p)
    voltages, flux, _, resistance = model.models
    ref = _reference(ods, r_p)
    np.testing.assert_allclose(_series(voltages, "V_R = R_p (I_p - I_ni)").y, ref["V_R"])
    np.testing.assert_allclose(_series(voltages, "residual V_B - V_R - V_I").y,
                               ref["V_B"] - ref["V_R"] - ref["V_I"])
    np.testing.assert_allclose(_series(flux, "closure Phi_B - Phi_R - Phi_I").y, ref["Phi_closure"])
    np.testing.assert_allclose(_series(resistance, "R_p given").y, r_p * 1e6)
    assert not [s for s in voltages.series if "implied" in s.label]


def test_the_non_inductive_current_is_used_and_stated(ods):
    model = _extract(ods, non_inductive_current=1e3)
    ref = _reference(ods, 0.0, I_ni=1e3)
    closing = _series(model.models[3], "R_closing = (V_B - V_I)/(I_p - I_ni)")
    np.testing.assert_allclose(closing.y, ref["R_closing"] * 1e6)
    assert "I_ni = 1000 A (given)" in model.suptitle


def test_time_range_windows_the_balance(ods):
    full = _extract(ods).models[0].series[0].x
    window = (float(full[0]), float(full[4]))
    windowed = _extract(ods, time_range=window).models[0].series[0].x
    np.testing.assert_array_equal(windowed, full[:5])


@pytest.mark.parametrize("bad", [[1e-5, 2e-5], -1e-5, float("nan"), "spitzer"])
def test_a_resistance_that_is_not_one_value_per_slice_in_ohm_is_refused(ods, bad):
    with pytest.raises(ValueError, match="plasma_resistance"):
        _extract(ods, plasma_resistance=bad)


def test_the_view_leaves_the_ods_as_it_was(ods):
    before = sorted(ods.flat())
    _extract(ods, plasma_resistance=2e-5)
    assert sorted(ods.flat()) == before
