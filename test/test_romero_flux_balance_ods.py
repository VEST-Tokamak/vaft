"""Romero's balance on the packaged VEST equilibria (#781, #652).

The three packaged shots are the only multi-slice equilibria in the wheel, and
they store psi in both conventions -- 39915 in weber, 41524 and 41672 per
radian -- so the adapter's convention handling meets real data on both
branches.  The check that decides it is independent of the equilibrium: the
boundary volt-seconds must match the inboard midplane flux loop's over the
same window to within the gap between the two surfaces, where a 2 pi slip on
either branch would put the ratio near 6 or 0.16 (#652).
"""

import logging

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from vaft.omas.process_wrapper import compute_romero_flux_balance_ods
from vaft.omas.sample import sample_ods

# Flux-closure windows: every slice with current, and for 41672 not the
# collapsing reconstructions after 0.345 s.
WINDOWS = {39915: (0.316, 0.326), 41524: (0.328, 0.334), 41672: (0.322, 0.341)}
PER_RADIAN = {39915: False, 41524: True, 41672: True}


@pytest.fixture(scope="module", params=sorted(WINDOWS))
def balance(request):
    shot = request.param
    logging.disable(logging.WARNING)
    try:
        out = compute_romero_flux_balance_ods(
            sample_ods(shot), R_p=0.0, I_ni=0.0, time_range=WINDOWS[shot]
        )
    finally:
        logging.disable(logging.NOTSET)
    return shot, out


def test_both_storage_conventions_are_met_and_brought_to_romero_sign(balance):
    shot, out = balance
    assert out["psi_per_radian"] is PER_RADIAN[shot]
    # psi grows outward for positive current in these equilibria.
    assert out["flux_sign"] == -1.0


def test_the_boundary_volt_seconds_match_the_inboard_flux_loop(balance):
    # Flux loop nearest the inboard midplane: it sits a few centimetres off
    # the plasma, so its voltage differs from V_B by d(psi_loop - psi_B)/dt,
    # not by a factor.
    shot, out = balance
    ods = sample_ods(shot)
    loops = ods["magnetics.flux_loop"]
    inboard = min(
        range(len(loops)),
        key=lambda i: (abs(float(loops[i]["position.0.z"])), float(loops[i]["position.0.r"])),
    )
    loop_time = np.asarray(
        loops[inboard]["flux.time"] if "flux.time" in loops[inboard] else ods["magnetics.time"],
        dtype=float,
    )
    flux = np.interp(out["time"], loop_time, np.asarray(loops[inboard]["flux.data"], float))
    # Compared as volt-seconds over the window rather than sample by sample:
    # V_B crosses zero late in 41524, where a pointwise ratio means nothing.
    ratio = abs((flux[-1] - flux[0]) / out["Phi_B"][-1])
    assert 0.7 < ratio < 1.6, ratio


def test_the_closing_resistance_is_positive_and_spitzer_sized(balance):
    # 2-50 micro-ohm, falling through the discharge: a Spitzer ring at a few
    # tens of eV.  Not the 2 pi check -- a slip can stay inside these bounds.
    _, out = balance
    closing = out["R_closing"]
    assert np.all(closing > 0.0)
    assert np.all(closing < 1e-4)


def test_the_internal_inductance_is_a_peaked_ohmic_one(balance):
    _, out = balance
    assert np.all((out["li_3"] > 0.35) & (out["li_3"] < 0.65))
    np.testing.assert_allclose(out["L_i"], 0.5 * 4e-7 * np.pi * out["R0"] * out["li_3"], rtol=1e-6)


def test_the_boundary_budget_agrees_with_the_direct_flux_change(balance):
    _, out = balance
    assert out["Phi_B"][-1] == pytest.approx(out["Phi_B_direct"][-1], rel=0.08)


def test_a_window_reaching_a_currentless_slice_is_refused():
    logging.disable(logging.WARNING)
    try:
        with pytest.raises(ValueError, match="flux-closure window"):
            compute_romero_flux_balance_ods(sample_ods(39915), R_p=0.0, I_ni=0.0)
        with pytest.raises(ValueError, match="three"):
            compute_romero_flux_balance_ods(
                sample_ods(39915), R_p=0.0, I_ni=0.0, time_range=(0.316, 0.317)
            )
    finally:
        logging.disable(logging.NOTSET)
