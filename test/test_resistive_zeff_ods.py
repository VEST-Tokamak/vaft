"""Resistive Z_eff on packaged VEST data: the #652 / #1214 Sec. 4 conventions and Level C-D.

The three multi-slice samples store psi in both conventions -- 39915 in weber,
41524 and 41672 per radian -- so the boundary-flux reader meets both branches.
What decides a 2 pi slip is independent of the equilibrium: the boundary
volt-seconds against the inboard midplane flux loop, where a slip on either
branch puts the ratio near 6 or 0.16.  The Ejima loop voltage of
``loop_voltage_from_total_flux`` is checked against Romero's ``V_B`` on the
same data, which closes the branch #652 left uncovered.

48224 is the one kinetic sample: Spitzer/Sauter/Redl are compared on it
(Level C), and NEO's conductivity case on it is compared at the charge NEO
itself used (Level D, offline fixture, no solver).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas")
pytest.importorskip("skimage")

from vaft.omas.resistive_zeff import (
    detected_psi_per_radian,
    flux_surface_state_ods,
    romero_boundary_flux_ods,
)
from vaft.omas.sample import sample_ods
from vaft.process.resistive_zeff import (
    Smoothing,
    TabulatedConductivity,
    model_resistance,
    observed_resistance,
)

WINDOWS = {39915: (0.316, 0.326), 41524: (0.328, 0.334), 41672: (0.322, 0.341)}
PER_RADIAN = {39915: False, 41524: True, 41672: True}


@pytest.fixture(scope="module", params=sorted(WINDOWS))
def flux(request):
    shot = request.param
    logging.disable(logging.WARNING)
    try:
        return shot, romero_boundary_flux_ods(sample_ods(shot), time_range=WINDOWS[shot],
                                              source={"shot": shot})
    finally:
        logging.disable(logging.NOTSET)


def test_the_storage_branch_is_detected_not_defaulted(flux):
    shot, got = flux
    assert got.flux_normalization == (
        "stored Wb/rad, multiplied by 2 pi" if PER_RADIAN[shot] else "stored full Wb")
    ods = sample_ods(shot)
    assert detected_psi_per_radian(ods, 0) is PER_RADIAN[shot]


def test_the_boundary_voltage_drives_the_current_and_matches_the_flux_loop(flux):
    shot, got = flux
    observed = observed_resistance(got, I_ni=0.0, smoothing=Smoothing("none"))
    # Romero's sign: the early flat-top V_B is positive for positive I_p.
    assert np.all(got.I_p > 0.0)
    assert observed.V_B[0] > 0.0
    ods = sample_ods(shot)
    loops = ods["magnetics.flux_loop"]
    inboard = min(range(len(loops)), key=lambda i: (abs(float(loops[i]["position.0.z"])),
                                                    float(loops[i]["position.0.r"])))
    loop_time = np.asarray(loops[inboard]["flux.time"] if "flux.time" in loops[inboard]
                           else ods["magnetics.time"], dtype=float)
    loop = np.interp(got.time, loop_time, np.asarray(loops[inboard]["flux.data"], float))
    ratio = abs((loop[-1] - loop[0]) / (got.psi_boundary[-1] - got.psi_boundary[0]))
    assert 0.7 < ratio < 1.6, ratio  # a 2 pi slip lands at 0.17-0.22 or 6.6-8.6


def test_the_ejima_loop_voltage_is_minus_v_b_on_the_same_data(flux):
    """#652's uncovered branch: +2 pi dpsi/dt on per-radian psi vs -dpsi/dt full Wb."""
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
    from vaft.formula.equilibrium import loop_voltage_from_total_flux

    shot, got = flux
    ods = sample_ods(shot)
    idx = [int(i) for i in got.source["time_index"].split(",")]
    stored = np.array([float(ods["equilibrium.time_slice"][i]["global_quantities.psi_boundary"])
                       for i in idx])
    per_radian = stored * ods_psi_to_wb_per_radian_factor(ods, idx[0])
    ejima = np.asarray(loop_voltage_from_total_flux(got.time, per_radian, psi_per_radian=True))
    romero = -np.gradient(got.psi_boundary, got.time)
    # Same magnitude; the sign differs by the stored COCOS times Romero's sign flip.
    np.testing.assert_allclose(np.abs(ejima), np.abs(romero), rtol=1e-9, atol=1e-9)
    assert np.allclose(ejima, -got.flux_sign * romero, rtol=1e-9, atol=1e-9)


def test_the_observed_resistance_is_the_closing_resistance_of_the_balance(flux):
    from vaft.omas.process_wrapper import compute_romero_flux_balance_ods

    shot, got = flux
    observed = observed_resistance(got, I_ni=0.0, smoothing=Smoothing("none"))
    reference = compute_romero_flux_balance_ods(sample_ods(shot), R_p=0.0, I_ni=0.0,
                                                time_range=WINDOWS[shot])
    np.testing.assert_allclose(observed.R_p, reference["R_closing"], rtol=1e-10)
    assert observed.provenance["current_source_assumption"].startswith("ohmic")


def test_a_fallback_only_flux_unit_is_refused(monkeypatch):
    import vaft.omas.resistive_zeff as module

    monkeypatch.setattr(module, "detected_psi_per_radian", lambda ods, idx: None)
    with pytest.raises(ValueError, match="not a detection"):
        romero_boundary_flux_ods(sample_ods(39915), time_range=WINDOWS[39915])


def test_disagreeing_detectors_are_refused(monkeypatch):
    import vaft.omas.resistive_zeff as module

    monkeypatch.setattr(module, "detected_psi_per_radian", lambda ods, idx: True)
    with pytest.raises(ValueError, match="2 pi slip"):
        romero_boundary_flux_ods(sample_ods(39915), time_range=WINDOWS[39915])


# --- Level C / D on the 48224 kinetic sample ---------------------------------------------


@pytest.fixture(scope="module")
def state48224():
    logging.disable(logging.WARNING)
    try:
        return flux_surface_state_ods(sample_ods(48224), time_slice=0)
    finally:
        logging.disable(logging.NOTSET)


def test_the_parallel_spitzer_resistance_agrees_with_the_j_tor_reference(state48224):
    """Sec. 7.1's j_tor^2 integral and the <J.B> form differ by geometry only."""
    from vaft.omas.process_wrapper import compute_ohmic_heating_power_from_core_profiles

    power = compute_ohmic_heating_power_from_core_profiles(
        sample_ods(48224), time_slice=0, Z_eff=2.0, ln_Lambda=17.0)
    reference = power / state48224.I_p**2
    parallel = model_resistance(state48224, 2.0, model="spitzer_nrl", ln_lambda=17.0).R_p
    assert 0.8 < parallel / reference < 1.25, parallel / reference
    assert state48224.source["excluded_current_fraction"] < 0.01


def test_vest_trapped_particles_multiply_the_spitzer_resistance(state48224):
    sp = model_resistance(state48224, 2.0, model="sauter_spitzer", ln_lambda="sauter").R_p
    sauter = model_resistance(state48224, 2.0, model="sauter", ln_lambda="sauter").R_p
    redl = model_resistance(state48224, 2.0, model="redl", ln_lambda="sauter").R_p
    assert 1.5 < sauter / sp < 5.0, sauter / sp
    assert abs(redl / sauter - 1.0) < 0.2, redl / sauter


def test_neo_conductivity_matches_sauter_resistance_at_its_own_charge(state48224):
    """Level D: NEO enters as a tabulated model, at the charge its species list gives."""
    from vaft.code.gacode.neo.outputs import collect_neo_outputs
    from vaft.machine_mapping.neoclassical import core_profiles_from_neo

    fixtures = Path(__file__).parent / "data" / "gacode"
    ods = sample_ods(48224)
    native = collect_neo_outputs(fixtures / "neo_vest_48224_conductivity")
    charge = float(np.nanmean(np.asarray(native.effective_charge, dtype=float)))
    core_profiles_from_neo(ods, collect_neo_outputs(fixtures / "neo_vest_48224_profile"),
                           time=float(ods["core_profiles.profiles_1d.0.time"])
                           if "core_profiles.profiles_1d.0.time" in ods else 0.3,
                           time_index=0, conductivity=native)
    rho_cp = np.asarray(ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"], float)
    sigma_cp = np.asarray(ods["core_profiles.profiles_1d.0.conductivity_parallel"], float)
    ts = ods["equilibrium.time_slice"][state48224.source["eq_index"]]["profiles_1d"]
    psi = np.asarray(ts["psi"], float)
    psi_norm_eq = (psi - psi[0]) / (psi[-1] - psi[0])
    rho_eq = np.asarray(ts["rho_tor_norm"], float)
    finite = np.isfinite(sigma_cp)
    psi_of_rho = np.interp(rho_cp[finite], rho_eq, psi_norm_eq)
    table = TabulatedConductivity("neo", psi_of_rho, sigma_cp[finite], z_eff=charge)
    band = state48224.restricted(psi_of_rho.min(), psi_of_rho.max())
    neo = model_resistance(band, charge, model=table, ln_lambda="sauter").R_p
    sauter = model_resistance(band, charge, model="sauter", ln_lambda="sauter").R_p
    assert 0.8 < neo / sauter < 1.25, neo / sauter
