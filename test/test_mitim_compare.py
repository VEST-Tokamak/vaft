"""The VAFT <-> MITIM coordinate bridge and the TGLF local-input comparison (#1588 A2).

Pure functions and a stub MITIM driver; no MITIM, no GACODE, no solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code import mitim


class _Profile:
    rmin = np.linspace(0.0, 0.3, 31)
    rho = np.sqrt(np.linspace(0.0, 1.0, 31))   # nonlinear on purpose


def test_the_bridge_round_trips_and_is_not_the_identity():
    roa = np.array([0.3, 0.5, 0.8])
    rho = mitim.rho_tor_norm_at(_Profile(), roa)
    assert not np.allclose(rho, roa)
    assert mitim.r_over_a_at(_Profile(), rho) == pytest.approx(roa, abs=2e-3)


def test_the_bridge_refuses_to_extrapolate_or_guess():
    with pytest.raises(ValueError, match="outside"):
        mitim.rho_tor_norm_at(_Profile(), [1.2])

    class Flat:
        rmin = np.zeros(5)
        rho = np.linspace(0, 1, 5)

    with pytest.raises(ValueError, match="rmin/rho"):
        mitim.rho_tor_norm_at(Flat(), [0.5])


def test_input_tglf_is_parsed_with_fortran_logicals_and_comments(tmp_path):
    path = tmp_path / "input.tglf"
    path.write_text("# MITIM header\nSAT_RULE = 3\nUSE_BPER=.true.\nUNITS = 'CGYRO'\n"
                    "RLTS_1 = 1.5D0  # comment\n\nQ_LOC=3.8\n")
    parsed = mitim.read_input_tglf(path)
    assert parsed == {"SAT_RULE": 3, "USE_BPER": True, "UNITS": "CGYRO", "RLTS_1": 1.5, "Q_LOC": 3.8}


def test_the_comparison_separates_physics_from_controls_and_defaults_from_disagreement():
    vaft = {"Q_LOC": 3.8, "RLTS_1": 1.0, "SAT_RULE": 3, "USE_BPER": False, "XNU_MODEL": 2, "AS_2": 0.8}
    other = {"Q_LOC": 3.8001, "RLTS_1": 1.2, "SAT_RULE": 3, "USE_BPER": False, "UNITS": "CGYRO",
             "AS_2": 0.8}
    rows = {row["key"]: row for row in mitim.compare_tglf_inputs(vaft, other, rtol=1e-3, effective=False)}
    assert rows["Q_LOC"]["status"] == "agree" and rows["Q_LOC"]["kind"] == "physics"
    assert rows["RLTS_1"]["status"] == "differ" and rows["RLTS_1"]["rel_diff"] == pytest.approx(0.2 / 1.2)
    assert rows["XNU_MODEL"]["status"] == "vaft_only" and rows["XNU_MODEL"]["kind"] == "control"
    assert rows["UNITS"]["status"] == "mitim_only"
    assert rows["SAT_RULE"]["status"] == rows["USE_BPER"]["status"] == "agree"


def test_controls_tglf_overwrites_at_start_up_are_not_differences():
    """tglf_startup.f90 (USE_PRESETS hard-coded true) for SAT_RULE 3."""
    vaft = {"SAT_RULE": 3, "XNU_MODEL": 2, "WDIA_TRAPPED": 0.0, "UNITS": "GYRO", "NKY": 12,
            "GEOMETRY_FLAG": 1, "Q_SA": 2.0}
    other = {"SAT_RULE": 3, "XNU_MODEL": 3, "WDIA_TRAPPED": 1.0, "UNITS": "CGYRO", "NKY": 19,
             "GEOMETRY_FLAG": 1}
    rows = {row["key"]: row["status"] for row in mitim.compare_tglf_inputs(vaft, other)}
    assert rows["XNU_MODEL"] == rows["WDIA_TRAPPED"] == rows["UNITS"] == "agree"
    assert rows["NKY"] == "differ"        # a real difference survives
    assert "Q_SA" not in rows             # s-alpha inputs are unused with GEOMETRY_FLAG=1
    effective = mitim.effective_tglf_controls({"SAT_RULE": 0, "NMODES": 3, "UNITS": "CGYRO",
                                               "USE_BPER": True, "ALPHA_MACH": 1.0})
    assert effective["NMODES"] == 4 and effective["UNITS"] == "GYRO" and effective["ALPHA_MACH"] == 0.0


def test_local_neo_fluxes_are_rescaled_to_profile_mode_normalisation():
    """Profile mode normalises by species 1 (n_1, T_1); MITIM's local mode by n_e, T_ref."""
    local = {"particle_flux": np.array([[2.0e-4], [-4.0e-5], [3.0e-5]]),
             "energy_flux": np.array([[-8.0e-5], [-1.2e-4], [4.0e-5]]), "r_over_a": np.array([0.4])}
    profile = mitim.neo_local_to_profile_normalisation(local, {"DENS_1": 0.8, "TEMP_1": 1.0})
    assert profile["particle_flux"] == pytest.approx(local["particle_flux"] / 0.8)
    assert profile["energy_flux"] == pytest.approx(local["energy_flux"] / 0.8)
    hot = mitim.neo_local_to_profile_normalisation(local, {"DENS_1": 1.0, "TEMP_1": 4.0})
    assert hot["particle_flux"] == pytest.approx(local["particle_flux"] / 2.0)
    assert hot["energy_flux"] == pytest.approx(local["energy_flux"] / 8.0)


def test_neo_fluxes_are_compared_per_radius_species_and_channel():
    vaft = {"r_over_a": np.array([0.4, 0.5]),
            "particle_flux": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "energy_flux": np.array([[5.0, 6.0], [7.0, 8.0]])}
    mitim_runs = {0.5: {"particle_flux": np.array([2.0, 4.0004]), "energy_flux": np.array([6.0, 9.0])},
                  0.6: {"particle_flux": np.array([1.0, 1.0]), "energy_flux": np.array([1.0, 1.0])}}
    rows = mitim.compare_neo_fluxes(vaft, mitim_runs, rtol=1e-3)
    at = {(r["r_over_a"], r["channel"], r["species"]): r["status"] for r in rows}
    assert at[(0.5, "particle_flux", 1)] == at[(0.5, "particle_flux", 2)] == "agree"
    assert at[(0.5, "energy_flux", 2)] == "differ"      # 9 vs 8
    assert at[(0.6, "particle_flux", None)] == "vaft_missing"


def test_the_conversion_returns_only_what_it_converts_and_checks_the_mass():
    local = {"particle_flux": np.array([1.0]), "energy_flux": np.array([1.0]), "r_over_a": np.array([0.4]),
             "momentum_flux": np.array([9.0]), "bootstrap_current": np.array([9.0])}
    out = mitim.neo_local_to_profile_normalisation(local, {"DENS_1": 0.8, "TEMP_1": 1.0, "MASS_1": 0.50397},
                                                   reference_mass_1=0.50397)
    assert set(out) == {"r_over_a", "particle_flux", "energy_flux"}
    with pytest.raises(ValueError, match="deuterium"):
        mitim.neo_local_to_profile_normalisation(local, {"DENS_1": 0.8, "TEMP_1": 1.0, "MASS_1": 1.0},
                                                 reference_mass_1=0.50397)


def test_a_species_mismatch_is_reported_not_truncated():
    vaft = {"r_over_a": np.array([0.5]), "particle_flux": np.array([[1.0], [2.0], [3.0]]),
            "energy_flux": np.array([[1.0], [2.0], [3.0]])}
    short = {0.5: {"particle_flux": np.array([1.0, 2.0]), "energy_flux": np.array([1.0, 2.0])}}
    assert {r["status"] for r in mitim.compare_neo_fluxes(vaft, short)} == {"species_mismatch"}
    same = {0.5: {"particle_flux": np.array([1.0, 2.0, 3.0]), "energy_flux": np.array([1.0, 2.0, 3.0])}}
    swapped = mitim.compare_neo_fluxes(vaft, same, vaft_charges=[1, 6, -1], mitim_charges={0.5: [6, 1, -1]})
    assert {r["status"] for r in swapped} == {"species_mismatch"}
    assert {r["status"] for r in mitim.compare_neo_fluxes(vaft, same, vaft_charges=[1, 6, -1],
                                                          mitim_charges={0.5: [1, 6, -1]})} == {"agree"}


def test_local_neo_charges_come_from_its_input():
    assert mitim.neo_input_charges({"N_SPECIES": 3, "Z_1": 1.0, "Z_2": 6.0, "Z_3": -1.0}) == [1.0, 6.0, -1.0]
    vaft = {"r_over_a": np.array([0.5]), "particle_flux": np.array([[1.0]]), "energy_flux": np.array([[1.0]])}
    with pytest.raises(ValueError, match="neo_input_charges"):
        mitim.compare_neo_fluxes(vaft, {0.5: {"particle_flux": np.array([1.0]), "energy_flux": np.array([1.0])}},
                                 vaft_charges=[1.0], mitim_charges={0.5: None})


def _loop_result(true, start, recovered, residual, *, target=(1.0, 2.0), model=(1.0, 2.0)):
    return {"r_over_a": [0.3, 0.6], "aLte_true": list(true), "aLte_start": list(start),
            "aLte_recovered": list(recovered), "model_minus_required_at_start_MWm2": list(residual),
            "target_after_manufacture_MWm2": list(target), "transport_after_manufacture_MWm2": list(model),
            "final_model_MWm2": [1.0, 2.0], "final_required_MWm2": [1.0, 2.0]}


def test_the_closed_loop_report_measures_recovery_and_manufacture():
    report = mitim.closed_loop_report(
        _loop_result([1.0, 2.0], [1.3, 2.6], [1.01, 2.0], [0.5, 0.7], target=(1.0, 2.02)), 0.3)
    assert report.recovery_error == pytest.approx(0.01 / 1.01)
    assert report.manufacture_error == pytest.approx(0.02 / 2.02)
    assert report.final_flux_error == 0.0 and report.sign_ok


def test_the_residual_sign_follows_the_gradient_offset_even_for_a_hollow_profile():
    """R = Q_model - Q_required takes the sign of a/L_Te(start) - a/L_Te(true).

    Inside a hollow profile a/L_Te < 0: a 30 % "steeper" start is more negative there,
    so R < 0 inside and > 0 outside, and the convention still holds.
    """
    hollow = mitim.closed_loop_report(
        _loop_result([-1.0, 2.0], [-1.3, 2.6], [-1.0, 2.0], [-0.2, 0.5]), 0.3)
    assert hollow.sign_ok
    flipped = mitim.closed_loop_report(
        _loop_result([-1.0, 2.0], [-1.3, 2.6], [-1.0, 2.0], [0.2, 0.5]), 0.3)
    assert not flipped.sign_ok
