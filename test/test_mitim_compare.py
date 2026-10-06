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
