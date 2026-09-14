"""TRANSP into the kinetic container: the one conversion under vaft.code."""

from __future__ import annotations

import numpy as np
import pytest
from transp_cdf_fixtures import write_transp_cdf

from vaft.code import transp
from vaft.code.transp.outputs import TranspFormatError
from vaft.code.transp.profiles import kinetic_profiles_from_slice


@pytest.fixture()
def run(tmp_path):
    expected = write_transp_cdf(tmp_path)
    with transp.read_transp_output(expected["path"]) as output:
        yield output, expected


# --- the radial coordinate ----------------------------------------------------


def test_psi_norm_is_plflx_over_plflxa_and_the_edge_is_the_last_closed_surface(run):
    """Not (P - P[0]) / (P[-1] - P[0]), which would call the innermost zone
    boundary the magnetic axis."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    np.testing.assert_allclose(profiles.psi_norm, expected["psi_norm"], rtol=1e-5)
    assert profiles.psi_norm[-1] == pytest.approx(1.0, rel=1e-6)
    assert profiles.psi_norm[0] > 0.0
    assert profiles.normalization.method == "transp_plflx_over_plflxa"


def test_the_default_grid_is_xb_and_leaves_the_flux_untouched(run):
    """psi_norm and the E x B frequency are both XB quantities, so choosing XB
    interpolates neither -- only the smooth kinetic profiles move."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    assert len(profiles) == len(expected["xb"])
    assert "interpolated" not in profiles.provenance["psi_norm"]
    assert "interpolated" not in profiles.provenance["omega_exb"]
    assert "interpolated onto XB" in profiles.provenance["n_e"]


def test_on_xb_the_flux_quantities_reach_the_container_untouched(run):
    """Not merely unlabelled as interpolated: bit-for-bit what the file says,
    since a round trip through the other grid would change the values while
    leaving every provenance string intact."""
    output, _ = run
    state = output.slice(0.20)
    profiles = kinetic_profiles_from_slice(state)
    np.testing.assert_array_equal(profiles.omega_exb, transp.exb_frequency(state))
    np.testing.assert_array_equal(profiles.psi_norm, np.asarray(state.psi_norm_xb, dtype=float))


def test_the_x_grid_is_available_and_costs_the_edge(run):
    """Reported rather than hidden: on X the outermost point is no longer the
    last closed flux surface."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid="X")
    assert len(profiles) == len(expected["x"])
    assert profiles.psi_norm[-1] < 1.0
    assert "interpolated onto X" in profiles.provenance["psi_norm"]
    assert "interpolated" not in profiles.provenance["n_e"]


def test_which_grid_was_chosen_is_part_of_the_answer(run):
    output, _ = run
    for grid in ("XB", "X"):
        profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid=grid)
        assert f"on the {grid} grid" in profiles.provenance["target_grid"]


def test_a_third_grid_is_refused(run):
    output, _ = run
    with pytest.raises(TranspFormatError, match="must be 'X' or 'XB'"):
        kinetic_profiles_from_slice(output.slice(0.20), target_grid="rho")


def test_both_grids_are_kept_so_the_choice_can_be_undone(run):
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    np.testing.assert_allclose(profiles.extras["x"], expected["x"], rtol=1e-6)
    np.testing.assert_allclose(profiles.extras["xb"], expected["xb"], rtol=1e-6)


# --- the E x B frequency ------------------------------------------------------


def test_the_exb_frequency_is_minus_the_potential_gradient_in_flux(run):
    """VRPOT is in volts and PLFLX in webers per radian, and a volt per weber
    is an inverse second -- so the result is rad/s with nothing converted."""
    output, expected = run
    state = output.slice(0.20)
    omega = transp.exb_frequency(state)
    expected_omega = -np.gradient(
        np.asarray(state.on_xb("VRPOT"), dtype=float),
        np.asarray(state.on_xb("PLFLX"), dtype=float),
        edge_order=2,
    )
    np.testing.assert_array_equal(omega, expected_omega)
    assert len(omega) == len(expected["xb"])
    assert "rad/s" in kinetic_profiles_from_slice(state).provenance["omega_exb"]


def test_the_exb_frequency_refuses_units_it_was_not_derived_for(tmp_path):
    """The cancellation that makes this rad/s is specifically volts over
    webers per radian; a different pair needs a conversion this does not make."""
    expected = write_transp_cdf(tmp_path, units_override={"VRPOT": "KILOVOLTS"})
    with transp.read_transp_output(expected["path"]) as output:
        with pytest.raises(TranspFormatError, match="VRPOT declares 'KILOVOLTS'"):
            transp.exb_frequency(output.slice(0.20))


# --- units come from the file -------------------------------------------------


def test_the_units_are_read_from_the_file_and_converted_once(run):
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    # NE is N/CM**3 in the file and m^-3 in the container.
    on_xb = np.interp(expected["xb"], expected["x"], expected["n_e"][1])
    np.testing.assert_allclose(profiles.n_e, on_xb * 1e6, rtol=1e-6)
    assert profiles.unit("n_e") == "m^-3"
    assert "N/CM**3" in profiles.provenance["n_e"]


def test_an_unrecognised_unit_raises_rather_than_being_guessed(tmp_path):
    """A wrong guess between N/CM**3 and N/M**3 is a silent factor of a
    million, which is what a --density-unit flag on the converter this
    replaces could produce."""
    expected = write_transp_cdf(tmp_path, units_override={"NE": "PARTICLES/BARN"})
    with transp.read_transp_output(expected["path"]) as output:
        with pytest.raises(TranspFormatError, match="PARTICLES/BARN"):
            kinetic_profiles_from_slice(output.slice(0.20))


def test_a_unit_of_the_wrong_dimension_raises(tmp_path):
    expected = write_transp_cdf(tmp_path, units_override={"NE": "EV"})
    with transp.read_transp_output(expected["path"]) as output:
        with pytest.raises(TranspFormatError, match="but n_e is carried in"):
            kinetic_profiles_from_slice(output.slice(0.20))


def test_a_variable_the_run_does_not_carry_is_simply_absent(tmp_path):
    """The fixture writes no NI, so n_i is absent rather than invented."""
    expected = write_transp_cdf(tmp_path)
    with transp.read_transp_output(expected["path"]) as output:
        profiles = kinetic_profiles_from_slice(output.slice(0.20))
    assert profiles.n_i is None
    assert profiles.n_e is not None


# --- time ---------------------------------------------------------------------


def test_the_sample_actually_taken_is_recorded(run):
    """A run's samples are irregular, so which one was used is part of the
    answer rather than an implementation detail."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.24))
    assert f"t = {expected['time'][1]:.6f} s" in profiles.provenance["psi_norm"]


def test_read_transp_profiles_is_the_one_shot_form(tmp_path):
    expected = write_transp_cdf(tmp_path)
    profiles = transp.read_transp_profiles(expected["path"], time_s=0.20)
    with transp.read_transp_output(expected["path"]) as output:
        direct = kinetic_profiles_from_slice(output.slice(0.20))
    np.testing.assert_array_equal(profiles.psi_norm, direct.psi_norm)
    np.testing.assert_array_equal(profiles.n_e, direct.n_e)


def test_the_result_is_a_kinetic_profile_set_a_kin_writer_accepts(tmp_path):
    """The point of the conversion: what comes out can be written as a .kin."""
    from vaft.data import read_kin, write_kin

    expected = write_transp_cdf(tmp_path)
    profiles = transp.read_transp_profiles(expected["path"], time_s=0.20)
    # .kin needs n_i, which this run does not carry; supply it explicitly
    # rather than having the converter invent one.
    from dataclasses import replace

    written = write_kin(replace(profiles, n_i=profiles.n_e * 0.9), tmp_path / "out.kin")
    back = read_kin(written)
    np.testing.assert_allclose(back.psi_norm, profiles.psi_norm, rtol=1e-7)
    np.testing.assert_allclose(back.omega_exb, profiles.omega_exb, rtol=1e-7)
