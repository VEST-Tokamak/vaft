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


def test_psi_norm_divides_by_plflxa_and_not_by_the_last_flux_point(tmp_path):
    """In a real run PLFLXA and PLFLX[-1] are equal, so which one is read is
    invisible there. This fixture breaks that identity on purpose -- it is
    unphysical, and it is the only way to see the difference."""
    expected = write_transp_cdf(tmp_path, plflxa_scale=2.0)
    with transp.read_transp_output(expected["path"]) as output:
        profiles = kinetic_profiles_from_slice(output.slice(0.20))
    np.testing.assert_allclose(profiles.psi_norm, expected["psi_norm"] / 2.0, rtol=1e-5)
    assert profiles.psi_norm[-1] == pytest.approx(0.5, rel=1e-5)


def test_the_endpoints_are_recorded_as_they_came_out(run):
    """Not as they ought to look. On X the edge is not 1.0, and a record
    saying otherwise would let a caller "renormalise" to what it already
    has."""
    output, _ = run
    for grid in ("XB", "X"):
        profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid=grid)
        assert profiles.normalization.axis_value == pytest.approx(profiles.psi_norm[0])
        assert profiles.normalization.edge_value == pytest.approx(profiles.psi_norm[-1])
    assert kinetic_profiles_from_slice(
        output.slice(0.20), target_grid="X"
    ).normalization.edge_value < 1.0


def test_the_source_file_travels_with_the_result(run):
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    assert profiles.source == str(expected["path"])


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
    # Which end matters: going XB -> X the clamp is at the *axis*, where a
    # constant is a poor stand-in for a quantity varying quadratically.
    assert "1 clamped at the axis end" in profiles.provenance["psi_norm"]


def test_going_the_other_way_clamps_the_edge_instead(run):
    output, _ = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    assert "1 clamped at the edge" in profiles.provenance["n_e"]
    assert "axis end" not in profiles.provenance["n_e"]


def test_which_grid_was_chosen_is_part_of_the_answer(run):
    output, _ = run
    for grid in ("XB", "X"):
        profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid=grid)
        assert f"on the {grid} grid" in profiles.provenance["target_grid"]


def test_a_third_grid_is_refused(run):
    output, _ = run
    with pytest.raises(TranspFormatError, match="must be 'X' or 'XB'"):
        kinetic_profiles_from_slice(output.slice(0.20), target_grid="rho")


def test_both_grids_are_recorded_without_being_passed_off_as_profiles(run):
    """They were in `extras` once, and a MARS writer duly emitted PROFx.IN and
    PROFxb.IN: `extras` means "a quantity this container has no field for",
    and a radial grid is not a quantity."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20))
    assert profiles.extras == {}
    note = profiles.provenance["target_grid"]
    assert f"{expected['x'][0]:.6g}" in note and f"{expected['xb'][-1]:.6g}" in note


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


def test_a_flux_that_stalls_is_refused_rather_than_producing_nans(tmp_path):
    """np.gradient divides by the spacing, so a repeated PLFLX value puts NaN
    into the middle of omega_exb with nothing but a numpy warning."""
    expected = write_transp_cdf(tmp_path, repeat_flux_at=3)
    with transp.read_transp_output(expected["path"]) as output:
        with pytest.raises(TranspFormatError, match="PLFLX does not increase"):
            transp.exb_frequency(output.slice(0.20))


def test_the_exb_guard_covers_the_flux_as_well_as_the_potential(tmp_path):
    expected = write_transp_cdf(tmp_path, units_override={"PLFLX": "WEBERS"})
    with transp.read_transp_output(expected["path"]) as output:
        with pytest.raises(TranspFormatError, match="PLFLX declares 'WEBERS'"):
            transp.exb_frequency(output.slice(0.20))


def test_a_run_without_the_potential_is_refused_by_name(tmp_path):
    """A bare KeyError from three layers down is not an answer."""
    import xarray as xr

    expected = write_transp_cdf(tmp_path)
    with xr.open_dataset(expected["path"], engine="scipy", decode_times=False) as ds:
        trimmed = ds.drop_vars("VRPOT").load()
    expected["path"].unlink()
    trimmed.to_netcdf(expected["path"], format="NETCDF3_CLASSIC")
    with transp.read_transp_output(expected["path"]) as output:
        with pytest.raises(TranspFormatError, match="carries no VRPOT"):
            kinetic_profiles_from_slice(output.slice(0.20))


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
    """A run with no rotation at all yields no omega_tor, rather than one
    invented from whatever else is lying about."""
    expected = write_transp_cdf(tmp_path, rotation_variables=())
    with transp.read_transp_output(expected["path"]) as output:
        profiles = kinetic_profiles_from_slice(output.slice(0.20))
    assert profiles.omega_tor is None
    assert profiles.n_e is not None and profiles.omega_exb is not None


def test_every_mapped_field_is_converted_and_named(run):
    """Not just n_e: a source swap between TE and TI, or a missing ladder
    rung, is invisible if only one field is ever read."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid="X")
    for field, source, values, factor in (
        ("n_e", "NE", expected["n_e"], 1e6),
        ("n_i", "NI", expected["n_i"], 1e6),
        ("T_e", "TE", expected["T_e"], 1.0),
        ("T_i", "TI", expected["T_i"], 1.0),
    ):
        np.testing.assert_allclose(
            getattr(profiles, field), values[1] * factor, rtol=1e-6
        )
        assert source in profiles.provenance[field]
    # TE and TI are different shapes in the fixture on purpose, so a swapped
    # source is a wrong profile rather than a scale error.
    assert not np.allclose(profiles.T_e / profiles.T_e[0], profiles.T_i / profiles.T_i[0])


def test_temperatures_in_kilo_electronvolts_are_converted(tmp_path):
    expected = write_transp_cdf(tmp_path, temperature_units="KEV")
    with transp.read_transp_output(expected["path"]) as output:
        profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid="X")
    np.testing.assert_allclose(profiles.T_e, expected["T_e"][1] * 1e3, rtol=1e-6)
    assert "KEV" in profiles.provenance["T_e"]


def test_the_measured_rotation_is_preferred_and_the_choice_is_named(run):
    """A run can carry several toroidal angular velocities and they are not
    the same quantity -- in the reference run every point of OMEG_VTR differs
    from OMEGA by more than a tenth of the peak."""
    output, expected = run
    profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid="X")
    np.testing.assert_allclose(
        profiles.omega_tor, expected["rotation"]["OMEG_VTR"][1], rtol=1e-6
    )
    assert "OMEG_VTR" in profiles.provenance["omega_tor"]
    assert "chosen over OMEGA" in profiles.provenance["omega_tor"]


def test_the_next_rotation_is_used_when_the_preferred_one_is_absent(tmp_path):
    expected = write_transp_cdf(tmp_path, rotation_variables=("OMEGA",))
    with transp.read_transp_output(expected["path"]) as output:
        profiles = kinetic_profiles_from_slice(output.slice(0.20), target_grid="X")
    np.testing.assert_allclose(
        profiles.omega_tor, expected["rotation"]["OMEGA"][1], rtol=1e-6
    )
    assert "OMEGA" in profiles.provenance["omega_tor"]


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
