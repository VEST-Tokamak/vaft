"""The native TRANSP transcript: grids, units, time, and the torque quantities."""

from __future__ import annotations

import numpy as np
import pytest
from transp_cdf_fixtures import UNITS, write_transp_cdf

from vaft.code import transp
from vaft.code.transp.outputs import TranspFormatError


@pytest.fixture()
def run(tmp_path):
    expected = write_transp_cdf(tmp_path)
    with transp.read_transp_output(expected["path"]) as output:
        yield output, expected


# --- grids: resolved by dimension name, never by length -----------------------


def test_the_two_radial_grids_are_the_same_length(run):
    """Which is exactly why nothing may resolve a grid by counting.

    A real MAST run has len(X) == len(XB) == 20, and the legacy converter
    picked a variable's grid by comparing array lengths -- so every
    zone-boundary quantity came back as a zone-centre one.
    """
    output, expected = run
    assert len(expected["x"]) == len(expected["xb"])
    assert output.grid_of("NE") == "X"
    assert output.grid_of("PLFLX") == "XB"
    assert np.all(expected["x"] < expected["xb"])  # interleaved


def test_a_zone_centre_variable_is_refused_on_the_boundary_grid(run):
    output, _ = run
    state = output.slice(0.20)
    with pytest.raises(TranspFormatError, match="NE is on the X grid"):
        state.on_xb("NE")
    with pytest.raises(TranspFormatError, match="PLFLX is on the XB grid"):
        state.on_x("PLFLX")


def test_the_refusal_says_that_interpolating_is_the_callers_choice(run):
    output, _ = run
    with pytest.raises(TranspFormatError, match="explicitly"):
        output.slice(0.20).on_xb("NE")


def test_a_variable_that_is_not_a_profile_has_no_grid(run):
    output, _ = run
    assert output.grid_of("PLFLXA") is None
    with pytest.raises(TranspFormatError, match="not a radial profile"):
        output.slice(0.20).on_x("PLFLXA")


# --- units: from the file, never from an argument -----------------------------


def test_units_come_from_the_variables_own_attribute(run):
    output, _ = run
    assert output.units("NE") == UNITS["NE"] == "N/CM**3"
    assert output.units("TQIN") == "Nt-M/CM3"
    assert output.units("PLFLX") == "Wb/rad"


def test_nothing_is_converted_on_the_way_in(run):
    """This layer is a transcript: NE is still per cubic centimetre."""
    output, expected = run
    state = output.slice(0.20)
    np.testing.assert_allclose(state.on_x("NE"), expected["n_e"][1], rtol=1e-6)
    assert state.on_x("NE").max() > 1e12  # cm^-3, not m^-3


def test_a_variable_with_no_units_attribute_reads_as_empty(run):
    output, _ = run
    assert output.units("X") == ""


def test_a_catalogued_variable_is_described_by_this_adapter(run):
    output, _ = run
    assert output.describe("NE") == "electron density [cm^-3]"
    assert "TQIN" in output.describe("TQTOTNB")  # the placeholder points at its replacement


def test_an_uncatalogued_variable_is_described_from_what_the_file_says(run):
    """Q is in the file and not in VARIABLE_DESCRIPTIONS, so the file answers."""
    output, _ = run
    described = output.describe("Q")
    assert "SAFETY FACTOR" in described
    assert "no units declared" in described  # Q carries an empty units attribute
    assert "TRANSP's own description" in described


def test_describing_a_variable_the_file_lacks_does_not_raise(run):
    output, _ = run
    assert "not in" in output.describe("NOT_A_TRANSP_VARIABLE")


# --- time ---------------------------------------------------------------------


def test_the_chosen_sample_is_part_of_the_answer(run):
    """Samples are irregular, so which one was taken has to travel with it."""
    output, expected = run
    state = output.slice(0.24)
    assert state.time_index == 1
    assert state.time_s == pytest.approx(expected["time"][1])


def test_both_time_dimensions_are_accepted(run):
    output, _ = run
    assert set(transp.TIME_DIMENSIONS) == {"TIME", "TIME3"}
    assert "TIME" in output.dims and "TIME3" in output.dims
    np.testing.assert_allclose(output.time, output.variable("TIME3").values)


def test_a_variable_without_a_time_axis_is_returned_whole(run):
    output, expected = run
    value = output.slice(0.20).variable("PLFLXA")
    assert np.asarray(value).shape == ()


# --- integrity ----------------------------------------------------------------


def test_the_files_own_wb_per_radian_statement_is_checked(tmp_path):
    """PLFLX2PI / PLFLX is 2*pi in a sound file; if it is not, the flux is
    not what it says and psi_norm would be silently wrong."""
    expected = write_transp_cdf(tmp_path, plflx2pi_factor=3.0)
    with pytest.raises(TranspFormatError, match=r"PLFLX2PI / PLFLX is 3\.0"):
        transp.read_transp_output(expected["path"])


def test_a_file_that_is_not_transp_output_is_refused(tmp_path):
    import xarray as xr

    path = tmp_path / "notatransp.CDF"
    xr.Dataset({"foo": (("bar",), np.arange(3.0))}).to_netcdf(path, format="NETCDF3_CLASSIC")
    with pytest.raises(TranspFormatError, match="does not look like"):
        transp.read_transp_output(path)


def test_a_run_directory_is_refused_with_a_useful_message(tmp_path):
    with pytest.raises(IsADirectoryError, match="collect_transp_outputs"):
        transp.read_transp_output(tmp_path)


# --- torque -------------------------------------------------------------------


def test_tqtotnb_is_refused_by_name_and_points_at_tqin(run):
    """It is a dimensionless zero in every run checked -- an empty placeholder."""
    output, _ = run
    with pytest.raises(TranspFormatError, match="empty placeholder"):
        output.variable("TQTOTNB")
    with pytest.raises(TranspFormatError, match="Use TQIN"):
        output.variable("TQTOTNB")


def test_enclosed_torque_pairs_the_two_cm_based_quantities_once(run):
    """TQIN is N m / cm^3 and DVOL is cm^3, so the product is already N m.

    Converting one of them to SI and not the other is a factor of a million.
    """
    output, expected = run
    state = output.slice(0.20)
    psi_norm, torque = transp.enclosed_torque(state)

    per_zone = expected["torque_density"][1] * expected["zone_volume"][1]
    np.testing.assert_allclose(torque, np.cumsum(per_zone), rtol=1e-6)
    assert torque[-1] == pytest.approx(per_zone.sum(), rel=1e-6)


def test_enclosed_torque_lands_on_the_zone_boundaries(run):
    """Summing zones 1..i gives the torque inside XB[i], not inside X[i]."""
    output, expected = run
    state = output.slice(0.20)
    psi_norm, torque = transp.enclosed_torque(state)
    assert len(torque) == len(expected["xb"]) == len(state.xb)
    np.testing.assert_allclose(psi_norm, expected["xb"], rtol=1e-6)
    assert psi_norm[-1] == pytest.approx(1.0)


def test_the_input_torque_density_keeps_its_own_units(run):
    output, expected = run
    state = output.slice(0.20)
    np.testing.assert_allclose(transp.input_torque_density(state), expected["torque_density"][1])
    assert state.units("TQIN") == "Nt-M/CM3"


# --- normalized flux ----------------------------------------------------------


def test_psi_norm_is_plflx_over_the_enclosed_flux(run):
    """Not (P - P[0]) / (P[-1] - P[0]), which would call the innermost zone
    boundary the magnetic axis and move the whole grid inward."""
    output, expected = run
    state = output.slice(0.20)
    np.testing.assert_allclose(state.psi_norm_xb, expected["plflx"][1] / expected["plflxa"][1], rtol=1e-6)
    assert state.psi_norm_xb[0] > 0.0
    edge_axis = (expected["plflx"][1] - expected["plflx"][1][0]) / (
        expected["plflx"][1][-1] - expected["plflx"][1][0]
    )
    assert edge_axis[0] == 0.0
    assert state.psi_norm_xb[0] != pytest.approx(edge_axis[0])


# --- collection ---------------------------------------------------------------


def test_collecting_a_directory_reports_what_it_holds(tmp_path):
    expected = write_transp_cdf(tmp_path)
    result = transp.collect_transp_outputs(tmp_path)
    assert result.returncode is None  # nothing was run
    assert result.runid == expected["runid"]
    assert expected["path"] in result.outputs["cdf"]


def test_collecting_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        transp.collect_transp_outputs(tmp_path / "nope")


def test_an_empty_directory_is_not_an_error(tmp_path):
    result = transp.collect_transp_outputs(tmp_path)
    assert result.outputs["cdf"] == () and result.runid == ""
