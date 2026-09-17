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


def test_a_grid_is_resolved_by_dimension_name_not_by_length(run):
    """A real MAST run has len(X) == len(XB) == 20, and the legacy converter
    picked a variable's grid by comparing array lengths -- so every
    zone-boundary quantity came back as a zone-centre one."""
    output, _ = run
    assert output.dims["X"] == output.dims["XB"]  # nothing can be told apart by counting
    assert output.grid_of("NE") == "X"
    assert output.grid_of("PLFLX") == "XB"


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


def test_a_variable_on_a_third_radial_dimension_has_no_grid(run):
    """79 of the reference file's variables are on RMAJM, THETA and the like."""
    output, expected = run
    assert output.grid_of("BDENS") is None
    state = output.slice(0.20)
    np.testing.assert_allclose(state.variable("BDENS"), expected["beam_density"][1], rtol=1e-6)
    with pytest.raises(TranspFormatError, match="not a radial profile"):
        state.on_x("BDENS")


def test_grids_written_without_a_time_axis_are_returned_whole(tmp_path):
    """X and XB are (TIME3, X) in the runs looked at, but need not be, and
    indexing axis 0 of a plain (X,) grid would return one radial point."""
    expected = write_transp_cdf(tmp_path, static_grids=True)
    with transp.read_transp_output(expected["path"]) as output:
        state = output.slice(0.20)
        np.testing.assert_allclose(state.x, expected["x"], rtol=1e-6)
        np.testing.assert_allclose(state.xb, expected["xb"], rtol=1e-6)
        assert len(state.on_x("NE")) == len(state.x)


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


def test_a_variable_with_no_units_attribute_reads_as_empty(tmp_path):
    expected = write_transp_cdf(tmp_path, no_units_variable=True)
    with transp.read_transp_output(expected["path"]) as output:
        assert "units" not in output._open["NOUNITS"].attrs
        assert output.units("NOUNITS") == ""
        assert "no units declared" in output.describe("NOUNITS")


def test_a_catalogued_variable_is_described_by_this_adapter(run):
    output, _ = run
    assert output.describe("NE") == "electron density [cm^-3]"
    assert "TQIN" in output.describe("TQTOTNB")  # the placeholder points at its replacement


def test_an_uncatalogued_variable_is_described_from_what_the_file_says(run):
    """Q is in the file and not in VARIABLE_DESCRIPTIONS, so the file answers."""
    output, _ = run
    described = output.describe("Q")
    assert "SAFETY FACTOR" in described
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


def test_a_scalar_time_series_and_a_profile_are_sampled_at_the_same_instant(run):
    """BPHXB is on TIME and NE on TIME3; a slice must reach both at its own time."""
    output, expected = run
    state = output.slice(0.30)
    assert state.time_index == 2
    assert float(state.variable("BPHXB")) == pytest.approx(-0.8, rel=1e-6)
    np.testing.assert_allclose(state.on_x("NE"), expected["n_e"][2], rtol=1e-6)


def test_a_dimensionless_scalar_is_returned_whole(run):
    """423 of the reference file's variables carry no axis at all."""
    output, _ = run
    assert np.asarray(output.slice(0.20).variable("NLTAUP")).shape == ()


def test_the_two_time_axes_must_hold_the_same_instants(tmp_path):
    """A sample index is chosen on TIME and applied to profiles on TIME3, so a
    file where they disagree cannot say which time a profile is from."""
    expected = write_transp_cdf(tmp_path, times=(0.1, 0.2, 0.3), times3=(0.5, 0.6, 0.7))
    with pytest.raises(TranspFormatError, match="do not hold the same instants"):
        transp.read_transp_output(expected["path"])


def test_time_axes_of_different_lengths_are_refused(tmp_path):
    expected = write_transp_cdf(tmp_path, times=(0.1, 0.2, 0.3, 0.4), times3=(0.1, 0.2, 0.3))
    with pytest.raises(TranspFormatError, match="do not hold the same instants"):
        transp.read_transp_output(expected["path"])


# --- integrity ----------------------------------------------------------------


def test_the_files_own_wb_per_radian_statement_is_checked(tmp_path):
    """PLFLX2PI / PLFLX is 2*pi in a sound file; if it is not, the flux is
    not what it says and psi_norm would be silently wrong."""
    expected = write_transp_cdf(tmp_path, plflx2pi_factor=3.0)
    with pytest.raises(TranspFormatError, match=r"PLFLX2PI / PLFLX is 3\.0"):
        transp.read_transp_output(expected["path"])


def test_a_flux_that_is_entirely_zero_is_refused_rather_than_skipped(tmp_path):
    """A zeroed tail is what a truncated classic netCDF reads back as, so an
    unusable PLFLX must not simply skip the check and report the file sound."""
    expected = write_transp_cdf(tmp_path, flux_edge=(0.0, 0.0, 0.0))
    with pytest.raises(TranspFormatError, match="no non-zero finite value"):
        transp.read_transp_output(expected["path"])


def test_a_truncated_file_is_refused_rather_than_read_back_as_zeros(tmp_path):
    expected = write_transp_cdf(tmp_path)
    whole = expected["path"].read_bytes()
    expected["path"].write_bytes(whole[: len(whole) - 400])
    with pytest.raises(TranspFormatError, match="killed mid-write"):
        transp.read_transp_output(expected["path"])


def test_a_grid_that_stops_increasing_at_a_later_sample_is_refused(tmp_path):
    """The grids are declared time-varying, so checking only the first sample
    would pass a run whose tail was never written."""
    expected = write_transp_cdf(tmp_path, reverse_xb_at=2)
    with pytest.raises(TranspFormatError, match="increase outward; they do not at sample 2"):
        transp.read_transp_output(expected["path"])


def test_grids_of_different_lengths_are_refused(tmp_path):
    expected = write_transp_cdf(tmp_path, boundary_points=8)
    with pytest.raises(TranspFormatError, match="one zone centre per zone boundary"):
        transp.read_transp_output(expected["path"])


def test_a_zone_centre_outside_its_own_boundary_is_refused(tmp_path):
    """Both grids still increase, so only the interleaving catches this."""
    expected = write_transp_cdf(tmp_path, centres_outside=True)
    with pytest.raises(TranspFormatError, match="swapped or misaligned"):
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


def test_a_closed_file_says_so_rather_than_half_answering(tmp_path):
    """Whatever was read before close() would still answer out of the cache,
    and anything else would fail deep in xarray on a NoneType."""
    expected = write_transp_cdf(tmp_path)
    output = transp.read_transp_output(expected["path"])
    output.variable("NE")  # cached
    output.close()
    with pytest.raises(TranspFormatError, match="is closed"):
        output.variable("TE")
    with pytest.raises(TranspFormatError, match="is closed"):
        output.units("NE")
    output.close()  # idempotent


# --- torque -------------------------------------------------------------------


def test_tqtotnb_is_refused_by_name_and_points_at_tqin(run):
    """It is a zero-valued scalar in every run checked -- an empty placeholder --
    even though it declares torque-density units."""
    output, _ = run
    assert output.units("TQTOTNB") == "Nt-M/CM3"
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
    _, torque = transp.enclosed_torque(state)

    per_zone = expected["torque_density"][1] * expected["zone_volume"][1]
    np.testing.assert_allclose(torque, np.cumsum(per_zone), rtol=1e-6)
    assert torque[-1] == pytest.approx(per_zone.sum(), rel=1e-6)


def test_enclosed_torque_lands_on_the_zone_boundaries(run):
    """Summing zones 1..i gives the torque inside XB[i], not inside X[i]."""
    output, expected = run
    state = output.slice(0.20)
    psi_norm, torque = transp.enclosed_torque(state)
    assert len(torque) == len(state.xb) == expected["zones"]
    assert psi_norm[-1] == pytest.approx(1.0)


def test_the_input_torque_density_and_zone_volume_keep_their_own_units(run):
    output, expected = run
    state = output.slice(0.20)
    np.testing.assert_allclose(transp.input_torque_density(state), expected["torque_density"][1])
    np.testing.assert_allclose(transp.zone_volume(state), expected["zone_volume"][1])
    assert state.units("TQIN") == "Nt-M/CM3"
    assert state.units("DVOL") == "CM**3"


# --- normalized flux ----------------------------------------------------------


def test_psi_norm_is_plflx_over_the_enclosed_flux(run):
    """Not (P - P[0]) / (P[-1] - P[0]), which would call the innermost zone
    boundary the magnetic axis and move the whole grid inward."""
    output, expected = run
    state = output.slice(0.20)
    np.testing.assert_allclose(
        state.psi_norm_xb, expected["plflx"][1] / expected["plflxa"][1], rtol=1e-6
    )
    edge_axis = (expected["plflx"][1] - expected["plflx"][1][0]) / (
        expected["plflx"][1][-1] - expected["plflx"][1][0]
    )
    assert state.psi_norm_xb[0] > 0.0
    assert state.psi_norm_xb[0] != pytest.approx(edge_axis[0])


def test_psi_norm_normalizes_by_this_times_edge_flux(run):
    """PLFLXA rises through a run -- 0.0272 to 0.0765 Wb/rad in the reference
    file -- so taking its first sample regardless of time puts the last closed
    surface off the end of the grid."""
    output, expected = run
    assert expected["plflxa"][0] != pytest.approx(expected["plflxa"][-1])
    for index, time_s in enumerate(expected["time"]):
        state = output.slice(float(time_s))
        assert state.psi_norm_xb[-1] == pytest.approx(1.0, rel=1e-5)
        np.testing.assert_allclose(
            state.psi_norm_xb, expected["plflx"][index] / expected["plflxa"][index], rtol=1e-5
        )


def test_psi_norm_is_not_the_radial_coordinate(run):
    """XB is sqrt of normalized toroidal flux; psi_norm is poloidal. In the
    reference run they are 0.0049 against 0.05 at the innermost boundary."""
    output, expected = run
    state = output.slice(0.20)
    assert state.psi_norm_xb[0] != pytest.approx(state.xb[0], rel=1e-3)
    np.testing.assert_allclose(state.psi_norm_xb, expected["psi_norm"], rtol=1e-5)


# --- collection ---------------------------------------------------------------


def test_collecting_a_directory_reports_what_it_holds(tmp_path):
    expected = write_transp_cdf(tmp_path)
    result = transp.collect_transp_outputs(tmp_path)
    assert result.returncode is None  # nothing was run
    assert result.runid == expected["runid"]
    assert expected["path"] in result.outputs["cdf"]


def test_the_particle_history_sibling_is_not_taken_for_the_run(tmp_path):
    """<runid>PH.CDF is a separate product, and it is routinely written last --
    so a reader that picked the newest .CDF would report the wrong runid."""
    run = write_transp_cdf(tmp_path, runid="45453X01")
    history = tmp_path / "45453X01PH.CDF"
    history.write_bytes(run["path"].read_bytes())
    result = transp.collect_transp_outputs(tmp_path)
    assert result.runid == "45453X01"
    assert result.outputs["cdf"] == (run["path"],)
    assert result.outputs["other_cdf"] == (history,)


def test_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        transp.collect_transp_outputs(tmp_path / "nope")


def test_an_empty_directory_is_not_an_error(tmp_path):
    result = transp.collect_transp_outputs(tmp_path)
    assert result.outputs["cdf"] == () and result.runid == ""
