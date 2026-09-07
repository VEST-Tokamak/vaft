"""The ideal-GPEC profile output: resonant metrics, complex decoding, n from the file."""

from __future__ import annotations

import netCDF4
import numpy as np
import pytest
from gpec_nc_fixtures import write_control_nc, write_profile_nc

from vaft.code.gpec import (
    GpecProfileOutput,
    read_gpec_netcdf,
    read_gpec_profile_output,
    read_resonant_table,
)



@pytest.fixture()
def profile(tmp_path):
    expected = write_profile_nc(tmp_path)
    return read_gpec_profile_output(tmp_path / "gpec_profile_output_n1.nc"), expected


# --- globals ------------------------------------------------------------------


def test_reads_the_globals_including_shot_time_and_machine(profile):
    output, expected = profile
    assert isinstance(output, GpecProfileOutput)
    assert output.n_tor == 1
    assert output.machine == "VEST"
    assert output.shot == expected["shot"] == 48226
    assert output.time == float(expected["time"])
    assert output.version == "v1.5.5-test"


@pytest.mark.parametrize("helicity", [-1.0, +1.0])
def test_the_helicity_gpec_ran_with_is_read_from_the_file(tmp_path, helicity):
    """Both signs occur: the shipped recon example runs at +1."""
    write_profile_nc(tmp_path, helicity=helicity)
    output = read_gpec_profile_output(tmp_path / "gpec_profile_output_n1.nc")
    assert output.helicity == helicity


@pytest.mark.parametrize("value", ["", None])
def test_an_empty_or_absent_helicity_is_none_rather_than_a_crash(tmp_path, value):
    write_profile_nc(tmp_path)
    path = tmp_path / "gpec_profile_output_n1.nc"
    with netCDF4.Dataset(path, "a") as ds:
        if value is None:
            ds.delncattr("helicity")
        else:
            ds.helicity = value
    assert read_gpec_profile_output(path).helicity is None


def test_a_machine_without_a_name_reads_as_empty(tmp_path):
    """The shipped recon example carries machine=''."""
    write_profile_nc(tmp_path, machine="")
    assert read_gpec_profile_output(tmp_path / "gpec_profile_output_n1.nc").machine == ""


# --- decoding -----------------------------------------------------------------


def test_complex_variables_decode_from_the_leading_i_dimension(profile):
    output, expected = profile
    np.testing.assert_allclose(output.Phi_res, expected["Phi_res"])
    np.testing.assert_allclose(output.I_res, expected["I_res"])
    assert np.iscomplexobj(output.Phi_res) and np.iscomplexobj(output.I_res)
    assert not np.iscomplexobj(output.w_isl)


def test_nothing_is_conjugated_on_the_way_in(profile):
    """The container is a transcript; the raw/conjugate question is separate.

    Covers the real-space ``*_fun`` pair as well as the spectral one, since
    those are the quantities GPEC itself treats differently by helicity.
    """
    output, expected = profile
    np.testing.assert_allclose(output.b_n, expected["b_n"])
    np.testing.assert_allclose(output.b_n_fun, expected["b_n_fun"])
    np.testing.assert_allclose(output.xi_n, expected["b_n"] * 10.0)
    np.testing.assert_allclose(output.xi_n_fun, expected["b_n_fun"] * 10.0)
    assert output.b_n.imag.any() and output.b_n_fun.imag.any()


def test_arrays_keep_their_native_shape_and_dtype(profile):
    """Native dimension order, and integer mode numbers stay integers."""
    output, expected = profile
    m_count, psi_count = expected["m_out"].size, expected["psi_n"].size
    theta_count = expected["theta"].size
    assert output.b_n.shape == (m_count, psi_count)
    assert output.b_n_fun.shape == (theta_count, psi_count)
    assert output.R.shape == (theta_count, psi_count)
    assert output.dims["b_n"] == ("m_out", "psi_n")
    assert output.dims["b_n_fun"] == ("theta_dcon", "psi_n")
    # An index array must stay usable as an index.
    assert np.issubdtype(output.m_out.dtype, np.integer)
    assert output.b_n[np.searchsorted(output.m_out, -1)].shape == (psi_count,)


def test_character_matrices_come_back_as_strings(profile):
    output, expected = profile
    assert output.extras["coil_name"] == expected["coil_names"]


def test_units_are_copied_where_the_file_has_them(profile):
    output, _ = profile
    assert output.units["Phi_res"] == "T"  # flux normalized by surface area
    assert output.units["I_res"] == "A"
    assert output.units["w_isl"] == "psi_n"
    # GPEC leaves many variables without units; the mapping is partial.
    assert "K_isl" not in output.units
    assert output.units.get("K_isl") is None


# --- rational surfaces --------------------------------------------------------


def test_resonant_poloidal_mode_number_is_n_times_q(tmp_path):
    write_profile_nc(tmp_path, n=3, rational_q=(1.0, 4.0 / 3.0))
    output = read_gpec_profile_output(tmp_path / "gpec_profile_output_n3.nc")
    assert output.n_tor == 3
    np.testing.assert_allclose(output.m_rational, [3.0, 4.0])
    assert output.n_rational == 2


def test_resonant_table_selects_by_dimension_not_by_length(profile):
    """``coil_index`` has as many entries as there are rational surfaces here.

    Matching on length would fold it, and the ``coil_name`` strings beside
    it, into a table of physical quantities.
    """
    output, expected = profile
    table = output.resonant_table()
    assert len(expected["coil_names"]) == output.n_rational  # the collision
    assert "coil_index" not in table and "coil_name" not in table
    assert {"psi_n_rational", "m_rational", "Phi_res", "I_res", "w_isl", "K_isl"} <= set(table)
    assert "T_e_rational" in table  # a rational-dimensioned extra belongs
    assert {value.shape for value in table.values()} == {expected["q_rational"].shape}


def test_the_cheap_reader_returns_the_same_table(profile, tmp_path):
    output, _ = profile
    cheap = read_resonant_table(tmp_path / "gpec_profile_output_n1.nc")
    assert set(cheap) == set(output.resonant_table())
    for name, values in cheap.items():
        np.testing.assert_allclose(values, output.resonant_table()[name])


# --- robustness ---------------------------------------------------------------


def test_every_variable_in_the_file_is_accounted_for(profile, tmp_path):
    """The preservation invariant: named or extra, nothing is dropped."""
    output, _ = profile
    with netCDF4.Dataset(tmp_path / "gpec_profile_output_n1.nc") as ds:
        in_file = set(ds.variables) - {"i"}
    assert in_file == set(output.dims)
    for name in in_file:
        value = getattr(output, name, None) if hasattr(output, name) else output.extras.get(name)
        assert value is not None, name


def test_a_named_variable_in_an_unexpected_shape_is_still_read(tmp_path):
    """Decoding follows the file's own dimensions, not a static list.

    A ``Phi_res`` written without the ``i`` axis (or a ``w_isl`` written with
    it) must not vanish, and must not keep an undecoded ``(2, N)`` layout.
    """
    write_profile_nc(tmp_path)
    path = tmp_path / "gpec_profile_output_n1.nc"
    import xarray as xr

    with xr.open_dataset(path) as ds:
        edited = ds.copy(deep=True)
    edited["Phi_res"] = (("psi_n_rational",), np.real(edited["Phi_res"].values[0]))
    edited["w_isl"] = (("i", "psi_n_rational"), np.stack([edited["w_isl"].values] * 2))
    edited.to_netcdf(tmp_path / "edited.nc")

    output = read_gpec_profile_output(tmp_path / "edited.nc")
    assert output.Phi_res is not None and not np.iscomplexobj(output.Phi_res)
    assert output.Phi_res.shape == (2,)
    assert np.iscomplexobj(output.w_isl) and output.w_isl.shape == (2,)
    assert {value.shape for value in output.resonant_table().values()} == {(2,)}


def test_a_trimmed_extract_still_reads(tmp_path):
    """A committed extract keeps psi_n-dimensioned variables and drops the rest."""
    write_profile_nc(tmp_path, trimmed=True)
    output = read_gpec_profile_output(tmp_path / "gpec_profile_output_n1.nc")
    assert output.n_rational == 2
    assert output.Phi_res is not None
    assert output.psi_n is not None and output.q is not None
    assert output.b_n is None and output.R is None
    assert output.m_rational is not None


def test_n_comes_from_the_attribute_not_the_filename(tmp_path):
    """C-08: the legacy readers fell back to n=1 when parsing failed."""
    write_profile_nc(tmp_path, n=2)
    renamed = tmp_path / "gpec_profile_output_n7.nc"
    (tmp_path / "gpec_profile_output_n2.nc").rename(renamed)
    with pytest.warns(UserWarning):
        output = read_gpec_profile_output(renamed)
    assert output.n_tor == 2


# --- wiring into the run result ------------------------------------------------


def test_the_result_exposes_the_profile_lazily(tmp_path):
    write_control_nc(tmp_path)
    write_profile_nc(tmp_path)
    result = read_gpec_netcdf(tmp_path)
    assert result.source_paths["profile"].endswith("gpec_profile_output_n1.nc")
    # Reading the run directory must not have loaded it.
    assert "profile" not in result.__dict__
    assert result.profile.n_tor == result.control.n_tor == 1
    assert result.profile is result.profile  # cached after first access


def test_an_unreadable_profile_leaves_the_control_output_usable(tmp_path):
    """A run killed mid-write must not cost the caller its control data."""
    write_control_nc(tmp_path)
    (tmp_path / "gpec_profile_output_n1.nc").write_bytes(b"truncated")
    result = read_gpec_netcdf(tmp_path)
    assert result.control.n_tor == 1
    with pytest.raises(Exception):
        _ = result.profile


def test_a_mode_mismatch_between_control_and_profile_is_refused(tmp_path):
    write_control_nc(tmp_path, n=1)
    write_profile_nc(tmp_path, n=2)
    (tmp_path / "gpec_profile_output_n2.nc").rename(tmp_path / "gpec_profile_output_n1.nc")
    result = read_gpec_netcdf(tmp_path)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="profile file is n=2"):
        _ = result.profile


def test_an_explicit_profile_path_is_used(tmp_path):
    write_control_nc(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    write_profile_nc(elsewhere)
    from vaft.code.gpec import GpecIdealResult

    result = GpecIdealResult.from_netcdf(
        tmp_path, profile_path=elsewhere / "gpec_profile_output_n1.nc"
    )
    assert result.profile is not None
    assert result.source_paths["profile"] == str(elsewhere / "gpec_profile_output_n1.nc")


def test_a_run_without_a_profile_file_still_reads(tmp_path):
    write_control_nc(tmp_path)
    result = read_gpec_netcdf(tmp_path)
    assert result.profile is None
    assert "profile" not in result.source_paths


def test_the_json_transcript_carries_the_resonant_table(tmp_path):
    write_control_nc(tmp_path)
    write_profile_nc(tmp_path)
    transcript = read_gpec_netcdf(tmp_path).to_dict()
    assert "resonant_table" in transcript
    assert "Phi_res" in transcript["resonant_table"]
    assert "m_rational" in transcript["resonant_table"]
