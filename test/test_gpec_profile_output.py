"""The ideal-GPEC profile output: resonant metrics, complex decoding, n from the file."""

from __future__ import annotations

import numpy as np
import pytest
from gpec_nc_fixtures import write_control_nc, write_profile_nc

from vaft.code.gpec import GpecProfileOutput, read_gpec_netcdf, read_gpec_profile_output


@pytest.fixture()
def profile(tmp_path):
    expected = write_profile_nc(tmp_path)
    return read_gpec_profile_output(tmp_path / "gpec_profile_output_n1.nc"), expected


def test_reads_globals_including_the_helicity_gpec_ran_with(profile):
    output, expected = profile
    assert isinstance(output, GpecProfileOutput)
    assert output.n_tor == 1
    assert output.machine == "VEST"
    assert output.helicity == expected["helicity"] == -1.0
    assert output.version == "v1.5.5-test"


def test_complex_variables_decode_from_the_leading_i_dimension(profile):
    output, expected = profile
    np.testing.assert_allclose(output.Phi_res, expected["Phi_res"])
    np.testing.assert_allclose(output.I_res, expected["I_res"])
    assert np.iscomplexobj(output.Phi_res) and np.iscomplexobj(output.I_res)
    # Real-valued metrics stay real.
    assert not np.iscomplexobj(output.w_isl)


def test_resonant_poloidal_mode_number_is_n_times_q(profile):
    output, expected = profile
    np.testing.assert_allclose(output.m_rational, 1 * expected["q_rational"])
    assert output.n_rational == expected["q_rational"].size


def test_resonant_mode_number_scales_with_the_toroidal_mode(tmp_path):
    write_profile_nc(tmp_path, n=3, rational_q=(1.0, 4.0 / 3.0))
    output = read_gpec_profile_output(tmp_path / "gpec_profile_output_n3.nc")
    assert output.n_tor == 3
    np.testing.assert_allclose(output.m_rational, [3.0, 4.0])


def test_units_are_preserved_verbatim(profile):
    output, _ = profile
    assert output.units["Phi_res"] == "T"  # flux normalized by surface area
    assert output.units["I_res"] == "A"
    assert output.units["w_isl"] == "psi_n"


def test_unnamed_variables_are_kept_as_decoded_extras(profile):
    output, _ = profile
    assert "b_eul" in output.extras
    assert np.iscomplexobj(output.extras["b_eul"])
    np.testing.assert_allclose(output.extras["b_eul"], output.b_n * 2.0)
    # A real extra stays real.
    assert not np.iscomplexobj(output.extras["T_e_rational"])


def test_resonant_table_is_aligned_and_includes_extras(profile):
    output, expected = profile
    table = output.resonant_table()
    assert {"psi_n_rational", "m_rational", "Phi_res", "I_res", "w_isl", "K_isl"} <= set(table)
    assert "T_e_rational" in table  # rational-shaped extra
    assert {value.shape for value in table.values()} == {expected["q_rational"].shape}


def test_a_trimmed_extract_still_reads(tmp_path):
    """A committed regression extract carries the rational block only."""
    write_profile_nc(tmp_path, trimmed=True)
    output = read_gpec_profile_output(tmp_path / "gpec_profile_output_n1.nc")
    assert output.n_rational == 2
    assert output.Phi_res is not None
    assert output.b_n is None and output.psi_n is None
    assert output.m_rational is not None


def test_n_comes_from_the_attribute_not_the_filename(tmp_path):
    """C-08: the legacy readers fell back to n=1 when parsing failed."""
    write_profile_nc(tmp_path, n=2)
    renamed = tmp_path / "gpec_profile_output_n7.nc"
    (tmp_path / "gpec_profile_output_n2.nc").rename(renamed)
    with pytest.warns(UserWarning):
        output = read_gpec_profile_output(renamed)
    assert output.n_tor == 2


def test_the_result_loader_picks_the_profile_file_up(tmp_path):
    write_control_nc(tmp_path)
    write_profile_nc(tmp_path)
    result = read_gpec_netcdf(tmp_path)
    assert result.profile is not None
    assert result.profile.n_tor == result.control.n_tor == 1
    assert result.source_paths["profile"].endswith("gpec_profile_output_n1.nc")


def test_a_mode_mismatch_between_control_and_profile_is_refused(tmp_path):
    write_control_nc(tmp_path, n=1)
    write_profile_nc(tmp_path, n=1)
    write_profile_nc(tmp_path, n=2)
    (tmp_path / "gpec_profile_output_n1.nc").unlink()
    (tmp_path / "gpec_profile_output_n2.nc").rename(tmp_path / "gpec_profile_output_n1.nc")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="profile file is n=2"):
        read_gpec_netcdf(tmp_path)


def test_a_run_without_a_profile_file_still_reads(tmp_path):
    write_control_nc(tmp_path)
    result = read_gpec_netcdf(tmp_path)
    assert result.profile is None
    assert "profile" not in result.source_paths
