"""GPEC's sectioned ASCII outputs: singcoup matrices, response parameters and relatives."""

from __future__ import annotations

import numpy as np
import pytest
from gpec_ascii_fixtures import (
    RESPONSE_GLOSSARY,
    SINGCOUP_SECTIONS,
    write_control,
    write_empty_singfld,
    write_recon,
    write_response,
    write_singcoup_matrix,
    write_singcoup_svd,
    write_singfld,
)

from vaft.code.gpec import (
    GpecAsciiOutput,
    GpecAsciiTable,
    read_gpec_ascii,
    read_gpec_response,
    read_gpec_singcoup,
)


@pytest.fixture()
def singcoup(tmp_path):
    expected = write_singcoup_matrix(tmp_path)
    return read_gpec_singcoup(tmp_path / "gpec_singcoup_matrix_n1.out"), expected


@pytest.fixture()
def response(tmp_path):
    expected = write_response(tmp_path)
    return read_gpec_response(tmp_path / "gpec_response_n1.out"), expected


# --- file level ---------------------------------------------------------------


def test_reads_the_kind_version_and_file_scalars(singcoup):
    output, _ = singcoup
    assert isinstance(output, GpecAsciiOutput)
    assert output.kind == "GPEC_SINGCOUP_MATRIX"
    assert output.description.startswith("Coupling matrices")
    assert output.version == "v1.5.5-test"
    assert output.attrs["jac_out"] == "boozer"
    assert output.attrs["tmag_out"] == 1
    assert isinstance(output.attrs["psilim"], float)
    assert isinstance(output.attrs["mpert"], int)


def test_free_text_above_the_version_is_kept_as_notes(tmp_path):
    """``gpec_singcoup_svd`` explains its half-area weighting there."""
    expected = write_singcoup_svd(tmp_path)
    output = read_gpec_singcoup(tmp_path / "gpec_singcoup_svd_n1.out")
    assert output.notes == expected["notes"]
    assert output.kind == "GPEC_SINGCOUP_SVD"


def test_a_scalar_whose_key_contains_a_space_is_still_a_scalar(tmp_path):
    """``gpec_control`` writes ``vacuum energy =  3.28697546E+000``."""
    expected = write_control(tmp_path)
    output = read_gpec_ascii(tmp_path / "gpec_control_n1.out")
    for key, value in expected["scalars"].items():
        assert output.attrs[key] == pytest.approx(value)
    assert output.attrs["mpert"] == expected["m"].size


def test_a_named_reader_refuses_the_wrong_kind_of_file(tmp_path):
    write_response(tmp_path)
    with pytest.raises(ValueError, match="is a GPEC_RESPONSE file, not GPEC_SINGCOUP"):
        read_gpec_singcoup(tmp_path / "gpec_response_n1.out")


def test_a_run_directory_is_refused_with_a_useful_message(tmp_path):
    with pytest.raises(IsADirectoryError, match="read_gpec_netcdf"):
        read_gpec_ascii(tmp_path)


# --- structure ----------------------------------------------------------------


def test_every_coupling_matrix_keeps_its_own_identity(singcoup):
    """The legacy reader collapsed all five into one flat list of surfaces.

    A caller could then not tell an effective-resonant-field coupling from an
    island-width one, which is the loss this container exists to prevent.
    """
    output, _ = singcoup
    assert output.titles == tuple(title for title, _ in SINGCOUP_SECTIONS)
    symbols = {section.tables[0].columns[1] for section in output.sections}
    assert symbols == {f"real({symbol})" for _, symbol in SINGCOUP_SECTIONS}


def test_gpecs_own_typo_in_a_section_title_is_preserved(singcoup):
    output, _ = singcoup
    assert "The coupling matrix to to penetrated resonant fields" in output.titles


def test_one_block_per_rational_surface_carrying_its_own_scalars(singcoup):
    """Real files satisfy blocks-per-section == msing and rows-per-block == mpert."""
    output, expected = singcoup
    for section in output.sections:
        assert len(section.tables) == output.attrs["msing"] == len(expected["rational_q"])
        for table, q in zip(section.tables, expected["rational_q"]):
            assert table.header["q"] == q
            assert isinstance(table.header["psi"], float)
            assert len(table) == output.attrs["mpert"] == expected["m"].size


def test_a_section_title_that_repeats_is_not_collapsed(response):
    """``gpec_response`` writes two different "Eigenvectors" sections."""
    output, _ = response
    assert output.titles.count("Eigenvectors") == 2
    first, second = output.sections_named("Eigenvectors")
    assert first.tables[0].columns != second.tables[0].columns
    assert "real(K_x^L)" in first.tables[0].columns
    assert "real(V_L)" in second.tables[0].columns


def test_an_untitled_block_is_not_filed_under_the_next_sections_title(tmp_path):
    """``gpec_singfld`` writes the resonant table before any title.

    Merging it into the overlap section would hand a caller asking for
    overlap data the resonant table instead.
    """
    expected = write_singfld(tmp_path)
    output = read_gpec_ascii(tmp_path / "gpec_singfld_n1.out")
    assert output.titles == ("", "Overlap fields, overlap singular currents, and overlap islands")
    resonant = output.sections[0].table
    assert resonant.columns[:2] == ("q", "psi")
    np.testing.assert_allclose(resonant.data, expected["resonant"])
    assert output.sections[1].table.columns[0] == "mode"


def test_looking_up_a_missing_section_says_what_is_there(singcoup):
    output, _ = singcoup
    with pytest.raises(KeyError, match="no section 'Nope'"):
        output.section("Nope")


def test_the_single_block_shortcut_refuses_a_multi_block_section(singcoup, response):
    output, _ = singcoup
    with pytest.raises(ValueError, match="holds 2 blocks"):
        _ = output.sections[0].table
    energy, _ = response
    assert isinstance(energy.section("Energy for dcon eigenmodes").table, GpecAsciiTable)


# --- data ---------------------------------------------------------------------


def test_columns_and_rows_are_transcribed_in_file_order(response):
    output, expected = response
    table = output.section("Energy for dcon eigenmodes").table
    assert table.columns == ("mode", "ev0", "ev1", "iv1")
    np.testing.assert_allclose(table.column("mode"), expected["modes"])
    np.testing.assert_allclose(table.data[:, 1:], expected["energy"])
    assert table.data.shape == (expected["modes"].size, 4)


def test_a_fortran_number_without_its_exponent_marker_is_read(response):
    """A real run wrote a denormal as ``5.84973725-321``; ``float`` rejects it.

    Rejecting the row instead would silently promote it to a column header and
    split the section in two, which is what happened before.
    """
    output, expected = response
    table = output.section("Eigenvalues (e) and Singular Values (s)").table
    assert len(table) == expected["modes"].size
    assert table.columns == ("mode", "e_L", "e_rho")
    assert table.column("e_rho")[-1] == pytest.approx(expected["denormal"], rel=1e-6)


def test_a_column_lookup_names_the_columns_it_has(response):
    output, _ = response
    table = output.section("Energy for dcon eigenmodes").table
    with pytest.raises(KeyError, match=r"no column 'nope'.*\['mode', 'ev0'"):
        table.column("nope")


def test_a_repeated_column_name_is_refused_rather_than_guessed(tmp_path):
    """``gpec_singfld``'s overlap block writes ``overlap(%)`` once per matrix."""
    expected = write_singfld(tmp_path)
    table = read_gpec_ascii(tmp_path / "gpec_singfld_n1.out").sections[1].table
    assert table.columns.count("overlap(%)") == expected["overlap_matrices"]

    with pytest.raises(KeyError, match=r"appears 3 times in this block \(at \[3, 6, 9\]\)"):
        table.column("overlap(%)")
    np.testing.assert_allclose(table.column("overlap(%)", occurrence=1), expected["overlap"][:, 6])
    with pytest.raises(KeyError, match="no occurrence 9"):
        table.column("overlap(%)", occurrence=9)
    with pytest.raises(ValueError, match=r"columns \['overlap\(%\)'\] appear more than once"):
        table.to_dict()


def test_complex_columns_compose_the_stored_pair_and_nothing_more(singcoup):
    """No conjugation, no phase convention: real + 1j*imag, exactly."""
    output, expected = singcoup
    for section in output.sections:
        symbol = section.tables[0].columns[1][len("real(") : -1]
        for table, q in zip(section.tables, expected["rational_q"]):
            composed = table.complex_column(symbol)
            np.testing.assert_allclose(composed.real, table.column(f"real({symbol})"))
            np.testing.assert_allclose(composed.imag, table.column(f"imag({symbol})"))
            np.testing.assert_allclose(composed, expected["blocks"][(section.title, q)])


def test_complex_names_lists_only_complete_pairs(response):
    """The second Eigenvectors block has a ``real(V_L)`` with no ``imag`` partner."""
    output, _ = response
    first, second = output.sections_named("Eigenvectors")
    assert first.table.complex_names == ("K_x^L",)
    assert "real(V_L)" in second.table.columns
    assert second.table.complex_names == ()
    with pytest.raises(KeyError, match="no column 'imag\\(V_L\\)'"):
        second.table.complex_column("V_L")


def test_to_dict_gives_the_columns_in_file_order(response):
    output, expected = response
    table = output.section("Energy for dcon eigenmodes").table
    assert list(table.to_dict()) == list(table.columns)
    np.testing.assert_allclose(table.to_dict()["ev0"], expected["energy"][:, 0])


# --- prose, comments, and nothing dropped -------------------------------------


def test_a_glossary_is_not_read_as_scalars(response):
    """``Lambda = Inductance`` is shaped exactly like ``jac_type = hamada``.

    Neither becomes a scalar: the whole family is kept verbatim as the
    section's legend, so no guess is made about which is which.
    """
    output, _ = response
    section = output.section("Eigenvalues (e) and Singular Values (s)")
    assert section.legend == tuple(line.strip() for line in RESPONSE_GLOSSARY)
    assert section.tables[0].header == {}
    for key in ("Lambda", "L", "rho", "P", "jac_type"):
        assert key not in output.attrs


def test_a_versionless_file_with_fortran_comments_still_reads(tmp_path):
    """``gpec_recon_*`` has no version line and comments between header and rows."""
    expected = write_recon(tmp_path)
    output = read_gpec_ascii(tmp_path / "gpec_recon_integration_sol1.out")
    assert output.kind == ""
    table = output.sections[0].table
    assert table.columns == ("psi", "c2_mu0", "k_xin2_re")
    assert len(table) == expected["rows"]
    for comment in expected["comments"]:
        assert comment in output.sections[0].legend


def test_a_row_that_does_not_fit_the_block_is_kept_verbatim(tmp_path):
    """The trailing summary rows of ``gpec_recon_integration`` are the real case.

    Padding or truncating them into the block's width would invent data.
    """
    write_recon(tmp_path)
    output = read_gpec_ascii(tmp_path / "gpec_recon_integration_sol1.out")
    assert len(output.unparsed) == 1
    number, text = output.unparsed[0]
    assert number == len(
        (tmp_path / "gpec_recon_integration_sol1.out").read_text(encoding="utf-8").splitlines()
    )
    assert len(text.split()) == 2  # two values where the block has three columns


def test_data_with_no_column_header_above_it_is_kept(tmp_path):
    write_singcoup_matrix(tmp_path)
    path = tmp_path / "gpec_singcoup_matrix_n1.out"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(6, "  1.0E-05  2.0E-06")  # before any section starts
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = read_gpec_ascii(path)
    assert output.unparsed == ((7, "  1.0E-05  2.0E-06"),)


def test_fixtures_leave_nothing_unparsed(singcoup, response):
    output, _ = singcoup
    assert output.unparsed == ()
    output, _ = response
    assert output.unparsed == ()


def test_a_block_with_no_rows_keeps_its_columns(tmp_path):
    """A run with msing = 0 writes a column header and nothing under it."""
    expected = write_empty_singfld(tmp_path)
    output = read_gpec_ascii(tmp_path / "gpec_vsingfld_n1.out")
    assert output.attrs["msing"] == 0
    table = output.sections[0].table
    assert table.columns == expected["columns"]
    assert len(table) == 0
    assert table.data.shape == (0, len(expected["columns"]))
    assert output.unparsed == ()


def test_the_source_path_is_recorded(singcoup, tmp_path):
    output, _ = singcoup
    assert output.path == str(tmp_path / "gpec_singcoup_matrix_n1.out")
