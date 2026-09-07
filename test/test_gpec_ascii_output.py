"""GPEC's sectioned ASCII outputs: singcoup matrices and response parameters."""

from __future__ import annotations

import numpy as np
import pytest
from gpec_ascii_fixtures import (
    SINGCOUP_SECTIONS,
    write_empty_singfld,
    write_response,
    write_singcoup_matrix,
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


def test_the_toroidal_mode_number_is_not_invented_from_the_file_name(singcoup):
    """These files do not record n; nothing here should pretend otherwise."""
    output, _ = singcoup
    assert not hasattr(output, "n_tor")
    assert "n" not in output.attrs


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


def test_a_column_lookup_names_the_columns_it_has(response):
    output, _ = response
    table = output.section("Stability indices").table
    with pytest.raises(KeyError, match=r"no column 'nope'.*\['mode', 's', 'se'\]"):
        table.column("nope")


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


def test_complex_names_lists_only_complete_pairs(singcoup, response):
    output, _ = singcoup
    assert output.sections[0].tables[0].complex_names == ("C_f",)
    # A table with no real/imag pair reports none.
    energy, _ = response
    assert energy.section("Energy for dcon eigenmodes").table.complex_names == ()


def test_to_dict_gives_the_columns_in_file_order(response):
    output, expected = response
    table = output.section("Energy for dcon eigenmodes").table
    assert list(table.to_dict()) == list(table.columns)
    np.testing.assert_allclose(table.to_dict()["ev0"], expected["energy"][:, 0])


# --- nothing dropped, nothing mangled -----------------------------------------


def test_a_prose_legend_is_not_mangled_into_a_scalar(response):
    """``rho = Reluctance (power norm)`` is key=value shaped but is prose."""
    output, _ = response
    section = output.section("Eigenvalues (e) and Singular Values (s)")
    assert "rho = Reluctance (power norm)" in section.legend
    assert "P = Permeability   *Complex (not Hermitian)" in section.legend
    assert "rho" not in output.attrs
    # A clean pair on its own line is a scalar, and is kept.
    assert section.tables[0].header["jac_type"] == "hamada"


def test_real_files_leave_nothing_unparsed(singcoup, response):
    output, _ = singcoup
    assert output.unparsed == ()
    output, _ = response
    assert output.unparsed == ()


def test_an_unreadable_line_is_kept_rather_than_dropped(tmp_path):
    """A format change must surface as data, not as silence."""
    write_singcoup_matrix(tmp_path)
    path = tmp_path / "gpec_singcoup_matrix_n1.out"
    lines = path.read_text(encoding="utf-8").splitlines()
    # A row with one column too many, in the middle of a block.
    lines.insert(10, "  -1  1.0E-05  2.0E-06  3.0E-07")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = read_gpec_ascii(path)
    assert len(output.unparsed) == 1
    number, text = output.unparsed[0]
    assert number == 11
    assert "3.0E-07" in text


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
