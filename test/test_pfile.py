"""The Osborne pfile: the transcript, its round trip, and the unit ladder."""

from __future__ import annotations

import numpy as np
import pytest
from pfile_fixtures import SECTIONS, write_pfile_file

from vaft.data.pfile import (
    PFILE_SECTION_ORDER,
    PFILE_UNITS,
    PFILE_TO_CONTAINER,
    PFile,
    PFileFormatError,
    PFileSection,
    kinetic_profiles_from_pfile,
    read_pfile,
    write_pfile,
)

#: The first six lines of a pfile, written out here rather than taken from
#: the module: the species header, the `%.6f` species rows, the section
#: header shape and the `%.8e` data rows are the format, and a test that
#: imported them could only compare the writer with itself.
EXPECTED_OPENING = [
    "3 N Z A of ION SPECIES",
    " 6.000000   6.000000   12.010700",
    " 1.000000   1.000000   2.000000",
    " 1.000000   1.000000   2.000000",
    "41 psinorm ne(10^20/m^3) dne/dpsiN",
    " 0.00000000e+00   4.00000000e-01   -3.25000000e-01",
]


@pytest.fixture()
def run(tmp_path):
    expected = write_pfile_file(tmp_path)
    return read_pfile(expected["path"]), expected


# --- the transcript -----------------------------------------------------------


def test_every_section_is_read_in_file_order_with_its_own_unit(tmp_path):
    """File order, not the format's order: a file written backwards has to
    come back backwards, or nothing downstream can tell the two apart."""
    expected = write_pfile_file(tmp_path, scrambled=True)
    pfile = read_pfile(expected["path"])
    assert pfile.keys() == tuple(expected["keys"])
    assert pfile.keys() == tuple(reversed(PFILE_SECTION_ORDER))
    assert pfile.unit("ne") == "10^20/m^3"
    assert pfile.unit("te") == "KeV"
    assert pfile.unit("kpol") == "km/s/T"


def test_writing_puts_the_sections_back_into_the_formats_order(tmp_path):
    """However they arrived. A consumer reads a pfile top to bottom, so the
    order is part of the format rather than of the particular file."""
    expected = write_pfile_file(tmp_path, scrambled=True)
    written = read_pfile(write_pfile(read_pfile(expected["path"]), tmp_path / "out"))
    assert written.keys() == PFILE_SECTION_ORDER


def test_the_optional_pressure_block_lands_after_ptot(tmp_path):
    expected = write_pfile_file(tmp_path, pplas=True)
    written = read_pfile(write_pfile(read_pfile(expected["path"]), tmp_path / "out"))
    keys = list(written.keys())
    assert keys[keys.index("ptot") + 1] == "pplas"


def test_the_optional_pressure_block_is_not_written_first_when_ptot_is_absent(tmp_path):
    """`pplas` belongs after `ptot`; with no `ptot` to follow it goes last.
    At the front it would put a pressure block above the densities."""
    expected = write_pfile_file(tmp_path, only=("ne", "te"), pplas=True)
    written = read_pfile(write_pfile(read_pfile(expected["path"]), tmp_path / "out"))
    assert written.keys() == ("ne", "te", "pplas")


def test_an_empty_unit_is_an_empty_unit_and_not_a_missing_one(run):
    """omghb really is written `omghb()` in every reference file."""
    pfile, _ = run
    assert pfile.unit("omghb") == ""
    assert pfile.section("omghb").label == "Hahm-Burrell ExB shearing rate"


def test_the_derivative_column_is_kept_as_written(run):
    """It is data. Whatever produced the reference files did not use the
    stencil the previous writer recomputed with, so a reader that dropped
    this column would make the file unreproducible."""
    pfile, expected = run
    section = pfile.section("te")
    np.testing.assert_allclose(section.derivative, expected["derivatives"]["te"], rtol=1e-8)
    recomputed = np.gradient(section.values, section.psi_norm, edge_order=2)
    assert not np.allclose(section.derivative, recomputed, rtol=1e-3)


def test_the_species_block_becomes_species(run):
    pfile, expected = run
    assert [(s.n, s.z, s.a) for s in pfile.species] == [tuple(r) for r in expected["species"]]
    # The block carries no labels, so position is what names a row.
    assert [s.label for s in pfile.species] == ["impurity", "main ion", "fast ion"]


def test_a_section_this_module_does_not_catalogue_is_kept(tmp_path):
    expected = write_pfile_file(tmp_path, unknown_section=True)
    pfile = read_pfile(expected["path"])
    assert "zeff" in pfile.keys()
    assert pfile.unit("zeff") == "-"
    assert pfile.section("zeff").label == ""  # not catalogued, still present


def test_the_arrays_are_read_only(run):
    """A profile set that says it is what the file said has to be."""
    pfile, _ = run
    section = pfile.section("ne")
    assert not section.values.flags.writeable
    assert not section.psi_norm.flags.writeable
    assert not section.derivative.flags.writeable
    with pytest.raises(ValueError):
        section.values[0] = 0.0


def test_a_section_key_is_kept_as_written(tmp_path):
    """Not normalised. Lowercasing `NE` would quietly map it onto `n_e`,
    turning a section this module has never seen into one it thinks it
    understands."""
    path = tmp_path / "p000000.00000"
    path.write_text(
        "3 psinorm NE(10^20/m^3) dNE/dpsiN\n"
        " 0.00000000e+00   1.00000000e+00   0.00000000e+00\n"
        " 5.00000000e-01   2.00000000e+00   0.00000000e+00\n"
        " 1.00000000e+00   3.00000000e+00   0.00000000e+00\n",
        encoding="utf-8",
    )
    pfile = read_pfile(path)
    assert pfile.keys() == ("NE",)
    assert "NE" in kinetic_profiles_from_pfile(pfile).extras


def test_a_grid_other_than_201_points_reads(tmp_path):
    """The reader this replaces tested `line.startswith('201 psinorm')`, so
    every other grid was silently unparseable."""
    expected = write_pfile_file(tmp_path, points=137)
    pfile = read_pfile(expected["path"])
    assert len(pfile) == 137
    assert len(pfile.section("ne").values) == 137


def test_asking_for_a_section_that_is_absent_names_what_is_there(run):
    pfile, _ = run
    with pytest.raises(KeyError, match="no section 'zeff'"):
        pfile.section("zeff")


# --- what the file must not be allowed to say ---------------------------------


def test_sections_on_different_coordinates_are_refused(tmp_path):
    """All 22 sections share one psinorm column, bit for bit, in all 57
    reference files. Taking the first and assuming the rest would put each
    profile on a coordinate that is not its own."""
    expected = write_pfile_file(tmp_path, mismatched_psi_in="ti")
    with pytest.raises(PFileFormatError, match="section ti is on a different psinorm"):
        read_pfile(expected["path"])


def test_a_section_shorter_than_it_declares_is_refused(tmp_path):
    expected = write_pfile_file(tmp_path, only=("ne", "te"), short_section="te")
    with pytest.raises(PFileFormatError, match="declares 42 rows but the file ends"):
        read_pfile(expected["path"])


def test_a_ragged_row_is_named_rather_than_skipped(tmp_path):
    """The readers this replaces skipped a short row and carried on, so a
    damaged file became a shorter profile with no diagnostic."""
    expected = write_pfile_file(tmp_path, ragged_row=3)
    with pytest.raises(PFileFormatError, match=r"section ne row 4 has 2 columns"):
        read_pfile(expected["path"])


def test_a_row_with_too_many_columns_is_refused(tmp_path):
    """Not merely too few: an extra column means the file is not what its
    header says, and truncating it silently would discard data."""
    expected = write_pfile_file(tmp_path, extra_column_in_row=2)
    with pytest.raises(PFileFormatError, match=r"section ne row 3 has 4 columns"):
        read_pfile(expected["path"])


def test_a_line_that_is_not_a_section_header_is_not_read_as_one(tmp_path):
    """Any line beginning with an integer used to become a section, so a
    trailing note above three numbers parsed into a profile named `of`."""
    expected = write_pfile_file(tmp_path, only=("ne",), footer="3 columns of something else")
    pfile = read_pfile(expected["path"])
    assert pfile.keys() == ("ne",)


def test_a_row_of_numbers_outside_any_block_is_refused(tmp_path):
    """Which is what a species count short of the rows beneath it leaves --
    skipping it is how a species, or a profile's tail, disappears."""
    expected = write_pfile_file(tmp_path, only=("ne",))
    text = expected["path"].read_text(encoding="utf-8").replace(
        "3 N Z A of ION SPECIES", "2 N Z A of ION SPECIES", 1
    )
    expected["path"].write_text(text, encoding="utf-8")
    with pytest.raises(PFileFormatError, match="outside any block"):
        read_pfile(expected["path"])


def test_a_section_declaring_no_rows_is_refused(tmp_path):
    path = tmp_path / "empty_section"
    path.write_text("0 psinorm ne(10^20/m^3) dne/dpsiN\n", encoding="utf-8")
    with pytest.raises(PFileFormatError, match="declares 0 rows"):
        read_pfile(path)


def test_a_file_with_no_sections_is_refused(tmp_path):
    path = tmp_path / "notapfile"
    path.write_text("some notes\nand nothing numeric\n", encoding="utf-8")
    with pytest.raises(PFileFormatError, match="holds no profile sections"):
        read_pfile(path)


def test_a_species_block_of_another_length_is_numbered_rather_than_guessed(tmp_path):
    """The three-row convention is impurity, main ion, fast ion. The writer
    this reads after omits the impurity row when a run has none, so a
    two-row block is main ion and fast ion -- and labelling deuterium as the
    impurity would be worse than not labelling it."""
    expected = write_pfile_file(tmp_path, species_rows=2)
    pfile = read_pfile(expected["path"])
    assert [s.label for s in pfile.species] == ["species 1", "species 2"]
    assert pfile.species[0].a == pytest.approx(12.0107)


def test_a_file_with_no_species_block_reads_and_writes(tmp_path):
    expected = write_pfile_file(tmp_path, species=False, only=("ne", "te"))
    pfile = read_pfile(expected["path"])
    assert pfile.species == ()
    out = write_pfile(pfile, tmp_path / "again")
    assert out.read_text(encoding="utf-8") == expected["path"].read_text(encoding="utf-8")


def test_a_section_on_its_own_coordinate_is_refused_when_built_in_memory(tmp_path):
    """read_pfile checks this; so must the container. The writer pairs the
    file's coordinate with each section's values, so a section on its own
    grid would be written out reprojected onto one that is not its own."""
    psi = np.linspace(0.0, 1.0, 5)
    ok = PFileSection(key="ne", unit="10^20/m^3", psi_norm=psi, values=np.arange(5.0))
    shifted = PFileSection(key="te", unit="KeV", psi_norm=psi + 0.25, values=np.arange(5.0))
    with pytest.raises(PFileFormatError, match="different radial coordinate"):
        PFile(psi_norm=psi, sections=(ok, shifted))


def test_one_quantity_may_not_appear_twice(tmp_path):
    section = PFileSection(key="ne", unit="10^20/m^3", psi_norm=np.linspace(0, 1, 4),
                           values=np.ones(4))
    with pytest.raises(PFileFormatError, match="appears twice"):
        PFile(psi_norm=np.linspace(0, 1, 4), sections=(section, section))


# --- writing ------------------------------------------------------------------


def test_a_pfile_round_trips_byte_for_byte(run, tmp_path):
    pfile, expected = run
    out = write_pfile(pfile, tmp_path / "again")
    assert out.read_text(encoding="utf-8") == expected["path"].read_text(encoding="utf-8")


def test_the_written_format_is_the_one_the_reference_files_carry(run, tmp_path):
    pfile, _ = run
    text = write_pfile(pfile, tmp_path / "out").read_text(encoding="utf-8")
    assert text.splitlines()[:6] == EXPECTED_OPENING


def test_recomputing_the_derivatives_changes_the_file(run, tmp_path):
    """Which is exactly why preserving them is the default: the previous
    writer recomputed every column and so reproduced none of the 57
    reference files."""
    pfile, expected = run
    kept = write_pfile(pfile, tmp_path / "kept").read_text(encoding="utf-8")
    fresh = write_pfile(pfile, tmp_path / "fresh", recompute_derivatives=True).read_text(
        encoding="utf-8"
    )
    assert kept == expected["path"].read_text(encoding="utf-8")
    assert fresh != kept


def test_a_section_built_without_a_derivative_gets_one(tmp_path):
    psi = np.linspace(0.0, 1.0, 9)
    values = psi**2
    pfile = PFile(
        psi_norm=psi,
        sections=(PFileSection(key="ne", unit="10^20/m^3", psi_norm=psi, values=values),),
    )
    written = read_pfile(write_pfile(pfile, tmp_path / "made"))
    np.testing.assert_allclose(
        written.section("ne").derivative,
        np.gradient(values, psi, edge_order=2),
        rtol=1e-7,
    )


def test_writing_refuses_a_coordinate_that_does_not_increase(tmp_path):
    expected = write_pfile_file(tmp_path, descending_psi=True, only=("ne",))
    pfile = read_pfile(expected["path"])
    with pytest.raises(PFileFormatError, match="psinorm does not increase"):
        write_pfile(pfile, tmp_path / "backwards")


def test_writing_refuses_a_derivative_that_is_not_finite(tmp_path):
    """Both columns are ones consumers spline."""
    psi = np.linspace(0.0, 1.0, 5)
    section = PFileSection(
        key="ne", unit="10^20/m^3", psi_norm=psi, values=np.ones(5),
        derivative=np.array([np.nan, 1.0, 2.0, 3.0, 4.0]),
    )
    with pytest.raises(PFileFormatError, match="derivatives that are not finite"):
        write_pfile(PFile(psi_norm=psi, sections=(section,)), tmp_path / "nan")


def test_writing_refuses_a_value_that_is_not_finite(tmp_path):
    psi = np.linspace(0.0, 1.0, 5)
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    pfile = PFile(
        psi_norm=psi,
        sections=(PFileSection(key="ne", unit="10^20/m^3", psi_norm=psi, values=values),),
    )
    with pytest.raises(PFileFormatError, match="not finite"):
        write_pfile(pfile, tmp_path / "nan")


def test_an_uncatalogued_section_is_written_after_the_known_ones(tmp_path):
    """Even one that arrived in the middle of the file."""
    expected = write_pfile_file(tmp_path, unknown_section_in_the_middle=True)
    pfile = read_pfile(expected["path"])
    assert pfile.keys()[2] == "zeff"  # where the file put it
    written = read_pfile(write_pfile(pfile, tmp_path / "moved"))
    assert written.keys()[-1] == "zeff"


def test_an_uncatalogued_section_appended_to_a_file_survives(tmp_path):
    expected = write_pfile_file(tmp_path, unknown_section=True)
    pfile = read_pfile(expected["path"])
    written = read_pfile(write_pfile(pfile, tmp_path / "again"))
    assert written.keys()[-1] == "zeff"
    assert written.keys()[: len(PFILE_SECTION_ORDER)] == PFILE_SECTION_ORDER


# --- the conversion, and C-27 -------------------------------------------------


def test_the_three_rotation_sections_stay_three_quantities(run):
    """The regression for C-27.

    The reader this replaces chose a section by asking whether "omeg" was a
    *substring* of the header. That matches `omeg` and `omegp` alike, and the
    later section overwrote the earlier, so what survived under an E x B name
    was `omegp` -- the poloidal contribution -- while `omgeb`, the actual
    E x B rotation, matched nothing and was dropped. Three quantities, one
    standing in for another.
    """
    pfile, expected = run
    profiles = kinetic_profiles_from_pfile(pfile)

    np.testing.assert_allclose(profiles.omega_tor, expected["values"]["omeg"] * 1e3)
    np.testing.assert_allclose(profiles.omega_exb, expected["values"]["omgeb"] * 1e3)
    np.testing.assert_allclose(profiles.omega_pol, expected["values"]["omegp"] * 1e3)

    for a, b in (("omega_tor", "omega_exb"), ("omega_tor", "omega_pol"),
                 ("omega_exb", "omega_pol")):
        assert not np.array_equal(getattr(profiles, a), getattr(profiles, b)), (a, b)


def test_the_mapping_is_by_exact_key_not_by_substring():
    """`omeg` is a prefix of `omegp`; `omgeb` contains neither."""
    assert PFILE_TO_CONTAINER["omeg"] == "omega_tor"
    assert PFILE_TO_CONTAINER["omegp"] == "omega_pol"
    assert PFILE_TO_CONTAINER["omgeb"] == "omega_exb"
    assert len(set(PFILE_TO_CONTAINER.values())) == len(PFILE_TO_CONTAINER)


def test_the_unit_ladder_converts_each_mapped_field(run):
    pfile, expected = run
    profiles = kinetic_profiles_from_pfile(pfile)
    np.testing.assert_allclose(profiles.n_e, expected["values"]["ne"] * 1e20)
    np.testing.assert_allclose(profiles.T_e, expected["values"]["te"] * 1e3)
    np.testing.assert_allclose(profiles.p_total, expected["values"]["ptot"] * 1e3)
    np.testing.assert_allclose(profiles.e_radial, expected["values"]["er"] * 1e3)
    assert profiles.unit("n_e") == "m^-3"
    assert profiles.unit("T_e") == "eV"


def test_the_unmapped_rotation_family_is_kept_in_its_own_units(run):
    pfile, expected = run
    profiles = kinetic_profiles_from_pfile(pfile)
    assert set(profiles.extras) == {
        "omgvb", "omgpp", "ommvb", "ommpp", "omevb", "omepp", "kpol", "omghb",
        "vtor1", "vpol1",
    }
    np.testing.assert_allclose(profiles.extras["omgpp"], expected["values"]["omgpp"])
    assert "kRad/s" in profiles.provenance["omgpp"]
    assert "not converted" in profiles.provenance["omgpp"]


def test_an_unconvertible_unit_on_a_mapped_section_raises(tmp_path):
    """Guessing a factor is how a density ends up wrong by a million."""
    expected = write_pfile_file(tmp_path, unknown_unit_in="ne")
    pfile = read_pfile(expected["path"])
    with pytest.raises(PFileFormatError, match="furlongs/fortnight"):
        kinetic_profiles_from_pfile(pfile)


def test_an_unconvertible_unit_on_an_unmapped_section_does_not(tmp_path):
    expected = write_pfile_file(tmp_path, unknown_unit_in="kpol")
    profiles = kinetic_profiles_from_pfile(read_pfile(expected["path"]))
    np.testing.assert_allclose(profiles.extras["kpol"], expected["values"]["kpol"])


def test_the_radial_coordinate_is_not_rescaled_on_read(run):
    pfile, expected = run
    profiles = kinetic_profiles_from_pfile(pfile)
    assert profiles.normalization.method == "as_read"
    np.testing.assert_allclose(profiles.psi_norm, expected["psi_norm"])


def test_the_species_block_reaches_the_container(run):
    pfile, _ = run
    profiles = kinetic_profiles_from_pfile(pfile)
    assert [s.label for s in profiles.species] == ["impurity", "main ion", "fast ion"]
    assert profiles.species[0].a == pytest.approx(12.0107)


def test_provenance_names_the_pfile_section_and_the_conversion(run):
    pfile, _ = run
    profiles = kinetic_profiles_from_pfile(pfile)
    assert profiles.provenance["omega_exb"].endswith("section omgeb [kRad/s] -> rad/s")
    assert profiles.provenance["n_e"].endswith("section ne [10^20/m^3] -> m^-3")


def test_the_remaining_mapped_fields_convert(run):
    pfile, expected = run
    profiles = kinetic_profiles_from_pfile(pfile)
    for section, field, factor in (
        ("ni", "n_i", 1e20), ("nz1", "n_z", 1e20), ("nb", "n_fast", 1e20),
        ("ti", "T_i", 1e3), ("pb", "p_fast", 1e3),
    ):
        np.testing.assert_allclose(
            getattr(profiles, field), expected["values"][section] * factor
        )
    assert profiles.T_z is None  # a pfile carries no impurity temperature


def test_the_module_and_the_fixture_agree_about_the_format():
    """The fixture is the format written out independently; if the two drift,
    one of them has stopped describing the reference files."""
    assert tuple(key for key, _ in SECTIONS) == PFILE_SECTION_ORDER
    assert {key: unit for key, unit in SECTIONS} == {
        key: PFILE_UNITS[key] for key, _ in SECTIONS
    }
