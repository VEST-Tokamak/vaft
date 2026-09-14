"""The Osborne pfile: the transcript, its round trip, and the unit ladder."""

from __future__ import annotations

import numpy as np
import pytest
from pfile_fixtures import SECTIONS, write_pfile_file

from vaft.data.pfile import (
    PFILE_SECTION_ORDER,
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


def test_every_section_is_read_in_file_order_with_its_own_unit(run):
    pfile, expected = run
    assert pfile.keys() == tuple(expected["keys"])
    assert pfile.keys() == PFILE_SECTION_ORDER  # the fixture writes all 22
    assert pfile.unit("ne") == "10^20/m^3"
    assert pfile.unit("te") == "KeV"
    assert pfile.unit("kpol") == "km/s/T"


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


def test_a_file_with_no_sections_is_refused(tmp_path):
    path = tmp_path / "notapfile"
    path.write_text("some notes\nand nothing numeric\n", encoding="utf-8")
    with pytest.raises(PFileFormatError, match="holds no profile sections"):
        read_pfile(path)


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


def test_every_section_name_in_the_fixture_is_catalogued():
    """The fixture is the format; if it grows a section the module has never
    heard of, that is a gap rather than a fixture quirk."""
    assert tuple(key for key, _ in SECTIONS) == PFILE_SECTION_ORDER
