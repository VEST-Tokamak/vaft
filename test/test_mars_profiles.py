"""MARS ``PROF*.IN``: two rotations, two quantities, nothing scaled."""

from __future__ import annotations

import numpy as np
import pytest
from mars_profile_fixtures import DECK_FILES, write_mars_deck

from vaft.data.kinetic_profiles import KineticProfiles
from vaft.data.mars_profiles import (
    MARS_FIELD_FILES,
    MARS_PROFILE_FILES,
    MARS_RADIAL_VARIABLES,
    MarsProfile,
    MarsProfileFormatError,
    read_mars_profile,
    read_mars_profiles,
    write_mars_profile,
    write_mars_profiles,
)

#: The first three lines of a MARS profile, written out here rather than
#: taken from the module: the `<count> <irad>` header and the `%.18e`
#: two-column rows *are* the format, and a test that imported them could only
#: compare the writer with itself.
EXPECTED_OPENING = [
    "41 1",
    "0.000000000000000000e+00 3.999999999999999181e+19",
    "2.500000000000000139e-02 3.999999999999998362e+19",
]


@pytest.fixture()
def deck(tmp_path):
    expected = write_mars_deck(tmp_path / "run")
    return read_mars_profiles(expected["directory"]), expected


# --- C-27: two files, two quantities ------------------------------------------


def test_the_two_rotation_files_are_two_quantities(deck):
    """The regression for C-27's MARS half.

    MARS reads PROFROT.IN under NPROFR into its fluid-rotation array and
    PROFWE.IN under NPROFWE into its E x B array; they differ by the ion
    diamagnetic frequency. The reader this replaces mapped both to one key
    and then skipped PROFWE.IN whenever PROFROT.IN had parsed, so the E x B
    file was unreachable -- and a second reader used the opposite precedence.
    """
    profiles, expected = deck
    assert profiles.omega_tor is not None
    assert profiles.omega_exb is not None
    np.testing.assert_array_equal(profiles.omega_tor, expected["values"]["PROFROT.IN"])
    np.testing.assert_array_equal(profiles.omega_exb, expected["values"]["PROFWE.IN"])
    assert not np.array_equal(profiles.omega_tor, profiles.omega_exb)


def test_neither_rotation_file_takes_precedence_over_the_other(tmp_path):
    """A deck with only one of them yields only that one -- not the other
    standing in for it."""
    only_rot = write_mars_deck(
        tmp_path / "rot", files=("PROFDEN.IN", "PROFROT.IN")
    )
    only_we = write_mars_deck(tmp_path / "we", files=("PROFDEN.IN", "PROFWE.IN"))

    rot = read_mars_profiles(only_rot["directory"])
    assert rot.omega_tor is not None and rot.omega_exb is None

    we = read_mars_profiles(only_we["directory"])
    assert we.omega_exb is not None and we.omega_tor is None


def test_the_mapping_names_each_file_once():
    assert MARS_PROFILE_FILES["PROFROT.IN"] == "omega_tor"
    assert MARS_PROFILE_FILES["PROFWE.IN"] == "omega_exb"
    assert len(set(MARS_PROFILE_FILES.values())) == len(MARS_PROFILE_FILES)


def test_writing_will_not_fill_the_fluid_rotation_file_from_the_exb_one(tmp_path):
    """Which is what produced the committed decks: one column written into
    both files, telling MARS the plasma rotates at its E x B frequency."""
    psi = np.linspace(0.0, 1.0, 9)
    profiles = KineticProfiles(psi_norm=psi, n_e=np.full(9, 4e19), omega_exb=-1e4 * psi)
    written = write_mars_profiles(profiles, tmp_path / "out")
    assert [p.name for p in written] == ["PROFDEN.IN", "PROFWE.IN"]
    assert not (tmp_path / "out" / "PROFROT.IN").exists()


def test_a_copied_rotation_pair_is_recorded(tmp_path):
    """Every committed deck has the two files byte-identical. That is not an
    error, but it is worth saying out loud."""
    expected = write_mars_deck(tmp_path / "run", identical_rotations=True)
    profiles = read_mars_profiles(expected["directory"])
    np.testing.assert_array_equal(profiles.omega_tor, profiles.omega_exb)
    assert "copied from the other" in profiles.provenance["omega_tor"]
    assert "copied from the other" in profiles.provenance["omega_exb"]


def test_a_deck_whose_rotations_differ_carries_no_such_note(deck):
    profiles, _ = deck
    assert "copied from the other" not in profiles.provenance["omega_tor"]


# --- the file itself ----------------------------------------------------------


def test_a_profile_round_trips_byte_for_byte(deck, tmp_path):
    _, expected = deck
    for name in DECK_FILES:
        source = expected["directory"] / name
        out = write_mars_profile(read_mars_profile(source), tmp_path / name)
        assert out.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_the_written_format_is_the_one_the_decks_carry(deck, tmp_path):
    _, expected = deck
    profile = read_mars_profile(expected["directory"] / "PROFDEN.IN")
    text = write_mars_profile(profile, tmp_path / "PROFDEN.IN").read_text(encoding="utf-8")
    assert text.splitlines()[:3] == EXPECTED_OPENING


def test_the_radial_variable_key_survives_a_single_file_round_trip(tmp_path):
    expected = write_mars_deck(tmp_path / "run", irad=2, files=("PROFDEN.IN",))
    source = expected["directory"] / "PROFDEN.IN"
    profile = read_mars_profile(source)
    assert profile.irad == 2
    out = write_mars_profile(profile, tmp_path / "again.IN")
    assert out.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_the_arrays_are_read_only(deck):
    profiles, expected = deck
    profile = read_mars_profile(expected["directory"] / "PROFDEN.IN")
    assert not profile.values.flags.writeable
    assert not profile.coordinate.flags.writeable
    with pytest.raises(ValueError):
        profile.values[0] = 0.0


def test_nothing_is_scaled(deck):
    """MARS reads these in SI and converts them itself when its NEXPV flag
    says the file carries absolute values. A reader that normalised would be
    wrong by the Alfven frequency."""
    profiles, expected = deck
    np.testing.assert_allclose(profiles.n_e, expected["values"]["PROFDEN.IN"])
    assert profiles.n_e.max() > 1e19  # m^-3, as written
    assert abs(profiles.omega_tor).max() > 1e4  # rad/s, not omega / omega_A


def test_the_density_file_fills_the_electron_density_only(deck):
    """MARS carries one density; equating the ion density with it is a
    modelling choice a caller makes, not one a reader may make for them."""
    profiles, _ = deck
    assert profiles.n_e is not None
    assert profiles.n_i is None


def test_the_files_abscissa_is_s_and_becomes_psi_norm_by_squaring(deck):
    """The header's second field names the abscissa, and MARS branches on it
    in every one of its profile readers: key 1 is s, the square root of the
    normalised poloidal flux, and key 2 is the toroidal equivalent. Reading
    the column straight into psi_norm puts every point at the square root of
    where it belongs -- 0.5 in the file is psi_N = 0.25 -- which is what the
    converter this replaces did."""
    profiles, expected = deck
    np.testing.assert_array_equal(profiles.psi_norm, expected["coordinate"] ** 2)
    assert profiles.normalization.method == "mars_s_squared"
    assert profiles.normalization.axis_value == pytest.approx(expected["coordinate"][0])
    assert profiles.normalization.edge_value == pytest.approx(expected["coordinate"][-1])
    # And it really is a different coordinate, not a relabelling.
    mid = len(profiles) // 2
    assert profiles.psi_norm[mid] != pytest.approx(expected["coordinate"][mid])


def test_the_file_keeps_its_own_abscissa(deck):
    """MarsProfile is the file; the conversion belongs to the deck reader."""
    _, expected = deck
    profile = read_mars_profile(expected["directory"] / "PROFDEN.IN")
    np.testing.assert_array_equal(profile.coordinate, expected["coordinate"])
    assert "poloidal" in profile.radial_variable
    np.testing.assert_array_equal(profile.psi_norm(), expected["coordinate"] ** 2)


def test_a_toroidal_flux_deck_is_refused_rather_than_guessed_at(tmp_path):
    """Key 2 is the square root of the *toroidal* flux. Reaching poloidal
    flux from it needs an equilibrium, which a file-format reader has not
    got."""
    expected = write_mars_deck(tmp_path / "run", irad=2)
    with pytest.raises(MarsProfileFormatError, match="needs an equilibrium"):
        read_mars_profiles(expected["directory"])
    profile = read_mars_profile(expected["directory"] / "PROFDEN.IN")
    assert "toroidal" in profile.radial_variable
    with pytest.raises(MarsProfileFormatError, match="needs an equilibrium"):
        profile.psi_norm()


def test_a_deck_whose_files_disagree_about_their_radial_variable_is_refused(tmp_path):
    expected = write_mars_deck(tmp_path / "run", irad_in={"PROFROT.IN": 2})
    with pytest.raises(MarsProfileFormatError, match="disagree about their own radial"):
        read_mars_profiles(expected["directory"])


def test_a_radial_variable_key_mars_would_stop_on_is_refused(tmp_path):
    expected = write_mars_deck(tmp_path / "run", irad=3, files=("PROFDEN.IN",))
    with pytest.raises(MarsProfileFormatError, match="not a MARS radial-variable key"):
        read_mars_profile(expected["directory"] / "PROFDEN.IN")
    with pytest.raises(MarsProfileFormatError, match="not a MARS radial-variable key"):
        MarsProfile(coordinate=np.linspace(0, 1, 3), values=np.ones(3), irad=7)


def test_writing_converts_the_coordinate_back_and_says_which_it_is(tmp_path):
    """MARS wants s. Writing psi_norm out under key 1 would tell it the
    column is already s and put every point at the square of where it
    belongs."""
    psi = np.array([0.0, 0.25, 0.64, 1.0])
    profiles = KineticProfiles(psi_norm=psi, n_e=np.full(4, 4e19))
    written = write_mars_profiles(profiles, tmp_path / "out")
    text = written[0].read_text(encoding="utf-8").splitlines()
    assert text[0] == "4 1"
    np.testing.assert_allclose(
        [float(line.split()[0]) for line in text[1:]], np.sqrt(psi)
    )


def test_provenance_names_the_file_and_what_mars_calls_it(deck):
    profiles, _ = deck
    assert profiles.provenance["omega_tor"].startswith("PROFROT.IN -- fluid rotation")
    assert "NPROFWE" in profiles.provenance["omega_exb"]


# --- what a deck must not be allowed to say -----------------------------------


def test_files_on_different_coordinates_are_refused(tmp_path):
    """The reader this replaces warned and kept the first file's coordinate,
    grafting every other file's values onto it point by point."""
    expected = write_mars_deck(tmp_path / "run", mismatched_psi_in="PROFTI.IN")
    with pytest.raises(MarsProfileFormatError, match="different radial coordinate"):
        read_mars_profiles(expected["directory"])


def test_files_of_different_lengths_are_refused(tmp_path):
    """This case the legacy reader did not even warn about: it stored the
    mismatched array and failed much later, naming no file."""
    expected = write_mars_deck(tmp_path / "run", short_psi_in="PROFTE.IN")
    with pytest.raises(MarsProfileFormatError, match="PROFTE.IN is on 40 points"):
        read_mars_profiles(expected["directory"])


def test_a_header_count_that_disagrees_with_the_file_is_refused(tmp_path):
    expected = write_mars_deck(tmp_path / "run", wrong_count_in="PROFDEN.IN")
    with pytest.raises(MarsProfileFormatError, match="declares 42 rows and carries 41"):
        read_mars_profiles(expected["directory"])


def test_a_ragged_row_is_named_rather_than_skipped(tmp_path):
    expected = write_mars_deck(tmp_path / "run", ragged_row_in="PROFDEN.IN")
    with pytest.raises(MarsProfileFormatError, match=r"PROFDEN.IN row 3 has 1 columns"):
        read_mars_profiles(expected["directory"])


def test_a_file_with_no_header_is_refused(tmp_path):
    path = tmp_path / "PROFDEN.IN"
    path.write_text("0.0 1.0\n1.0 2.0\n", encoding="utf-8")
    with pytest.raises(MarsProfileFormatError, match="row count and a radial-variable key"):
        read_mars_profile(path)


def test_a_directory_with_no_profiles_is_refused(tmp_path):
    tmp_path.joinpath("README.md").write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(MarsProfileFormatError, match="holds no PROF"):
        read_mars_profiles(tmp_path)


def test_a_missing_directory_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_mars_profiles(tmp_path / "nope")


def test_files_differing_by_a_rounding_error_are_still_refused(tmp_path):
    """The contract is one coordinate, not nearly one: a deck whose files
    went through different float round trips is two coordinates, and
    averaging them away is the defect being fixed, one decimal place down."""
    expected = write_mars_deck(tmp_path / "run", files=("PROFDEN.IN", "PROFTE.IN"))
    path = expected["directory"] / "PROFTE.IN"
    lines = path.read_text(encoding="utf-8").splitlines()
    head, rows = lines[0], lines[1:]
    nudged = []
    for row in rows:
        x, y = row.split()
        nudged.append(f"{float(x) * (1 + 1e-12):.18e} {y}")
    path.write_text("\n".join([head] + nudged) + "\n", encoding="utf-8")
    with pytest.raises(MarsProfileFormatError, match="different radial coordinate"):
        read_mars_profiles(expected["directory"])


def test_a_blank_line_among_the_rows_is_tolerated(tmp_path):
    """Fortran list-directed READ skips a blank record, so MARS reads such a
    file; the count still has to match the rows that are there."""
    expected = write_mars_deck(
        tmp_path / "run", files=("PROFDEN.IN",), blank_line_in="PROFDEN.IN"
    )
    profile = read_mars_profile(expected["directory"] / "PROFDEN.IN")
    assert len(profile) == expected["points"]


def test_a_section_declaring_no_rows_is_refused(tmp_path):
    path = tmp_path / "PROFDEN.IN"
    path.write_text("0 1\n", encoding="utf-8")
    with pytest.raises(MarsProfileFormatError, match="declares 0 rows"):
        read_mars_profile(path)


def test_a_header_with_a_third_field_is_refused(tmp_path):
    """PROFROTC.IN's header is count, key and a harmonic count -- a different
    format. Reading its first two fields and discarding the third would lose
    data silently."""
    path = tmp_path / "PROFROTC.IN"
    path.write_text("2 1 4\n0.0 1.0\n1.0 2.0\n", encoding="utf-8")
    with pytest.raises(MarsProfileFormatError, match="third field"):
        read_mars_profile(path)


def test_writing_refuses_a_coordinate_that_does_not_increase(tmp_path):
    """MARS splines against it, as GPEC and the pfile consumers do."""
    profile = MarsProfile(
        coordinate=np.array([0.0, 0.5, 0.4, 1.0]), values=np.ones(4)
    )
    with pytest.raises(MarsProfileFormatError, match="does not increase"):
        write_mars_profile(profile, tmp_path / "PROFDEN.IN")


def test_writing_refuses_a_coordinate_that_is_not_finite(tmp_path):
    profile = MarsProfile(
        coordinate=np.array([0.0, np.nan, 1.0]), values=np.ones(3)
    )
    with pytest.raises(MarsProfileFormatError, match="coordinates that are not finite"):
        write_mars_profile(profile, tmp_path / "PROFDEN.IN")


def test_a_profile_of_mismatched_lengths_is_refused():
    with pytest.raises(MarsProfileFormatError, match="values against"):
        MarsProfile(coordinate=np.linspace(0, 1, 4), values=np.ones(3))


def test_a_scalar_is_not_a_profile():
    with pytest.raises(MarsProfileFormatError, match="1-D array"):
        MarsProfile(coordinate=1.0, values=2.0)


def test_a_non_numeric_token_names_the_file_and_the_row(tmp_path):
    path = tmp_path / "PROFDEN.IN"
    path.write_text("3 1\n0.0 1.0\n0.5 abc\n1.0 3.0\n", encoding="utf-8")
    with pytest.raises(MarsProfileFormatError, match="PROFDEN.IN row 2 is not two numbers"):
        read_mars_profile(path)


def test_a_fortran_exponent_is_read(tmp_path):
    path = tmp_path / "PROFDEN.IN"
    path.write_text("2 1\n0.0D+00 1.0D+19\n1.0D+00 2.0D+19\n", encoding="utf-8")
    profile = read_mars_profile(path)
    np.testing.assert_allclose(profile.values, [1e19, 2e19])


def test_the_field_and_file_tables_are_exact_inverses():
    assert MARS_FIELD_FILES == {v: k for k, v in MARS_PROFILE_FILES.items()}
    assert set(MARS_RADIAL_VARIABLES) == {1, 2}


def test_writing_refuses_a_value_that_is_not_finite(tmp_path):
    psi = np.linspace(0.0, 1.0, 4)
    profile = MarsProfile(coordinate=psi, values=np.array([1.0, np.nan, 3.0, 4.0]))
    with pytest.raises(MarsProfileFormatError, match="not finite"):
        write_mars_profile(profile, tmp_path / "PROFDEN.IN")


def test_writing_a_set_that_holds_nothing_mars_wants_is_refused(tmp_path):
    profiles = KineticProfiles(psi_norm=np.linspace(0, 1, 5), p_total=np.ones(5))
    with pytest.raises(MarsProfileFormatError, match="nothing to write"):
        write_mars_profiles(profiles, tmp_path / "out")


# --- what a deck may say that this module does not know -----------------------


def test_an_unrecognised_profile_is_kept_rather_than_dropped(tmp_path):
    expected = write_mars_deck(tmp_path / "run", unknown_file=True)
    profiles = read_mars_profiles(expected["directory"])
    assert "ZEFF" in profiles.extras
    np.testing.assert_allclose(profiles.extras["ZEFF"], expected["values"]["PROFZEFF.IN"])
    assert "not catalogued" in profiles.provenance["ZEFF"]


def test_an_unrecognised_profile_is_written_back(tmp_path):
    """"Kept rather than dropped" has to survive the writer, or a deck read
    and written loses the file the reader promised to keep."""
    expected = write_mars_deck(tmp_path / "run", unknown_file=True)
    profiles = read_mars_profiles(expected["directory"])
    written = write_mars_profiles(profiles, tmp_path / "out")
    assert (tmp_path / "out" / "PROFZEFF.IN").exists()
    assert [p.name for p in written] == sorted(list(DECK_FILES) + ["PROFZEFF.IN"])
    bare = write_mars_profiles(profiles, tmp_path / "bare", include_extras=False)
    assert [p.name for p in bare] == sorted(DECK_FILES)


def test_a_file_in_another_mars_layout_does_not_make_the_deck_unreadable(tmp_path):
    """PROFPA.IN carries one column per species and PROFROTC.IN a harmonic
    block. They belong to legitimate decks, and refusing the whole directory
    over one would turn "kept rather than dropped" into "all or nothing"."""
    expected = write_mars_deck(tmp_path / "run", unreadable_file=True)
    profiles = read_mars_profiles(expected["directory"])
    assert profiles.available()  # the rest of the deck read
    assert "present but not read" in profiles.provenance["PROFPA.IN"]


def test_a_catalogued_file_in_the_wrong_shape_is_still_refused(tmp_path):
    """The tolerance above is for files this module does not claim to read."""
    expected = write_mars_deck(tmp_path / "run", ragged_row_in="PROFDEN.IN")
    with pytest.raises(MarsProfileFormatError, match="PROFDEN.IN row 3"):
        read_mars_profiles(expected["directory"])


def test_a_deck_round_trips_through_the_container(deck, tmp_path):
    profiles, expected = deck
    written = write_mars_profiles(profiles, tmp_path / "out")
    assert [p.name for p in written] == sorted(DECK_FILES)
    for path in written:
        assert path.read_text(encoding="utf-8") == (
            expected["directory"] / path.name
        ).read_text(encoding="utf-8")
