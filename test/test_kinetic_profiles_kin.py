"""The kinetic-profile container and GPEC ``.kin`` I/O."""

from __future__ import annotations

import numpy as np
import pytest
from kinetic_profile_fixtures import TRUNCATED_SPAN, profile_columns, write_kin_file

from vaft.data.kinetic_profiles import (
    KINETIC_UNITS,
    KIN_COLUMNS,
    KineticProfiles,
    normalize_psi,
    read_kin,
    write_kin,
)

#: The header line GPEC's own example files carry, byte for byte -- written
#: out here rather than imported, so that changing the module's constant
#: fails a test instead of moving the goalposts with it.  It is the first line
#: of all 65 .kin files under the reference tree, DIII-D and MAST alike.
GPEC_KIN_HEADER = (
    "             psi         ni(m^-3)         ne(m^-3)"
    "           ti(eV)           te(eV)      wexb(rad/s)"
)


@pytest.fixture()
def kin(tmp_path):
    expected = write_kin_file(tmp_path)
    return read_kin(tmp_path / "synthetic.kin"), expected


# --- the container ------------------------------------------------------------


def test_reads_the_six_columns_in_gpecs_order(kin):
    profiles, expected = kin
    assert KIN_COLUMNS == ("psi_norm", "n_i", "n_e", "T_i", "T_e", "omega_exb")
    for name, values in expected.items():
        np.testing.assert_allclose(getattr(profiles, name), values)
    assert profiles.available() == ("n_e", "n_i", "T_e", "T_i", "omega_exb")


def test_units_are_the_containers_own_not_the_files(kin):
    profiles, _ = kin
    assert KINETIC_UNITS["n_e"] == "m^-3"
    assert KINETIC_UNITS["T_e"] == "eV"
    assert KINETIC_UNITS["omega_exb"] == "rad/s"
    assert profiles.unit("n_e") == "m^-3"


def test_the_units_table_and_the_container_are_reachable_from_vaft_data():
    """The pfile and MARS readers convert through this table and no other, so
    it has to be importable from where its docstring says it is."""
    import vaft.data as data

    assert data.KINETIC_UNITS["n_e"] == "m^-3"
    assert data.KIN_HEADER == GPEC_KIN_HEADER
    assert "n_e" in data.PROFILE_FIELDS and "psi_norm" not in data.PROFILE_FIELDS
    assert data.kinetic_profiles.__name__ == "vaft.data.kinetic_profiles"


def test_a_profile_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="every profile is on the same radial coordinate"):
        KineticProfiles(psi_norm=np.linspace(0, 1, 5), n_e=np.zeros(4))


def test_a_missing_profile_names_what_is_there(kin):
    profiles, _ = kin
    with pytest.raises(KeyError, match="no profile 'p_total'"):
        profiles.field("p_total")


def test_extra_columns_are_kept_rather_than_dropped(tmp_path):
    write_kin_file(tmp_path, extra_column=True)
    profiles = read_kin(tmp_path / "synthetic.kin")
    # Numbered as a person reading the file would: the seventh column is 7.
    assert "column_7" in profiles.extras
    np.testing.assert_allclose(profiles.extras["column_7"], np.arange(len(profiles)))
    np.testing.assert_allclose(profiles.field("column_7"), np.arange(len(profiles)))
    assert profiles.provenance["column_7"].endswith("column 7")


def test_a_profile_set_cannot_be_edited_behind_its_provenance(kin):
    """frozen= stops the attributes being rebound and nothing else, so the
    arrays and mappings are sealed too: a set that says "as read" has to be."""
    profiles, _ = kin
    with pytest.raises(ValueError):
        profiles.psi_norm[0] = 42.0
    with pytest.raises(ValueError):
        profiles.n_e[0] = 0.0
    with pytest.raises(TypeError):
        profiles.provenance["n_e"] = "somewhere else"


def test_sealing_does_not_reach_back_into_the_callers_array():
    """A read-only *view*, not a read-only array: passing an array in must not
    make the caller's own copy unwritable."""
    psi = np.linspace(0.0, 1.0, 5)
    KineticProfiles(psi_norm=psi)
    psi[0] = 0.5  # still the caller's array
    assert psi[0] == 0.5


# --- C-10: the radial coordinate is never rescaled on read --------------------


def test_a_truncated_coordinate_survives_a_round_trip(tmp_path):
    """The regression for C-10.

    A real .kin spans 0.00495 to 1.0.  The legacy readers stretched that onto
    [0, 1] and the writer put the stretched values back, so one round trip
    moved every interior point permanently.
    """
    write_kin_file(tmp_path, span=TRUNCATED_SPAN)
    profiles = read_kin(tmp_path / "synthetic.kin")
    assert profiles.psi_norm[0] == pytest.approx(TRUNCATED_SPAN[0])
    assert profiles.normalization.method == "as_read"

    out = write_kin(profiles, tmp_path / "again.kin")
    assert out.read_text(encoding="utf-8") == (tmp_path / "synthetic.kin").read_text(encoding="utf-8")


def test_normalizing_is_an_explicit_operation_that_records_itself(tmp_path):
    write_kin_file(tmp_path, span=TRUNCATED_SPAN)
    profiles = read_kin(tmp_path / "synthetic.kin")

    stretched = normalize_psi(profiles, method="min_max")
    assert stretched.psi_norm[0] == 0.0 and stretched.psi_norm[-1] == 1.0
    assert stretched.normalization.method == "min_max"
    assert stretched.normalization.axis_value == pytest.approx(TRUNCATED_SPAN[0])
    # The measurable delta the legacy code applied silently.
    assert abs(stretched.psi_norm[1] - profiles.psi_norm[1]) > 1e-3
    # And the original is untouched: the container is frozen.
    assert profiles.psi_norm[0] == pytest.approx(TRUNCATED_SPAN[0])


def test_explicit_normalization_needs_its_edge():
    profiles = KineticProfiles(psi_norm=np.linspace(0.1, 0.9, 5))
    with pytest.raises(ValueError, match="needs edge="):
        normalize_psi(profiles, method="explicit")
    scaled = normalize_psi(profiles, method="explicit", edge=2.0)
    np.testing.assert_allclose(scaled.psi_norm, np.linspace(0.05, 0.45, 5))
    assert scaled.normalization.edge_value == 2.0


def test_an_axis_and_edge_without_a_method_is_refused():
    """The silent rescale, with a provenance record vouching for it: a default
    method would ignore both arguments and stretch the coordinate instead."""
    profiles = KineticProfiles(psi_norm=np.linspace(0.00495, 1.0, 5))
    with pytest.raises(TypeError, match="method"):
        normalize_psi(profiles, axis=0.0, edge=1.0)
    with pytest.raises(ValueError, match="takes the axis and edge from"):
        normalize_psi(profiles, method="min_max", axis=0.0, edge=1.0)


def test_an_unknown_normalization_method_is_refused():
    profiles = KineticProfiles(psi_norm=np.linspace(0.1, 0.9, 5))
    with pytest.raises(ValueError, match="must be 'min_max' or 'explicit'"):
        normalize_psi(profiles, method="stretch")


# --- GPEC's own header/footer rule --------------------------------------------


def test_a_header_and_footer_are_ignored_the_way_gpec_ignores_them(tmp_path):
    """"No lines start with a number" is the file's actual contract."""
    expected = write_kin_file(tmp_path, footer=True)
    profiles = read_kin(tmp_path / "synthetic.kin")
    assert len(profiles) == expected["psi_norm"].size
    np.testing.assert_allclose(profiles.psi_norm, expected["psi_norm"])


def test_the_table_stops_at_the_footer_rather_than_resuming_after_it(tmp_path):
    """GPEC's readtable takes the *first contiguous block*: it stops at the
    first non-numeric line and ignores the rest of the file. Collecting every
    numeric line anywhere would read the summary row below the footer as one
    more data point, silently disagreeing with GPEC about the profile."""
    expected = write_kin_file(tmp_path, numeric_footer=True)
    profiles = read_kin(tmp_path / "synthetic.kin")
    assert len(profiles) == expected["psi_norm"].size
    assert profiles.psi_norm[-1] == pytest.approx(expected["psi_norm"][-1])


def test_a_word_that_python_would_read_as_a_number_is_still_header_text(tmp_path):
    """GPEC looks at one character: a line starting with a letter is header,
    even when it is "nan" or "Infinity", which float() accepts."""
    expected = write_kin_file(tmp_path, header="nan is not a data line")
    profiles = read_kin(tmp_path / "synthetic.kin")
    assert len(profiles) == expected["psi_norm"].size


def test_fortran_exponents_are_read_rather_than_dropped(tmp_path):
    """A .kin written by a Fortran tool carries 1.0D+00, which GPEC's
    list-directed read accepts and Python's float() does not -- so a reader
    that classifies by float() would silently discard every such row."""
    expected = write_kin_file(tmp_path, fortran_exponents=True)
    profiles = read_kin(tmp_path / "synthetic.kin")
    assert len(profiles) == expected["psi_norm"].size
    np.testing.assert_allclose(profiles.n_e, expected["n_e"], rtol=1e-7)


def test_a_ragged_row_is_named_rather_than_dying_inside_numpy(tmp_path):
    write_kin_file(tmp_path, ragged_row=3)
    with pytest.raises(ValueError, match="data row 4 has 5 columns against 6"):
        read_kin(tmp_path / "synthetic.kin")


def test_a_file_with_no_header_reads(tmp_path):
    expected = write_kin_file(tmp_path, header=False)
    profiles = read_kin(tmp_path / "synthetic.kin")
    np.testing.assert_allclose(profiles.n_e, expected["n_e"])


def test_a_file_with_no_data_rows_is_refused(tmp_path):
    (tmp_path / "empty.kin").write_text("psi ni ne ti te wexb\nnothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        read_kin(tmp_path / "empty.kin")


def test_too_few_columns_is_refused(tmp_path):
    (tmp_path / "short.kin").write_text("  1.0   2.0   3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a .kin file has 6"):
        read_kin(tmp_path / "short.kin")


# --- C-27: the rotation column is omega_E, not toroidal rotation --------------


def test_writing_refuses_a_set_that_has_only_toroidal_rotation(tmp_path):
    columns = profile_columns()
    omega_tor = -columns.pop("omega_exb")
    profiles = KineticProfiles(**columns, omega_tor=omega_tor)
    with pytest.raises(ValueError, match="rotation column is omega_E"):
        write_kin(profiles, tmp_path / "wrong.kin")


def test_writing_names_whatever_is_missing(tmp_path):
    columns = profile_columns()
    del columns["T_i"]
    with pytest.raises(ValueError, match=r"missing \['T_i'\]"):
        write_kin(KineticProfiles(**columns), tmp_path / "short.kin")


def test_an_all_zero_rotation_column_is_refused_unless_asked(tmp_path):
    """GPEC substitutes 1e-9 for every zero, so the file would not mean what it says."""
    columns = profile_columns()
    columns["omega_exb"] = np.zeros_like(columns["psi_norm"])
    profiles = KineticProfiles(**columns)
    with pytest.raises(ValueError, match="replaces every zero with 1e-9"):
        write_kin(profiles, tmp_path / "zero.kin")
    written = write_kin(profiles, tmp_path / "zero.kin", allow_zero_rotation=True)
    assert read_kin(written).omega_exb.max() == 0.0


def test_a_partly_zero_rotation_column_is_refused_too(tmp_path):
    """GPEC substitutes per element, not per column, so a core that was
    zero-filled by a converter is the same defect on half the profile."""
    columns = profile_columns()
    columns["omega_exb"] = columns["omega_exb"].copy()
    columns["omega_exb"][:5] = 0.0
    with pytest.raises(ValueError, match="zero at 5 of"):
        write_kin(KineticProfiles(**columns), tmp_path / "part.kin")


def test_a_non_finite_value_is_refused(tmp_path):
    """GPEC's own warning at the 1e-9 substitution says a NaN ruins the
    whole spline; nothing downstream recovers from one."""
    columns = profile_columns()
    columns["T_e"] = columns["T_e"].copy()
    columns["T_e"][7] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        write_kin(KineticProfiles(**columns), tmp_path / "nan.kin")


def test_a_coordinate_that_does_not_increase_is_refused(tmp_path):
    """GPEC splines against psi and assumes an increasing abscissa; a shuffled
    column produces garbage there rather than an error."""
    columns = profile_columns()
    psi = columns["psi_norm"].copy()
    psi[3], psi[4] = psi[4], psi[3]
    columns["psi_norm"] = psi
    with pytest.raises(ValueError, match="does not increase"):
        write_kin(KineticProfiles(**columns), tmp_path / "shuffled.kin")


# --- writing ------------------------------------------------------------------


def test_the_written_header_is_the_one_gpecs_examples_carry(kin, tmp_path):
    """Byte for byte against a literal, not against the module's own constant:
    a header with the right tokens and the wrong spacing reads identically and
    is no longer the line every real .kin carries."""
    profiles, _ = kin
    text = write_kin(profiles, tmp_path / "out.kin").read_text(encoding="utf-8")
    assert text.splitlines()[0] == GPEC_KIN_HEADER
    # A header is a comment: the file still reads without it.
    bare = write_kin(profiles, tmp_path / "bare.kin", header=False)
    np.testing.assert_allclose(read_kin(bare).psi_norm, profiles.psi_norm)


def test_provenance_records_where_each_column_came_from(kin):
    profiles, _ = kin
    # n_e is the file's third column: psi, ni, ne.
    assert profiles.provenance["psi_norm"].endswith("column 1")
    assert profiles.provenance["n_e"].endswith("column 3")
    assert profiles.source.endswith("synthetic.kin")
    assert profiles.normalization.source == "synthetic.kin"
