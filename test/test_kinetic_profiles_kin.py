"""The kinetic-profile container and GPEC ``.kin`` I/O."""

from __future__ import annotations

import numpy as np
import pytest
from kinetic_profile_fixtures import TRUNCATED_SPAN, profile_columns, write_kin_file

from vaft.data.kinetic_profiles import (
    KIN_COLUMNS,
    UNITS,
    KineticProfiles,
    PsiNormalization,
    normalize_psi,
    read_kin,
    write_kin,
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
    assert UNITS["n_e"] == "m^-3" and UNITS["T_e"] == "eV" and UNITS["omega_exb"] == "rad/s"
    assert profiles.unit("n_e") == "m^-3"


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
    assert "column_6" in profiles.extras
    np.testing.assert_allclose(profiles.extras["column_6"], np.arange(len(profiles)))


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


# --- writing ------------------------------------------------------------------


def test_the_written_header_is_the_one_gpecs_examples_carry(kin, tmp_path):
    profiles, _ = kin
    text = write_kin(profiles, tmp_path / "out.kin").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert "psi" in first and "ni(m^-3)" in first and "wexb(rad/s)" in first
    # A header is a comment: the file still reads without it.
    bare = write_kin(profiles, tmp_path / "bare.kin", header=False)
    np.testing.assert_allclose(read_kin(bare).psi_norm, profiles.psi_norm)


def test_provenance_records_where_each_column_came_from(kin):
    profiles, _ = kin
    assert profiles.provenance["n_e"].endswith("column 2")
    assert profiles.source.endswith("synthetic.kin")
    assert profiles.normalization.source == "synthetic.kin"
