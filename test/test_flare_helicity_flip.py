"""Conjugating a GPEC BRZPHI field for FLARE.

FLARE reconstructs ``B = Re(C exp(-i n phi))``, so conjugating the stored
coefficients flips the helicity the trace assumes. The operation is
text-level on purpose: FLARE splines the ASCII, so re-formatting the numbers
would change what it reads even where the value is unchanged.
"""

from __future__ import annotations

import pytest

from vaft.code.flare import BRZPHI_IMAGINARY_COLUMNS, write_helicity_flipped_field

HEADER = (
    " GPEC_BRZPHI: Total perturbed field\n"
    " v1.5.5-378-gf06e6ab\n"
    "\n"
    "   n  =     1\n"
    "   nr =     2  nz =     2\n"
    "\n"
    "  l                r                z        real(b_r)        imag(b_r)"
    "        real(b_z)        imag(b_z)      real(b_phi)      imag(b_phi)\n"
)
ROWS = (
    "  0  8.39999974E-001 -1.60000002E+000 -1.53691428E-004  9.75592631E-005"
    " -9.91222792E-005  7.09735557E-005  1.50157577E-005  1.86837439E-004\n"
    "  0  8.39999974E-001 -1.57419357E+000 -1.57140008E-004 -9.98944809E-005"
    " -9.97368958E-005  7.18798110E-005  1.52077629E-005  1.90713045E-004\n"
)


@pytest.fixture
def field(tmp_path):
    path = tmp_path / "gpec_brzphi_n1.out"
    path.write_text(HEADER + ROWS)
    return path


def _rows(path):
    return [
        line.split() for line in path.read_text().splitlines() if len(line.split()) == 9
        and not line.lstrip().startswith("l ")
    ]


def test_only_the_imaginary_columns_change(field):
    before = _rows(field)
    after = _rows(write_helicity_flipped_field(field))
    assert len(after) == len(before) == 2
    for original, flipped in zip(before, after):
        for index, (was, now) in enumerate(zip(original, flipped)):
            if index in BRZPHI_IMAGINARY_COLUMNS:
                assert float(now) == -float(was)
            else:
                assert now == was, f"column {index} was re-formatted"


def test_the_sign_is_flipped_on_the_token_not_through_a_float(field):
    """FLARE splines the ASCII, so a value that round-trips through a float
    and back can differ in its last digit even when the number is the same.
    The digits have to survive verbatim."""
    flipped = _rows(write_helicity_flipped_field(field))
    assert flipped[0][4] == "-9.75592631E-005"
    # A leading minus is removed rather than turned into "--" or " -".
    assert flipped[1][4] == "9.98944809E-005"


def test_the_header_is_copied_verbatim(field):
    """n, nr and nz sit at fixed character positions, and FLARE reads them
    there."""
    out = write_helicity_flipped_field(field).read_text()
    for line in HEADER.splitlines():
        assert line in out.splitlines()


def test_the_default_name_marks_the_flip(field):
    assert write_helicity_flipped_field(field).name == "gpec_brzphi_n1_helicityflip.out"


def test_a_destination_may_be_named(field, tmp_path):
    target = tmp_path / "elsewhere.out"
    assert write_helicity_flipped_field(field, target) == target
    assert target.exists()


def test_flipping_twice_returns_the_original_numbers(field, tmp_path):
    """Conjugation is an involution, and the text-level flip has to be one
    too -- a sign written as "+x" on the way out would not come back."""
    once = write_helicity_flipped_field(field, tmp_path / "once.out")
    twice = write_helicity_flipped_field(once, tmp_path / "twice.out")
    assert _rows(twice) == _rows(field)


def test_a_file_with_no_data_row_is_refused(tmp_path):
    """Writing the copy anyway would leave a file named helicityflip that is
    identical to its input."""
    path = tmp_path / "gpec_brzphi_n1.out"
    path.write_text(HEADER)
    with pytest.raises(ValueError, match="no BRZPHI data row"):
        write_helicity_flipped_field(path)


def test_a_missing_source_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="perturbed-field input"):
        write_helicity_flipped_field(tmp_path / "absent.out")


def test_rows_are_found_by_content_not_by_position(tmp_path):
    """GPEC's header length varies between builds, so a fixed skip would
    treat a header line as data on one build and drop a row on another."""
    path = tmp_path / "gpec_brzphi_n1.out"
    path.write_text(" extra provenance line\n" + HEADER + ROWS)
    assert len(_rows(write_helicity_flipped_field(path))) == 2


def test_a_nine_token_line_that_is_not_numeric_stays_a_header(tmp_path):
    """The column-name line has nine tokens too."""
    path = tmp_path / "gpec_brzphi_n1.out"
    path.write_text(HEADER + ROWS)
    out = write_helicity_flipped_field(path).read_text()
    assert "real(b_r)        imag(b_r)" in out
