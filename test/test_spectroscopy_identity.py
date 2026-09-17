"""Atomic and spectroscopic identity: element, isotope, stage and charge.

These four are routinely conflated and must not be.  ``C III`` and ``C3+``
name *different* ions; ``D`` is hydrogen with a mass number rather than an
element of its own.  :mod:`vaft.spectroscopy` is where those distinctions are
defined, so this is where they are pinned.
"""

import pytest

from vaft.spectroscopy import (
    LineIdentity,
    Species,
    charge_state_of,
    describe_available,
    format_species,
    ionization_stage_of,
    matches,
    parse_emission_term,
    parse_line_label,
    parse_species,
)


# ---------------------------------------------------------------------------
# Spectroscopic stage against ionic charge
# ---------------------------------------------------------------------------

def test_stage_and_charge_differ_by_one_in_both_directions():
    assert charge_state_of(1) == 0 and ionization_stage_of(0) == 1
    assert charge_state_of(3) == 2 and ionization_stage_of(2) == 3
    for stage in range(1, 12):
        assert ionization_stage_of(charge_state_of(stage)) == stage


def test_a_stage_below_neutral_is_refused():
    with pytest.raises(ValueError):
        charge_state_of(0)
    with pytest.raises(ValueError):
        ionization_stage_of(-1)


@pytest.mark.parametrize(
    "spectroscopic, charge",
    [("CI", "C0"), ("CII", "C+"), ("CIII", "C2+"), ("CIV", "C3+"), ("OV", "O4+"),
     ("HeII", "He+")],
)
def test_both_notations_name_the_same_ion(spectroscopic, charge):
    """Passive spectroscopy says C III where other analysis says C2+."""
    assert parse_species(spectroscopic) == parse_species(charge)


def test_c_three_is_not_c_three_plus():
    """The off-by-one that makes the two notations easy to confuse."""
    assert parse_species("CIII") != parse_species("C3+")
    assert parse_species("CIII").charge_state == 2
    assert parse_species("C3+").ionization_stage == 4


def test_charge_may_be_written_on_either_side_of_the_sign():
    assert parse_species("C2+") == parse_species("C+2")


def test_charge_state_is_derived_so_it_cannot_drift():
    assert Species("C", ionization_stage=3).charge_state == 2
    assert Species("C").charge_state is None
    assert Species.from_charge_state("C", 2) == Species("C", None, 3)


# ---------------------------------------------------------------------------
# Element against isotope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "term, mass_number",
    [("H", 1), ("Hydrogen", 1), ("hydrogen", 1),
     ("D", 2), ("Deuterium", 2), ("deuterium", 2),
     ("T", 3), ("Tritium", 3)],
)
def test_hydrogen_isotopes_are_one_element(term, mass_number):
    """Deuterium is hydrogen with a mass number, never an element of its own."""
    species = parse_species(term)
    assert species.element == "H"
    assert species.mass_number == mass_number


def test_helium_needs_no_special_case():
    assert parse_species("He") == parse_species("Helium") == Species("He")
    assert parse_species("HeII").ionization_stage == 2


@pytest.mark.parametrize("term", ["C", "Carbon", "carbon"])
def test_element_names_resolve_in_any_case(term):
    assert parse_species(term) == Species("C")


def test_an_element_symbol_keeps_its_case_so_nickel_is_not_nitrogen():
    """``Ni`` is nickel; ``NI`` is neutral nitrogen.  Only case separates them.

    The Data Dictionary's own example, ``WI_4000``, depends on this reading.
    """
    assert parse_species("Ni") == Species("Ni")
    assert parse_species("NI") == Species("N", None, 1)
    assert parse_species("WI") == Species("W", None, 1)


# ---------------------------------------------------------------------------
# Reading a stored label
# ---------------------------------------------------------------------------

def test_the_data_dictionary_form_parses():
    identity = parse_line_label("WI_4000")
    assert identity.species == Species("W", None, 1)
    assert identity.wavelength_angstrom == 4000.0
    assert identity.series is None


def test_the_series_form_parses():
    identity = parse_line_label("H-alpha_6563")
    assert identity.species.element == "H"
    assert identity.series == "alpha"
    assert identity.wavelength_angstrom == 6563.0


def test_a_stored_label_claims_no_isotope_it_did_not_state():
    """An unmarked hydrogen line records hydrogen, not protium specifically."""
    assert parse_line_label("H-alpha_6563").species.mass_number is None
    # A selector, by contrast, does mean protium when it says H.
    assert parse_emission_term("H").species.mass_number == 1


def test_a_label_following_no_convention_is_not_guessed_at():
    assert parse_line_label("Bremsstrahlung_broadband") is None


@pytest.mark.parametrize("bad", ["", "Xq", "C IIII", "CVV", "alpha", 3.7, True, None, ["C"]])
def test_malformed_terms_resolve_to_nothing(bad):
    assert parse_species(bad) is None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

SAMPLE_LABELS = (
    "H-alpha_6563", "OI_7770", "H-beta_4861", "H-gamma_4340",
    "CII_4267", "CIII_1909", "OII_3726", "OV_629",
)


def _matched(term):
    parsed = parse_emission_term(term)
    return [
        label for label in SAMPLE_LABELS
        if parsed is not None
        and (identity := parse_line_label(label)) is not None
        and matches(parsed, identity)
    ]


@pytest.mark.parametrize(
    "term, expected",
    [
        ("CIII", ["CIII_1909"]),
        ("C2+", ["CIII_1909"]),
        ("C", ["CII_4267", "CIII_1909"]),
        ("Carbon", ["CII_4267", "CIII_1909"]),
        ("O", ["OI_7770", "OII_3726", "OV_629"]),
        ("H_alpha", ["H-alpha_6563"]),
        ("H", ["H-alpha_6563", "H-beta_4861", "H-gamma_4340"]),
    ],
)
def test_a_term_matches_at_its_own_level(term, expected):
    assert _matched(term) == expected


@pytest.mark.parametrize("separator", ["H_alpha", "H-alpha", "Halpha", "H alpha", "Hα"])
def test_a_line_is_reachable_however_it_is_spelled(separator):
    assert _matched(separator) == ["H-alpha_6563"]


@pytest.mark.parametrize("term", ["D", "Deuterium", "D_alpha", "D-alpha", "Dα", "T"])
def test_asking_for_deuterium_never_returns_hydrogen(term):
    """Hydrogen and deuterium stay distinct selectors, as the API promises.

    VEST runs hydrogen, so these match nothing here -- but they match nothing
    rather than quietly returning a line the data only ever called hydrogen.
    """
    assert _matched(term) == []


def test_hydrogen_does_not_include_deuterium():
    deuterium_line = LineIdentity(Species("H", 2), series="alpha")
    assert not matches(parse_emission_term("H"), deuterium_line)
    assert matches(parse_emission_term("D"), deuterium_line)


def test_a_series_member_of_an_ion_is_not_invented():
    """``C III alpha`` is not a line; reading it as one would fabricate data."""
    assert parse_emission_term("CIIIalpha") is None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_available_choices_are_named_semantically():
    described = describe_available(SAMPLE_LABELS)
    assert "C III" in described and "H_alpha" in described
    assert "elements: H, O, C" in described


def test_an_unparsed_label_is_still_reported():
    assert "Bremsstrahlung_broadband" in describe_available(["Bremsstrahlung_broadband"])


def test_species_are_spelled_the_way_spectroscopy_spells_them():
    assert format_species(Species("C", None, 3)) == "C III"
    assert format_species(Species("H", 2)) == "D"
    assert format_species(Species("He")) == "He"
