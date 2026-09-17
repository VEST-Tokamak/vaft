"""Atomic and spectroscopic identity: what a species or a spectral line *is*.

A spectral line carries more physics than its name suggests, and the concepts
are routinely conflated.  ``C III`` names an *ionization stage*; ``C2+`` names
the *charge state* of the same ion; ``carbon`` names the *element* of both.
``D`` is not an element at all -- it is hydrogen with a mass number.  This
module keeps those apart and converts between the notations that different
diagnostics and communities happen to prefer, so nothing downstream has to.

It exists because the IMAS Data Dictionary gives species metadata nowhere to
live.  Under ``spectrometer_uv.channel[:].processed_line[:]`` the only fields
are ``label``, ``wavelength_central``, ``intensity`` and ``radiance`` -- no
element, no ion, no isotope, no transition.  The Dictionary's answer is to
encode the species *in the label string*:

    "String identifying the processed line. To avoid ambiguities, the
    following syntax is used : element with ionization state_wavelength in
    Angstrom (e.g. WI_4000)"

So parsing the label is the Dictionary's own intended mechanism, not a
workaround, and it is why this module introduces **no table of species**: it
reads what a machine's data declares rather than deciding in advance which
lines exist.  The only tables here are the chemical elements, the three
hydrogen isotopes, and Greek series letters -- vocabulary, not physics.

Nothing here imports OMAS or touches an ODS, so ``vaft.plot`` may use it
directly; :mod:`vaft.plot.backend.recipes` reads the labels and this module
decides what they mean.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "ATOMIC_NUMBERS",
    "ELEMENT_NAMES",
    "ISOTOPE_NAMES",
    "SERIES_NAMES",
    "LineIdentity",
    "Species",
    "charge_state_of",
    "describe_available",
    "format_species",
    "ionization_stage_of",
    "matches",
    "parse_emission_term",
    "parse_line_label",
    "parse_species",
]

#: Nuclear charge by element symbol.  ``D`` and ``T`` appear because ADAS keys
#: species that way (see :func:`vaft.process.atomic.normalize_atomic_symbol`);
#: :class:`Species` records them as hydrogen with a mass number instead.
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "D": 1, "T": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Al": 13,
    "Si": 14, "S": 16, "Cl": 17, "Ar": 18, "Ca": 20, "Ti": 22,
    "Fe": 26, "Ni": 28, "Kr": 36, "Mo": 42, "Xe": 54, "W": 74,
}

#: Full English element names, lowercased.
ELEMENT_NAMES: dict[str, str] = {
    "hydrogen": "H", "deuterium": "D", "tritium": "T", "helium": "He",
    "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C",
    "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne",
    "aluminium": "Al", "aluminum": "Al", "silicon": "Si", "sulfur": "S",
    "sulphur": "S", "chlorine": "Cl", "argon": "Ar", "calcium": "Ca",
    "titanium": "Ti", "iron": "Fe", "nickel": "Ni", "krypton": "Kr",
    "molybdenum": "Mo", "xenon": "Xe", "tungsten": "W",
}

#: The hydrogen isotopes, as ``symbol -> mass number``.  An isotope is a
#: property of hydrogen, never an element of its own.
ISOTOPE_NAMES: dict[str, int] = {"H": 1, "D": 2, "T": 3}

#: Series members of a hydrogen-like spectrum, by their Greek letter.
SERIES_NAMES: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon")

_GREEK_SERIES = {
    "α": "alpha", "β": "beta", "γ": "gamma",
    "δ": "delta", "ε": "epsilon",
}

_ROMAN_VALUES = (
    ("XL", 40), ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1),
)


def charge_state_of(ionization_stage: int) -> int:
    """The ionic charge of a spectroscopic stage: ``C III`` -> ``C2+``.

    Spectroscopic notation counts from the neutral atom as ``I``, so the two
    differ by exactly one.  Keeping the conversion in one place is what stops
    ``C III`` and ``C3+`` -- genuinely different ions -- from being treated as
    the same species somewhere downstream.
    """
    if ionization_stage < 1:
        raise ValueError(f"ionization stage counts from I; got {ionization_stage}")
    return ionization_stage - 1


def ionization_stage_of(charge_state: int) -> int:
    """The spectroscopic stage of an ionic charge: ``C2+`` -> ``C III``."""
    if charge_state < 0:
        raise ValueError(f"charge state cannot be negative; got {charge_state}")
    return charge_state + 1


@dataclass(frozen=True)
class Species:
    """An element, optionally narrowed to one isotope and one charge.

    ``mass_number`` and ``ionization_stage`` are ``None`` when unspecified,
    which is what a label that names neither leaves behind.  ``charge_state``
    is derived rather than stored, so it can never disagree with the stage.
    """

    element: str
    mass_number: int | None = None
    ionization_stage: int | None = None

    @property
    def charge_state(self) -> int | None:
        """The ionic charge, or ``None`` when the stage is unspecified."""
        if self.ionization_stage is None:
            return None
        return charge_state_of(self.ionization_stage)

    @property
    def atomic_number(self) -> int | None:
        """Nuclear charge, or ``None`` for an element outside the table."""
        return ATOMIC_NUMBERS.get(self.element)

    @classmethod
    def from_charge_state(
        cls, element: str, charge_state: int, mass_number: int | None = None
    ) -> "Species":
        """Build from the ``C2+`` spelling rather than the ``C III`` one."""
        return cls(element, mass_number, ionization_stage_of(charge_state))


@dataclass(frozen=True)
class LineIdentity:
    """What one spectral line is: a species, a series member, a wavelength.

    Also used for a *selector*, where a ``None`` field means "unconstrained"
    rather than "unknown"; :func:`matches` reads the two senses correctly.
    ``label`` keeps the string this was parsed from so a line whose label
    follows no convention is still reachable by its literal name.
    """

    species: Species
    series: str | None = None
    wavelength_angstrom: float | None = None
    label: str = ""


def _fold(text: str) -> str:
    """Drop separators and map Greek series letters to their names."""
    folded = "".join(_GREEK_SERIES.get(char, char) for char in text.strip())
    return re.sub(r"[\s_\-]+", "", folded)


def _parse_roman(text: str) -> int | None:
    """Parse an uppercase Roman numeral, or ``None`` if it is not one."""
    if not text or any(char not in "IVXL" for char in text):
        return None
    total, rest = 0, text
    for symbol, value in _ROMAN_VALUES:
        while rest.startswith(symbol):
            total += value
            rest = rest[len(symbol) :]
    # Round-tripping rejects malformed numerals such as "IIII" or "VV".
    return total if total and _format_roman(total) == text else None


def _format_roman(value: int) -> str:
    out, rest = "", value
    for symbol, amount in _ROMAN_VALUES:
        while rest >= amount:
            out += symbol
            rest -= amount
    return out


def _parse_charge_suffix(text: str) -> int | None:
    """Parse ``2+``, ``+2``, ``+`` or ``0`` as an ionic charge."""
    if text == "0":
        return 0
    match = re.fullmatch(r"(\d*)\+(\d*)", text)
    if match is None:
        return None
    before, after = match.group(1), match.group(2)
    if before and after:
        return None
    digits = before or after
    return int(digits) if digits else 1


def _split_element(text: str) -> tuple[str, int | None, str] | None:
    """Split a leading element or isotope symbol from the rest of a term.

    The symbol is matched **case-sensitively**, because case is what separates
    ``Ni`` (nickel) from ``NI`` (neutral nitrogen) and ``WI`` (neutral
    tungsten, the Data Dictionary's own example) from a two-letter element.
    """
    for width in (2, 1):
        head, rest = text[:width], text[width:]
        if head in ATOMIC_NUMBERS:
            if head in ("D", "T"):
                return "H", ISOTOPE_NAMES[head], rest
            return head, None, rest
    return None


def _species_parts(folded: str) -> tuple[Species, bool] | None:
    """Parse a species, reporting whether an isotope was explicitly named.

    The flag is what lets a *selector* mean protium when it says ``H`` while a
    stored *label* that says ``H`` claims no isotope at all.  Without it the
    two could not be told apart, and ``emission="D"`` would match a line the
    data only ever called hydrogen.
    """
    named = ELEMENT_NAMES.get(folded.lower())
    if named is not None:
        if named in ("D", "T"):
            return Species("H", ISOTOPE_NAMES[named]), True
        return Species(named), False

    split = _split_element(folded)
    if split is None:
        return None
    element, mass_number, rest = split
    marked = mass_number is not None
    if not rest:
        return Species(element, mass_number), marked

    stage = _parse_roman(rest)
    if stage is None:
        charge = _parse_charge_suffix(rest)
        stage = None if charge is None else ionization_stage_of(charge)
    if stage is None:
        return None
    return Species(element, mass_number, stage), marked


def parse_species(text: Any) -> Species | None:
    """Resolve a species term to an identity, or ``None`` if it is not one.

    Accepts a full element name in any case (``carbon``, ``Deuterium``), a
    symbol with a spectroscopic stage (``CIII``, ``C III``, ``WI``), and a
    symbol with a charge state in either ordering (``C2+``, ``C+2``, ``C+``,
    ``C0``).  ``CIII`` and ``C2+`` resolve to the same species; ``CIII`` and
    ``C3+`` deliberately do not.

    This reads a *selector*, so bare hydrogen means protium: ``H`` will not go
    on to match a deuterium line.
    """
    parsed = _parse(text, as_selector=True)
    return None if parsed is None else parsed.species


def _parse(text: Any, *, as_selector: bool) -> LineIdentity | None:
    """Shared parse for selector terms and stored labels."""
    if text is None or isinstance(text, bool) or not isinstance(text, (str, bytes)):
        return None
    raw = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
    folded = _fold(unicodedata.normalize("NFC", raw))
    if not folded:
        return None

    series = None
    body = folded
    for candidate in SERIES_NAMES:
        if len(folded) > len(candidate) and folded.lower().endswith(candidate):
            series, body = candidate, folded[: -len(candidate)]
            break

    parts = _species_parts(body)
    if parts is None:
        return None
    species, marked = parts
    if series is not None and species.ionization_stage is not None:
        # "C III alpha" is not a thing: a series member belongs to a neutral
        # hydrogen-like spectrum, and reading it as one would invent a line.
        return None
    if as_selector and species.element == "H" and not marked:
        species = Species("H", 1, species.ionization_stage)
    return LineIdentity(species, series=series, label=str(raw))


def parse_emission_term(term: Any) -> LineIdentity | None:
    """Resolve a user-facing ``emission=`` term to a selector identity.

    A term names a species (``CIII``, ``Carbon``) or one series member of one
    isotope (``H_alpha``, ``D-alpha``, ``Hα``).  Unset fields mean
    "unconstrained", so ``Carbon`` matches every carbon line.
    """
    return _parse(term, as_selector=True)


def parse_line_label(label: Any) -> LineIdentity | None:
    """Parse a stored ``processed_line.label`` into an identity.

    Handles the Data Dictionary's ``WI_4000`` form and the series form VEST
    writes for hydrogen (``H-alpha_6563``), both with the wavelength in
    Angstrom.  Returns ``None`` for a label following neither convention; the
    caller then falls back to matching that label literally rather than
    guessing at its species.
    """
    if label is None or not isinstance(label, (str, bytes)):
        return None
    raw = label.decode("utf-8", "ignore") if isinstance(label, bytes) else label
    text = unicodedata.normalize("NFC", raw).strip()
    if not text:
        return None

    head, wavelength = text, None
    if "_" in text:
        candidate_head, _, tail = text.rpartition("_")
        if re.fullmatch(r"\d+(?:\.\d+)?", tail) and candidate_head:
            head, wavelength = candidate_head, float(tail)

    identity = _parse(head, as_selector=False)
    if identity is None:
        return None
    return LineIdentity(identity.species, identity.series, wavelength, label=str(raw))


def _isotope_agrees(wanted: int | None, found: int | None) -> bool:
    """Whether a selector's isotope admits the one a label recorded.

    An unmarked line is conventionally the light isotope, so ``H`` admits it;
    ``D`` and ``T`` never do -- asking for deuterium must not return a trace
    the data only ever called hydrogen.
    """
    if wanted is None:
        return True
    if found is None:
        return wanted == 1
    return wanted == found


def matches(term: LineIdentity, identity: LineIdentity) -> bool:
    """Whether a parsed selector selects a parsed line.

    Every field the selector leaves unset is unconstrained, so ``Carbon``
    matches C II and C III alike while ``CIII`` matches only its own stage.
    """
    if term.species.element != identity.species.element:
        return False
    if not _isotope_agrees(term.species.mass_number, identity.species.mass_number):
        return False
    if (
        term.species.ionization_stage is not None
        and term.species.ionization_stage != identity.species.ionization_stage
    ):
        return False
    if term.series is not None and term.series != identity.series:
        return False
    return True


def format_species(species: Species) -> str:
    """Spell a species the way spectroscopy does: ``C III``, ``D``, ``He``."""
    symbol = species.element
    if species.element == "H" and species.mass_number in (2, 3):
        symbol = {2: "D", 3: "T"}[species.mass_number]
    if species.ionization_stage is None:
        return symbol
    return f"{symbol} {_format_roman(species.ionization_stage)}"


def describe_available(labels: Iterable[str]) -> str:
    """Name the semantic choices a set of stored labels offers.

    Error messages built on this report species and elements a caller can
    actually ask for, rather than only the raw labels or a list of indices.
    """
    elements: list[str] = []
    species: list[str] = []
    literal: list[str] = []
    for label in labels:
        identity = parse_line_label(label)
        if identity is None:
            if label and label not in literal:
                literal.append(str(label))
            continue
        element = format_species(Species(identity.species.element, identity.species.mass_number))
        if element not in elements:
            elements.append(element)
        name = format_species(identity.species)
        if identity.series is not None:
            name = f"{element}_{identity.series}"
        if name not in species:
            species.append(name)
    parts = []
    if species:
        parts.append("lines and species: " + ", ".join(species))
    if elements:
        parts.append("elements: " + ", ".join(elements))
    if literal:
        parts.append("unparsed labels: " + ", ".join(literal))
    return "; ".join(parts) if parts else "none"
