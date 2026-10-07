"""Atomic identity: elements, isotopes, charge states and how species are written.

The canonical owner of VAFT's atomic vocabulary (#1711).  ``C III`` names an
*ionization stage*; ``C2+`` names the *charge state* of the same ion;
``carbon`` names the *element* of both.  ``D`` is not an element at all -- it
is hydrogen with a mass number.  This module keeps those apart and converts
between the notations that different diagnostics and communities prefer.

It owns **identity only**.  Densities, populations, solver species indices and
impurity fractions are higher-level models that consume an
:class:`AtomicSpecies`: the composition of :mod:`vaft.process.impurity`
(#1565), the canonical multi-species state of #1567, and the solver-specific
projections built on it.  Spectral-line identity lives one level up in
:mod:`vaft.data.spectroscopy`; atomic equations live in
:mod:`vaft.formula.atomic`.

Nothing here imports OMAS, an IMAS runtime, plotting or a solver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "ATOMIC_NUMBERS",
    "ELEMENT_NAMES",
    "ISOTOPE_NAMES",
    "STANDARD_ATOMIC_WEIGHTS",
    "AtomicSpecies",
    "charge_state_of",
    "format_species",
    "ionization_stage_of",
    "parse_species",
]

#: Nuclear charge by element symbol.  ``D`` and ``T`` appear because ADAS and
#: the plasma-composition layers key species that way (see
#: :func:`vaft.process.atomic.normalize_atomic_symbol`); :class:`AtomicSpecies`
#: records them as hydrogen with a mass number instead.
ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "D": 1, "T": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Al": 13,
    "Si": 14, "S": 16, "Cl": 17, "Ar": 18, "Ca": 20, "Ti": 22,
    "Fe": 26, "Ni": 28, "Kr": 36, "Mo": 42, "Xe": 54, "W": 74,
}

#: Standard atomic weight [u] of every symbol in :data:`ATOMIC_NUMBERS`: the
#: IUPAC 2021 abridged values, and the isotope masses for ``D`` and ``T``.
#: :data:`vaft.data.synthetic_kinetic_profiles.ION_SPECIES` is derived from it.
STANDARD_ATOMIC_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "H": 1.00784, "D": 2.01410, "T": 3.01605, "He": 4.0026, "Li": 6.94,
    "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.180, "Al": 26.982, "Si": 28.085, "S": 32.06,
    "Cl": 35.45, "Ar": 39.95, "Ca": 40.078, "Ti": 47.867, "Fe": 55.845,
    "Ni": 58.693, "Kr": 83.798, "Mo": 95.95, "Xe": 131.29, "W": 183.84,
})

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
class AtomicSpecies:
    """An element, optionally narrowed to one isotope and one charge.

    ``mass_number`` and ``ionization_stage`` are ``None`` when unspecified,
    which is what a label that names neither leaves behind.  ``charge_state``
    is derived rather than stored, so it can never disagree with the stage.
    Identity only: no density, population or solver index.
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
    ) -> "AtomicSpecies":
        """Build from the ``C2+`` spelling rather than the ``C III`` one."""
        return cls(element, mass_number, ionization_stage_of(charge_state))


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


def _species_parts(folded: str) -> tuple[AtomicSpecies, bool] | None:
    """Parse a separator-free species term, reporting whether an isotope was named.

    The flag is what lets a *selector* mean protium when it says ``H`` while a
    stored *label* that says ``H`` claims no isotope at all.
    """
    named = ELEMENT_NAMES.get(folded.lower())
    if named is not None:
        if named in ("D", "T"):
            return AtomicSpecies("H", ISOTOPE_NAMES[named]), True
        return AtomicSpecies(named), False

    split = _split_element(folded)
    if split is None:
        return None
    element, mass_number, rest = split
    marked = mass_number is not None
    if not rest:
        return AtomicSpecies(element, mass_number), marked

    stage = _parse_roman(rest)
    if stage is None:
        charge = _parse_charge_suffix(rest)
        stage = None if charge is None else ionization_stage_of(charge)
    if stage is None:
        return None
    return AtomicSpecies(element, mass_number, stage), marked


def _fold_species(text: Any) -> str | None:
    """Normalize a term (NFC, separators dropped), or ``None`` if it is not text."""
    import unicodedata

    if text is None or isinstance(text, bool) or not isinstance(text, (str, bytes)):
        return None
    raw = text.decode("utf-8", "ignore") if isinstance(text, bytes) else text
    folded = re.sub(r"[\s_\-]+", "", unicodedata.normalize("NFC", raw).strip())
    return folded or None


def parse_species(text: Any) -> AtomicSpecies | None:
    """Resolve a species term to an identity, or ``None`` if it is not one.

    Accepts a full element name in any case (``carbon``, ``Deuterium``), a
    symbol with a spectroscopic stage (``CIII``, ``C III``, ``WI``), and a
    symbol with a charge state in either ordering (``C2+``, ``C+2``, ``C+``,
    ``C0``).  ``CIII`` and ``C2+`` resolve to the same species; ``CIII`` and
    ``C3+`` deliberately do not.

    This reads a *selector*, so bare hydrogen means protium: ``H`` will not go
    on to match a deuterium line.  A spectral-series term such as ``H_alpha``
    names a line, not a species; :func:`vaft.data.spectroscopy.parse_emission_term`
    reads it.
    """
    folded = _fold_species(text)
    if folded is None:
        return None
    parts = _species_parts(folded)
    if parts is None:
        return None
    species, marked = parts
    if species.element == "H" and not marked:
        species = AtomicSpecies("H", 1, species.ionization_stage)
    return species


def format_species(species: AtomicSpecies) -> str:
    """Spell a species the way spectroscopy does: ``C III``, ``D``, ``He``."""
    symbol = species.element
    if species.element == "H" and species.mass_number in (2, 3):
        symbol = {2: "D", 3: "T"}[species.mass_number]
    if species.ionization_stage is None:
        return symbol
    return f"{symbol} {_format_roman(species.ionization_stage)}"
