"""Spectral-line identity: what a line *is*, and the IMAS ``processed_line.label`` codec.

The IMAS Data Dictionary gives species metadata nowhere to live.  Under
``spectrometer_uv.channel[:].processed_line[:]`` the only fields are
``label``, ``wavelength_central``, ``intensity`` and ``radiance`` -- no
element, no ion, no isotope, no transition.  The Dictionary's answer is to
encode the species *in the label string*:

    "String identifying the processed line. To avoid ambiguities, the
    following syntax is used : element with ionization state_wavelength in
    Angstrom (e.g. WI_4000)"

So parsing the label is the Dictionary's own intended mechanism, not a
workaround, and it is why this module introduces **no table of lines**: it
reads what a machine's data declares rather than deciding in advance which
lines exist.  The only vocabulary here is the Greek series letters; elements,
isotopes and charge states are :mod:`vaft.data.atomic`'s.

This module says what a line or selector *means*.  Which traces a plot
selects and how it presents them stays in :mod:`vaft.plot.backend.recipes`;
the labels a machine writes stay in :mod:`vaft.machine_mapping`.  Nothing here
imports OMAS, an IMAS runtime or plotting.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from vaft.data.atomic import AtomicSpecies, _species_parts, format_species

__all__ = [
    "SERIES_NAMES",
    "SpectralLineIdentity",
    "describe_available",
    "matches",
    "parse_emission_term",
    "parse_line_label",
]

#: Series members of a hydrogen-like spectrum, by their Greek letter.
SERIES_NAMES: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon")

_GREEK_SERIES = {
    "α": "alpha", "β": "beta", "γ": "gamma",
    "δ": "delta", "ε": "epsilon",
}


@dataclass(frozen=True)
class SpectralLineIdentity:
    """What one spectral line is: a species, a series member, a wavelength.

    Also used for a *selector*, where a ``None`` field means "unconstrained"
    rather than "unknown"; :func:`matches` reads the two senses correctly.
    ``label`` keeps the string this was parsed from so a line whose label
    follows no convention is still reachable by its literal name.
    """

    species: AtomicSpecies
    series: str | None = None
    wavelength_angstrom: float | None = None
    label: str = ""


def _fold(text: str) -> str:
    """Drop separators and map Greek series letters to their names."""
    folded = "".join(_GREEK_SERIES.get(char, char) for char in text.strip())
    return re.sub(r"[\s_\-]+", "", folded)


def _parse(text: Any, *, as_selector: bool) -> SpectralLineIdentity | None:
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
        species = AtomicSpecies("H", 1, species.ionization_stage)
    return SpectralLineIdentity(species, series=series, label=str(raw))


def parse_emission_term(term: Any) -> SpectralLineIdentity | None:
    """Resolve a user-facing ``emission=`` term to a selector identity.

    A term names a species (``CIII``, ``Carbon``) or one series member of one
    isotope (``H_alpha``, ``D-alpha``, ``Hα``).  Unset fields mean
    "unconstrained", so ``Carbon`` matches every carbon line.
    """
    return _parse(term, as_selector=True)


def parse_line_label(label: Any) -> SpectralLineIdentity | None:
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
    return SpectralLineIdentity(identity.species, identity.series, wavelength, label=str(raw))


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


def matches(term: SpectralLineIdentity, identity: SpectralLineIdentity) -> bool:
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
        element = format_species(AtomicSpecies(identity.species.element, identity.species.mass_number))
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
