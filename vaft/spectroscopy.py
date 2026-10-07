"""Deprecated alias of :mod:`vaft.data.atomic` and :mod:`vaft.data.spectroscopy`.

Atomic identity (elements, isotopes, charge states, species notation) moved to
:mod:`vaft.data.atomic` and spectral-line identity (the ``processed_line.label``
codec, ``emission=`` terms, line matching) to :mod:`vaft.data.spectroscopy`
(#1711): they are representation-neutral data semantics, not a top-level
domain.  Importing this module warns and re-exports the old names with their
old behaviour; ``Species`` is :class:`vaft.data.atomic.AtomicSpecies` and
``LineIdentity`` is :class:`vaft.data.spectroscopy.SpectralLineIdentity`.
Scheduled for removal two minor releases after the move ships (current
version 0.7.1; remove in 0.10.0).
"""

from __future__ import annotations

import warnings
from typing import Any

from vaft.data.atomic import (  # noqa: F401
    ATOMIC_NUMBERS,
    ELEMENT_NAMES,
    ISOTOPE_NAMES,
    charge_state_of,
    format_species,
    ionization_stage_of,
)
from vaft.data.atomic import AtomicSpecies as Species
from vaft.data.spectroscopy import (  # noqa: F401
    SERIES_NAMES,
    describe_available,
    matches,
    parse_emission_term,
    parse_line_label,
)
from vaft.data.spectroscopy import SpectralLineIdentity as LineIdentity

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


def parse_species(text: Any) -> Species | None:
    """Deprecated: the species of any ``emission=`` term, series terms included.

    The old behaviour, kept for this alias: ``H_alpha`` resolves to protium.
    :func:`vaft.data.atomic.parse_species` reads species terms only.
    """
    identity = parse_emission_term(text)
    return None if identity is None else identity.species


warnings.warn(
    "vaft.spectroscopy is deprecated and will be removed in vaft 0.10.0; import "
    "vaft.data.atomic (elements, species) or vaft.data.spectroscopy (spectral lines) instead",
    DeprecationWarning,
    stacklevel=2,
)
