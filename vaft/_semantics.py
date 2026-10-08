"""The ``Semantics`` docstring section shared by formulas and processes (#1702).

A ``Semantics`` section names, in the controlled vocabulary, the scientific
quantities a function takes and computes, so the generated ontology can
connect a function it otherwise could not reach::

    Semantics
    ---------
    consumes: plasma_current
    produces: q95

Each line is ``key: term, term, ...`` with the keys of :data:`FIELDS`; either
may be omitted, but not both, and none may repeat.  The section is optional:
add it only where a function's place among the scientific quantities is not
already stated elsewhere (a ``Reduction`` relation, a diagnostic mapping), and
never on a generic helper.

This module checks the *shape* only.  Whether every term names a quantity of
:mod:`vaft.plot.taxonomy` is checked where the vocabulary is known -- by the
ontology generator and its tests -- because ``vaft.formula`` does not import
``vaft.plot``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

#: The keys of a ``Semantics`` section, in the order they are written.
FIELDS: Tuple[str, ...] = ("consumes", "produces")

_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class Semantics:
    """The quantities one function consumes and produces, as vocabulary terms."""

    consumes: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"consumes": list(self.consumes), "produces": list(self.produces)}


def parse_semantics(text: Optional[str]) -> Tuple[Optional[Semantics], Tuple[str, ...]]:
    """The ``Semantics`` section's value and every problem with it.

    ``(None, ())`` when there is no section.  An unknown, repeated or empty
    key, a malformed term, or a section naming nothing yields
    ``(None, errors)``: a half-valid section is never returned.
    """
    if text is None or not text.strip():
        return None, ()
    errors: list[str] = []
    values: dict[str, Tuple[str, ...]] = {}
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        key, sep, rest = line.partition(":")
        key = key.strip()
        if not sep or key not in FIELDS:
            errors.append(f"Semantics line {line.strip()!r} is not one of {', '.join(f'{k}: ...' for k in FIELDS)}")
            continue
        if key in values:
            errors.append(f"Semantics {key} is given twice")
            continue
        terms = tuple(term.strip() for term in rest.split(",") if term.strip())
        if not terms:
            errors.append(f"Semantics {key} is empty")
        for term in terms:
            if not _TERM.match(term):
                errors.append(f"Semantics {key} term {term!r} is not a vocabulary name")
        if len(set(terms)) != len(terms):
            errors.append(f"Semantics {key} repeats a term")
        values[key] = terms
    if not errors and not values:
        errors.append("Semantics names nothing")
    if errors:
        return None, tuple(errors)
    return Semantics(values.get("consumes", ()), values.get("produces", ())), ()
