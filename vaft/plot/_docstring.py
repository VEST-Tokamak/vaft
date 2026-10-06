"""The plot docstring contract (issue #1505).

A formula answers *what is this quantity?*, a processing routine *how is this
input turned into this output?*  A plot answers a third question: *what does
this visualization show, how should it be read, and what scientific questions
can it support?*  The three share one parser (:mod:`vaft._docstring`) but not a
schema.  A canonical renderer -- the ``@renderer``-registered function
``vaft.plot.<name>`` -- documents itself as a one-line summary, optional prose,
and any of the sections below:

``Interpretation``
    What the figure exposes, which questions it is used for, which features a
    reader inspects, and how the drawn representation relates to the physical
    quantity.  Required once a plot adopts the contract.
``Options``
    The options that change the scientific representation or its reading --
    radial coordinate, normalization, uncertainty and validity display,
    reference lines, contour style, overlays.  The option *vocabulary* is
    defined structurally by the plotting layer (:mod:`vaft.plot.controls`,
    :class:`~vaft.plot.discovery.PlotCapability`); this section explains what
    the choices *mean* and never re-enumerates them.  Generic styling keywords
    are not listed.
``Convention``
    Sign, unit or coordinate conventions the reader must know to read a value.
``Applicability``
    The data, machine or regime the plot is meaningful for.
``Limitations``
    What must *not* be concluded from the plot alone.

plus numpydoc ``Parameters`` / ``Returns`` / ``Raises`` (no unit tags: a
renderer's arguments are a view model and presentation switches, not physical
quantities), ``References`` (``.. [1] text``), ``Notes``, ``See Also``,
``Examples`` and ``Warnings``.

The docstring is the single source of human-facing scientific guidance.
:class:`~vaft.plot.registry.PlotSpec` keeps answering the structural questions
(what model, which DD paths, which options); :func:`plot_documentation` parses
the renderer's docstring into a :class:`PlotDocumentation` that the generated
reference pages and GUI help surfaces both read, so neither keeps its own copy
of a plot's scientific description.  Only the standard library and
:mod:`vaft._docstring` are imported here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from vaft._docstring import (  # noqa: F401 -- re-exported for the docs catalog and tests
    DocstringContract,
    ParamDoc,
    ParsedDocstring,
    RaiseDoc,
    Reference,
    ReturnDoc,
    strip_roles,
)
from vaft._docstring import parse_docstring as _parse_docstring

__all__ = [
    "CUSTOM_SECTIONS",
    "PLOT_CONTRACT",
    "PlotDocumentation",
    "SECTION_VOCABULARY",
    "parse_docstring",
    "plot_documentation",
]

#: Section titles the contract allows, in the order they are rendered.
SECTION_VOCABULARY: tuple[str, ...] = (
    "Parameters",
    "Returns",
    "Raises",
    "Interpretation",
    "Options",
    "Convention",
    "Applicability",
    "Limitations",
    "References",
    "Notes",
    "See Also",
    "Examples",
    "Warnings",
)

#: The sections issue #1505 adds on top of numpydoc.
CUSTOM_SECTIONS: tuple[str, ...] = (
    "Interpretation",
    "Options",
    "Convention",
    "Applicability",
    "Limitations",
)

#: The section an adopted plot must carry.
REQUIRED_SECTION = "Interpretation"

PLOT_CONTRACT = DocstringContract(
    section_vocabulary=SECTION_VOCABULARY,
    custom_sections=CUSTOM_SECTIONS,
    item_sections=frozenset({"Parameters", "Returns", "Raises"}),
    unit_sections=frozenset(),
    reference_section="References",
    module_section_vocabulary=("Notes", "References", "Examples", "See Also"),
    presence={
        "has_interpretation": "Interpretation",
        "has_options": "Options",
        "has_limitations": "Limitations",
        "convention_sensitive": "Convention",
    },
)

#: Sections a structured consumer renders from ``parameters`` / ``returns`` /
#: ``raises`` / ``references`` rather than from their raw text.
_ITEM_SECTIONS = ("Parameters", "Returns", "Raises", "References")

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def parse_docstring(text: str | None) -> ParsedDocstring:
    """Parse a renderer docstring against the plot contract; never raises."""
    return _parse_docstring(text, PLOT_CONTRACT)


@dataclass(frozen=True)
class PlotDocumentation:
    """One canonical plot's docstring, parsed for documentation and GUI consumers.

    ``sections`` holds every section present as ``(title, text)`` in the
    contract's vocabulary order; the item sections are also parsed into
    ``parameters``, ``returns``, ``raises`` and ``references``.  ``errors``
    lists every way the docstring falls short of the contract, and
    ``conforming`` is true when there are none.  A GUI may show only
    :attr:`summary`, :attr:`interpretation` and :attr:`limitations`; the
    reference pages render everything.
    """

    name: str
    summary: str
    description: str
    sections: tuple[tuple[str, str], ...]
    parameters: tuple[ParamDoc, ...]
    returns: tuple[ReturnDoc, ...]
    raises: tuple[RaiseDoc, ...]
    references: tuple[Reference, ...]
    errors: tuple[str, ...]

    def section(self, title: str) -> str | None:
        """Text of the section called ``title``, or ``None``."""
        for name, text in self.sections:
            if name == title:
                return text
        return None

    @property
    def interpretation(self) -> str | None:
        """What the plot shows and which questions it supports."""
        return self.section("Interpretation")

    @property
    def options(self) -> str | None:
        """What the representation-changing options mean."""
        return self.section("Options")

    @property
    def convention(self) -> str | None:
        """Sign, unit or coordinate conventions needed to read a value."""
        return self.section("Convention")

    @property
    def applicability(self) -> str | None:
        """The data, machine or regime the plot is meaningful for."""
        return self.section("Applicability")

    @property
    def limitations(self) -> str | None:
        """What must not be concluded from the plot alone."""
        return self.section("Limitations")

    @property
    def see_also(self) -> str | None:
        """Related plots to inspect next."""
        return self.section("See Also")

    @property
    def conforming(self) -> bool:
        """The docstring meets the plot contract."""
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        """Plain data with Sphinx roles made literal, for the site snapshot and GUI panels."""
        return {
            "name": self.name,
            "summary": strip_roles(self.summary),
            "description": strip_roles(self.description),
            "sections": [
                {"title": title, "text": strip_roles(text)}
                for title, text in self.sections
                if title not in _ITEM_SECTIONS
            ],
            "parameters": [
                {"name": item.name, "type": item.type, "description": strip_roles(item.description)}
                for item in self.parameters
            ],
            "returns": [
                {"name": item.name, "type": item.type, "description": strip_roles(item.description)}
                for item in self.returns
            ],
            "raises": [
                {"type": item.type, "description": strip_roles(item.description)}
                for item in self.raises
            ],
            "references": [
                {"label": ref.label, "text": strip_roles(ref.text)} for ref in self.references
            ],
            "conforming": self.conforming,
            "errors": list(self.errors),
        }


def _structural_violations(parsed: ParsedDocstring, function: Any) -> list[str]:
    """What the contract requires of an adopted plot, beyond parsing cleanly.

    A one-line docstring parses without error, so ``conforming`` has to mean
    more than that: a summary sentence, an ``Interpretation`` section, and --
    when ``Parameters`` is present -- one item per named signature parameter
    in signature order.  Which plots must also state ``Limitations``, and that
    ``Options`` explains rather than re-enumerates a vocabulary, are policy and
    stay in ``test/test_plot_docstrings.py``.
    """
    if not parsed.summary:
        return ["missing summary line"]
    violations: list[str] = []
    if not parsed.summary.endswith("."):
        violations.append("the summary line must end with a period")
    if parsed.section(REQUIRED_SECTION) is None:
        violations.append(f"missing {REQUIRED_SECTION} section")
    if parsed.section("Parameters") is not None:
        expected = [
            p.name for p in inspect.signature(function).parameters.values() if p.kind not in _VARIADIC
        ]
        documented = [name.strip() for item in parsed.parameters for name in item.name.split(",")]
        if documented != expected:
            violations.append(f"Parameters documents {documented} but the signature has {expected}")
    return violations


def documentation_of(name: str, function: Any) -> PlotDocumentation:
    """Parse ``function``'s docstring as the documentation of plot ``name``."""
    target = inspect.unwrap(function)
    parsed = parse_docstring(target.__doc__)
    errors = list(dict.fromkeys([*parsed.errors, *_structural_violations(parsed, target)]))
    return PlotDocumentation(
        name=name,
        summary=parsed.summary,
        description=parsed.description,
        sections=parsed.sections,
        parameters=parsed.parameters,
        returns=parsed.returns,
        raises=parsed.raises,
        references=parsed.references,
        errors=tuple(errors),
    )


@lru_cache(maxsize=None)
def plot_documentation(name: str) -> PlotDocumentation:
    """The parsed docstring of canonical plot ``name``'s registered renderer.

    The renderer ``vaft.plot.<name>`` is the source; nothing else describes a
    plot's scientific meaning.  Raises ``KeyError`` for a name the registry
    does not hold, exactly as :func:`vaft.plot.registry.get_spec`.
    """
    import vaft.plot  # noqa: F401 -- importing the package registers every renderer

    from .registry import get_spec

    spec = get_spec(name)
    return documentation_of(spec.name, spec.renderer)
