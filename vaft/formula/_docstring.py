"""The formula docstring contract (issue #248).

Every public function in :mod:`vaft.formula` documents itself in one layout:
a one-line summary, a *Definition* paragraph (math in ``$..$``), numpydoc
``Parameters`` / ``Returns`` items whose description paragraph closes with a
unit tag such as ``[Wb/rad]`` or ``[-]``, and any of a fixed set of
underlined sections -- ``Convention``, ``Physical interpretation``,
``Assumptions``, ``Validity``, ``Limitations``, ``Numerical notes``,
``References``.  The docstring is the single source of truth: the catalog
in :mod:`vaft.formula.catalog` and the generated reference pages are both
read out of it through this module.

The parser itself lives in :mod:`vaft._docstring`, shared with the
processing layer's contract in :mod:`vaft.process._docstring`; this module
declares what a *formula* docstring must contain and binds the engine to it.
Only the standard library is used here, and nothing in ``vaft.formula``
imports this module at package-import time.
"""

from __future__ import annotations

from vaft._docstring import (  # noqa: F401 -- re-exported for the catalog and tests
    DocstringContract,
    ModuleDoc,
    NotationRow,
    ParamDoc,
    ParsedDocstring,
    RaiseDoc,
    Reference,
    ReturnDoc,
    strip_roles,
)
from vaft._docstring import parse_docstring as _parse_docstring
from vaft._docstring import parse_module_docstring as _parse_module_docstring

#: Section titles the contract allows, in the order they are rendered.
SECTION_VOCABULARY: tuple[str, ...] = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Convention",
    "Physical interpretation",
    "Assumptions",
    "Validity",
    "Limitations",
    "Numerical notes",
    "References",
    "Notes",
    "See Also",
    "Examples",
    "Warnings",
)

#: The provenance sections issue #248 adds on top of numpydoc.
CUSTOM_SECTIONS: tuple[str, ...] = (
    "Convention",
    "Physical interpretation",
    "Assumptions",
    "Validity",
    "Limitations",
    "Numerical notes",
)

#: An empirical formula opens its ``Validity`` section with this sentence.
EMPIRICAL_MARKER = "Empirical fit."

#: Section titles a module docstring may use.
MODULE_SECTION_VOCABULARY: tuple[str, ...] = (
    "Notation",
    "Conventions",
    "Notes",
    "References",
    "Examples",
    "See Also",
)

FORMULA_CONTRACT = DocstringContract(
    section_vocabulary=SECTION_VOCABULARY,
    custom_sections=CUSTOM_SECTIONS,
    item_sections=frozenset({"Parameters", "Returns", "Yields", "Raises"}),
    unit_sections=frozenset({"Parameters", "Returns", "Yields"}),
    reference_section="References",
    module_section_vocabulary=MODULE_SECTION_VOCABULARY,
    markers={"empirical": ("Validity", EMPIRICAL_MARKER)},
    presence={"convention_sensitive": "Convention"},
)


def parse_docstring(text: str | None) -> ParsedDocstring:
    """Parse a function docstring against the formula contract; never raises."""
    return _parse_docstring(text, FORMULA_CONTRACT)


def parse_module_docstring(text: str | None) -> ModuleDoc:
    """Parse a submodule docstring into title, overview and notation table."""
    return _parse_module_docstring(text, FORMULA_CONTRACT)
