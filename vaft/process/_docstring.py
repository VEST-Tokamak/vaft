"""The processing docstring contract (issue #252).

A formula answers *what is this quantity, where is it from, when is it
valid?*  A processing routine answers *how is this input turned into this
output?* -- so the two share a parser (:mod:`vaft._docstring`) but not a
schema.  Every public function in :mod:`vaft.process` documents itself as a
one-line summary, prose, numpydoc ``Parameters`` / ``Returns`` items whose
description paragraph closes with a unit tag (``[V]``, ``[s]``, ``[Wb/rad]``,
``[-]``, ``[any]``), and any of the sections below:

``Processing steps``
    The transformations in order, where they change what the output means.
``Input semantics`` / ``Output semantics``
    Where the data sits in the processing chain, as independent descriptors
    (raw or calibrated; filtered, baseline-subtracted, repaired; integrated;
    normalized; diagnostic-native or equilibrium-mapped; measured, fitted,
    reconstructed or synthetic) rather than one flat state.  Required only
    where the state genuinely changes.
``Defaults``
    Every default that materially affects the result, each classified:
    physical constant, literature value, diagnostic calibration, empirical
    estimate, validated workflow default, machine-specific setting,
    acquisition-era policy, legacy compatibility value, numerical convenience.
``Convention``
    Sign, coordinate, COCOS, Wb vs Wb/rad, causal vs zero-phase, and so on.
``Assumptions``
    What the implementation takes for granted.
``Applicability``
    Opens with :data:`MACHINE_INDEPENDENT` or :data:`VEST_SPECIFIC`, then
    the data, diagnostic, acquisition era or regime the routine is for.
``Limitations``
    Where the result is undefined, unreliable or refused.
``Provenance``
    ``.. [label] text`` entries, exactly as a formula's ``References`` --
    but a label may point at ``VEST_IMPAProcessing.m``, a ``vest.yaml`` key,
    a VAFT issue, a validated viewer, or a paper.  For a ported workflow the
    first kind is worth more than the last.

The catalog in :mod:`vaft.process.catalog` and the generated reference pages
are read out of these docstrings; nothing else describes a processing
routine.  Nothing in ``vaft.process`` imports this module at package-import
time.
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
    "Processing steps",
    "Input semantics",
    "Output semantics",
    "Defaults",
    "Convention",
    "Assumptions",
    "Applicability",
    "Limitations",
    "Provenance",
    "Notes",
    "See Also",
    "Examples",
    "Warnings",
)

#: The sections issue #252 adds on top of numpydoc.
CUSTOM_SECTIONS: tuple[str, ...] = (
    "Processing steps",
    "Input semantics",
    "Output semantics",
    "Defaults",
    "Convention",
    "Assumptions",
    "Applicability",
    "Limitations",
    "Provenance",
)

#: ``Applicability`` opens with exactly one of these.
MACHINE_INDEPENDENT = "Machine-independent."
VEST_SPECIFIC = "VEST-specific."

#: Section titles a module docstring may use.
MODULE_SECTION_VOCABULARY: tuple[str, ...] = (
    "Notation",
    "Conventions",
    "Notes",
    "Provenance",
    "Examples",
    "See Also",
)

PROCESS_CONTRACT = DocstringContract(
    section_vocabulary=SECTION_VOCABULARY,
    custom_sections=CUSTOM_SECTIONS,
    item_sections=frozenset({"Parameters", "Returns", "Yields", "Raises"}),
    unit_sections=frozenset({"Parameters", "Returns", "Yields"}),
    reference_section="Provenance",
    module_section_vocabulary=MODULE_SECTION_VOCABULARY,
    markers={
        "machine_independent": ("Applicability", MACHINE_INDEPENDENT),
        "vest_specific": ("Applicability", VEST_SPECIFIC),
    },
    presence={
        "convention_sensitive": "Convention",
        "has_processing_steps": "Processing steps",
        "has_input_semantics": "Input semantics",
        "has_output_semantics": "Output semantics",
    },
)


def machine_scope(parsed: ParsedDocstring) -> str | None:
    """``"independent"``, ``"vest"`` or ``None`` when ``Applicability`` declares neither."""
    if parsed.flags.get("machine_independent"):
        return "independent"
    if parsed.flags.get("vest_specific"):
        return "vest"
    return None


def parse_docstring(text: str | None) -> ParsedDocstring:
    """Parse a function docstring against the processing contract; never raises."""
    return _parse_docstring(text, PROCESS_CONTRACT)


def parse_module_docstring(text: str | None) -> ModuleDoc:
    """Parse a submodule docstring into title, overview and notation table."""
    return _parse_module_docstring(text, PROCESS_CONTRACT)
