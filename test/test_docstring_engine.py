"""`vaft._docstring` is one parser serving two contracts (issue #417).

The formula tests exercise the engine through the formula contract and are
left exactly as they were.  These tests exercise it through a *second*
contract, so that what is contract and what is engine cannot drift back
together: a section one layer allows is rejected under the other, citations
are read from whichever section the contract names, and flags come from the
contract's markers rather than from anything hard-coded.
"""

from __future__ import annotations

import pytest

from vaft._docstring import DocstringContract, parse_docstring, parse_module_docstring
from vaft.formula._docstring import FORMULA_CONTRACT
from vaft.process._docstring import (
    MACHINE_INDEPENDENT,
    PROCESS_CONTRACT,
    VEST_SPECIFIC,
    machine_scope,
)

_PROCESS = """Reconstruct samples saturated at an acquisition limit.

Samples within ``tolerance`` of ``clip_value`` are replaced by a spline.

Parameters
----------
time : np.ndarray
    Sample times [s].
data : np.ndarray
    Waveform with saturated samples [any].

Returns
-------
np.ndarray
    Repaired waveform, unsaturated samples preserved exactly [any].

Processing steps
----------------
1. Mask samples within tolerance of any rail.
2. Fit a cubic spline through the rest.

Applicability
-------------
Machine-independent.  Callers supply the rail and tolerance.

Provenance
----------
.. [1] ``vest.yaml``, ``diamagnetic_rogowski.clip_values``; issue #285.
.. [2] VEST_IMPAProcessing.m, the legacy repair.
"""


def test_the_process_contract_reads_provenance_as_citations():
    parsed = parse_docstring(_PROCESS, PROCESS_CONTRACT)
    assert parsed.errors == ()
    assert [ref.label for ref in parsed.references] == ["1", "2"]
    assert parsed.references[1].text.startswith("VEST_IMPAProcessing.m")
    assert [title for title, _ in parsed.sections] == [
        "Parameters", "Returns", "Processing steps", "Applicability", "Provenance",
    ]


def test_markers_produce_flags_named_by_the_contract():
    parsed = parse_docstring(_PROCESS, PROCESS_CONTRACT)
    assert parsed.flags["machine_independent"] is True
    assert parsed.flags["vest_specific"] is False
    assert parsed.flags["has_processing_steps"] is True
    assert parsed.flags["convention_sensitive"] is False
    assert machine_scope(parsed) == "independent"
    # the formula contract's flags are not part of this parse at all
    assert "empirical" not in parsed.flags
    assert parsed.empirical is False


def test_vest_specific_marker_is_recognised():
    text = _PROCESS.replace(MACHINE_INDEPENDENT, VEST_SPECIFIC)
    assert machine_scope(parse_docstring(text, PROCESS_CONTRACT)) == "vest"


def test_a_section_one_contract_allows_is_unknown_under_the_other():
    formula_only = "S.\n\nValidity\n--------\nEmpirical fit. Dataset X.\n"
    under_process = parse_docstring(formula_only, PROCESS_CONTRACT)
    assert "unknown section header 'Validity'" in "\n".join(under_process.errors)

    process_only = "S.\n\nApplicability\n-------------\nMachine-independent.\n"
    under_formula = parse_docstring(process_only, FORMULA_CONTRACT)
    assert "unknown section header 'Applicability'" in "\n".join(under_formula.errors)


def test_references_under_the_process_contract_are_just_prose():
    """The process contract cites under Provenance; a References section is not in its vocabulary."""
    text = "S.\n\nReferences\n----------\n.. [1] A paper.\n"
    parsed = parse_docstring(text, PROCESS_CONTRACT)
    assert parsed.references == ()
    assert "unknown section header 'References'" in "\n".join(parsed.errors)


def test_unit_tags_are_only_demanded_where_the_contract_says():
    contract = DocstringContract(
        section_vocabulary=("Parameters", "Returns"),
        custom_sections=(),
        item_sections=frozenset({"Parameters", "Returns"}),
        unit_sections=frozenset({"Returns"}),
        reference_section=None,
        module_section_vocabulary=(),
    )
    text = "S.\n\nParameters\n----------\nx : float\n    No tag.\n\nReturns\n-------\nfloat\n    No tag either.\n"
    parsed = parse_docstring(text, contract)
    assert parsed.parameters[0].unit is None
    assert [e for e in parsed.errors if "Parameters" in e] == []
    assert any("Returns" in e and "unit tag" in e for e in parsed.errors)


def test_a_contract_cannot_name_sections_outside_its_own_vocabulary():
    with pytest.raises(ValueError, match="custom sections outside"):
        DocstringContract(("Notes",), ("Validity",), frozenset(), frozenset(), None, ())
    with pytest.raises(ValueError, match="reference section"):
        DocstringContract(("Notes",), (), frozenset(), frozenset(), "References", ())
    with pytest.raises(ValueError, match="marker"):
        DocstringContract(("Notes",), (), frozenset(), frozenset(), None, (),
                          markers={"x": ("Validity", "Empirical fit.")})


def test_module_docstrings_use_each_contracts_own_vocabulary():
    text = "Title.\n\nOverview.\n\nProvenance\n----------\nPorted from VFIT.\n"
    assert parse_module_docstring(text, PROCESS_CONTRACT).errors == ()
    assert "unknown section header 'Provenance'" in "\n".join(
        parse_module_docstring(text, FORMULA_CONTRACT).errors
    )


def test_the_two_contracts_share_the_numpydoc_core():
    core = {"Parameters", "Returns", "Yields", "Raises", "Notes", "See Also", "Examples", "Warnings"}
    assert core <= set(FORMULA_CONTRACT.section_vocabulary)
    assert core <= set(PROCESS_CONTRACT.section_vocabulary)
    assert FORMULA_CONTRACT.unit_sections == PROCESS_CONTRACT.unit_sections
    assert FORMULA_CONTRACT.reference_section != PROCESS_CONTRACT.reference_section
