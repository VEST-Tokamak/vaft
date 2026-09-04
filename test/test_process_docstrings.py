"""Every public processing function documents itself under the issue #252 contract.

The contract is parsed by :mod:`vaft.process._docstring` and its structural
half -- parameters match the signature and carry units, there is a Returns
with units, Applicability declares a machine scope -- is checked by the
catalog itself, so a function's ``conforming`` flag means the same thing on
the documentation site as it does here.  This file adds the half that is
physics judgement rather than parseable content: which functions need no
Provenance because they are pure numerics, which are multi-stage and must
list their Processing steps, which change the processing state and must say
so, and which are convention-sensitive.

``PENDING`` names the submodules not yet brought under the contract.  Each
of #418-#421 removes its modules from it, and a test below refuses to let a
module leave early or stay once it conforms, so the split is enforceable
rather than aspirational.
"""

from __future__ import annotations

import pytest

import vaft.process
from vaft.process import catalog
from vaft.process._docstring import (
    CUSTOM_SECTIONS,
    MACHINE_INDEPENDENT,
    SECTION_VOCABULARY,
    VEST_SPECIFIC,
)

#: Submodules whose functions are not yet documented under the contract.
#: Sub-issue B (#418): magnetics, electromagnetics, fluctuation.
#: Sub-issue C (#419): equilibrium, cocos.
#: Sub-issue D (#420): profile, atomic.
#: Sub-issue E (#421): impa, soft_x_rays, langmuir, camera_geometry.
#: ``onset`` landed after #252 was scoped (#409) and is unassigned.
PENDING = frozenset({
    "atomic",
    "camera_geometry",
    "cocos",
    "electromagnetics",
    "equilibrium",
    "fluctuation",
    "impa",
    "langmuir",
    "magnetics",
    "onset",
    "profile",
    "soft_x_rays",
})

#: Pure numerics and bookkeeping: no source adds anything.
DEFINITIONAL = frozenset({
    "line_average_density",
    "linear_baseline",
    "quadratic_baseline",
    "exp_baseline",
    "describe_time_grid",
    "detrend_moving_average",
    "butterworth_lowpass",
    "butterworth_bandpass",
    "is_signal_active",
    "signal_on_offset",
    "process_signal",
    "time_derivative",
    "filter_dataframe",
    "log_transform",
    "analyze_significance",
    "compute_metrics",
    "get_residuals",
    "get_correlation_matrix",
    "get_individual_correlations",
    "confinement_time_histogram",
})

#: Multi-stage routines: the order of operations decides what the output means.
PIPELINE = frozenset({
    "repair_clipped_interval",
    "vest_coil_current_noise_reduction",
    "anti_alias_filter",
    "resample_to_time",
    "process_signal",
    "subtract_baseline",
    "signal_on_offset",
    "time_derivative",
    "filter_dataframe",
    "generate_core_profiles_history_dataframe",
    "perform_ols_regression",
    "compute_metrics",
})

#: Routines whose output sits at a different place in the processing chain
#: from their input, and must say so in Input/Output semantics.  Sub-issues
#: C and D add the equilibrium mappers, the profile fitters and the
#: reconstructions.
STATEFUL = frozenset({
    "repair_clipped_interval",
})

#: Sign, phase, coordinate or normalisation choices change the number.
CONVENTION_SENSITIVE = frozenset({
    "line_average_density",
    "smooth",
    "butterworth_lowpass",
    "butterworth_bandpass",
    "detrend_moving_average",
    "vest_coil_current_noise_reduction",
    "describe_time_grid",
    "anti_alias_filter",
    "resample_to_time",
    "process_signal",
    "signal_on_offset",
    "infer_signal_orientation",
    "time_derivative",
    "filter_dataframe",
    "log_transform",
    "perform_ols_regression",
    "compute_metrics",
})

SPECS = [spec for spec in catalog.list_processes() if spec.category not in PENDING]
IDS = [spec.qualname for spec in SPECS]


# --- the split ------------------------------------------------------------------


def test_pending_names_real_categories():
    assert PENDING <= set(catalog.CATEGORIES), sorted(PENDING - set(catalog.CATEGORIES))


def test_pending_is_exactly_the_set_of_non_conforming_categories():
    """A module may not leave PENDING early, nor linger once it conforms."""
    actual = {doc.name for doc in catalog.categories() if not doc.conforming}
    assert actual == PENDING, {
        "should be pending": sorted(actual - PENDING),
        "should be removed from PENDING": sorted(PENDING - actual),
    }


def test_something_is_under_the_contract():
    assert SPECS, "every category is pending; the contract enforces nothing"


# --- structural, per function ------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_has_a_summary_line(spec):
    assert spec.summary, "missing docstring or summary"
    assert spec.summary.endswith("."), spec.summary


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_conforms_structurally(spec):
    assert spec.conforming, "\n".join(spec.errors)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_every_section_is_in_the_vocabulary(spec):
    for title, _ in spec.sections:
        assert title in SECTION_VOCABULARY, title


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_applicability_declares_exactly_one_scope(spec):
    if spec.deprecated:
        pytest.skip("deprecated shim: summary only")
    text = spec.section("Applicability") or ""
    assert text.startswith((MACHINE_INDEPENDENT, VEST_SPECIFIC)), text[:60]
    assert not (text.startswith(MACHINE_INDEPENDENT) and VEST_SPECIFIC in text[:40])
    assert spec.machine_scope in ("independent", "vest")


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_a_vest_specific_routine_says_which_data(spec):
    """`VEST-specific.` alone is not applicability; it must say for what."""
    if spec.machine_scope != "vest":
        pytest.skip("not VEST-specific")
    text = spec.section("Applicability") or ""
    assert len(text) > len(VEST_SPECIFIC) + 20, text


# --- policy, per function ------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_ported_and_empirical_routines_record_their_provenance(spec):
    if spec.deprecated or spec.name in DEFINITIONAL:
        pytest.skip("definitional or deprecated")
    assert spec.references, "no Provenance section; a ported or empirical routine must name its source"
    for ref in spec.references:
        assert ref.text, f"empty provenance entry [{ref.label}]"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_multi_stage_routines_list_their_processing_steps(spec):
    if spec.name not in PIPELINE:
        pytest.skip("single-step")
    steps = spec.section("Processing steps") or ""
    assert steps, "missing Processing steps"
    assert "1." in steps and "2." in steps, "Processing steps must be an ordered list"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_stateful_routines_describe_input_and_output_semantics(spec):
    if spec.name not in STATEFUL:
        pytest.skip("state unchanged")
    assert spec.section("Input semantics"), "missing Input semantics"
    assert spec.section("Output semantics"), "missing Output semantics"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_convention_sensitive_routines_state_their_convention(spec):
    if spec.name not in CONVENTION_SENSITIVE:
        assert not spec.convention_sensitive, (
            "carries a Convention section but is not in the CONVENTION_SENSITIVE policy list"
        )
        pytest.skip("not convention-sensitive")
    assert spec.convention_sensitive, "missing Convention section"
    assert len(spec.section("Convention") or "") > 20


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_defaults_that_matter_are_classified(spec):
    """A Defaults section must say what kind of value each default is."""
    text = spec.section("Defaults")
    if text is None:
        pytest.skip("no Defaults section")
    kinds = (
        "physical constant", "literature value", "diagnostic calibration",
        "empirical", "validated-workflow default", "validated workflow default",
        "machine-specific", "acquisition-era", "legacy compatibility",
        "numerical convenience", "assumed value", "hard-coded", "conventional",
    )
    assert any(kind in text for kind in kinds), text[:120]


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_documented_defects_are_tracked_by_issue(spec):
    text = spec.section("Limitations") or ""
    if "tracked in" in text.lower():
        assert "#" in text, "a tracked limitation must name its GitHub issue number"


def test_policy_lists_name_real_functions():
    known = {spec.name for spec in catalog.list_processes()}
    known |= {alias for spec in catalog.list_processes() for alias in spec.aliases}
    for name, policy in (
        ("DEFINITIONAL", DEFINITIONAL),
        ("PIPELINE", PIPELINE),
        ("STATEFUL", STATEFUL),
        ("CONVENTION_SENSITIVE", CONVENTION_SENSITIVE),
    ):
        assert policy <= known, (name, sorted(policy - known))


def test_policy_lists_only_govern_functions_under_the_contract():
    """A name on a policy list for a pending module is a promise nobody checks."""
    governed = {spec.name for spec in SPECS}
    for name, policy in (
        ("DEFINITIONAL", DEFINITIONAL),
        ("PIPELINE", PIPELINE),
        ("STATEFUL", STATEFUL),
        ("CONVENTION_SENSITIVE", CONVENTION_SENSITIVE),
    ):
        assert policy <= governed, (name, sorted(policy - governed))


# --- module docstrings ----------------------------------------------------------


@pytest.mark.parametrize(
    "doc", [d for d in catalog.categories() if d.name not in PENDING], ids=lambda d: d.name
)
def test_module_docstrings_carry_a_title_and_overview(doc):
    assert doc.title.endswith("."), doc.title
    assert doc.overview or doc.notation or doc.conventions


def test_vocabulary_lists_every_custom_section():
    assert set(CUSTOM_SECTIONS) <= set(SECTION_VOCABULARY)
