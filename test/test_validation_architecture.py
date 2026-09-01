"""The validation layer's architectural contract (issue #253).

`vaft.validation` evaluates the credibility of data and computational results;
it does not generate them, it does not render them, and it does not decide
whether a production stage may use them.  These tests hold that boundary, and
hold the four native IMAS status fields apart from one another, because both
properties are invisible in ordinary use and would erode silently.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from omas import ODS

from vaft.validation import CATEGORIES, ValidationStatus
from vaft.validation import imas as validity


def _in_subprocess(source: str) -> str:
    """Run `source` in a clean interpreter and return its stdout.

    Import side effects are what is being measured, so they cannot be observed
    from inside a session where the test suite has already imported half the
    package.
    """
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# The namespace and its boundaries
# ---------------------------------------------------------------------------

def test_importing_the_validation_core_pulls_in_no_plotting_or_filedb():
    """The core must be usable in a headless, database-free context.

    Production QA consumes validation, never the other way round.  If importing
    the namespace dragged in matplotlib or the FileDB layer, that direction
    would already be reversed regardless of what the module docstrings claim.
    """
    leaked = _in_subprocess(
        "import sys, vaft.validation\n"
        "print(','.join(sorted(m for m in sys.modules if m.startswith(('matplotlib', 'vaft.database')))))"
    )
    assert leaked == "", f"importing vaft.validation pulled in: {leaked}"


def test_verification_is_a_validation_category_not_a_separate_namespace():
    import importlib.util

    assert "verification" in CATEGORIES
    assert importlib.util.find_spec("vaft.verification") is None


def test_every_historical_import_site_still_resolves():
    """The package migration must not break the workflow or its callers.

    `vaft/validation.py` became `vaft/validation/production_qa.py`; the Snakefile
    and six test modules import these names from `vaft.validation` directly.
    """
    from vaft import validation

    for name in (
        "STAGE_METRICS",
        "STAGE_PRECONDITIONS",
        "STAGE_VALIDATION_PLOTS",
        "ValidationPlot",
        "mhd_linear_run_coverage_model",
        "raw_acquisition_qa_model",
        "render_stage_plots",
        "stage_plot_filenames",
        "stages",
        "validation_plots",
        # Reached by name from test_efit_validation_semantics.py: the migration
        # resolves anything production_qa defines, not only its `__all__`.
        "_efit_metrics",
    ):
        assert getattr(validation, name) is not None

    with pytest.raises(AttributeError, match="has no attribute 'not_a_real_name'"):
        validation.not_a_real_name


def test_no_exported_name_is_shadowed_by_a_submodule():
    """A submodule silently wins over a same-named lazy export, import-order
    dependent -- the hazard `test_api_layer_boundaries` guards for
    `vaft.machine_mapping`, applied to this package as it grows domains.
    """
    from vaft import validation

    package_dir = Path(validation.__file__).parent
    submodules = {path.stem for path in package_dir.glob("*.py")} - {"__init__"}
    assert not sorted(set(validation.__all__) & submodules)
    # The package discovers its own submodules rather than listing them, so a
    # new domain module is reachable the moment it exists.
    assert validation._SUBMODULES == submodules
    for name in sorted(submodules):
        assert getattr(validation, name).__name__ == f"vaft.validation.{name}"


# ---------------------------------------------------------------------------
# The status vocabulary
# ---------------------------------------------------------------------------

def test_not_available_is_not_a_pass():
    assert ValidationStatus.NOT_AVAILABLE != ValidationStatus.PASS
    assert ValidationStatus.INDETERMINATE != ValidationStatus.PASS
    assert ValidationStatus.NOT_AVAILABLE != ValidationStatus.INDETERMINATE


def test_a_report_of_statuses_serializes_deterministically():
    report = {
        "verification": {"structure": ValidationStatus.PASS},
        "diagnostic_fit": {"ip": ValidationStatus.NOT_AVAILABLE},
    }
    encoded = json.dumps(report, sort_keys=True)
    assert encoded == json.dumps(report, sort_keys=True)
    assert json.loads(encoded) == {
        "diagnostic_fit": {"ip": "not_available"},
        "verification": {"structure": "pass"},
    }


def test_a_metric_needs_no_status():
    """#253 §6: a calculated number is not automatically a validation result.

    Nothing in the vocabulary forces a metric provider to produce a verdict, so
    a plain float and a compact mapping are both complete answers.
    """
    from vaft.formula.statistics import rms

    assert isinstance(rms([1.0, -1.0]), float)


# ---------------------------------------------------------------------------
# Native IMAS validity
# ---------------------------------------------------------------------------

@pytest.fixture
def probe() -> tuple[ODS, str]:
    """One B-probe channel on a five-sample grid, carrying no validity yet."""
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = np.linspace(0.0, 0.4, 5)
    ods["magnetics.b_field_pol_probe.0.field.data"] = np.arange(5, dtype=float)
    return ods, "magnetics.b_field_pol_probe.0.field"


def test_absent_validity_is_distinguishable_from_valid(probe):
    ods, base = probe
    assert validity.read_validity(ods, base) is None
    assert validity.read_validity_timed(ods, base) is None

    ods[f"{base}.validity"] = validity.VALIDITY_VALID
    assert validity.read_validity(ods, base) == 0
    assert validity.read_validity(ods, base) is not None


def test_an_ods_without_validity_leaves_every_consumer_unchanged(probe):
    """Absence of an assessment is not a rejection.

    Data produced before the quality layer existed must select exactly as it
    always did, which is what makes the layer safe to roll out.
    """
    ods, base = probe
    assert validity.validity_mask(ods, base).all()
    assert validity.valid_fraction(ods, base) == 1.0
    assert not validity.validity_mask(ods, base, default=False).any()


def test_validity_timed_is_authoritative_and_positional(probe):
    """A channel that fails partway through stays usable for the earlier part."""
    ods, base = probe
    ods[f"{base}.validity_timed"] = np.array([0, 0, 0, -2, -2])

    mask = validity.validity_mask(ods, base)
    assert mask.tolist() == [True, True, True, False, False]

    early = validity.validity_codes(ods, base, times=np.array([0.05]))
    late = validity.validity_codes(ods, base, times=np.array([0.35]))
    assert early.tolist() == [0]
    assert late.tolist() == [-2]


def test_the_scalar_aggregate_is_the_worst_state_reached(probe):
    assert validity.aggregate_validity([1, 1, 1]) == validity.VALIDITY_CERTIFIED
    assert validity.aggregate_validity([1, 0, 1]) == validity.VALIDITY_VALID
    assert validity.aggregate_validity([0, 0, -1]) == validity.VALIDITY_SUSPECT
    assert validity.aggregate_validity([0, -1, -2]) == validity.VALIDITY_INVALID
    # A waveform with no samples carries no usable datum, which is a different
    # statement from `read_validity_timed` returning None for an absent node.
    assert validity.aggregate_validity([]) == validity.VALIDITY_INVALID


def test_the_scalar_never_overrides_the_timed_field(probe):
    """The aggregate is a summary, not a veto.

    A channel good for the first half of a discharge aggregates to `invalid`,
    and a consumer reading only that scalar would throw away usable samples.
    """
    ods, base = probe
    ods[f"{base}.validity_timed"] = np.array([0, 0, 0, -2, -2])
    ods[f"{base}.validity"] = validity.VALIDITY_INVALID

    assert validity.read_validity(ods, base) == validity.VALIDITY_INVALID
    assert validity.validity_mask(ods, base).sum() == 3


def test_a_scalar_alone_is_broadcast_over_the_record(probe):
    ods, base = probe
    ods[f"{base}.validity"] = validity.VALIDITY_SUSPECT
    assert validity.validity_mask(ods, base).tolist() == [False] * 5
    assert validity.validity_mask(ods, base, min_validity=validity.VALIDITY_SUSPECT).all()


def test_validity_resolves_against_the_root_time_when_the_node_has_none(probe):
    """`validity_timed` is coordinated on `<node>.time`, which a homogeneous ODS
    does not store -- the shape every packaged VEST sample round-trips into.
    """
    ods, base = probe
    assert f"{base}.time" not in ods
    assert validity.resolve_signal_time(ods, base).size == 5

    ods[f"{base}.time"] = np.linspace(1.0, 1.4, 5)
    assert validity.resolve_signal_time(ods, base)[0] == 1.0


def test_projecting_a_per_sample_assessment_writes_both_native_nodes(probe):
    ods, base = probe
    scalar = validity.write_validity(ods, base, [0, 0, -1, 0, 0])

    assert scalar == validity.VALIDITY_SUSPECT
    assert ods[f"{base}.validity"] == validity.VALIDITY_SUSPECT
    assert np.asarray(ods[f"{base}.validity_timed"]).tolist() == [0, 0, -1, 0, 0]


def test_a_projection_that_would_not_resolve_in_time_is_refused(probe):
    ods, base = probe
    with pytest.raises(ValueError, match="time coordinate has 5"):
        validity.write_validity(ods, base, [0, 0])

    bare = ODS(consistency_check=False)
    with pytest.raises(ValueError, match="no time coordinate"):
        validity.write_validity(bare, base, [0, 0])


def test_valid_fraction_is_a_metric_not_a_verdict(probe):
    """#253 §7: how much of the record survives is evidence; what fraction is
    enough is the consumer's policy and is deliberately not encoded here.
    """
    ods, base = probe
    ods[f"{base}.validity_timed"] = np.array([0, 0, 0, -2, -2])

    assert validity.valid_fraction(ods, base) == pytest.approx(0.6)
    window = np.asarray(ods["magnetics.time"]) < 0.25
    assert validity.valid_fraction(ods, base, window=window) == 1.0
    assert np.isnan(validity.valid_fraction(ods, base, window=np.zeros(5, dtype=bool)))


# ---------------------------------------------------------------------------
# The status fields that are not validity
# ---------------------------------------------------------------------------

def test_output_flag_convergence_and_validity_stay_distinct():
    """Three different questions that a single generic flag would conflate.

    A run can succeed on invalid channels; a converged slice can come from a
    failed run's leftovers.  Each reader answers only its own question, and
    none of them can answer another's.
    """
    ods = ODS(consistency_check=False)
    ods["equilibrium.code.output_flag"] = np.array([0, -1])
    ods["equilibrium.time_slice.0.convergence.result.name"] = "converged"
    ods["equilibrium.time_slice.0.convergence.result.index"] = 1

    assert validity.read_output_flag(ods).tolist() == [0, -1]
    assert validity.read_output_flag(ods, time_slice=1) == -1
    assert validity.read_convergence_result(ods, time_slice=0) == {
        "name": "converged",
        "index": 1,
    }
    # An equilibrium IDS has no `validity`, and a magnetics channel has neither
    # an output flag nor a convergence result. The readers are not substitutes.
    assert validity.read_validity(ods, "equilibrium.time_slice.0") is None
    assert validity.read_convergence_result(ods, time_slice=1) is None
    assert validity.read_output_flag(ods, "magnetics") is None


def test_a_successful_run_flag_says_nothing_about_source_validity(probe):
    ods, base = probe
    ods["equilibrium.code.output_flag"] = np.array([0])
    ods[f"{base}.validity"] = validity.VALIDITY_INVALID

    assert validity.read_output_flag(ods, time_slice=0) == 0
    assert validity.read_validity(ods, base) == validity.VALIDITY_INVALID
