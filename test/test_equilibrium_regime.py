"""Which cohort a reconstructed slice belongs to (issue #76).

The rule these pin is a choice, and the tests say which choice and why, so a
later change has to argue with the measurement rather than with a number.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.equilibrium_regime import (
    EquilibriumRegimePolicy,
    vest_equilibrium_regime_policy,
)
from vaft.validation import (
    PHASES,
    classify_equilibrium_regime,
    classify_equilibrium_regimes,
)

POLICY = vest_equilibrium_regime_policy()


def _ods(currents, times=None, limloc=None):
    """A reconstruction carrying only what the classifier reads."""
    ods = ODS(consistency_check=False)
    values = np.asarray(currents, dtype=float)
    grid = np.asarray(
        times if times is not None else np.arange(values.size) * 1e-3 + 0.30,
        dtype=float,
    )
    ods["equilibrium.time"] = grid
    for index, value in enumerate(values):
        ods[f"equilibrium.time_slice.{index}.constraints.ip.measured"] = float(value)
        if limloc is not None and limloc[index] is not None:
            ods[
                f"equilibrium.code.parameters.time_slice.{index}.aeqdsk.limloc"
            ] = limloc[index]
    return ods


# --- the rule ---------------------------------------------------------------


def test_the_phase_rule_is_a_level_and_says_why():
    """#76: the rule must be explicit and configurable, not a hidden branch.

    The level rule was chosen over the slope rule after measuring both: they
    agree on 295 of 300 reference slices, so the choice was made on what sets
    the threshold's scale. The slope rule scales by that shot's max |dIp/dt|,
    which is set by the termination -- 2.73, 9.38 and 2.34 MA/s over three
    discharges. The level rule scales by peak current, a property of the
    plasma being classified.
    """
    assert POLICY.phase_rule == "level"
    assert POLICY.flat_fraction == 0.9
    assert POLICY.status["phase"] == "measured"
    # The evidence travels with the rule, not in a commit message.
    provenance = POLICY.provenance["phase"]
    assert "295 of 300" in provenance
    assert "not causal" in provenance, "the known cost must be stated"


def test_a_slice_is_flat_when_it_is_near_its_own_discharge_peak():
    labels = classify_equilibrium_regimes(
        _ods([20_000, 60_000, 100_000, 95_000, 40_000]), policy=POLICY
    )
    assert [item.phase for item in labels] == [
        "ramp_up",
        "ramp_up",
        "flat",
        "flat",
        "ramp_down",
    ]
    # Every label carries the number that produced it.
    assert labels[2].fraction_of_peak == pytest.approx(1.0)
    assert labels[3].fraction_of_peak == pytest.approx(0.95)
    assert labels[4].fraction_of_peak == pytest.approx(0.40)


def test_the_flat_boundary_is_where_the_configuration_puts_it():
    """Exactly at the threshold counts as flat; a hair below does not."""
    labels = classify_equilibrium_regimes(
        _ods([100_000, 90_000, 89_999]), policy=POLICY
    )
    assert [item.phase for item in labels] == ["flat", "flat", "ramp_down"]


def test_the_vacuum_cut_is_efits_own_and_not_a_second_opinion():
    """A slice EFIT calls vacuum is not a slice with a phase.

    The threshold is the same number as `CUTIP`; two different plasma-current
    floors in one pipeline is the confusion #708 spent a commit removing.
    """
    from vaft.code.efit.config import EFITInitializationConfig

    assert POLICY.vacuum_current_amperes == EFITInitializationConfig().current_threshold

    labels = classify_equilibrium_regimes(
        _ods([14_999, 15_000, 100_000]), policy=POLICY
    )
    assert [item.phase for item in labels] == ["vacuum", "ramp_up", "flat"]


def test_the_slope_rule_is_still_reachable_by_name():
    """Two studies were written against it, so an A/B must be able to ask."""
    slope = EquilibriumRegimePolicy(
        phase_rule="slope",
        flat_fraction=POLICY.flat_fraction,
        slope_fraction=POLICY.slope_fraction,
        vacuum_current_amperes=POLICY.vacuum_current_amperes,
        topology_labels=POLICY.topology_labels,
        status=POLICY.status,
        provenance=POLICY.provenance,
    )
    currents = [20_000, 60_000, 100_000, 100_500, 40_000]
    by_level = [i.phase for i in classify_equilibrium_regimes(_ods(currents), policy=POLICY)]
    by_slope = [i.phase for i in classify_equilibrium_regimes(_ods(currents), policy=slope)]
    assert by_level != by_slope, "the two rules must be distinguishable"
    assert set(by_slope) <= set(PHASES)


# --- unknown is not a negative answer ---------------------------------------


def test_an_unreadable_current_is_unknown_and_not_vacuum():
    """The distinction #76 asks for, and the one that is easy to lose.

    A missing current means nobody looked. Calling it `vacuum` would put it in
    a cohort with slices that were measured and found empty, and every
    statistic over that cohort would then be wrong in a direction nothing
    reports.
    """
    labels = classify_equilibrium_regimes(_ods([np.nan, 100_000]), policy=POLICY)
    assert labels[0].phase == "unknown"
    assert labels[0].current is None
    assert "plasma_current_unreadable" in labels[0].reasons
    assert labels[1].phase == "flat"


def test_an_unnamed_boundary_is_unknown_and_not_limited():
    """Same rule on the topology side: absent, unrecognised, and named differ."""
    ods = _ods([100_000, 100_000, 100_000], limloc=["IN", "ZZZZ", None])
    labels = classify_equilibrium_regimes(ods, policy=POLICY)

    assert labels[0].topology == "inboard_limited"
    assert labels[0].limloc == "IN" and labels[0].reasons == ()

    assert labels[1].topology == "unknown"
    assert labels[1].limloc == "ZZZZ", "the raw value is kept even when unmapped"
    assert "limloc_unrecognised" in labels[1].reasons

    assert labels[2].topology == "unknown"
    assert labels[2].limloc is None
    assert "limloc_absent" in labels[2].reasons


def test_a_diverted_label_exists_but_vest_has_not_produced_one():
    """The map covers what EFIT can write, which is not a claim about VEST.

    Shot 46742 was run end to end for exactly this question and its eight
    converged slices are all `IN`. Listing `DN` does not assert that VEST
    diverts, and this test exists so the two are not confused later.
    """
    assert POLICY.topology("DN") == "double_null"
    assert POLICY.topology("SNT") == "single_null_top"
    assert POLICY.topology("VAC") == "vacuum"
    assert POLICY.status["topology"] == "inferred"
    assert "Nothing here claims VEST diverts" in POLICY.provenance["topology"]


# --- the record -------------------------------------------------------------


def test_the_label_carries_its_inputs_so_a_cohort_can_be_recut():
    """#76: derived labels must preserve their input quantities."""
    labels = classify_equilibrium_regimes(_ods([100_000, 50_000]), policy=POLICY)
    record = labels[0].as_dict()
    assert set(record) == {
        "time", "phase", "topology", "current", "current_rate",
        "fraction_of_peak", "limloc", "rule", "reasons",
    }
    assert record["rule"] == "level"
    assert record["current"] == pytest.approx(100_000.0)
    # JSON-safe: no numpy scalars leak into the record.
    import json

    json.dumps(record)


def test_one_slice_is_selected_from_the_discharge_not_classified_alone():
    ods = _ods([20_000, 100_000, 40_000])
    single = classify_equilibrium_regime(ods, time_index=1, policy=POLICY)
    assert single.phase == "flat"
    assert single == classify_equilibrium_regimes(ods, policy=POLICY)[1]

    with pytest.raises(IndexError):
        classify_equilibrium_regime(ods, time_index=9, policy=POLICY)


def test_a_product_with_no_slices_classifies_to_nothing():
    empty = ODS(consistency_check=False)
    assert classify_equilibrium_regimes(empty, policy=POLICY) == ()
    with pytest.raises(ValueError, match="no equilibrium time slices"):
        classify_equilibrium_regime(empty, time_index=0, policy=POLICY)


def test_the_packaged_reference_classifies_end_to_end():
    """The rule applied to a real reconstruction, not only to a fixture."""
    from vaft.omas.sample import sample_ods

    labels = classify_equilibrium_regimes(sample_ods(), policy=POLICY)
    assert labels, "the packaged sample carries slices"
    assert {item.phase for item in labels} <= set(PHASES)
    # 39915 peaks at ~80 kA and decays; it has a flat top and a ramp-down, and
    # its last slice falls under the vacuum cut.
    assert {"flat", "ramp_down", "vacuum"} <= {item.phase for item in labels}
    # No a-file parameters on this product, so the topology is honestly unknown.
    assert all(item.topology == "unknown" for item in labels)
    assert all("limloc_absent" in item.reasons for item in labels)
