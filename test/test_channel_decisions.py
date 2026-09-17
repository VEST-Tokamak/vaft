"""The per-channel, per-slice decision contract (issue #296).

Pure: no ODS.  The mapping from state to weight is stated once and is
deterministic; the invariants keep a decision from saying two things at once.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vaft.validation.channel_decision import (
    MISSING,
    PROVENANCE_MEASURED,
    REASON_NO_SIGNAL,
    REASON_PROJECTED_VALIDITY,
    RECOVERED,
    REJECTED,
    STATE_NAMES,
    SUSPECT,
    USABLE,
    ChannelDecision,
    ChannelDecisions,
    decision_from_codes,
    missing_decision,
    rejected_decision,
    usable_decision,
    weight_factor_for,
)


def test_state_codes_and_names_are_aligned():
    assert [STATE_NAMES[c] for c in (USABLE, SUSPECT, REJECTED, MISSING, RECOVERED)] == [
        "usable", "suspect", "rejected", "missing", "recovered",
    ]


def test_weight_factor_mapping_is_deterministic():
    state = np.array([USABLE, SUSPECT, REJECTED, MISSING])
    assert weight_factor_for(state).tolist() == [1.0, 1.0, 0.0, 0.0]
    assert weight_factor_for(state, suspect_weight_factor=0.25).tolist() == [1.0, 0.25, 0.0, 0.0]
    with pytest.raises(ValueError, match="backend's own weight"):
        weight_factor_for(np.array([RECOVERED]))


def test_invariants_reject_weight_on_rejected_or_missing_slices():
    with pytest.raises(ValueError, match="cannot carry weight"):
        ChannelDecision("flux_loop", 0, "L0", np.array([REJECTED, USABLE]), np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="cannot carry weight"):
        ChannelDecision("flux_loop", 0, "L0", np.array([MISSING]), np.array([1.0]))
    with pytest.raises(ValueError, match="shape"):
        ChannelDecision("flux_loop", 0, "L0", np.array([USABLE, USABLE]), np.array([1.0]))
    with pytest.raises(ValueError, match="unknown state"):
        ChannelDecision("flux_loop", 0, "L0", np.array([7]), np.array([1.0]))


def test_a_recovered_slice_needs_a_finite_value_and_a_named_backend():
    state = np.array([USABLE, RECOVERED])
    with pytest.raises(ValueError, match="finite value"):
        ChannelDecision("b_field_pol_probe", 3, "P3", state, np.array([1.0, 0.0]), provenance="fit")
    with pytest.raises(ValueError, match="name its backend"):
        ChannelDecision("b_field_pol_probe", 3, "P3", state, np.array([1.0, 0.0]), value=np.array([np.nan, 0.2]))
    with pytest.raises(ValueError, match="without a recovered slice"):
        ChannelDecision("b_field_pol_probe", 3, "P3", np.array([USABLE]), np.array([1.0]), provenance="fit")
    decision = ChannelDecision(
        "b_field_pol_probe", 3, "P3", state, np.array([1.0, 0.0]), value=np.array([np.nan, 0.2]),
        uncertainty=np.array([np.nan, 0.01]), provenance="gaussian_fit4",
    )
    assert decision.recovered_mask().tolist() == [False, True]
    assert decision.state_at(1) == RECOVERED and not decision.all_usable
    assert decision.provenance != PROVENANCE_MEASURED


def test_the_container_checks_keys_and_grid_and_replaces_entries():
    times = np.array([0.30, 0.31, 0.32])
    a = usable_decision("flux_loop", 0, "L0", 3)
    b = rejected_decision("flux_loop", 1, "L1", 3, reason="manual_exclusion_list:66")
    decisions = ChannelDecisions(times, {("flux_loop", 0): a, ("flux_loop", 1): b})
    assert decisions.indices("flux_loop") == (0, 1)
    assert decisions.state_at("flux_loop", 1, 2) == REJECTED
    assert decisions.get("b_field_pol_probe", 0) is None
    with pytest.raises(ValueError, match="holds a decision for"):
        ChannelDecisions(times, {("flux_loop", 5): a})
    with pytest.raises(ValueError, match="slices on a grid"):
        ChannelDecisions(times[:2], {("flux_loop", 0): a})
    swapped = decisions.replaced(rejected_decision("flux_loop", 0, "L0", 3, reason="x"))
    assert swapped.state_at("flux_loop", 0, 0) == REJECTED
    assert decisions.state_at("flux_loop", 0, 0) == USABLE  # the original is untouched
    assert swapped.summary() == {"flux_loop": {"usable": 0, "suspect": 0, "rejected": 6, "missing": 0, "recovered": 0}}


def test_decision_from_codes_reads_unassessed_as_usable_and_the_floor_as_policy():
    usable = decision_from_codes("flux_loop", 0, "L0", None, 4, min_validity=0, suspect_code=-1)
    assert usable.all_usable and usable.reasons == ()
    codes = np.array([0, 0, -2, -2])
    railed = decision_from_codes("flux_loop", 0, "L0", codes, 4, min_validity=0, suspect_code=-1)
    assert railed.state.tolist() == [USABLE, USABLE, REJECTED, REJECTED]
    assert railed.weight_factor.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert REASON_PROJECTED_VALIDITY in railed.reasons
    with pytest.raises(ValueError, match="codes for"):
        decision_from_codes("flux_loop", 0, "L0", codes, 3, min_validity=0, suspect_code=-1)


def test_suspect_is_rejected_unless_a_factor_is_configured():
    codes = np.array([0, -1, -2])
    strict = decision_from_codes("flux_loop", 0, "L0", codes, 3, min_validity=0, suspect_code=-1)
    assert strict.state.tolist() == [USABLE, REJECTED, REJECTED]
    lenient = decision_from_codes(
        "flux_loop", 0, "L0", codes, 3, min_validity=0, suspect_code=-1, suspect_weight_factor=0.5
    )
    assert lenient.state.tolist() == [USABLE, SUSPECT, REJECTED]
    assert lenient.weight_factor.tolist() == [1.0, 0.5, 0.0]


def test_missing_and_rejected_helpers_carry_their_reasons():
    assert missing_decision("flux_loop", 2, "L2", 2).reasons == (REASON_NO_SIGNAL,)
    rejected = rejected_decision("flux_loop", 2, "L2", 2, reason=("a", "b"))
    assert rejected.reasons == ("a", "b") and rejected.weight_factor.tolist() == [0.0, 0.0]


def test_as_dict_round_trips_through_json_with_state_names():
    decision = ChannelDecision(
        "b_field_pol_probe", 3, "P3", np.array([USABLE, RECOVERED]), np.array([1.0, 0.0]),
        value=np.array([np.nan, 0.2]), provenance="gaussian_fit4", reasons=("r",),
    )
    decisions = ChannelDecisions(np.array([0.30, 0.31]), {("b_field_pol_probe", 3): decision, ("flux_loop", 0): usable_decision("flux_loop", 0, "L0", 2)})
    payload = json.loads(json.dumps(decisions.as_dict(only_notable=True), default=float))
    assert payload["schema_version"] == 1 and payload["times"] == [0.30, 0.31]
    (entry,) = payload["entries"]
    assert entry["state"] == ["usable", "recovered"] and entry["provenance"] == "gaussian_fit4"
    assert len(decisions.as_dict()["entries"]) == 2
