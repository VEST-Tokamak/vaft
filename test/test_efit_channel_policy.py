"""The acceptance policy that decides EFIT's magnetic channels (issue #296).

It reads the projected validity and the routine manual list, nothing else:
no detector, no vacuum model, no machine list of its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.validation.channel_decision import (
    MISSING,
    REASON_CONDEMNED,
    REASON_MANUAL_LIST,
    REASON_NO_SIGNAL,
    REASON_PROJECTED_VALIDITY,
    REJECTED,
    SUSPECT,
    USABLE,
)
from vaft.validation.efit_channels import (
    condemned_channels,
    decide_efit_channels,
    efit_probe_count,
    split_manual_rejections,
)
from vaft.validation.imas import VALIDITY_INVALID, VALIDITY_VALID, write_validity

N = 2500
TIME = np.linspace(0.26, 0.36, N)


def _ods(n_probes: int = 4, n_loops: int = 3, *, missing=()) -> ODS:
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = TIME
    for index in range(n_probes):
        base = f"magnetics.b_field_pol_probe.{index}"
        ods[f"{base}.name"] = f"probe{index}"
        ods[f"{base}.position.r"] = 0.05
        ods[f"{base}.position.z"] = 0.1
        if ("b_field_pol_probe", index) not in missing:
            ods[f"{base}.field.data"] = 0.05 * np.sin(2 * np.pi * 20 * TIME)
    for index in range(n_loops):
        base = f"magnetics.flux_loop.{index}"
        ods[f"{base}.name"] = f"loop{index}"
        ods[f"{base}.position.0.r"] = 0.6
        ods[f"{base}.position.0.z"] = 0.3
        if ("flux_loop", index) not in missing:
            ods[f"{base}.flux.data"] = 0.01 * np.cos(2 * np.pi * 20 * TIME)
    return ods


def test_split_manual_rejections_inverts_the_combined_index():
    assert split_manual_rejections([65, 74, 3], nbprobe=64) == {
        ("flux_loop", 0): 65, ("flux_loop", 9): 74, ("b_field_pol_probe", 2): 3,
    }
    with pytest.raises(ValueError, match="one-based"):
        split_manual_rejections([0], nbprobe=64)


def test_a_channel_usable_at_one_slice_and_rejected_at_another():
    ods = _ods()
    codes = np.where(TIME >= 0.31, VALIDITY_INVALID, VALIDITY_VALID)
    write_validity(ods, "magnetics.b_field_pol_probe.1.field", codes)
    decisions = decide_efit_channels(ods, [0.30, 0.32], nbprobe=4)
    probe = decisions.get("b_field_pol_probe", 1)
    assert probe.state.tolist() == [USABLE, REJECTED]
    assert probe.weight_factor.tolist() == [1.0, 0.0]
    assert probe.reasons == (REASON_PROJECTED_VALIDITY,)
    assert decisions.get("b_field_pol_probe", 0).all_usable  # unassessed: usable by default


def test_missing_condemned_and_manual_channels_carry_their_reasons():
    ods = _ods(missing=(("flux_loop", 2),))
    write_validity(ods, "magnetics.b_field_pol_probe.2.field", [VALIDITY_INVALID] * N)
    decisions = decide_efit_channels(ods, [0.30, 0.32], nbprobe=4, manual_rejections=[2, 4 + 1])
    assert decisions.get("flux_loop", 2).state.tolist() == [MISSING, MISSING]
    assert decisions.get("flux_loop", 2).reasons == (REASON_NO_SIGNAL,)
    condemned = decisions.get("b_field_pol_probe", 2)
    assert condemned.state.tolist() == [REJECTED, REJECTED]
    assert condemned.reasons == (REASON_CONDEMNED,)
    manual_probe = decisions.get("b_field_pol_probe", 1)
    assert manual_probe.state.tolist() == [REJECTED, REJECTED]
    assert manual_probe.reasons == (f"{REASON_MANUAL_LIST}:2",)
    manual_loop = decisions.get("flux_loop", 0)
    assert manual_loop.reasons == (f"{REASON_MANUAL_LIST}:5",)
    assert condemned_channels(ods, nbprobe=4) == {2}


def test_a_condemned_channel_on_the_manual_list_records_both_reasons():
    ods = _ods()
    write_validity(ods, "magnetics.b_field_pol_probe.0.field", [VALIDITY_INVALID] * N)
    decision = decide_efit_channels(ods, [0.30], nbprobe=4, manual_rejections=[1]).get("b_field_pol_probe", 0)
    assert decision.reasons == (REASON_CONDEMNED, f"{REASON_MANUAL_LIST}:1")


def test_probes_beyond_efit_geometry_get_no_decision():
    decisions = decide_efit_channels(_ods(n_probes=6), [0.30], nbprobe=4)
    assert decisions.indices("b_field_pol_probe") == (0, 1, 2, 3)
    assert decisions.get("b_field_pol_probe", 5) is None


def test_suspects_mark_usable_slices_suspect_with_the_configured_factor():
    ods = _ods()
    decisions = decide_efit_channels(
        ods, [0.30, 0.32], nbprobe=4, suspects={("flux_loop", 1): "model_disagreement"}, suspect_weight_factor=0.5
    )
    loop = decisions.get("flux_loop", 1)
    assert loop.state.tolist() == [SUSPECT, SUSPECT] and loop.weight_factor.tolist() == [0.5, 0.5]
    assert loop.reasons == ("suspect:model_disagreement",)
    report_only = decide_efit_channels(ods, [0.30], nbprobe=4, suspects={("flux_loop", 1): "x"})
    assert report_only.get("flux_loop", 1).weight_factor.tolist() == [1.0]


def test_no_detector_and_no_model_runs_here(monkeypatch):
    import vaft.validation.magnetics as quality

    def boom(*args, **kwargs):
        raise AssertionError("the policy must not assess signals")

    monkeypatch.setattr(quality, "validate_magnetics_signals", boom)
    decisions = decide_efit_channels(_ods(), [0.30, 0.31], nbprobe=4)
    assert all(decision.all_usable for decision in decisions.entries.values())


def test_the_packaged_shot_rejects_exactly_the_condemned_probe_and_the_manual_loops():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    nbprobe = efit_probe_count(ods)
    assert nbprobe == 64
    times = np.asarray(ods["equilibrium.time"], dtype=float)
    decisions = decide_efit_channels(ods, times, manual_rejections=[65, 66, 67, 68, 72, 74])
    rejected = {key for key, decision in decisions.entries.items() if np.all(decision.state == REJECTED)}
    assert rejected == {("b_field_pol_probe", 25)} | {("flux_loop", i) for i in (0, 1, 2, 3, 7, 9)}
    assert decisions.get("b_field_pol_probe", 25).reasons == (REASON_CONDEMNED,)
    assert decisions.get("flux_loop", 9).reasons == (f"{REASON_MANUAL_LIST}:74",)
    assert decisions.get("flux_loop", 9).name == "Flux Loop - #14"
    assert decisions.indices("b_field_pol_probe") == tuple(range(64))
