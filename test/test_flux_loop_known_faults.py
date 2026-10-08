"""Flux-loop faults recorded in vest.yaml, pinned at their boundaries (#1543).

Each boundary comes from a per-shot scan of every loop's integrated amplitude:
the shot before and the shot after are asserted, so a range edited by one
shot fails here rather than silently admitting a dead loop to EFIT.
"""

from __future__ import annotations

import pytest

from vaft.machine_mapping.magnetics import (
    VestConfigurationError,
    _pinned_channel_index,
    known_magnetics_faults,
)

C4_04 = ("b_field_pol_probe", 45)


def _loops(shot: int) -> list[int]:
    return sorted(index for kind, index in known_magnetics_faults(shot) if kind == "flux_loop")


@pytest.mark.parametrize(
    ("shot", "dead"),
    [
        (44393, []),
        (44394, [7]),
        (44922, [7]),
        (44923, []),
        (44927, []),
        (44928, [7, 8, 9, 10]),
        (45027, [7, 8, 9, 10]),
        (45028, []),
        # #1793: lower-inboard #14/#15 off during plasma, then #12/#13/#15 flat zero
        (46748, []),
        (46749, [9, 10]),
        (47985, [9, 10]),
        (47986, [3, 9, 10]),
        (48038, [3, 9, 10]),
        (48039, [3, 9]),
        (48392, [3, 9]),
        (48393, [3]),
        (48815, [3]),
        (48816, [3, 7, 8, 10]),
        (48926, [3, 7, 8, 10]),
    ],
)
def test_dead_flux_loops_at_each_boundary(shot, dead):
    assert _loops(shot) == dead


@pytest.mark.parametrize("shot", [44394, 44922, 44923, 44928, 45027, 46749, 47986, 48039, 48393, 48816, 48940])
def test_a_flux_loop_revision_keeps_the_always_on_probe_fault(shot):
    """Revisions replace `probes`; C4-04 must be restated in each one (#977)."""
    assert C4_04 in known_magnetics_faults(shot)


def test_loop_indices_count_within_flux_loop_and_are_pinned_by_position():
    assert _pinned_channel_index({"index": 3, "r": 0.592, "z": -0.685}, "flux_loop", context="t") == 3
    with pytest.raises(VestConfigurationError, match="not at r="):
        _pinned_channel_index({"index": 3, "r": 0.592, "z": 0.685}, "flux_loop", context="t")
    with pytest.raises(VestConfigurationError, match="out of range"):
        _pinned_channel_index({"index": 11, "r": 0.0, "z": 0.0}, "flux_loop", context="t")
