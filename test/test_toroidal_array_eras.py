"""Which toroidal Mirnov array a shot has, and what it can therefore resolve.

VEST has had two arrays and a gap between them, and that is a fact about the
machine rather than about any analysis. It is recorded in ``vest.yaml`` and
answered here.

Evidence for the boundaries, both in the ``toroidal_mirnov_reference`` block:
the 2023-04 magnetics log's History sheet records field 207 as existing only
to shot 35520, and the archived raw dumps for 44740 and 45531 contain none of
fields 207/209/241 while the packaged 39915 maps all three with zero samples.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from vaft.machine_mapping.magnetics import (
    FLUCTUATION_MIRNOV_FIRST_SHOT,
    TOROIDAL_MIRNOV_REFERENCE_CHANNELS,
    TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT,
    toroidal_array_for_shot,
    toroidal_mirnov_reference_channels,
)

LEGACY = Path(__file__).resolve().parents[1] / "vaft" / "data" / "legacy"
REFERENCE_FIELDS = (207, 209, 241)


# ---------------------------------------------------------------------------
# The boundaries
# ---------------------------------------------------------------------------


def test_the_two_arrays_do_not_overlap():
    """A gap, not a handover -- which is why the middle era resolves nothing."""
    assert TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT < FLUCTUATION_MIRNOV_FIRST_SHOT


@pytest.mark.parametrize(
    "shot,name",
    [
        (30000, "phase_reference"),
        (TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT, "phase_reference"),
        (TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT + 1, None),
        (39915, None),
        (FLUCTUATION_MIRNOV_FIRST_SHOT - 1, None),
        (FLUCTUATION_MIRNOV_FIRST_SHOT, "fluctuation"),
        (45531, "fluctuation"),
    ],
)
def test_the_array_a_shot_has(shot, name):
    assert toroidal_array_for_shot(shot)["name"] == name


def test_the_gap_resolves_no_mode_number_at_all():
    """Not "the fit is hard here" -- there is no second position to fit across."""
    era = toroidal_array_for_shot(39915)
    assert era["angles_deg"] == ()
    assert era["alias_step"] is None


@pytest.mark.parametrize(
    "shot,angles,step",
    [
        (30000, (75.0, 135.0, 195.0, 315.0), 6),
        (45531, (135.0, 225.0, 315.0), 4),
    ],
)
def test_each_array_states_the_alias_step_it_imposes(shot, angles, step):
    """`n` is resolved only modulo this, and the number comes from the geometry."""
    era = toroidal_array_for_shot(shot)
    assert era["angles_deg"] == pytest.approx(angles)
    assert era["alias_step"] == step


def test_the_reported_angles_are_imas_phi_not_clock_angles():
    """The fluctuation identifiers say 45/135/225; those are clock angles."""
    era = toroidal_array_for_shot(45531)
    assert era["clocks"] == pytest.approx((1.5, 4.5, 7.5))
    assert era["angles_deg"] == pytest.approx((135.0, 225.0, 315.0))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_inventory_is_ungated():
    """shot=0 means "the machine description", the reading the whole module uses."""
    assert len(toroidal_mirnov_reference_channels(0)) == len(TOROIDAL_MIRNOV_REFERENCE_CHANNELS)
    assert len(TOROIDAL_MIRNOV_REFERENCE_CHANNELS) == 4


def test_a_shot_past_the_boundary_has_no_reference_channels():
    assert toroidal_mirnov_reference_channels(TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT) != ()
    assert toroidal_mirnov_reference_channels(TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT + 1) == ()


def test_the_returned_channels_are_copies():
    first = toroidal_mirnov_reference_channels(0)
    first[0]["clock"] = 99.0
    assert toroidal_mirnov_reference_channels(0)[0]["clock"] != 99.0


def test_the_disputed_channel_is_still_marked_as_such():
    """Field 171 stays until issue #825 is settled, but never silently."""
    disputed = [c for c in TOROIDAL_MIRNOV_REFERENCE_CHANNELS if "disputed" in c]
    assert [c["field_code"] for c in disputed] == [171]
    assert "#825" in disputed[0]["disputed"]


# ---------------------------------------------------------------------------
# The evidence, checked rather than cited
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shot", [44740, 45531])
def test_the_archives_past_the_boundary_carry_no_reference_field(shot):
    """The claim the boundary rests on, verified against the packaged archives."""
    archive = LEGACY / f"shot_{shot}.json.gz"
    if not archive.exists():
        pytest.skip(f"{archive.name} is a repository-only sample")
    with gzip.open(archive, "rt") as handle:
        payload = json.load(handle)
    present = set(payload.get("fields", {}))
    assert shot > TOROIDAL_MIRNOV_REFERENCE_LAST_SHOT
    assert not [f for f in REFERENCE_FIELDS if str(f) in present or f in present]
