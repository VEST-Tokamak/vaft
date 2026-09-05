"""The wall segmentation the reduced-basis contract is built on (vfit #8/#9)."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.wall_segments import (
    GAP_FACTOR,
    SEGMENT_DEFINITION_VERSION,
    WallSegment,
    cluster_by_z_gap,
    segment_digest,
    segment_loops,
    segment_membership,
)


def _add_loop(ods, index, name, rc, zc, w, h):
    base = f"pf_passive.loop.{index}"
    if name is not None:
        ods[f"{base}.name"] = name
    ods[f"{base}.resistance"] = 1.0e-3
    ods[f"{base}.element.0.geometry.geometry_type"] = 1
    ods[f"{base}.element.0.geometry.outline.r"] = np.array([rc - w / 2, rc + w / 2, rc + w / 2, rc - w / 2])
    ods[f"{base}.element.0.geometry.outline.z"] = np.array([zc - h / 2, zc - h / 2, zc + h / 2, zc + h / 2])


def _interleaved_ods() -> ODS:
    """Two structures interleaved in index order plus a W11-style sliver --
    the fixture test_tokamaker_vessel.py uses, because index contiguity must
    never be relied on."""
    wa = [("WA", dict(rc=0.803, zc=float(zc), w=0.006, h=0.05)) for zc in np.arange(-0.175, 0.176, 0.05)]
    wb = []
    for rc in (0.20, 0.25, 0.30):
        wb.append(("WB", dict(rc=rc, zc=+0.5, w=0.05, h=0.02)))
        wb.append(("WB", dict(rc=rc, zc=-0.5, w=0.05, h=0.02)))
    entries = []
    for pair in zip(wa, wb):
        entries.extend(pair)
    entries.extend(wa[len(wb):])
    entries.append(("W11", dict(rc=0.1, zc=0.0, w=0.002, h=0.05)))
    ods = ODS(consistency_check=False)
    for index, (name, params) in enumerate(entries):
        _add_loop(ods, index, name, **params)
    return ods


def test_loops_are_grouped_by_name_regardless_of_index_order():
    segments = segment_loops(_interleaved_ods())
    by_id = {seg.id: seg for seg in segments}
    assert set(by_id) == {"WA", "WB_L", "WB_U", "W11"}
    assert by_id["WA"].size == 8
    assert by_id["WB_L"].size == 3 and by_id["WB_U"].size == 3
    assert by_id["W11"].size == 1
    # interleaved indices: WA takes even positions, WB odd
    assert set(by_id["WA"].index[:6].tolist()) == {0, 2, 4, 6, 8, 10}


def test_a_mirrored_pair_is_split_into_lower_and_upper_segments():
    by_id = {seg.id: seg for seg in segment_loops(_interleaved_ods())}
    assert by_id["WB_L"].z_center < 0.0 < by_id["WB_U"].z_center
    assert by_id["WB_L"].name == by_id["WB_U"].name == "WB"


def test_a_segment_contiguous_through_the_midplane_stays_whole():
    by_id = {seg.id: seg for seg in segment_loops(_interleaved_ods())}
    assert "WA" in by_id and "WA_L" not in by_id
    assert by_id["WA"].index.size == 8


def test_a_split_that_does_not_straddle_the_midplane_is_numbered_not_mirrored():
    ods = ODS(consistency_check=False)
    for index, zc in enumerate((0.1, 0.15, 0.6, 0.65)):  # two clusters, both above z=0
        _add_loop(ods, index, "WC", rc=0.5, zc=zc, w=0.02, h=0.04)
    ids = [seg.id for seg in segment_loops(ods)]
    assert ids == ["WC_1", "WC_2"]


def test_every_loop_belongs_to_exactly_one_segment():
    ods = _interleaved_ods()
    segments = segment_loops(ods)
    membership = segment_membership(segments, len(ods["pf_passive.loop"]))
    assert (membership >= 0).all()
    assert sum(seg.size for seg in segments) == len(ods["pf_passive.loop"])


def test_a_segment_map_that_misses_or_repeats_a_loop_is_refused():
    a = WallSegment("A", "A", np.array([0, 1]), 0.0, 0.0)
    b = WallSegment("B", "B", np.array([1, 2]), 0.0, 0.0)
    with pytest.raises(ValueError, match="more than one segment"):
        segment_membership([a, b], 3)
    with pytest.raises(ValueError, match="belong to no segment"):
        segment_membership([a], 3)


def test_segments_are_ordered_by_first_element_and_the_digest_is_stable():
    ods = _interleaved_ods()
    first = segment_loops(ods)
    second = segment_loops(ods)
    assert [seg.id for seg in first] == [seg.id for seg in second]
    assert [int(seg.index.min()) for seg in first] == sorted(int(seg.index.min()) for seg in first)
    assert segment_digest(first) == segment_digest(second)
    assert len(segment_digest(first)) == 12


def test_an_unnamed_loop_is_refused():
    ods = _interleaved_ods()
    _add_loop(ods, len(ods["pf_passive.loop"]), None, rc=0.5, zc=0.0, w=0.01, h=0.01)
    with pytest.raises(ValueError, match="carries no name"):
        segment_loops(ods)


def test_cluster_by_z_gap_uses_the_median_height_as_its_scale():
    z_lo = np.array([0.0, 0.05, 0.10, 0.30])
    z_hi = z_lo + 0.05
    clusters = cluster_by_z_gap(z_lo, z_hi, gap_factor=GAP_FACTOR)
    assert [c.tolist() for c in clusters] == [[0, 1, 2], [3]]
    # a wider tolerance absorbs the gap
    assert len(cluster_by_z_gap(z_lo, z_hi, gap_factor=4.0)) == 1


def test_tokamaker_vessel_export_uses_the_same_rule():
    """One rule for one question: what the TokaMaker adapter calls a region is
    what the reduced basis calls a segment."""
    from vaft.code.tokamaker.vessel import _loop_rectangles, _split_z_clusters

    ods = _interleaved_ods()
    rects = _loop_rectangles(ods)
    wb_clusters = _split_z_clusters(rects["WB"])
    assert len(wb_clusters) == 2 and all(len(c) == 3 for c in wb_clusters)
    assert len(_split_z_clusters(rects["WA"])) == 1


# --- the packaged machine ----------------------------------------------------

#: Ordered by first element index: the packaged geometry lists each upper
#: half before its lower one, and W8/W6/W10/W7 interleave, so the four upper
#: halves precede the four lower ones.
PACKAGED_IDS = [
    "W1", "W9_U", "W9_L", "W2_U", "W2_L", "W3_U", "W3_L", "W4", "W5_U", "W5_L",
    "W8_U", "W6_U", "W10_U", "W7_U", "W8_L", "W6_L", "W10_L", "W7_L", "W11",
]


@pytest.fixture(scope="module")
def packaged():
    from vaft.omas.sample import sample_ods

    return sample_ods()


def test_the_packaged_wall_segments_into_nineteen_conductors(packaged):
    segments = segment_loops(packaged)
    assert [seg.id for seg in segments] == PACKAGED_IDS
    assert sum(seg.size for seg in segments) == 950
    by_id = {seg.id: seg for seg in segments}
    assert by_id["W1"].size == 240 and by_id["W4"].size == 230 and by_id["W11"].size == 230
    for name in ("W2", "W3", "W5", "W6", "W7", "W8", "W9", "W10"):
        assert by_id[f"{name}_L"].size == by_id[f"{name}_U"].size
        # mirrored to within a centimetre; the real machine is not exactly symmetric
        assert by_id[f"{name}_L"].z_center == pytest.approx(-by_id[f"{name}_U"].z_center, abs=1e-2)
    assert SEGMENT_DEFINITION_VERSION == "vest-name-zgap-1.5-v1"
