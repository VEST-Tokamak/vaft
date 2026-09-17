"""Island geometry: the critical width GPEC does not provide, and overlap."""

from __future__ import annotations

import numpy as np
import pytest
from resonant_table_fixtures import island_chain, resonant_table

from vaft.process.perturbation import (
    COINCIDENT_POLICIES,
    IslandPairs,
    chirikov,
    critical_island_width,
    group_coincident_islands,
    island_overlap_width,
    island_pairs,
    nearest_surface_spacing,
    penetration_ratio,
)


# --- the critical width -------------------------------------------------------


def test_the_critical_width_is_the_distance_to_the_nearest_neighbour():
    """An island overlaps its neighbour when it is as wide as the gap to it,
    so the width at which that happens is the gap."""
    psi = np.array([0.2, 0.5, 0.6, 1.0])
    np.testing.assert_allclose(
        critical_island_width(psi), [0.3, 0.1, 0.1, 0.4]
    )


def test_the_critical_width_is_returned_in_the_callers_order():
    psi = np.array([1.0, 0.2, 0.6, 0.5])
    ordered = critical_island_width(np.sort(psi))
    got = critical_island_width(psi)
    np.testing.assert_allclose(got[np.argsort(psi)], ordered)


def test_a_lone_surface_has_no_neighbour_and_no_critical_width():
    assert np.isnan(critical_island_width([0.5])).all()
    assert critical_island_width([]).size == 0


def test_two_surfaces_at_the_same_place_are_refused():
    """Islands of different helicity do sit on one surface, and the spacing
    between surfaces means nothing until they are one island."""
    with pytest.raises(ValueError, match="group_coincident_islands"):
        critical_island_width([0.3, 0.5, 0.5])


@pytest.mark.parametrize("psi,width,message", [
    ([0.1, 0.2], [-0.5, 0.1], "negative width"),
    ([0.1, 0.2], [np.nan, 0.1], "non-finite width"),
    ([0.1, np.nan], [0.1, 0.1], "non-finite position"),
])
def test_an_island_that_is_not_a_real_island_is_refused(psi, width, message):
    """A NaN width silently breaks an overlap chain -- its gap is NaN, so the
    pair reads as separated -- and a NaN position defeats the tie check."""
    with pytest.raises(ValueError):
        penetration_ratio(psi, width)
    with pytest.raises(ValueError):
        island_pairs(psi, width)


def test_a_width_that_is_not_one_per_surface_is_refused():
    with pytest.raises(ValueError, match="one per rational surface"):
        penetration_ratio([0.3, 0.5], [0.01, 0.02, 0.03])


# --- the identity that makes the rest trustworthy -----------------------------


def test_the_surface_chirikov_is_the_width_over_the_critical_width():
    psi = np.array([0.2, 0.5, 0.6, 1.0])
    width = np.array([0.15, 0.05, 0.02, 0.2])
    np.testing.assert_array_equal(
        chirikov(psi, width), width / critical_island_width(psi)
    )
    np.testing.assert_array_equal(chirikov(psi, width), penetration_ratio(psi, width))


def test_the_ratio_is_one_exactly_where_islands_touch():
    psi = np.array([0.4, 0.6, 0.8])
    # The middle island is exactly as wide as the gap either side.
    width = np.array([0.05, 0.2, 0.05])
    assert penetration_ratio(psi, width)[1] == pytest.approx(1.0)


def test_the_two_definitions_answer_different_questions():
    """The surface form asks whether one island reaches its nearest
    neighbour; the pair form whether two particular islands reach each
    other. They disagree by design, and the pair form has one fewer entry."""
    # Widths must vary for the two to separate: with every island the same
    # width on a uniform grid both reduce to w / spacing, which is why a
    # fixture of equal widths cannot tell them apart.
    psi = np.array([0.4, 0.5, 0.6, 0.7])
    width = np.array([0.02, 0.15, 0.03, 0.04])
    surface = chirikov(psi, width)
    pair = chirikov(psi, width, definition="pair")
    assert pair.size == surface.size - 1
    assert not np.allclose(pair, surface[:-1])
    # The pair form is the mean of the two half widths over the spacing.
    assert pair[0] == pytest.approx(0.5 * (0.02 + 0.15) / 0.1)
    # The surface form is one island against its own nearest gap.
    assert surface[1] == pytest.approx(0.15 / 0.1)


def test_an_unknown_definition_is_refused():
    psi, width = island_chain()
    with pytest.raises(ValueError, match="'surface' or 'pair'"):
        chirikov(psi, width, definition="global")


# --- pairs --------------------------------------------------------------------


def test_a_pair_gap_is_the_spacing_less_the_two_half_widths():
    psi = np.array([0.4, 0.7])
    width = np.array([0.2, 0.1])
    pairs = island_pairs(psi, width)
    assert len(pairs) == 1
    assert pairs.spacing[0] == pytest.approx(0.3)
    assert pairs.gap[0] == pytest.approx(0.3 - 0.15)
    assert pairs.inner_right[0] == pytest.approx(0.5)
    assert pairs.outer_left[0] == pytest.approx(0.65)
    assert not pairs.overlaps[0]


def test_separatrices_that_meet_exactly_count_as_touching():
    """Rather than as apart by nothing, which is what floating point makes of
    an exact contact."""
    # Chosen so the gap is exactly +0.0 in binary floating point: with a
    # spacing whose rounding happens to fall the other way, `gap < 0` would
    # carry the test and the tolerance would never be exercised.
    psi = np.array([0.25, 0.75])
    width = np.array([0.25, 0.75])  # half-widths sum to exactly the spacing
    pairs = island_pairs(psi, width)
    assert pairs.gap[0] == 0.0
    assert not (pairs.gap[0] < 0.0), "this case must rest on the tolerance, not the sign"
    assert pairs.overlaps[0]


def test_a_round_off_gap_scales_with_the_spacing():
    """A fixed absolute tolerance is a different criterion at every radius:
    the same physics at a spacing of 1e-3 and of 1.0 must read the same."""
    for spacing in (1.0e-3, 1.0):
        psi = np.array([0.0, spacing])
        width = np.array([spacing, spacing])  # exact contact
        residual = 1.0e-13 * spacing          # a round-off-scale gap
        assert island_pairs(psi, width + np.array([0.0, -2 * residual])).overlaps[0]
        # And a gap far above round-off is a real separation at either scale.
        assert not island_pairs(psi, width * np.array([1.0, 0.5])).overlaps[0]


def test_a_pair_can_be_mapped_back_to_the_rows_it_came_from():
    """The pairs are sorted; without the indices a caller with unsorted input
    could not say which helicities a pair joins."""
    psi = np.array([0.7, 0.4, 0.5, 0.6])
    width = np.array([0.04, 0.02, 0.15, 0.03])
    pairs = island_pairs(psi, width)
    np.testing.assert_array_equal(psi[pairs.inner_index], pairs.inner_psi_norm)
    np.testing.assert_array_equal(psi[pairs.outer_index], pairs.outer_psi_norm)
    assert pairs.inner_index.tolist() == [1, 2, 3]


def test_comparing_two_pair_sets_does_not_raise():
    psi, width = island_chain()
    a, b = island_pairs(psi, width), island_pairs(psi, width)
    assert (a == b) is False  # identity, not an elementwise array comparison
    assert isinstance(a, IslandPairs)


def test_fewer_than_two_islands_make_no_pairs():
    assert len(island_pairs([0.5], [0.01])) == 0
    assert len(island_pairs([], [])) == 0


# --- overlap width ------------------------------------------------------------


def test_a_stochastic_patch_in_the_core_is_not_an_edge_layer():
    """The question is whether the region touches the wall. Deciding it from
    whether the outermost *pair* overlaps -- which is what the code this
    replaces did -- reports a patch at psi_n = 0.2 as an edge layer 0.83
    wide."""
    psi = np.array([0.20, 0.25, 0.30, 0.35])
    width = np.full(4, 0.06)
    assert island_pairs(psi, width).overlaps.all(), "it must overlap for this to bite"
    result = island_overlap_width(psi, width, separatrix=1.0)
    assert result.width == 0.0
    assert not result.edge_connected


def test_an_outermost_pair_that_does_not_touch_breaks_the_chain():
    psi, width = island_chain(overlapping_outer=False)
    inner = island_pairs(psi, width)
    assert inner.overlaps[:-2].any(), "the inner pairs must overlap for this to bite"
    result = island_overlap_width(psi, width)
    assert result.width == 0.0
    assert not result.edge_connected


def test_a_separatrix_inside_the_island_stack_is_not_a_reached_boundary():
    """It has nothing to be a width of."""
    result = island_overlap_width([0.20, 0.25, 0.30, 0.35], [0.06] * 4, separatrix=0.1)
    assert result.width == 0.0
    assert not result.edge_connected


def test_a_chain_reaching_the_boundary_is_measured_in_from_it():
    psi, width = island_chain()
    assert psi[-1] + 0.5 * width[-1] == pytest.approx(1.0), "the fixture must reach it"
    result = island_overlap_width(psi, width, separatrix=1.0)
    assert result.edge_connected
    # No gap anywhere in this chain, so the region runs to the innermost
    # island's inner separatrix. (Written as a plain assertion: a conditional
    # expression here silently becomes `assert True` when the gap is None.)
    assert result.first_gap_psi_norm is None
    assert result.width == pytest.approx(1.0 - (psi[0] - 0.5 * width[0]))


def test_a_separatrix_other_than_one_is_used(tmp_path):
    psi, width = island_chain(separatrix=0.9)
    at_nine = island_overlap_width(psi, width, separatrix=0.9)
    at_one = island_overlap_width(psi, width, separatrix=1.0)
    assert at_nine.edge_connected and at_nine.separatrix_psi_norm == 0.9
    assert not at_one.edge_connected, "the chain stops at 0.9, so 1.0 is not reached"
    assert at_nine.width == pytest.approx(0.9 - (psi[0] - 0.5 * width[0]))


def test_the_width_cannot_exceed_the_boundary_it_was_measured_from():
    """A fixed upper bound of one reports a width larger than the plasma
    whenever the innermost island reaches past the axis."""
    psi = np.array([0.02, 0.35, 0.68, 0.95])
    width = np.full(4, 0.4)
    unclipped = 0.95 - (psi[0] - 0.2)
    assert unclipped > 0.95, "the innermost island must reach past the axis"
    result = island_overlap_width(psi, width, separatrix=0.95)
    assert result.edge_connected
    assert result.width == pytest.approx(0.95)


def test_the_walk_stops_at_the_first_gap_going_inward():
    psi, width = island_chain(count=6, break_at=1)
    result = island_overlap_width(psi, width)
    assert result.edge_connected
    pairs = island_pairs(psi, width)
    broken = np.nonzero(~pairs.overlaps)[0]
    expected = 0.5 * (pairs.inner_right[broken[-1]] + pairs.outer_left[broken[-1]])
    assert result.first_gap_psi_norm == pytest.approx(expected)


def test_a_chain_with_no_gap_at_all_runs_to_its_innermost_island():
    psi, width = island_chain(count=4)
    assert island_pairs(psi, width).overlaps.all()
    result = island_overlap_width(psi, width, separatrix=1.0)
    assert result.first_gap_psi_norm is None
    assert result.width == pytest.approx(1.0 - (psi[0] - 0.5 * width[0]))


def test_the_onset_scale_is_the_inverse_square_of_the_outer_pair_chirikov():
    """Island width goes as the square root of the field, so the field has to
    grow by 1/kappa^2 for the outermost pair to reach contact."""
    psi, width = island_chain(overlapping_outer=False)
    result = island_overlap_width(psi, width)
    outer = island_pairs(psi, width).chirikov[-1]
    assert result.onset_scale == pytest.approx(1.0 / outer**2)





def test_a_non_finite_separatrix_is_refused():
    psi, width = island_chain()
    with pytest.raises(ValueError, match="separatrix must be finite"):
        island_overlap_width(psi, width, separatrix=float("nan"))


def test_both_chirikov_maxima_travel_with_the_result():
    psi, width = island_chain()
    result = island_overlap_width(psi, width)
    assert result.max_surface_chirikov == pytest.approx(np.nanmax(chirikov(psi, width)))
    assert result.max_pair_chirikov == pytest.approx(
        np.nanmax(chirikov(psi, width, definition="pair"))
    )
    assert result.policy == "envelope"


# --- coincident surfaces ------------------------------------------------------


def test_islands_on_one_surface_become_one_island():
    psi, width, members = group_coincident_islands([0.5, 0.5, 0.8], [0.02, 0.03, 0.01])
    np.testing.assert_allclose(psi, [0.5, 0.8])
    np.testing.assert_allclose(width, [0.03, 0.01])
    assert members == ((0, 1), (2,))


def test_the_two_policies_differ_and_neither_is_derivable_from_the_other():
    envelope = group_coincident_islands([0.5, 0.5], [0.02, 0.03])[1]
    rss = group_coincident_islands([0.5, 0.5], [0.02, 0.03], policy="rss")[1]
    assert envelope[0] == pytest.approx(0.03)
    assert rss[0] == pytest.approx(np.sqrt(0.02**2 + 0.03**2))
    assert rss[0] > envelope[0]
    assert set(COINCIDENT_POLICIES) == {"envelope", "rss"}


def test_the_policy_changes_the_number_and_not_only_the_label():
    """Coincident surfaces are the only configuration real data overlaps in,
    so the policy has to reach the width rather than just the record."""
    psi, width = island_chain(coincident=True)
    envelope = island_overlap_width(psi, width, policy="envelope")
    rss = island_overlap_width(psi, width, policy="rss")
    assert envelope.policy == "envelope" and rss.policy == "rss"
    assert rss.max_surface_chirikov > envelope.max_surface_chirikov


def test_a_cluster_sits_at_its_members_mean():
    psi, width, _ = group_coincident_islands([0.40, 0.50, 0.60], [0.01] * 3, psi_tol=0.2)
    assert psi[0] == pytest.approx(0.5)


def test_clustering_is_single_linkage_and_can_reach_further_than_the_tolerance():
    """Each island is compared with the last one added, so a chain of steps
    each inside psi_tol collapses however far the chain runs."""
    positions = [0.50, 0.505, 0.51, 0.515]
    psi, _, members = group_coincident_islands(positions, [0.01] * 4, psi_tol=0.006)
    assert psi.size == 1 and members == ((0, 1, 2, 3),)
    assert positions[-1] - positions[0] > 0.006


def test_how_close_counts_as_the_same_surface_is_a_parameter():
    apart = group_coincident_islands([0.5, 0.5 + 1e-6], [0.02, 0.03], psi_tol=1e-9)
    together = group_coincident_islands([0.5, 0.5 + 1e-6], [0.02, 0.03], psi_tol=1e-5)
    assert apart[0].size == 2 and together[0].size == 1


def test_an_unknown_policy_or_a_negative_tolerance_is_refused():
    with pytest.raises(ValueError, match="policy must be one of"):
        group_coincident_islands([0.5], [0.01], policy="sum")
    with pytest.raises(ValueError, match="psi_tol must be non-negative"):
        group_coincident_islands([0.5], [0.01], psi_tol=-1.0)


def test_grouping_an_empty_set_is_empty():
    psi, width, members = group_coincident_islands([], [])
    assert psi.size == 0 and width.size == 0 and members == ()


# --- against the shape of a real table ----------------------------------------


def test_the_critical_width_is_defined_where_gpec_leaves_its_own_at_zero():
    """GPEC writes w_isl_v_crit as zeros on an ideal run, which is the whole
    reason this is computed rather than read."""
    table = resonant_table()
    assert np.all(table["w_isl_v_crit"] == 0.0)
    w_crit = critical_island_width(table["psi_n_rational"])
    assert np.all(np.isfinite(w_crit)) and np.all(w_crit > 0.0)


def test_the_recomputed_chirikov_is_the_one_the_file_carries():
    """The identity everything here rests on. The fixture's K_isl is built
    the way GPEC builds it -- the island width over the nearest-neighbour
    spacing -- so this is the same check the real runs pass bit-for-bit."""
    table = resonant_table()
    np.testing.assert_array_equal(
        chirikov(table["psi_n_rational"], table["w_isl"]), table["K_isl"]
    )
    np.testing.assert_array_equal(
        chirikov(table["psi_n_rational"], table["w_isl_v"]), table["K_isl_v"]
    )
