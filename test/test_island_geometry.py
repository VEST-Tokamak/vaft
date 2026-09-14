"""Island geometry: the critical width GPEC does not provide, and overlap."""

from __future__ import annotations

import numpy as np
import pytest
from resonant_table_fixtures import island_chain, resonant_table

from vaft.process.perturbation import (
    COINCIDENT_POLICIES,
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


def test_fewer_than_two_islands_make_no_pairs():
    assert len(island_pairs([0.5], [0.01])) == 0
    assert len(island_pairs([], [])) == 0


# --- overlap width ------------------------------------------------------------


def test_an_overlap_that_does_not_reach_the_boundary_has_zero_width():
    """The quantity is about the edge being connected to the wall. Islands
    overlapping deep inside are not that, however much they overlap."""
    psi, width = island_chain(overlapping_outer=False)
    inner = island_pairs(psi, width)
    assert inner.overlaps[:-2].any(), "the inner pairs must overlap for this to bite"
    result = island_overlap_width(psi, width)
    assert result.width == 0.0
    assert not result.edge_connected


def test_a_chain_reaching_the_boundary_is_measured_in_from_it():
    psi, width = island_chain()
    result = island_overlap_width(psi, width, separatrix=1.0)
    assert result.edge_connected
    assert result.width > 0.0
    assert result.width == pytest.approx(1.0 - result.first_gap_psi_norm) if (
        result.first_gap_psi_norm is not None
    ) else True


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


def test_the_width_is_clipped_to_the_plasma():
    psi, width = island_chain(count=4, start=0.1)
    assert 0.0 <= island_overlap_width(psi, width, separatrix=1.0).width <= 1.0


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


def test_the_policy_travels_into_the_overlap_result():
    psi, width = island_chain()
    assert island_overlap_width(psi, width, policy="rss").policy == "rss"


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
