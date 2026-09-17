"""Aggregating a resonant response over several toroidal mode numbers.

The fixtures here are deliberately not the shipped corpus. The DIII-D ideal
example carries n = 1, 2 and 3, but only n = 1 is driven -- ``Phi_res`` peaks
at 5.7e-4, 7.6e-21 and 3.8e-10 -- so every composite computed on it is one
mode plus two zeros, and a peak that trivially equals its own bound cannot
distinguish an interference that works from one that does not. What the real
runs do establish is the alignment, and ``test_alignment`` below encodes the
three facts measured on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.perturbation import (
    AlignedSurface,
    CompositeDrive,
    DELTA_E_COMBINATIONS,
    SURFACE_MEMBERSHIPS,
    align_surfaces_by_q,
    composite_drive_at_q,
    helical_phase_sweep,
    q_composite_table,
    reduce_delta_e,
)


def table(q, values=None, psi=None, column="Phi_res"):
    """A resonant table for one mode, as read_resonant_table returns."""
    out = {"q_rational": np.asarray(q, dtype=float)}
    if psi is not None:
        out["psi_n_rational"] = np.asarray(psi, dtype=float)
    if values is not None:
        out[column] = np.asarray(values, dtype=complex)
    return out


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def test_a_surface_is_shared_when_the_rationals_are_equal_not_merely_close():
    """q is exactly m / n in double precision, and IEEE division is correctly
    rounded, so 4/3 and 8/6 are the same double. Matching on the rational
    needs no tolerance at all."""
    aligned = align_surfaces_by_q({3: table([4 / 3]), 6: table([8 / 6])})
    assert len(aligned) == 1
    assert aligned[0].resonant_modes == (3, 6)
    assert {x.n: x.m for x in aligned[0].members} == {3: 4, 6: 8}


def test_a_mode_that_cannot_resonate_is_distinguished_from_one_that_is_missing():
    """n * q integral or not is decidable arithmetic; the legacy dropped both
    cases from the group identically, so a truncated run looked exactly like
    a mode with no rational there."""
    # q = 5/2 resonates on n = 2 (m = 5) but never on n = 1: 1 * 2.5 is not
    # an integer. q = 2 and q = 3 are integral for both, so where a run does
    # not report them it stopped short rather than being unable to resonate.
    aligned = align_surfaces_by_q({1: table([2.0]), 2: table([2.5, 3.0])})
    by_q = {round(s.q, 6): {x.n: x.membership for x in s.members} for s in aligned}
    assert by_q[2.0] == {1: "resonant", 2: "absent"}
    assert by_q[2.5] == {1: "non_resonant", 2: "resonant"}
    assert by_q[3.0] == {1: "absent", 2: "resonant"}
    assert set(SURFACE_MEMBERSHIPS) == {"resonant", "non_resonant", "absent"}


def test_the_poloidal_mode_number_is_carried_not_re_derived():
    aligned = align_surfaces_by_q({2: table([1.5, 2.5], psi=[0.3, 0.7])})
    member = aligned[0].members[0]
    assert (member.m, member.index, member.psi_norm) == (3, 0, 0.3)
    assert aligned[1].members[0].m == 5


def test_surfaces_come_back_ascending_in_q():
    aligned = align_surfaces_by_q({1: table([4.0, 2.0, 3.0])})
    assert [s.q for s in aligned] == [2.0, 3.0, 4.0]


def test_a_reported_surface_whose_n_times_q_is_not_integral_is_refused():
    """It is not a rational surface, whatever the file calls it."""
    with pytest.raises(ValueError, match="not an integer"):
        align_surfaces_by_q({2: table([2.03])})


def test_the_integrality_tolerance_is_inclusive_at_its_own_boundary():
    """A q exactly `tolerance` away from a rational is accepted; one past it
    is not. Which side the boundary falls on is invisible unless a test
    stands on it."""
    q = 2.0 + 1e-6
    # The boundary is wherever this q actually lands, not where a decimal
    # literal suggests: 3 * ((6 + 1e-6) / 3) is not 6 + 1e-6.
    exactly = abs(3 * q - round(3 * q))
    aligned = align_surfaces_by_q({3: table([q])}, integrality_tolerance=exactly)
    assert aligned[0].q == pytest.approx(2.0)
    with pytest.raises(ValueError, match="not an integer"):
        align_surfaces_by_q({3: table([q])}, integrality_tolerance=np.nextafter(exactly, 0.0))


def test_a_non_finite_q_is_refused_rather_than_matched():
    """The legacy compared abs(q - target) > tol, which is False for a nan,
    so a nan q was returned as the nearest match."""
    with pytest.raises(ValueError, match="q = nan"):
        align_surfaces_by_q({1: table([2.0, np.nan])})


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_mode_number_that_is_not_a_positive_integer_is_refused(bad):
    with pytest.raises(ValueError, match="positive integers"):
        align_surfaces_by_q({bad: table([2.0])})


def test_a_table_without_q_is_refused():
    with pytest.raises(ValueError, match="no q_rational"):
        align_surfaces_by_q({1: {"Phi_res": np.array([1.0])}})


def test_positions_must_match_the_surfaces_they_position():
    with pytest.raises(ValueError, match="3 values of q against 2 positions"):
        align_surfaces_by_q({1: table([2.0, 3.0, 4.0], psi=[0.5, 0.8])})


def test_no_tables_is_no_surfaces():
    assert align_surfaces_by_q({}) == ()


def test_alignment():
    """The three facts measured on the DIII-D reference runs, as a fixture.

    n = 1, 2 and 3 of shot 147131 report 4, 8 and 12 rational surfaces; their
    union is 16 distinct q; and every one of the 32 gaps is a mode that
    cannot resonate there -- not one is a run that stopped short.
    """
    q_by_n = {
        1: [2.0, 3.0, 4.0, 5.0],
        2: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        3: [4 / 3, 5 / 3, 2.0, 7 / 3, 8 / 3, 3.0, 10 / 3, 11 / 3, 4.0, 13 / 3, 14 / 3, 5.0],
    }
    aligned = align_surfaces_by_q({n: table(q) for n, q in q_by_n.items()})
    assert len(aligned) == 16
    memberships = [x.membership for s in aligned for x in s.members]
    assert len(memberships) == 16 * 3
    assert memberships.count("resonant") == 4 + 8 + 12
    assert memberships.count("non_resonant") == 48 - 24
    assert memberships.count("absent") == 0
    shared = [s.q for s in aligned if len(s.resonant_modes) == 3]
    assert shared == [2.0, 3.0, 4.0, 5.0]


# --------------------------------------------------------------------------
# The composite drive
# --------------------------------------------------------------------------


def test_one_mode_peaks_at_its_own_magnitude():
    drive = composite_drive_at_q({1: table([2.0], [3.0 + 4.0j])}, 2.0, "Phi_res")
    assert drive.peak == pytest.approx(5.0, abs=1e-9)
    assert drive.quadrature == pytest.approx(5.0)
    assert drive.linear_bound == pytest.approx(5.0)


def test_the_bound_is_reached_only_when_the_phases_happen_to_line_up():
    """Aligning every mode at one angle means solving ``n u + arg(z_n) = 0``
    for every n at once -- one unknown against as many constraints as there
    are modes. It is a coincidence, not the usual case, even for two modes,
    which is the whole reason ``peak`` is reported rather than ``sum |z|``.
    """
    # arg(z_n) = -n * 1.1: every constraint is solved by u = 1.1.
    lined_up = {n: table([6.0], [np.exp(-1j * n * 1.1)]) for n in (1, 2, 3)}
    drive = composite_drive_at_q(lined_up, 6.0, "Phi_res")
    assert drive.peak == pytest.approx(3.0, abs=1e-8)
    assert drive.peak_angle == pytest.approx(1.1, abs=1e-6)

    # A pair whose phases are not in that progression falls short already.
    pair = {1: table([6.0], [1.0]), 2: table([6.0], [1.0j])}
    short = composite_drive_at_q(pair, 6.0, "Phi_res")
    assert short.linear_bound == pytest.approx(2.0)
    assert short.peak < 1.77
    assert short.quadrature == pytest.approx(np.sqrt(2.0))


def test_the_peak_is_a_gauge_invariant_and_the_angle_is_not():
    """Shifting the helical origin by d takes z_n to z_n exp(i n d). The
    curve translates; its height does not."""
    values = {1: 1.0 + 0.5j, 2: -0.3 + 0.8j, 3: 0.6 - 0.2j}
    base = {n: table([6.0], [z]) for n, z in values.items()}
    shift = 0.7
    moved = {n: table([6.0], [z * np.exp(1j * n * shift)]) for n, z in values.items()}
    first, second = (composite_drive_at_q(t, 6.0, "Phi_res") for t in (base, moved))
    assert second.peak == pytest.approx(first.peak, rel=1e-10)
    assert second.quadrature == pytest.approx(first.quadrature)
    assert second.peak_angle == pytest.approx(
        float(np.mod(first.peak_angle - shift, 2 * np.pi)), abs=1e-6
    )


def test_the_reported_peak_is_the_largest_value_the_sweep_finds():
    """The refinement is not decoration: a coarse grid alone misses the peak
    of the highest harmonic."""
    modes = {n: table([12.0], [complex(np.cos(n), np.sin(2 * n))]) for n in (1, 5, 11)}
    drive = composite_drive_at_q(modes, 12.0, "Phi_res")
    dense = helical_phase_sweep(
        modes, 12.0, "Phi_res", angles=np.linspace(0.0, 2 * np.pi, 400_001)
    )
    assert drive.peak == pytest.approx(dense.max(), rel=1e-9)
    assert drive.peak >= dense.max() - 1e-9


def test_a_high_mode_number_still_finds_its_peak():
    """The sum is a trigonometric polynomial of degree n_max, so it has up to
    2 n_max maxima. A fixed grid resolves none of them once n_max is large,
    and a search bracketed around the highest sample then refines the wrong
    one: before the grid was sized from n_max, 139 of 200 random four-mode
    cases with n up to 500 came back below the true peak, the worst by 17%.
    """
    rng = np.random.default_rng(3)
    for _ in range(12):
        modes = sorted(rng.choice(np.arange(1, 501), size=4, replace=False).tolist())
        q = float(np.lcm.reduce(modes))
        tables = {n: table([q], [complex(rng.normal(), rng.normal())]) for n in modes}
        drive = composite_drive_at_q(tables, q, "Phi_res")
        dense = helical_phase_sweep(
            tables, q, "Phi_res", angles=np.linspace(0.0, 2 * np.pi, 60 * max(modes))
        )
        assert drive.peak >= dense.max() - 1e-9 * abs(dense.max())


def test_the_peak_is_never_negative():
    """The sum has zero mean over a period, so its maximum cannot be below
    zero. A search that refined into a minimum used to report one: three
    modes (1, 4, 7) on a three-point grid gave -0.394 where the peak is
    +2.100."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        modes = sorted(rng.choice(np.arange(1, 30), size=3, replace=False).tolist())
        q = float(np.lcm.reduce(modes))
        tables = {n: table([q], [complex(rng.normal(), rng.normal())]) for n in modes}
        assert composite_drive_at_q(tables, q, "Phi_res").peak >= 0.0


def test_a_grid_too_coarse_for_the_top_harmonic_is_refused_not_used():
    """Silently sampling below the mode's own period is how the peak came
    back negative."""
    tables = {n: table([28.0], [1.0 + 0j]) for n in (1, 4, 7)}
    with pytest.raises(ValueError, match="at least 113"):
        composite_drive_at_q(tables, 28.0, "Phi_res", angle_points=3)
    with pytest.raises(ValueError, match="whole number"):
        composite_drive_at_q(tables, 28.0, "Phi_res", angle_points=200.5)


def test_the_contribution_comes_from_the_surface_that_was_asked_for():
    """Every other numeric fixture here has one surface per mode, so reading
    index 0 regardless of the q requested passes them all. This one does
    not: each mode carries three surfaces with distinct values."""
    tables = {
        1: table([2.0, 3.0, 4.0], [10.0, 20.0, 40.0]),
        2: table([2.0, 3.0, 4.0], [1.0, 2.0, 4.0]),
    }
    for q, expected in ((2.0, 11.0), (3.0, 22.0), (4.0, 44.0)):
        drive = composite_drive_at_q(tables, q, "Phi_res")
        assert drive.linear_bound == pytest.approx(expected)
        assert drive.peak == pytest.approx(expected, abs=1e-9)
        assert drive.contributions == (
            complex(expected * 10 / 11), complex(expected / 11)
        )


def test_the_table_carries_each_row_from_its_own_surface():
    tables = {
        1: table([2.0, 3.0], [10.0, 20.0], psi=[0.4, 0.8]),
        2: table([2.0, 3.0], [1.0, 2.0], psi=[0.4, 0.8]),
    }
    got = q_composite_table(tables, "Phi_res")
    assert got["linear_bound"].tolist() == pytest.approx([11.0, 22.0])
    assert got["psi_norm_1"].tolist() == pytest.approx([0.4, 0.8])


def test_a_q_taken_from_the_table_matches_the_surface_it_came_from():
    """align_surfaces_by_q reports float(Fraction), which need not be the
    double the caller read out of the file; matching those on equality made
    a q from the very table being passed in fail to resolve."""
    wobbled = 1.3333333333333335
    assert wobbled != 4 / 3
    tables = {3: table([wobbled], [2.0])}
    assert composite_drive_at_q(tables, wobbled, "Phi_res").linear_bound == pytest.approx(2.0)
    assert helical_phase_sweep(tables, wobbled, "Phi_res", angles=[0.0])[0] == pytest.approx(2.0)


def test_a_mode_resonating_twice_at_one_q_is_refused_not_overwritten():
    """A reversed-shear q profile really does, and one SurfaceMember per mode
    cannot hold both -- so the second row used to silently replace the
    first."""
    with pytest.raises(ValueError, match="two surfaces at q"):
        align_surfaces_by_q({1: table([2.0, 2.0], psi=[0.3, 0.9])})


@pytest.mark.parametrize("bad", [0.0, -2.0])
def test_a_non_positive_q_is_refused(bad):
    """q = 0 is integral for every mode number, so it would mark every mode
    resonant at a surface that is not one."""
    with pytest.raises(ValueError, match="a safety factor is positive"):
        align_surfaces_by_q({1: table([bad])})


def test_a_column_the_wrong_length_or_the_wrong_shape_is_refused():
    """A netCDF Phi_res is (2, N) real/imaginary; complex(values[i]) on it
    raised TypeError where the docstring promises ValueError."""
    with pytest.raises(ValueError, match="1 values of 'Phi_res' against 2 surfaces"):
        composite_drive_at_q({1: {"q_rational": [2.0, 3.0], "Phi_res": [1.0]}}, 3.0, "Phi_res")
    raw = {"q_rational": [2.0], "Phi_res": np.array([[1.0], [2.0]])}
    with pytest.raises(ValueError, match="one value per surface"):
        composite_drive_at_q({1: raw}, 2.0, "Phi_res")


def test_a_weight_rescales_a_mode_and_a_unit_weight_re_phases_it():
    """A unit-modulus weight leaves every magnitude alone, so the bound and
    the quadrature sum do not move -- but it changes how the modes line up,
    so the peak does. That asymmetry is what a phasing study varies."""
    one = {1: table([6.0], [1.0]), 2: table([6.0], [1.0])}
    plain = composite_drive_at_q(one, 6.0, "Phi_res")
    halved = composite_drive_at_q(one, 6.0, "Phi_res", weights={2: 0.5})
    turned = composite_drive_at_q(one, 6.0, "Phi_res", weights={2: 1j})
    assert halved.linear_bound == pytest.approx(1.5)
    assert turned.linear_bound == pytest.approx(plain.linear_bound)
    assert turned.quadrature == pytest.approx(plain.quadrature)
    assert plain.peak == pytest.approx(2.0, abs=1e-9)
    assert turned.peak < plain.peak


def test_the_sweep_and_the_drive_describe_one_curve():
    modes = {n: table([6.0], [complex(n, -n)]) for n in (1, 2, 3)}
    drive = composite_drive_at_q(modes, 6.0, "Phi_res")
    at_peak = helical_phase_sweep(modes, 6.0, "Phi_res", angles=[drive.peak_angle])
    assert at_peak[0] == pytest.approx(drive.peak, rel=1e-9)


def test_a_q_no_mode_resonates_at_is_an_error_not_a_row_of_nan():
    """The legacy returned nan rows, so a sweep at a q with no surface looked
    like a sweep that had simply found nothing."""
    with pytest.raises(ValueError, match="no mode in .* resonates at q"):
        composite_drive_at_q({1: table([2.0], [1.0])}, 2.5, "Phi_res")


def test_a_missing_column_on_a_resonating_mode_is_an_error():
    both = {1: table([2.0], [1.0]), 2: table([2.0], None)}
    with pytest.raises(ValueError, match="no 'Phi_res' column"):
        composite_drive_at_q(both, 2.0, "Phi_res")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_contribution_or_weight_is_refused(bad):
    with pytest.raises(ValueError, match="non-finite"):
        composite_drive_at_q({1: table([2.0], [complex(bad, 0)])}, 2.0, "Phi_res")
    with pytest.raises(ValueError, match="weight for n = 1 is not finite"):
        composite_drive_at_q(
            {1: table([2.0], [1.0])}, 2.0, "Phi_res", weights={1: complex(bad, 0)}
        )


def test_a_weight_for_a_mode_that_was_not_given_is_refused():
    """Silently ignoring it would apply a phasing the caller believes is in
    effect."""
    with pytest.raises(ValueError, match="not in tables"):
        composite_drive_at_q({1: table([2.0], [1.0])}, 2.0, "Phi_res", weights={4: 1j})


def test_the_sweep_refuses_a_non_finite_angle():
    with pytest.raises(ValueError, match="every angle must be finite"):
        helical_phase_sweep({1: table([2.0], [1.0])}, 2.0, "Phi_res", angles=[0.0, np.nan])


# --------------------------------------------------------------------------
# The q-composite table
# --------------------------------------------------------------------------


def test_every_row_carries_every_column():
    """The legacy emitted a per-mode column only where that mode was present,
    so the frame's columns varied by row."""
    tables = {1: table([2.0, 3.0], [1.0, 2.0]), 2: table([2.0, 2.5], [1.0j, 3.0])}
    got = q_composite_table(tables, "Phi_res")
    lengths = {len(v) for v in got.values()}
    assert lengths == {3}
    assert set(got) == {
        "q", "n_modes", "peak", "peak_angle", "quadrature", "linear_bound",
        "psi_norm_1", "m_1", "psi_norm_2", "m_2",
    }


def test_a_mode_that_does_not_reach_a_surface_is_marked_not_dropped():
    tables = {1: table([2.0, 3.0], [1.0, 2.0]), 2: table([2.5], [3.0])}
    got = q_composite_table(tables, "Phi_res")
    row = int(np.argmin(np.abs(got["q"] - 2.5)))
    assert got["m_1"][row] == -1
    assert np.isnan(got["psi_norm_1"][row])
    assert got["m_2"][row] == 5
    assert got["n_modes"][row] == 1


def test_shared_only_keeps_the_surfaces_every_mode_reaches():
    tables = {1: table([2.0, 3.0], [1.0, 2.0]), 2: table([2.0, 2.5], [1.0j, 3.0])}
    got = q_composite_table(tables, "Phi_res", shared_only=True)
    assert got["q"].tolist() == [2.0]
    assert set(got["n_modes"].tolist()) == {2}


def test_shared_only_with_nothing_shared_is_an_error():
    tables = {1: table([2.0], [1.0]), 2: table([2.5], [1.0])}
    with pytest.raises(ValueError, match="no surface is resonant on every one"):
        q_composite_table(tables, "Phi_res", shared_only=True)


def test_the_peak_never_exceeds_the_bound_and_the_quadrature_never_does_either():
    rng = np.random.default_rng(11)
    tables = {}
    for n in (1, 2, 3, 4):
        q = np.array([12 / k for k in (6, 4, 3, 2)])
        tables[n] = table(q, rng.normal(size=q.size) + 1j * rng.normal(size=q.size))
    got = q_composite_table(tables, "Phi_res")
    assert np.all(got["peak"] <= got["linear_bound"] * (1 + 1e-12))
    assert np.all(got["quadrature"] <= got["linear_bound"] * (1 + 1e-12))


# --------------------------------------------------------------------------
# Combining delta_e
# --------------------------------------------------------------------------


def test_quadrature_and_linear_differ_and_quadrature_is_the_default():
    values = {1: 3.0, 2: 4.0}
    assert reduce_delta_e(values) == pytest.approx(5.0)
    assert reduce_delta_e(values, combination="linear") == pytest.approx(7.0)


def test_both_combinations_scale_with_a_uniform_weight():
    values = {1: 3.0, 2: 4.0}
    for how in ("quadrature", "linear"):
        plain = reduce_delta_e(values, combination=how)
        scaled = reduce_delta_e(values, combination=how, weights={1: 2.0, 2: 2.0})
        assert scaled == pytest.approx(2.0 * plain)


def test_a_negative_weight_is_refused_as_a_negative_value_would_be():
    """It cancels one mode against another exactly as a negative value does,
    and quadrature squares it away without noticing."""
    with pytest.raises(ValueError, match="weight for n = 1 is negative"):
        reduce_delta_e({1: 3.0, 2: 4.0}, weights={1: -1.0, 2: 1.0})


def test_the_combinations_are_exactly_the_two_that_are_offered():
    assert DELTA_E_COMBINATIONS == ("quadrature", "linear")


def test_a_negative_delta_e_is_refused():
    """It is the magnitude of a projection. A negative one means the caller
    passed something else, and the linear combination would quietly cancel
    against it."""
    with pytest.raises(ValueError, match="negative"):
        reduce_delta_e({1: 1.0, 2: -1.0}, combination="linear")


def test_a_complex_delta_e_is_refused_rather_than_silently_realified():
    """The legacy's 'coherent' branch wrapped its inputs in complex() and
    took a magnitude, which did nothing because they were already real."""
    with pytest.raises(ValueError, match="not real"):
        reduce_delta_e({1: 1.0 + 1.0j})


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_value_or_weight_is_refused(bad):
    with pytest.raises(ValueError, match="not finite"):
        reduce_delta_e({1: bad})
    with pytest.raises(ValueError, match="not finite"):
        reduce_delta_e({1: 1.0}, weights={1: bad})


def test_no_modes_is_an_error_not_a_zero():
    """A combination over nothing is not 0.0, which would read as a run with
    no edge overlap."""
    with pytest.raises(ValueError, match="nothing to combine"):
        reduce_delta_e({})


def test_partial_weights_are_refused_in_both_directions():
    with pytest.raises(ValueError, match="modes \\[2\\] have none"):
        reduce_delta_e({1: 1.0, 2: 1.0}, weights={1: 1.0})
    with pytest.raises(ValueError, match="name modes with no value"):
        reduce_delta_e({1: 1.0}, weights={1: 1.0, 5: 1.0})


def test_an_unknown_combination_is_refused():
    with pytest.raises(ValueError, match="quadrature.*linear"):
        reduce_delta_e({1: 1.0}, combination="coherent")
