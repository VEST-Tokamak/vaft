"""Rational-surface resolution and its equilibrium/profile overlays (issue #506).

One resolver, :func:`vaft.process.equilibrium.rational_surfaces`, answers
``q_target -> radius``; the 2-D flux-surface contours and the 1-D profile
markers of :mod:`vaft.plot` are drawn from its records and never search ``q``
themselves.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vaft.process.equilibrium import (
    RATIONAL_SURFACE_ABSENT,
    RATIONAL_SURFACE_PRESENT,
    find_rational_surfaces,
    rational_surfaces,
)


PSI = np.linspace(0.0, 1.0, 101)


def _one(surfaces, q_target):
    (match,) = [s for s in surfaces if s.q_target == pytest.approx(q_target)]
    return match


# --- resolver ------------------------------------------------------------------


def test_a_monotonic_profile_has_one_root_where_q_crosses():
    q = 1.0 + 3.0 * PSI  # q = 2 at psi_N = 1/3
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0])
    assert surface.status == RATIONAL_SURFACE_PRESENT and surface.present
    (root,) = surface.roots
    assert root.psi_norm == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert root.rho_pol_norm == pytest.approx(np.sqrt(1.0 / 3.0))
    assert root.rho_tor_norm is None and root.root_index == 0


def test_a_value_the_profile_never_reaches_is_absent_not_invented():
    q = 1.0 + 3.0 * PSI
    (surface,) = rational_surfaces(PSI, q, q_targets=[5.0])
    assert surface.status == RATIONAL_SURFACE_ABSENT and not surface.present
    assert surface.roots == ()


def test_an_exact_node_is_one_root():
    q = np.linspace(1.0, 3.0, 5)  # q = 2 exactly at node 2
    psi = np.linspace(0.0, 1.0, 5)
    (surface,) = rational_surfaces(psi, q, q_targets=[2.0])
    assert [r.psi_norm for r in surface.roots] == [0.5]


@pytest.mark.parametrize("target, where", [(1.0, 0.0), (4.0, 1.0)])
def test_roots_on_the_axis_and_on_the_edge_are_kept(target, where):
    q = 1.0 + 3.0 * PSI
    (surface,) = rational_surfaces(PSI, q, q_targets=[target])
    assert [r.psi_norm for r in surface.roots] == [where]


def test_reversed_shear_returns_both_roots_ordered_outward():
    q = 3.0 - 6.0 * PSI * (1.0 - PSI)  # min 1.5 at psi_N = 0.5
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0])
    positions = [r.psi_norm for r in surface.roots]
    assert len(positions) == 2 and positions[0] < 0.5 < positions[1]
    assert [r.root_index for r in surface.roots] == [0, 1]
    for position in positions:
        assert 3.0 - 6.0 * position * (1.0 - position) == pytest.approx(2.0, abs=1e-3)


def test_several_targets_come_back_one_record_each_ascending():
    q = 1.0 + 3.0 * PSI
    surfaces = rational_surfaces(PSI, q, q_targets=[3.0, 1.5, 2.0])
    assert [s.q_target for s in surfaces] == [1.5, 2.0, 3.0]
    assert all(s.present for s in surfaces)


def test_harmonics_with_one_ratio_share_one_surface_and_keep_both_names():
    q = 1.0 + 3.0 * PSI
    surfaces = rational_surfaces(PSI, q, resonances=[(2, 1), (4, 2), (3, 2)])
    assert [s.q_target for s in surfaces] == [1.5, 2.0]
    q2 = _one(surfaces, 2.0)
    assert q2.harmonics == ((2, 1), (4, 2)) and not q2.requested_q
    assert len(q2.roots) == 1
    # a plain value meets the harmonics that reduce to it
    merged = rational_surfaces(PSI, q, q_targets=[2.0], resonances=[(2, 1)])
    assert len(merged) == 1 and merged[0].requested_q and merged[0].harmonics == ((2, 1),)


def test_a_nan_gap_is_never_bridged():
    q = 1.0 + 3.0 * PSI
    q[30:40] = np.nan  # q = 2 (psi_N = 1/3) falls inside the gap
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0])
    assert surface.status == RATIONAL_SURFACE_ABSENT
    # find_rational_surfaces drops the samples and interpolates across them
    assert find_rational_surfaces(PSI, q, 1, m_range=(2, 2))["psi_n_rational"].size == 1


def test_nothing_is_extrapolated_beyond_the_profile():
    psi = np.linspace(0.0, 0.8, 81)
    q = 1.0 + 3.0 * psi  # reaches 3.4 at the last sample; 3.5 lies beyond
    (surface,) = rational_surfaces(psi, q, q_targets=[3.5])
    assert surface.status == RATIONAL_SURFACE_ABSENT


def test_rho_tor_norm_is_carried_only_when_supplied():
    q = 1.0 + 3.0 * PSI
    rho_tor = PSI ** 0.7
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0], rho_tor_norm=rho_tor)
    (root,) = surface.roots
    assert root.rho_tor_norm == pytest.approx(np.interp(root.psi_norm, PSI, rho_tor))
    assert root.rho_pol_norm == pytest.approx(np.sqrt(root.psi_norm))


def test_the_sign_of_q_does_not_decide_resonance():
    q = 1.0 + 3.0 * PSI
    positive = rational_surfaces(PSI, q, q_targets=[2.0], resonances=[(3, 2)])
    negative = rational_surfaces(PSI, -q, q_targets=[2.0], resonances=[(3, -2)])
    assert [[r.psi_norm for r in s.roots] for s in positive] == [
        [r.psi_norm for r in s.roots] for s in negative
    ]


@pytest.mark.parametrize("kwargs", [
    {}, {"q_targets": [0.0]}, {"q_targets": [np.nan]}, {"resonances": [(2, 0)]},
    {"resonances": [(2.5, 1)]}, {"resonances": [2]},
])
def test_bad_requests_are_refused(kwargs):
    with pytest.raises(ValueError):
        rational_surfaces(PSI, 1.0 + PSI, **kwargs)


def test_a_non_increasing_grid_is_refused():
    with pytest.raises(ValueError, match="increase"):
        rational_surfaces(PSI[::-1], 1.0 + PSI, q_targets=[1.5])
