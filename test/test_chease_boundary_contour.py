"""Choosing the plasma boundary when an equilibrium offers several contours.

A limited equilibrium offers one contour at ``target_psin``.  A diverted one
offers the core surface *and* the divertor lobe, and ``find_contours`` returns
the lobe first whenever it sits at lower Z -- which on a lower single null it
does.  Handing that lobe to CHEASE as the boundary is the defect
:class:`~vaft.code.chease.BoundaryContourPolicy` exists to prevent, so the
tests below build a diverted psi map where the naive ``contours[0]`` really is
the lobe rather than asserting the choice against a map that never had one.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.code.chease import (
    DEFAULT_BOUNDARY_CONTOUR_POLICY,
    PENALIZING_BOUNDARY_CONTOUR_POLICY,
    BoundaryContourPolicy,
    _polygon_area,
    _target_boundary,
    score_boundary_contour,
)

CORE_AXIS = (1.0, 0.0)
LOBE_CENTRE = (0.45, -1.5)


def _geqdsk(
    psi: np.ndarray,
    *,
    r_left: float,
    r_dim: float,
    z_dim: float,
    axis: tuple[float, float],
    boundary: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, object]:
    nw, nh = psi.shape
    r_bbbs, z_bbbs = boundary if boundary is not None else (np.array([]), np.array([]))
    return {
        "NW": nw,
        "NH": nh,
        "RLEFT": r_left,
        "RDIM": r_dim,
        "ZMID": 0.0,
        "ZDIM": z_dim,
        "PSIRZ": psi,
        "SIMAG": 0.0,
        "SIBRY": 1.0,
        "RMAXIS": axis[0],
        "ZMAXIS": axis[1],
        "RBBBS": r_bbbs,
        "ZBBBS": z_bbbs,
    }


def _elliptic(r: np.ndarray, z: np.ndarray, centre, semi) -> np.ndarray:
    """Normalized flux of one nested-ellipse family, 0 at ``centre``."""
    rr, zz = np.meshgrid(r, z, indexing="ij")
    return ((rr - centre[0]) / semi[0]) ** 2 + ((zz - centre[1]) / semi[1]) ** 2


def _diverted_map() -> dict[str, object]:
    """A core plasma plus a disjoint lower divertor lobe at the same psi_N.

    ``psi`` is the pointwise minimum of two ellipse families, so every level
    below 1 has two contours: the core surface around ``CORE_AXIS`` and a small
    closed lobe around ``LOBE_CENTRE``.  The lobe sits at low Z, which is where
    ``skimage.measure.find_contours`` starts, so it is returned first.
    """
    r = np.linspace(0.05, 2.05, 201)
    z = np.linspace(-2.0, 2.0, 201)
    core = _elliptic(r, z, CORE_AXIS, (0.6, 1.0))
    lobe = _elliptic(r, z, LOBE_CENTRE, (0.12, 0.15))
    psi = np.minimum(core, lobe)
    return _geqdsk(psi, r_left=0.05, r_dim=2.0, z_dim=4.0, axis=CORE_AXIS, boundary=None)


def _raw_contours(geqdsk, target_psin: float) -> list[np.ndarray]:
    measure = pytest.importorskip("skimage.measure")
    r = np.linspace(geqdsk["RLEFT"], geqdsk["RLEFT"] + geqdsk["RDIM"], geqdsk["NW"])
    z = np.linspace(-geqdsk["ZDIM"] / 2.0, geqdsk["ZDIM"] / 2.0, geqdsk["NH"])
    out = []
    for contour in measure.find_contours(np.asarray(geqdsk["PSIRZ"]).T, target_psin):
        out.append(
            np.column_stack(
                [
                    np.interp(contour[:, 1], np.arange(r.size), r),
                    np.interp(contour[:, 0], np.arange(z.size), z),
                ]
            )
        )
    return out


def test_the_first_contour_of_a_diverted_map_really_is_the_divertor_lobe():
    # Guards the premise of every test below: without it they would prove
    # nothing, because a one-contour map has no wrong branch to pick.
    geqdsk = _diverted_map()
    contours = _raw_contours(geqdsk, 0.99)
    assert len(contours) == 2
    first = contours[0]
    assert abs(np.mean(first[:, 1]) - LOBE_CENTRE[1]) < 0.2
    assert _polygon_area(first) < 0.1 * max(_polygon_area(c) for c in contours)


def test_target_boundary_picks_the_core_surface_not_the_lobe():
    geqdsk = _diverted_map()
    chosen = _target_boundary(geqdsk, 0.99)
    areas = [_polygon_area(c) for c in _raw_contours(geqdsk, 0.99)]
    assert _polygon_area(chosen) == pytest.approx(max(areas), rel=1e-9)
    # The property that actually matters: the returned curve encloses the axis.
    assert chosen[:, 0].min() < CORE_AXIS[0] < chosen[:, 0].max()
    assert chosen[:, 1].min() < CORE_AXIS[1] < chosen[:, 1].max()


def _large_lobe_map() -> dict[str, object]:
    """A compact core beside a divertor lobe of *larger* enclosed area.

    The discriminating case: on area alone the lobe wins, so a chooser that
    ranks by size picks it.  Physically this is a near-separatrix level whose
    legs sweep a wide private-flux region while the core surface is small.
    """
    r = np.linspace(0.05, 2.05, 241)
    z = np.linspace(-2.0, 2.0, 241)
    core = _elliptic(r, z, CORE_AXIS, (0.25, 0.35))
    lobe = _elliptic(r, z, (1.0, -1.2), (0.8, 0.5))
    psi = np.minimum(core, lobe)
    return _geqdsk(psi, r_left=0.05, r_dim=2.0, z_dim=4.0, axis=CORE_AXIS, boundary=None)


def test_the_axis_bonus_and_not_the_area_is_what_decides_it():
    geqdsk = _large_lobe_map()
    contours = _raw_contours(geqdsk, 0.99)
    assert len(contours) == 2
    by_area = sorted(contours, key=_polygon_area)
    lobe, core = by_area[1], by_area[0]
    # Premise: here the lobe really is the larger of the two.
    assert abs(np.mean(lobe[:, 1]) + 1.2) < 0.2
    assert _polygon_area(lobe) > _polygon_area(core)

    # Area alone picks the lobe; the axis bonus picks the plasma.
    no_bonus = BoundaryContourPolicy(axis_bonus=0.0, closed_bonus=0.0)
    assert score_boundary_contour(lobe, CORE_AXIS, no_bonus)[0] > score_boundary_contour(
        core, CORE_AXIS, no_bonus
    )[0]
    assert score_boundary_contour(core, CORE_AXIS)[0] > score_boundary_contour(
        lobe, CORE_AXIS
    )[0]

    chosen = _target_boundary(geqdsk, 0.99)
    assert _polygon_area(chosen) == pytest.approx(_polygon_area(core), rel=1e-9)
    assert chosen[:, 1].min() < CORE_AXIS[1] < chosen[:, 1].max()


def _straddling_axis_map():
    """A contour that crosses R = 0, which no plasma boundary does."""
    r = np.linspace(-0.5, 2.0, 201)
    z = np.linspace(-1.5, 1.5, 201)
    psi = _elliptic(r, z, (0.3, 0.0), (0.6, 0.8))
    stored = (
        np.array([0.9, 1.1, 1.1, 0.9, 0.9]),
        np.array([-0.2, -0.2, 0.2, 0.2, -0.2]),
    )
    return _geqdsk(psi, r_left=-0.5, r_dim=2.5, z_dim=3.0, axis=(0.3, 0.0), boundary=stored)


def test_a_contour_crossing_r_zero_is_rejected_and_the_stored_boundary_is_used():
    geqdsk = _straddling_axis_map()
    contours = _raw_contours(geqdsk, 0.99)
    assert contours and min(np.mean(c[:, 0] > 0.0) for c in contours) < 0.95
    chosen = _target_boundary(geqdsk, 0.99)
    # The g-file's own RBBBS/ZBBBS, five points, not a 0.99 contour.
    assert chosen.shape == (5, 2)
    assert np.all(chosen[:, 0] > 0.0)


def test_the_penalizing_policy_keeps_the_least_bad_contour_instead():
    geqdsk = _straddling_axis_map()
    chosen = _target_boundary(geqdsk, 0.99, PENALIZING_BOUNDARY_CONTOUR_POLICY)
    assert chosen.shape[0] > 5
    assert chosen[:, 0].min() < 0.0  # the very contour "reject" refuses


def test_every_candidate_rejected_and_no_stored_boundary_is_an_error():
    geqdsk = _straddling_axis_map()
    geqdsk["RBBBS"] = np.array([])
    geqdsk["ZBBBS"] = np.array([])
    with pytest.raises(ValueError, match="No usable target boundary contour"):
        _target_boundary(geqdsk, 0.99)


def test_the_policy_refuses_a_rule_it_does_not_implement():
    with pytest.raises(ValueError, match="positive_r_rule"):
        BoundaryContourPolicy(positive_r_rule="ignore")
    with pytest.raises(ValueError, match="positive_r_fraction"):
        BoundaryContourPolicy(positive_r_fraction=95.0)


def test_the_config_carries_the_policy_to_the_boundary_choice():
    from vaft.code.chease import CHEASEConfig

    assert CHEASEConfig().boundary_contour_policy == DEFAULT_BOUNDARY_CONTOUR_POLICY
    config = CHEASEConfig(boundary_contour_policy=PENALIZING_BOUNDARY_CONTOUR_POLICY)
    assert config.boundary_contour_policy.positive_r_rule == "penalize"
