"""Inboard indentation (bean shaping) in the Miller parameterization (#941).

``indentation`` adds ``b sin^2(theta) cos(theta)`` to R. It has to leave every
existing surface alone at b = 0, keep the four cardinal points where they
were, dent the high-field side exactly past the analytic onset
``b > (1 - alpha)^2/2``, and be refused where the curve would cross itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.equilibrium import MillerSurface
from vaft.process.equilibrium import evaluate_miller, fit_miller_surface

THETA = np.linspace(0.0, 2*np.pi, 400, endpoint=False)
BASE = dict(r=0.22, r0=0.9, z0=-0.03, kappa=1.7, delta=0.32)


def _surface(**kw) -> MillerSurface:
    return MillerSurface(**{**BASE, **kw})


def _inboard_curvature(b: float, delta: float = BASE["delta"]) -> float:
    """d2R/du2 at the inboard midplane (theta = pi + u), by finite differences."""
    h = 1e-4
    surface = _surface(delta=delta, indentation=b)
    r = np.array([evaluate_miller(surface, np.pi + u)[0] for u in (-h, 0.0, h)])
    return float((r[0] - 2*r[1] + r[2]) / h**2)


def test_zero_indentation_is_the_old_surface_to_the_bit():
    r, z = evaluate_miller(_surface(), THETA)
    alpha = np.arcsin(BASE["delta"])
    expected_r = BASE["r0"] + BASE["r"]*np.cos(THETA + alpha*np.sin(THETA))
    expected_z = BASE["z0"] + BASE["kappa"]*BASE["r"]*np.sin(THETA)
    assert np.array_equal(r, expected_r) and np.array_equal(z, expected_z)
    assert _surface().indentation == 0.0


def test_indentation_leaves_the_cardinal_points_in_place():
    cardinal = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    plain = np.column_stack(evaluate_miller(_surface(), cardinal))
    bean = np.column_stack(evaluate_miller(_surface(indentation=0.6), cardinal))
    np.testing.assert_allclose(bean, plain, atol=1e-15)


@pytest.mark.parametrize("delta", [0.0, 0.32, -0.3, 0.6])
def test_bean_onset_matches_the_analytic_criterion(delta):
    onset = (1 - np.arcsin(delta))**2 / 2
    assert _inboard_curvature(onset - 0.02, delta) > 0      # convex: R grows away from the midplane
    assert _inboard_curvature(onset + 0.02, delta) < 0      # concave: the midplane is dented outward
    expected = 2*BASE["r"]*((1 - np.arcsin(delta))**2/2 - (onset + 0.02))
    assert _inboard_curvature(onset + 0.02, delta) == pytest.approx(expected, rel=1e-3)


def test_outboard_stays_convex_below_its_bound():
    alpha = np.arcsin(BASE["delta"])
    bound = (1 + alpha)**2 / 2
    h = 1e-4
    for b, sign in ((bound - 0.05, -1), (bound + 0.05, +1)):
        r = [evaluate_miller(_surface(indentation=b), u)[0] for u in (-h, 0.0, h)]
        assert np.sign((r[0] - 2*r[1] + r[2]) / h**2) == sign


def test_self_intersecting_indentation_is_refused():
    limit = -np.sqrt(1 - BASE["delta"]**2)
    with pytest.raises(ValueError, match="indentation"):
        evaluate_miller(_surface(indentation=limit), THETA)
    r, z = evaluate_miller(_surface(indentation=limit + 1e-3), THETA)
    # Still a simple curve: at every height the outboard side is outboard of the inboard one.
    upper = (THETA > -1e-12) & (THETA < np.pi/2)
    for t in THETA[upper][1:]:
        r_out = evaluate_miller(_surface(indentation=limit + 1e-3), t)[0]
        r_in = evaluate_miller(_surface(indentation=limit + 1e-3), np.pi - t)[0]
        assert r_out > r_in


@pytest.mark.parametrize("squareness", [False, True])
def test_indented_surface_round_trips_through_the_fitter(squareness):
    truth = _surface(zeta=0.1 if squareness else 0.0, indentation=0.35)
    fit = fit_miller_surface(evaluate_miller(truth, THETA), squareness=squareness, indentation=True)
    assert fit.accepted, fit.reason
    assert fit.surface.indentation == pytest.approx(0.35, abs=2e-3)
    assert fit.surface.delta == pytest.approx(0.32, abs=2e-3)
    assert fit.surface.kappa == pytest.approx(1.7, abs=2e-3)


def test_plain_surface_fits_zero_indentation():
    fit = fit_miller_surface(evaluate_miller(_surface(), THETA), indentation=True)
    assert fit.accepted and abs(fit.surface.indentation) < 1e-3


def test_indentation_is_opt_in():
    bean = evaluate_miller(_surface(indentation=0.6), THETA)
    default = fit_miller_surface(bean)
    assert default.surface.indentation == 0.0
    assert not default.accepted                       # a bean is not five-parameter Miller-like
    assert fit_miller_surface(bean, indentation=True).accepted


def test_default_fit_is_unchanged_by_the_indentation_work():
    """Pinned from origin/develop before #941 (bit-identical there on macOS)."""
    rng = np.random.default_rng(7)
    theta = np.linspace(0, 2*np.pi, 300, endpoint=False)
    r, z = evaluate_miller(MillerSurface(0.22, 0.9, -0.03, 1.7, 0.32), theta)
    r = r + 1e-3*rng.standard_normal(r.size); z = z + 1e-3*rng.standard_normal(z.size)
    s = fit_miller_surface((r, z)).surface
    expected = (0.21998494211178976, 0.8997816786170715, -0.030157138796743203, 1.700347712135167, 0.31878788793594054)
    np.testing.assert_allclose((s.r, s.r0, s.z0, s.kappa, s.delta), expected, rtol=1e-9)
    assert s.zeta == 0.0 and s.indentation == 0.0


def test_bean_residuals_are_orthogonal_to_the_fitted_curve():
    """A wrong g' or g'' in the Newton projection would tilt these away from the normal."""
    rng = np.random.default_rng(3)
    truth = _surface(indentation=0.5)
    r, z = evaluate_miller(truth, THETA)
    r = r + 5e-4*rng.standard_normal(r.size); z = z + 5e-4*rng.standard_normal(z.size)
    fit = fit_miller_surface((r, z), indentation=True)
    dense = np.linspace(0, 2*np.pi, 200000, endpoint=False)
    curve = np.column_stack(evaluate_miller(fit.surface, dense))
    tangent = np.gradient(curve, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    from scipy.spatial import cKDTree
    index = cKDTree(curve).query(fit.reconstructed.points)[1]
    residual = fit.contour.points - fit.reconstructed.points
    length = np.linalg.norm(residual, axis=1)
    keep = length > 1e-4
    cosine = np.abs(np.sum(residual[keep]*tangent[index[keep]], axis=1)) / length[keep]
    assert np.max(cosine) < 0.02
