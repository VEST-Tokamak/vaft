"""The two published approximations to the virial closure coefficient alpha.

`alpha = 2<R Bz^2>/<R Bp^2>` is a volume integral over the plasma, so it needs
the internal poloidal field -- which a filament or current-element
reconstruction does not determine well. Bongard et al. (2016) give two
approximations that avoid it, and this module checks both against the volume
integral of an analytic Solov'ev equilibrium, which has a real field and
therefore a real alpha:

* `alpha_1 = 2 kappa^2 / (1 + kappa^2)` from the boundary elongation alone;
* `alpha_2` from the field in a thin annulus conformal to the LCFS.

The tolerances here are measured, not aspirational: they are the errors the
sweep below actually produces, rounded outward. They are regression bounds on
the implementation, not claims about how accurate the physics is.
"""

import numpy as np
import pytest
from skimage import measure

from vaft.data.equilibrium import SolovevConstraint
from vaft.formula.equilibrium import (
    virial_alpha_approx_from_kappa,
    virial_alpha_from_R_Bz_Bp_dl,
    virial_li_from_S_alpha_mu,
)
from vaft.process.equilibrium import (
    contour_shape_parameters,
    efit_virial_volume_integrals,
    evaluate_solovev,
    scale_boundary_conformal,
    solve_solovev_constraints,
    virial_alpha_conformal_annulus,
    virial_alpha_thin_annulus,
)

R0 = 1.0
#: (elongation, inverse aspect ratio) pairs spanning the VEST range and beyond.
SHAPES = [(1.0, 0.5), (1.2, 0.5), (1.4, 0.5), (1.6, 0.5), (1.8, 0.5), (2.0, 0.5),
          (1.0, 0.7), (1.2, 0.7), (1.4, 0.7), (1.6, 0.7), (1.8, 0.7), (2.0, 0.7)]


def _solovev(eps, kappa, delta=0.3, n_points=24):
    """Solov'ev equilibrium whose psi = 0 contour approximates the target shape."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    a = eps * R0
    r_b = R0 + a * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    z_b = kappa * a * np.sin(theta)
    constraints = [
        SolovevConstraint(float(r), float(z), "psi", 0.0) for r, z in zip(r_b, z_b)
    ]
    return solve_solovev_constraints(
        constraints, pprime=-1200.0, ffprime=0.08, rref=R0
    )


def _largest_closed_contour(psi, r, z, level=0.0):
    best = None
    for seg in measure.find_contours(psi, level=level):
        r_pts = np.interp(seg[:, 0], np.arange(r.size), r)
        z_pts = np.interp(seg[:, 1], np.arange(z.size), z)
        if abs(r_pts[0] - r_pts[-1]) > 1e-9 or r_pts.size < 50:
            continue
        area = 0.5 * abs(np.sum(r_pts[:-1] * z_pts[1:] - r_pts[1:] * z_pts[:-1]))
        if best is None or area > best[0]:
            best = (area, r_pts, z_pts)
    assert best is not None, "the Solov'ev psi = 0 level has no closed contour"
    return best[1], best[2]


def _case(eps, kappa, nr=257, nz=385):
    """Analytic equilibrium plus everything the alpha estimates need."""
    model = _solovev(eps, kappa)
    a = eps * R0
    r = np.linspace(max(1e-3, R0 - 1.6 * a), R0 + 1.6 * a, nr)
    z = np.linspace(-2.0 * kappa * a, 2.0 * kappa * a, nz)
    r_mesh, z_mesh = np.meshgrid(r, z, indexing="ij")
    field = evaluate_solovev(model, r_mesh, z_mesh)
    r_bdry, z_bdry = _largest_closed_contour(field["psi"], r, z)
    boundary_field = evaluate_solovev(model, r_bdry, z_bdry)
    return {
        "r_mesh": r_mesh, "z_mesh": z_mesh,
        "b_r": field["b_r"], "b_z": field["b_z"],
        "r_bdry": r_bdry, "z_bdry": z_bdry,
        "b_r_bdry": boundary_field["b_r"], "b_z_bdry": boundary_field["b_z"],
        "kappa": contour_shape_parameters(r_bdry, z_bdry)["elongation"],
        "alpha_true": efit_virial_volume_integrals(
            r_mesh, z_mesh, r_bdry, z_bdry, field["b_r"], field["b_z"]
        )["alpha"],
    }


# --------------------------------------------------------------------------
# alpha_1: from the boundary elongation
# --------------------------------------------------------------------------

def test_alpha_1_is_unity_for_a_circular_boundary_and_two_when_infinitely_elongated():
    assert virial_alpha_approx_from_kappa(1.0) == pytest.approx(1.0)
    assert virial_alpha_approx_from_kappa(1e6) == pytest.approx(2.0, abs=1e-9)


def test_alpha_1_increases_with_elongation():
    values = [virial_alpha_approx_from_kappa(k) for k in (1.0, 1.2, 1.5, 2.0, 3.0)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert all(1.0 <= v <= 2.0 for v in values)


def test_alpha_1_rejects_a_degenerate_elongation():
    assert np.isnan(virial_alpha_approx_from_kappa(np.nan))


@pytest.mark.parametrize("kappa,eps", SHAPES)
def test_alpha_1_matches_the_solovev_volume_integral_within_ten_percent(kappa, eps):
    """The accuracy Bongard et al. report, measured on an analytic equilibrium.

    The error is systematically negative and grows as the plasma gets rounder
    and fatter -- alpha_1 knows only the elongation, so it cannot see the
    aspect-ratio contribution that keeps the true alpha above 1 even at
    kappa = 1.
    """
    case = _case(eps, kappa)
    approx = virial_alpha_approx_from_kappa(case["kappa"])
    error = (approx - case["alpha_true"]) / case["alpha_true"]
    assert abs(error) < 0.10, f"kappa={kappa} eps={eps}: alpha_1 error {error:.1%}"


# --------------------------------------------------------------------------
# alpha_2: from a thin annulus conformal to the LCFS
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kappa,eps", SHAPES)
def test_alpha_2_matches_the_solovev_volume_integral_within_ten_percent(kappa, eps):
    case = _case(eps, kappa)
    result = virial_alpha_conformal_annulus(
        case["r_mesh"], case["z_mesh"], case["b_r"], case["b_z"],
        case["r_bdry"], case["z_bdry"], thickness=0.1,
    )
    assert result["valid"], result["reason"]
    error = (result["alpha"] - case["alpha_true"]) / case["alpha_true"]
    assert abs(error) < 0.10, f"kappa={kappa} eps={eps}: alpha_2 error {error:.1%}"


def test_alpha_2_converges_to_the_thin_annulus_limit_as_thickness_shrinks():
    """The t -> 0 limit is exact only with the support-function weight.

    A conformal annulus has local width t * (x - c).n, not a uniform t, so the
    limiting line integral carries that weight. Weighting every segment equally
    instead converges to a different number entirely -- the check below pins
    both halves of that statement.
    """
    case = _case(0.7, 1.6, nr=385, nz=577)
    thin = virial_alpha_thin_annulus(
        case["r_bdry"], case["z_bdry"], case["b_r_bdry"], case["b_z_bdry"]
    )
    assert np.isfinite(thin)

    gaps = []
    for thickness in (0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01):
        result = virial_alpha_conformal_annulus(
            case["r_mesh"], case["z_mesh"], case["b_r"], case["b_z"],
            case["r_bdry"], case["z_bdry"], thickness=thickness,
        )
        assert result["valid"], f"t={thickness}: {result['reason']}"
        gaps.append(abs(result["alpha"] - thin))

    assert all(b < a for a, b in zip(gaps, gaps[1:])), f"non-monotone: {gaps}"
    assert gaps[-1] < 0.01 * abs(thin), f"t=0.01 gap {gaps[-1]:.5f} vs alpha {thin:.5f}"

    uniform = virial_alpha_thin_annulus(
        case["r_bdry"], case["z_bdry"], case["b_r_bdry"], case["b_z_bdry"],
        mode="uniform",
    )
    assert abs(uniform - thin) > 10.0 * gaps[-1]


def test_alpha_2_abstains_rather_than_describing_the_grid():
    """A thin annulus on a coarse grid resolves nothing; NaN beats a number."""
    case = _case(0.7, 1.6, nr=129, nz=193)
    result = virial_alpha_conformal_annulus(
        case["r_mesh"], case["z_mesh"], case["b_r"], case["b_z"],
        case["r_bdry"], case["z_bdry"], thickness=1e-4,
    )
    assert not result["valid"]
    assert np.isnan(result["alpha"])
    assert "cells" in result["reason"]
    assert result["n_cells"] > 0.0  # the count that triggered it is reported


def test_alpha_2_abstains_when_the_boundary_leaves_the_grid():
    case = _case(0.7, 1.6, nr=129, nz=193)
    r_small = np.linspace(0.9, 1.1, 41)
    z_small = np.linspace(-0.1, 0.1, 41)
    r_mesh, z_mesh = np.meshgrid(r_small, z_small, indexing="ij")
    result = virial_alpha_conformal_annulus(
        r_mesh, z_mesh, np.zeros_like(r_mesh), np.ones_like(r_mesh),
        case["r_bdry"], case["z_bdry"], thickness=0.1,
    )
    assert not result["valid"] and np.isnan(result["alpha"])
    assert "beyond the field grid" in result["reason"]


@pytest.mark.parametrize("thickness", [0.0, 1.0, 1.5, -0.1, np.nan])
def test_alpha_2_rejects_a_thickness_outside_the_open_unit_interval(thickness):
    case = _case(0.7, 1.6, nr=129, nz=193)
    result = virial_alpha_conformal_annulus(
        case["r_mesh"], case["z_mesh"], case["b_r"], case["b_z"],
        case["r_bdry"], case["z_bdry"], thickness=thickness,
    )
    assert not result["valid"] and np.isnan(result["alpha"])


def test_alpha_2_is_invariant_to_boundary_orientation():
    """A reversed contour is the same plasma; both estimates must not notice."""
    case = _case(0.7, 1.6, nr=129, nz=193)
    kwargs = dict(thickness=0.15)
    ccw = virial_alpha_conformal_annulus(
        case["r_mesh"], case["z_mesh"], case["b_r"], case["b_z"],
        case["r_bdry"], case["z_bdry"], **kwargs,
    )
    cw = virial_alpha_conformal_annulus(
        case["r_mesh"], case["z_mesh"], case["b_r"], case["b_z"],
        case["r_bdry"][::-1], case["z_bdry"][::-1], **kwargs,
    )
    assert ccw["valid"] and cw["valid"]
    assert cw["alpha"] == pytest.approx(ccw["alpha"], rel=1e-9)

    thin_ccw = virial_alpha_thin_annulus(
        case["r_bdry"], case["z_bdry"], case["b_r_bdry"], case["b_z_bdry"]
    )
    thin_cw = virial_alpha_thin_annulus(
        case["r_bdry"][::-1], case["z_bdry"][::-1],
        case["b_r_bdry"][::-1], case["b_z_bdry"][::-1],
    )
    assert thin_cw == pytest.approx(thin_ccw, rel=1e-9)


def test_alpha_2_scales_out_of_the_field_magnitude():
    """alpha is a ratio: doubling B leaves it alone."""
    case = _case(0.7, 1.6, nr=129, nz=193)
    args = (case["r_mesh"], case["z_mesh"])
    base = virial_alpha_conformal_annulus(
        *args, case["b_r"], case["b_z"], case["r_bdry"], case["z_bdry"], thickness=0.15
    )
    scaled = virial_alpha_conformal_annulus(
        *args, 3.0 * case["b_r"], 3.0 * case["b_z"],
        case["r_bdry"], case["z_bdry"], thickness=0.15,
    )
    assert scaled["alpha"] == pytest.approx(base["alpha"], rel=1e-12)


def test_thin_annulus_rejects_an_unknown_mode():
    case = _case(0.7, 1.6, nr=129, nz=193)
    with pytest.raises(ValueError, match="conformal"):
        virial_alpha_thin_annulus(
            case["r_bdry"], case["z_bdry"], case["b_r_bdry"], case["b_z_bdry"],
            mode="psi_band",
        )


# --------------------------------------------------------------------------
# supporting geometry
# --------------------------------------------------------------------------

def test_conformal_scaling_preserves_shape_and_shrinks_area_quadratically():
    theta = np.linspace(0.0, 2.0 * np.pi, 201)
    r_b = 1.0 + 0.3 * np.cos(theta)
    z_b = 0.5 * np.sin(theta)
    scale = 0.8
    r_in, z_in = scale_boundary_conformal(r_b, z_b, scale)

    def area(r, z):
        return 0.5 * abs(np.sum(r[:-1] * z[1:] - r[1:] * z[:-1]))

    assert area(r_in, z_in) == pytest.approx(scale**2 * area(r_b, z_b), rel=1e-9)
    # Same shape: elongation is scale-invariant.
    assert contour_shape_parameters(r_in, z_in)["elongation"] == pytest.approx(
        contour_shape_parameters(r_b, z_b)["elongation"], rel=1e-9
    )


def test_conformal_scaling_rejects_a_nonpositive_scale():
    theta = np.linspace(0.0, 2.0 * np.pi, 51)
    with pytest.raises(ValueError, match="positive"):
        scale_boundary_conformal(np.cos(theta) + 2.0, np.sin(theta), 0.0)


def test_thin_limit_quadrature_abstains_on_a_zero_field():
    zeros = np.zeros(8)
    assert np.isnan(virial_alpha_from_R_Bz_Bp_dl(np.ones(8), zeros, zeros, np.ones(8)))


# --------------------------------------------------------------------------
# why the choice of alpha matters
# --------------------------------------------------------------------------

def test_an_alpha_error_is_amplified_in_the_bongard_li():
    """`l_i = (S1 + S2 - 2 mui - 3 S3)/(3 alpha - 2)` divides by `3 alpha - 2`.

    The numerator does not depend on alpha, so the local sensitivity is
    `dln l_i/dln alpha = -3 alpha/(3 alpha - 2)`, about -1.9 near alpha = 1.4.
    A *finite* 10% error is the secant rather than the derivative,
    `3 alpha e/(3 alpha (1 + e) - 2)`, which is 1.60 there -- so 10% in alpha
    costs ~16% in l_i, approaching ~19% for small errors. Either way it is an
    amplification, which is why the two alpha approximations are kept separate
    and compared rather than one being picked silently.
    """
    S1, S2, S3, mui = 2.2, -0.4, 0.6, -0.12
    alpha = 1.4

    def amplification(rel_error):
        base = virial_li_from_S_alpha_mu(S1, S2, S3, alpha, mui)
        moved = virial_li_from_S_alpha_mu(S1, S2, S3, alpha * (1.0 + rel_error), mui)
        return abs((moved - base) / base) / abs(rel_error)

    # Finite 10% step: the exact secant.
    li_change = 3.0 * alpha * 0.10 / (3.0 * alpha * 1.10 - 2.0)
    assert amplification(0.10) == pytest.approx(li_change / 0.10, rel=1e-9)
    assert amplification(0.10) == pytest.approx(1.603, abs=1e-3)

    # Vanishing step: the logarithmic derivative.
    derivative = 3.0 * alpha / (3.0 * alpha - 2.0)
    assert amplification(1e-7) == pytest.approx(derivative, rel=1e-4)
    assert derivative == pytest.approx(1.909, abs=1e-3)

    # Both exceed one: an alpha error never shrinks on its way into l_i.
    assert amplification(0.10) > 1.5 and derivative > 1.5
