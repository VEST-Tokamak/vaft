"""Prescribed X-point topology through the Solov'ev constraint path (#938).

The constraint solve used to be hard-wired to the five-term up-down symmetric
basis, so an X-point could only come from :func:`solovev_example`'s private
Cerfon-Freidberg solve.  These tests hold the generalized path to what #938
asks of a diverted analytic equilibrium: the prescribed null is a saddle, not
just a stationary point; the separatrix passes through it; a single null has
no mirrored twin; and the gridded record recovers the null within a
resolution-dependent error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0 as MU0

from vaft.data.equilibrium import SolovevConstraint, SolovevEquilibrium
from vaft.process.equilibrium import (
    evaluate_solovev,
    find_stationary_points,
    solovev_example,
    solovev_shape_constraints,
    solovev_to_equilibrium,
    solovev_xpoint_constraints,
    solve_solovev_constraints,
)

R0, A_MINOR, KAPPA, DELTA = 0.4, 0.235, 1.7, 0.35
PSI0, A_CF = 0.01, 0.9


def _solve(topology: str, *, x_point=None) -> SolovevEquilibrium:
    basis = "cerfon_freidberg" if "single_null" in topology else "cerfon_freidberg_even"
    constraints = solovev_shape_constraints(
        major_radius=R0, minor_radius=A_MINOR, elongation=KAPPA, triangularity=DELTA,
        topology=topology, x_point=x_point,
    )
    return solve_solovev_constraints(
        constraints, pprime=-PSI0*(1 - A_CF)/(MU0*R0**4), ffprime=-PSI0*A_CF/R0**2,
        rref=R0, f_boundary=0.1*R0, basis=basis,
    )


def _grid(n: int = 129):
    half = 1.1*KAPPA*A_MINOR + 0.35*A_MINOR
    r = np.linspace(R0 - 1.3*A_MINOR, R0 + 1.3*A_MINOR, n)
    z = np.linspace(-half, half, int(np.ceil(n*2*half/(2.6*A_MINOR))) | 1)
    return r, z


def _saddle(model: SolovevEquilibrium, r: float, z: float) -> dict[str, float]:
    v = evaluate_solovev(model, r, z)
    return {
        "psi": float(v["psi"]), "grad": float(np.hypot(v["dpsi_dr"], v["dpsi_dz"])),
        "det": float(v["d2psi_dr2"]*v["d2psi_dz2"] - v["d2psi_drdz"]**2),
    }


DEFAULT_X = (R0 - 1.1*DELTA*A_MINOR, 1.1*KAPPA*A_MINOR)


def test_second_derivatives_match_finite_differences_in_every_basis():
    rng = np.random.default_rng(3)
    for basis, size in (("classic", 5), ("cerfon_freidberg_even", 7), ("cerfon_freidberg", 12)):
        model = SolovevEquilibrium(rng.normal(size=size)*1e-3, -800.0, 0.02, R0, basis=basis)
        r, z, h = 0.45, 0.13, 1e-5
        v = evaluate_solovev(model, r, z)
        def f(rr, zz, key):
            return float(evaluate_solovev(model, rr, zz)[key])
        assert v["d2psi_dr2"] == pytest.approx((f(r+h, z, "dpsi_dr")-f(r-h, z, "dpsi_dr"))/(2*h), rel=1e-6, abs=1e-9)
        assert v["d2psi_dz2"] == pytest.approx((f(r, z+h, "dpsi_dz")-f(r, z-h, "dpsi_dz"))/(2*h), rel=1e-6, abs=1e-9)
        assert v["d2psi_drdz"] == pytest.approx((f(r, z+h, "dpsi_dr")-f(r, z-h, "dpsi_dr"))/(2*h), rel=1e-6, abs=1e-9)
        assert v["dpsi_dr"] == pytest.approx((f(r+h, z, "psi")-f(r-h, z, "psi"))/(2*h), rel=1e-6, abs=1e-9)


def test_extended_bases_satisfy_grad_shafranov_with_the_model_source():
    for topology in ("smooth", "double_null", "lower_single_null"):
        model = _solve(topology)
        r, z = np.meshgrid(np.linspace(0.25, 0.6, 7), np.linspace(-0.3, 0.3, 7), indexing="ij")
        v = evaluate_solovev(model, r, z)
        delta_star = v["d2psi_dr2"] - v["dpsi_dr"]/r + v["d2psi_dz2"]
        np.testing.assert_allclose(delta_star, v["grad_shafranov_source"], rtol=1e-9, atol=1e-9*np.max(np.abs(v["grad_shafranov_source"])))


def test_basis_size_is_checked_against_the_coefficients():
    with pytest.raises(ValueError, match="requires 7"):
        SolovevEquilibrium(np.zeros(5), -1.0, 0.0, 1.0, basis="cerfon_freidberg_even")
    with pytest.raises(ValueError, match="basis must be one of"):
        SolovevEquilibrium(np.zeros(5), -1.0, 0.0, 1.0, basis="spline")
    with pytest.raises(ValueError, match="at least twelve"):
        solve_solovev_constraints(solovev_xpoint_constraints(0.3, -0.4), pprime=-1.0, ffprime=0.0, rref=1.0, basis="cerfon_freidberg")


def test_rank_deficient_and_ill_conditioned_sets_fail_explicitly():
    # A vertical slope on the midplane is identically zero in a symmetric basis.
    midplane = [SolovevConstraint(r, 0.0, "dpsi_dz", 0.0) for r in (0.3, 0.4, 0.5, 0.6, 0.7)]
    with pytest.raises(ValueError, match="rank deficient"):
        solve_solovev_constraints(midplane, pprime=-1.0, ffprime=0.0, rref=0.4)
    nearly_same = [SolovevConstraint(0.4 + 1e-9*i, 0.1, "psi", 0.0) for i in range(4)] + [SolovevConstraint(0.4, 0.0, "psi", -1.0)]
    with pytest.raises(ValueError, match="rank deficient|ill-conditioned"):
        solve_solovev_constraints(nearly_same, pprime=-1.0, ffprime=0.0, rref=0.4)


def test_overdetermined_solve_reports_its_residual():
    constraints = solovev_shape_constraints(major_radius=R0, minor_radius=A_MINOR, elongation=KAPPA, triangularity=DELTA)
    classic = solve_solovev_constraints(constraints, pprime=-1e3, ffprime=-0.01, rref=R0)
    exact = solve_solovev_constraints(constraints, pprime=-1e3, ffprime=-0.01, rref=R0, basis="cerfon_freidberg_even")
    assert classic.residual_norm > 0 and classic.metadata["constraint_count"] == 7
    assert exact.residual_norm < 1e-12 and exact.rank == 7
    assert np.isfinite(exact.metadata["condition_number"])


def test_combination_constraints_validate_their_kinds():
    with pytest.raises(ValueError, match="at least one"):
        SolovevConstraint(0.4, 0.0, "combination", 0.0)
    with pytest.raises(ValueError, match="combination kinds"):
        SolovevConstraint(0.4, 0.0, "combination", 0.0, (("curl", 1.0),))
    with pytest.raises(ValueError, match="only meaningful"):
        SolovevConstraint(0.4, 0.0, "psi", 0.0, (("psi", 1.0),))


def test_xpoint_helper_expands_into_the_three_generic_conditions():
    constraints = solovev_xpoint_constraints(0.3, -0.4, psi=0.02)
    assert [(c.kind, c.value) for c in constraints] == [("psi", 0.02), ("dpsi_dr", 0.0), ("dpsi_dz", 0.0)]
    with pytest.raises(ValueError):
        solovev_xpoint_constraints(0.0, 0.1)


def test_double_null_nulls_are_saddles_and_the_exported_boundary_reaches_them():
    model = _solve("double_null")
    for z_x in (DEFAULT_X[1], -DEFAULT_X[1]):
        point = _saddle(model, DEFAULT_X[0], z_x)
        assert abs(point["psi"]) < 1e-10*PSI0
        assert point["grad"] < 1e-8*PSI0/R0
        assert point["det"] < 0
    r, z = _grid()
    eq = solovev_to_equilibrium(model, r, z)
    assert eq.lcfs.closed and eq.metadata["basis"] == "cerfon_freidberg_even"
    distance = np.hypot(eq.lcfs.r[:, None] - DEFAULT_X[0], eq.lcfs.z[:, None] - np.array([DEFAULT_X[1], -DEFAULT_X[1]]))
    assert np.all(distance.min(axis=0) < 3*(r[1] - r[0]))


def test_single_null_is_asymmetric_with_no_mirrored_xpoint():
    model = _solve("lower_single_null")
    lower = _saddle(model, DEFAULT_X[0], -DEFAULT_X[1])
    assert abs(lower["psi"]) < 1e-10*PSI0 and lower["grad"] < 1e-8*PSI0/R0 and lower["det"] < 0
    mirror = _saddle(model, DEFAULT_X[0], DEFAULT_X[1])
    assert mirror["grad"] > 1e-3*PSI0/R0
    r, z = _grid()
    eq = solovev_to_equilibrium(model, r, z)
    assert eq.metadata["topology_assumptions"].startswith("axisymmetric; up-down asymmetry")
    saddles = find_stationary_points(eq, kind="x")
    near_plasma = [s for s in saddles if abs(s.r - R0) < 1.3*A_MINOR and abs(s.psi_n - 1.0) < 0.2]
    assert any(np.hypot(s.r - DEFAULT_X[0], s.z + DEFAULT_X[1]) < 2*(r[1] - r[0]) for s in near_plasma)
    assert not any(s.z > 0 for s in near_plasma)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    psi = evaluate_solovev(model, rm, zm)["psi"]
    assert np.max(np.abs(psi - psi[:, ::-1])) > 1e-2*np.max(np.abs(psi))


def test_upper_single_null_is_the_lower_one_mirrored():
    lower, upper = _solve("lower_single_null"), _solve("upper_single_null")
    r, z = np.meshgrid(np.linspace(0.25, 0.6, 9), np.linspace(-0.4, 0.4, 9), indexing="ij")
    np.testing.assert_allclose(evaluate_solovev(upper, r, z)["psi"], evaluate_solovev(lower, r, -z)["psi"], rtol=1e-9, atol=1e-12)


def test_custom_xpoint_is_honoured_and_wrong_side_is_rejected():
    custom = (0.33, 0.45)
    model = _solve("double_null", x_point=custom)
    point = _saddle(model, *custom)
    assert abs(point["psi"]) < 1e-10*PSI0 and point["grad"] < 1e-8*PSI0/R0 and point["det"] < 0
    with pytest.raises(ValueError, match="below the midplane"):
        solovev_shape_constraints(major_radius=R0, minor_radius=A_MINOR, elongation=KAPPA,
                                  triangularity=DELTA, topology="lower_single_null", x_point=custom)
    with pytest.raises(ValueError, match="no X-point"):
        solovev_shape_constraints(major_radius=R0, minor_radius=A_MINOR, elongation=KAPPA,
                                  triangularity=DELTA, x_point=custom)
    with pytest.raises(ValueError, match="topology must be one of"):
        solovev_shape_constraints(major_radius=R0, minor_radius=A_MINOR, elongation=KAPPA,
                                  triangularity=DELTA, topology="limited")


def test_gridded_xpoint_recovery_improves_with_resolution():
    model = _solve("lower_single_null")
    errors = []
    for n in (65, 129, 257):
        eq = solovev_to_equilibrium(model, *_grid(n))
        saddles = find_stationary_points(eq, kind="x")
        errors.append(min(np.hypot(s.r - DEFAULT_X[0], s.z + DEFAULT_X[1]) for s in saddles))
    assert errors[-1] < 1e-6*A_MINOR
    # Each doubling of the grid cuts the error by well over the factor 4 of a
    # second-order scheme; the bicubic spline converges faster than that.
    assert errors[1] < errors[0]/4 and errors[2] < errors[1]/4


#: solovev_example's coefficients at its defaults, from the private solve it
#: used before #938 routed it through solve_solovev_constraints.
PRE_938_COEFFICIENTS = {
    "limited": [0.099795655024286, -0.197076838872786, 0.013834966156041, -0.035043003112827,
                0.001643722252377, 0.001431757655644, 5.1316827437e-05] + [0.0]*5,
    "double_null": [0.09869133117978, -0.090898281576533, -0.055147006300509, -0.055644729156053,
                    0.071588968111366, -0.073201245627753, -0.003217419072674] + [0.0]*5,
    "single_null": [0.099259643393476, -0.143769405225254, -0.02088575481824, -0.045694595091302,
                    0.036736363204704, -0.035822422917134, -0.001577098340442, 0.023533460265841,
                    0.125231708207842, -0.079735650131932, -0.017631392242738, 0.002570425723112],
}


def test_solovev_example_keeps_its_pre_938_coefficients():
    for topology, expected in PRE_938_COEFFICIENTS.items():
        np.testing.assert_allclose(solovev_example(topology).metadata["coefficients"], expected, rtol=0, atol=2e-14)


def test_solovev_example_still_solves_a_large_aspect_ratio():
    """The condition limit guards user constraint sets, not the fixed Cerfon-Freidberg system."""
    eq = solovev_example("limited", aspect_ratio=100.0, elongation=1.7, resolution=65)
    assert eq.lcfs.closed
