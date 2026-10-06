"""vaft.formula.dimensional (#1621): Buckingham-Pi groups and the Connor-Kadomtsev constraint, exactly."""

from fractions import Fraction

import numpy as np
import pytest

from vaft.formula import dimensional as d
from vaft.formula.equilibrium import (
    dimensionless_scaling_coeffs_from_engineering_scaling_coeffs,
    kadomtsev_constraint_from_engineering_exponents,
)

#: The collisional, finite-beta confinement problem: varied state, time, constants.
VARIABLES = ("n", "T", "B", "R", "tau", "e", "m_i", "mu0", "eps0")
#: Conventional groups as exponent vectors over VARIABLES (fixed q, epsilon, kappa).
OMEGA_TAU = (0, 0, 1, 0, 1, 1, -1, 0, 0)                                   # e B tau / m_i
RHO_STAR = (0, Fraction(1, 2), -1, -1, 0, -1, Fraction(1, 2), 0, 0)        # sqrt(m_i T) / (e B R)
BETA = (1, 1, -2, 0, 0, 0, 0, 1, 0)                                        # mu0 n T / B^2
NU_STAR = (1, -2, 0, 1, 0, 4, 0, 0, -2)                                    # n e^4 R / (eps0^2 T^2)
DEBYE = (1, -1, 0, 2, 0, 2, 0, 0, -1)                                      # (R / lambda_D)^2 = n e^2 R^2 / (eps0 T)
STATE = (0, 1, 2, 3)                                                       # n, T, B, R


def test_dimension_matrix_and_monomials():
    D = d.dimension_matrix(["B", "e", "tau", "m_i"])
    assert D.shape == (4, 4)
    for group in (OMEGA_TAU, RHO_STAR, BETA, NU_STAR, DEBYE):
        assert all(v == 0 for v in d.monomial_dimension(group, VARIABLES))
    # tau itself is a time: (L, M, T, I) = (0, 0, 1, 0).
    assert list(d.monomial_dimension((0, 0, 0, 0, 1, 0, 0, 0, 0), VARIABLES)) == [0, 0, 1, 0]
    with pytest.raises(KeyError):
        d.dimension_matrix(["n", "not_a_variable"])


def test_buckingham_pi_gives_five_groups_and_the_conventional_ones_are_a_basis_change():
    basis = d.dimensionless_groups(VARIABLES)
    assert len(basis) == len(VARIABLES) - 4
    for vector in basis:
        assert all(v == 0 for v in d.monomial_dimension(vector, VARIABLES))
        assert all(Fraction(v).denominator == 1 for v in vector)  # integer, exact
    conventional = [OMEGA_TAU, RHO_STAR, BETA, NU_STAR, DEBYE]
    coefficients = np.array([[float(c) for c in d.express_in_basis(g, basis)] for g in conventional])
    # The five conventional groups span the same space: an invertible change of basis.
    assert abs(np.linalg.det(coefficients)) > 1e-9
    with pytest.raises(ValueError, match="span"):
        d.express_in_basis((0, 0, 0, 0, 1, 0, 0, 0, 0), basis)  # tau alone is not dimensionless


def test_large_denominators_stay_exact_and_floats_read_as_decimals():
    big = d.rational_null_space([[Fraction(1, 999983), Fraction(1, 999979), Fraction(1, 999961), Fraction(1, 999959)]])
    for v in big:
        assert sum(Fraction(1, q) * x for q, x in zip((999983, 999979, 999961, 999959), v)) == 0
    assert list(d.monomial_dimension([0.123456789], ["tau"])) == [0, 0, Fraction("0.123456789"), 0]


def test_rational_null_space_is_exact():
    basis = d.rational_null_space([[1, 2, 3], [2, 4, 6]])
    assert len(basis) == 2
    for v in basis:
        assert 1 * v[0] + 2 * v[1] + 3 * v[2] == 0
    assert d.rational_null_space([[1, 0], [0, 1]]) == []
    with pytest.raises(ValueError):
        d.rational_null_space([[1, 2], [3]])


def _state(group):
    return [group[i] for i in STATE]


def test_kadomtsev_constraint_follows_from_dropping_the_debye_group():
    # All four non-time groups span the state space: dimensional analysis alone constrains nothing.
    assert d.similarity_constraint_vectors([_state(g) for g in (RHO_STAR, BETA, NU_STAR, DEBYE)]) == []
    # Quasi-neutrality (no Debye group) leaves one constraint: k = (8, 2, 5, -4) on (n, T, B, R).
    (k,) = d.similarity_constraint_vectors([_state(g) for g in (RHO_STAR, BETA, NU_STAR)])
    assert [int(v) for v in k] == [8, 2, 5, -4]


@pytest.mark.parametrize("seed", range(20))
def test_constraint_on_omega_tau_is_the_engineering_kadomtsev_residual(seed):
    rng = np.random.default_rng(seed)
    a_I, a_B, a_P, a_n, a_R = rng.uniform(-1.5, 2.0, 5)
    if abs(1 + a_P) < 0.05:
        a_P += 0.2
    (k,) = d.similarity_constraint_vectors([_state(g) for g in (RHO_STAR, BETA, NU_STAR)])
    x = [float(v) for v in d.state_exponents_from_engineering_exponents(a_I, a_B, a_P, a_n, a_R)]
    x[2] += 1.0  # Omega_i tau_E = e B tau / m_i
    residual_from_similarity = float(np.dot([float(v) for v in k], x)) * (1 + a_P)
    residual = kadomtsev_constraint_from_engineering_exponents(a_I, a_B, a_P, a_n, a_R)
    assert residual_from_similarity == pytest.approx(-residual, abs=1e-12)


def test_ipb98_completed_scaling_is_in_the_span_and_matches_the_published_indices():
    # IPB98(y,2) at fixed epsilon, kappa: aI 0.93, aB 0.15, aP -0.69, an 0.41, aR 1.97.
    x = list(d.state_exponents_from_engineering_exponents("0.93", "0.15", "-0.69", "0.41", "1.97"))
    x[2] += 1
    groups = [_state(g) for g in (RHO_STAR, BETA, NU_STAR)]
    (k,) = d.similarity_constraint_vectors(groups)
    off = float(sum(kv * xv for kv, xv in zip(k, x)))
    assert abs(off) < 0.2  # IPB98 nearly satisfies the constraint: k . x = +0.03 here
    mu_rho, mu_beta, mu_nu = dimensionless_scaling_coeffs_from_engineering_scaling_coeffs(
        0.93, 0.15, -0.69, 0.41, 0.19, 1.97, 0.58, 0.78)[:3]  # a_M, a_R, a_eps, a_kappa
    assert mu_rho == pytest.approx(-2.70, abs=0.05) and mu_beta == pytest.approx(-0.90, abs=0.05)


def test_state_exponents_are_exact_for_rationals_and_float_otherwise():
    exact = d.state_exponents_from_engineering_exponents(1, 0, Fraction(-1, 2), 0, 0)
    assert list(exact) == [-1, -1, 2, -1] and all(isinstance(v, Fraction) for v in exact)
    approx = d.state_exponents_from_engineering_exponents(1.0, 0.0, -0.5)
    assert approx.dtype == float and list(approx) == [-1.0, -1.0, 2.0, -1.0]


def test_alpha_p_minus_one_is_singular():
    with pytest.raises(ValueError, match="singular"):
        d.state_exponents_from_engineering_exponents(1.0, 0.0, -1.0)
