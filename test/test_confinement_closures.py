"""Errors in variables, Kadomtsev completion and closures (#548 Sec. 6-8; Lane D)."""

import numpy as np
import pytest

from vaft.formula.equilibrium import (
    dimensionless_scaling_coeffs_from_engineering_scaling_coeffs,
    kadomtsev_constraint_from_engineering_exponents,
)
from vaft.process.confinement import (
    closure_constraint,
    dimensionless_confinement_indices,
    fit_confinement_scaling,
    fit_confinement_scaling_odr,
    fit_constrained_confinement_scaling,
    infer_kadomtsev_size_exponent,
)

IPB98 = {"i_p": 0.93, "b_t": 0.15, "p_net": -0.69, "n_e": 0.41}


def _sample(seed=3, noise=0.05):
    rng = np.random.default_rng(seed)
    shot = np.repeat(np.arange(15), 4)
    ip = np.exp(rng.normal(np.log(1e5), 0.3, 15))[shot] * np.exp(rng.normal(0, 0.05, shot.size))
    bt = np.exp(rng.normal(np.log(0.18), 0.15, 15))[shot]
    p = np.exp(rng.normal(np.log(2e5), 0.4, shot.size))
    tau = (1e-3 * (ip / 1e5) ** 0.9 * (bt / 0.18) ** 0.5 * (p / 2e5) ** -0.6
           * np.exp(rng.normal(0, noise, shot.size)))
    return tau, {"i_p": ip, "b_t": bt, "p_net": p}, shot


def test_ipb98_completes_to_its_published_size_exponent():
    alpha_r, se = infer_kadomtsev_size_exponent(IPB98, np.diag([0.01, 0.04, 0.0, 0.0]))
    assert alpha_r == pytest.approx(1.9725, abs=1e-12)  # IPB98(y,2) has R^1.97 at fixed epsilon
    assert se == pytest.approx(np.sqrt(0.25**2 * 0.01 + 1.25**2 * 0.04))
    k = kadomtsev_constraint_from_engineering_exponents(0.93, 0.15, -0.69, 0.41, alpha_r)
    assert k == pytest.approx(0.0, abs=1e-12)


def test_density_free_completion_treats_alpha_n_as_zero():
    a = {k: v for k, v in IPB98.items() if k != "n_e"}
    alpha_r, _ = infer_kadomtsev_size_exponent(a, np.zeros((3, 3)))
    assert alpha_r == pytest.approx(1.9725 - 2 * 0.41)


@pytest.mark.parametrize("alpha", [{"i_p": 1.0, "b_t": 0.1}, {"i_p": 1, "b_t": 0, "p_net": -0.5, "r": 1}])
def test_completion_rejects_unknown_or_missing_exponents(alpha):
    with pytest.raises(ValueError):
        infer_kadomtsev_size_exponent(alpha, np.eye(len(alpha)))


def test_dimensionless_indices_of_ipb98_and_their_propagated_error():
    cov = np.diag([0.02, 0.05, 0.01, 0.03]) ** 2
    out = dimensionless_confinement_indices(IPB98, cov)
    np.testing.assert_allclose(out["value"][:3], [-2.70, -0.90, -0.01], atol=0.03)
    assert out["value"][3] == pytest.approx(-0.93 / 0.31)
    # Linear propagation agrees with a Monte Carlo through the same map.
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(list(IPB98.values()), cov, 4000)

    def mu_rho(v):
        a_r = (v[0] + 5 * v[1] + 3 * v[2] + 8 * v[3] + 5) / 4
        return dimensionless_scaling_coeffs_from_engineering_scaling_coeffs(
            v[0], v[1], v[2], v[3], 0, a_r, 0, 0)[0]

    assert np.std([mu_rho(v) for v in draws]) == pytest.approx(out["stderr"][0], rel=0.15)


@pytest.mark.parametrize("mu_rho", [-2.0, -2.5, -3.0])
def test_closure_constraint_fixes_the_completed_mu_rho(mu_rho):
    rng = np.random.default_rng(int(-10 * mu_rho))
    for a_i, a_b, a_n in rng.uniform(-1, 2, (20, 3)):
        coef, rhs = closure_constraint(mu_rho, ["i_p", "b_t", "p_net", "n_e"])
        # Solve the constraint for a_P, then check the completed index.
        a_p = (rhs - coef["i_p"] * a_i - coef["b_t"] * a_b - coef["n_e"] * a_n) / coef["p_net"]
        out = dimensionless_confinement_indices(
            {"i_p": a_i, "b_t": a_b, "p_net": a_p, "n_e": a_n}, np.zeros((4, 4)))
        assert out["value"][0] == pytest.approx(mu_rho, abs=1e-9)


def test_constrained_fit_satisfies_its_constraint_and_costs_fit_quality():
    tau, x, shot = _sample()
    free = fit_confinement_scaling(tau, x, shot)
    for mu_rho in (-2.0, -3.0):
        coef, rhs = closure_constraint(mu_rho, list(x))
        fit = fit_constrained_confinement_scaling(tau, x, shot, coef, rhs)
        assert sum(coef[k] * v for k, v in fit.exponents().items()) == pytest.approx(rhs, abs=1e-10)
        assert fit.rmse_log >= free.rmse_log - 1e-12
        # The projected covariance has no variance along the constraint.
        c = np.array([0.0] + [coef[k] for k in x])
        assert float(c @ fit.cov @ c) == pytest.approx(0.0, abs=1e-12)


def test_a_true_constraint_leaves_the_fit_nearly_unchanged():
    tau, x, shot = _sample(noise=0.0)
    free = fit_confinement_scaling(tau, x, shot)
    coef = {"i_p": 1.0, "b_t": 1.0}  # the truth has 0.9 + 0.5 = 1.4
    fit = fit_constrained_confinement_scaling(tau, x, shot, coef, 1.4)
    np.testing.assert_allclose(fit.coef, free.coef, atol=1e-9)


def test_constraint_rejects_unknown_names_and_zero_rows():
    tau, x, shot = _sample()
    with pytest.raises(ValueError):
        fit_constrained_confinement_scaling(tau, x, shot, {"r": 1.0}, 0.0)
    with pytest.raises(ValueError):
        fit_constrained_confinement_scaling(tau, x, shot, {"i_p": 0.0}, 0.0)


def test_odr_reduces_to_least_squares_as_the_predictor_error_vanishes():
    tau, x, _ = _sample()
    out = fit_confinement_scaling_odr(tau * x["p_net"], x, sigma_log_response=0.05,
                                      sigma_log_predictors={"i_p": 0, "b_t": 0, "p_net": 1e-6})
    np.testing.assert_allclose(out["coef"], out["ols_coef"], atol=1e-6)
    with pytest.raises(ValueError, match="no predictor"):
        fit_confinement_scaling_odr(tau, x, sigma_log_response=0.05,
                                    sigma_log_predictors={"i_p": 0, "b_t": 0, "p_net": 0})


def test_odr_matches_the_classical_deming_slope():
    """One noisy predictor: the Deming regression closed form."""
    rng = np.random.default_rng(11)
    xt = rng.normal(0, 1, 200)
    xm, ym = xt + rng.normal(0, 0.3, 200), 0.7 * xt + rng.normal(0, 0.2, 200)
    lam = 0.2**2 / 0.3**2
    sxx, syy, sxy = np.var(xm), np.var(ym), np.cov(xm, ym, bias=True)[0, 1]
    deming = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy**2)) / (2 * sxy)
    out = fit_confinement_scaling_odr(np.exp(ym), {"p_net": np.exp(xm)}, sigma_log_response=0.2,
                                      sigma_log_predictors={"p_net": 0.3})
    assert out["coef"][1] == pytest.approx(deming, rel=1e-9)


def test_odr_undoes_the_attenuation_of_a_noisy_predictor():
    """Least squares flattens the slope of a noisy predictor; ODR with its error does not."""
    rng = np.random.default_rng(7)
    p_true = np.exp(rng.normal(0.0, 0.4, 400))
    w = p_true**0.4 * np.exp(rng.normal(0, 0.02, 400))
    p_meas = p_true * np.exp(rng.normal(0, 0.3, 400))
    out = fit_confinement_scaling_odr(w, {"p_net": p_meas}, sigma_log_response=0.02,
                                      sigma_log_predictors={"p_net": 0.3})
    assert abs(out["ols_coef"][1] - 0.4) > 0.1
    assert out["coef"][1] == pytest.approx(0.4, abs=0.05)


def test_odr_requires_an_error_for_every_predictor():
    tau, x, _ = _sample()
    with pytest.raises(ValueError, match="b_t"):
        fit_confinement_scaling_odr(tau, x, sigma_log_response=0.1,
                                    sigma_log_predictors={"i_p": 0.0, "p_net": 0.1})


def test_restricted_leverage_has_one_fewer_degree_of_freedom():
    tau, x, shot = _sample()
    coef, rhs = closure_constraint(-3.0, list(x))
    fit = fit_constrained_confinement_scaling(tau, x, shot, coef, rhs)
    assert np.sum(fit.leverage) == pytest.approx(len(fit.coef) - 1, abs=1e-9)
    free = fit_confinement_scaling(tau, x, shot)
    assert np.sum(free.leverage) == pytest.approx(len(free.coef), abs=1e-9)
