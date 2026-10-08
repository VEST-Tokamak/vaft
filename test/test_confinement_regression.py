"""Regression and identifiability for confinement scaling (#548 Sec. 3, 4, 6; Lane D)."""

import numpy as np
import pytest

from vaft.process.confinement import (
    assess_predictor_identifiability,
    bootstrap_confinement_scaling,
    fit_confinement_scaling,
    leave_one_group_out_scaling,
)

TRUE = {"i_p": 0.9, "b_t": 0.5, "p": -0.6}


def _sample(seed=3, groups=15, per=4, shot_sd=0.1, noise=0.05):
    rng = np.random.default_rng(seed)
    shot = np.repeat(np.arange(groups), per)
    ip = np.exp(rng.normal(np.log(1e5), 0.3, groups))[shot] * np.exp(rng.normal(0, 0.05, shot.size))
    bt = np.exp(rng.normal(np.log(0.18), 0.15, groups))[shot]
    p = np.exp(rng.normal(np.log(2e5), 0.4, shot.size))
    tau = (1e-3 * (ip / 1e5) ** 0.9 * (bt / 0.18) ** 0.5 * (p / 2e5) ** -0.6
           * np.exp(rng.normal(0, shot_sd, groups))[shot] * np.exp(rng.normal(0, noise, shot.size)))
    return tau, {"i_p": ip, "b_t": bt, "p": p}, shot


def test_noise_free_power_law_is_recovered_exactly():
    tau, x, shot = _sample(shot_sd=0.0, noise=0.0)
    fit = fit_confinement_scaling(tau, x, shot)
    for name, value in TRUE.items():
        assert fit.exponents()[name] == pytest.approx(value, abs=1e-10)
    assert fit.rmse_log == pytest.approx(0.0, abs=1e-10) and fit.n_groups == 15


def test_cluster_errors_exceed_naive_errors_for_a_shot_level_predictor():
    """B_T is constant within a shot, so a shot effect is invisible to iid errors."""
    tau, x, shot = _sample(shot_sd=0.2, noise=0.02)
    fit = fit_confinement_scaling(tau, x, shot)
    j = fit.names.index("b_t")
    assert fit.stderr[j] > fit.stderr_iid[j]


def test_intervals_use_student_t_with_groups_minus_one():
    from scipy import stats

    tau, x, shot = _sample()
    fit = fit_confinement_scaling(tau, x, shot)
    half = stats.t.ppf(0.975, fit.n_groups - 1) * fit.stderr
    np.testing.assert_allclose(fit.ci95[:, 1] - fit.coef, half)


def test_unusable_rows_are_dropped_and_reported():
    tau, x, shot = _sample()
    tau = tau.copy()
    tau[[0, 5]] = [np.nan, -1.0]
    fit = fit_confinement_scaling(tau, x, shot)
    assert fit.n == tau.size - 2 and not fit.mask[0] and not fit.mask[5]


def test_robust_fit_resists_one_wild_row():
    tau, x, shot = _sample(shot_sd=0.0, noise=0.02)
    tau = tau.copy()
    tau[7] *= 20.0
    plain = fit_confinement_scaling(tau, x, shot)
    robust = fit_confinement_scaling(tau, x, shot, robust=True)
    err = lambda f: abs(f.exponents()["p"] - TRUE["p"])  # noqa: E731
    assert err(robust) < err(plain)
    assert int(np.argmax(plain.cooks_distance)) == 7


@pytest.mark.parametrize("groups", [np.zeros(60), np.arange(3)])
def test_fit_rejects_one_group_or_mismatched_groups(groups):
    tau, x, _ = _sample()
    with pytest.raises(ValueError):
        fit_confinement_scaling(tau, x, groups)


def test_identifiability_flags_collinear_predictors():
    rng = np.random.default_rng(0)
    a = np.exp(rng.normal(0, 0.3, 50))
    b = a**1.0 * np.exp(rng.normal(0, 0.01, 50))
    c = np.exp(rng.normal(0, 0.3, 50))
    out = assess_predictor_identifiability({"a": a, "b": b, "c": c})
    assert out["vif"]["a"] > 100 and out["vif"]["c"] < 2
    assert out["correlation"]["a"]["b"] > 0.99 and out["rank"] == 4 and out["n"] == 50
    exact = assess_predictor_identifiability({"a": a, "b": a, "c": c})
    assert exact["rank"] == 3 and exact["vif"]["a"] == float("inf")


def test_group_bootstrap_brackets_the_truth_and_counts_degenerate_draws():
    tau, x, shot = _sample()
    boot = bootstrap_confinement_scaling(tau, x, shot, n_boot=400, seed=1)
    lo, hi = boot["percentile95"][boot["names"].index("p")]
    assert lo < TRUE["p"] < hi
    # Two shots, one field each: half the draws repeat one shot and lose B_T.
    two = shot < 2
    small = bootstrap_confinement_scaling(tau[two], {k: v[two] for k, v in x.items()}, shot[two],
                                          n_boot=200, seed=0)
    assert small["rejected"] > 50


def test_leave_one_group_out_holds_out_whole_groups():
    tau, x, shot = _sample(shot_sd=0.0, noise=0.0)
    out = leave_one_group_out_scaling(tau, x, shot)
    assert out["coef"].shape == (15, 4)
    np.testing.assert_allclose(out["log_error"], 0.0, atol=1e-9)
    noisy = leave_one_group_out_scaling(*_sample())
    assert noisy["rmse_log"] > fit_confinement_scaling(*_sample()).rmse_log


def test_a_constant_predictor_is_refused_not_given_an_exponent():
    """Pseudo-inverse solvers return a confident, arbitrary exponent here."""
    tau, x, shot = _sample()
    x = dict(x, b_t=np.full_like(x["b_t"], 0.18))
    with pytest.raises(ValueError, match="b_t"):
        fit_confinement_scaling(tau, x, shot)
    x["b_t"] = 0.18 * (1.0 + 1e-12 * np.arange(x["b_t"].size))
    with pytest.raises(ValueError, match="b_t"):
        fit_confinement_scaling(tau, x, shot)
    out = assess_predictor_identifiability(x)
    assert out["constant"] == ["b_t"] and out["vif"]["b_t"] == float("inf")
    assert np.isnan(out["correlation"]["b_t"]["i_p"])


def test_an_exactly_collinear_design_is_refused():
    tau, x, shot = _sample()
    x = dict(x, p2=x["p"] ** 2)
    with pytest.raises(ValueError, match="rank"):
        fit_confinement_scaling(tau, x, shot)


def test_robust_iid_error_is_the_robust_estimators_own():
    tau, x, shot = _sample(shot_sd=0.0, noise=0.02)
    tau = tau.copy()
    tau[7] *= 20.0
    plain = fit_confinement_scaling(tau, x, shot)
    robust = fit_confinement_scaling(tau, x, shot, robust=True)
    assert robust.stderr_iid[1] < plain.stderr_iid[1]


# --- #1621: principal directions and the engineering/dimensionless equivalence ----------


def test_principal_directions_find_the_locked_combination():
    from vaft.process.confinement import predictor_principal_directions

    rng = np.random.default_rng(1621)
    shots = np.repeat(np.arange(10), 6)
    i_p = np.exp(rng.uniform(-1, 1, shots.size))
    b_t = np.exp(rng.uniform(-1, 1, shots.size))
    p = i_p * np.exp(rng.normal(0, 0.02, shots.size))  # P follows I_p almost exactly
    out = predictor_principal_directions({"i_p": i_p, "b_t": b_t, "p": p}, groups=shots)
    sv = out["singular_values"]
    assert sv == sorted(sv, reverse=True) and sv[-1] / sv[0] < 0.05
    weakest = np.asarray(out["directions"][-1])
    # The barely varied combination is ln P - ln I_p, with B_T out of it.
    assert abs(weakest[1]) < 0.05 and weakest[0] == pytest.approx(-weakest[2], abs=0.05)
    assert out["effective_rank"] == 2 and {"between", "within"} <= set(out)
    # The split is a decomposition of each direction's spread.
    np.testing.assert_allclose(np.square(out["between"]["projected"]) + np.square(out["within"]["projected"]),
                               np.square(sv), rtol=1e-10)
    # Fewer rows than predictors: the unconstrained directions come back with zero singular value.
    few = predictor_principal_directions({k: v[:3] for k, v in {"a": i_p, "b": b_t, "c": p, "d": i_p * b_t}.items()})
    assert len(few["singular_values"]) == 4 and few["singular_values"][-1] == 0.0
    with pytest.raises(ValueError, match="groups"):
        predictor_principal_directions({"i_p": i_p, "b_t": b_t}, groups=shots[:5])
    with pytest.raises(ValueError, match="does not vary"):
        predictor_principal_directions({"i_p": i_p, "b_t": np.ones_like(i_p)})


def _dimensionless_synthetic(n_rows=150, seed=0):
    rng = np.random.default_rng(seed)
    n, t, b, r, q = (np.exp(rng.uniform(-1, 1, n_rows)) for _ in range(5))
    rho, beta, nu = t**0.5 / (b * r), n * t / b**2, n * r / t**2
    omega_tau = rho**-2.7 * beta**-0.9 * nu**-0.01 * q**-3.0
    tau = omega_tau / b
    engineering = {"i_p": b * r / q, "b_t": b, "p_net": n * t * r**3 / tau, "n_e": n, "r": r}
    return tau, engineering, omega_tau, {"rho": rho, "beta": beta, "nu": nu, "q": q}


def test_engineering_and_direct_dimensionless_routes_agree_on_exact_data():
    from vaft.process.confinement import dimensionless_confinement_indices, fit_confinement_scaling

    tau, engineering, omega_tau, groups = _dimensionless_synthetic()
    rows = np.arange(tau.size)
    route_a = fit_confinement_scaling(tau, engineering, rows)
    keys = ["i_p", "b_t", "p_net", "n_e"]
    idx = [route_a.names.index(k) for k in keys]
    mu_a = dimensionless_confinement_indices({k: route_a.exponents()[k] for k in keys},
                                             np.asarray(route_a.cov)[np.ix_(idx, idx)])
    route_b = fit_confinement_scaling(omega_tau, groups, rows).exponents()
    np.testing.assert_allclose(mu_a["value"], [route_b[k] for k in ("rho", "beta", "nu", "q")], atol=1e-9)
    np.testing.assert_allclose(mu_a["value"], [-2.7, -0.9, -0.01, -3.0], atol=1e-9)
    # The completed size exponent equals the one the exact data carry.
    assert mu_a["alpha_R"] == pytest.approx(route_a.exponents()["r"], abs=1e-9)



# --- #1713: thermal vs global energy basis -------------------------------------------------


def test_resolver_is_strict_and_records_the_thermal_for_global_approximation():
    from vaft.process.confinement import THERMAL_AS_GLOBAL, resolve_observed_confinement

    th = np.array([1e-3, 2e-3, np.nan])
    thermal = resolve_observed_confinement(th, "thermal")
    np.testing.assert_array_equal(thermal.tau[:2], th[:2])
    assert np.isnan(thermal.tau[2])
    assert thermal.approximation is None and list(thermal.energy_basis_used) == ["thermal", "thermal", ""]
    # A global scaling with only a thermal time is not silently accepted...
    strict = resolve_observed_confinement(th, "global")
    assert np.all(np.isnan(strict.tau)) and "no global" in strict.reason
    # ...unless the approximation is asked for, and then it is recorded.
    relaxed = resolve_observed_confinement(th, "global", thermal_as_global=True)
    np.testing.assert_array_equal(relaxed.tau[:2], th[:2])
    assert relaxed.approximation == THERMAL_AS_GLOBAL
    # A global observation wins where it exists.
    mixed = resolve_observed_confinement(th, "global", tau_global=np.array([1.5e-3, np.nan, np.nan]),
                                         thermal_as_global=True)
    assert mixed.tau[0] == 1.5e-3 and list(mixed.energy_basis_used[:2]) == ["global", "thermal"]
    # An unaudited basis is unknown: NaN unless the approximation is asked for.
    assert np.all(np.isnan(resolve_observed_confinement(th, "unaudited").tau))
    assert "unaudited" in resolve_observed_confinement(th, "unaudited", thermal_as_global=True).approximation
    with pytest.raises(ValueError):
        resolve_observed_confinement(th, "total")


def test_fast_ions_separate_the_thermal_and_global_h_factors():
    from vaft.process.confinement import resolve_observed_confinement

    p = np.array([1e6, 2e6])
    w_th, w_fast = np.array([1e5, 1.6e5]), np.array([3e4, 6e4])
    tau_th, tau_global = w_th / p, (w_th + w_fast) / p
    prediction = np.array([0.08, 0.07])
    h_thermal = resolve_observed_confinement(tau_th, "thermal", tau_global=tau_global).tau / prediction
    h_global = resolve_observed_confinement(tau_th, "global", tau_global=tau_global).tau / prediction
    np.testing.assert_allclose(h_global / h_thermal, (w_th + w_fast) / w_th)
    assert np.all(h_global > h_thermal)
