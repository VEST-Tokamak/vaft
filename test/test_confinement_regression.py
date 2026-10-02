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
