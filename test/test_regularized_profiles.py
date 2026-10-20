"""Independent analytic truth for positive regular flux functions."""
import numpy as np
import pytest

from vaft.formula.regularized_profiles import (
    RegularizedProfileConfig as Config,
    evaluate_regularized_profile,
    fit_regularized_profile,
    profile_basis,
)
from vaft.formula.utils import fit_profile


@pytest.mark.parametrize("basis", ["bernstein2", "bernstein3", "bspline6"])
@pytest.mark.parametrize("hollow", [False, True])
def test_linear_flux_truth_regular_axis_without_zero_flux_slope(basis, hollow):
    x = np.linspace(0, 1.3, 35)
    y = 2 + (1 if hollow else -1)*x
    cfg = Config(basis=basis, domain_max=1.3, regularization=0)
    fit = fit_regularized_profile(x, y, config=cfg)
    assert fit.status == "available", fit.reason
    np.testing.assert_allclose(fit.evaluate(x), y, atol=2e-5)
    slope = fit.evaluate([0.], derivative=1)[0]
    assert abs(slope) > .9
    rho = np.array([0., 1e-6])
    assert abs(np.diff(fit.evaluate(rho*rho))[0]/1e-6) < 2e-6
    assert fit.evaluate([1.])[0] > 0
    assert np.isnan(fit.evaluate([1.31])[0])
    assert fit.diagnostics["sigma"] is None


def test_robust_outlier_preserves_weights_and_large_physical_scale():
    x = np.linspace(0, 1.2, 40)
    y = (3-x)*1e19
    y[20] += 1e21
    fit = fit_regularized_profile(x, y, config=Config(domain_max=1.2, scale=1e19, huber_delta=1))
    assert fit.status == "available", fit.reason
    assert fit.diagnostics["robust_weights"][20] < .02
    assert np.mean(abs(fit.evaluate(x)/1e19-(3-x))) < .2


def test_rank_convergence_and_invalid_inputs():
    assert fit_regularized_profile([.5]*8, [1.]*8).status == "failed"
    f = fit_regularized_profile(np.linspace(0, 1, 20), np.linspace(1, 5, 20)**2,
                                config=Config(max_nfev=1))
    assert f.status == "failed"
    with pytest.raises(ValueError):
        f.evaluate([.5])
    for x, y in [([0, np.nan], [1, 2]), ([-1, 0], [1, 2]), ([0, 1], [1e300]*2)]:
        with pytest.raises(ValueError):
            fit_regularized_profile(x, y)


def test_partition_derivatives_and_parameter_roundtrip():
    for basis in ("bernstein2", "bernstein3", "bspline6"):
        cfg = Config(basis=basis, domain_max=1.4)
        x = np.linspace(0, 1.4, 100)
        b = profile_basis(x, cfg)
        np.testing.assert_allclose(b.sum(axis=-1), 1, atol=1e-14)
        assert b.min() > -1e-14
        theta = np.linspace(-2, 2, cfg.coefficient_count)
        expected = cfg.scale*b@(cfg.floor+np.logaddexp(0, theta))
        np.testing.assert_allclose(evaluate_regularized_profile(x, theta, cfg), expected)
        h = 1e-5
        finite = (evaluate_regularized_profile(x[1:-1]+h, theta, cfg)-
                  evaluate_regularized_profile(x[1:-1]-h, theta, cfg))/(2*h)
        np.testing.assert_allclose(evaluate_regularized_profile(x[1:-1], theta, cfg, derivative=1), finite, atol=1e-7)


def test_legacy_tuple_unchanged_and_free_polynomial_has_axis_linear_term():
    rho = np.linspace(0, 1, 20)
    result = fit_profile(rho, 2+rho, None, rho, order=3,
                         fitting_function="free_polynomial", uncertainty_option=0)
    assert len(result) == 4
    np.testing.assert_allclose(result[0], 2+rho, atol=1e-7)
    np.testing.assert_array_equal(result[1], 0)
    assert result[3][1] > .99
