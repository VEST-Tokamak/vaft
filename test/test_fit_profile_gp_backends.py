"""The scipy and scikit-learn Gaussian-process backends agree (issue #426).

`vaft.formula.utils` imported scikit-learn at module scope for one optional
fitting mode, so every module that needed `gradient` paid the whole import --
`import vaft.process.profile` cost 1604 modules against the 837 of `import
vaft`. The default `'gp'` mode is now implemented against scipy and
scikit-learn is an optional dependency.

Replacing a fitting backend is only defensible if the replacement computes the
same thing, so that is asserted here rather than assumed: first as pure linear
algebra with the hyperparameters fixed, then end to end through `fit_profile`
with both optimising their own.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from vaft.formula.utils import _gp_posterior, fit_profile, gp_fit

sklearn = pytest.importorskip("sklearn", reason="the sklearn backend is optional")


def _noisy_exponential(n: int = 25, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.exp(-3.0 * x) + rng.normal(0.0, 0.02, n)
    return x, y, np.full(n, 0.02)


def test_the_posteriors_are_the_same_linear_algebra_at_fixed_hyperparameters():
    """No optimiser involved, so any difference here is an algebra error."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    x, y, y_std = _noisy_exponential()
    x_eval = np.linspace(0.0, 1.0, 21)
    scale = float(np.std(y))
    y_normalised = (y - float(np.mean(y))) / scale
    noise = (y_std / scale) ** 2

    mean_scipy, std_scipy = _gp_posterior(x, y_normalised, noise, x_eval, 1.0, 0.3)

    kernel = ConstantKernel(1.0, "fixed") * RBF(0.3, "fixed")
    reference = GaussianProcessRegressor(kernel=kernel, alpha=noise, optimizer=None)
    reference.fit(x[:, None], y_normalised)
    mean_sklearn, std_sklearn = reference.predict(x_eval[:, None], return_std=True)

    np.testing.assert_allclose(mean_scipy, mean_sklearn, rtol=0, atol=1e-8)
    np.testing.assert_allclose(std_scipy, std_sklearn, rtol=0, atol=1e-8)


def test_both_backends_agree_end_to_end_through_fit_profile():
    """Each optimises its own marginal likelihood and lands in the same place."""
    x, y, y_std = _noisy_exponential()
    x_eval = np.linspace(0.0, 1.0, 21)

    scipy_mean, scipy_std, _, _ = fit_profile(x, y, y_std, x_eval, fitting_function="gp")
    sklearn_mean, sklearn_std, _, _ = fit_profile(
        x, y, y_std, x_eval, fitting_function="gp_sklearn"
    )

    np.testing.assert_allclose(scipy_mean, sklearn_mean, rtol=0, atol=1e-5)
    np.testing.assert_allclose(scipy_std, sklearn_std, rtol=0, atol=1e-5)


def test_the_noise_must_reach_the_kernel_in_normalized_units():
    """The convention bug the replacement exposed, pinned so it cannot return.

    ``normalize_y=True`` standardises the target but leaves ``alpha`` alone, so
    handing scikit-learn the raw variance understates the measurement noise by
    ``var(y)``. The old call did exactly that, which made the fit follow the
    scatter far more closely than the stated uncertainties justify.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    x, y, y_std = _noisy_exponential()
    x_eval = np.linspace(0.0, 1.0, 21)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.3, (0.05, 5.0))

    def predict(alpha):
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=5,
            random_state=0,
        )
        model.fit(x[:, None], y)
        return model.predict(x_eval[:, None])

    correct = predict((y_std / float(np.std(y))) ** 2)
    understated = predict(y_std**2)
    reference, _ = gp_fit(x, y, y_std, x_eval)

    np.testing.assert_allclose(reference, correct, rtol=0, atol=1e-5)
    assert np.max(np.abs(reference - understated)) > 1e-3, (
        "the raw-variance call should differ noticeably; if it no longer does, "
        "sklearn changed how normalize_y interacts with alpha"
    )


@pytest.mark.parametrize(
    "module", ["vaft.formula.utils", "vaft.formula.stability", "vaft.process.profile"]
)
def test_importing_the_module_does_not_load_scikit_learn(module):
    code = f"import sys, {module}; print('sklearn' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", f"{module} still imports scikit-learn"


def test_the_scipy_backend_warns_that_it_cannot_use_an_sklearn_kernel():
    from sklearn.gaussian_process.kernels import RBF

    x, y, y_std = _noisy_exponential(n=12)
    with pytest.warns(RuntimeWarning, match="scikit-learn kernel"):
        fit_profile(x, y, y_std, x, fitting_function="gp", gp_kernel=RBF(0.3))


# --- the fit is trained once, and the callable re-evaluates it -------------------


def test_the_returned_callable_reuses_the_fit_instead_of_refitting():
    """The callable is evaluated many times; the hyperparameter search runs once.

    `vaft.process.profile.core_profiles` evaluates the returned callable on the
    equilibrium grid and `FittedProfile` keeps it for later re-evaluation, so a
    callable that refits pays the whole marginal-likelihood optimisation on
    every call.  The sklearn branch gets this for free by handing back a fitted
    estimator's `predict`; the scipy branch has to be told.
    """
    from vaft.formula import utils as formula_utils

    calls = {"n": 0}
    original = formula_utils._gp_negative_log_marginal_likelihood

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    x, y, y_std = _noisy_exponential(n=20)
    x_eval = np.linspace(0.0, 1.0, 40)

    formula_utils._gp_negative_log_marginal_likelihood = counting
    try:
        y_fit, _, fit_function, _ = fit_profile(
            x, y, y_std, x_eval, fitting_function="gp"
        )
        after_fit = calls["n"]
        assert after_fit > 0, "the fit should have optimised something"
        for _ in range(3):
            fit_function(x_eval)
        assert calls["n"] == after_fit, (
            f"the callable re-ran the hyperparameter search: {calls['n'] - after_fit} "
            "extra objective evaluations over three calls"
        )
    finally:
        formula_utils._gp_negative_log_marginal_likelihood = original

    # Re-evaluating the same fit on the same grid must reproduce it exactly.
    np.testing.assert_allclose(fit_function(x_eval), y_fit)


def test_a_non_finite_target_scale_falls_back_the_same_way_in_both_branches():
    """NaN is truthy, so ``float(np.std(...)) or 1.0`` would accept it (#714)."""
    from vaft.formula.utils import _target_scale

    assert _target_scale([np.nan, 1.0]) == 1.0
    assert _target_scale([2.0, 2.0]) == 1.0
    assert _target_scale([0.0, 2.0]) == 1.0


def test_a_non_finite_anchor_is_refused_by_name_not_by_scipy():
    """`fit_profile` masks its data, then appends the anchor after the mask.

    A non-finite anchor therefore reaches the covariance solve untouched, where
    scipy raises "array must not contain infs or NaNs" from three frames down,
    naming neither the argument nor the caller (#714).
    """
    x, y, y_std = _noisy_exponential(n=10)
    with pytest.raises(ValueError, match="gp_anchor"):
        fit_profile(
            x, y, y_std, x,
            fitting_function="gp",
            gp_anchor=(np.array([1.2]), np.array([np.nan]), np.array([0.01])),
        )
