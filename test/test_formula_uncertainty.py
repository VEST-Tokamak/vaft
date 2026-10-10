"""Per-formula uncertainty propagation: the optional contract and evaluation API (#1874).

The pilot formulas keep their signatures, values and cost; the analytic
Jacobians they carry agree with a well-scaled finite difference; correlated
and independent inputs reproduce the closed forms; an unknown uncertainty, a
mismatched input and an out-of-domain point are refused; and the first-order
result says when it stops describing the spread.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vaft.formula import equilibrium, sensitivity
from vaft.formula._propagation import analytic_jacobian

ROOT = Path(__file__).resolve().parents[1]
TAU = equilibrium.confinement_time_from_P_loss_W_th
OHMIC = equilibrium.ohmic_heating_power_from_I_p_V_res
POINT = {"P_loss": 1.0e5, "W_th": 2.0e3}


def propagate(formula=TAU, inputs=POINT, **kwargs):
    return sensitivity.propagate_formula_uncertainty(formula, inputs, **kwargs)


# --- the fast path is untouched ---------------------------------------------


def test_the_pilot_formulas_keep_their_signature_value_and_identity():
    assert list(inspect.signature(TAU).parameters) == ["P_loss", "W_th"]
    assert list(inspect.signature(OHMIC).parameters) == ["I_p", "V_res"]
    assert TAU(1.0e5, 2.0e3) == 2.0e-2 and isinstance(TAU(1.0e5, 2.0e3), float)
    assert OHMIC(1.0e5, 0.5) == 5.0e4
    # the decorator returns the function itself: no wrapper on the call path
    assert TAU.__module__ == "vaft.formula.equilibrium" and not hasattr(TAU, "__wrapped__")
    assert analytic_jacobian(TAU)[1] == ("P_loss", "W_th")


def test_the_catalog_reads_the_attribute_the_decorator_writes():
    from vaft.formula import _propagation, catalog

    assert catalog._JACOBIAN_ATTRIBUTE == _propagation._ATTRIBUTE
    assert catalog.analytic_jacobian(TAU) == _propagation.analytic_jacobian(TAU)


def test_importing_the_formula_package_does_not_load_the_propagation_api():
    code = "import sys, vaft.formula; print('vaft.formula.sensitivity' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


# --- analytic derivatives against finite differences -------------------------


@pytest.mark.parametrize(("formula", "point"), [
    (TAU, {"P_loss": 1.0e5, "W_th": 2.0e3}),
    (TAU, {"P_loss": 3.0e3, "W_th": 7.0e1}),
    (OHMIC, {"I_p": 1.0e5, "V_res": 0.4}),
])
def test_the_analytic_jacobian_agrees_with_a_scaled_finite_difference(formula, point):
    analytic = propagate(formula, point, std=[1.0, 1.0], derivative="analytic")
    numeric = propagate(formula, point, std=[1.0, 1.0], derivative="finite_difference")
    assert analytic.derivative_method == "analytic" and numeric.derivative_method == "finite_difference"
    np.testing.assert_allclose(analytic.jacobian, numeric.jacobian, rtol=1e-7)


def test_the_finite_difference_step_follows_each_inputs_own_scale():
    # inputs 8 orders of magnitude apart; a unit-floored step would ruin the small one
    point = {"P_loss": 2.0e-3, "W_th": 4.0e5}
    analytic = propagate(TAU, point, std=[1e-4, 1e4], derivative="analytic")
    numeric = propagate(TAU, point, std=[1e-4, 1e4], derivative="finite_difference")
    np.testing.assert_allclose(numeric.jacobian, analytic.jacobian, rtol=1e-7)


# --- closed forms, independent and correlated --------------------------------


def test_independent_inputs_reproduce_the_relative_error_sum():
    sigma_p, sigma_w = 1.0e4, 1.0e2
    result = propagate(std=[sigma_p, sigma_w])
    tau = POINT["W_th"] / POINT["P_loss"]
    expected = tau * np.hypot(sigma_p / POINT["P_loss"], sigma_w / POINT["W_th"])
    np.testing.assert_allclose(result.std, [expected], rtol=1e-12)
    assert result.method == "linear" and result.input_names == ("P_loss", "W_th")


def test_a_positive_correlation_reduces_the_spread_of_a_ratio():
    sigma_p, sigma_w, rho = 1.0e4, 2.0e2, 0.8
    cov = np.array([[sigma_p ** 2, rho * sigma_p * sigma_w], [rho * sigma_p * sigma_w, sigma_w ** 2]])
    correlated = propagate(covariance=cov)
    independent = propagate(std=[sigma_p, sigma_w])
    P, W = POINT["P_loss"], POINT["W_th"]
    rel2 = (sigma_p / P) ** 2 + (sigma_w / W) ** 2 - 2 * rho * sigma_p * sigma_w / (P * W)
    np.testing.assert_allclose(correlated.std, [W / P * np.sqrt(rel2)], rtol=1e-12)
    assert correlated.std[0] < independent.std[0]


def test_a_bilinear_product_with_correlation():
    I, V, sI, sV, rho = 1.0e5, 0.5, 2.0e3, 0.05, -0.3
    cov = np.array([[sI ** 2, rho * sI * sV], [rho * sI * sV, sV ** 2]])
    result = propagate(OHMIC, {"I_p": I, "V_res": V}, covariance=cov)
    expected = V ** 2 * sI ** 2 + I ** 2 * sV ** 2 + 2 * I * V * rho * sI * sV
    np.testing.assert_allclose(result.covariance, [[expected]], rtol=1e-12)


def test_input_order_is_the_covariance_order():
    sigma_p, sigma_w = 1.0e4, 1.0e2
    forward = propagate(std=[sigma_p, sigma_w], input_names=("P_loss", "W_th"))
    reverse = propagate(std=[sigma_w, sigma_p], input_names=("W_th", "P_loss"))
    np.testing.assert_allclose(forward.std, reverse.std, rtol=1e-12)
    np.testing.assert_allclose(forward.jacobian[:, ::-1], reverse.jacobian, rtol=1e-12)


def test_a_fixed_input_carries_no_uncertainty():
    result = propagate(input_names=("W_th",), std=[1.0e2])
    assert result.fixed == {"P_loss": 1.0e5}
    np.testing.assert_allclose(result.std, [1.0e2 / 1.0e5], rtol=1e-12)
    assert result.jacobian.shape == (1, 1)


def test_variance_and_standard_deviation_are_not_confused():
    by_variance = propagate(covariance=[1.0e8, 1.0e4])  # 1-D covariance means variances
    by_std = propagate(std=[1.0e4, 1.0e2])
    np.testing.assert_allclose(by_variance.covariance, by_std.covariance, rtol=1e-12)
    np.testing.assert_allclose(by_std.std ** 2, np.diag(by_std.covariance), rtol=1e-12)


def test_the_output_covariance_is_symmetric_and_positive_semidefinite():
    cov = np.array([[1.0e8, 5.0e5], [5.0e5, 1.0e4]])
    out = propagate(covariance=cov).covariance
    np.testing.assert_allclose(out, out.T)
    assert np.linalg.eigvalsh(out).min() >= -1e-18


# --- refused inputs -----------------------------------------------------------


@pytest.mark.parametrize(("kwargs", "message"), [
    ({}, "unknown, not zero"),
    ({"std": [1.0, 1.0], "covariance": [1.0, 1.0]}, "exactly one"),
    ({"std": [np.nan, 1.0]}, "known"),
    ({"covariance": [[1.0, np.nan], [np.nan, 1.0]]}, "NaN is an unknown"),
    ({"std": [1.0]}, "entries for 2 inputs"),
    ({"std": [-1.0, 1.0]}, "non-negative"),
    ({"covariance": [[1.0, 2.0], [0.0, 1.0]]}, "symmetric"),
    ({"covariance": [[1.0, 2.0], [2.0, 1.0]]}, "positive semi-definite"),
    ({"std": [1.0, 1.0], "method": "bootstrap"}, "method"),
    ({"std": [1.0, 1.0], "derivative": "autodiff"}, "derivative"),
    ({"std": [1.0, 1.0], "input_names": ("P_loss", "P_loss")}, "repeats"),
    ({"std": [1.0, 1.0], "input_names": ("P_loss", "W")}, "missing"),
])
def test_an_ambiguous_or_invalid_request_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        propagate(**kwargs)


def test_an_unknown_parameter_and_an_array_input_are_refused():
    with pytest.raises(ValueError, match="no parameter"):
        propagate(inputs={**POINT, "R0": 0.4}, std=[1.0, 1.0])
    with pytest.raises(ValueError, match="not a scalar"):
        propagate(inputs={"P_loss": np.array([1.0, 2.0]), "W_th": 1.0}, input_names=("P_loss", "W_th"),
                  std=[1.0, 1.0])


def test_a_point_outside_the_domain_is_refused_not_propagated():
    with pytest.raises(ValueError, match="outside its domain"):
        propagate(inputs={"P_loss": 0.0, "W_th": 1.0e3}, std=[1.0, 1.0], derivative="analytic")


def test_analytic_is_refused_for_a_formula_without_one():
    with pytest.raises(ValueError, match="carries no analytic Jacobian"):
        propagate(equilibrium.beta_toroidal_from_p_B0, {"p_average": 1.0e3, "B0": 0.3}, std=[1.0, 0.01],
                  derivative="analytic")


def test_a_zero_input_needs_an_explicit_step():
    with pytest.raises(ValueError, match="give fd_step"):
        propagate(OHMIC, {"I_p": 1.0e5, "V_res": 0.0}, std=[1.0, 0.01], derivative="finite_difference")
    result = propagate(OHMIC, {"I_p": 1.0e5, "V_res": 0.0}, std=[1.0, 0.01], derivative="finite_difference",
                       fd_step={"I_p": 1.0, "V_res": 1e-6})
    np.testing.assert_allclose(result.jacobian, [[0.0, 1.0e5]], atol=1e-6)


# --- validity of the first-order result ----------------------------------------


def test_linearity_grows_with_the_relative_spread_and_is_infinite_across_the_pole():
    small = propagate(std=[1.0e3, 0.0]).linearity
    large = propagate(std=[5.0e4, 0.0]).linearity
    across = propagate(std=[1.0e5, 0.0]).linearity  # P_loss - 1 sigma reaches the pole
    assert small < 0.02 < large < 1.0
    assert across == float("inf")
    # exact inputs test nothing: no ratio is reported
    assert propagate(std=[0.0, 0.0]).linearity is None


def test_linear_and_sampled_propagation_agree_when_linear_and_part_when_not():
    near = propagate(std=[2.0e3, 2.0e1])
    near_mc = propagate(std=[2.0e3, 2.0e1], method="monte_carlo", samples=40000, seed=1)
    assert near_mc.method == "monte_carlo" and near_mc.jacobian is None and near_mc.derivative_method is None
    np.testing.assert_allclose(near_mc.std, near.std, rtol=0.03)
    far = propagate(std=[4.0e4, 2.0e1])
    far_mc = propagate(std=[4.0e4, 2.0e1], method="monte_carlo", samples=40000, seed=1)
    # 1/P is convex with a pole inside a 40 % spread: sampling is much wider than first order,
    # and the linearity ratio says so before anyone compares
    assert far_mc.std[0] > 1.2 * far.std[0]
    assert far.linearity > 0.3


def test_monte_carlo_reports_rejected_draws():
    result = propagate(std=[6.0e4, 2.0e1], method="monte_carlo", samples=5000, seed=2)
    assert result.rejected == 0  # 1/P is finite for any nonzero draw ...
    assert result.samples + result.rejected == 5000


# --- the catalog contract -------------------------------------------------------


def test_the_catalog_reports_the_contract_and_requires_the_section():
    from vaft.formula import catalog

    for name in ("equilibrium.confinement_time_from_P_loss_W_th", "equilibrium.ohmic_heating_power_from_I_p_V_res"):
        spec = catalog.describe(name)
        assert spec.uncertainty_propagation and spec.analytic_jacobian and spec.errors == ()
        assert "Uncertainty propagation" in dict(spec.sections)
        row = spec.as_dict()
        assert row["uncertainty_propagation"] and row["analytic_jacobian"]

    from vaft.formula._propagation import jacobian

    @jacobian(lambda x: [[1.0]], wrt=("x",))
    def toy(x):
        """Toy.

        Parameters
        ----------
        x : float [m]
            A length.

        Returns
        -------
        float [m]
            The length.
        """
        return x

    spec = catalog._spec(toy, "toy", "utils", "vaft.formula.utils", ())
    assert any("Uncertainty propagation" in error for error in spec.errors)
