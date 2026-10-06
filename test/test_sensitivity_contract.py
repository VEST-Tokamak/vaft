"""The #1642 sensitivity contract: generic kernels and their interpretation (Lane AP).

The acceptance demonstrations of #1642 at the cost of a unit test:

* an analytic derivative already in VAFT (the exact Green's-function field,
  ``B = curl(psi)``) validated against a finite-difference one;
* a local ``J Sigma J^T`` (the one `dimensionless_confinement_indices` already
  returns) compared with Monte Carlo through the same map;
* the same map near ``alpha_P = -1``, where its docstring says the linear
  propagation is poor, showing the local Jacobian failing -- and the two
  diagnostics that say so.

No solver runs; the costliest call is 4000 evaluations of a closed-form map.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from vaft.formula.green import green_br_bz_exact, green_psi_exact
from vaft.formula.sensitivity import (
    finite_difference_jacobian,
    linear_covariance_propagation,
    linearity_ratio,
    monte_carlo_propagation,
    singular_value_spectrum,
)
from vaft.process.confinement import dimensionless_confinement_indices
from vaft.validation import ValidationStatus
from vaft.validation.sensitivity import (
    Jacobian,
    as_evidence,
    compare_jacobians,
    compare_linear_to_monte_carlo,
    scan_evidence,
)

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def test_finite_differences_recover_a_known_jacobian():
    def f(x):
        return np.array([x[0] ** 2 * x[1], np.sin(x[1]), 3.0 * x[0]])

    x = np.array([1.3, 0.4])
    exact = np.array([[2 * 1.3 * 0.4, 1.3 ** 2], [0.0, math.cos(0.4)], [3.0, 0.0]])
    assert finite_difference_jacobian(f, x) == pytest.approx(exact, rel=1e-8, abs=1e-9)
    assert finite_difference_jacobian(f, x, scheme="forward") == pytest.approx(exact, rel=1e-6, abs=1e-7)
    with pytest.raises(ValueError):
        finite_difference_jacobian(f, x, scheme="backward")
    with pytest.raises(ValueError):
        finite_difference_jacobian(f, x, step=0.0)


def test_covariance_propagation_keeps_correlations():
    jac = np.array([[1.0, 1.0]])
    independent = linear_covariance_propagation(jac, np.array([1.0, 1.0]))
    anticorrelated = linear_covariance_propagation(jac, np.array([[1.0, -1.0], [-1.0, 1.0]]))
    assert independent[0, 0] == pytest.approx(2.0)
    assert anticorrelated[0, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        linear_covariance_propagation(jac, np.eye(3))
    with pytest.raises(ValueError):
        linear_covariance_propagation(jac, np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_monte_carlo_is_seeded_and_matches_a_linear_map():
    jac = np.array([[2.0, -1.0], [0.5, 0.5]])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    first = monte_carlo_propagation(lambda x: jac @ x, [1.0, 2.0], cov, samples=20000, seed=3)
    again = monte_carlo_propagation(lambda x: jac @ x, [1.0, 2.0], cov, samples=20000, seed=3)
    assert np.array_equal(first["covariance"], again["covariance"])
    np.testing.assert_allclose(first["covariance"], linear_covariance_propagation(jac, cov), rtol=0.05, atol=2e-3)
    assert first["mean"] == pytest.approx(jac @ [1.0, 2.0], abs=0.01)
    with pytest.raises(ValueError):
        monte_carlo_propagation(lambda x: x, [0.0], [1.0], samples=1)


def test_singular_value_spectrum_finds_the_unidentified_combination():
    # The output depends on x0 + x1 only: (1, -1) is not identifiable.
    jac = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
    spectrum = singular_value_spectrum(jac)
    assert spectrum["rank"] == 1 and spectrum["nullity"] == 1
    assert spectrum["condition_number"] == math.inf
    null = spectrum["null_space"][:, 0]
    assert abs(null @ np.array([1.0, 1.0])) < 1e-12
    full = singular_value_spectrum(np.diag([10.0, 0.1]))
    assert full["rank"] == 2 and full["condition_number"] == pytest.approx(100.0)
    # Column scaling makes commensurate parameters of different units.
    assert singular_value_spectrum(np.diag([10.0, 0.1]), column_scale=[0.1, 10.0])["condition_number"] == (
        pytest.approx(1.0))


def test_linearity_ratio_is_zero_for_a_linear_map_and_grows_with_curvature():
    jac = np.array([[2.0, 0.0]])
    assert linearity_ratio(lambda x: jac @ x, [1.0, 1.0], [0.3, 0.2], jac) == pytest.approx(0.0, abs=1e-12)
    # f = x^2 at x = 1: J = 2; a step d gives d^2 / (2d + d^2) of the response missed.
    for d in (0.01, 0.5):
        r = linearity_ratio(lambda x: x ** 2, [1.0], [d], [[2.0]])
        assert r == pytest.approx(d / (2 + d))
    assert math.isnan(linearity_ratio(lambda x: np.zeros(1), [1.0], [0.1], [[0.0]]))


# ---------------------------------------------------------------------------
# a derivative validated against another (#1642 acceptance)
# ---------------------------------------------------------------------------


def _green_jacobians(r_src=0.45, z_src=0.1, point=(0.30, -0.05)):
    """d psi / d(r, z) of a unit ring current: analytic (the field) and finite-difference."""

    def psi(x):
        return np.atleast_1d(green_psi_exact(x[0], x[1], r_src, z_src))

    r, z = point
    br, bz = green_br_bz_exact(r, z, r_src, z_src)
    # Full-weber psi: B_z = (1/2 pi r) dpsi/dr, B_r = -(1/2 pi r) dpsi/dz.
    analytic = np.array([[2 * np.pi * r * float(bz), -2 * np.pi * r * float(br)]])
    numeric = finite_difference_jacobian(psi, np.array([r, z]))
    names = dict(inputs=("r", "z"), outputs=("psi",), kind="forward", perturbation="physical")
    return (Jacobian("analytic", matrix=analytic, source="vaft.formula.green.green_br_bz_exact", **names),
            Jacobian("finite_difference", matrix=numeric, source="vaft.formula.green.green_psi_exact", **names))


def test_the_analytic_green_field_is_the_derivative_of_its_flux():
    analytic, numeric = _green_jacobians()
    result = compare_jacobians(analytic, numeric, tolerance=(1e-6, 1e-4))
    assert result["max_relative_error"] < 1e-6
    assert result["status"] == "pass"
    evidence = as_evidence(result, key="green.flux_field_derivative")
    assert (evidence.axis, evidence.status) == ("numerical", ValidationStatus.PASS)


def test_a_comparison_without_a_tolerance_carries_no_verdict():
    analytic, numeric = _green_jacobians()
    result = compare_jacobians(analytic, numeric)
    assert "status" not in result
    assert as_evidence(result, key="k").status is ValidationStatus.INDETERMINATE


def test_jacobians_align_by_name_and_refuse_self_comparison():
    a = Jacobian("analytic", "forward", "physical", ("x", "y"), ("f",), matrix=[[1.0, 2.0]])
    b = Jacobian("finite_difference", "forward", "physical", ("y", "x"), ("f",), matrix=[[2.0, 1.0]])
    assert compare_jacobians(a, b)["max_relative_error"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="repetition"):
        compare_jacobians(a, Jacobian("analytic", "forward", "physical", ("x", "y"), ("f",), matrix=[[1.0, 2.0]]))
    with pytest.raises(ValueError):
        Jacobian("guess", "forward", "physical", ("x",), ("f",), matrix=[[1.0]])
    with pytest.raises(ValueError):
        Jacobian("analytic", "forward", "physical", ("x",), ("f",))


def test_matrix_free_and_explicit_jacobians_agree():
    matrix = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]])
    free = Jacobian("autodiff", "observation", "physical", ("a", "b", "c"), ("y1", "y2"), jvp=lambda v: matrix @ v)
    explicit = Jacobian("finite_difference", "observation", "physical", ("a", "b", "c"), ("y1", "y2"), matrix=matrix)
    assert free.apply([1.0, 1.0, 1.0]) == pytest.approx(matrix @ [1.0, 1.0, 1.0])
    assert compare_jacobians(explicit, free)["max_relative_error"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# local against sampled, and where the local Jacobian breaks (#1642 s16)
# ---------------------------------------------------------------------------

NAMES = ("i_p", "b_t", "p_net")


def _confinement(p_net: float):
    alpha = {"i_p": 0.9, "b_t": 0.2, "p_net": p_net}
    cov = np.diag([0.05 ** 2, 0.1 ** 2, 0.05 ** 2])
    zero = np.zeros((3, 3))

    def indices(vector):
        return dimensionless_confinement_indices(dict(zip(NAMES, vector)), zero)["value"]

    return alpha, cov, indices, np.array([alpha[name] for name in NAMES])


def test_the_process_covariance_is_the_generic_kernels():
    alpha, cov, indices, x = _confinement(-0.6)
    reported = dimensionless_confinement_indices(alpha, cov)["cov"]
    generic = linear_covariance_propagation(finite_difference_jacobian(indices, x), cov)
    assert generic == pytest.approx(reported, rel=1e-4)


def test_linear_propagation_holds_far_from_the_singularity_and_fails_near_it():
    tolerance = (0.1, 0.5)  # this test's choice for the demonstration, in ln(sd ratio)
    alpha, cov, indices, x = _confinement(-0.6)
    linear = linear_covariance_propagation(finite_difference_jacobian(indices, x), cov)
    sampled = monte_carlo_propagation(indices, x, cov, samples=4000, seed=1)
    healthy = compare_linear_to_monte_carlo(linear, sampled, tolerance=tolerance)
    assert healthy["status"] == "pass"
    assert healthy["sampling_standard_error"] == pytest.approx(1 / math.sqrt(2 * 3999))

    # 1 + alpha_P = 0.07, one standard error 0.05 away from the pole.
    alpha, cov, indices, x = _confinement(-0.93)
    jac = finite_difference_jacobian(indices, x)
    linear = linear_covariance_propagation(jac, cov)
    sampled = monte_carlo_propagation(indices, x, cov, samples=4000, seed=1)
    broken = compare_linear_to_monte_carlo(linear, sampled, outputs=("mu_rho", "mu_beta", "mu_nu", "mu_q"),
                                           tolerance=tolerance)
    assert broken["status"] == "fail"
    assert broken["max_abs_log_sd_ratio"] > 2.0
    # The cheap diagnostic says the same before any sampling is paid for.
    assert linearity_ratio(indices, x, np.array([0.0, 0.0, 0.05]), jac) > 0.5
    assert linearity_ratio(*_confinement(-0.6)[2:], np.array([0.0, 0.0, 0.05]),
                           finite_difference_jacobian(*_confinement(-0.6)[2:])) < 0.2
    evidence = as_evidence(broken, key="confinement.indices_linear_uq", axis="inference", cost="expensive")
    assert evidence.status is ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# targeted scans (#1663) and layering
# ---------------------------------------------------------------------------


def _scan_report():
    def spread(value):
        return {"n": 5, "median": 1.0, "relative_half_iqr": value}

    return {
        "rows": 240,
        "configurations": [],
        "slices": [
            {"shot": 39915, "time_ms": 312, "model_form": {"wmhd": spread(0.10), "li": spread(0.02)},
             "model_form_admissible": {"wmhd": spread(0.30), "li": {"n": 0}}},
            {"shot": 39915, "time_ms": 317, "model_form": {"wmhd": spread(0.20), "li": spread(float("nan"))},
             "model_form_admissible": {"wmhd": spread(0.50), "li": spread(0.04)}},
        ],
        "marginal": {"dia": {"4.0": {"wmhd": spread(0.0)}}},
    }


def test_a_weight_scan_report_is_model_form_inference_evidence():
    evidence = scan_evidence(_scan_report())
    assert (evidence.axis, evidence.cost, evidence.status) == ("inference", "expensive",
                                                               ValidationStatus.INDETERMINATE)
    medians = evidence.metrics["median_relative_half_iqr"]
    assert medians["wmhd"] == {"good": pytest.approx(0.15), "admissible": pytest.approx(0.40)}
    assert medians["li"] == {"good": pytest.approx(0.02), "admissible": pytest.approx(0.04)}
    assert evidence.metrics["slices"] == 2 and evidence.metrics["records"] == 240
    assert scan_evidence({"slices": []}).status is ValidationStatus.NOT_AVAILABLE


def test_the_validation_side_imports_no_plotting_database_or_workflow():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, vaft.validation.sensitivity, vaft.formula.sensitivity\n"
         "print(','.join(sorted(m for m in sys.modules if m.startswith("
         "('matplotlib', 'vaft.database', 'vaft.plot', 'vaft.code', 'vaft.process', 'omas')))))"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
