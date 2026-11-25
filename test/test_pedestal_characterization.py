"""Pedestal characterization (#2335): conventions, covariance, localized resolution and status."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula import fit_profile
from vaft.formula.equilibrium import modified_tanh_profile, modified_tanh_profile_derivative
from vaft.formula.utils import (
    EPED_TANH_PARAMETERS,
    EpedTanhFit,
    _eped_tanh_model,
    fit_eped_tanh,
    tanh_pedestal_geometry,
)
from vaft.process._pedestal import _descriptors, _eped_tanh_derivative
from vaft.process.profile import (
    PEDESTAL_STATUSES,
    PedestalCharacterization,
    characterize_pedestal,
    pedestal_top,
)

TRUTH = dict(f0=3.0, f_ped=1.2, f_sep=0.1, x_ped=0.93, width=0.035, alpha=1.4, beta=1.8)
X = np.linspace(0.0, 1.05, 140)


def _h_mode(x=X, noise=0.01, seed=0, **overrides):
    parameters = {**TRUTH, **overrides}
    y = _eped_tanh_model(x, *parameters.values())
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, np.size(x))
    return y


def _truth_descriptors(**overrides):
    names = PedestalCharacterization.__dataclass_fields__["descriptor_names"].default
    return dict(zip(names, _descriptors(np.array(list({**TRUTH, **overrides}.values())))))


# --- the convention (formula layer) ---------------------------------------------


def test_geometry_is_groebner_full_width():
    top, foot, full, s_top, s_foot, s_full = tanh_pedestal_geometry(0.93, 0.035)
    assert (top, foot, full) == pytest.approx((0.895, 0.965, 0.07))
    # Unavailable uncertainty is nan, not zero.
    assert np.isnan([s_top, s_foot, s_full]).all()


def test_geometry_uncertainty_matches_the_issue_formula_and_a_monte_carlo():
    cov = np.array([[4e-6, 1.5e-6], [1.5e-6, 1e-6]])  # (x_c, w)
    _, _, _, s_top, s_foot, s_full = tanh_pedestal_geometry(0.93, 0.035, cov)
    var_c, var_delta, cov_c_delta = cov[0, 0], 4 * cov[1, 1], 2 * cov[0, 1]
    assert s_top**2 == pytest.approx(var_c + var_delta / 4 - cov_c_delta)
    assert s_foot**2 == pytest.approx(var_c + var_delta / 4 + cov_c_delta)
    assert s_full == pytest.approx(2e-3)
    draws = np.random.default_rng(0).multivariate_normal([0.93, 0.035], cov, 200_000)
    assert np.std(draws[:, 0] - draws[:, 1]) == pytest.approx(s_top, rel=0.01)
    assert np.std(draws[:, 0] + draws[:, 1]) == pytest.approx(s_foot, rel=0.01)


def test_geometry_refuses_a_non_positive_width_and_a_bad_covariance():
    with pytest.raises(ValueError, match="positive"):
        tanh_pedestal_geometry(0.9, 0.0)
    with pytest.raises(ValueError, match="2 x 2"):
        tanh_pedestal_geometry(0.9, 0.03, np.eye(3))


def test_eped_tanh_and_modified_tanh_agree_when_full_width_is_twice_the_native_scale():
    """The pure-tanh components of the two kernels are the same curve with Delta = 2w."""
    psi = np.linspace(0.6, 1.1, 301)
    x_c, w, f_ped, f_sep = 0.93, 0.035, 1.2, 0.1
    eped = f_sep + 0.5 * (f_ped - f_sep) * (1.0 - np.tanh((psi - x_c) / w))
    groebner = modified_tanh_profile(psi, pedestal_height=f_ped - f_sep, pedestal_position=x_c,
                                     pedestal_width=2 * w, edge_value=f_sep)
    np.testing.assert_allclose(eped, groebner, rtol=0, atol=1e-13)
    top, foot, *_ = tanh_pedestal_geometry(x_c, w)
    # z = +-1 at the canonical top and foot of the Groebner form.
    assert 2 * (x_c - top) / (2 * w) == pytest.approx(1.0)
    assert 2 * (x_c - foot) / (2 * w) == pytest.approx(-1.0)


def test_the_analytic_derivative_matches_finite_differences_and_the_groebner_kernel():
    theta = np.array(list(TRUTH.values()))
    x = np.linspace(0.05, 1.04, 97)
    h = 1e-6
    fd = (_eped_tanh_model(x + h, *theta) - _eped_tanh_model(x - h, *theta)) / (2 * h)
    np.testing.assert_allclose(_eped_tanh_derivative(x, *theta), fd, rtol=1e-5, atol=1e-6)
    # Outside the gated core the gradient is the Groebner kernel's with Delta = 2w.
    edge = np.linspace(0.94, 1.05, 23)
    expected = modified_tanh_profile_derivative(edge, pedestal_height=1.1, pedestal_position=0.93,
                                                pedestal_width=0.07)
    np.testing.assert_allclose(_eped_tanh_derivative(edge, *theta), expected, rtol=1e-12)


def test_the_barrier_step_shares_the_convention():
    from vaft.data import BarrierStep

    step = BarrierStep(position=0.93, width=0.07, height=1.0)  # full width
    top, foot, *_ = tanh_pedestal_geometry(0.93, 0.035)  # native scale
    assert (step.knee, step.foot) == pytest.approx((top, foot))


# --- the structured fit -----------------------------------------------------------


def test_fit_eped_tanh_keeps_covariance_status_and_names():
    fit = fit_eped_tanh(X, _h_mode())
    assert isinstance(fit, EpedTanhFit) and fit.success and fit.n_converged >= 1
    assert fit.parameters == EPED_TANH_PARAMETERS
    assert fit.units == ("y", "y", "y", "x", "x", "-", "-")
    assert fit.covariance.shape == (7, 7) and fit.covariance_available
    assert np.all(fit.std > 0)
    assert fit.values["x_ped"] == pytest.approx(0.93, abs=0.005)
    assert fit.degrees_of_freedom == X.size - 7
    assert fit.pedestal_max_correlation < 0.995
    assert fit.bound_saturated == ()


def test_fit_profile_keeps_the_four_tuple_and_coefficients_and_reports_real_uncertainty():
    y = _h_mode()
    y_eval, y_std, function, coeffs = fit_profile(X, y, None, X, fitting_function="eped_tanh")
    fit = function.eped_fit
    np.testing.assert_array_equal(coeffs, fit.coefficients)
    np.testing.assert_array_equal(y_eval, function(X))
    # No longer the zero placeholder: the linearized fit uncertainty.
    assert y_std.shape == y_eval.shape and np.all(y_std > 0) and np.all(np.isfinite(y_std))
    np.testing.assert_allclose(y_std, fit.profile_std(X))


def test_unavailable_covariance_is_nan_not_zero():
    fit = fit_eped_tanh(X, _h_mode())
    blind = EpedTanhFit(**{**{f: getattr(fit, f) for f in EpedTanhFit.__dataclass_fields__},
                           "covariance": np.full((7, 7), np.nan)})
    assert not blind.covariance_available
    assert np.isnan(blind.profile_std(X[:5])).all()
    assert np.isnan(blind.std).all()


def test_fit_eped_tanh_refuses_unmasked_input():
    y = _h_mode()
    with pytest.raises(ValueError, match="finite"):
        fit_eped_tanh(X, np.where(X > 0.5, np.nan, y))
    with pytest.raises(ValueError, match="y_std"):
        fit_eped_tanh(X, y, np.zeros_like(y))


def test_fit_profile_is_the_single_start_legacy_fit():
    y = _h_mode(noise=0.15, seed=11, f_ped=0.8)
    _, _, function, coeffs = fit_profile(X, y, None, X, fitting_function="eped_tanh")
    assert function.eped_fit.n_starts == 1
    np.testing.assert_array_equal(coeffs, fit_eped_tanh(X, y, n_starts=1).coefficients)


def test_the_legacy_seed_wins_ties_so_coefficients_are_reproducible():
    """With one start the fit is the pre-#2335 single curve_fit; the multi-start
    keeps exactly that answer unless another start is strictly better."""
    y = _h_mode(noise=0.02, seed=3)
    single = fit_eped_tanh(X, y, n_starts=1)
    multi = fit_eped_tanh(X, y, n_starts=4)
    assert single.n_starts == 1 and multi.n_starts == 4
    np.testing.assert_array_equal(single.coefficients, multi.coefficients)


# --- characterization -------------------------------------------------------------


def test_known_descriptors_are_recovered():
    result = characterize_pedestal(X, _h_mode(noise=0.003), quantity="T_e")
    truth = _truth_descriptors()
    assert result.status == "present" and result.reason == ""
    for name in ("top", "center", "foot"):
        assert getattr(result, name) == pytest.approx(truth[name], abs=0.003), name
    assert result.full_width == pytest.approx(0.07, abs=0.004)
    assert result.native_width == pytest.approx(0.035, abs=0.002)
    for name in ("value_top", "value_foot", "value_separatrix", "drop", "height"):
        assert getattr(result, name) == pytest.approx(truth[name], rel=0.03, abs=0.01), name
    assert result.gradient_max < 0
    assert result.gradient_max == pytest.approx(truth["gradient_max"], rel=0.05)
    assert result.gradient_position == pytest.approx(0.93, abs=0.003)
    assert result.foot_location == "inside"
    # Asymptotic height and evaluated drop are different numbers.
    assert result.drop == pytest.approx(np.tanh(1.0) * result.height, rel=0.05)


def test_the_gradient_of_a_pure_tanh_is_minus_height_over_full_width():
    truth = _truth_descriptors()
    assert truth["gradient_max"] == pytest.approx(-1.1 / 0.07, rel=0.02)


def test_an_mtanh_profile_is_characterized_in_the_shared_convention():
    """Not self-consistency: data from Groebner's form, fitted by the EPED tanh."""
    y = modified_tanh_profile(X, pedestal_height=2.0, pedestal_position=0.92,
                              pedestal_width=0.06, core_slope=0.0, edge_value=0.1)
    y = y + 0.6 * np.clip(1 - (X / 0.92) ** 2, 0, None) ** 1.5
    result = characterize_pedestal(X, y, quantity="T_e")
    assert result.status == "present"
    assert result.full_width == pytest.approx(0.06, abs=0.01)
    assert result.top == pytest.approx(0.89, abs=0.01)
    assert result.foot == pytest.approx(0.95, abs=0.01)


def test_foot_location_relative_to_the_lcfs():
    outside = characterize_pedestal(X, _h_mode(noise=0.003, x_ped=0.98, width=0.03), quantity="n_e")
    assert outside.status == "present" and outside.foot > 1.0
    assert outside.foot_location == "outside"
    at = characterize_pedestal(X, _h_mode(noise=0.003, x_ped=0.97, width=0.03), quantity="n_e")
    assert at.foot_location == "at"


def test_covariance_uncertainties_cover_a_synthetic_ensemble():
    """One-sigma intervals contain the truth about 68 % of the time."""
    x = np.linspace(0.4, 1.05, 60)
    sigma = 0.03
    truth = _truth_descriptors()
    names = ("top", "center", "foot", "full_width", "value_top", "drop")
    hits = {name: 0 for name in names}
    runs = 60
    for seed in range(runs):
        y = _h_mode(x, noise=sigma, seed=1000 + seed)
        result = characterize_pedestal(x, y, quantity="T_e", value_std=np.full(x.size, sigma))
        assert result.status == "present"
        for name in names:
            hits[name] += abs(getattr(result, name) - truth[name]) <= result.uncertainties[name]
    for name, count in hits.items():
        assert 0.5 <= count / runs <= 0.88, (name, count / runs)


def test_descriptor_intervals_use_the_propagated_sigma():
    result = characterize_pedestal(X, _h_mode(noise=0.01), quantity="T_e")
    low, high = result.interval("top")
    assert high - low == pytest.approx(2 * result.uncertainties["top"])
    assert result.descriptor_covariance.shape == (12, 12)


def test_evidence_counts_independent_samples_and_records_the_definition():
    result = characterize_pedestal(X, _h_mode(), quantity="T_e")
    evidence = result.evidence
    assert evidence["n_used"] == np.count_nonzero((X >= 0.4) & (X <= 1.05))
    assert evidence["n_gradient_region"] >= 2 and evidence["n_top_region"] >= 2
    assert evidence["coordinate"] == "psi_norm" and "2 w" in evidence["width_definition"]
    assert evidence["extrapolated"] is False
    assert evidence["drop_significance"] > 3

    # Duplicated positions are not independent observations.
    doubled = characterize_pedestal(np.repeat(X, 2), np.repeat(_h_mode(), 2), quantity="T_e")
    assert doubled.evidence["n_used"] == 2 * evidence["n_used"]
    assert doubled.evidence["n_unique"] == evidence["n_unique"]
    assert doubled.evidence["n_gradient_region"] == evidence["n_gradient_region"]


def test_pure_noise_is_never_present():
    for seed in range(4):
        y = 2.0 + np.random.default_rng(seed).normal(0.0, 0.5, X.size)
        result = characterize_pedestal(X, y, quantity="T_e")
        assert result.status in {"unresolved", "absent"}, seed
        # Noise this large cannot establish absence either.
        assert result.status == "unresolved" and "absence not established" in result.reason


@pytest.mark.parametrize("profile", [
    lambda x: 1.0 - 0.3 * x**2,
    lambda x: np.clip(1 - x**2, 0, None) ** 2,
    lambda x: (1 - x / 1.1) ** 3,
])
def test_smooth_profiles_are_absent(profile):
    result = characterize_pedestal(X, profile(X), quantity="n_e")
    assert result.status == "absent", result.reason
    assert result.evidence["absence_evidence_adequate"]


def test_a_weak_pedestal_in_noise_is_unresolved_not_absent():
    y = _h_mode(noise=0.1, f_ped=0.2, f0=2.0)
    result = characterize_pedestal(X, y, quantity="T_e")
    assert result.status == "unresolved"


def test_a_spatially_unresolved_pedestal_is_unresolved():
    """Seven positions, each sampled three times within 2e-4: twenty-one
    samples pass min_points, but only seven are independent and the step's
    width is not resolved."""
    x = np.array([0.45, 0.6, 0.72, 0.82, 0.9, 0.99, 1.04] * 3) + np.repeat([0, 1e-4, 2e-4], 7)
    y = _h_mode(x, noise=0.005, seed=2)
    result = characterize_pedestal(x, y, quantity="T_e")
    assert result.evidence["n_used"] == 21 and result.evidence["n_unique"] == 7
    assert result.evidence["n_gradient_region"] <= 1
    assert result.status == "unresolved", result.reason


@pytest.mark.parametrize("seed", [0, 3, 8])
def test_an_insignificant_real_pedestal_is_never_absent(seed):
    """Thirty noisy samples of a true pedestal: failing to resolve it is not
    evidence that it is not there."""
    x = np.linspace(0.4, 1.05, 30)
    y = _h_mode(x, noise=0.1, seed=seed, f_ped=0.5)
    result = characterize_pedestal(x, y, quantity="T_e")
    assert result.status in {"present", "unresolved"}, result.reason


def test_irregular_sampling_is_characterized():
    rng = np.random.default_rng(7)
    x = np.sort(np.concatenate([rng.uniform(0.4, 0.85, 15), rng.uniform(0.85, 1.05, 30)]))
    result = characterize_pedestal(x, _h_mode(x, noise=0.01, seed=4), quantity="T_e")
    assert result.status == "present"
    assert result.center == pytest.approx(0.93, abs=0.01)


def test_minimum_data_is_checked_after_masking_invalid_sigma():
    """Twenty-two samples in the window, five with an invalid sigma: the
    count the fit actually sees is seventeen, which is not enough."""
    x = np.linspace(0.4, 1.05, 22)
    y = _h_mode(x, noise=0.005)
    sigma = np.full(x.size, 0.01)
    sigma[[1, 4, 9, 13, 20]] = [0.0, -1.0, np.nan, np.inf, 0.0]
    result = characterize_pedestal(x, y, quantity="T_e", value_std=sigma)
    assert result.status == "not_available"
    assert result.evidence["n_invalid_sigma"] == 5 and result.evidence["n_used"] == 17
    assert "after masking" in result.reason
    assert np.isnan(result.top)


def test_absence_needs_edge_coverage():
    x = np.linspace(0.4, 0.9, 80)  # stops well inside the LCFS
    result = characterize_pedestal(x, 1.0 - 0.3 * x**2, quantity="n_e")
    assert result.status == "unresolved"
    assert result.evidence["absence_evidence_adequate"] is False


def test_statuses_are_from_the_declared_vocabulary():
    assert PEDESTAL_STATUSES == ("present", "absent", "unresolved", "fit_failed", "not_available")
    for result in (
        characterize_pedestal(X, _h_mode(), quantity="T_e"),
        characterize_pedestal(X, 1.0 - 0.3 * X**2, quantity="T_e"),
        characterize_pedestal(X[:5], _h_mode()[:5], quantity="T_e"),
    ):
        assert result.status in PEDESTAL_STATUSES


def test_independent_quantities_are_characterized_independently():
    """n_e and T_e need not share a pedestal; each result says which it is."""
    ne = characterize_pedestal(X, _h_mode(noise=0.003, x_ped=0.95, width=0.02), quantity="n_e")
    te = characterize_pedestal(X, _h_mode(noise=0.003, x_ped=0.9, width=0.04), quantity="T_e")
    assert (ne.quantity, te.quantity) == ("n_e", "T_e")
    assert ne.center == pytest.approx(0.95, abs=0.005) and te.center == pytest.approx(0.9, abs=0.005)


def test_coordinate_is_validated():
    with pytest.raises(ValueError, match="coordinate must be one of"):
        characterize_pedestal(X, _h_mode(), quantity="T_e", coordinate="r_minor")


# --- the legacy wrapper ------------------------------------------------------------


def test_pedestal_top_carries_status_canonical_boundaries_and_legacy_inner_edge():
    result = pedestal_top(X, _h_mode(), quantity="p_total", acceptance="localized")
    c = result.characterization
    assert result.status == result.detection_status == "present"
    assert result.position == c.center and result.width == c.native_width
    assert result.top == pytest.approx(c.top) and result.foot == pytest.approx(c.foot)
    assert result.full_width == pytest.approx(2 * result.width)
    # inner_edge keeps its historical value, a quarter full width inside the centre.
    assert result.inner_edge == result.position - 0.5 * result.width
    assert result.top < result.inner_edge < result.center


def test_the_fallback_is_labelled_a_partition_policy_with_the_real_status():
    result = pedestal_top(X, 1.0 - 0.3 * X**2, quantity="n_e")  # legacy default
    assert result.method == "fallback" and result.position == 0.85
    assert result.status == "absent"
    assert result.top is None and result.foot is None and result.full_width is None


def test_positional_construction_still_works():
    from vaft.process.profile import PedestalTop

    legacy = PedestalTop(0.93, "eped_fit", "p_total", "psi_norm", 0.06)
    assert legacy.status == "" and legacy.detection_status == "present"
    assert legacy.inner_edge == pytest.approx(0.90) and legacy.top == pytest.approx(0.87)
    assert PedestalTop(0.85, "fallback", "p_total", "psi_norm").detection_status == "not_available"


@pytest.mark.parametrize("seed, noise, method, position, width, reason", [
    # Recorded with the pre-#2335 pedestal_top on develop f878d31ae.
    (0, 0.02, "eped_fit", 0.9293154875825744, 0.03174426351505591, ""),
    (0, 0.1, "eped_fit", 0.9266388859869035, 0.022182527668107205, ""),
    (0, 0.3, "fallback", 0.85, None, "width rests on a bound of the model's parameter box"),
])
def test_legacy_acceptance_reproduces_the_pre_2335_result(seed, noise, method, position, width, reason):
    y = _h_mode(noise=noise, seed=seed)
    legacy = pedestal_top(X, y, quantity="T_e", acceptance="legacy")
    assert legacy.method == method and legacy.reason == reason
    assert legacy.position == pytest.approx(position, abs=1e-7)
    if width is None:
        assert legacy.width is None
    else:
        assert legacy.width == pytest.approx(width, abs=1e-7)
    # The status of the characterization is still recorded.
    assert legacy.status in PEDESTAL_STATUSES


def test_the_default_reproduces_develop_on_the_reference_cases():
    """The default stays the pre-#2335 acceptance until the localized test is
    qualified on VEST/GPEC benchmarks (#2335 section 7)."""
    import json
    from pathlib import Path

    table = json.loads((Path(__file__).parent / "data" / "pedestal_top_develop_f878d31ae.json").read_text())
    disagreements = 0
    for row in table["rows"]:
        y = _h_mode(noise=row["noise"], seed=row["seed"])
        result = pedestal_top(X, y, quantity="T_e")
        assert result.method == row["method"] and result.reason == row["reason"], row
        assert result.position == pytest.approx(row["position"], abs=1e-7), row
        if row["width"] is None:
            assert result.width is None
        else:
            assert result.width == pytest.approx(row["width"], abs=1e-7), row
        # The localized verdict is still recorded, for information.
        assert result.status in PEDESTAL_STATUSES and result.characterization is not None
        disagreements += result.from_fit != (result.status == "present")
    assert len(table["rows"]) == 48
    assert disagreements > 0  # the noisy cases where the two tests differ are visible


def test_localized_acceptance_is_opt_in_and_follows_the_status():
    for seed in range(12):
        y = _h_mode(noise=0.3, seed=seed)
        legacy = pedestal_top(X, y, quantity="T_e")
        if legacy.from_fit and legacy.status != "present":
            break
    else:  # pragma: no cover - the reference table shows such cases exist
        pytest.fail("no case where the two acceptances disagree")
    localized = pedestal_top(X, y, quantity="T_e", acceptance="localized")
    assert localized.method == "fallback" and localized.status == legacy.status
    assert localized.reason == legacy.characterization.reason


def test_an_unknown_acceptance_is_refused():
    with pytest.raises(ValueError, match="acceptance"):
        pedestal_top(X, _h_mode(), quantity="T_e", acceptance="strict")
