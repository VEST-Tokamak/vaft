"""The EPED-style pedestal fit and the boundary it produces (decision D-05)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula import fit_profile
from vaft.formula.utils import _eped_tanh_model
from vaft.process.profile import (
    COORDINATES,
    PEDESTAL_FALLBACK_PSI_NORM,
    PEDESTAL_RESOLUTION_FACTOR,
    FittedProfile,
    PedestalTop,
    pedestal_top,
)

TRUTH = dict(f0=3.0, f_ped=1.2, f_sep=0.1, x_ped=0.93, width=0.035, alpha=1.4, beta=1.8)


def _h_mode(points=140, noise=0.01, seed=0, **overrides):
    parameters = {**TRUTH, **overrides}
    x = np.linspace(0.0, 1.05, points)
    y = _eped_tanh_model(x, *parameters.values())
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, x.size)
    return x, y, parameters


# --- the model ----------------------------------------------------------------


def test_the_model_is_continuous_at_the_pedestal_top_and_reaches_the_axis_value():
    x = np.linspace(0.0, 1.05, 400)
    y = _eped_tanh_model(x, **TRUTH)
    assert y[0] == pytest.approx(TRUTH["f0"], rel=1e-6)
    # No step where the gated core term switches off.
    assert np.abs(np.diff(y)).max() < 0.2


def test_the_model_does_not_diverge_away_from_the_pedestal():
    """The legacy modified tanh multiplied its slope across the whole domain."""
    far = _eped_tanh_model(np.array([-2.0, 3.0]), **TRUTH)
    assert np.all(np.isfinite(far))
    assert np.all(np.abs(far) < 10.0 * TRUTH["f0"])


def test_the_fit_recovers_known_parameters(): 
    x, y, truth = _h_mode()
    _, _, _, coefficients = fit_profile(x, y, None, x, fitting_function="eped_tanh")
    assert coefficients[3] == pytest.approx(truth["x_ped"], abs=0.01)
    assert coefficients[4] == pytest.approx(truth["width"], abs=0.01)


# --- the boundary -------------------------------------------------------------


def test_an_h_mode_profile_gives_a_fitted_boundary():
    x, y, truth = _h_mode()
    result = pedestal_top(x, y, quantity="p_total")
    assert isinstance(result, PedestalTop)
    assert result.method == "eped_fit" and result.from_fit
    assert result.position == pytest.approx(truth["x_ped"], abs=0.01)
    assert result.width == pytest.approx(truth["width"], abs=0.01)
    assert result.reason == ""
    assert result.quantity == "p_total"


def test_the_result_carries_the_fit_as_a_fitted_profile():
    x, y, _ = _h_mode()
    result = pedestal_top(x, y, quantity="p_total")
    assert isinstance(result.fit, FittedProfile)
    assert result.fit.method == "eped_tanh"
    assert result.fit.coordinate == "psi_norm"
    # Callable like any other fit, and it reproduces the profile it came from.
    np.testing.assert_allclose(result.fit(x[-5:]), y[-5:], atol=0.05)


def test_an_l_mode_profile_falls_back_and_says_why():
    """No pedestal to find is the intended answer, not a failure."""
    x = np.linspace(0.0, 1.05, 140)
    result = pedestal_top(x, 1.0 - 0.3 * x**2, quantity="n_e")
    assert result.method == "fallback" and not result.from_fit
    assert result.position == PEDESTAL_FALLBACK_PSI_NORM == 0.85
    assert result.reason
    assert result.fit is None and result.width is None


def test_too_few_points_in_the_window_falls_back():
    x, y, _ = _h_mode(points=200)
    inside = (x >= 0.4) & (x <= 1.05)
    result = pedestal_top(x[inside][:6], y[inside][:6], quantity="p_total")
    assert result.method == "fallback"
    assert "min_points" in result.reason


def test_pure_noise_does_not_pass_as_a_pedestal():
    """The criterion is the fitted curve's own variation, not its parameters.

    On this profile the fit returns ``f_ped - f_sep = 2.2`` against a residual
    of 0.5 -- so a height-versus-residual test would accept it -- while the
    curve it actually draws varies by only 0.65 across the window.
    """
    x = np.linspace(0.0, 1.05, 140)
    noise = np.random.default_rng(3).normal(0.0, 0.5, x.size)
    result = pedestal_top(x, np.full_like(x, 2.0) + noise, quantity="T_e")
    assert result.method == "fallback"
    assert "no pedestal resolved" in result.reason


def test_the_resolution_factor_is_what_separates_noise_from_a_pedestal():
    x, y, _ = _h_mode()
    assert pedestal_top(x, y, quantity="p_total").method == "eped_fit"
    assert PEDESTAL_RESOLUTION_FACTOR == 3.0


@pytest.mark.parametrize("method", ["eped_fit", "fallback"])
def test_every_result_records_the_method_that_produced_it(method):
    """D-05: a reduction that does not record this cannot be compared."""
    if method == "eped_fit":
        x, y, _ = _h_mode()
    else:
        x = np.linspace(0.0, 1.05, 140)
        y = 1.0 - 0.3 * x**2
    result = pedestal_top(x, y, quantity="p_total")
    assert result.method == method
    assert (result.reason == "") is (method == "eped_fit")


# --- the coordinate contract ---------------------------------------------------


def test_the_position_is_in_the_declared_coordinate_and_is_not_converted():
    x, y, truth = _h_mode()
    as_psi = pedestal_top(x, y, quantity="p_total", coordinate="psi_norm")
    as_rho = pedestal_top(x, y, quantity="p_total", coordinate="rho_tor_norm")
    assert as_psi.coordinate == "psi_norm" and as_rho.coordinate == "rho_tor_norm"
    assert as_psi.position == pytest.approx(as_rho.position)
    assert as_rho.fit.coordinate == "rho_tor_norm"


def test_an_unknown_coordinate_is_refused():
    x, y, _ = _h_mode()
    with pytest.raises(ValueError, match="coordinate must be one of"):
        pedestal_top(x, y, quantity="p_total", coordinate="r_minor")
    assert "psi_norm" in COORDINATES


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        pedestal_top([0.0, 0.5, 1.0], [1.0, 2.0], quantity="n_e")


def test_the_quantity_is_required():
    x, y, _ = _h_mode()
    with pytest.raises(TypeError):
        pedestal_top(x, y)


def test_uncertainties_are_accepted_as_weights():
    x, y, truth = _h_mode(noise=0.02, seed=5)
    sigma = np.full_like(x, 0.02)
    weighted = pedestal_top(x, y, quantity="p_total", value_std=sigma)
    assert weighted.method == "eped_fit"
    assert weighted.position == pytest.approx(truth["x_ped"], abs=0.02)


def test_non_finite_samples_are_dropped_rather_than_poisoning_the_fit():
    x, y, truth = _h_mode()
    y = y.copy()
    y[10] = np.nan
    y[-3] = np.inf
    result = pedestal_top(x, y, quantity="p_total")
    assert result.method == "eped_fit"
    assert result.position == pytest.approx(truth["x_ped"], abs=0.01)
