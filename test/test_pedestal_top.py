"""The EPED-style pedestal fit and the boundary it produces (decision D-05)."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula import fit_profile
from vaft.formula.utils import _eped_tanh_model
from vaft.process.profile import (
    COORDINATES,
    PEDESTAL_FALLBACK_PSI_NORM,
    PEDESTAL_MAX_WIDTH,
    PEDESTAL_RESOLUTION_FACTOR,
    CoordinateUnavailableError,
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
    for seed in range(8):
        noise = np.random.default_rng(seed).normal(0.0, 0.5, x.size)
        result = pedestal_top(x, np.full_like(x, 2.0) + noise, quantity="T_e")
        assert result.method == "fallback", seed
        assert result.reason


def test_a_tanh_too_wide_to_be_a_pedestal_falls_back():
    """The model's box allows a width of 0.3, which is not an edge feature.

    Over one MAST campaign's 171 fits, 25 came back between 0.20 and 0.29 --
    ramps across half the minor radius, every one of them reporting itself as
    a measured pedestal before this criterion existed.
    """
    x, y, truth = _h_mode()
    assert PEDESTAL_MAX_WIDTH == 0.15
    accepted = pedestal_top(x, y, quantity="p_total")
    assert accepted.method == "eped_fit" and accepted.width < PEDESTAL_MAX_WIDTH

    narrow = pedestal_top(x, y, quantity="p_total", max_width=0.5 * truth["width"])
    assert narrow.method == "fallback"
    assert "not a pedestal" in narrow.reason


def test_both_rejection_criteria_are_reachable():
    """Neither the width test nor the variation test is dead code."""
    x, y, truth = _h_mode()
    width_reason = pedestal_top(x, y, quantity="p_total", max_width=0.001).reason
    assert "wide" in width_reason

    reasons = set()
    for seed in range(8):
        noise = np.random.default_rng(seed).normal(0.0, 0.5, x.size)
        reasons.add(pedestal_top(x, np.full_like(x, 2.0) + noise, quantity="T_e").reason)
    assert any("varies by" in reason for reason in reasons)
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


def test_the_position_is_in_whichever_coordinate_the_samples_were_in():
    """Two grids for the same profile give two different positions, each
    labelled with the coordinate it was computed in.  Nothing is converted."""
    x, y, truth = _h_mode()
    as_psi = pedestal_top(x, y, quantity="p_total", coordinate="psi_norm")
    # The same profile against sqrt(psi): a genuinely different grid.
    as_rho = pedestal_top(np.sqrt(x), y, quantity="p_total", coordinate="rho_pol_norm")

    assert as_psi.coordinate == "psi_norm" and as_rho.coordinate == "rho_pol_norm"
    assert as_rho.position == pytest.approx(np.sqrt(as_psi.position), abs=0.02)
    assert as_rho.position != pytest.approx(as_psi.position, abs=0.01)
    assert as_rho.fit.coordinate == "rho_pol_norm"


def test_the_fallback_is_refused_outside_psi_norm():
    """0.85 is a psi_norm position; returning it under another label would put
    the boundary a whole pedestal width away and still call it D-05."""
    x = np.linspace(0.0, 1.05, 140)
    flat = 1.0 - 0.3 * x**2
    with pytest.raises(CoordinateUnavailableError, match="is a psi_norm position"):
        pedestal_top(x, flat, quantity="n_e", coordinate="rho_tor_norm")
    # An explicit fallback in the caller's own coordinate is fine.
    result = pedestal_top(x, flat, quantity="n_e", coordinate="rho_tor_norm", fallback=0.92)
    assert result.method == "fallback" and result.position == 0.92


def test_the_inner_edge_is_half_a_width_inside_the_centre():
    x, y, _ = _h_mode()
    result = pedestal_top(x, y, quantity="p_total")
    assert result.inner_edge == pytest.approx(result.position - 0.5 * result.width)
    assert pedestal_top(x, 1.0 - 0.3 * x**2, quantity="n_e").inner_edge is None


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


def test_uncertainties_actually_weight_the_fit():
    """A *constant* sigma is a no-op for the fitted parameters, so this uses a
    varying one: distrusting the edge has to move the answer."""
    x, y, truth = _h_mode(noise=0.02, seed=5)
    sigma = np.where(x > truth["x_ped"], 1.0, 0.001)

    plain = pedestal_top(x, y, quantity="p_total")
    weighted = pedestal_top(x, y, quantity="p_total", value_std=sigma)
    assert plain.method == weighted.method == "eped_fit"
    assert weighted.position != pytest.approx(plain.position, abs=1e-9)


def test_a_value_std_of_the_wrong_length_is_refused():
    x, y, _ = _h_mode()
    with pytest.raises(ValueError, match="value_std has 5 samples"):
        pedestal_top(x, y, quantity="p_total", value_std=np.ones(5))


def test_non_finite_samples_are_dropped_rather_than_poisoning_the_fit():
    x, y, truth = _h_mode()
    y = y.copy()
    y[10] = np.nan
    y[-3] = np.inf
    result = pedestal_top(x, y, quantity="p_total")
    assert result.method == "eped_fit"
    assert result.position == pytest.approx(truth["x_ped"], abs=0.01)


# --- profiles the legacy model could not fit ----------------------------------


def test_an_all_negative_profile_is_fitted():
    """A rotation profile is negative everywhere.

    The legacy fitter could not do this at all: for a narrow-range negative
    profile its parameter box inverts, and otherwise its seed falls outside
    the box and SciPy refuses it.
    """
    x, y, truth = _h_mode()
    result = pedestal_top(x, -y, quantity="omega_tor")
    assert result.method == "eped_fit"
    assert result.position == pytest.approx(truth["x_ped"], abs=0.02)


def test_a_profile_from_a_different_functional_form_is_fitted():
    """Not self-consistency: this profile is an mtanh, not the fitted model."""
    x = np.linspace(0.0, 1.05, 140)
    x_ped, width = 0.94, 0.03
    z = (x - x_ped) / width
    y = 0.1 + 0.5 * (2.4 - 0.1) * (1.0 - np.tanh(z)) * (1.0 + 0.15 * np.where(z < 0, -z, 0.0))
    result = pedestal_top(x, y, quantity="T_e")
    assert result.method == "eped_fit"
    assert result.position == pytest.approx(x_ped, abs=0.03)
    assert result.width == pytest.approx(width, abs=0.03)


def test_the_model_refuses_a_non_positive_width():
    """A negative width inverts the pedestal silently; bounds protect the fit
    path, and this protects a direct caller."""
    with pytest.raises(ValueError, match="width must be positive"):
        _eped_tanh_model(np.array([0.5]), 3.0, 1.2, 0.1, 0.93, -0.035, 1.4, 1.8)


def test_the_core_shape_exponents_cannot_go_below_one():
    """Below one the gated core term has an infinite slope -- at the pedestal
    top for beta, at the magnetic axis for alpha."""
    from vaft.formula.utils import eped_tanh_bounds

    x, y, _ = _h_mode()
    lower, upper = eped_tanh_bounds(x, y)
    assert lower[5] >= 1.0 and lower[6] >= 1.0
