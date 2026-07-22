"""Robustness tests for vaft.formula.utils.fit_profile."""

import numpy as np
import pytest

from vaft.formula.utils import fit_profile


@pytest.fixture()
def profile_data():
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 0.9, 15)
    y_true = 300.0 * (1.0 - x**2) ** 1.5
    y = y_true + rng.normal(0.0, 5.0, x.size)
    y_std = np.full(x.size, 8.0)
    x_eval = np.linspace(0.0, 1.0, 50)
    return x, y, y_std, x_eval


def test_large_magnitude_data_fits_without_normalization(profile_data):
    """Raw SI densities (~1e18) must not silently return the initial guess."""
    x, _, _, x_eval = profile_data
    y = 5e18 * (1.0 - x**2)
    _, _, fit, _ = fit_profile(x, y, None, x_eval, order=3,
                               fitting_function='polynomial')
    assert float(fit(np.array([0.0]))[0]) > 1e18


def test_zero_sigma_point_is_dropped(profile_data):
    """A zero-uncertainty channel must not hijack the weighted fit."""
    x, y, y_std, x_eval = profile_data
    y_ref, _, fit_ref, _ = fit_profile(x, y, y_std, x_eval, order=3,
                                       fitting_function='polynomial')
    y_bad = y.copy()
    y_bad[7] = 0.0
    y_std_bad = y_std.copy()
    y_std_bad[7] = 0.0
    y_fit, _, fit_bad, _ = fit_profile(x, y_bad, y_std_bad, x_eval, order=3,
                                       fitting_function='polynomial')
    axis_ref = float(fit_ref(np.array([0.0]))[0])
    axis_bad = float(fit_bad(np.array([0.0]))[0])
    assert abs(axis_bad - axis_ref) / axis_ref < 0.2


def test_nan_point_is_masked_not_fatal(profile_data):
    x, y, y_std, x_eval = profile_data
    y_bad = y.copy()
    y_bad[3] = np.nan
    y_eval, _, _, _ = fit_profile(x, y_bad, y_std, x_eval, order=3,
                                  fitting_function='polynomial')
    assert np.all(np.isfinite(y_eval))


def test_linear_mode_sorts_x():
    x = np.array([0.5, 0.1, 0.9, 0.3])
    y = np.array([5.0, 1.0, 9.0, 3.0])  # y = 10x
    x_eval = np.array([0.2, 0.4, 0.7])
    y_eval, _, _, _ = fit_profile(x, y, None, x_eval, fitting_function='linear')
    np.testing.assert_allclose(y_eval, [2.0, 4.0, 7.0], rtol=1e-12)


def test_gp_accepts_none_y_std(profile_data):
    pytest.importorskip("sklearn")
    x, y, _, x_eval = profile_data
    y_eval, _, _, _ = fit_profile(x, y, None, x_eval, fitting_function='gp')
    assert np.all(np.isfinite(y_eval))


def test_list_inputs_accepted():
    x = [0.0, 0.2, 0.4, 0.6, 0.8]
    y = [100.0, 90.0, 70.0, 40.0, 15.0]
    y_eval, _, _, _ = fit_profile(x, y, None, [0.1, 0.5], order=2,
                                  fitting_function='polynomial')
    assert np.all(np.isfinite(y_eval))


def test_too_few_valid_points_raises():
    with pytest.raises(ValueError):
        fit_profile([0.1, 0.5], [np.nan, 3.0], None, [0.2], order=2,
                    fitting_function='polynomial')
