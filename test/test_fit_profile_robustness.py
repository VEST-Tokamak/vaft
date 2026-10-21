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


def test_linear_mode_is_nan_outside_the_data_by_default():
    # numpy.interp holds the end values; a profile past the last channel must
    # come back missing, not as a flat artefact (#359).
    x = np.array([0.2, 0.4, 0.6])
    y = np.array([2.0, 4.0, 6.0])
    x_eval = np.array([0.0, 0.3, 0.6, 0.9])
    with pytest.warns(UserWarning, match=r"nan at 2 evaluation point\(s\) outside the data range"):
        y_eval, y_std, fit, coeffs = fit_profile(x, y, None, x_eval, fitting_function='linear')
    np.testing.assert_allclose(y_eval[1:3], [3.0, 6.0], rtol=1e-12)
    assert np.isnan(y_eval[[0, 3]]).all()
    assert np.isnan(y_std[[0, 3]]).all()
    np.testing.assert_array_equal(y_std[1:3], 0.0)
    assert np.isnan(fit(np.array([1.0]))).all()
    assert coeffs is None


def test_linear_mode_extrapolates_only_when_asked():
    x = np.array([0.2, 0.4, 0.6])
    y = np.array([2.0, 4.0, 6.0])
    x_eval = np.array([0.0, 0.3, 0.9])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y_eval, y_std, fit, _ = fit_profile(x, y, None, x_eval, fitting_function='linear',
                                            extrapolate=True)
    np.testing.assert_allclose(y_eval, [2.0, 3.0, 6.0], rtol=1e-12)
    np.testing.assert_array_equal(y_std, 0.0)
    np.testing.assert_allclose(fit(np.array([1.0])), [6.0])


def test_linear_mode_inside_the_data_does_not_warn():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y_eval, _, _, _ = fit_profile([0.0, 1.0], [1.0, 3.0], None, [0.0, 0.5, 1.0],
                                      fitting_function='linear')
    np.testing.assert_allclose(y_eval, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("mode", ["sqrt", "sqrt_exponential"])
def test_sqrt_modes_warn_when_they_clip_negative_data(mode):
    x = np.linspace(0.0, 0.9, 10)
    y = 10.0 * (1.0 - x**2)
    y[-2:] = [-0.5, -0.2]
    with pytest.warns(UserWarning, match="clipped 2 negative data point"):
        fit_profile(x, y, None, x, order=3, fitting_function=mode)


def test_sqrt_modes_are_silent_on_non_negative_data():
    import warnings
    x = np.linspace(0.0, 0.9, 10)
    y = 10.0 * np.sqrt(1.0 - x**2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fit_profile(x, y, None, x, order=3, fitting_function='sqrt')
