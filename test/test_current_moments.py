"""Toroidal current-density moments (#943, current-distribution part).

Analytic distributions with known moments -- a uniform disk, a rotated
elliptical Gaussian, a skewed profile, a filament -- pin the definitions; the
Solov'ev family pins the equilibrium path, including its sign in every COCOS.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vaft.process.equilibrium import (
    current_centroid,
    current_covariance,
    current_moment,
    derive_current_moments,
    fractional_cell_weights_from_boundary,
    solovev_example,
)


def _grid(n=201, half=0.5, r0=1.0):
    r = np.linspace(r0 - half, r0 + half, n); z = np.linspace(-half, half, n)
    return r, z, *np.meshgrid(r, z, indexing="ij")


def _disk_weights(r, z, rc, zc, a):
    t = np.linspace(0, 2*np.pi, 721)
    return fractional_cell_weights_from_boundary(r, z, rc + a*np.cos(t), zc + a*np.sin(t))


def test_uniform_disk_has_the_textbook_moments():
    r, z, _, _ = _grid()
    a, jc = 0.3, 2.0e6
    w = _disk_weights(r, z, 1.02, 0.05, a)
    total, rc, zc = current_centroid(np.full((r.size, z.size), jc), r, z, weights=w)
    assert total == pytest.approx(jc*np.pi*a**2, rel=2e-3)
    assert (rc, zc) == (pytest.approx(1.02, abs=1e-4), pytest.approx(0.05, abs=1e-4))
    cov = current_covariance(np.full((r.size, z.size), jc), r, z, weights=w)
    np.testing.assert_allclose(cov, [[a**2/4, 0], [0, a**2/4]], atol=2e-4)
    for p, q in ((3, 0), (2, 1), (1, 2), (0, 3)):
        assert abs(current_moment(np.full((r.size, z.size), jc), r, z, p, q, weights=w)) < 1e-5


def test_elliptical_gaussian_covariance_and_principal_axes():
    r, z, rm, zm = _grid(301, 0.6)
    angle, sr, sz = 0.4, 0.06, 0.12
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    sigma = rot @ np.diag([sr**2, sz**2]) @ rot.T
    d = np.stack((rm - 1.0, zm + 0.02), axis=-1)
    j = 1e6*np.exp(-0.5*np.einsum("...i,ij,...j->...", d, np.linalg.inv(sigma), d))
    cov = current_covariance(j, r, z)
    np.testing.assert_allclose(cov, sigma, rtol=1e-4, atol=1e-8)
    values, vectors = np.linalg.eigh(cov)
    np.testing.assert_allclose(np.sqrt(values), [sr, sz], rtol=1e-4)
    assert abs(abs(vectors[:, 1] @ rot[:, 1]) - 1) < 1e-6


def test_central_moments_are_translation_invariant_and_raw_ones_are_not():
    r, z, rm, zm = _grid(241, 0.6)

    def blob(r0, z0):
        return np.exp(-((rm - r0)**2/0.01 + (zm - z0)**2/0.02))*(1 + 3*(rm - r0))

    a, b = blob(1.0, 0.0), blob(1.05, 0.05)
    for p, q in ((2, 0), (1, 1), (3, 0), (1, 2), (0, 4)):
        assert current_moment(b, r, z, p, q) == pytest.approx(current_moment(a, r, z, p, q), rel=1e-4, abs=1e-12)
    raw_a = current_moment(a, r, z, 1, 0, center=(0.0, 0.0))
    raw_b = current_moment(b, r, z, 1, 0, center=(0.0, 0.0))
    assert raw_b - raw_a == pytest.approx(0.05, rel=1e-6)         # raw first moment = centroid
    unnormalized = current_moment(a, r, z, 2, 0, normalize=False)
    assert unnormalized == pytest.approx(current_moment(a, r, z, 2, 0)*current_centroid(a, r, z)[0])


def test_skewness_changes_sign_under_a_vertical_mirror_and_vanishes_with_symmetry():
    r, z, rm, zm = _grid(241, 0.6)
    skewed = np.exp(-((rm - 1)**2 + zm**2)/0.02)*(1 + 4*zm)
    mirrored = skewed[:, ::-1]
    assert current_moment(skewed, r, z, 0, 3) == pytest.approx(-current_moment(mirrored, r, z, 0, 3))
    assert abs(current_moment(skewed, r, z, 0, 3)) > 1e-5
    symmetric = np.exp(-((rm - 1)**2 + zm**2)/0.02)*(1 + 4*(rm - 1))
    for p, q in ((0, 1), (1, 1), (0, 3), (2, 1)):
        assert abs(current_moment(symmetric, r, z, p, q)) < 1e-12


def test_single_filament_is_the_lowest_order_limit():
    r, z, _, _ = _grid(101)
    j = np.zeros((r.size, z.size)); j[60, 45] = 1e9
    total, rc, zc = current_centroid(j, r, z)
    assert (rc, zc) == (r[60], z[45]) and total > 0
    for p, q in ((2, 0), (1, 1), (0, 2), (3, 0), (0, 3)):
        assert current_moment(j, r, z, p, q) == 0.0


def test_fractional_cells_converge_with_resolution_and_beat_a_mask():
    a = 0.3
    errors, mask_errors = [], []
    for n in (51, 101, 201):
        r, z, rm, zm = _grid(n)
        j = np.ones((n, n))
        w = _disk_weights(r, z, 1.0, 0.0, a)
        mask = ((rm - 1)**2 + zm**2 <= a**2).astype(float)
        errors.append(abs(current_moment(j, r, z, 2, 0, weights=w) - a**2/4))
        mask_errors.append(abs(current_moment(j, r, z, 2, 0, weights=mask) - a**2/4))
    assert errors[2] < errors[0] and errors[2] < 3e-5
    assert sum(errors) < sum(mask_errors)


def test_bad_inputs_are_refused():
    r, z, _, _ = _grid(21)
    with pytest.raises(ValueError, match="shape"):
        current_centroid(np.ones((20, 21)), r, z)
    with pytest.raises(ValueError, match="zero"):
        current_centroid(np.zeros((21, 21)), r, z)
    with pytest.raises(ValueError, match="non-negative"):
        current_moment(np.ones((21, 21)), r, z, -1, 0)


@pytest.mark.parametrize("convention", [11, 2, 1, 17])
def test_equilibrium_current_matches_the_recorded_ip_in_every_cocos(convention):
    eq = solovev_example("limited", convention=convention)
    moments = derive_current_moments(eq)
    assert moments.total_current == pytest.approx(eq.ip, rel=2e-3)
    assert moments.current_density_source == "grad_shafranov_flux_functions"
    assert abs(moments.central_moments[(0, 3)]) < 1e-12        # up-down symmetric
    assert abs(moments.central_moments[(1, 1)]) < 1e-12
    reference = derive_current_moments(solovev_example("limited"))
    for key, value in reference.central_moments.items():
        assert moments.central_moments[key] == pytest.approx(value, rel=1e-9, abs=1e-15)


def test_single_null_current_is_up_down_asymmetric():
    moments = derive_current_moments(solovev_example("single_null"), max_order=3)
    assert set(moments.central_moments) == {(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)}
    assert abs(moments.central_moments[(0, 3)]) > 1e-5
    assert moments.covariance.shape == (2, 2)


def test_supplied_current_density_is_used_and_labelled():
    eq = solovev_example("limited")
    derived = derive_current_moments(eq)
    from vaft.process._equilibrium_moments import _flux_function_current

    supplied = derive_current_moments(eq, j_tor=2*_flux_function_current(eq))
    assert supplied.current_density_source == "supplied"
    assert supplied.total_current == pytest.approx(2*derived.total_current)
    assert supplied.central_moments[(2, 0)] == pytest.approx(derived.central_moments[(2, 0)])


def test_a_record_without_a_convention_needs_an_explicit_current():
    eq = dataclasses.replace(solovev_example("limited"), convention=None)
    with pytest.raises(ValueError, match="COCOS"):
        derive_current_moments(eq)
    with pytest.raises(ValueError, match="max_order"):
        derive_current_moments(solovev_example("limited"), max_order=1)


def test_an_ambiguous_but_agreeing_convention_is_enough_and_a_wrong_one_is_warned_about():
    from vaft.data.resources import sample_geqdsk
    from vaft.process.equilibrium import as_equilibrium

    gfile = sample_geqdsk()
    native = as_equilibrium(gfile)                      # COCOS 1 or 2: same sigma_Bp and flux unit
    assert native.convention.cocos is None
    assert derive_current_moments(native).total_current == pytest.approx(native.ip, rel=2e-3)
    with pytest.warns(UserWarning, match="factor 6.28"):
        derive_current_moments(as_equilibrium(gfile, convention=11))
