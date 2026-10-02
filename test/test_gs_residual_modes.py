"""Mode-resolved Grad-Shafranov residual on nested flux surfaces (#948).

The projection shares its angle with the Fourier surface representation
(#945): projecting R or Z itself must give that surface's own R and Z
coefficients, which pins the angle, the phase and the normalization at once.
An exact Solov'ev equilibrium must have a residual at discretization level.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vaft.process.equilibrium import (
    as_equilibrium,
    fit_fourier_surface_sequence,
    grad_shafranov_residual_modes,
    solovev_example,
)

LEVELS = [0.2, 0.4, 0.6, 0.8]


@pytest.fixture(scope="module")
def single_null():
    return solovev_example("single_null", resolution=257)


@pytest.mark.parametrize("coordinate, family", [("r", ("r_cos", "r_sin")), ("z", ("z_cos", "z_sin"))])
def test_projecting_a_coordinate_returns_the_surface_harmonics(single_null, coordinate, family):
    eq = single_null
    rm, zm = np.meshgrid(eq.r, eq.z, indexing="ij")
    modes = grad_shafranov_residual_modes(eq, LEVELS, max_mode=6, residual_map=rm if coordinate == "r" else zm)
    surfaces = fit_fourier_surface_sequence(eq, LEVELS, modes=6)
    for row, fit in enumerate(surfaces.fits):
        np.testing.assert_allclose(modes.cos[row], getattr(fit.surface, family[0]), atol=2e-4)
        np.testing.assert_allclose(modes.sin[row], getattr(fit.surface, family[1]), atol=2e-4)


def test_constant_residual_is_pure_m0(single_null):
    shape = (single_null.r.size, single_null.z.size)
    modes = grad_shafranov_residual_modes(single_null, LEVELS, max_mode=5, residual_map=np.full(shape, 3.0))
    np.testing.assert_allclose(modes.cos[:, 0], 3.0, rtol=1e-12)
    assert np.max(np.abs(modes.cos[:, 1:])) < 1e-12 and np.max(np.abs(modes.sin)) < 1e-12
    np.testing.assert_allclose(modes.rms, 3.0, rtol=1e-12)


def test_exact_solovev_has_a_discretization_level_residual(single_null):
    modes = grad_shafranov_residual_modes(single_null, LEVELS, max_mode=4)
    assert modes.scale > 0
    assert np.max(modes.rms)/modes.scale < 1e-3
    assert np.max(np.abs(np.r_[modes.cos.ravel(), modes.sin.ravel()]))/modes.scale < 1e-3


def test_a_source_error_shows_up_as_the_expected_harmonics(single_null):
    """p' scaled by 1.05: the residual is -0.05 * (-mu0 R^2 p'), so m = 0 dominates and m = 1 follows R^2."""
    eq = single_null
    wrong = dataclasses.replace(eq, pprime=1.05*np.asarray(eq.pprime))
    modes = grad_shafranov_residual_modes(wrong, LEVELS, max_mode=4)
    exact = grad_shafranov_residual_modes(eq, LEVELS, max_mode=4)
    assert np.all(modes.rms > 50*exact.rms)
    assert np.all(np.abs(modes.cos[:, 0]) > np.abs(modes.cos[:, 2]))
    assert np.all(modes.amplitude(1) > modes.amplitude(3))


def test_symmetric_equilibrium_has_no_odd_asymmetric_residual():
    eq = solovev_example("limited", resolution=257)
    rm, zm = np.meshgrid(eq.r, eq.z, indexing="ij")
    modes = grad_shafranov_residual_modes(eq, LEVELS, max_mode=4, residual_map=rm**2 + zm**2)
    # Even in Z -> no sine terms, to the up-down symmetry of the traced contours.
    assert np.max(np.abs(modes.sin)) < 1e-5*np.max(np.abs(modes.cos[:, 0]))


def test_flux_unit_and_inputs_are_checked(single_null):
    from vaft.data.equilibrium import EquilibriumConvention

    unknown = dataclasses.replace(single_null, convention=EquilibriumConvention())
    with pytest.raises(ValueError, match="flux unit"):
        grad_shafranov_residual_modes(unknown, LEVELS)
    with pytest.raises(ValueError, match="grid shape"):
        grad_shafranov_residual_modes(single_null, LEVELS, residual_map=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        grad_shafranov_residual_modes(single_null, LEVELS, max_mode=-1)
    assert grad_shafranov_residual_modes(single_null, [0.5, 1.4], max_mode=2).skipped == (1.4,)


def test_packaged_reconstruction_residual_is_resolved_by_mode():
    from vaft.data.resources import sample_geqdsk

    modes = grad_shafranov_residual_modes(as_equilibrium(sample_geqdsk()), LEVELS, max_mode=4)
    assert modes.cos.shape == (4, 5) and np.all(np.isfinite(modes.rms))
    assert np.all(np.diff(modes.rms) > 0)                        # force balance degrades toward the edge
