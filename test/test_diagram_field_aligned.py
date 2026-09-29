"""Field-aligned coordinates part 2 (#1075): eigenmode, k_x(theta), basis, shear, flux tube."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _field_aligned as fa
from vaft.diagram._scene import Label
from vaft.formula.stability import (
    ballooning_radial_wavenumber,
    s_alpha_ballooning_eigenmode,
    s_alpha_ballooning_stable,
    s_alpha_curvature_drive,
)


@pytest.mark.parametrize("s, alpha", [(1.0, 0.3), (1.0, 0.7), (1.0, 1.2), (0.4, 0.5), (1.5, 2.0), (0.5, 2.5)])
def test_the_eigenmode_agrees_with_newcomb_and_localises_when_unstable(s, alpha):
    g, theta, F = s_alpha_ballooning_eigenmode(s, alpha)
    stable = s_alpha_ballooning_stable(s, alpha)
    assert (g < 0.0) == stable  # an independent integrator decides the same way
    np.testing.assert_allclose(F, F[::-1], atol=1e-6)  # even
    tail = np.max(np.abs(F[np.abs(theta) > 4 * math.pi]))
    if not stable:
        assert F[len(F) // 2] == pytest.approx(1.0)  # peaked at the outboard midplane
        assert tail < 0.05  # it balloons: localised near theta = 0
    else:
        assert tail > 0.1  # the stable top eigenvector is continuum-like, it fills the interval


def test_the_radial_wavenumber_is_the_lambda_of_the_s_alpha_model():
    theta = np.linspace(-6.0, 6.0, 25)
    for s, alpha, t0 in ((1.0, 0.0, 0.0), (0.7, 0.5, 0.3)):
        lam = ballooning_radial_wavenumber(1.0, s, theta, alpha, t0)
        # s_alpha_curvature_drive's K = cos(theta) + Lambda sin(theta), with the same Lambda
        K = s_alpha_curvature_drive(theta, s, alpha, t0)
        np.testing.assert_allclose(K, np.cos(theta) + lam * np.sin(theta), atol=1e-12)
    # alpha = 0: the sheared slab, k_x = k_x0 + k_y s z with k_x0 = -k_y s theta0
    assert ballooning_radial_wavenumber(2.0, 0.8, 3.0, 0.0, 0.5) == pytest.approx(2.0 * 0.8 * 3.0 - 2.0 * 0.8 * 0.5)
    with pytest.raises(ValueError):
        ballooning_radial_wavenumber(np.inf, 1.0, 0.0)


def test_the_basis_has_b_along_the_lines_and_grad_alpha_across():
    model = vaft.diagram.field_aligned_basis(q=3.0).model
    b, g = np.asarray(model["B_direction"]), np.asarray(model["grad_alpha_direction"])
    assert np.dot(b, g) == pytest.approx(0.0, abs=1e-12)  # in (phi, theta) coordinates: grad alpha . B = 0
    assert b[1] / b[0] == pytest.approx(1.0 / 3.0)  # d theta / d phi = 1/q along the line


def test_shear_rotates_the_phase_fronts():
    model = vaft.diagram.magnetic_shear_field_aligned(shear=1.5).model
    np.testing.assert_allclose(model["k_x"], 1.5 * np.asarray(model["theta"]))
    assert model["k_x"][2] == 0.0  # untilted at theta = 0


def test_the_figure_contrasts_an_unstable_and_a_stable_surface():
    chart = vaft.diagram.ballooning_eigenfunction().model
    assert chart.parameters["growth_rate_squared_unstable"] > 0 > chart.parameters["growth_rate_squared_stable"]
    assert not s_alpha_ballooning_stable(*fa.UNSTABLE) and s_alpha_ballooning_stable(*fa.STABLE)


def test_flux_tube_steps_and_labels_off():
    assert vaft.diagram.flux_tube_patch().model["steps"][-1] == "flux_tube"
    for build in (vaft.diagram.field_aligned_basis, vaft.diagram.flux_tube_patch,
                  vaft.diagram.magnetic_shear_field_aligned, vaft.diagram.ballooning_eigenfunction):
        assert not [i for i in build(labels=False).scene.items
                    if isinstance(i, Label) and i.role not in ("axes", "ticks")]
