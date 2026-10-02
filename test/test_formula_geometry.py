"""Geometric approximations (#1062): the slab, cylinder and local reduction agree where they should."""

import numpy as np
import pytest

from vaft.formula.geometry import (
    cylindrical_parallel_wavenumber,
    cylindrical_safety_factor_from_r_B,
    local_slab_from_cylinder,
    shear_length_from_q_R0_s,
    sheared_slab_field,
    sheared_slab_parallel_wavenumber,
    slab_parallel_wavenumber,
)
from vaft.formula.stability import helical_phase


def test_a_straight_slab_has_k_par_equal_to_k_z():
    B = [0.0, 0.0, 2.0]
    assert slab_parallel_wavenumber([3.0, -1.0, 0.5], B) == pytest.approx(0.5)
    assert slab_parallel_wavenumber([3.0, -1.0, 0.0], B) == pytest.approx(0.0)  # flute-like
    np.testing.assert_allclose(slab_parallel_wavenumber([[0, 0, 1], [0, 0, -2]], B), [1.0, -2.0])
    with pytest.raises(ValueError):
        slab_parallel_wavenumber([1, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        slab_parallel_wavenumber([1, 0], [0, 0, 1])


def test_the_sheared_slab_tilts_with_x_and_its_k_par_is_the_projection():
    L_s, B0, k_y = -4.0, 1.5, 7.0
    x = np.array([-0.02, 0.0, 0.01, 0.03])
    B = sheared_slab_field(x, B0, L_s)
    np.testing.assert_allclose(B[:, 1] / B[:, 2], x / L_s)
    np.testing.assert_allclose(np.linalg.norm(B, axis=-1), B0, rtol=1e-4)
    exact = slab_parallel_wavenumber(np.broadcast_to([0.0, k_y, 0.0], B.shape), B)
    np.testing.assert_allclose(sheared_slab_parallel_wavenumber(x, k_y, L_s), exact, rtol=1e-4, atol=1e-12)
    assert sheared_slab_parallel_wavenumber(0.0, k_y, L_s) == 0.0
    assert sheared_slab_parallel_wavenumber(0.0, k_y, L_s, k_z=0.3) == pytest.approx(0.3)
    for bad in (0.0, np.inf):
        with pytest.raises(ValueError):
            sheared_slab_field(0.0, B0, bad)
        with pytest.raises(ValueError):
            sheared_slab_parallel_wavenumber(0.0, k_y, bad)


def test_the_cylindrical_q_agrees_with_the_engineering_q_cyl():
    from vaft.formula.constants import MU0
    from vaft.formula.equilibrium import q_cyl_from_B_R_epsilon_kappa_I

    a, R0, B_z, I = 0.3, 1.5, 2.0, 2.0e5
    B_theta = MU0 * I / (2 * np.pi * a)  # Ampere's law at the edge of a round column
    q = cylindrical_safety_factor_from_r_B(a, B_theta, B_z, R0)
    assert q == pytest.approx(q_cyl_from_B_R_epsilon_kappa_I(B_z, R0, a / R0, 1.0, I), rel=1e-3)
    with pytest.raises(ValueError):
        cylindrical_safety_factor_from_r_B(a, 0.0, B_z, R0)


def test_the_shear_length_is_a_magnitude():
    assert shear_length_from_q_R0_s(-2.0, 1.5, 1.0) == pytest.approx(3.0)
    assert shear_length_from_q_R0_s(2.0, 1.5, -1.0) == pytest.approx(3.0)


@pytest.mark.parametrize("m, n", [(2, 1), (3, 2), (1, 1)])
def test_k_par_vanishes_exactly_on_the_rational_surface(m, n):
    R0 = 1.8
    assert cylindrical_parallel_wavenumber(m, n, m / n, R0) == 0.0
    # q rising outward: positive inside, negative outside
    assert cylindrical_parallel_wavenumber(m, n, 0.9 * m / n, R0) > 0
    assert cylindrical_parallel_wavenumber(m, n, 1.1 * m / n, R0) < 0
    for bad in ((0, n), (m, 0), (m, -1)):
        with pytest.raises(ValueError):
            cylindrical_parallel_wavenumber(*bad, 1.0, R0)


@pytest.mark.parametrize("m, n", [(2, 1), (3, 2), (4, 3)])
def test_the_local_slab_reproduces_the_cylinder_to_first_order(m, n):
    R0, q0, qa, a = 1.6, 1.0, 4.0, 0.5
    q = lambda r: q0 + (qa - q0) * (r / a) ** 2  # noqa: E731
    r_s = a * np.sqrt((m / n - q0) / (qa - q0))
    s_hat = r_s * (2 * (qa - q0) * r_s / a**2) / (m / n)
    k_y, L_s, k_z = local_slab_from_cylinder(m, n, r_s, R0, m / n, s_hat)
    assert k_y == pytest.approx(m / r_s) and k_z == 0.0  # field-aligned on the rational surface
    assert L_s < 0 and abs(L_s) == pytest.approx(shear_length_from_q_R0_s(m / n, R0, s_hat))
    # along the straightened torus the same harmonic has k = -n/R0: the phase is helical_phase's
    theta, phi = 0.4, 1.1
    assert k_y * (r_s * theta) - n / R0 * (R0 * phi) == pytest.approx(float(helical_phase(theta, phi, m, n)))
    # and k_par(x) agrees to first order near r_s -- the returned tuple feeds the slab directly
    for x in (1e-4 * a, -1e-4 * a, 1e-3 * a):
        cylinder = cylindrical_parallel_wavenumber(m, n, q(r_s + x), R0)
        slab = sheared_slab_parallel_wavenumber(x, *local_slab_from_cylinder(m, n, r_s, R0, m / n, s_hat))
        assert slab == pytest.approx(cylinder, rel=5 * abs(x) / r_s + 1e-9)


def test_off_the_rational_surface_k_z_is_the_cylinders_k_par():
    for q_0 in (1.7, 2.0, 2.4):
        k_y, L_s, k_z = local_slab_from_cylinder(2, 1, 0.3, 1.5, q_0, 0.8)
        assert k_z == pytest.approx(cylindrical_parallel_wavenumber(2, 1, q_0, 1.5))
        assert sheared_slab_parallel_wavenumber(0.0, k_y, L_s, k_z) == pytest.approx(k_z)


def test_local_slab_rejects_what_has_no_slab():
    with pytest.raises(ValueError):
        local_slab_from_cylinder(2, 1, 0.3, 1.5, 2.0, 0.0)
    with pytest.raises(ValueError):
        local_slab_from_cylinder(2, 1, -0.3, 1.5, 2.0, 1.0)
    with pytest.raises(ValueError):
        local_slab_from_cylinder(2, 1, 0.3, 1.5, -2.0, 1.0)
    with pytest.raises(ValueError):
        shear_length_from_q_R0_s(2.0, 1.5, 0.0)
