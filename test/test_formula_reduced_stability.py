"""Local interchange criteria and the reduced internal kink (#1635): values, limits and identities."""

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.formula.stability import (
    bussac_internal_kink_energy,
    bussac_poloidal_beta,
    ggj_ideal_interchange_index,
    ggj_resistive_interchange_index,
    magnetic_well_from_specific_volume,
    mercier_criterion_circular,
    suydam_criterion,
)


def test_suydam_value_and_the_shearless_limit():
    r, B, q, dq, dp = 0.2, 0.5, 1.5, 4.0, -3.0e3
    expected = r * B**2 / (8 * MU0) * (dq / q) ** 2 + dp
    assert suydam_criterion(r, B, q, dq, dp) == pytest.approx(expected, rel=1e-12)
    # no shear: any outward pressure fall violates it, a flat pressure is marginal
    assert suydam_criterion(r, B, q, 0.0, -1.0) < 0.0
    assert suydam_criterion(r, B, q, 0.0, 0.0) == 0.0
    # shear stabilises: enough of it restores the criterion
    assert suydam_criterion(r, B, q, 40.0, dp) > 0.0
    # q's sign convention does not matter, only q'/q
    assert suydam_criterion(r, B, -q, -dq, dp) == pytest.approx(expected)


def test_suydam_is_vectorised():
    r = np.array([0.1, 0.2, 0.3])
    out = suydam_criterion(r, 0.5, 1.2, 2.0, -1.0e3)
    assert out.shape == (3,)
    assert out[2] > out[0]  # the shear term grows with r


def test_mercier_is_suydam_with_the_toroidal_average_curvature():
    rng = np.random.default_rng(1635)
    r, B, q, dq, dp = rng.uniform(0.05, 0.5, 5), rng.uniform(0.1, 1, 5), rng.uniform(0.5, 4, 5), \
        rng.normal(0, 5, 5), rng.normal(0, 1e4, 5)
    np.testing.assert_allclose(mercier_criterion_circular(r, B, q, dq, dp),
                               suydam_criterion(r, B, q, dq, dp) - dp * q**2, rtol=1e-12)


def test_mercier_toroidal_curvature_stabilises_above_q_one():
    # no shear, pressure falling outward: unstable below q = 1, stable above it, marginal at q = 1
    assert mercier_criterion_circular(0.2, 0.5, 0.8, 0.0, -1.0e3) < 0.0
    assert mercier_criterion_circular(0.2, 0.5, 2.0, 0.0, -1.0e3) > 0.0
    assert mercier_criterion_circular(0.2, 0.5, 1.0, 0.0, -1.0e3) == pytest.approx(0.0, abs=1e-9)


def test_ggj_indices_obey_the_dr_di_identity():
    rng = np.random.default_rng(7)
    E, F, H = rng.normal(size=(3, 50))
    d_i = ggj_ideal_interchange_index(E, F, H)
    d_r = ggj_resistive_interchange_index(E, F, H)
    np.testing.assert_allclose(d_r, d_i + (H - 0.5) ** 2, rtol=1e-12, atol=1e-12)
    assert np.all(d_r >= d_i - 1e-15)
    # a surface can be Mercier stable yet resistive-interchange unstable
    assert ggj_ideal_interchange_index(0.1, 0.0, 0.0) < 0.0
    assert ggj_resistive_interchange_index(0.1, 0.0, 0.0) > 0.0


def test_magnetic_well_is_zero_for_a_straight_cylinder_and_signed_otherwise():
    flux = np.linspace(0.01, 1.0, 101)
    assert np.allclose(magnetic_well_from_specific_volume(flux, np.full_like(flux, 2.0)), 0.0)
    # V' = V0 (1 - c Phi): V'' < 0, a well, W = c Phi / (1 - c Phi); a linear V' is differentiated exactly
    c = 0.3
    w = magnetic_well_from_specific_volume(flux, 5.0 * (1.0 - c * flux))
    np.testing.assert_allclose(w, c * flux / (1.0 - c * flux), rtol=1e-10)
    assert np.all(w > 0.0)
    hill = magnetic_well_from_specific_volume(flux, 5.0 * (1.0 + c * flux))
    assert np.all(hill < 0.0)


def test_bussac_poloidal_beta_from_a_parabolic_pressure():
    p0, a, r1, B_theta = 1.0e4, 1.0, 0.4, 0.2
    r = np.linspace(0.0, r1, 2001)
    p = p0 * (1 - r**2 / a**2)
    mean = 2.0 / r1**2 * np.trapezoid(p * r, r)
    # analytic: <p>_1 - p(r1) = p0 r1^2 / (2 a^2)
    assert mean - p[-1] == pytest.approx(p0 * r1**2 / (2 * a**2), rel=1e-6)
    beta = bussac_poloidal_beta(mean, p[-1], B_theta)
    assert beta == pytest.approx(2 * MU0 * p0 * r1**2 / (2 * a**2) / B_theta**2, rel=1e-6)


def test_bussac_energy_changes_sign_at_the_critical_beta():
    crit = np.sqrt(13.0 / 144.0)
    assert crit == pytest.approx(0.3005, abs=1e-4)
    assert bussac_internal_kink_energy(0.9 * crit, 0.9) > 0.0
    assert bussac_internal_kink_energy(1.1 * crit, 0.9) < 0.0
    assert bussac_internal_kink_energy(crit, 0.9) == pytest.approx(0.0, abs=1e-15)
    # the drive scales with 1 - q0
    assert bussac_internal_kink_energy(0.1, 0.8) == pytest.approx(2 * bussac_internal_kink_energy(0.1, 0.9))


@pytest.mark.parametrize("call", [
    lambda: suydam_criterion(0.0, 1.0, 1.0, 1.0, 1.0),
    lambda: suydam_criterion(0.1, 1.0, 0.0, 1.0, 1.0),
    lambda: mercier_criterion_circular(-0.1, 1.0, 1.0, 1.0, 1.0),
    lambda: mercier_criterion_circular(0.1, np.nan, 1.0, 1.0, 1.0),
    lambda: ggj_ideal_interchange_index(np.inf, 0.0, 0.0),
    lambda: magnetic_well_from_specific_volume([0.0, 1.0], [1.0, 1.0]),
    lambda: magnetic_well_from_specific_volume([0.0, 2.0, 1.0], [1.0, 1.0, 1.0]),
    lambda: magnetic_well_from_specific_volume([0.0, 1.0, 2.0], [1.0, 0.0, 1.0]),
    lambda: bussac_poloidal_beta(1.0, 0.0, 0.0),
    lambda: bussac_internal_kink_energy(0.1, 1.0),
])
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
