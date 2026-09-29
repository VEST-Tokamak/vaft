"""SOL blob formulas (#1211): reference scales, the two closure limits, the regime scalings."""

import numpy as np
import pytest

from vaft.formula.blob import (
    blob_collisionality,
    blob_reference_size,
    blob_reference_velocity,
    blob_regime_velocities,
    inertial_blob_velocity,
    interpolated_blob_velocity,
    sheath_connected_blob_velocity,
)
from vaft.formula.constants import ME, QE

MD = 3.3435837724e-27


def _nstx():
    # Myra et al., Phys. Plasmas 13 (2006) 092509, p. -2: n_e = 3e12 cm^-3, T_e = 20 eV, a_b = 2 cm, L = 3.5 R,
    # R = 150 cm, B = 2.5 kG, deuterium
    Te, B, R = 20.0, 0.25, 1.5
    cs = np.sqrt(Te * QE / MD)
    rho = cs / (QE * B / MD)
    return cs, rho, 3.5 * R, R, B


def test_nstx_reference_scales_match_myra_2006():
    cs, rho, L, R, B = _nstx()
    ds = blob_reference_size(rho, L, R)
    vs = blob_reference_velocity(cs, ds, R)
    # Myra's cgs forms, Eqs. (2) and (3): a_hat = 0.018 a_b B^{4/5} R^{1/5} / (L^{2/5} T^{2/5}), v_* = 5.1e6 ...
    a_hat_myra = 0.018 * 2.0 * 2500.0**0.8 * 150.0**0.2 / (525.0**0.4 * 20.0**0.4)
    v_star_myra = 5.1e6 * 525.0**0.2 * 20.0**0.7 / (2500.0**0.4 * 150.0**0.6) / 100.0  # cm/s -> m/s
    assert 0.02 / ds == pytest.approx(a_hat_myra, rel=0.05)
    assert vs == pytest.approx(v_star_myra, rel=0.05)
    # the collisionality is its definition; Myra's numeric Eq. (1) folds in his own nu_ei and ln Lambda
    assert blob_collisionality(2.0e6, L, QE * B / ME, rho) == pytest.approx(2.0e6 * L / (QE * B / ME * rho))


def test_the_two_limits_meet_at_the_reference_scales():
    cs, rho, L, R, _ = _nstx()
    ds = blob_reference_size(rho, L, R)
    vs = blob_reference_velocity(cs, ds, R)
    assert sheath_connected_blob_velocity(cs, rho, ds, L, R) == pytest.approx(vs)
    assert inertial_blob_velocity(cs, ds, R) == pytest.approx(vs)
    # and scale as delta^-2 and delta^1/2 away from it
    assert sheath_connected_blob_velocity(cs, rho, 2 * ds, L, R) == pytest.approx(vs / 4)
    assert inertial_blob_velocity(cs, 4 * ds, R) == pytest.approx(2 * vs)
    assert inertial_blob_velocity(cs, ds, R, relative_amplitude=0.25) == pytest.approx(0.5 * vs)


def test_the_interpolation_reduces_to_both_limits():
    d = np.array([1e-3, 1e3])
    for f in (1.0, 0.3):
        v = interpolated_blob_velocity(d, relative_amplitude=f)
        assert v[0] == pytest.approx(np.sqrt(f) * np.sqrt(d[0]), rel=1e-3)
        assert v[1] == pytest.approx(f / d[1] ** 2, rel=1e-3)
    assert interpolated_blob_velocity(1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        interpolated_blob_velocity(1.0, relative_amplitude=1.5)


def test_regime_boundaries_are_where_scalings_meet():
    eps = 0.1
    theta = np.array([0.03, 1.0, 30.0])
    d = theta**0.4
    # Lambda = Theta: RB = RX; Lambda = eps Theta: RX = C_i; Lambda = 1: RX = C_s; Theta = 1/eps: C_i = C_s
    v = blob_regime_velocities(d, theta, eps)
    np.testing.assert_allclose(v.resistive_ballooning, v.resistive_x_point)
    v = blob_regime_velocities(d, eps * theta, eps)
    np.testing.assert_allclose(v.resistive_x_point, v.connected_ideal_interchange)
    v = blob_regime_velocities(d, 1.0, eps)
    np.testing.assert_allclose(v.resistive_x_point, v.sheath_connected)
    v = blob_regime_velocities((1 / eps) ** 0.4, 1.0, eps)
    assert v.connected_ideal_interchange == pytest.approx(v.sheath_connected)
    with pytest.raises(ValueError):
        blob_regime_velocities(1.0, 1.0, 1.5)


def test_inputs_are_refused():
    for bad in ((0.0, 1.0, 1.0), (1.0, -1.0, 1.0)):
        with pytest.raises(ValueError):
            blob_reference_size(*bad)
    with pytest.raises(ValueError):
        blob_collisionality(-1.0, 1.0, 1.0, 1.0)
