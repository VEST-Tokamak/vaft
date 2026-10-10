"""Volume-based ballooning normalisations (#1637): exact reduction to the s-alpha (CHT) definitions."""

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.formula.equilibrium import ballooning_alpha_from_volume, shear_from_volume


def _circular(r, R0=1.5, B0=0.8, p0=2.0e4):
    """A large-aspect-ratio circular equilibrium, analytically, with psi the poloidal flux per radian."""
    q = 1.0 + 2.0 * r**2
    dq_dr = 4.0 * r
    dp_dr = -4.0 * p0 * r * (1.0 - r**2)
    dpsi_dr = r * B0 / q                  # R0 B_p with B_p = r B0 / (q R0)
    V = 2.0 * np.pi**2 * R0 * r**2
    dV_dr = 4.0 * np.pi**2 * R0 * r
    return dict(q=q, dq_dr=dq_dr, dp_dr=dp_dr, dpsi_dr=dpsi_dr, V=V, dV_dr=dV_dr, R0=R0, B0=B0)


def test_volume_shear_is_the_s_alpha_shear_for_circles():
    r = np.linspace(0.05, 0.6, 12)
    e = _circular(r)
    s = shear_from_volume(e["V"], e["dV_dr"] / e["dpsi_dr"], e["q"], e["dq_dr"] / e["dpsi_dr"])
    np.testing.assert_allclose(s, r * e["dq_dr"] / e["q"], rtol=1e-12)


def test_volume_shear_does_not_depend_on_the_flux_label_or_its_sign():
    r = np.linspace(0.05, 0.6, 12)
    e = _circular(r)
    base = shear_from_volume(e["V"], e["dV_dr"] / e["dpsi_dr"], e["q"], e["dq_dr"] / e["dpsi_dr"])
    for factor in (2 * np.pi, -1.0, 1.0 / 3.7):  # Wb instead of Wb/rad, a COCOS sign, psi_N
        dpsi = factor * e["dpsi_dr"]
        np.testing.assert_allclose(shear_from_volume(e["V"], e["dV_dr"] / dpsi, e["q"], e["dq_dr"] / dpsi), base,
                                   rtol=1e-12)


def test_volume_alpha_is_the_cht_alpha_for_circles():
    r = np.linspace(0.05, 0.6, 12)
    e = _circular(r)
    alpha = ballooning_alpha_from_volume(e["V"], e["dV_dr"] / e["dpsi_dr"], e["dp_dr"] / e["dpsi_dr"], e["R0"])
    cht = -2.0 * MU0 * e["R0"] * e["q"] ** 2 * e["dp_dr"] / e["B0"] ** 2
    np.testing.assert_allclose(alpha, cht, rtol=1e-12)
    assert np.all(alpha > 0.0)  # pressure falls outward


def test_volume_alpha_needs_flux_per_radian():
    r = np.linspace(0.05, 0.6, 12)
    e = _circular(r)
    per_radian = ballooning_alpha_from_volume(e["V"], e["dV_dr"] / e["dpsi_dr"], e["dp_dr"] / e["dpsi_dr"], e["R0"])
    full = 2.0 * np.pi * e["dpsi_dr"]  # the full poloidal flux in Wb
    in_weber = ballooning_alpha_from_volume(e["V"], e["dV_dr"] / full, e["dp_dr"] / full, e["R0"])
    np.testing.assert_allclose(in_weber * (2.0 * np.pi) ** 2, per_radian, rtol=1e-12)
    # the sign of psi cancels
    flipped = -e["dpsi_dr"]
    np.testing.assert_allclose(
        ballooning_alpha_from_volume(e["V"], e["dV_dr"] / flipped, e["dp_dr"] / flipped, e["R0"]), per_radian,
        rtol=1e-12)


@pytest.mark.parametrize("call", [
    lambda: shear_from_volume(0.0, 1.0, 1.0, 1.0),
    lambda: shear_from_volume(1.0, 0.0, 1.0, 1.0),
    lambda: shear_from_volume(1.0, 1.0, 0.0, 1.0),
    lambda: shear_from_volume(1.0, 1.0, 1.0, np.nan),
    lambda: ballooning_alpha_from_volume(-1.0, 1.0, 1.0, 1.0),
    lambda: ballooning_alpha_from_volume(1.0, 1.0, 1.0, 0.0),
    lambda: ballooning_alpha_from_volume(1.0, np.inf, 1.0, 1.0),
])
def test_bad_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
