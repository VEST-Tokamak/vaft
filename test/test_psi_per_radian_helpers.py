"""The three flux helpers say which flux family they take (#354).

q_from_phi, surface_poloidal_flux_from_psi_boundary and
loop_voltage_from_total_flux used to hard-code one family each -- full weber
for the first, per radian for the other two -- with no way to say otherwise.
Each now takes ``psi_per_radian``; the defaults are unchanged.  Pinned on the
packaged 39915 equilibrium, which carries the same flux in both families (the
g-file's per-radian SIMAG/SIBRY and the ODS's full-weber profiles_1d.psi),
against the q the equilibrium itself stores.
"""

import logging

import numpy as np
import pytest

from vaft.data.resources import sample_geqdsk
from vaft.formula.equilibrium import (
    loop_voltage_from_total_flux,
    q_from_phi,
    surface_poloidal_flux_from_psi_boundary,
    toroidal_electric_field,
)


@pytest.fixture(scope="module")
def equilibrium():
    logging.disable(logging.WARNING)
    try:
        g = sample_geqdsk("efit/g039915.00319")
        ts = g.to_omas()["equilibrium.time_slice.0"]
    finally:
        logging.disable(logging.NOTSET)
    psi = np.asarray(ts["profiles_1d.psi"], float)
    psi_radian = np.linspace(float(g["SIMAG"]), float(g["SIBRY"]), psi.size)
    return psi, psi_radian, np.asarray(ts["profiles_1d.phi"], float), np.asarray(ts["profiles_1d.q"], float)


def test_q_from_phi_reproduces_the_stored_q_on_either_flux_family(equilibrium):
    psi, psi_radian, phi, q = equilibrium
    assert (psi[-1] - psi[0]) / (psi_radian[-1] - psi_radian[0]) == pytest.approx(2 * np.pi)
    interior = slice(3, -3)
    from_weber = q_from_phi(psi, phi)
    np.testing.assert_allclose(np.abs(from_weber[interior]), np.abs(q[interior]), rtol=5e-3)
    np.testing.assert_allclose(
        q_from_phi(psi_radian, phi, psi_per_radian=True), from_weber, rtol=1e-12
    )
    # The default reads the flux as full weber, so a per-radian one is 2 pi off.
    np.testing.assert_allclose(q_from_phi(psi_radian, phi), 2 * np.pi * from_weber, rtol=1e-12)


def test_the_surface_flux_is_the_same_weber_from_either_family(equilibrium):
    psi, psi_radian, _, _ = equilibrium
    assert surface_poloidal_flux_from_psi_boundary(psi_radian[-1]) == pytest.approx(psi[-1], rel=1e-12)
    assert surface_poloidal_flux_from_psi_boundary(psi[-1], psi_per_radian=False) == psi[-1]


def test_the_loop_voltage_is_the_same_from_either_family():
    t = np.linspace(0.0, 1e-3, 11)
    psi_weber = 0.02 + 3.0 * t
    np.testing.assert_allclose(
        loop_voltage_from_total_flux(t, psi_weber / (2 * np.pi)),
        loop_voltage_from_total_flux(t, psi_weber, psi_per_radian=False),
        rtol=1e-12,
    )
    np.testing.assert_allclose(loop_voltage_from_total_flux(t, psi_weber, psi_per_radian=False), 3.0)


def test_the_loop_voltage_is_minus_the_lenz_voltage_of_the_toroidal_field():
    # Stated in the docstring: on a full-weber flux the two kernels differ by
    # sign (Ejima's +dpsi/dt against Lenz's -dpsi/dt), not by 2 pi.
    t = np.linspace(0.0, 1e-3, 11)
    psi = 0.02 + 3.0 * t
    radius = 0.4
    lenz = 2 * np.pi * radius * toroidal_electric_field(radius, np.gradient(psi, t))
    np.testing.assert_allclose(
        loop_voltage_from_total_flux(t, psi, psi_per_radian=False), -lenz, rtol=1e-12
    )


def test_a_greens_function_flux_gives_the_greens_field_with_the_documented_cocos():
    # vaft.formula.green's Conventions section: cocos=13 reproduces the
    # Green's-function B_z; the default is 2 pi too large, cocos=11 flips it.
    from vaft.formula.equilibrium import vertical_magnetic_field_from_psi
    from vaft.formula.green import green_br_bz_exact, green_psi_exact

    r = np.linspace(0.2, 0.8, 601)
    z = np.zeros_like(r)
    psi = np.asarray(green_psi_exact(r, z, 0.5, 0.1), float)
    _, b_z = green_br_bz_exact(r, z, 0.5, 0.1)
    interior = slice(5, -5)

    def ratio(cocos):
        return np.median(vertical_magnetic_field_from_psi(psi, r, z, cocos=cocos)[interior] / b_z[interior])

    assert ratio(13) == pytest.approx(1.0, rel=1e-3)
    assert ratio(11) == pytest.approx(-1.0, rel=1e-3)
    assert ratio(None) == pytest.approx(2 * np.pi, rel=1e-3)
