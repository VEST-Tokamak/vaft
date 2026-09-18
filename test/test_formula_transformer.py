"""Romero's exact plasma-transformer identities (#781, child A).

Every identity here is exact, so the tests pin them against each other and
against the one analytic equilibrium that has a closed form -- a uniform
current channel -- rather than restating the algebra.
"""

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.formula.equilibrium import (
    inductive_voltage_from_dW_magdt_I_p,
    li_3_from_Bp2_volume_integral,
)
from vaft.formula.transformer import (
    current_weighted_flux_from_psi_j_dS,
    equilibrium_surface_voltage_from_I_p_dL_i_V_R,
    equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R,
    internal_inductance_from_psi_C_psi_B_I_p,
    internal_inductance_rate_from_I_p_V_R_V_C,
    plasma_current_rate_from_L_i_V_B_V_C_V_R,
    resistive_voltage_from_R_p_I_p_I_ni,
)


def _uniform_channel(ip, r0=0.4, a=0.25, psi_b=-0.013, rings=4000):
    """Large-aspect-ratio uniform current: psi falls by mu0 R0 I / 2 to the edge.

    B_theta = mu0 I r / (2 pi a^2), so the full flux between the axis and r is
    2 pi R0 int B_theta dr = mu0 R0 I r^2 / (2 a^2), decreasing outward for
    positive I -- Romero's sign.
    """
    edges = np.linspace(0.0, a, rings + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    ds = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    psi = psi_b + 0.5 * MU0 * r0 * ip * (1.0 - r**2 / a**2)
    j = np.full_like(r, ip / (np.pi * a**2))
    return psi, j, ds, psi_b, r0


@pytest.mark.parametrize("ip", [1.0e5, -2.0e5])
def test_a_uniform_channel_has_li_3_one_half_through_its_fluxes(ip):
    psi, j, ds, psi_b, r0 = _uniform_channel(ip)
    psi_c = current_weighted_flux_from_psi_j_dS(psi, j, ds)
    l_i = internal_inductance_from_psi_C_psi_B_I_p(psi_c, psi_b, ip)
    assert l_i == pytest.approx(MU0 * r0 / 4.0, rel=1e-6)
    # The same L_i as the field-energy route: int B_p^2 dV = mu0^2 I^2 R0 / 4.
    li_3 = li_3_from_Bp2_volume_integral(MU0**2 * ip**2 * r0 / 4.0, ip, r0)
    assert 2.0 * l_i / (MU0 * r0) == pytest.approx(li_3, rel=1e-6)


def test_an_inverted_flux_convention_is_refused_not_returned():
    psi, j, ds, psi_b, _ = _uniform_channel(1.0e5)
    psi_c = current_weighted_flux_from_psi_j_dS(-psi, j, ds)
    with pytest.raises(ValueError, match="Romero"):
        internal_inductance_from_psi_C_psi_B_I_p(psi_c, -psi_b, 1.0e5)


def test_cells_outside_the_plasma_are_excluded_by_zero_area():
    psi, j, ds, _, _ = _uniform_channel(1.0e5)
    inside = current_weighted_flux_from_psi_j_dS(psi, j, ds)
    padded = current_weighted_flux_from_psi_j_dS(
        np.r_[psi, 1.0], np.r_[j, 5.0e6], np.r_[ds, 0.0]
    )
    assert padded == inside


# A self-consistent history: arbitrary psi_C(t), psi_B(t), I_p(t) in Romero's
# convention define L_i, V_C and V_B; eq. (39) then fixes V_R, and eq. (40)
# must return the plasma current's own derivative.
def psi_c(t):
    return 0.02 + 0.03 * t - 0.01 * t**2


def psi_b(t):
    return 0.01 + 0.005 * np.sin(3.0 * t)


def i_p(t):
    return 1.0e5 * (1.0 + 0.4 * t)


def _d(f, t, h=1e-5):
    return (f(t + h) - f(t - h)) / (2.0 * h)


@pytest.mark.parametrize("t", [0.1, 0.35, 0.8])
def test_the_two_rate_equations_and_the_flux_definitions_close(t):
    def l_i(tt):
        return internal_inductance_from_psi_C_psi_B_I_p(psi_c(tt), psi_b(tt), i_p(tt))

    v_c, v_b = -_d(psi_c, t), -_d(psi_b, t)
    v_r = v_c + 0.5 * i_p(t) * _d(l_i, t)  # eq. (39) solved for V_R
    assert internal_inductance_rate_from_I_p_V_R_V_C(i_p(t), v_r, v_c) == pytest.approx(
        _d(l_i, t), rel=1e-8
    )
    assert plasma_current_rate_from_L_i_V_B_V_C_V_R(l_i(t), v_b, v_c, v_r) == pytest.approx(
        _d(i_p, t), rel=1e-6
    )


def test_the_energy_form_of_the_inductive_voltage_is_exact_while_l_i_changes():
    # equilibrium's (1/I) dW/dt with W = L_i I^2 / 2 is Romero's V_ind exactly,
    # with the half on dL_i -- not d(L_i I)/dt, which differs by I dL_i / 2.
    t = 0.35

    def l_i(tt):
        return internal_inductance_from_psi_C_psi_B_I_p(psi_c(tt), psi_b(tt), i_p(tt))

    def energy(tt):
        return 0.5 * l_i(tt) * i_p(tt) ** 2

    v_ind = inductive_voltage_from_dW_magdt_I_p(_d(energy, t), i_p(t))
    exact = l_i(t) * _d(i_p, t) + 0.5 * i_p(t) * _d(l_i, t)
    assert v_ind == pytest.approx(exact, rel=1e-8)
    circuit = _d(lambda tt: l_i(tt) * i_p(tt), t)
    assert circuit - v_ind == pytest.approx(0.5 * i_p(t) * _d(l_i, t), rel=1e-6)

def test_the_two_v_c_routes_invert_the_two_rate_equations():
    l_i, ip, v_b, v_r, v_c = 3.1e-7, 1.2e5, 1.7, 0.9, 1.3
    dl = internal_inductance_rate_from_I_p_V_R_V_C(ip, v_r, v_c)
    di = plasma_current_rate_from_L_i_V_B_V_C_V_R(l_i, v_b, v_c, v_r)
    assert equilibrium_surface_voltage_from_I_p_dL_i_V_R(ip, dl, v_r) == pytest.approx(v_c, rel=1e-14)
    assert equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R(l_i, di, v_b, v_r) == pytest.approx(
        v_c, rel=1e-14
    )
    # Together they are the internal-energy balance, whatever V_C is.
    assert l_i * di + 0.5 * ip * dl == pytest.approx(v_b - v_r, rel=1e-14)


def test_a_flat_loop_voltage_profile_is_the_steady_state():
    assert internal_inductance_rate_from_I_p_V_R_V_C(1.0e5, 0.7, 0.7) == 0.0
    assert plasma_current_rate_from_L_i_V_B_V_C_V_R(3.0e-7, 0.7, 0.7, 0.7) == 0.0


def test_the_resistive_voltage_needs_an_explicit_non_inductive_current():
    assert resistive_voltage_from_R_p_I_p_I_ni(2.0e-3, 1.0e5, 2.0e4) == pytest.approx(160.0)
    with pytest.raises(TypeError):
        resistive_voltage_from_R_p_I_p_I_ni(2.0e-3, 1.0e5)  # noqa: the point of the test


@pytest.mark.parametrize(
    "call",
    [
        lambda: resistive_voltage_from_R_p_I_p_I_ni(-1.0, 1.0, 0.0),
        lambda: internal_inductance_rate_from_I_p_V_R_V_C(0.0, 1.0, 1.0),
        lambda: plasma_current_rate_from_L_i_V_B_V_C_V_R(0.0, 1.0, 1.0, 1.0),
        lambda: equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R(-1e-7, 1.0, 1.0, 1.0),
        lambda: equilibrium_surface_voltage_from_I_p_dL_i_V_R(np.nan, 1.0, 1.0),
        lambda: current_weighted_flux_from_psi_j_dS([1.0, 2.0], [1.0, -1.0], [1.0, 1.0]),
        lambda: current_weighted_flux_from_psi_j_dS([1.0], [1.0], [-1.0]),
        lambda: current_weighted_flux_from_psi_j_dS([1.0, 2.0], [1.0, 1.0, 1.0], 1.0),
        lambda: internal_inductance_from_psi_C_psi_B_I_p(1.0, 0.0, 0.0),
    ],
    ids=[
        "negative_resistance", "zero_current_rate", "zero_L_i", "negative_L_i",
        "nan_current", "no_net_current", "negative_area", "no_broadcast", "zero_I_p",
    ],
)
def test_a_non_physical_input_raises(call):
    with pytest.raises(ValueError):
        call()
