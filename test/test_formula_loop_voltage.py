"""Boundary and internal inductive voltage, split term by term (#782).

Romero's balance is V_B = -dpsi_B/dt with psi_B = L_e I_p + sum_j M_pj I_j on
one side and V_B = R_p I_p + (1/I_p) d(L_i I_p^2 / 2)/dt on the other.  What
is pinned: the four boundary terms are the product rule of psi_B, the two
internal terms are the energy derivative, and the lumped circuit of #967
closes the balance only with the one-half on dL_i/dt.
"""

import numpy as np
import pytest

from vaft.formula.startup import (
    boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j,
    internal_inductive_voltage_terms_from_L_i_I_p,
    plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p,
)


# Smooth, all-varying histories: plasma, its boundary, two coils and their
# coupling all change at once, so no term can hide behind another's zero.
def L_e(t):
    return 1.0e-6 * (1.0 + 0.3 * t + 0.2 * t**2)


def L_i(t):
    return 0.4e-6 * (1.0 + 0.5 * np.sin(t))


def I_p(t):
    return 1.0e5 * (0.2 + t - 0.1 * t**3)


def M(t):
    return np.array([-2.0e-6 * (1.0 + 0.1 * t), 0.7e-6 * np.cos(t)])


def I_j(t):
    return np.array([5.0e3 * (1.0 - 2.0 * t), -1.2e3 * (1.0 + t**2)])


def _d(f, t, h=1e-5):
    return (f(t + h) - f(t - h)) / (2.0 * h)


def _psi_b(t):
    return L_e(t) * I_p(t) + np.sum(M(t) * I_j(t))


T = 0.37


def _boundary_terms(t=T):
    return boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(
        L_e(t), I_p(t), _d(I_p, t), _d(L_e, t),
        M_pj_H=M(t), I_j_A=I_j(t), dI_j_dt_A_s=_d(I_j, t), dM_pj_dt_H_s=_d(M, t),
    )


def test_the_four_terms_are_the_product_rule_of_the_boundary_flux():
    assert sum(_boundary_terms()) == pytest.approx(-_d(_psi_b, T), rel=1e-8)


def test_each_term_is_the_one_its_name_says():
    t = T
    ramp, shape, drive, geometry = _boundary_terms(t)
    assert ramp == pytest.approx(-L_e(t) * _d(I_p, t), rel=1e-14)
    assert shape == pytest.approx(-I_p(t) * _d(L_e, t), rel=1e-14)
    assert drive == pytest.approx(-np.sum(M(t) * _d(I_j, t)), rel=1e-14)
    assert geometry == pytest.approx(-np.sum(I_j(t) * _d(M, t)), rel=1e-14)


def test_without_coils_only_the_plasma_terms_remain():
    ramp, shape, drive, geometry = boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(
        1.0e-6, 1.0e5, 2.0e6, 3.0e-6
    )
    assert (ramp, shape) == (pytest.approx(-2.0, rel=1e-14), pytest.approx(-0.3, rel=1e-14))
    assert (drive, geometry) == (0.0, 0.0)
    assert all(isinstance(term, float) for term in (ramp, shape, drive, geometry))


def test_a_coil_time_series_sums_over_the_last_axis():
    times = np.array([0.1, 0.2, 0.3])
    mutual = np.stack([M(t) for t in times])
    current = np.stack([I_j(t) for t in times])
    rate = np.stack([_d(I_j, t) for t in times])
    ramp, shape, drive, geometry = boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(
        L_e(times), I_p(times), _d(I_p, times), M_pj_H=mutual, I_j_A=current,
        dI_j_dt_A_s=rate,
    )
    assert drive.shape == geometry.shape == ramp.shape == (3,)
    np.testing.assert_allclose(drive, -np.sum(mutual * rate, axis=-1), rtol=1e-14)
    np.testing.assert_array_equal(geometry, 0.0)


def test_a_fixed_mutual_broadcasts_over_a_coil_current_waveform():
    # Fixed geometry, measured coil currents: M is per coil, I_j per time.
    times = np.array([0.1, 0.2, 0.3])
    mutual = M(0.0)
    rate = np.stack([_d(I_j, t) for t in times])
    _, _, drive, geometry = boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(
        1.0e-6, 1.0e5, 0.0, M_pj_H=mutual,
        I_j_A=np.stack([I_j(t) for t in times]), dI_j_dt_A_s=rate,
    )
    np.testing.assert_allclose(drive, -(rate @ mutual), rtol=1e-14)
    np.testing.assert_array_equal(geometry, np.zeros(3))


def test_the_internal_terms_are_the_energy_derivative():
    t = T
    ramp, profile = internal_inductive_voltage_terms_from_L_i_I_p(
        L_i(t), I_p(t), _d(I_p, t), _d(L_i, t)
    )

    def energy(tt):
        return 0.5 * L_i(tt) * I_p(tt) ** 2

    assert I_p(t) * (ramp + profile) == pytest.approx(_d(energy, t), rel=1e-8)
    assert profile == pytest.approx(0.5 * I_p(t) * _d(L_i, t), rel=1e-14)


def _romero_residual(dL_p_dt):
    # Drive the lumped circuit with the external terms, then compare the
    # boundary voltage it implies with the resistive plus internal side.
    t, r_p = T, 3.0e-3
    ip, dle, dli = I_p(t), _d(L_e, t), _d(L_i, t)
    _, _, drive, geometry = _boundary_terms(t)
    ip_dot = plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p(
        drive + geometry, r_p, ip, L_e(t) + L_i(t), dL_p_dt(dle, dli)
    )
    ramp, shape, _, _ = boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(
        L_e(t), ip, ip_dot, dle
    )
    v_b = ramp + shape + drive + geometry
    internal = sum(internal_inductive_voltage_terms_from_L_i_I_p(L_i(t), ip, ip_dot, dli))
    return v_b - (r_p * ip + internal), 0.5 * ip * dli


def test_the_lumped_circuit_closes_romero_with_half_the_internal_rate():
    # The voltages here are volts; the residual is round-off on them.
    residual, _ = _romero_residual(lambda dle, dli: dle + 0.5 * dli)
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_the_naive_inductance_rate_misses_by_the_internal_profile_term():
    # Passing dL_e + dL_i puts an extra I dL_i / 2 into the lumped balance,
    # and Romero's balance is off by exactly that.
    residual, overstatement = _romero_residual(lambda dle, dli: dle + dli)
    assert residual == pytest.approx(overstatement, rel=1e-9)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"M_pj_H": [1e-6], "I_j_A": [1.0]}, "together"),
        ({"dM_pj_dt_H_s": [0.0]}, "without the coils"),
        ({"M_pj_H": [1e-6, 2e-6], "I_j_A": [1.0, 2.0, 3.0], "dI_j_dt_A_s": [0.0]}, "broadcast"),
        ({"L_e_H": 0.0}, "L_e_H"),
        ({"I_p_A": np.nan}, "I_p_A"),
    ],
)
def test_the_boundary_split_refuses_an_inconsistent_input(kwargs, match):
    call = {"L_e_H": 1e-6, "I_p_A": 1e5, "dI_p_dt_A_s": 1e6}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        boundary_loop_voltage_terms_from_L_e_I_p_M_pj_I_j(**call)


@pytest.mark.parametrize("kwargs", [{"L_i_H": -1e-7}, {"dL_i_dt_H_s": np.inf}])
def test_the_internal_split_refuses_a_non_physical_input(kwargs):
    call = {"L_i_H": 1e-7, "I_p_A": 1e5, "dI_p_dt_A_s": 1e6}
    call.update(kwargs)
    with pytest.raises(ValueError):
        internal_inductive_voltage_terms_from_L_i_I_p(**call)
