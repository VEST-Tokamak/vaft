r"""The plasma as a transformer secondary: Romero's exact current and inductance balance.

Romero's reduction of Poynting's theorem to three voltages -- the boundary
loop voltage $V_B$, the resistive voltage $V_R$ and the voltage $V_C$ at the
current-weighted "equilibrium" flux surface -- closes the plasma current and
the internal inductance exactly, with no assumption about the current profile.
This module holds those identities and the definitions they rest on.  It owns
no time derivative and no integral over time: $V_C = -\dot\psi_C$, the
volt-second budget and any closure for $V_C$ belong to the process layer.

Notation
--------
I_p      : plasma current                                        [A]
L_i      : dimensional internal inductance, 2 W_int / I_p^2      [H]
psi      : poloidal flux through the toroidal circle, full       [Wb]
psi_B    : psi at the plasma boundary                            [Wb]
psi_C    : current-weighted flux average, int psi j dS / I_p     [Wb]
V_B      : boundary loop voltage, -d psi_B / dt                  [V]
V_C      : voltage at the equilibrium surface, -d psi_C / dt     [V]
V_R      : resistive voltage, R_p (I_p - I_ni)                   [V]
R_p      : plasma resistance                                     [Ohm]
I_ni     : non-inductively driven current                        [A]

Conventions
-----------
**Romero's signs throughout.**  $\psi$ is the full flux through the toroidal
circle through a point, in weber and not per radian, and every voltage is
$V = -\dot\psi$ (Lenz).  In this convention $\psi_C - \psi_B = L_i I_p$ has the
sign of $I_p$, and a positive $V_B$ sustains a positive $I_p$ -- the same sense
as the loop voltage of
:func:`vaft.formula.startup.plasma_current_derivative_lumped_from_V_loop_R_p_I_p_L_p`.
An equilibrium stored in another COCOS must be brought to it first; the one
function here that reads a flux refuses a result whose sign says it was not.

References
----------
.. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010) 115002.
"""

from __future__ import annotations

import numpy as np

from ._exports import public_names


def _maybe_scalar(value, *inputs):
    """Return a float when every input was scalar."""
    if all(np.isscalar(item) or np.ndim(item) == 0 for item in inputs):
        return float(value)
    return value


def _finite(name, value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return array


def _nonzero(name, value):
    array = _finite(name, value)
    if np.any(array == 0.0):
        raise ValueError(f"{name} must be non-zero; got {value!r}")
    return array


def resistive_voltage_from_R_p_I_p_I_ni(R_p_ohm, I_p_A, I_ni_A):
    r"""Resistive voltage of the plasma, net of non-inductive current drive.

    $$V_R = R_p\,(I_p - \hat I)$$

    Parameters
    ----------
    R_p_ohm : float or np.ndarray
        Plasma resistance, finite and non-negative [Ohm].
    I_p_A : float or np.ndarray
        Plasma current, finite [A].
    I_ni_A : float or np.ndarray
        Non-inductively driven current, finite; ``0`` asserts a purely
        Ohmic plasma [A].

    Returns
    -------
    float or np.ndarray
        Resistive voltage [V].

    Raises
    ------
    ValueError
        A non-finite input or a negative resistance.

    Convention
    ----------
    **``I_ni_A`` has no default on purpose.**  $\hat I = 0$ is a statement
    that nothing but the transformer drives current, true of a pure-Ohmic
    VEST discharge and false with EC, NBI or helicity injection; a caller
    must make it.  $R_p$ and $\hat I$ are Romero's resistance-weighted
    definitions, $\int\eta j^2 dS/I_p^2$ and $\int\eta j\hat j dS/\int\eta j^2
    dS\cdot I_p$, which are what make $I_p V_R$ the dissipated power; a
    Spitzer ring resistance is an estimate of the first.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eqs. (17)-(21).

    See Also
    --------
    vaft.formula.startup.plasma_resistance_uniform_ellipse_from_eta_R0_a_kappa
    """
    resistance = _finite("R_p_ohm", R_p_ohm)
    if np.any(resistance < 0.0):
        raise ValueError(f"R_p_ohm must be non-negative; got {R_p_ohm!r}")
    current = _finite("I_p_A", I_p_A)
    driven = _finite("I_ni_A", I_ni_A)
    return _maybe_scalar(resistance * (current - driven), R_p_ohm, I_p_A, I_ni_A)


def internal_inductance_rate_from_I_p_V_R_V_C(I_p_A, V_R_V, V_C_V):
    r"""Rate of change of the internal inductance, Romero's exact balance.

    $$\dot L_i = \frac{2\,(V_R - V_C)}{I_p}$$

    Parameters
    ----------
    I_p_A : float or np.ndarray
        Plasma current, finite and non-zero [A].
    V_R_V : float or np.ndarray
        Resistive voltage, finite [V].
    V_C_V : float or np.ndarray
        Voltage at the equilibrium flux surface, finite [V].

    Returns
    -------
    float or np.ndarray
        $\mathrm{d}L_i/\mathrm{d}t$ [H/s].

    Raises
    ------
    ValueError
        A non-finite input or a zero plasma current.

    Convention
    ----------
    Romero's signs, as in the module docstring.  Exact: no current-profile
    model enters.  $L_i$ is steady exactly when $V_C = V_R$, and the full
    steady state is $V_C = V_R = V_B$, a flat loop-voltage profile.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eq. (39).

    See Also
    --------
    plasma_current_rate_from_L_i_V_B_V_C_V_R
    equilibrium_surface_voltage_from_I_p_dL_i_V_R
    """
    current = _nonzero("I_p_A", I_p_A)
    rate = 2.0 * (_finite("V_R_V", V_R_V) - _finite("V_C_V", V_C_V)) / current
    return _maybe_scalar(rate, I_p_A, V_R_V, V_C_V)


def plasma_current_rate_from_L_i_V_B_V_C_V_R(L_i_H, V_B_V, V_C_V, V_R_V):
    r"""Rate of change of the plasma current, Romero's exact balance.

    $$\dot I_p = \frac{V_B + V_C - 2V_R}{L_i}$$

    Parameters
    ----------
    L_i_H : float or np.ndarray
        Dimensional internal inductance, finite and positive [H].
    V_B_V : float or np.ndarray
        Boundary loop voltage, finite [V].
    V_C_V : float or np.ndarray
        Voltage at the equilibrium flux surface, finite [V].
    V_R_V : float or np.ndarray
        Resistive voltage, finite [V].

    Returns
    -------
    float or np.ndarray
        $\mathrm{d}I_p/\mathrm{d}t$ [A/s].

    Raises
    ------
    ValueError
        A non-finite input or a non-positive internal inductance.

    Convention
    ----------
    Romero's signs.  Together with
    :func:`internal_inductance_rate_from_I_p_V_R_V_C` it gives
    $L_i\dot I_p + \tfrac12 I_p\dot L_i = V_B - V_R$, the energy balance of
    the internal field, without assuming how $L_i$ evolves.  Only the
    internal inductance appears: the external one and the coils are inside
    $V_B$.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eq. (40).

    See Also
    --------
    internal_inductance_rate_from_I_p_V_R_V_C
    equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R
    """
    inductance = _finite("L_i_H", L_i_H)
    if np.any(inductance <= 0.0):
        raise ValueError(f"L_i_H must be positive; got {L_i_H!r}")
    drive = _finite("V_B_V", V_B_V) + _finite("V_C_V", V_C_V) - 2.0 * _finite("V_R_V", V_R_V)
    return _maybe_scalar(drive / inductance, L_i_H, V_B_V, V_C_V, V_R_V)


def equilibrium_surface_voltage_from_I_p_dL_i_V_R(I_p_A, dL_i_dt_H_s, V_R_V):
    r"""Voltage at the equilibrium flux surface, inferred from the inductance history.

    $$V_C^{(L)} = V_R - \frac{I_p}{2}\,\dot L_i$$

    Parameters
    ----------
    I_p_A : float or np.ndarray
        Plasma current, finite [A].
    dL_i_dt_H_s : float or np.ndarray
        Rate of change of the internal inductance, finite [H/s].
    V_R_V : float or np.ndarray
        Resistive voltage, finite [V].

    Returns
    -------
    float or np.ndarray
        $V_C$ [V].

    Raises
    ------
    ValueError
        A non-finite input.

    Convention
    ----------
    Romero's eq. (39) solved for $V_C$.  This and
    :func:`equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R` read the same
    $V_C$ off two different measured histories, and a third route is
    $-\dot\psi_C$ from the equilibria; their disagreement is #781's first
    validation target, and it lands on whichever input is worst -- usually
    $R_p$, which enters both.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eq. (39).

    See Also
    --------
    equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R
    current_weighted_flux_from_psi_j_dS
    """
    value = _finite("V_R_V", V_R_V) - 0.5 * _finite("I_p_A", I_p_A) * _finite(
        "dL_i_dt_H_s", dL_i_dt_H_s
    )
    return _maybe_scalar(value, I_p_A, dL_i_dt_H_s, V_R_V)


def equilibrium_surface_voltage_from_L_i_dI_p_V_B_V_R(L_i_H, dI_p_dt_A_s, V_B_V, V_R_V):
    r"""Voltage at the equilibrium flux surface, inferred from the current history.

    $$V_C^{(I)} = L_i\,\dot I_p - V_B + 2V_R$$

    Parameters
    ----------
    L_i_H : float or np.ndarray
        Dimensional internal inductance, finite and non-negative [H].
    dI_p_dt_A_s : float or np.ndarray
        Rate of change of the plasma current, finite [A/s].
    V_B_V : float or np.ndarray
        Boundary loop voltage, finite [V].
    V_R_V : float or np.ndarray
        Resistive voltage, finite [V].

    Returns
    -------
    float or np.ndarray
        $V_C$ [V].

    Raises
    ------
    ValueError
        A non-finite input or a negative internal inductance.

    Convention
    ----------
    Romero's eq. (40) solved for $V_C$; see
    :func:`equilibrium_surface_voltage_from_I_p_dL_i_V_R` for the other
    route and what their difference measures.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eq. (40).

    See Also
    --------
    equilibrium_surface_voltage_from_I_p_dL_i_V_R
    """
    inductance = _finite("L_i_H", L_i_H)
    if np.any(inductance < 0.0):
        raise ValueError(f"L_i_H must be non-negative; got {L_i_H!r}")
    value = (
        inductance * _finite("dI_p_dt_A_s", dI_p_dt_A_s)
        - _finite("V_B_V", V_B_V)
        + 2.0 * _finite("V_R_V", V_R_V)
    )
    return _maybe_scalar(value, L_i_H, dI_p_dt_A_s, V_B_V, V_R_V)


def current_weighted_flux_from_psi_j_dS(psi_Wb, j_phi_A_m2, dS_m2):
    r"""Current-weighted average of the poloidal flux over the plasma cross-section.

    $$\psi_C = \frac{\int_\Omega \psi\,j_\phi\,dS}{\int_\Omega j_\phi\,dS}$$

    Parameters
    ----------
    psi_Wb : array_like
        Poloidal flux at each cell inside the boundary, full flux in Romero's
        sign, finite [Wb].
    j_phi_A_m2 : array_like
        Toroidal current density at the same cells, finite [A m^-2].
    dS_m2 : array_like
        Poloidal cross-section area of each cell, finite and non-negative;
        broadcast against the other two [m^2].

    Returns
    -------
    float
        $\psi_C$ [Wb].

    Raises
    ------
    ValueError
        A non-finite input, a negative cell area, arrays that do not
        broadcast, or a net enclosed current of zero.

    Convention
    ----------
    **The cells must be the plasma, not the grid.**  Cells outside the
    boundary carry vessel or coil current in a free-boundary solution and
    would pull $\psi_C$ toward the wall; mask them out, or pass
    $dS = 0$ for them.  The quadrature is the plain cell sum -- the caller
    chooses the cells, which is the one approximation this makes.

    $V_C = -\dot\psi_C$ is the voltage the other functions here take; the
    time derivative is the process layer's.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eq. (34).

    See Also
    --------
    internal_inductance_from_psi_C_psi_B_I_p
    """
    psi = _finite("psi_Wb", psi_Wb)
    current_density = _finite("j_phi_A_m2", j_phi_A_m2)
    area = _finite("dS_m2", dS_m2)
    if np.any(area < 0.0):
        raise ValueError("dS_m2 must be non-negative")
    try:
        psi, current_density, area = np.broadcast_arrays(psi, current_density, area)
    except ValueError:
        raise ValueError("psi_Wb, j_phi_A_m2 and dS_m2 do not broadcast") from None
    enclosed = float(np.sum(current_density * area))
    if enclosed == 0.0:
        raise ValueError("the cells enclose no net current; psi_C is undefined")
    return float(np.sum(psi * current_density * area)) / enclosed


def internal_inductance_from_psi_C_psi_B_I_p(psi_C_Wb, psi_B_Wb, I_p_A):
    r"""Dimensional internal inductance from the equilibrium and boundary fluxes.

    $$L_i = \frac{\psi_C - \psi_B}{I_p}$$

    Parameters
    ----------
    psi_C_Wb : float or np.ndarray
        Current-weighted flux average, Romero's sign, finite [Wb].
    psi_B_Wb : float or np.ndarray
        Boundary flux, same convention, finite [Wb].
    I_p_A : float or np.ndarray
        Plasma current, finite and non-zero [A].

    Returns
    -------
    float or np.ndarray
        Internal inductance [H].

    Raises
    ------
    ValueError
        A non-finite input, a zero current, or a negative result -- which
        means the flux is in a convention opposite to Romero's.

    Convention
    ----------
    **Exact, and a sign check on the flux convention.**  With
    $W_{p,\mathrm{int}} = \tfrac12(\psi_C - \psi_B) I_p$ this is
    $2W_{p,\mathrm{int}}/I_p^2$, the same $L_i$ as
    :func:`vaft.formula.equilibrium.internal_inductance_from_W_int_Ip`, so it
    is positive for any current profile.  A negative value is therefore not a
    physical answer but an inverted $\psi$ -- a COCOS whose flux grows
    outward for positive current -- and is refused rather than returned.  A
    per-radian flux is $2\pi$ too small and passes that check: convert it
    before calling.

    References
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eqs. (35)-(36).

    See Also
    --------
    current_weighted_flux_from_psi_j_dS
    """
    current = _nonzero("I_p_A", I_p_A)
    inductance = (_finite("psi_C_Wb", psi_C_Wb) - _finite("psi_B_Wb", psi_B_Wb)) / current
    if np.any(inductance < 0.0):
        raise ValueError(
            "psi_C - psi_B has the opposite sign to I_p, so L_i would be negative: "
            "the flux is not in Romero's convention (negate it, or check its COCOS)"
        )
    return _maybe_scalar(inductance, psi_C_Wb, psi_B_Wb, I_p_A)


__all__ = public_names(globals())
