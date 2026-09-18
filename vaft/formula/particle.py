"""
Single-particle motion in magnetic and electric fields.

This module provides the gyration and guiding-centre drift relations of a
charged particle, and a Boris integrator of the full Lorentz-force orbit
against which the drift relations can be checked.

Notation
--------
q      : particle charge, signed                         [C]
m      : particle mass                                   [kg]
B      : magnetic field (vector or magnitude)            [T]
E      : electric field                                  [V/m]
v_perp : speed perpendicular to B                        [m/s]
v_par  : speed parallel to B                             [m/s]
omega_c: signed gyrofrequency qB/m                       [rad/s]
rho    : Larmor radius                                   [m]
R_c    : radius-of-curvature vector of the field line    [m]
"""

from typing import Callable, Tuple

import numpy as np

__all__ = [
    "gyrofrequency",
    "larmor_radius",
    "exb_drift_velocity",
    "grad_b_drift_velocity",
    "curvature_drift_velocity",
    "boris_orbit",
]


def _vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"{name} must be a 3-vector (last axis of length 3), not shape {arr.shape}")
    return arr


def gyrofrequency(q, m, B):
    r"""Signed gyrofrequency of a charged particle.

    $$\omega_c = \frac{qB}{m}$$

    Parameters
    ----------
    q : float or np.ndarray
        Particle charge, signed [C].
    m : float or np.ndarray
        Particle mass [kg].
    B : float or np.ndarray
        Magnetic-field magnitude [T].

    Returns
    -------
    float or np.ndarray
        Gyrofrequency, positive for positive charge [rad/s].

    Raises
    ------
    ValueError
        ``m`` is not positive.

    Convention
    ----------
    Signed: $\omega_c > 0$ for an ion, $< 0$ for an electron. An ion gyrates
    in the left-handed sense about $\mathbf{B}$ (clockwise seen with
    $\mathbf{B}$ pointing at the viewer), an electron in the right-handed
    sense; the sign carries that. Use ``abs`` for the frequency alone.

    Physical interpretation
    -----------------------
    The rate at which the Lorentz force turns the perpendicular velocity:
    the fastest time scale of magnetised motion, and the reason drifts are
    slow by comparison.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled
           Fusion*, 3rd ed., Springer (2016), Sec. 2.2.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 2.2.
    """
    m = np.asarray(m, dtype=float)
    if np.any(m <= 0.0):
        raise ValueError("m must be positive")
    return np.asarray(q, dtype=float) * np.asarray(B, dtype=float) / m


def larmor_radius(q, m, v_perp, B):
    r"""Larmor (gyro) radius of a charged particle.

    $$\rho = \frac{m\,v_\perp}{|q|\,B}$$

    Parameters
    ----------
    q : float or np.ndarray
        Particle charge; only its magnitude enters [C].
    m : float or np.ndarray
        Particle mass [kg].
    v_perp : float or np.ndarray
        Speed perpendicular to the magnetic field [m/s].
    B : float or np.ndarray
        Magnetic-field magnitude [T].

    Returns
    -------
    float or np.ndarray
        Radius of the gyro-orbit [m].

    Raises
    ------
    ValueError
        ``q`` or ``B`` is zero.

    Convention
    ----------
    Always positive: the gyration sense is in ``gyrofrequency``. $v_\perp$ is
    the speed in the guiding-centre frame, i.e. after removing any drift.

    Physical interpretation
    -----------------------
    The scale below which a particle does not follow field lines; drift
    approximations need it small against the field's gradient length.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled
           Fusion*, 3rd ed., Springer (2016), Sec. 2.2.
    """
    q = np.abs(np.asarray(q, dtype=float))
    B = np.abs(np.asarray(B, dtype=float))
    if np.any(q == 0.0) or np.any(B == 0.0):
        raise ValueError("q and B must be non-zero")
    return np.asarray(m, dtype=float) * np.abs(np.asarray(v_perp, dtype=float)) / (q * B)


def exb_drift_velocity(E, B):
    r"""E-cross-B drift velocity of the guiding centre.

    $$\mathbf{v}_E = \frac{\mathbf{E}\times\mathbf{B}}{B^{2}}$$

    Parameters
    ----------
    E : array_like
        Electric field, last axis the three Cartesian components [V/m].
    B : array_like
        Magnetic field, last axis the three Cartesian components [T].

    Returns
    -------
    np.ndarray
        Drift velocity, same shape as the broadcast inputs [m/s].

    Raises
    ------
    ValueError
        An input is not a 3-vector, or ``B`` vanishes.

    Convention
    ----------
    Right-handed Cartesian components. The drift is independent of charge
    and mass, so ions and electrons move together and no current flows.

    Physical interpretation
    -----------------------
    Over each gyration the particle gains energy on one side of its orbit
    and loses it on the other, so its radius changes around the orbit and
    the guiding centre steps sideways -- perpendicular to both fields.

    Assumptions
    -----------
    $E_\perp \ll cB$ (non-relativistic drift); uniform fields over a Larmor
    radius and slow in time compared with the gyroperiod.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled
           Fusion*, 3rd ed., Springer (2016), Sec. 2.2.2.
    """
    E = _vector(E, "E")
    B = _vector(B, "B")
    B2 = np.sum(B * B, axis=-1, keepdims=True)
    if np.any(B2 == 0.0):
        raise ValueError("B must be non-zero")
    return np.cross(E, B) / B2


def grad_b_drift_velocity(q, m, v_perp, B, grad_B):
    r"""Grad-B drift velocity of the guiding centre.

    $$\mathbf{v}_{\nabla B} = \frac{m v_\perp^{2}}{2qB}\,
    \frac{\mathbf{B}\times\nabla B}{B^{2}}$$

    Parameters
    ----------
    q : float
        Particle charge, signed [C].
    m : float
        Particle mass [kg].
    v_perp : float or np.ndarray
        Speed perpendicular to the magnetic field [m/s].
    B : array_like
        Magnetic field, last axis the three Cartesian components [T].
    grad_B : array_like
        Gradient of the field magnitude $|\mathbf{B}|$ [T/m].

    Returns
    -------
    np.ndarray
        Drift velocity [m/s].

    Raises
    ------
    ValueError
        An input is not a 3-vector, ``q`` is zero, or ``B`` vanishes.

    Convention
    ----------
    Charge-signed: ions and electrons drift in opposite directions, which is
    what makes this drift carry current.

    Physical interpretation
    -----------------------
    The orbit is tighter where the field is stronger, so the guiding centre
    walks perpendicular to both $\mathbf{B}$ and its gradient.

    Assumptions
    -----------
    $\rho \ll B/|\nabla B|$; the drift is the first-order average over a
    gyration.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled
           Fusion*, 3rd ed., Springer (2016), Sec. 2.3.1.
    """
    B = _vector(B, "B")
    grad_B = _vector(grad_B, "grad_B")
    q = float(q)
    if q == 0.0:
        raise ValueError("q must be non-zero")
    Bmag = np.linalg.norm(B, axis=-1, keepdims=True)
    if np.any(Bmag == 0.0):
        raise ValueError("B must be non-zero")
    v_perp = np.asarray(v_perp, dtype=float)[..., None] if np.ndim(v_perp) else float(v_perp)
    return float(m) * v_perp ** 2 / (2.0 * q * Bmag) * np.cross(B, grad_B) / Bmag ** 2


def curvature_drift_velocity(q, m, v_par, B, R_c):
    r"""Curvature drift velocity of the guiding centre.

    $$\mathbf{v}_R = \frac{m v_\parallel^{2}}{qB^{2}}\,
    \frac{\mathbf{R}_c\times\mathbf{B}}{R_c^{2}}$$

    Parameters
    ----------
    q : float
        Particle charge, signed [C].
    m : float
        Particle mass [kg].
    v_par : float or np.ndarray
        Speed along the magnetic field [m/s].
    B : array_like
        Magnetic field, last axis the three Cartesian components [T].
    R_c : array_like
        Radius-of-curvature vector, from the centre of curvature to the
        field line [m].

    Returns
    -------
    np.ndarray
        Drift velocity [m/s].

    Raises
    ------
    ValueError
        An input is not a 3-vector, ``q`` is zero, or ``B`` or ``R_c``
        vanishes.

    Convention
    ----------
    $\mathbf{R}_c$ points *outward*, from the centre of curvature to the
    line, as in Chen. Charge-signed like the grad-B drift. In a vacuum field
    $\nabla\times\mathbf{B} = 0$ the two drifts add in the same direction.

    Physical interpretation
    -----------------------
    The centrifugal force of streaming along a curved line, $m v_\parallel^2
    / R_c$, acts like any other perpendicular force $\mathbf{F}$ and drives
    $\mathbf{F}\times\mathbf{B}/(qB^2)$.

    Assumptions
    -----------
    $\rho \ll R_c$.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled
           Fusion*, 3rd ed., Springer (2016), Sec. 2.3.2.
    """
    B = _vector(B, "B")
    R_c = _vector(R_c, "R_c")
    q = float(q)
    if q == 0.0:
        raise ValueError("q must be non-zero")
    B2 = np.sum(B * B, axis=-1, keepdims=True)
    R2 = np.sum(R_c * R_c, axis=-1, keepdims=True)
    if np.any(B2 == 0.0) or np.any(R2 == 0.0):
        raise ValueError("B and R_c must be non-zero")
    v_par = np.asarray(v_par, dtype=float)[..., None] if np.ndim(v_par) else float(v_par)
    return float(m) * v_par ** 2 / (q * B2) * np.cross(R_c, B) / R2


def boris_orbit(q, m, x0, v0, E_field: Callable, B_field: Callable, dt, n_steps) -> Tuple[np.ndarray, np.ndarray]:
    r"""Charged-particle orbit from the Lorentz force, by the Boris scheme.

    $$m\,\frac{\mathrm{d}\mathbf{v}}{\mathrm{d}t} = q\,(\mathbf{E} + \mathbf{v}\times\mathbf{B}),
    \qquad \frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t} = \mathbf{v}$$

    Parameters
    ----------
    q : float
        Particle charge, signed [C].
    m : float
        Particle mass [kg].
    x0 : array_like
        Initial position [m].
    v0 : array_like
        Initial velocity [m/s].
    E_field : callable
        ``E_field(x)`` returning the electric field at position ``x`` [V/m].
    B_field : callable
        ``B_field(x)`` returning the magnetic field at position ``x`` [T].
    dt : float
        Time step [s].
    n_steps : int
        Number of steps [-].

    Returns
    -------
    positions : np.ndarray
        Positions, shape ``(n_steps + 1, 3)``, the first being ``x0`` [m].
    velocities : np.ndarray
        ``velocities[i + 1]`` is the velocity over the step from
        ``positions[i]`` to ``positions[i + 1]``; ``velocities[0]`` is
        ``v0``. Shape ``(n_steps + 1, 3)`` [m/s].

    Raises
    ------
    ValueError
        ``m``, ``dt`` or ``n_steps`` is not positive, or a vector is not a
        3-vector.

    Convention
    ----------
    Non-relativistic leapfrog: velocities live at the half steps between
    positions (half electric kick, magnetic rotation, half kick, then the
    position advance). ``v0`` is taken as the velocity half a step before
    ``x0``. Right-handed Cartesian components.

    Physical interpretation
    -----------------------
    The full orbit the drift formulas average: integrated in a uniform or
    slowly varying field, its guiding centre moves at ``exb_drift_velocity``,
    ``grad_b_drift_velocity`` and ``curvature_drift_velocity``.

    Numerical notes
    ---------------
    The magnetic rotation is exact in angle up to the $\tan(\omega_c\Delta t
    /2)$ phase error, so with $E = 0$ the kinetic energy is conserved to
    round-off for any step; the scheme is volume-preserving and has no
    secular energy drift. Keep $|\omega_c|\Delta t \lesssim 0.1$ for an
    accurate gyrophase. Fixed step, so the output is deterministic.

    References
    ----------
    .. [1] J. P. Boris, Proc. 4th Conf. Numer. Sim. Plasmas, NRL (1970), 3.
    .. [2] C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer
           Simulation*, McGraw-Hill (1985), Sec. 4-4.
    .. [3] H. Qin et al., Phys. Plasmas 20, 084503 (2013).
    """
    m = float(m)
    dt = float(dt)
    n_steps = int(n_steps)
    if not m > 0.0 or not dt > 0.0 or n_steps <= 0:
        raise ValueError("m, dt and n_steps must be positive")
    x = _vector(x0, "x0").copy()
    v = _vector(v0, "v0").copy()
    positions = np.empty((n_steps + 1, 3))
    velocities = np.empty((n_steps + 1, 3))
    positions[0], velocities[0] = x, v
    k = float(q) * dt / (2.0 * m)
    for i in range(n_steps):
        e = _vector(E_field(x), "E_field(x)")
        b = _vector(B_field(x), "B_field(x)")
        v_minus = v + k * e
        t = k * b
        s = 2.0 * t / (1.0 + t @ t)
        v_prime = v_minus + np.cross(v_minus, t)
        v = v_minus + np.cross(v_prime, s) + k * e
        x = x + dt * v
        positions[i + 1], velocities[i + 1] = x, v
    return positions, velocities
