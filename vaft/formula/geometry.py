"""
Canonical geometric approximations: slab, sheared slab, cylinder, local reduction.

Also the canonical slab field configurations: a Harris current sheet and an
X-point, in the same slab frame.

The reductions plasma theory uses between a real torus and an analytic model,
kept apart along two independent axes: the *geometry* (slab, cylindrical,
toroidal) and the *ordering* (local versus global; large aspect ratio
$\\epsilon = a/R_0 \\ll 1$). A slab can carry $q$, $\\hat s$, $R_0$ or a
$1/R_0$ curvature inherited from a torus and still be a slab; large aspect
ratio is an ordering, not a geometry. The guide page *Geometric
approximations* sets out what each representation retains and neglects.

Notation
--------
x, y, z : slab coordinates -- radial, binormal, parallel               [m]
r       : minor radius of the cylinder                                 [m]
theta   : poloidal angle                                               [rad]
R0      : major radius the torus is straightened at                     [m]
q       : safety factor                                                 [-]
s_hat   : magnetic shear r q'/q                                         [-]
L_s     : shear length of a sheared slab, signed                        [m]
m, n    : poloidal and toroidal mode numbers                            [-]
k_y, k_z: slab wavenumbers                                              [1/m]
k_par   : parallel wavenumber k.B/B                                     [1/m]
a       : current-sheet half-thickness                                  [m]

Conventions
-----------
Perturbations are $e^{i(m\\theta - n\\phi)}$ as in
``vaft.formula.stability.helical_phase``: both mode numbers positive, the
helicity in the minus sign; $q$ is a magnitude and $\\hat s$ keeps the sign of
$d|q|/dr$, so reversed shear flips the sign of $L_s$. A sheared slab
is $\\mathbf B = B_0(\\hat{\\mathbf z} + (x/L_s)\\hat{\\mathbf y})$ with $z$
along the field at $x = 0$. The local slab of a cylinder at $r_0$ is that
field-aligned frame: $z$ along $\\mathbf B(r_0)$, $y$ the binormal in the
surface, $x = r - r_0$. There the harmonic has $k_y = m/r_0$ and
$k_z = (m - nq_0)/(q_0R_0)$, zero on a rational surface; along the
straightened torus ($z = R_0\\phi$) it would be $-n/R_0$ instead. Positive
shear makes $L_s = -q_0R_0/\\hat s$ negative in that frame.

References
----------
.. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
       Ch. 3 and 6.
.. [2] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
       Ch. 9 and 11.
"""

from typing import Tuple

import numpy as np

from .constants import MU0

__all__ = [
    "slab_parallel_wavenumber",
    "sheared_slab_field",
    "sheared_slab_parallel_wavenumber",
    "shear_length_from_q_R0_s",
    "cylindrical_safety_factor_from_r_B",
    "cylindrical_parallel_wavenumber",
    "cylindrical_poloidal_field",
    "peaked_current_safety_factor",
    "local_slab_from_cylinder",
    "harris_sheet_field",
    "harris_sheet_current_density",
    "x_point_flux",
]


def _vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"{name} must be a 3-vector (last axis of length 3), not shape {arr.shape}")
    return arr


def _mode(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive mode number, not {value!r}")
    return int(value)


def slab_parallel_wavenumber(k, B):
    r"""Component of a wavevector along the magnetic field.

    $$k_\parallel = \frac{\mathbf k\cdot\mathbf B}{|\mathbf B|}$$

    Parameters
    ----------
    k : array_like
        Wavevector $(k_x, k_y, k_z)$, last axis of length 3 [1/m].
    B : array_like
        Magnetic field at the same point, last axis of length 3 [T].

    Returns
    -------
    float or np.ndarray
        Parallel wavenumber, signed along $\mathbf B$ [1/m].

    Raises
    ------
    ValueError
        A vector is not 3-dimensional or $|\mathbf B| = 0$.

    Convention
    ----------
    Signed: positive when $\mathbf k$ has a component along $\mathbf B$. In a
    straight slab $\mathbf B = B_0\hat{\mathbf z}$ it is $k_z$; neither
    curvature nor shear enters unless the field carries them.

    Physical interpretation
    -----------------------
    How fast a perturbation varies along field lines. Electrons stream along
    $\mathbf B$, so modes with $k_\parallel \to 0$ -- flute-like,
    field-aligned -- are the ones that escape parallel damping.

    References
    ----------
    .. [1] F. F. Chen, *Introduction to Plasma Physics and Controlled Fusion*,
           3rd ed., Springer (2016), Ch. 4.
    """
    k = _vector(k, "k")
    B = _vector(B, "B")
    b = np.linalg.norm(B, axis=-1)
    if np.any(b == 0.0):
        raise ValueError("|B| must be non-zero")
    result = np.sum(k * B, axis=-1) / b
    return float(result) if np.ndim(result) == 0 else result


def sheared_slab_field(x, B0, L_s):
    r"""Magnetic field of the sheared slab at radial position $x$.

    $$\mathbf B = B_0\left(\hat{\mathbf z} + \frac{x}{L_s}\,\hat{\mathbf y}\right)$$

    Parameters
    ----------
    x : float or np.ndarray
        Distance from the reference surface [m].
    B0 : float
        Field along $z$, the field direction at $x = 0$ [T].
    L_s : float
        Signed shear length [m].

    Returns
    -------
    np.ndarray
        $(B_x, B_y, B_z)$, last axis of length 3 [T].

    Raises
    ------
    ValueError
        ``L_s`` is zero or not finite.

    Convention
    ----------
    $x$ radial, $y$ binormal, $z$ along the field at $x = 0$. The sign of
    ``L_s`` is the sign of the field-line tilt $dB_y/dx$; the slab reached from
    a positive-shear cylinder by ``local_slab_from_cylinder`` has $L_s < 0$.

    Physical interpretation
    -----------------------
    The field direction rotates with $x$ while its magnitude stays $B_0$ to
    first order: magnetic shear without curvature. It is the local model of
    the region around a rational surface.

    Assumptions
    -----------
    $|x| \ll |L_s|$, so $|\mathbf B| \simeq B_0$; no curvature, no
    equilibrium gradient in the field strength.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 8.3.
    """
    L_s = float(L_s)
    if not np.isfinite(L_s) or L_s == 0.0:
        raise ValueError(f"L_s must be finite and non-zero, not {L_s!r}")
    x = np.asarray(x, dtype=float)
    B0 = float(B0)
    return np.stack([np.zeros_like(x), B0 * x / L_s, np.full_like(x, B0)], axis=-1)


def sheared_slab_parallel_wavenumber(x, k_y, L_s, k_z=0.0):
    r"""Parallel wavenumber across a sheared slab, to first order in $x/L_s$.

    $$k_\parallel(x) = k_z + k_y\,\frac{x}{L_s}$$

    Parameters
    ----------
    x : float or np.ndarray
        Distance from the reference surface [m].
    k_y : float
        Binormal wavenumber [1/m].
    L_s : float
        Signed shear length, as in ``sheared_slab_field`` [m].
    k_z : float
        Wavenumber along the field at $x = 0$; zero when $x = 0$ is the
        resonant surface [1/m].

    Returns
    -------
    float or np.ndarray
        $k_\parallel$ [1/m].

    Raises
    ------
    ValueError
        ``L_s`` is zero or not finite.

    Convention
    ----------
    The field of ``sheared_slab_field``; ``slab_parallel_wavenumber`` of that
    field to first order. With $k_z = 0$ the resonance $k_\parallel = 0$ is at
    $x = 0$ and $k_\parallel' = k_y/L_s$.

    Physical interpretation
    -----------------------
    A perturbation is field-aligned only on one surface; away from it,
    $|k_\parallel|$ grows linearly, which is what localises tearing layers,
    drift waves and ballooning-type modes around rational surfaces.

    Assumptions
    -----------
    $|x| \ll |L_s|$, dropping the $O(x^2/L_s^2)$ change of $|\mathbf B|$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 8.3.
    """
    L_s = float(L_s)
    if not np.isfinite(L_s) or L_s == 0.0:
        raise ValueError(f"L_s must be finite and non-zero, not {L_s!r}")
    result = float(k_z) + float(k_y) * np.asarray(x, dtype=float) / L_s
    return float(result) if np.ndim(result) == 0 else result


def shear_length_from_q_R0_s(q, R0, s_hat):
    r"""Magnitude of the shear length a torus or cylinder gives its local slab.

    $$L_s = \frac{q R_0}{|\hat s|}$$

    Parameters
    ----------
    q : float or np.ndarray
        Safety factor at the reference surface [-].
    R0 : float
        Major radius [m].
    s_hat : float or np.ndarray
        Magnetic shear $\hat s = (r/q)\,dq/dr$, for example ``shear_from_r_q`` [-].

    Returns
    -------
    float or np.ndarray
        $|L_s|$, the distance across the surfaces at which the field tilts by
        $B_y/B_z = 1$ in the local slab [m].

    Raises
    ------
    ValueError
        ``s_hat`` is zero.

    Convention
    ----------
    Unsigned: magnitudes of $q$ and $\hat s$, whatever sign a COCOS gives
    them. The sign a slab needs depends on how its $y$ is oriented, see
    ``local_slab_from_cylinder`` for VAFT's.

    Physical interpretation
    -----------------------
    Short $L_s$ means strong shear: a mode stays field-aligned over a thinner
    layer. Local turbulence and drift-wave models keep this tokamak
    coefficient while discarding the rest of the geometry.

    Assumptions
    -----------
    Large aspect ratio and a locality $|x| \ll r$ around the surface.

    References
    ----------
    .. [1] W. Horton, Rev. Mod. Phys. 71 (1999) 735, Sec. II.
    """
    s_hat = np.asarray(s_hat, dtype=float)
    if np.any(s_hat == 0.0):
        raise ValueError("s_hat must be non-zero: a shearless surface has no finite shear length")
    result = np.abs(np.asarray(q, dtype=float)) * float(R0) / np.abs(s_hat)
    return float(result) if np.ndim(result) == 0 else result


def cylindrical_safety_factor_from_r_B(r, B_theta, B_z, R0):
    r"""Safety factor of a straight cylinder standing in for a torus.

    $$q(r) = \frac{r B_z}{R_0 B_\theta}$$

    Parameters
    ----------
    r : float or np.ndarray
        Minor radius [m].
    B_theta : float or np.ndarray
        Poloidal (azimuthal) field at $r$ [T].
    B_z : float or np.ndarray
        Axial field, the stand-in for $B_\phi$ [T].
    R0 : float
        Major radius the torus is straightened at; the cylinder has length $2\pi R_0$ [m].

    Returns
    -------
    float or np.ndarray
        $q(r)$ [-].

    Raises
    ------
    ValueError
        ``B_theta`` is zero somewhere.

    Convention
    ----------
    $z = R_0\phi$ and $B_z \leftrightarrow B_\phi$: the field-line pitch
    $d\phi/d\theta$ of the torus becomes $dz/(R_0 d\theta)$. Signs follow
    the fields; take magnitudes for the usual positive $q$.

    Physical interpretation
    -----------------------
    Toroidal transits per poloidal transit. It is the leading $O(1)$ limit of
    the toroidal $q$ under $\epsilon = a/R_0 \ll 1$: no $1/R$ variation, no
    shaping, no Shafranov shift.

    Assumptions
    -----------
    A periodic cylinder; a screw pinch has the same $q$ with $R_0$ set by
    the imposed period.

    Reduction
    ---------
    input: profile_1d
    output: profile_1d
    kind: normalization
    locality: flux_surface_local
    role: state_coordinate

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.3.
    .. [2] J. P. Freidberg, *Ideal MHD*, Cambridge University Press (2014),
           Ch. 11 (screw pinch).
    """
    B_theta = np.asarray(B_theta, dtype=float)
    if np.any(B_theta == 0.0):
        raise ValueError("B_theta must be non-zero")
    result = np.asarray(r, dtype=float) * np.asarray(B_z, dtype=float) / (float(R0) * B_theta)
    return float(result) if np.ndim(result) == 0 else result


def cylindrical_parallel_wavenumber(m_pol, n_tor, q, R0):
    r"""Parallel wavenumber of an $m/n$ harmonic on a cylindrical surface of safety factor $q$.

    $$k_\parallel = \frac{m - nq}{qR_0}$$

    Parameters
    ----------
    m_pol : int
        Poloidal mode number [-].
    n_tor : int
        Toroidal mode number [-].
    q : float or np.ndarray
        Safety factor of the surface [-].
    R0 : float
        Major radius the cylinder is straightened at [m].

    Returns
    -------
    float or np.ndarray
        $k_\parallel$; zero exactly where $q = m/n$ [1/m].

    Raises
    ------
    ValueError
        A mode number is not positive, or ``q`` is zero.

    Convention
    ----------
    The harmonic is $e^{i(m\theta - n\phi)}$ (``helical_phase``) with
    $z = R_0\phi$, and $B \simeq B_z$ ($B_\theta \ll B_z$). $q$ is the
    positive magnitude, as the mode numbers are: a negative $q$ from a COCOS
    sign never reaches $m/n$. With $q$ rising outward, $k_\parallel$ falls
    through zero at the rational surface.

    Physical interpretation
    -----------------------
    The resonance condition in wave-number form: the perturbation is
    constant along field lines where $q(r_s) = m/n$, and a rational surface is
    the place $k_\parallel = 0$.

    Assumptions
    -----------
    Cylinder, large aspect ratio: $B_\theta/B_z = r/(qR_0) \ll 1$.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.4.
    """
    m = _mode(m_pol, "m_pol")
    n = _mode(n_tor, "n_tor")
    q = np.asarray(q, dtype=float)
    if np.any(q == 0.0):
        raise ValueError("q must be non-zero")
    result = (m - n * q) / (q * float(R0))
    return float(result) if np.ndim(result) == 0 else result


def cylindrical_poloidal_field(r, I_enclosed):
    r"""Poloidal field of a straight current column by Ampère's law.

    $$B_\theta(r) = \frac{\mu_0 I(r)}{2\pi r}$$

    Parameters
    ----------
    r : float or np.ndarray
        Minor radius, positive [m].
    I_enclosed : float or np.ndarray
        Axial current inside radius $r$ [A].

    Returns
    -------
    float or np.ndarray
        $B_\theta$ [T].

    Raises
    ------
    ValueError
        ``r`` is not positive.

    Convention
    ----------
    $B_\theta$ has the sign of the enclosed current along $+z$, right-handed
    about it. Exact for a cylindrically symmetric current distribution.

    Physical interpretation
    -----------------------
    Only the current inside $r$ sets the field at $r$: a centrally peaked
    current makes $B_\theta$ rise fast and then fall as $1/r$, which is what
    shapes $q(r)$.

    Reduction
    ---------
    input: profile_1d
    output: profile_1d
    kind: normalization
    locality: flux_surface_local
    role: profile_descriptor

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.3.
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("r must be positive")
    from .constants import MU0

    result = MU0 * np.asarray(I_enclosed, dtype=float) / (2.0 * np.pi * r)
    return float(result) if np.ndim(result) == 0 else result


def peaked_current_safety_factor(x, q_a, nu):
    r"""Cylindrical $q$ profile of the peaked current $j \propto (1 - x^2)^\nu$.

    $$q(x) = \frac{q_a\,x^2}{1 - (1 - x^2)^{\nu + 1}},\qquad q(0) = \frac{q_a}{\nu + 1}$$

    Parameters
    ----------
    x : float or np.ndarray
        Normalised minor radius $r/a$, in $[0, 1]$ [-].
    q_a : float
        Edge safety factor $q(a)$ [-].
    nu : float
        Current peaking exponent, non-negative; $\nu = 0$ is a flat current [-].

    Returns
    -------
    float or np.ndarray
        $q(x)$, rising from $q_a/(\nu + 1)$ on axis to $q_a$ at the edge [-].

    Raises
    ------
    ValueError
        ``x`` lies outside $[0, 1]$, ``q_a`` is not positive or ``nu`` is negative.

    Convention
    ----------
    The enclosed current is $I(x) = I_a[1 - (1 - x^2)^{\nu+1}]$; with
    ``cylindrical_poloidal_field`` and ``cylindrical_safety_factor_from_r_B``
    this is $q = rB_z/(R_0B_\theta)$. The axis value is the limit $x \to 0$.

    Physical interpretation
    -----------------------
    The more peaked the current (larger $\nu$), the lower $q$ on axis and the
    stronger the shear outside: $q_a/q_0 = \nu + 1$ is fixed by the peaking
    alone.

    Assumptions
    -----------
    Cylinder, large aspect ratio, the standard model profile; not a
    reconstructed equilibrium.

    References
    ----------
    .. [1] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 3.3 and 6.8 (the profile used for tearing and kink examples).
    """
    x = np.asarray(x, dtype=float)
    q_a, nu = float(q_a), float(nu)
    if np.any(x < 0.0) or np.any(x > 1.0 + 1e-12):
        raise ValueError("x must lie in [0, 1]")
    x = np.minimum(x, 1.0)
    if q_a <= 0.0 or nu < 0.0:
        raise ValueError(f"q_a must be positive and nu non-negative, not {q_a!r} and {nu!r}")
    x2 = x * x
    with np.errstate(divide="ignore"):  # x = 1: log1p(-1) = -inf, expm1(-inf) = -1, denom = 1
        denom = -np.expm1((nu + 1.0) * np.log1p(-x2))  # 1 - (1 - x^2)^(nu+1), exact at small x
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(x2 > 1e-12, q_a * x2 / np.where(denom > 0.0, denom, 1.0), q_a / (nu + 1.0))
    return float(result) if np.ndim(result) == 0 else result


def local_slab_from_cylinder(m_pol, n_tor, r_0, R0, q_0, s_hat) -> Tuple[float, float, float]:
    r"""The field-aligned sheared slab of a cylinder at $r_0$: $(k_y, L_s, k_z)$ of an $m/n$ harmonic.

    $$k_y = \frac{m}{r_0},\qquad L_s = -\frac{q_0 R_0}{\hat s},\qquad k_z = \frac{m - nq_0}{q_0R_0}$$

    Parameters
    ----------
    m_pol : int
        Poloidal mode number [-].
    n_tor : int
        Toroidal mode number [-].
    r_0 : float
        Radius of the reference surface [m].
    R0 : float
        Major radius [m].
    q_0 : float
        Safety factor at $r_0$, a positive magnitude [-].
    s_hat : float
        Magnetic shear at $r_0$ [-].

    Returns
    -------
    k_y : float
        Binormal wavenumber $m/r_0$ [1/m].
    L_s : float
        Signed shear length of the local slab [m].
    k_z : float
        Wavenumber along the field at $r_0$; zero when $q_0 = m/n$ [1/m].

    Raises
    ------
    ValueError
        A mode number is not positive, ``r_0``, ``R0`` or ``q_0`` is not
        positive, or ``s_hat`` is zero.

    Convention
    ----------
    $x = r - r_0$; $z$ along $\mathbf B(r_0)$ and $y$ the binormal in the
    surface, the frame of ``sheared_slab_field``, so
    ``sheared_slab_parallel_wavenumber(x, *local_slab_from_cylinder(...))``
    is the cylinder's $k_\parallel$ to first order in $x$. The harmonic is
    $e^{i(m\theta - n\phi)}$; its wavenumber along the straightened torus
    ($z = R_0\phi$) would be $-n/R_0$, which is not this $k_z$. With
    $\hat s > 0$ the field line, followed in $z$, drifts to $-y$ faster on the
    outer surfaces, which makes $L_s$ negative in this frame.

    Physical interpretation
    -----------------------
    The bridge from global mode numbers to local wavenumbers: on a rational
    surface $q_0 = m/n$ gives $k_z = 0$, and the shear becomes
    $k_\parallel' = k_y/L_s = -k_y\hat s/(q_0R_0)$.

    Assumptions
    -----------
    Locality $|x| \ll r_0$ with $q \simeq q_0 + q_0'x$; large aspect ratio
    ($B_\theta \ll B_z$). The same slab follows from a toroidal equilibrium by
    a local expansion, without the cylinder in between.

    References
    ----------
    .. [1] H. P. Furth, J. Killeen and M. N. Rosenbluth, Phys. Fluids 6
           (1963) 459.
    .. [2] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press (2011),
           Sec. 6.8.
    """
    m = _mode(m_pol, "m_pol")
    n = _mode(n_tor, "n_tor")
    r_0, R0, q_0, s_hat = float(r_0), float(R0), float(q_0), float(s_hat)
    if not (r_0 > 0.0 and R0 > 0.0 and q_0 > 0.0):
        raise ValueError(f"r_0, R0 and q_0 must be positive, not {r_0!r}, {R0!r} and {q_0!r}")
    if s_hat == 0.0 or not np.isfinite(s_hat):
        raise ValueError(f"s_hat must be finite and non-zero, not {s_hat!r}")
    return m / r_0, -q_0 * R0 / s_hat, (m - n * q_0) / (q_0 * R0)


# ------------------------------------------------------------------
# Canonical slab field configurations: current sheet, X-point
# ------------------------------------------------------------------


def harris_sheet_field(x, B0, a):
    r"""Reconnecting field of a Harris current sheet.

    $$B_y(x) = B_0\tanh\frac{x}{a}$$

    Parameters
    ----------
    x : float or np.ndarray
        Distance from the sheet centre, along its normal [m].
    B0 : float
        Asymptotic reconnecting field, signed [T].
    a : float
        Sheet half-thickness, positive [m].

    Returns
    -------
    float or np.ndarray
        $B_y$ [T].

    Raises
    ------
    ValueError
        ``a`` is not positive and finite.

    Convention
    ----------
    Slab frame of ``sheared_slab_field``: $x$ the sheet normal, $y$ the
    reconnecting direction, $z$ the current (and guide-field) direction;
    $B_y(-x) = -B_y(x)$. A guide field $B_g\hat{\mathbf z}$ adds without
    changing $J_z$ (``harris_sheet_current_density``).

    Physical interpretation
    -----------------------
    The field reverses across a layer of thickness $\sim 2a$; for
    $|x| \ll a$ it is the sheared slab with $B_y' = B_0/a$ and no guide field.
    In the original Harris equilibrium the magnetic pressure is balanced by a
    plasma pressure $\propto \mathrm{sech}^2(x/a)$.

    Assumptions
    -----------
    One-dimensional, $\partial_y = \partial_z = 0$; the profile, not the
    kinetic (drifting-Maxwellian) closure that makes it an exact Vlasov
    equilibrium.

    References
    ----------
    .. [1] E. G. Harris, Nuovo Cimento 23, 115 (1962).
    .. [2] D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge
           University Press (2000), Sec. 3.1.
    """
    a = _positive(a, "a")
    return float(B0) * np.tanh(np.asarray(x, dtype=float) / a)


def harris_sheet_current_density(x, B0, a):
    r"""Current density of a Harris current sheet, from Ampere's law.

    $$J_z(x) = \frac{1}{\mu_0}\frac{dB_y}{dx} = \frac{B_0}{\mu_0 a}\,\mathrm{sech}^2\frac{x}{a}$$

    Parameters
    ----------
    x : float or np.ndarray
        Distance from the sheet centre, along its normal [m].
    B0 : float
        Asymptotic reconnecting field, signed [T].
    a : float
        Sheet half-thickness, positive [m].

    Returns
    -------
    float or np.ndarray
        $J_z$ [A/m^2].

    Raises
    ------
    ValueError
        ``a`` is not positive and finite.

    Convention
    ----------
    Right-handed $(x, y, z)$ with $\mu_0\mathbf J = \nabla\times\mathbf B$,
    so $\mu_0J_z = \partial_xB_y$: $B_0 > 0$ ($B_y$ along $+y$ above the
    sheet) gives current along $+z$. The sheet carries $2B_0/\mu_0$ per unit
    length in $y$.

    Physical interpretation
    -----------------------
    The field reversal of ``harris_sheet_field`` and a localized current are
    the same object: the current is confined to $|x| \lesssim a$, where the
    field changes sign.

    Assumptions
    -----------
    As ``harris_sheet_field``; displacement current neglected.

    References
    ----------
    .. [1] E. G. Harris, Nuovo Cimento 23, 115 (1962).
    .. [2] D. Biskamp, *Magnetic Reconnection in Plasmas*, Cambridge
           University Press (2000), Sec. 3.1.
    """
    a = _positive(a, "a")
    return float(B0) / (MU0 * a) / np.cosh(np.asarray(x, dtype=float) / a) ** 2


def x_point_flux(x, y, B_prime):
    r"""Flux function of a current-free magnetic X-point.

    $$\psi(x, y) = \frac{B'}{2}\left(x^2 - y^2\right),\qquad
      \mathbf B_\perp = \hat{\mathbf z}\times\nabla\psi = B'\,(y, x)$$

    Parameters
    ----------
    x : float or np.ndarray
        Coordinate along the inflow (sheet normal) direction [m].
    y : float or np.ndarray
        Coordinate along the outflow direction [m].
    B_prime : float
        Field gradient $B'$, non-zero [T/m].

    Returns
    -------
    float or np.ndarray
        $\psi$; in-plane field lines lie on its contours [T m].

    Raises
    ------
    ValueError
        ``B_prime`` is zero or not finite.

    Convention
    ----------
    The same $\mathbf B_\perp = \hat{\mathbf z}\times\nabla\psi$ as
    ``slab_perturbed_flux``: along the inflow axis $B_y = B'x$ reverses
    across $x = 0$ as in a current sheet, along the outflow axis
    $B_x = B'y$. The separatrices are $y = \pm x$, the contour
    $\psi = 0$ through the X-point.

    Physical interpretation
    -----------------------
    The lowest-order field about a null: four branches meeting at the
    X-point, $\nabla^2\psi = 0$ so no current. A current along $z$ at the null
    changes the separatrix angle away from $90^\circ$ (collapse towards a
    sheet). Near each X-point of a tearing island the flux of
    ``slab_perturbed_flux`` has this saddle form.

    Assumptions
    -----------
    Two-dimensional, $\partial_z = 0$; expansion to second order about the
    null, valid for $|x|, |y|$ small against the scale of the surrounding
    field.

    References
    ----------
    .. [1] E. R. Priest and T. G. Forbes, *Magnetic Reconnection*, Cambridge
           University Press (2000), Sec. 2.1.
    """
    B_prime = float(B_prime)
    if not np.isfinite(B_prime) or B_prime == 0.0:
        raise ValueError(f"B_prime must be finite and non-zero, not {B_prime!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * B_prime * (x * x - y * y)


def _positive(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, not {value!r}")
    return value
