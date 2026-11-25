"""Divergence-free particle benchmark fields and adapters to existing VAFT fields.

All factories return static Cartesian SI-valued ParticleField objects. Analytic
examples are controlled reference geometries, not reconstructed VEST equilibria.
"""

from __future__ import annotations

import numpy as np

from vaft.data.particle import ParticleField

__all__ = [
    "uniform_particle_field",
    "slab_particle_field",
    "mirror_particle_field",
    "toroidal_particle_field",
    "equilibrium_particle_field",
    "filament_particle_field",
]


def _positive(**values):
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def uniform_particle_field(B=(0.0, 0.0, 1.0), E=(0.0, 0.0, 0.0)):
    """Construct a uniform magnetic field with a consistent crossed electric field.

    Parameters
    ----------
    B : array_like, optional
        Nonzero Cartesian magnetic field; unit z field is a reference default [T].
    E : array_like, optional
        Uniform electric field; zero is the magnetic-only default [V/m].

    Returns
    -------
    ParticleField
        Uniform field with symmetric gauge A = B cross x / 2 [-].

    Applicability
    -------------
    Machine-independent. Exact uniform-field benchmark.

    Provenance
    ----------
    .. [1] F. F. Chen, Introduction to Plasma Physics and Controlled Fusion,
       3rd ed. (2016), Sec. 2.2.
    """
    B, E = np.asarray(B, float), np.asarray(E, float)
    if (
        B.shape != (3,)
        or E.shape != (3,)
        or not np.all(np.isfinite(np.r_[B, E]))
        or np.linalg.norm(B) == 0
    ):
        raise ValueError("finite 3-vector fields and nonzero B required")
    B, E = B.copy(), E.copy()
    electric = np.any(E != 0)
    return ParticleField(
        lambda x: B.copy(),
        E=(lambda x: E.copy()) if electric else None,
        potential=(lambda x: -float(E @ x)) if electric else None,
        jacobian=lambda x: np.zeros((3, 3)),
        vector_potential=lambda x: np.cross(B, x) / 2,
        derivative_step=1e-4,
        name="uniform Cartesian field",
        provenance={
            "gauge": "A=B cross x/2; Phi=-E dot x", "geometry": "uniform",
            "B_T": B.tolist(), "E_V_per_m": E.tolist(),
        },
    )


def slab_particle_field(B0=1.0, length=30.0):
    """Construct a divergence-free slab gradient Bz = B0 (1 + x/length).

    Parameters
    ----------
    B0 : float, optional
        Positive reference field, a benchmark default [T].
    length : float, optional
        Positive gradient scale, a benchmark default [m].

    Returns
    -------
    ParticleField
        Slab field restricted to abs(x) < length/2 [-].

    Applicability
    -------------
    Machine-independent. Weak-gradient drift benchmark when rho/length is small.

    Provenance
    ----------
    .. [1] F. F. Chen, Introduction to Plasma Physics and Controlled Fusion,
       3rd ed. (2016), Sec. 2.3.
    """
    _positive(B0=B0, length=length)
    J = np.zeros((3, 3))
    J[2, 0] = B0 / length
    return ParticleField(
        lambda x: np.array([0.0, 0.0, B0 * (1 + x[0] / length)]),
        jacobian=lambda x: J.copy(),
        contains=lambda x: abs(x[0]) < length / 2,
        vector_potential=lambda x: np.array(
            [0.0, B0 * (x[0] + x[0] ** 2 / (2 * length)), 0.0]
        ),
        name="divergence-free slab gradient",
        provenance={"gauge": "Ay=B0(x+x^2/(2L))", "length_m": length, "B0_T": B0},
    )


def mirror_particle_field(B0=1.0, length=10.0):
    """Construct an exactly solenoidal axisymmetric parabolic mirror field.

    Parameters
    ----------
    B0 : float, optional
        Minimum on-axis field, a benchmark default [T].
    length : float, optional
        Half length of the accepted mirror domain, a benchmark default [m].

    Returns
    -------
    ParticleField
        Field with Bz = B0(1+z^2/length^2) and Br = -r B0 z/length^2 [-].

    Convention
    ----------
    The domain is abs(z) < length and r < length/4. On axis, s=z and the
    nominal mirror ratio at the two ends is two. This is a current-carrying
    analytic benchmark, not an assertion of a vacuum-coil solution.

    Applicability
    -------------
    Machine-independent. Magnetized mirror and harmonic-bounce benchmark.

    Provenance
    ----------
    .. [1] Curl of A=(-y f(z)/2, x f(z)/2, 0), f=B0(1+z^2/L^2),
       an exactly divergence-free benchmark.
    """
    _positive(B0=B0, length=length)
    c = B0 / length**2

    def B(x):
        a, b, z = x
        return np.array([-c * a * z, -c * b * z, B0 + c * z * z])

    def J(x):
        a, b, z = x
        return np.array(
            [[-c * z, 0.0, -c * a], [0.0, -c * z, -c * b], [0.0, 0.0, 2 * c * z]]
        )

    return ParticleField(
        B,
        jacobian=J,
        vector_potential=lambda x: (
            np.array([-x[1], x[0], 0.0]) * (B0 + c * x[2] ** 2) / 2
        ),
        contains=lambda x: abs(x[2]) < length and np.hypot(x[0], x[1]) < length / 4,
        name="solenoidal parabolic magnetic mirror",
        provenance={
            "gauge": "A=(-y f/2,x f/2,0)",
            "length_m": length,
            "B_min_T": B0,
            "B0_T": B0,
            "nominal_end_ratio": 2.0,
            "on_axis_arc_coordinate": "z",
        },
    )


def toroidal_particle_field(
    R0=3.0,
    B0=1.0,
    poloidal=0.08,
    *,
    perturbation=0.0,
    mode=6,
    minor_radius=1.0,
    derivative_step=1e-4,
    constant_magnitude=False,
):
    """Construct an axisymmetric torus with an optional curl-defined 3-D perturbation.

    Parameters
    ----------
    R0 : float, optional
        Positive major radius, a benchmark default [m].
    B0 : float, optional
        Reference toroidal field, a benchmark default [T].
    poloidal : float, optional
        Nonnegative coefficient c in psi=c((R-R0)^2+z^2), a benchmark default [T].
    perturbation : float, optional
        Signed perturbation amplitude relative to B0, zero is axisymmetry [-].
    mode : int, optional
        Positive toroidal harmonic number, a benchmark default [-].
    minor_radius : float, optional
        Positive radial/vertical domain extent, less than R0 [m].
    derivative_step : float, optional
        Step retained for independent derivative audits; the benchmark supplies
        an analytic Jacobian. This is a numerical convenience [m].
    constant_magnitude : bool, optional
        Use Bphi=B0 with poloidal=perturbation=0 for a local pure-curvature
        benchmark; default False uses the vacuum F/R component [-].

    Returns
    -------
    ParticleField
        Curl-defined field, vector potential and explicit toroidal domain [-].

    Convention
    ----------
    phi=atan2(y,x), e_phi=(-sin(phi),cos(phi),0). The physical flux per
    radian is R A_phi; no COCOS label is inferred for this Cartesian benchmark.
    A_z=-B0 R0 log(R/R0)+a B0 R0 (R/R0)^n cos(n phi)/n. The perturbation
    represents a controlled vacuum transverse harmonic, not a plasma response.

    Applicability
    -------------
    Machine-independent. Axisymmetric orbit and static symmetry-breaking tests.

    Limitations
    -----------
    The circular current-carrying poloidal field is not a Grad-Shafranov solve.
    Use reconstructed equilibria for machine-specific orbit claims.

    Provenance
    ----------
    .. [1] Analytic curl of the stated vector potential.
    """
    _positive(R0=R0, B0=B0, minor_radius=minor_radius, derivative_step=derivative_step)
    if (
        minor_radius >= R0
        or not np.isfinite(poloidal)
        or poloidal < 0
        or not np.isfinite(perturbation)
    ):
        raise ValueError(
            "require 0 < minor_radius < R0, poloidal >= 0 and finite perturbation"
        )
    if isinstance(mode, bool) or int(mode) != mode or mode < 1:
        raise ValueError("mode must be a positive integer")
    if constant_magnitude and (poloidal != 0 or perturbation != 0):
        raise ValueError("pure-curvature benchmark requires poloidal=perturbation=0")

    def parts(x):
        R = np.hypot(x[0], x[1])
        phi = np.arctan2(x[1], x[0])
        radial = np.array([x[0] / R, x[1] / R, 0.0])
        toroidal = np.array([-x[1] / R, x[0] / R, 0.0])
        return R, phi, radial, toroidal

    def B(x):
        R, phi, radial, toroidal = parts(x)
        amplitude = perturbation * B0 * (R / R0) ** (mode - 1)
        br = -2 * poloidal * x[2] / R - amplitude * np.sin(mode * phi)
        bt = (B0 if constant_magnitude else B0 * R0 / R) - amplitude * np.cos(
            mode * phi
        )
        return (
            br * radial
            + bt * toroidal
            + np.array([0.0, 0.0, 2 * poloidal * (R - R0) / R])
        )

    def J(x):
        R, phi, radial, toroidal = parts(x)
        amp = perturbation * B0 * (R / R0) ** (mode - 1)
        br = -2 * poloidal * x[2] / R - amp * np.sin(mode * phi)
        bt = (B0 if constant_magnitude else B0 * R0 / R) - amp * np.cos(mode * phi)
        br_R = 2 * poloidal * x[2] / R**2 - amp * (mode - 1) / R * np.sin(mode * phi)
        bt_R = (0.0 if constant_magnitude else -B0 * R0 / R**2) - amp * (
            mode - 1
        ) / R * np.cos(mode * phi)
        br_phi = -mode * amp * np.cos(mode * phi)
        bt_phi = mode * amp * np.sin(mode * phi)
        derivative_R = (
            br_R * radial
            + bt_R * toroidal
            + np.array([0.0, 0.0, 2 * poloidal * R0 / R**2])
        )
        derivative_phi = (br_phi - bt) * radial + (br + bt_phi) * toroidal
        derivative_z = -2 * poloidal / R * radial
        return (
            np.outer(derivative_R, radial)
            + np.outer(derivative_phi, toroidal / R)
            + np.outer(derivative_z, [0.0, 0.0, 1.0])
        )

    def A(x):
        R, phi, _, toroidal = parts(x)
        psi = poloidal * ((R - R0) ** 2 + x[2] ** 2)
        az = -B0 * R if constant_magnitude else -B0 * R0 * np.log(R / R0)
        az += perturbation * B0 * R0 / mode * (R / R0) ** mode * np.cos(mode * phi)
        return psi / R * toroidal + np.array([0.0, 0.0, az])

    return ParticleField(
        B,
        jacobian=J,
        vector_potential=A,
        derivative_step=derivative_step,
        contains=lambda x: (
            abs(np.hypot(x[0], x[1]) - R0) < minor_radius and abs(x[2]) < minor_radius
        ),
        name="constant-magnitude curved field"
        if constant_magnitude
        else "analytic toroidal orbit benchmark",
        provenance={
            "gauge": "A_phi=psi/R; A_z specified in toroidal_particle_field",
            "R0_m": R0,
            "B0_T": B0,
            "poloidal_T": poloidal,
            "perturbation": perturbation,
            "mode": int(mode),
            "axisymmetric": perturbation == 0,
            "minor_radius_m": minor_radius,
        },
    )


def equilibrium_particle_field(equilibrium, *, derivative_step=1e-4):
    """Adapt an existing VAFT equilibrium field interpolator to particle tracing.

    Parameters
    ----------
    equilibrium : EquilibriumData
        Validated equilibrium with explicit resolved COCOS convention [-].
    derivative_step : float, optional
        Cartesian field derivative step, a numerical convenience [m].

    Returns
    -------
    ParticleField
        Cartesian field restricted to the equilibrium map domain [-].

    Processing steps
    ----------------
    Reuse the equilibrium interpolator and its COCOS sign/flux handling. Convert
    its (B_R,B_Z,B_phi) components to Cartesian order and enforce map bounds.

    Applicability
    -------------
    Machine-independent. A resolved static equilibrium snapshot.

    Limitations
    -----------
    The underlying interpolator clips F beyond the profile range; this adapter
    does not validate the equilibrium or invent a vector-potential gauge.

    Provenance
    ----------
    .. [1] vaft.process.equilibrium.make_equilibrium_field_interpolator.
    """
    from vaft.process.equilibrium import make_equilibrium_field_interpolator

    if equilibrium.convention.cocos is None:
        raise ValueError("particle equilibrium adapter requires resolved COCOS")
    native = make_equilibrium_field_interpolator(equilibrium)
    r, z = equilibrium.r, equilibrium.z
    if r is None or z is None:
        raise ValueError("equilibrium map axes are required")

    def B(x):
        R = np.hypot(x[0], x[1])
        br, bz, bt = native(R, x[2])
        return np.array([(br * x[0] - bt * x[1]) / R, (br * x[1] + bt * x[0]) / R, bz])

    return ParticleField(
        B,
        derivative_step=derivative_step,
        contains=lambda x: r[0] < np.hypot(x[0], x[1]) < r[-1] and z[0] < x[2] < z[-1],
        name="VAFT equilibrium particle field",
        provenance={
            "cocos": equilibrium.convention.cocos,
            "field_source": "make_equilibrium_field_interpolator; F edge clipping retained",
        },
    )


def filament_particle_field(
    points_xyz, currents_a, *, derivative_step, exclusion_radius
):
    """Adapt existing closed-filament Biot-Savart fields with a matching potential.

    Parameters
    ----------
    points_xyz : sequence
        Closed Cartesian filament polylines, the Biot-Savart native input [m].
    currents_a : array_like
        One current per filament [A].
    derivative_step : float
        Explicit Cartesian numerical derivative step [m].
    exclusion_radius : float
        Positive excluded distance from every straight filament segment [m].

    Returns
    -------
    ParticleField
        Static vacuum field and line-integral vector potential [-].

    Processing steps
    ----------------
    Reuse the existing midpoint-rule Biot-Savart field. Evaluate A with the
    same midpoint quadrature, so curl(A) is exactly the represented B.

    Applicability
    -------------
    Machine-independent. Closed vacuum filaments away from conductor singularities.

    Limitations
    -----------
    No finite conductor radius, plasma response or wall currents are modeled.

    Provenance
    ----------
    .. [1] vaft.process.coils_non_axisymmetric.biot_savart_filaments.
    """
    from vaft.formula.constants import MU0
    from vaft.process.coils_non_axisymmetric import biot_savart_filaments

    _positive(derivative_step=derivative_step, exclusion_radius=exclusion_radius)
    paths = [np.asarray(p, float).copy() for p in points_xyz]
    currents = np.asarray(currents_a, float).copy()
    if (
        len(paths) == 0
        or currents.shape != (len(paths),)
        or not np.all(np.isfinite(currents))
    ):
        raise ValueError("finite currents must match nonempty filament paths")
    for path in paths:
        if (
            path.ndim != 2
            or path.shape[1] != 3
            or len(path) < 4
            or not np.all(np.isfinite(path))
            or not np.array_equal(path[0], path[-1])
        ):
            raise ValueError(
                "filaments must be finite explicitly closed Cartesian polylines"
            )
        if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) == 0):
            raise ValueError("filament segments must have nonzero length")
    starts = np.concatenate([p[:-1] for p in paths])
    dl = np.concatenate([np.diff(p, axis=0) for p in paths])
    lengths = np.linalg.norm(dl, axis=1)
    tangent = dl / lengths[:, None]
    weights = np.concatenate(
        [np.full(len(p) - 1, current) for p, current in zip(paths, currents)]
    )

    def contains(x):
        along = np.sum((x - starts) * tangent, axis=1)
        closest = starts + np.clip(along, 0, lengths)[:, None] * tangent
        return np.all(np.linalg.norm(x - closest, axis=1) > exclusion_radius)

    def A(x):
        midpoints = starts + dl / 2
        distance = np.linalg.norm(x - midpoints, axis=1)
        return MU0 / (4 * np.pi) * np.sum((weights / distance)[:, None] * dl, axis=0)

    return ParticleField(
        lambda x: biot_savart_filaments(paths, currents, np.asarray(x)[None, :])[0],
        vector_potential=A,
        contains=contains,
        derivative_step=derivative_step,
        name="closed-filament vacuum field",
        provenance={
            "gauge": "free-space line-integral A with midpoint quadrature",
            "source": "biot_savart_filaments",
            "exclusion_radius_m": exclusion_radius,
        },
    )
