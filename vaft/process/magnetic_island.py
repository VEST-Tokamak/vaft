"""An analytic magnetic island placed on an axisymmetric equilibrium (issue #886).

The equilibrium is the only source of plasma geometry. Nothing here takes a
major radius, an elongation or a Shafranov shift: the resonant surface comes
from the equilibrium's safety factor, the poloidal angle from its
straight-field-line map, and the island's extent from the flux surfaces'
spacing on the outboard midplane. The island is a prescribed constant-psi
perturbation on top of that, not a solution of anything.

Three objects are kept apart, because they are different physics and a phantom
that conflates them cannot say which one a synthetic signal is sensitive to::

    equilibrium      psi(R, Z), q(psi_N)                       (input)
    island topology  helical flux Omega(R, Z; t, phi)          magnetic_island_topology
    emissivity       epsilon(R, Z; t, phi)                      island_emissivity

The diagnostic response, a line integral of the emissivity, lives in
:mod:`vaft.process.line_of_sight` and :mod:`vaft.process.soft_x_rays`.

Notation
--------
psi_N     : normalized poloidal flux, 0 on axis and 1 on the boundary      [-]
psi_N,s   : psi_N of the resonant surface q = m/n                          [-]
theta*    : PEST straight-field-line poloidal angle                     [rad]
phi       : IMAS toroidal angle, counter-clockwise seen from above      [rad]
xi        : helical phase m theta* - sigma n phi - alpha(t)             [rad]
x         : outboard-midplane distance of a point's flux surface from the
            resonant surface, positive outward                             [m]
W         : full island width at the outboard midplane                    [m]
Omega     : helical flux 8 (x/W)^2 - cos xi, -1 at the O-point, 1 on the
            separatrix                                                     [-]

Conventions
-----------
**W is the full width at the outboard midplane.** ``x`` is a flux label
measured in metres: the outboard-midplane distance between the flux surface
through a point and the resonant surface. On the outboard midplane it is the
exact radial distance, so the separatrix branches through an outboard O-point
are exactly ``W`` apart; elsewhere the island is bounded by the same flux
surfaces, as a constant-psi island is, and its geometric width follows the
flux expansion, wider where the surfaces spread.

**The helical phase is resonant by construction.** ``sigma`` is the sign of
``d phi / d theta*`` along a field line, read from the equilibrium's field
direction, so ``xi`` is constant along every field line on the resonant
surface whichever way the current and the toroidal field point. ``phi`` is the
IMAS toroidal angle, the frame of ``soft_x_rays.channel.line_of_sight.phi``.

**O-point at xi = 0, X-point at xi = pi.** ``alpha(t) = phase +
angular_frequency (t - t0)`` rotates the island rigidly; at fixed ``phi`` the
O-point moves to larger ``theta*`` when ``alpha`` grows. **The sign of
``angular_frequency`` is therefore a poloidal direction, not a toroidal one**:
at fixed ``theta*`` the pattern moves in the IMAS toroidal angle at
``d phi / dt = -angular_frequency / (sigma n)``, so the same spec propagates
the opposite way toroidally when the helicity flips (reversed plasma current
or toroidal field). A caller matching a measured toroidal phase velocity must
fold ``IslandTopology.helicity`` in.

Provenance
----------
.. [1] Choi, G. J., Rev. Mod. Plasma Phys. 8, 11 (2024), for the constant-psi
   island, its normalized helical flux and the helical angle.
.. [2] Escande, D. F. and Momo, B., Rev. Mod. Plasma Phys. 8, 16 (2024), for
   defining island width through flux coordinates rather than a Euclidean
   radius.
.. [3] Grimm, Dewar and Manickam, J. Comput. Phys. 49, 94 (1983), for the
   straight-field-line angle through
   :func:`vaft.process.equilibrium.straight_field_line_map`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from .equilibrium import (
    StraightFieldLineMap,
    as_equilibrium,
    calculate_q_profile_from_psi,
    find_rational_surfaces,
    straight_field_line_map,
)

__all__ = [
    "IslandTopology",
    "MagneticIslandSpec",
    "equilibrium_safety_factor",
    "island_emissivity",
    "magnetic_island_topology",
    "resolve_rational_surface",
]


@dataclass(frozen=True)
class MagneticIslandSpec:
    """The prescribed island: mode numbers, width and phase, nothing geometric.

    ``width`` is the full width at the outboard midplane in metres.
    ``phase`` and ``angular_frequency`` set ``alpha(t) = phase +
    angular_frequency (t - t0)``. ``psi_n_s`` overrides the resonant surface
    for a controlled study; left ``None`` it is found from ``q = m/n``, and
    ``surface_index`` picks among several crossings, innermost first.
    ``helicity`` overrides the sign read from the equilibrium, for a record
    whose convention cannot supply it.
    """

    m: int
    n: int
    width: float
    phase: float = 0.0
    angular_frequency: float = 0.0
    t0: float = 0.0
    psi_n_s: float | None = None
    surface_index: int = 0
    helicity: int | None = None

    def __post_init__(self) -> None:
        if int(self.m) != self.m or self.m < 1:
            raise ValueError(f"m must be a positive integer, not {self.m!r}")
        if int(self.n) != self.n or self.n < 1:
            raise ValueError(
                f"n must be a positive integer, not {self.n!r}; the helicity, not the sign "
                "of n, carries the field-line direction"
            )
        if not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError(f"width must be a positive full width in metres, not {self.width!r}")
        if self.helicity not in (None, 1, -1):
            raise ValueError(f"helicity must be +1, -1 or None, not {self.helicity!r}")
        if self.psi_n_s is not None and not 0.0 < self.psi_n_s < 1.0:
            raise ValueError(f"psi_n_s must lie strictly inside (0, 1), not {self.psi_n_s!r}")

    def alpha(self, time: float) -> float:
        """Island phase ``alpha(t)`` [rad]."""
        return float(self.phase + self.angular_frequency * (float(time) - self.t0))


@dataclass(frozen=True)
class IslandTopology:
    """The helical flux of one island on an equilibrium grid, at one ``(t, phi)``.

    Arrays are indexed ``(R, Z)`` like the flux map, and are NaN (or False)
    outside the equilibrium boundary. ``rephase`` gives the same island at
    another time or toroidal angle without rebuilding the geometry.

    ``normal_displacement`` keeps the issue's name but is a flux label in
    metres, ``R_out(psi) - R_out(psi_s)``: the true normal distance only on the
    outboard midplane. ``grad_psi_s`` is ``|grad psi|`` at the resonant
    surface's outboard-midplane point, in the record's flux unit per metre,
    reported for converting ``W`` to a flux width; ``q_s`` is NaN when the
    surface was placed by ``psi_n_s`` on a record that carries no q.
    """

    spec: MagneticIslandSpec
    r: np.ndarray
    z: np.ndarray
    psi_n: np.ndarray
    inside_boundary: np.ndarray
    psi_n_s: float
    q_s: float
    helicity: int
    grad_psi_s: float
    width_psi_n: float
    time: float
    phi: float
    theta_star: np.ndarray
    normal_displacement: np.ndarray
    helical_phase: np.ndarray
    helical_flux: np.ndarray
    inside_separatrix: np.ndarray
    o_points: np.ndarray
    x_points: np.ndarray
    rational_surface: np.ndarray
    sfl_map: StraightFieldLineMap

    def rephase(self, *, time: float | None = None, phi: float | None = None) -> "IslandTopology":
        """The same island at another time and/or toroidal angle [-]."""
        time = self.time if time is None else float(time)
        phi = self.phi if phi is None else float(phi)
        fields = _phase_fields(self.spec, self.helicity, self.theta_star,
                               self.normal_displacement, time, phi)
        o_pts, x_pts = _o_and_x_points(self.spec, self.helicity, self.rational_surface, time, phi)
        return replace(self, time=time, phi=phi, o_points=o_pts, x_points=x_pts, **fields)


def _coerce(equilibrium):
    from vaft.data.equilibrium import EquilibriumData

    if isinstance(equilibrium, EquilibriumData):
        return equilibrium
    return as_equilibrium(equilibrium)


def _cocos_for_q(eq) -> int:
    convention = eq.convention
    if convention.cocos is not None:
        return int(convention.cocos)
    if convention.psi_per_radian is True:
        return 1
    if convention.psi_per_radian is False:
        return 11
    raise ValueError(
        "this equilibrium carries no q profile and its flux storage family is unknown, so the "
        "2*pi in q cannot be fixed; declare a COCOS index or supply q"
    )


def equilibrium_safety_factor(equilibrium) -> tuple[np.ndarray, np.ndarray]:
    """The magnitude of the safety factor on normalized flux, as the equilibrium knows it.

    Parameters
    ----------
    equilibrium : EquilibriumData or source
        An equilibrium record, or anything
        :func:`vaft.process.equilibrium.as_equilibrium` accepts [-].

    Returns
    -------
    tuple of numpy.ndarray
        ``(psi_n, abs(q))``, sorted by increasing ``psi_n`` [-].

    Raises
    ------
    ValueError
        The record carries neither a q profile nor the flux map and poloidal
        current function to compute one, or its flux storage family is unknown.

    Convention
    ----------
    The record's own profile when it has one; otherwise q is computed from the
    flux map and ``F`` by :func:`calculate_q_profile_from_psi` on 97 interior
    levels, which is the case for an analytic Solov'ev export. Only the
    magnitude is returned: resonance is ``|q| = m/n``, and the direction is the
    helicity's job.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A computed profile starts at ``psi_n = 0.02`` and ends at ``0.98``, so a
    resonance closer to the axis or the boundary than that is not found.

    Provenance
    ----------
    .. [1] :func:`vaft.process.equilibrium.calculate_q_profile_from_psi`, the
       contour-integral safety factor it falls back to.
    """
    eq = _coerce(equilibrium)
    span = eq.psi_boundary - eq.psi_axis
    if eq.q is not None and eq.psi_1d is not None and eq.q.size == eq.psi_1d.size:
        psi_n = (eq.psi_1d - eq.psi_axis) / span
        q = np.abs(eq.q)
    else:
        if eq.psi is None or eq.f is None or eq.psi_1d is None:
            raise ValueError(
                "the equilibrium has no q profile and lacks the flux map or F to compute one"
            )
        psi_n = np.linspace(0.02, 0.98, 97)
        q = np.abs(np.asarray(calculate_q_profile_from_psi(
            eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary, psi_n,
            axis_rz=eq.magnetic_axis, cocos=_cocos_for_q(eq),
        ), dtype=float))
    order = np.argsort(psi_n)
    psi_n, q = psi_n[order], q[order]
    keep = np.isfinite(q)
    return psi_n[keep], q[keep]


def resolve_rational_surface(equilibrium, m: int, n: int, *, index: int = 0) -> tuple[float, float]:
    """Where the equilibrium's own safety factor equals ``m/n``.

    Parameters
    ----------
    equilibrium : EquilibriumData or source
        An equilibrium record, or anything
        :func:`vaft.process.equilibrium.as_equilibrium` accepts [-].
    m : int
        Poloidal mode number [-].
    n : int
        Toroidal mode number [-].
    index : int, optional
        Which crossing when ``q`` crosses ``m/n`` more than once, innermost
        first [-].

    Returns
    -------
    tuple of float
        ``(psi_n_s, q_s)``: the normalized flux of the resonant surface and the
        profile's value there, which equals ``m/n`` to the root tolerance [-].

    Raises
    ------
    ValueError
        ``q = m/n`` does not occur in the profile, or ``index`` asks for a
        crossing that does not exist.

    Convention
    ----------
    Resonance is ``|q| = m/n``. The crossings are bracketed by
    :func:`find_rational_surfaces` and each is refined by a root solve on a
    cubic spline through the profile, so the answer is not limited to the
    linear interpolation between samples.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Only as good as the profile the equilibrium carries; a reconstruction with
    a poorly constrained core q places a core resonance equally poorly.

    Provenance
    ----------
    .. [1] :func:`vaft.process.equilibrium.find_rational_surfaces` for the
       bracketing.
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq

    psi_n, q = equilibrium_safety_factor(equilibrium)
    target = float(m) / float(n)
    found = find_rational_surfaces(psi_n, q, n, m_range=(m, m))
    coarse = np.asarray(found["psi_n_rational"], dtype=float)
    coarse = coarse[np.asarray(found["m"]) == m]
    if coarse.size == 0:
        raise ValueError(
            f"q = {m}/{n} = {target:.4g} does not occur in this equilibrium: |q| spans "
            f"[{np.min(q):.4g}, {np.max(q):.4g}] over psi_N in [{psi_n[0]:.3g}, {psi_n[-1]:.3g}]"
        )
    coarse = np.sort(coarse)
    if not 0 <= index < coarse.size:
        raise ValueError(f"q = {m}/{n} crosses {coarse.size} time(s); index {index} does not exist")
    guess = float(coarse[index])
    unique, first = np.unique(psi_n, return_index=True)
    # anti-alias: not time-domain -- q over normalized flux.
    spline = CubicSpline(unique, q[first])
    j = int(np.clip(np.searchsorted(unique, guess), 1, unique.size - 1))
    lo, hi = unique[j - 1], unique[j]
    f_lo, f_hi = spline(lo) - target, spline(hi) - target
    if f_lo * f_hi > 0.0:
        return guess, float(spline(guess))
    root = float(brentq(lambda s: float(spline(s)) - target, lo, hi, xtol=1e-12))
    return root, float(spline(root))


def _helicity(eq, sfl: StraightFieldLineMap, r_s: float, z_s: float, psi_n_s: float) -> int:
    """Sign of ``d phi / d theta*`` along a field line, in the IMAS toroidal frame.

    ``B_Z = -k dpsi/dR / R`` with ``sign(k) = sigma_RphiZ sigma_Bp``, and the
    IMAS toroidal field is ``sigma_RphiZ F / R``; on the outboard midplane the
    geometric poloidal direction is +Z, so the sign is
    ``-sigma_Bp sign(F) sign(dpsi/dR)`` and ``sigma_RphiZ`` cancels.
    """
    from vaft.data.cocos import cocos_spec

    convention = eq.convention
    if convention.cocos is not None:
        sigma_bp = cocos_spec(int(convention.cocos)).sigma_bp
    else:
        options = {cocos_spec(int(c)).sigma_bp for c in (convention.candidates or ())}
        if len(options) != 1:
            raise ValueError(
                "the equilibrium's COCOS orientation is not identified, so the field-line "
                "helicity cannot be read from it; set MagneticIslandSpec.helicity"
            )
        sigma_bp = options.pop()
    if eq.f is None or eq.psi_1d is None:
        raise ValueError("the equilibrium has no F profile; set MagneticIslandSpec.helicity")
    psi_s = eq.psi_axis + psi_n_s * (eq.psi_boundary - eq.psi_axis)
    order = np.argsort(eq.psi_1d)
    # anti-alias: not time-domain -- F over poloidal flux.
    f_s = float(np.interp(psi_s, eq.psi_1d[order], eq.f[order]))
    dpsi_dr = float(sfl._spline.ev(r_s, z_s, dx=1))
    sign = -sigma_bp * np.sign(f_s) * np.sign(dpsi_dr)
    if sign == 0:
        raise ValueError("F or dpsi/dR vanishes on the resonant surface; set the helicity")
    return int(sign)


def _phase_fields(spec, helicity, theta_star, displacement, time, phi):
    xi = spec.m * theta_star - helicity * spec.n * phi - spec.alpha(time)
    xi = np.mod(xi, 2.0 * np.pi)
    xi = np.where(xi >= 2.0 * np.pi, xi - 2.0 * np.pi, xi)
    omega = 8.0 * (displacement / spec.width) ** 2 - np.cos(xi)
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(omega) & (omega <= 1.0)
    return {"helical_phase": xi, "helical_flux": omega, "inside_separatrix": inside}


def _o_and_x_points(spec, helicity, surface, time, phi):
    theta_s, r_s, z_s = surface
    base = helicity * spec.n * phi + spec.alpha(time)
    k = np.arange(spec.m)
    targets = {
        "o": np.mod((base + 2.0 * np.pi * k) / spec.m, 2.0 * np.pi),
        "x": np.mod((base + np.pi + 2.0 * np.pi * k) / spec.m, 2.0 * np.pi),
    }
    theta_ext = np.concatenate((theta_s - 2.0 * np.pi, theta_s, theta_s + 2.0 * np.pi))
    r_ext = np.tile(r_s, 3)
    z_ext = np.tile(z_s, 3)
    out = []
    for key in ("o", "x"):
        t = targets[key]
        # anti-alias: not time-domain -- surface geometry over the SFL angle.
        out.append(np.column_stack((np.interp(t, theta_ext, r_ext), np.interp(t, theta_ext, z_ext))))
    return out[0], out[1]


def magnetic_island_topology(
    equilibrium,
    island: MagneticIslandSpec,
    *,
    time: float = 0.0,
    phi: float = 0.0,
    sfl_map: StraightFieldLineMap | None = None,
) -> IslandTopology:
    """The helical flux of an ``m/n`` island placed on an equilibrium.

    Parameters
    ----------
    equilibrium : EquilibriumData or source
        The axisymmetric equilibrium, the only source of geometry; anything
        :func:`vaft.process.equilibrium.as_equilibrium` accepts [-].
    island : MagneticIslandSpec
        Mode numbers, outboard-midplane full width and phase [-].
    time : float, optional
        Time at which to evaluate the island phase [s].
    phi : float, optional
        IMAS toroidal angle of the poloidal plane [rad].
    sfl_map : StraightFieldLineMap, optional
        A straight-field-line map already built for this equilibrium, to reuse
        [-].

    Returns
    -------
    IslandTopology
        The resonant surface, the helicity, the flux width, and on the
        equilibrium grid ``theta_star``, ``normal_displacement``,
        ``helical_phase``, ``helical_flux`` and ``inside_separatrix``, with the
        O- and X-point positions [-].

    Raises
    ------
    ValueError
        The resonant surface does not exist, the boundary is not closed on the
        grid, or the helicity cannot be read from the record and was not given.

    Convention
    ----------
    ``Omega = 8 (x/W)^2 - cos(xi)`` with ``xi = m theta* - sigma n phi -
    alpha(t)``: ``Omega = -1`` at the O-point (``xi = 0``, ``x = 0``) and
    ``Omega = 1`` on the separatrix and at the X-point (``xi = pi``), and the
    separatrix branches through an outboard O-point are exactly ``W`` apart.
    ``x = R_out(psi) - R_out(psi_s)``, where ``R_out`` is the outboard-midplane
    radius of a flux surface, so ``x`` is a flux label in metres, exact on the
    outboard midplane and growing outward; ``theta*`` is PEST, zero on the outboard midplane; ``phi``
    is the IMAS toroidal angle. ``sigma`` is the sign of ``d phi / d theta*``
    along a field line from the equilibrium's field, so ``xi`` is constant along
    field lines on the resonant surface.

    Applicability
    -------------
    Machine-independent.

    Processing steps
    ----------------
    1. Build (or reuse) the straight-field-line map of the equilibrium.
    2. Find ``psi_N,s`` from ``q = m/n``, unless the spec overrides it.
    3. Solve the resonant surface; take the helicity from the field direction
       at its outboard-midplane point.
    4. Form the outboard-midplane displacement and the helical phase on the grid, and
       from them the helical flux and the separatrix mask.
    5. Locate the O- and X-points on the resonant surface.

    Limitations
    -----------
    A prescribed constant-psi island: no island-induced change of the
    equilibrium, no finite-width distortion beyond the quadratic flux model,
    and no island that crosses the boundary is treated specially. The
    displacement is first order in the flux offset, so an island wide against
    the local flux-surface curvature is only approximately symmetric.

    Provenance
    ----------
    .. [1] Choi, G. J., Rev. Mod. Plasma Phys. 8, 11 (2024), the constant-psi
       helical flux.
    .. [2] Le, T. X. K. et al., Plasma Phys. Control. Fusion 68, 065003 (2026),
       a rotating-island SXR forward model of the same scope on a spherical
       tokamak.
    """
    eq = _coerce(equilibrium)
    if eq.psi is None or eq.magnetic_axis is None or eq.psi_axis is None or eq.psi_boundary is None:
        raise ValueError("the equilibrium needs a flux map, its axis and boundary flux, and an axis")
    sfl = sfl_map if sfl_map is not None else straight_field_line_map(
        eq.psi, eq.r, eq.z, eq.psi_axis, eq.psi_boundary, eq.magnetic_axis)
    if island.psi_n_s is None:
        psi_n_s, q_s = resolve_rational_surface(eq, island.m, island.n, index=island.surface_index)
    else:
        psi_n_s = float(island.psi_n_s)
        # The override places the island; q there is reported when the record
        # can supply it, and is not a precondition for placing it.
        try:
            prof_psi, prof_q = equilibrium_safety_factor(eq)
        except ValueError:
            q_s = float("nan")
        else:
            # anti-alias: not time-domain -- q over normalized flux.
            q_s = float(np.interp(psi_n_s, prof_psi, prof_q))

    surface = sfl.surface(psi_n_s, n_theta=1024)
    r_out, z_out = float(surface["r"][0]), float(surface["z"][0])
    grad_s = float(surface["grad_psi"][0])
    span = eq.psi_boundary - eq.psi_axis
    helicity = island.helicity if island.helicity is not None else _helicity(
        eq, sfl, r_out, z_out, psi_n_s)

    rr, zz = np.meshgrid(eq.r, eq.z, indexing="ij")
    psi_n = (eq.psi - eq.psi_axis) / span
    theta_star = sfl.theta_star(rr, zz)
    inside = np.isfinite(theta_star)
    outboard = sfl.outboard_radius
    r_s = float(outboard(psi_n_s))
    displacement = np.where(inside, outboard(np.clip(psi_n, 0.0, 1.0)) - r_s, np.nan)
    width_psi_n = float(sfl.psi_norm(r_s + 0.5 * island.width, z_out)
                        - sfl.psi_norm(r_s - 0.5 * island.width, z_out))
    fields = _phase_fields(island, helicity, theta_star, displacement, time, phi)

    theta_s = np.unwrap(surface["theta_star"])
    if theta_s[-1] < theta_s[0]:
        raise ValueError("the straight-field-line angle does not increase along the surface")
    rational = (theta_s, surface["r"], surface["z"])
    o_pts, x_pts = _o_and_x_points(island, helicity, rational, time, phi)
    return IslandTopology(
        spec=island, r=eq.r, z=eq.z, psi_n=psi_n, inside_boundary=inside,
        psi_n_s=float(psi_n_s), q_s=float(q_s), helicity=int(helicity), grad_psi_s=grad_s,
        width_psi_n=width_psi_n, time=float(time), phi=float(phi),
        theta_star=theta_star, normal_displacement=displacement,
        o_points=o_pts, x_points=x_pts, rational_surface=np.vstack(rational), sfl_map=sfl,
        **fields,
    )


def _profile_function(profile) -> Callable[[np.ndarray], np.ndarray]:
    if profile is None:
        return lambda psi_n: np.zeros_like(np.asarray(psi_n, dtype=float))
    if callable(profile):
        return lambda psi_n: np.asarray(profile(np.asarray(psi_n, dtype=float)), dtype=float)
    try:
        grid, values = (np.asarray(part, dtype=float).reshape(-1) for part in profile)
    except (TypeError, ValueError):
        raise ValueError("profile must be a callable of psi_N or a (psi_N, values) pair") from None
    if grid.size != values.size or grid.size < 2:
        raise ValueError("a tabulated profile needs equal-length psi_N and values, at least two")
    order = np.argsort(grid)
    # anti-alias: not time-domain -- emissivity profile over normalized flux.
    return lambda psi_n: np.interp(psi_n, grid[order], values[order])


def island_emissivity(
    topology: IslandTopology,
    *,
    profile: Any = None,
    model: str = "flatten",
    amplitude: float = 1.0,
    smoothing: float = 0.1,
    hard_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """A soft-X-ray emissivity phantom from an island topology.

    Parameters
    ----------
    topology : IslandTopology
        The island on its equilibrium grid, from
        :func:`magnetic_island_topology` [-].
    profile : callable or tuple of array_like, optional
        Axisymmetric background ``epsilon_0(psi_N)``, as a callable or a
        ``(psi_N, values)`` table; required by ``model="flatten"``, zero when
        omitted for ``model="island"`` [emissivity unit].
    model : str, optional
        ``"flatten"`` drives the emissivity inside the island toward its
        resonant-surface value; ``"island"`` adds ``amplitude`` times the island
        mask [-].
    amplitude : float, optional
        ``A``: the flattening fraction for ``"flatten"``, the added emissivity
        for ``"island"`` [- or emissivity unit].
    smoothing : float, optional
        Width of the ``tanh`` edge of the island mask, in helical-flux units [-].
    hard_mask : bool, optional
        Use the binary mask ``Omega <= 1`` instead of the smooth one [-].

    Returns
    -------
    tuple of numpy.ndarray
        ``(epsilon, delta_epsilon)`` on the equilibrium grid, indexed
        ``(R, Z)``, zero outside the boundary; ``delta_epsilon`` is
        ``epsilon - epsilon_0`` [emissivity unit].

    Raises
    ------
    ValueError
        An unknown model, a flattening model without a profile, a non-positive
        smoothing, or a malformed profile table.

    Convention
    ----------
    The mask is ``M = (1 - tanh((Omega - 1)/smoothing)) / 2``, one inside the
    separatrix and zero outside it. ``"island"``: ``delta_epsilon = A M``.
    ``"flatten"``: ``delta_epsilon = A M (epsilon_0(psi_N,s) -
    epsilon_0(psi_N))``, so ``A = 0`` returns the axisymmetric background and
    ``A = 1`` flattens it to the resonant-surface value inside the island.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A phenomenological response, not a transport calculation: the flattening
    is instantaneous and complete to the separatrix, with no dependence on the
    island's size against a critical width, and no atomic or filter physics.

    Provenance
    ----------
    .. [1] Le, T. X. K. et al., Plasma Phys. Control. Fusion 68, 065003 (2026),
       for modelling an island's SXR signature as flattened emissivity.
    .. [2] Dreval, M. B. et al., Plasma Phys. Control. Fusion 65, 035001
       (2023), for a prescribed emissivity perturbation on magnetic surfaces.
    """
    if model not in ("flatten", "island"):
        raise ValueError(f"model must be 'flatten' or 'island', not {model!r}")
    if model == "flatten" and profile is None:
        raise ValueError("the flattening model needs a background profile")
    if not hard_mask and not smoothing > 0.0:
        raise ValueError(f"smoothing must be positive, not {smoothing!r}")
    background_of = _profile_function(profile)
    inside = topology.inside_boundary
    psi_n = np.clip(np.where(inside, topology.psi_n, 0.0), 0.0, 1.0)
    background = np.where(inside, background_of(psi_n), 0.0)
    omega = np.where(inside, topology.helical_flux, np.inf)
    if hard_mask:
        mask = (omega <= 1.0).astype(float)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            mask = np.where(inside, 0.5 * (1.0 - np.tanh((omega - 1.0) / smoothing)), 0.0)
    if model == "island":
        delta = amplitude * mask
    else:
        at_surface = float(background_of(np.array([topology.psi_n_s]))[0])
        delta = amplitude * mask * (at_surface - background)
    delta = np.where(inside, delta, 0.0)
    return background + delta, delta
