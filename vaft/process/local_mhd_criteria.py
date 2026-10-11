"""Local MHD stability criteria evaluated from one reconstructed equilibrium (#1902).

Mercier's ideal-interchange index ``D_I`` and the Glasser-Greene-Johnson
resistive-interchange index ``D_R`` with its geodesic coefficient ``H``, from
flux-surface averages of the equilibrium itself -- no stability code.  They
are the quantities DCON and RDCON report as ``di``, ``dr`` and ``h``, and the
formulas are DCON's (``dcon/mercier.f``, ``mercier_scan``) rewritten free of
its coordinate normalisation, so the two can be compared surface by surface.

    psi(R, Z), F, p, q ─ trace each level ─▶ <1/|grad chi|^2>, <B^2/|grad chi|^2>,
                                             <1/B^2>, <1/(B^2 |grad chi|^2)>, <B^2>, V'
                       ─ GGJ / Mercier ─────▶ D_I, H, D_R = D_I + (H - 1/2)^2

Notation
--------
chi      : total poloidal flux, 2 pi psi with psi per radian, oriented to
           increase outward                                                [Wb]
'        : d/d chi                                                       [1/Wb]
V        : volume enclosed by a surface; V' = closed_integral(R dl / |grad psi|) [m^3/Wb]
F        : |R B_phi|                                                      [T m]
p        : mu0 times the pressure                                          [T^2]
<X>      : volume-weighted flux-surface average, closed_integral(X R dl / |grad psi|) / V'
D_I      : Mercier index; > 0 is ideal-interchange unstable                 [-]
D_R      : GGJ resistive-interchange index; > 0 is resistive-interchange unstable [-]
H        : GGJ geodesic coefficient                                         [-]

Conventions
-----------
**DCON's sign convention.**  ``D_I < 0`` is Mercier stable, the opposite sign
to :func:`vaft.formula.stability.mercier_criterion_circular` and the same as
:func:`vaft.formula.stability.ggj_ideal_interchange_index`.  With
``T = 2 pi F p' V' <1/|grad chi|^2> / q'``::

    D_I = -1/4 + T (1 - T) + p' (V'/q')^2 <B^2/|grad chi|^2>
          * [ p' (<1/B^2> + (2 pi F)^2 <1/(B^2 |grad chi|^2)>) - V''/V' ]
    H   = 2 pi F p' V' / q' * ( <1/|grad chi|^2> - <B^2/|grad chi|^2> / <B^2> )
    D_R = D_I + (H - 1/2)^2

**Orientation.**  The indices are invariant under the COCOS sign choices,
so they are evaluated in DCON's orientation: ``chi`` increases outward, ``F``
and ``q`` are taken by magnitude, and ``p'`` is then negative for a pressure
that falls outward.

**E and F are not separated.**  Only ``E + F`` enters both indices
(``D_I = E + F + H - 1/4``, ``D_R = E + F + H^2``), so it is reported as one
number, ``E_plus_F``.

Provenance
----------
.. [GGJ] A. H. Glasser, J. M. Greene and J. L. Johnson, Phys. Fluids 18, 875 (1975).
.. [DCON] GPEC ``dcon/mercier.f``, ``mercier_scan``: the surface averages and
   the D_I / H / D_R expressions transcribed here.
.. [1902] Issue #1902, equilibrium-native local criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "InterchangeProfile",
    "SurfaceCurvature",
    "curvature_map",
    "flux_surface_curvature",
    "interchange_comparison",
    "mercier_ggj_from_ods",
    "mercier_ggj_profile",
]

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class InterchangeProfile:
    """Mercier / GGJ interchange indices on flux surfaces of one equilibrium."""

    #: Normalized poloidal flux of each evaluated surface [-].
    psi_n: np.ndarray
    #: Mercier ideal-interchange index; > 0 unstable [-].
    D_I: np.ndarray
    #: GGJ resistive-interchange index; > 0 unstable [-].
    D_R: np.ndarray
    #: GGJ geodesic coefficient [-].
    H: np.ndarray
    #: ``E + F`` of GGJ, ``D_I - H + 1/4`` [-].
    E_plus_F: np.ndarray
    #: dV/dchi, chi the total poloidal flux increasing outward [m^3/Wb].
    dV_dchi: np.ndarray
    #: |q| from the flux map on each surface, F/(2 pi) closed(dl/(R |grad psi|)) [-].
    q: np.ndarray
    #: d|q|/dchi on each surface, from the map [1/Wb].
    dq_dchi: np.ndarray
    #: The five surface averages, keyed as in the module notation [SI].
    averages: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mercier_unstable(self) -> np.ndarray:
        """``D_I > 0``: the universal Mercier criterion, nothing more [-]."""
        return self.D_I > 0

    @property
    def resistive_interchange_unstable(self) -> np.ndarray:
        """``D_R > 0``: the universal GGJ criterion [-]."""
        return self.D_R > 0


def mercier_ggj_profile(
    psi_grid,
    R,
    Z,
    psi_axis: float,
    psi_boundary: float,
    levels_norm: Sequence[float],
    *,
    profile_psi_n: Sequence[float],
    f: Sequence[float],
    pressure: Sequence[float],
    q: Sequence[float],
    axis_rz: tuple[float, float],
    min_points: int = 40,
    oversample: int = 2,
    q_from_map: bool = True,
) -> InterchangeProfile:
    """Mercier ``D_I``, GGJ ``H`` and ``D_R`` on flux surfaces of a flux map.

    Parameters
    ----------
    psi_grid : array_like
        Poloidal flux on the grid, indexed ``(R, Z)``, per radian [Wb/rad].
    R : array_like
        Major-radius grid axis [m].
    Z : array_like
        Height grid axis [m].
    psi_axis : float
        Flux on the magnetic axis [Wb/rad].
    psi_boundary : float
        Flux on the boundary [Wb/rad].
    levels_norm : sequence of float
        Normalized flux of the surfaces to evaluate, strictly inside ``(0, 1)``
        and increasing; at least five [-].
    profile_psi_n : sequence of float
        Normalized flux grid of ``f``, ``pressure`` and ``q``, increasing [-].
    f : sequence of float
        ``F = R B_phi`` on ``profile_psi_n``, either sign [T m].
    pressure : sequence of float
        Plasma pressure on ``profile_psi_n`` [Pa].
    q : sequence of float
        Safety factor on ``profile_psi_n``, either sign; used for ``q'`` only
        when ``q_from_map`` is false, and otherwise compared with the map's
        ``q`` as a consistency check [-].
    axis_rz : tuple of float
        Magnetic-axis position, which picks the confined contour [m].
    min_points : int, optional
        Fewest contour vertices for a surface to count as resolved; a
        numerical-resolution control [-].
    oversample : int, optional
        Contours are traced on the flux map refined this many times by its own
        quintic spline; a numerical-resolution control [-].
    q_from_map : bool, optional
        Take ``q`` and ``q'`` from the flux map on each surface, as DCON does,
        instead of differentiating the given profile.  A choice of method: the
        given profile's derivative near a flat-q core is noise-dominated
        (Limitations) [-].

    Returns
    -------
    profile : InterchangeProfile
        The indices, ``E + F``, ``dV/dchi`` and the five averages per level;
        an unresolved level is NaN [-].

    Raises
    ------
    ValueError
        Shapes disagree, the levels are not inside ``(0, 1)`` and increasing,
        fewer than five are given, ``psi_axis`` equals ``psi_boundary``, or
        ``oversample`` is not positive.

    Processing steps
    ----------------
    1. Trace each level and keep the closed contour enclosing the axis.
    2. On it, ``|grad psi|`` from a quintic spline of the flux map and
       ``B^2 = (|grad psi|^2 + F^2) / R^2``; volume-weighted averages.
    3. ``p'`` from a cubic spline of the pressure; ``V''`` and (by default)
       ``q'`` on each surface itself, via ``d/dpsi closed(g dl/|grad psi|) =
       closed(div(g grad psi / |grad psi|^2) dl/|grad psi|)`` with ``g = R``
       and ``1/R`` and second derivatives of the spline, so no difference is
       taken across levels; all per unit ``chi``.
    4. ``D_I``, ``H``, ``D_R`` as in the module notation.

    Convention
    ----------
    DCON's: ``D_I < 0`` and ``D_R < 0`` are stable; ``chi`` increases outward
    and ``F``, ``q`` enter by magnitude, which leaves the indices unchanged
    under COCOS sign choices.  ``pressure`` is in pascal and multiplied by
    ``mu0`` here.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Measured against DCON (mpsi 512, mtheta 1024) with the defaults, D_I on
    a 0.01 grid of levels: the packaged VEST 39915@319 g-file (129 x 129)
    within 3.3 % for psi_N 0.05-0.1, 0.5 % for 0.1-0.3 and 0.1 % beyond; the
    upstream DIII-D-like example (257 x 257) within 0.2 % to psi_N 0.9 and
    3.4 % in its steep pedestal (0.95-0.96).  With ``q_from_map=False`` the
    VEST core error rises to 19 % (0.05-0.1) and 5 % (0.1-0.3).  Near the
    axis the flux gradient vanishes and every average loses accuracy.  DCON
    clamps p >= 0; this uses the pressure as given.  Ideal, static,
    isotropic pressure; no flow.

    Provenance
    ----------
    .. [DCON] GPEC ``dcon/mercier.f`` ``mercier_scan``.
    .. [GGJ] Glasser, Greene and Johnson, Phys. Fluids 18, 875 (1975).
    """
    from scipy.interpolate import CubicSpline, RectBivariateSpline

    from .equilibrium import _closed_contour, _enclosing_segment, extract_flux_surface_contours

    psi = np.asarray(psi_grid, dtype=float)
    r_axis, z_axis = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    if psi.shape != (r_axis.size, z_axis.size):
        raise ValueError(f"psi_grid {psi.shape} is not indexed (R, Z) = ({r_axis.size}, {z_axis.size})")
    levels = np.asarray(levels_norm, dtype=float)
    if levels.size < 5 or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("levels_norm must be at least five increasing values inside (0, 1)")
    if psi_axis == psi_boundary:
        raise ValueError("psi_axis equals psi_boundary")
    x = np.asarray(profile_psi_n, dtype=float)
    profiles = [np.asarray(v, dtype=float) for v in (f, pressure, q)]
    if any(v.shape != x.shape for v in profiles) or np.any(np.diff(x) <= 0):
        raise ValueError("f, pressure and q must lie on the increasing profile_psi_n grid")
    f_spline = CubicSpline(x, np.abs(profiles[0]))
    p_spline = CubicSpline(x, MU0 * profiles[1])
    q_spline = CubicSpline(x, np.abs(profiles[2]))
    flux = RectBivariateSpline(r_axis, z_axis, psi, kx=5, ky=5)
    chi_span = 2.0 * np.pi * abs(psi_boundary - psi_axis)

    if int(oversample) < 1:
        raise ValueError("oversample must be a positive integer")
    r_fine = np.linspace(r_axis[0], r_axis[-1], (r_axis.size - 1) * int(oversample) + 1)
    z_fine = np.linspace(z_axis[0], z_axis[-1], (z_axis.size - 1) * int(oversample) + 1)
    psi_fine = psi if int(oversample) == 1 else flux(r_fine, z_fine)
    contours = extract_flux_surface_contours(psi_fine, r_fine, z_fine, psi_axis, psi_boundary, levels)
    names = ("b2_over_gchi2", "inv_gchi2", "inv_b2", "inv_b2_gchi2", "b2")
    averages = {name: np.full(levels.size, np.nan) for name in names}
    dV = np.full(levels.size, np.nan)
    d2V = np.full(levels.size, np.nan)
    q_surface = np.full(levels.size, np.nan)
    dq_surface = np.full(levels.size, np.nan)
    orientation = np.sign(psi_boundary - psi_axis)
    for i, level in enumerate(levels):
        segment = _enclosing_segment(contours.get(float(level), []), axis_rz, min_points)
        if segment is None:
            continue
        rc, zc = _closed_contour(*segment)
        rm, zm = 0.5 * (rc[1:] + rc[:-1]), 0.5 * (zc[1:] + zc[:-1])
        dl = np.hypot(np.diff(rc), np.diff(zc))
        pr, pz = flux(rm, zm, dx=1, grid=False), flux(rm, zm, dy=1, grid=False)
        prr, pzz, prz = flux(rm, zm, dx=2, grid=False), flux(rm, zm, dy=2, grid=False), flux(rm, zm, dx=1, dy=1, grid=False)
        g2 = pr**2 + pz**2
        grad_psi = np.sqrt(g2)
        b2 = (g2 + float(f_spline(level)) ** 2) / rm**2
        grad_chi2 = (2.0 * np.pi) ** 2 * g2
        weight = rm * dl / grad_psi
        dV[i] = weight.sum()
        # V'' locally, with no difference across levels: d/dpsi closed(R dl/|grad psi|)
        # = closed(div(R grad psi / |grad psi|^2) dl/|grad psi|), the 2-D divergence in (R, Z).
        divergence = (pr * g2 + rm * (prr + pzz) * g2
                      - 2.0 * rm * (pr * pr * prr + 2.0 * pr * pz * prz + pz * pz * pzz)) / g2**2
        d2V[i] = orientation * float(np.sum(divergence * dl / grad_psi)) / (2.0 * np.pi)  # d/dchi = d/dpsi / 2 pi
        # q from the flux map, as DCON does, and q' by the same identity with weight
        # 1/R: q = F/(2 pi) closed(dl/(R |grad psi|)).
        f_here = float(f_spline(level))
        loop = float(np.sum(dl / (rm * grad_psi)))
        div_inv_r = (-pr / rm**2 * g2 + (prr + pzz) / rm * g2
                     - 2.0 / rm * (pr * pr * prr + 2.0 * pr * pz * prz + pz * pz * pzz)) / g2**2
        dloop = orientation * float(np.sum(div_inv_r * dl / grad_psi)) / (2.0 * np.pi)
        q_surface[i] = f_here * loop / (2.0 * np.pi)
        dq_surface[i] = (float(f_spline(level, 1)) / chi_span * loop + f_here * dloop) / (2.0 * np.pi)
        for name, values in zip(names, (b2 / grad_chi2, 1.0 / grad_chi2, 1.0 / b2, 1.0 / (b2 * grad_chi2), b2)):
            averages[name][i] = float(np.sum(values * weight) / dV[i])

    resolved = np.isfinite(dV)
    d_I = np.full(levels.size, np.nan)
    h = np.full(levels.size, np.nan)
    if resolved.any():
        lv = levels[resolved]
        v1 = dV[resolved]
        v2 = d2V[resolved]
        p1 = p_spline(lv, 1) / chi_span
        q1 = dq_surface[resolved] if q_from_map else q_spline(lv, 1) / chi_span
        two_pi_f = 2.0 * np.pi * f_spline(lv)
        a = {name: averages[name][resolved] for name in names}
        term = two_pi_f * p1 * v1 / q1 * a["inv_gchi2"]
        d_I[resolved] = (-0.25 + term * (1.0 - term)
                         + p1 * (v1 / q1) ** 2 * a["b2_over_gchi2"]
                         * (p1 * (a["inv_b2"] + two_pi_f**2 * a["inv_b2_gchi2"]) - v2 / v1))
        h[resolved] = two_pi_f * p1 * v1 / q1 * (a["inv_gchi2"] - a["b2_over_gchi2"] / a["b2"])
    given_q = q_spline(levels)
    q_mismatch = float(np.nanmax(np.abs(q_surface - given_q) / given_q)) if resolved.any() else float("nan")
    return InterchangeProfile(
        psi_n=levels, D_I=d_I, D_R=d_I + (h - 0.5) ** 2, H=h, E_plus_F=d_I - h + 0.25, dV_dchi=dV, q=q_surface, dq_dchi=dq_surface,
        averages=averages,
        provenance={"formulas": "DCON mercier_scan (dcon/mercier.f), coordinate-free", "oversample": int(oversample),
                    "V''": "local surface integral of div(R grad psi / |grad psi|^2)",
                    "q": "from the flux map (as DCON)" if q_from_map else "the given profile, spline derivative",
                    "q_map_vs_given_max_relative": q_mismatch,
                    "orientation": "chi increasing outward, |F|, |q|", "unresolved_levels": int((~resolved).sum())},
    )


def mercier_ggj_from_ods(ods: Any, time_slice: int = 0, *, levels_norm: Sequence[float] | None = None,
                         min_points: int = 40, oversample: int = 2) -> InterchangeProfile:
    """Mercier / GGJ indices of one ODS equilibrium slice.

    Parameters
    ----------
    ods : ODS
        Holding ``equilibrium.time_slice[time_slice]`` with ``profiles_2d[0]``
        ``psi`` on a rectangular grid, ``profiles_1d`` ``psi``, ``f``,
        ``pressure``, ``q``, and the axis and boundary flux and axis position [-].
    time_slice : int, optional
        The slice to read [-].
    levels_norm : sequence of float, optional
        Surfaces to evaluate; by default 0.05 to 0.98 in steps of 0.01, the
        range the reference comparison covers [-].
    min_points : int, optional
        Passed to :func:`mercier_ggj_profile`; a numerical-resolution control [-].
    oversample : int, optional
        Passed to :func:`mercier_ggj_profile`; a numerical-resolution control [-].

    Returns
    -------
    profile : InterchangeProfile
        As :func:`mercier_ggj_profile`, with the slice index and time in its
        provenance [-].

    Raises
    ------
    KeyError
        A required path is missing.

    Processing steps
    ----------------
    1. Read the slice's flux map, profiles and axis, refusing a missing path.
    2. Convert the flux to per radian and the profile grid to normalized flux.
    3. Evaluate with :func:`mercier_ggj_profile` and record the slice and time.

    Input semantics
    ---------------
    The ODS flux is converted to per radian with
    :func:`vaft.data.eqdsk.ods_psi_to_wb_per_radian_factor`, which reads the
    record's own convention.

    Output semantics
    ----------------
    Raw equilibrium-derived indices with no QA cut; comparing with a
    stability code is :func:`interchange_comparison`'s job.

    Convention
    ----------
    As :func:`mercier_ggj_profile`: DCON's sign and orientation.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1902] Issue #1902, evaluation from one reconstructed equilibrium ODS.
    """
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor
    from vaft.machine_mapping.utils import path_exists

    base = f"equilibrium.time_slice.{int(time_slice)}"
    required = ("profiles_2d.0.psi", "profiles_2d.0.grid.dim1", "profiles_2d.0.grid.dim2", "profiles_1d.psi",
                "profiles_1d.f", "profiles_1d.pressure", "profiles_1d.q", "global_quantities.psi_axis",
                "global_quantities.psi_boundary", "global_quantities.magnetic_axis.r",
                "global_quantities.magnetic_axis.z")
    for leaf in required:
        if not path_exists(ods, f"{base}.{leaf}"):
            raise KeyError(f"{base}.{leaf} is missing")
    factor = float(ods_psi_to_wb_per_radian_factor(ods, int(time_slice)))
    axis = float(ods[f"{base}.global_quantities.psi_axis"]) * factor
    boundary = float(ods[f"{base}.global_quantities.psi_boundary"]) * factor
    psi_1d = np.asarray(ods[f"{base}.profiles_1d.psi"], dtype=float) * factor
    psi_n = (psi_1d - axis) / (boundary - axis)
    order = np.argsort(psi_n)
    levels = np.arange(0.05, 0.985, 0.01) if levels_norm is None else np.asarray(levels_norm, dtype=float)
    profile = mercier_ggj_profile(
        np.asarray(ods[f"{base}.profiles_2d.0.psi"], dtype=float) * factor,
        np.asarray(ods[f"{base}.profiles_2d.0.grid.dim1"], dtype=float),
        np.asarray(ods[f"{base}.profiles_2d.0.grid.dim2"], dtype=float),
        axis, boundary, levels,
        profile_psi_n=psi_n[order],
        f=np.asarray(ods[f"{base}.profiles_1d.f"], dtype=float)[order],
        pressure=np.asarray(ods[f"{base}.profiles_1d.pressure"], dtype=float)[order],
        q=np.asarray(ods[f"{base}.profiles_1d.q"], dtype=float)[order],
        axis_rz=(float(ods[f"{base}.global_quantities.magnetic_axis.r"]),
                 float(ods[f"{base}.global_quantities.magnetic_axis.z"])),
        min_points=min_points,
        oversample=oversample,
    )
    time = None
    if path_exists(ods, f"{base}.time"):
        time = float(ods[f"{base}.time"])
    return InterchangeProfile(**{**profile.__dict__, "provenance": {**profile.provenance, "time_slice": int(time_slice),
                                                                     "time": time}})


def interchange_comparison(profile: InterchangeProfile, psi_n: Sequence[float], di: Sequence[float],
                           dr: Sequence[float], *, label: str) -> dict[str, np.ndarray]:
    """Equilibrium-native ``D_I``, ``D_R`` against a stability code's ``di``, ``dr``.

    Parameters
    ----------
    profile : InterchangeProfile
        From :func:`mercier_ggj_profile` [-].
    psi_n : sequence of float
        The code's normalized flux grid [-].
    di : sequence of float
        The code's Mercier index on that grid, DCON's sign [-].
    dr : sequence of float
        The code's resistive-interchange index [-].
    label : str
        Which code and run the comparison is against, kept on the result [-].

    Returns
    -------
    comparison : dict
        On the profile's levels: ``psi_n``, ``D_I``, ``di``, ``D_R``, ``dr``,
        the absolute differences ``dD_I``, ``dD_R``, the code's implied
        ``H`` (``0.5 - sqrt(dr - di)``, the root nearer the equilibrium's), and
        ``label`` [-].

    Raises
    ------
    ValueError
        The code's grid is not increasing or the arrays disagree in length.

    Processing steps
    ----------------
    1. Interpolate the code's ``di``, ``dr`` linearly onto the profile's levels
       inside its own range; NaN outside it.
    2. Differences and the code's implied ``H``, choosing the root of
       ``(H - 1/2)^2 = dr - di`` nearer the equilibrium's ``H``.

    Convention
    ----------
    Both in DCON's sign convention; no QA cut, no verdict.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1902] Issue #1902, comparison with DCON/RDCON on a reference slice.
    """
    x = np.asarray(psi_n, dtype=float)
    di, dr = np.asarray(di, dtype=float), np.asarray(dr, dtype=float)
    if di.shape != x.shape or dr.shape != x.shape or np.any(np.diff(x) <= 0):
        raise ValueError("psi_n must be increasing and di, dr on it")
    lv = profile.psi_n
    inside = (lv >= x[0]) & (lv <= x[-1])
    di_i = np.where(inside, np.interp(lv, x, di), np.nan)
    dr_i = np.where(inside, np.interp(lv, x, dr), np.nan)
    root = np.sqrt(np.clip(dr_i - di_i, 0.0, None))
    h_code = np.where(np.abs(0.5 - root - profile.H) <= np.abs(0.5 + root - profile.H), 0.5 - root, 0.5 + root)
    return {"psi_n": lv, "D_I": profile.D_I, "di": di_i, "D_R": profile.D_R, "dr": dr_i,
            "dD_I": profile.D_I - di_i, "dD_R": profile.D_R - dr_i, "H": profile.H, "H_code": h_code,
            "label": label}


@dataclass(frozen=True)
class SurfaceCurvature:
    """Field-line curvature and local shear along flux surfaces (#1902)."""

    #: Normalized flux of each traced surface [-].
    psi_n: np.ndarray
    #: Per surface: ``R``, ``Z`` [m], ``theta`` (geometric angle about the axis,
    #: from the outboard midplane, counter-clockwise in (R, Z)) [rad], ``dl`` [m],
    #: ``kappa_n``, ``kappa_g`` [1/m], ``local_shear`` [1/m^3], ``B`` [T].
    surfaces: tuple[Mapping[str, np.ndarray], ...]
    #: Per surface, the identity residuals the values must satisfy (Limitations) [-].
    checks: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any] = field(default_factory=dict)


class _OrientedField:
    """B, b, n and their spatial derivatives in one canonical orientation.

    psi is flipped to increase outward and F taken by magnitude, so the signs
    of the geodesic curvature and the local shear do not depend on the
    record's COCOS; the normal curvature does not depend on them at all.
    """

    def __init__(self, psi, r_axis, z_axis, psi_axis, psi_boundary, profile_psi_n, f, pressure):
        from scipy.interpolate import CubicSpline, RectBivariateSpline

        self.sign = float(np.sign(psi_boundary - psi_axis))
        self.axis, self.span = psi_axis, psi_boundary - psi_axis
        self.flux = RectBivariateSpline(r_axis, z_axis, self.sign * np.asarray(psi, dtype=float), kx=5, ky=5)
        self.f = CubicSpline(profile_psi_n, np.abs(np.asarray(f, dtype=float)))
        self.p = CubicSpline(profile_psi_n, MU0 * np.asarray(pressure, dtype=float))
        self.h = 1e-6 * max(float(np.ptp(r_axis)), float(np.ptp(z_axis)))

    def psi_n(self, r, z):
        return (self.sign * self.flux(r, z, grid=False) - self.axis) / self.span

    def at(self, r, z):
        pr, pz = self.flux(r, z, dx=1, grid=False), self.flux(r, z, dy=1, grid=False)
        g = np.hypot(pr, pz)
        B_vec = np.array([-pz / r, self.f(self.psi_n(r, z)) / r, pr / r])  # (R, phi, Z)
        B = np.linalg.norm(B_vec, axis=0)
        return B_vec, B, np.array([pr, 0.0 * pr, pz]) / g, g

    def derivative(self, fun, r, z):
        h = self.h
        return (fun(r + h, z) - fun(r - h, z)) / (2 * h), (fun(r, z + h) - fun(r, z - h)) / (2 * h)

    @staticmethod
    def curl(A, dA_dR, dA_dZ, r):
        return np.array([-dA_dZ[1], dA_dZ[0] - dA_dR[2], (A[1] + r * dA_dR[1]) / r])

    def quantities(self, r, z):
        B_vec, B, n, g = self.at(r, z)
        b = B_vec / B
        db_dR, db_dZ = self.derivative(lambda rr, zz: self.at(rr, zz)[0] / self.at(rr, zz)[1], r, z)
        kappa = np.array([b[0] * db_dR[0] + b[2] * db_dZ[0] - b[1] ** 2 / r,
                          b[0] * db_dR[1] + b[2] * db_dZ[1] + b[0] * b[1] / r,
                          b[0] * db_dR[2] + b[2] * db_dZ[2]])
        binormal = np.cross(b.T, n.T).T
        def h_vec(rr, zz):
            Bv, _, nn, gg = self.at(rr, zz)
            return np.cross(Bv.T, (nn * gg).T).T / gg**2
        h = h_vec(r, z)
        shear = np.sum(h * self.curl(h, *self.derivative(h_vec, r, z), r), axis=0)
        dB2_dR, dB2_dZ = self.derivative(lambda rr, zz: 0.5 * self.at(rr, zz)[1] ** 2, r, z)
        grad_b2 = np.array([dB2_dR, 0.0 * r, dB2_dZ])
        dp_dpsi = self.p(self.psi_n(r, z), 1) / self.span * self.sign
        return {
            "B": B, "grad_psi": g, "kappa_n": np.sum(kappa * n, axis=0), "kappa_g": np.sum(kappa * binormal, axis=0),
            "local_shear": shear,
            "kappa_dot_b": np.sum(kappa * b, axis=0),
            "kappa_n_force_balance": (np.sum(grad_b2 * n, axis=0) + dp_dpsi * self.sign * g) / B**2,
            "kappa_g_force_balance": np.sum(grad_b2 * binormal, axis=0) / B**2,
        }


def flux_surface_curvature(
    psi_grid,
    R,
    Z,
    psi_axis: float,
    psi_boundary: float,
    levels_norm: Sequence[float],
    *,
    profile_psi_n: Sequence[float],
    f: Sequence[float],
    pressure: Sequence[float],
    q: Sequence[float],
    axis_rz: tuple[float, float],
    min_points: int = 40,
    oversample: int = 2,
) -> SurfaceCurvature:
    """Normal and geodesic field-line curvature and local shear along flux surfaces.

    Parameters
    ----------
    psi_grid : array_like
        Poloidal flux on the grid, indexed ``(R, Z)``, per radian [Wb/rad].
    R : array_like
        Major-radius grid axis [m].
    Z : array_like
        Height grid axis [m].
    psi_axis : float
        Flux on the magnetic axis [Wb/rad].
    psi_boundary : float
        Flux on the boundary [Wb/rad].
    levels_norm : sequence of float
        Normalized flux of the surfaces to trace, inside ``(0, 1)`` [-].
    profile_psi_n : sequence of float
        Normalized flux grid of ``f``, ``pressure`` and ``q``, increasing [-].
    f : sequence of float
        ``F = R B_phi``, either sign [T m].
    pressure : sequence of float
        Plasma pressure [Pa].
    q : sequence of float
        Safety factor, either sign; only for the shear-identity check [-].
    axis_rz : tuple of float
        Magnetic-axis position [m].
    min_points : int, optional
        Fewest contour vertices for a resolved surface; a numerical-resolution
        control [-].
    oversample : int, optional
        Contours traced on the flux map refined this many times by its
        quintic spline; a numerical-resolution control [-].

    Returns
    -------
    curvature : SurfaceCurvature
        Per surface the contour, ``kappa_n``, ``kappa_g`` [1/m],
        ``local_shear`` [1/m^3], ``B`` [T], and the identity residuals in
        ``checks`` [-].

    Raises
    ------
    ValueError
        Shapes disagree or a level is outside ``(0, 1)``.

    Processing steps
    ----------------
    1. Orient: psi increasing outward, ``F`` by magnitude.
    2. Trace each level (axis-enclosing closed contour) on the refined map.
    3. At the segment midpoints: ``b``, ``n = grad psi/|grad psi|``,
       ``kappa = b . grad b`` (cylindrical, axisymmetric), ``kappa_n = kappa . n``,
       ``kappa_g = kappa . (b x n)``, ``local_shear = h . curl h`` with
       ``h = B x grad psi / |grad psi|^2``; derivatives by central
       differences of the spline-evaluated fields.
    4. Checks: ``max |kappa . b|``; ``kappa_n`` and ``kappa_g`` against force
       balance, ``(n . grad(B^2/2) + mu0 p' |grad psi|) / B^2`` and
       ``(b x n) . grad(B^2/2) / B^2``; and ``closed(local_shear R dl/|grad psi|)``
       against ``2 pi dq/dpsi`` from the map.

    Convention
    ----------
    ``n`` points outward (psi increasing outward), so ``kappa_n < 0`` -- the
    curvature vector pointing inward, as on the outboard side -- is the
    unfavourable sign for a pressure falling outward.  ``kappa_g`` is along
    ``b x n`` and ``local_shear`` is ``h . curl h`` in the canonical
    orientation (``B_phi > 0``, psi increasing outward); in it
    ``closed(local_shear R dl/|grad psi|) = 2 pi d|q|/dpsi``, the local
    shear whose surface average is the global shear.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Ideal static equilibrium (the force-balance checks assume it).  Near the
    axis the flux gradient vanishes and ``n`` is ill-defined; trace from
    psi_N of about 0.05 on a VEST-sized map.

    Provenance
    ----------
    .. [GJ68] J. M. Greene and J. L. Johnson, Plasma Phys. 10, 729 (1968): local
       shear ``h . curl h`` with ``h = B x grad psi/|grad psi|^2``.
    .. [1902] Issue #1902, normal/geodesic curvature and local shear maps.
    """
    from .equilibrium import _closed_contour, _enclosing_segment, extract_flux_surface_contours

    psi = np.asarray(psi_grid, dtype=float)
    r_axis, z_axis = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    if psi.shape != (r_axis.size, z_axis.size):
        raise ValueError(f"psi_grid {psi.shape} is not indexed (R, Z) = ({r_axis.size}, {z_axis.size})")
    levels = np.asarray(levels_norm, dtype=float)
    if np.any(levels <= 0) or np.any(levels >= 1):
        raise ValueError("levels_norm must lie inside (0, 1)")
    x = np.asarray(profile_psi_n, dtype=float)
    field_ = _OrientedField(psi, r_axis, z_axis, psi_axis, psi_boundary, x, f, pressure)
    r_fine = np.linspace(r_axis[0], r_axis[-1], (r_axis.size - 1) * int(oversample) + 1)
    z_fine = np.linspace(z_axis[0], z_axis[-1], (z_axis.size - 1) * int(oversample) + 1)
    psi_fine = psi if int(oversample) == 1 else field_.sign * field_.flux(r_fine, z_fine)
    contours = extract_flux_surface_contours(psi_fine, r_fine, z_fine, psi_axis, psi_boundary, levels)
    surfaces, checks = [], {name: np.full(levels.size, np.nan) for name in
                            ("kappa_dot_b", "kappa_n_force_balance", "kappa_g_force_balance", "shear_identity")}
    # q' from the map at the same levels, for the shear identity (the profile
    # needs five levels; pad with interior ones when fewer are asked for).
    check_levels = levels if levels.size >= 5 else np.unique(np.r_[levels, 0.5, 0.6, 0.7, 0.8, 0.9])
    profile = mercier_ggj_profile(psi, r_axis, z_axis, psi_axis, psi_boundary, check_levels, profile_psi_n=x, f=f,
                                  pressure=pressure, q=q, axis_rz=axis_rz, min_points=min_points, oversample=oversample)
    for i, level in enumerate(levels):
        segment = _enclosing_segment(contours.get(float(level), []), axis_rz, min_points)
        if segment is None:
            surfaces.append({})
            continue
        rc, zc = _closed_contour(*segment)
        r, z = 0.5 * (rc[1:] + rc[:-1]), 0.5 * (zc[1:] + zc[:-1])
        dl = np.hypot(np.diff(rc), np.diff(zc))
        values = field_.quantities(r, z)
        theta = np.mod(np.arctan2(z - axis_rz[1], r - axis_rz[0]), 2 * np.pi)
        surfaces.append({"R": r, "Z": z, "theta": theta, "dl": dl, **{k: values[k] for k in
                         ("kappa_n", "kappa_g", "local_shear", "B")}})
        scale_n = np.max(np.abs(values["kappa_n"]))
        checks["kappa_dot_b"][i] = float(np.max(np.abs(values["kappa_dot_b"])) / scale_n)
        checks["kappa_n_force_balance"][i] = float(np.max(np.abs(values["kappa_n"] - values["kappa_n_force_balance"])) / scale_n)
        checks["kappa_g_force_balance"][i] = float(np.max(np.abs(values["kappa_g"] - values["kappa_g_force_balance"]))
                                                   / max(np.max(np.abs(values["kappa_g"])), np.finfo(float).tiny))
        match = np.flatnonzero(np.isclose(profile.psi_n, level))
        if match.size and np.isfinite(profile.dq_dchi[match[0]]):
            integral = float(np.sum(values["local_shear"] * r * dl / values["grad_psi"]))
            dq_dpsi = profile.dq_dchi[match[0]] * 2.0 * np.pi  # chi = 2 pi psi
            checks["shear_identity"][i] = abs(integral / (2.0 * np.pi * dq_dpsi) - 1.0)
    return SurfaceCurvature(psi_n=levels, surfaces=tuple(surfaces), checks=checks,
                            provenance={"orientation": "psi increasing outward, B_phi > 0", "oversample": int(oversample)})


def curvature_map(
    psi_grid,
    R,
    Z,
    psi_axis: float,
    psi_boundary: float,
    *,
    profile_psi_n: Sequence[float],
    f: Sequence[float],
    pressure: Sequence[float],
    psi_n_max: float = 0.98,
    psi_n_min: float = 0.02,
) -> dict[str, np.ndarray]:
    """``kappa_n``, ``kappa_g`` and local shear on the (R, Z) grid inside the plasma.

    Parameters
    ----------
    psi_grid : array_like
        Poloidal flux, indexed ``(R, Z)``, per radian [Wb/rad].
    R : array_like
        Major-radius grid axis [m].
    Z : array_like
        Height grid axis [m].
    psi_axis : float
        Flux on the axis [Wb/rad].
    psi_boundary : float
        Flux on the boundary [Wb/rad].
    profile_psi_n : sequence of float
        Normalized flux grid of ``f`` and ``pressure`` [-].
    f : sequence of float
        ``F = R B_phi``, either sign [T m].
    pressure : sequence of float
        Plasma pressure [Pa].
    psi_n_max : float, optional
        Outermost normalized flux kept; a display choice, short of the
        boundary where ``psi_n`` stops labelling closed surfaces [-].
    psi_n_min : float, optional
        Innermost kept; a display choice, outside the axis where ``n`` is
        ill-defined [-].

    Returns
    -------
    maps : dict
        ``R``, ``Z`` (2-D, indexed ``(R, Z)``) [m], ``psi_n`` [-],
        ``kappa_n``, ``kappa_g`` [1/m], ``local_shear`` [1/m^3] and ``B`` [T]; NaN
        outside ``[psi_n_min, psi_n_max]`` [-].

    Raises
    ------
    ValueError
        ``psi_grid`` is not indexed ``(R, Z)``.

    Processing steps
    ----------------
    1. Orient and spline as :func:`flux_surface_curvature`.
    2. Evaluate the same pointwise quantities at every grid node.
    3. Mask nodes outside the requested normalized-flux band.

    Convention
    ----------
    As :func:`flux_surface_curvature`.  The mask is by normalized flux only:
    a private-flux region with psi_n in the band outside the separatrix is
    not removed.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1902] Issue #1902, R-Z maps of curvature and local shear.
    """
    psi = np.asarray(psi_grid, dtype=float)
    r_axis, z_axis = np.asarray(R, dtype=float), np.asarray(Z, dtype=float)
    if psi.shape != (r_axis.size, z_axis.size):
        raise ValueError(f"psi_grid {psi.shape} is not indexed (R, Z) = ({r_axis.size}, {z_axis.size})")
    field_ = _OrientedField(psi, r_axis, z_axis, psi_axis, psi_boundary, np.asarray(profile_psi_n, dtype=float),
                            f, pressure)
    rr, zz = np.meshgrid(r_axis, z_axis, indexing="ij")
    psi_n = field_.psi_n(rr.ravel(), zz.ravel()).reshape(rr.shape)
    inside = (psi_n >= psi_n_min) & (psi_n <= psi_n_max)
    out = {"R": rr, "Z": zz, "psi_n": psi_n}
    names = ("kappa_n", "kappa_g", "local_shear", "B")
    for name in names:
        out[name] = np.full(rr.shape, np.nan)
    if inside.any():
        values = field_.quantities(rr[inside], zz[inside])
        for name in names:
            out[name][inside] = values[name]
    return out

