"""Equilibrium context for core low-n MHD interpretation (#1798, child of #1797).

Descriptors only. :func:`core_q_context` answers *what the equilibrium looks
like* where core n=1 physics lives: q landmarks and the shape of the profile,
every rational crossing with its signed local shear, the q=1 surfaces,
how far q_min sits from low-order rationals, same-helicity (double-resonant)
pairs, low-shear regions under an explicit threshold, the plasma-boundary
topology with q95 and q_boundary kept apart, and the pressure context of those
regions where the profiles exist. It never says that an internal kink,
infernal mode, fishbone, sawtooth or double tearing is present: that needs a
stability calculation (#1635 reduced theory, #1429 DCON/RDCON, #1800 branch
tracking) consuming this context.

Built on the canonical primitives rather than beside them:
:func:`vaft.process.equilibrium.rational_surfaces` for every crossing
(``|q| = |m/n|``, never bridging a NaN gap),
:func:`vaft.formula.equilibrium.shear_from_r_q` for shear, and
:func:`vaft.process.equilibrium.derive_boundary_representation` with
:class:`vaft.data.equilibrium.Topology` for the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_RATIONALS",
    "CoreQContext",
    "RationalCrossing",
    "RationalProximity",
    "DoubleResonantPair",
    "LowShearRegion",
    "NearRationalLowShearRegion",
    "EnclosedPressure",
    "core_q_context",
    "core_q_context_from_profiles",
    "low_shear_regions",
]

#: Convenience default only: low-order core rationals. Not a physics claim that
#: these are the modes that matter; pass ``rationals=`` to ask for others.
DEFAULT_RATIONALS: tuple[tuple[int, int], ...] = ((1, 1), (3, 2), (2, 1), (5, 2), (3, 1))

#: Fraction of the profile's |q| scale below which a sample-to-sample change is
#: treated as flat when reading the profile's shape, so rounding noise in a flat
#: core does not invent extrema. Scaled by max|q|, not by the profile's own range,
#: which on a flat profile is the noise itself.
_SHAPE_RTOL = 1e-6


@dataclass(frozen=True)
class RationalCrossing:
    """One surface where ``|q| = m/n``, with the signed local shear there."""

    m: int
    n: int
    q_target: float
    psi_n: float
    rho: float
    shear: Optional[float]
    root_index: int


@dataclass(frozen=True)
class RationalProximity:
    """How far ``q_min`` sits from ``m/n``: a descriptor, never a surface claim."""

    m: int
    n: int
    q_rational: float
    delta_q: float
    crossing_count: int
    #: ``|delta_q|`` within ``tangency_atol`` and no crossing found: the profile
    #: may just touch ``m/n`` between samples, which a sign test cannot see.
    tangent_candidate: bool

    @property
    def abs_delta_q(self) -> float:
        return abs(self.delta_q)


@dataclass(frozen=True)
class DoubleResonantPair:
    """Two consecutive surfaces of the same ``m/n`` (a reversed-shear configuration).

    An equilibrium fact, not a double-tearing mode: that label needs resistive
    coupling evidence (RDCON Delta-prime, eigenmodes) this context does not have.
    """

    m: int
    n: int
    inner: RationalCrossing
    outer: RationalCrossing

    @property
    def delta_rho(self) -> float:
        return self.outer.rho - self.inner.rho


@dataclass(frozen=True)
class LowShearRegion:
    """One connected radial interval with ``|s| < threshold`` (threshold kept explicit)."""

    threshold: float
    rho_start: float
    rho_end: float
    psi_n_start: float
    psi_n_end: float
    contains_axis: bool
    contains_q_min: bool
    contains_q1_surface: bool
    q_low: float
    q_high: float
    mean_shear: float
    max_abs_shear: float
    #: Drive-context descriptors over the interval, None without a pressure profile.
    pressure_drop: Optional[float] = None
    mean_abs_pressure_gradient: Optional[float] = None
    max_abs_pressure_gradient: Optional[float] = None

    @property
    def width_rho(self) -> float:
        return self.rho_end - self.rho_start


@dataclass(frozen=True)
class NearRationalLowShearRegion:
    """Where ``|s| < shear_threshold`` and ``|q - m/n| < delta_q_threshold`` together.

    Named for what it is, not "infernal region": no stability calculation has
    been made.
    """

    m: int
    n: int
    shear_threshold: float
    delta_q_threshold: float
    rho_start: float
    rho_end: float
    min_abs_delta_q: float
    rho_at_min_abs_delta_q: float
    mean_abs_shear: float

    @property
    def width_rho(self) -> float:
        return self.rho_end - self.rho_start


@dataclass(frozen=True)
class EnclosedPressure:
    """Pressure inside one flux surface: ``integral p dV`` and its volume mean."""

    psi_n: float
    rho: float
    volume: float
    integrated_pressure: float

    @property
    def mean_pressure(self) -> float:
        return self.integrated_pressure / self.volume if self.volume > 0 else math.nan


@dataclass(frozen=True)
class CoreQContext:
    """Equilibrium descriptors for core low-n MHD interpretation at one time slice."""

    time_slice: Optional[int]
    #: Name of the radial coordinate every ``rho`` and every shear uses.
    coordinate: str
    #: Sign of q in the source convention (+1, -1, or 0 for mixed); every q here is |q|.
    q_sign: int
    q_axis: float
    q_min: float
    psi_n_at_q_min: float
    rho_at_q_min: float
    q_min_on_axis: bool
    #: ``monotonic``, ``weak_shear``, ``reversed_shear``, ``multi_extremum`` or ``undetermined``.
    q_profile_topology: str
    shear_sign_changes_rho: tuple[float, ...]
    q95: Optional[float]
    q_boundary: Optional[float]
    q_boundary_reason: str
    #: A :class:`vaft.data.equilibrium.Topology` value, or ``unknown``.
    boundary_topology: str
    boundary_reason: str
    rational_surfaces: tuple[RationalCrossing, ...]
    rational_proximity: tuple[RationalProximity, ...]
    double_resonant_pairs: tuple[DoubleResonantPair, ...]
    low_shear_regions: tuple[LowShearRegion, ...]
    near_rational_low_shear_regions: tuple[NearRationalLowShearRegion, ...]
    #: One entry per q=1 surface, the volume it encloses; empty without a volume profile.
    pressure_inside_q1: tuple[EnclosedPressure, ...]
    pressure_inside_q1_reason: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def q1_surfaces(self) -> tuple[RationalCrossing, ...]:
        return tuple(c for c in self.rational_surfaces if c.q_target == 1.0)

    @property
    def q1_surface_count(self) -> int:
        return len(self.q1_surfaces)

    def as_record(self) -> dict[str, Any]:
        """A JSON-ready summary; non-finite numbers become None."""

        def plain(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if hasattr(value, "__dataclass_fields__"):
                record = {name: plain(getattr(value, name)) for name in value.__dataclass_fields__}
                for name in ("width_rho", "delta_rho", "abs_delta_q", "mean_pressure"):
                    if hasattr(type(value), name):
                        record[name] = plain(getattr(value, name))
                return record
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, (float, np.floating)):
                return float(value) if math.isfinite(value) else None
            return value

        record = plain(self)
        record["q1_surface_count"] = self.q1_surface_count
        return record


def _shear(rho: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    from vaft.formula.equilibrium import shear_from_r_q

    shear = np.full(rho.shape, np.nan)
    finite = np.isfinite(rho) & np.isfinite(q_abs) & (q_abs > 0)
    if finite.sum() >= 2:
        shear[finite] = shear_from_r_q(rho[finite], q_abs[finite])
    return shear


def _interval_edges(x: np.ndarray, inside: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges ``[i, j]`` of consecutive True samples."""
    runs, i = [], 0
    while i < inside.size:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < inside.size and inside[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def low_shear_regions(rho, shear, threshold, *, psi_n=None, q=None, pressure=None) -> tuple[LowShearRegion, ...]:
    """Connected radial intervals where ``|s| < threshold``.

    Parameters
    ----------
    rho : array_like
        Radial coordinate, increasing; the one the shear was taken in [-].
    shear : array_like
        Magnetic shear on ``rho`` [-].
    threshold : float
        The ``|s|`` bound, always explicit: no value of it is a stability boundary [-].
    psi_n : array_like, optional
        Normalized poloidal flux on the same grid, for the ``psi_n`` edges [-].
    q : array_like, optional
        Safety factor on the same grid, for the q range and q=1 containment [-].
    pressure : array_like, optional
        Pressure on the same grid, for the drive-context descriptors [Pa].

    Returns
    -------
    regions : tuple of LowShearRegion
        Every interval, innermost first, at sample resolution [-].

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Intervals end at the last sample inside, without interpolating the
    threshold crossing, so a width is resolved to the grid spacing. An interval
    is closed only by a sample outside it: a non-finite shear sample breaks it.
    ``contains_q_min`` compares against the global ``|q|`` minimum on the grid.
    """
    if not threshold > 0:
        raise ValueError(f"threshold must be positive, not {threshold!r}")
    rho = np.asarray(rho, dtype=float).ravel()
    shear = np.asarray(shear, dtype=float).ravel()
    if rho.shape != shear.shape:
        raise ValueError(f"rho and shear differ in length: {rho.size} vs {shear.size}")
    psi = rho if psi_n is None else np.asarray(psi_n, dtype=float).ravel()
    q_abs = None if q is None else np.abs(np.asarray(q, dtype=float).ravel())
    p = None if pressure is None else np.asarray(pressure, dtype=float).ravel()
    inside = np.isfinite(shear) & (np.abs(shear) < threshold)
    i_min = None if q_abs is None or not np.isfinite(q_abs).any() else int(np.nanargmin(q_abs))
    regions = []
    for i, j in _interval_edges(rho, inside):
        sl = slice(i, j + 1)
        q_lo = q_hi = math.nan
        contains_q1 = False
        if q_abs is not None:
            q_lo, q_hi = float(np.nanmin(q_abs[sl])), float(np.nanmax(q_abs[sl]))
            contains_q1 = bool(q_lo <= 1.0 <= q_hi)
        pressure_fields: dict[str, Optional[float]] = {}
        if p is not None and np.isfinite(p[sl]).all():
            pressure_fields["pressure_drop"] = float(p[i] - p[j])
            if j > i:
                gradient = np.gradient(p[sl], rho[sl])
                pressure_fields["mean_abs_pressure_gradient"] = float(np.mean(np.abs(gradient)))
                pressure_fields["max_abs_pressure_gradient"] = float(np.max(np.abs(gradient)))
        regions.append(LowShearRegion(
            threshold=float(threshold),
            rho_start=float(rho[i]),
            rho_end=float(rho[j]),
            psi_n_start=float(psi[i]),
            psi_n_end=float(psi[j]),
            contains_axis=bool(i == 0),
            contains_q_min=bool(i_min is not None and i <= i_min <= j),
            contains_q1_surface=contains_q1,
            q_low=q_lo,
            q_high=q_hi,
            mean_shear=float(np.mean(shear[sl])),
            max_abs_shear=float(np.max(np.abs(shear[sl]))),
            **pressure_fields,
        ))
    return tuple(regions)


def _profile_topology(rho: np.ndarray, q_abs: np.ndarray, shear: np.ndarray, weak_shear: float):
    finite = np.isfinite(rho) & np.isfinite(q_abs)
    r, q = rho[finite], q_abs[finite]
    if r.size < 3:
        return "undetermined", ()
    dq = np.diff(q)
    flat = np.abs(dq) <= _SHAPE_RTOL * max(float(np.max(np.abs(q))), 1e-300)
    signs = np.sign(dq)[~flat]
    centres = (0.5 * (r[:-1] + r[1:]))[~flat]
    changes = [float(0.5 * (centres[k] + centres[k + 1])) for k in range(signs.size - 1) if signs[k] != signs[k + 1]]
    if signs.size == 0:
        return "weak_shear", ()
    if not changes:
        s = shear[finite]
        core = np.isfinite(s)
        if core.any() and np.nanmax(np.abs(s[core])) < weak_shear:
            return "weak_shear", ()
        return "monotonic", ()
    if len(changes) == 1 and signs[0] < 0 < signs[-1]:
        return "reversed_shear", tuple(changes)
    return "multi_extremum", tuple(changes)


def _near_rational_regions(rho, q_abs, shear, rationals, shear_threshold, dq_threshold):
    out = []
    for m, n in rationals:
        target = abs(m / n)
        delta = np.abs(q_abs - target)
        inside = np.isfinite(shear) & (np.abs(shear) < shear_threshold) & np.isfinite(delta) & (delta < dq_threshold)
        for i, j in _interval_edges(rho, inside):
            sl = slice(i, j + 1)
            k = i + int(np.argmin(delta[sl]))
            out.append(NearRationalLowShearRegion(
                m=int(m), n=int(n),
                shear_threshold=float(shear_threshold), delta_q_threshold=float(dq_threshold),
                rho_start=float(rho[i]), rho_end=float(rho[j]),
                min_abs_delta_q=float(delta[k]), rho_at_min_abs_delta_q=float(rho[k]),
                mean_abs_shear=float(np.mean(np.abs(shear[sl]))),
            ))
    return tuple(out)


def core_q_context_from_profiles(
    psi_n,
    q,
    *,
    rho_tor_norm=None,
    pressure=None,
    volume=None,
    rationals: Sequence[tuple[int, int]] = DEFAULT_RATIONALS,
    shear_thresholds: Sequence[float] = (0.1,),
    near_rational_delta_q: float = 0.1,
    weak_shear: float = 0.1,
    tangency_atol: float = 1e-3,
    boundary_topology: str = "unknown",
    boundary_reason: str = "not supplied",
    time_slice: Optional[int] = None,
) -> CoreQContext:
    """Core q-profile, rational-surface, low-shear and boundary context from 1-D profiles.

    Parameters
    ----------
    psi_n : array_like
        Normalized poloidal flux, increasing, 0 on axis and 1 at the boundary [-].
    q : array_like
        Safety factor on ``psi_n``, of either sign [-].
    rho_tor_norm : array_like, optional
        Normalized toroidal-flux radius on the same grid; without it every
        radius and shear is in ``sqrt(psi_n)`` and named so [-].
    pressure : array_like, optional
        Pressure on the same grid, for the region drive context [Pa].
    volume : array_like, optional
        Volume enclosed by each surface, for the pressure inside q=1 [m^3].
    rationals : sequence of (int, int), optional
        The ``(m, n)`` to locate and measure q_min against [-].
    shear_thresholds : sequence of float, optional
        Each ``|s|`` bound to report low-shear regions for; a scan, not one
        universal value [-].
    near_rational_delta_q : float, optional
        The ``|q - m/n|`` bound of the near-rational low-shear regions [-].
    weak_shear : float, optional
        ``|s|`` below which a profile with no extremum is called ``weak_shear``
        rather than ``monotonic`` [-].
    tangency_atol : float, optional
        ``|q_min - m/n|`` within which an uncrossed rational is flagged as a
        possible tangency between samples [-].
    boundary_topology : str, optional
        Boundary topology already known to the caller [-].
    boundary_reason : str, optional
        Why that topology was, or could not be, decided [-].
    time_slice : int, optional
        The slice index to record [-].

    Returns
    -------
    context : CoreQContext
        The descriptors, with every threshold and the radial coordinate
        recorded [-].

    Raises
    ------
    ValueError
        The profiles are not 1-D and of equal length, fewer than three samples
        are finite, ``psi_n`` does not increase, or a threshold is not positive.

    Processing steps
    ----------------
    1. Take ``|q|`` (resonance is ``|q| = |m/n|``) and record the sign the
       source gave it.
    2. Choose the radius (``rho_tor_norm`` or ``sqrt(psi_n)``) and take the
       shear ``(rho/|q|) d|q|/drho`` in it.
    3. Locate q_axis, q_min and its place, and read the profile's shape from
       the sign changes of ``d|q|``.
    4. Locate every rational crossing with :func:`rational_surfaces`, attach
       its interpolated shear, pair same-``m/n`` neighbours, and measure
       ``q_min - m/n`` for every requested rational.
    5. Find the low-shear and near-rational low-shear intervals for each
       threshold, and the pressure enclosed by each q=1 surface when a volume
       profile is given.

    Input semantics
    ---------------
    Profiles of one equilibrium slice on one grid; nothing is fabricated for a
    profile that is not given.

    Output semantics
    ----------------
    Descriptors of the equilibrium, not of a mode: no field says an
    instability is present.

    Convention
    ----------
    Every q is ``|q|``; ``q_sign`` keeps the source's sign. Shear is
    ``d ln|q| / d ln rho`` in the recorded coordinate, signed, so the two
    surfaces of a reversed-shear pair have opposite signs. ``q_axis`` is
    ``|q|`` at the first sample and ``q_min`` the smallest ``|q|``: they are
    equal only for a monotonic profile.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Linear interpolation between samples, as :func:`rational_surfaces`: a
    rational touched between two samples is not a crossing and appears only as
    ``tangent_candidate``. Interval edges are at sample resolution. q95 is
    interpolated at ``psi_n = 0.95``; ``q_boundary`` is reported only for a
    ``limited`` boundary, since on a separatrix it is not a finite boundary q.

    Provenance
    ----------
    .. [1798] Issue #1798, core q / rational / low-shear / boundary context.
    """
    from vaft.process.equilibrium import rational_surfaces

    psi = np.asarray(psi_n, dtype=float).ravel()
    q_raw = np.asarray(q, dtype=float).ravel()
    if psi.shape != q_raw.shape:
        raise ValueError(f"psi_n and q differ in length: {psi.size} vs {q_raw.size}")
    finite = np.isfinite(psi) & np.isfinite(q_raw)
    if finite.sum() < 3:
        raise ValueError("at least three finite samples are needed")
    if np.any(np.diff(psi[finite]) <= 0):
        raise ValueError("psi_n must increase")
    for value in (*shear_thresholds, near_rational_delta_q, weak_shear, tangency_atol):
        if not value > 0:
            raise ValueError(f"thresholds must be positive, got {value!r}")

    signs = np.sign(q_raw[finite])
    q_sign = int(signs[0]) if np.all(signs == signs[0]) else 0
    q_abs = np.abs(q_raw)
    if rho_tor_norm is not None:
        rho = np.asarray(rho_tor_norm, dtype=float).ravel()
        if rho.shape != psi.shape:
            raise ValueError("rho_tor_norm is not on the psi_n grid")
        coordinate = "rho_tor_norm"
    else:
        rho = np.sqrt(np.clip(psi, 0.0, None))
        coordinate = "rho_pol_norm (sqrt psi_n)"
    shear = _shear(rho, q_abs)

    i_min = int(np.nanargmin(np.where(finite, q_abs, np.nan)))
    i_axis = int(np.flatnonzero(finite)[0])
    topology, changes = _profile_topology(rho, q_abs, shear, weak_shear)

    surfaces = rational_surfaces(
        psi, q_raw, resonances=tuple((int(m), int(n)) for m, n in rationals),
        rho_tor_norm=None if rho_tor_norm is None else rho,
    )
    crossings: list[RationalCrossing] = []
    pairs: list[DoubleResonantPair] = []
    by_target: dict[float, list[RationalCrossing]] = {}
    for surface in surfaces:
        m, n = surface.harmonics[0]
        for root in surface.roots:
            r = root.rho_tor_norm if rho_tor_norm is not None else root.rho_pol_norm
            r = float(np.interp(root.psi_norm, psi[finite], rho[finite])) if r is None else float(r)
            ok = np.isfinite(shear)
            s = float(np.interp(root.psi_norm, psi[ok], shear[ok])) if ok.sum() >= 2 else None
            crossing = RationalCrossing(m=abs(int(m)), n=abs(int(n)), q_target=float(surface.q_target),
                                        psi_n=float(root.psi_norm), rho=r, shear=s, root_index=root.root_index)
            crossings.append(crossing)
            by_target.setdefault(float(surface.q_target), []).append(crossing)
    for target, items in by_target.items():
        for inner, outer in zip(items, items[1:]):
            pairs.append(DoubleResonantPair(m=inner.m, n=inner.n, inner=inner, outer=outer))

    q_min = float(q_abs[i_min])
    proximity = []
    for m, n in rationals:
        target = abs(m / n)
        count = len(by_target.get(float(target), []))
        delta = q_min - target
        proximity.append(RationalProximity(
            m=int(m), n=int(n), q_rational=float(target), delta_q=float(delta), crossing_count=count,
            tangent_candidate=bool(count == 0 and abs(delta) <= tangency_atol),
        ))

    regions: list[LowShearRegion] = []
    near: list[NearRationalLowShearRegion] = []
    for threshold in shear_thresholds:
        regions.extend(low_shear_regions(rho, shear, threshold, psi_n=psi, q=q_abs, pressure=pressure))
        near.extend(_near_rational_regions(rho, q_abs, shear, rationals, threshold, near_rational_delta_q))

    q95 = None
    if psi[finite][0] <= 0.95 <= psi[finite][-1]:
        q95 = float(np.interp(0.95, psi[finite], q_abs[finite]))
    if boundary_topology == "limited":
        q_boundary = float(q_abs[np.flatnonzero(finite)[-1]])
        q_boundary_reason = "limited boundary: |q| at the last closed surface"
    else:
        q_boundary = None
        q_boundary_reason = (
            "not reported: a separatrix q is not a finite boundary q"
            if boundary_topology not in ("unknown", "ambiguous")
            else f"not reported: boundary topology is {boundary_topology}"
        )

    enclosed: list[EnclosedPressure] = []
    if volume is None or pressure is None:
        enclosed_reason = "no volume profile" if volume is None else "no pressure profile"
    else:
        v = np.asarray(volume, dtype=float).ravel()
        p = np.asarray(pressure, dtype=float).ravel()
        ok = finite & np.isfinite(v) & np.isfinite(p)
        if v.shape != psi.shape or p.shape != psi.shape or ok.sum() < 2:
            enclosed_reason = "volume or pressure not usable on the psi_n grid"
        else:
            enclosed_reason = "integral of p dV from the axis to each q=1 surface (trapezoid)"
            cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (p[ok][1:] + p[ok][:-1]) * np.diff(v[ok]))))
            for crossing in (c for c in crossings if c.q_target == 1.0):
                enclosed.append(EnclosedPressure(
                    psi_n=crossing.psi_n, rho=crossing.rho,
                    volume=float(np.interp(crossing.psi_n, psi[ok], v[ok])),
                    integrated_pressure=float(np.interp(crossing.psi_n, psi[ok], cumulative)),
                ))

    return CoreQContext(
        time_slice=time_slice,
        coordinate=coordinate,
        q_sign=q_sign,
        q_axis=float(q_abs[i_axis]),
        q_min=q_min,
        psi_n_at_q_min=float(psi[i_min]),
        rho_at_q_min=float(rho[i_min]),
        q_min_on_axis=bool(i_min == i_axis),
        q_profile_topology=topology,
        shear_sign_changes_rho=changes,
        q95=q95,
        q_boundary=q_boundary,
        q_boundary_reason=q_boundary_reason,
        boundary_topology=str(boundary_topology),
        boundary_reason=str(boundary_reason),
        rational_surfaces=tuple(crossings),
        rational_proximity=tuple(proximity),
        double_resonant_pairs=tuple(pairs),
        low_shear_regions=tuple(regions),
        near_rational_low_shear_regions=tuple(near),
        pressure_inside_q1=tuple(enclosed),
        pressure_inside_q1_reason=enclosed_reason,
        provenance={
            "coordinate": coordinate,
            "shear": "d ln|q| / d ln rho in the recorded coordinate (vaft.formula.equilibrium.shear_from_r_q)",
            "rational_surfaces": "vaft.process.equilibrium.rational_surfaces, |q| = |m/n|, linear interpolation",
            "rationals": [list(pair) for pair in rationals],
            "shear_thresholds": [float(t) for t in shear_thresholds],
            "near_rational_delta_q": float(near_rational_delta_q),
            "weak_shear": float(weak_shear),
            "tangency_atol": float(tangency_atol),
            "q95": "|q| interpolated at psi_n = 0.95",
        },
    )


def core_q_context(ods: Any, time_slice: int = 0, **options: Any) -> CoreQContext:
    """Core q / rational / low-shear / boundary context of one ODS equilibrium slice.

    Parameters
    ----------
    ods : ODS
        Holding ``equilibrium.time_slice[time_slice].profiles_1d`` ``psi`` and
        ``q``, and ``global_quantities`` ``psi_axis``/``psi_boundary`` [-].
    time_slice : int, optional
        The slice to read; any further keyword (``rationals``,
        ``shear_thresholds``, ...) is passed to
        :func:`core_q_context_from_profiles` [-].

    Returns
    -------
    context : CoreQContext
        The descriptors of that slice [-].

    Raises
    ------
    KeyError
        The slice has no ``profiles_1d.psi`` or ``q``, or no axis/boundary flux.

    Processing steps
    ----------------
    1. Read ``psi``, ``q`` and, where present, ``rho_tor_norm``, ``pressure``
       and ``volume`` from ``profiles_1d``; normalize ``psi`` by the axis and
       boundary flux.
    2. Classify the boundary with :func:`derive_boundary_representation`; an
       equilibrium it cannot classify is ``unknown`` with the reason.
    3. Hand everything to :func:`core_q_context_from_profiles`.

    Input semantics
    ---------------
    One reconstructed equilibrium slice; read without creating ODS paths.

    Output semantics
    ----------------
    Descriptors of that equilibrium; no mode label.

    Convention
    ----------
    As :func:`core_q_context_from_profiles`. ``psi_n = (psi - psi_axis) /
    (psi_boundary - psi_axis)``, which is COCOS-independent.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    A profile missing from the slice is omitted, never synthesized: without
    ``volume`` there is no enclosed pressure, without ``pressure`` no drive
    context.

    Provenance
    ----------
    .. [1798] Issue #1798, core q / rational / low-shear / boundary context.
    """
    from vaft.machine_mapping.utils import path_exists

    base = f"equilibrium.time_slice.{int(time_slice)}"
    for leaf in ("profiles_1d.psi", "profiles_1d.q", "global_quantities.psi_axis", "global_quantities.psi_boundary"):
        if not path_exists(ods, f"{base}.{leaf}"):
            raise KeyError(f"{base}.{leaf} is missing")
    psi = np.asarray(ods[f"{base}.profiles_1d.psi"], dtype=float)
    axis, boundary = float(ods[f"{base}.global_quantities.psi_axis"]), float(ods[f"{base}.global_quantities.psi_boundary"])
    if boundary == axis:
        raise ValueError("psi_axis equals psi_boundary")
    psi_n = (psi - axis) / (boundary - axis)

    def optional(name: str):
        path = f"{base}.profiles_1d.{name}"
        if not path_exists(ods, path):
            return None
        value = np.asarray(ods[path], dtype=float)
        return value if value.shape == psi.shape else None

    try:
        from vaft.process._equilibrium_parametric import as_equilibrium, derive_boundary_representation

        representation = derive_boundary_representation(as_equilibrium(ods, time_index=int(time_slice)))
        topology = getattr(representation.topology, "value", str(representation.topology))
        reason = representation.reason or "derive_boundary_representation (flux map)"
    except Exception as exc:  # no psi map, no wall: an unclassified boundary, not an error
        topology, reason = "unknown", f"boundary not classified: {type(exc).__name__}: {exc}"

    options.setdefault("boundary_topology", topology)
    options.setdefault("boundary_reason", reason)
    return core_q_context_from_profiles(
        psi_n,
        np.asarray(ods[f"{base}.profiles_1d.q"], dtype=float),
        rho_tor_norm=optional("rho_tor_norm"),
        pressure=optional("pressure"),
        volume=optional("volume"),
        time_slice=int(time_slice),
        **options,
    )
