"""Resolved transport states: one qualified plasma state shared by TGLF, NEO and classical.

Routine transport analysis (issues #1428, #1431, #1435) asks the same upstream
question before any solver runs: *which* equilibrium and kinetic-profile slice
describe this instant, where did each kinetic quantity come from, and is the result
fit to hand to a model?  This module answers it once, so a TGLF and a NEO result
that are later compared describe the same plasma rather than two independently
resolved ones.

The chain::

    ODS (equilibrium + core_profiles)
    -> equilibrium slice at the state time, core_profiles slice within dt   [times]
    -> ion temperature: measured -> pressure-inferred (#1426) -> policy      [ti]
    -> composition closure via prepare_gacode_profile(impurity=, z_eff=)     [GACODEProfile]
    -> ResolvedTransportState (profile + per-quantity provenance + identity)
    -> solver readiness: assess_tglf_readiness / assess_neo_readiness

Nothing here runs a solver.  Execution stays with :mod:`vaft.code.gacode`; this
layer only decides what is run and records why.

Conventions
-----------
``time_efit_s`` is the equilibrium slice time and is the state's time.  The
core_profiles slice is matched to it **by time** within an explicit tolerance, never
by index.  ``r_over_a`` is GACODE's ``rmin/rmin[-1]`` on the converted profile, whose
outer edge is ``rho_max``; it is not ``rho_tor_norm``.  The ODS passed in is never
modified: the ion-temperature policy writes into a deep copy.

Notes
-----
The default ion-temperature policy and composition (H+ with C6+ at Z_eff = 2) are
VEST's, read through :mod:`vaft.machine_mapping.core_profiles`.  The mechanics are
machine-independent and every policy value is an argument.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_RHO_MAX",
    "DEFAULT_SURFACES",
    "DEFAULT_TIME_TOLERANCE_S",
    "EFIT_QUALITIES",
    "EFIT_LINEAGES",
    "RESOLVER_VERSION",
    "ReadinessReport",
    "ResolvedTransportState",
    "SurfaceReadiness",
    "TransportStateKey",
    "assess_neo_readiness",
    "assess_tglf_readiness",
    "inferred_ti_supported",
    "CLASSICAL_MODEL",
    "classical_heat_fluxes",
    "surface_toroidal_field",
    "physics_parameters",
    "resolve_transport_state",
    "run_identity",
    "transport_partition",
]

#: Bumped whenever a change here can alter a resolved state; part of every identity.
RESOLVER_VERSION = 2

#: The two EFIT lineages of lane K's State key contract v1 (#1454).
EFIT_LINEAGES = ("magnetics", "electron_kinetic")

#: Only these #1331 slice labels are resolved; an unreconstructible slice is refused.
EFIT_QUALITIES = ("good", "admissible")

#: The routine radial scan, as r/a.  Inside the Thomson span, clear of the axis where
#: VEST's hollow Te makes gradients ill-conditioned, at or below 0.8 so it sits well
#: inside ``DEFAULT_RHO_MAX``, and evenly spaced so NEO's ``N_RADIAL = 6`` between
#: 0.3 and 0.8 lands on exactly the same surfaces.
DEFAULT_SURFACES = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

#: The campaign core_profiles stage pairs its slices to EFIT within 0.5 ms
#: (``equilibrium_tolerance_ms`` in its manifest); the same window is used here.
DEFAULT_TIME_TOLERANCE_S = 5e-4

#: Outer edge of the converted GACODE profile, as rho_tor_norm (the reference notebooks').
DEFAULT_RHO_MAX = 0.95

#: Hydrogen, the VEST main ion written when core_profiles carries none.
_HYDROGEN = {"label": "H+", "z": 1.0, "a": 1.00794}


@dataclass(frozen=True)
class TransportStateKey:
    """Which plasma state: ``(shot, time_efit_s, efit_lineage)``.

    Lane K's State key contract v1 (#1454): ``time_efit_s`` is the EFIT slice time
    rounded to 1e-4 s, and ``efit_lineage`` is ``magnetics`` or ``electron_kinetic``.
    """

    shot: int
    time_efit_s: float
    efit_lineage: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_efit_s", round(float(self.time_efit_s), 4))
        if self.efit_lineage not in EFIT_LINEAGES:
            raise ValueError(
                f"efit_lineage must be one of {EFIT_LINEAGES}; got {self.efit_lineage!r}"
            )

    @property
    def time_ms(self) -> int:
        """The slice time in whole milliseconds, the #1331 label table's key [ms]."""
        return int(round(self.time_efit_s * 1e3))

    def as_dict(self) -> dict[str, Any]:
        return {
            "shot": int(self.shot),
            "time_efit_s": float(self.time_efit_s),
            "efit_lineage": self.efit_lineage,
        }

    def slug(self) -> str:
        """A path-safe name: ``<shot>/<lineage>/<time_ms>``."""
        return f"{int(self.shot)}/{self.efit_lineage}/{self.time_ms:05d}"


@dataclass
class ResolvedTransportState:
    """One plasma state ready to be handed to a transport model, or the reasons it is not.

    ``profile`` is the shared :class:`~vaft.code.gacode._profiles.GACODEProfile`
    (``None`` when ``status`` is ``"insufficient"``).  ``provenance`` names, per
    quantity, whether it is measured, reconstructed, inferred, assumed or derived.
    """

    key: TransportStateKey
    efit_quality: str
    quality_source: str
    status: str
    reasons: tuple[str, ...] = ()
    profile: Any = None
    times: Mapping[str, Any] = field(default_factory=dict)
    ti: Mapping[str, Any] = field(default_factory=dict)
    composition: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: The same state with the inferred-Ti gaps filled at a different ratio: a surface
    #: whose solver input changes between the two depends on the fill (not identity).
    fill_probe: Any = field(default=None, repr=False, compare=False)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def ti_lineage(self) -> str:
        return str(self.ti.get("lineage", "unresolved"))

    def identity_payload(self) -> dict[str, Any]:
        """Everything that, changed, makes this a different state.

        The resolved profile's own arrays are hashed, so a different input file, a
        different #1426 temperature or a change in the GACODE converter all change
        the identity without anyone bumping a version.  Paths and the Ti sigma
        (provenance only) are left out: moving a file or restating an uncertainty
        does not change the physics a solver is given.
        """
        inputs = {
            name: (value.get("sha256") if isinstance(value, Mapping) else value)
            for name, value in self.inputs.items()
        }
        return {
            "resolver_version": RESOLVER_VERSION,
            "key": self.key.as_dict(),
            "times": dict(self.times),
            "ti": {k: self.ti.get(k) for k in ("lineage", "method", "ratio", "temperature_sha256")},
            "composition": dict(self.composition),
            "inputs": inputs,
            "settings": dict(self.settings),
            "profile_sha256": _profile_digest(self.profile),
        }

    @property
    def identity(self) -> str:
        """sha256 of :meth:`identity_payload` (the upstream half of every run identity)."""
        return _digest(self.identity_payload())

    def summary(self) -> dict[str, Any]:
        """A JSON-ready record of the state without the profile arrays."""
        return {
            **self.key.as_dict(),
            "efit_quality": self.efit_quality,
            "quality_source": self.quality_source,
            "status": self.status,
            "reasons": list(self.reasons),
            "ti_lineage": self.ti_lineage,
            "times": dict(self.times),
            "ti": dict(self.ti),
            "composition": dict(self.composition),
            "provenance": _jsonable(self.provenance),
            "inputs": dict(self.inputs),
            "settings": dict(self.settings),
            "state_identity": self.identity,
        }


@dataclass
class SurfaceReadiness:
    """One requested surface: ``status`` is ``"ready"`` or a reason code."""

    r_over_a: float
    status: str
    detail: str = ""
    local_input: Any = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"


@dataclass
class ReadinessReport:
    """A solver's verdict on one state: ``ready``, ``conditional`` or ``insufficient``.

    ``conditions`` are the declared assumptions a ``conditional`` state rests on;
    ``reasons`` are why an ``insufficient`` one cannot run.
    """

    solver: str
    status: str
    reasons: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    surfaces: tuple[SurfaceReadiness, ...] = ()

    @property
    def runnable(self) -> bool:
        return self.status in ("ready", "conditional")

    def summary(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "status": self.status,
            "reasons": list(self.reasons),
            "conditions": list(self.conditions),
            "surfaces": [
                {"r_over_a": s.r_over_a, "status": s.status, "detail": s.detail}
                for s in self.surfaces
            ],
        }


# --------------------------------------------------------------------------- helpers


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


#: The GACODEProfile arrays a solver input is built from.
_PROFILE_FIELDS = (
    "rho", "rmin", "rmaj", "zmag", "q", "kappa", "delta", "zeta", "polflux", "ne", "te",
    "ni", "ti", "z", "mass", "masse", "z_eff", "vtor", "torfluxa", "rcentr", "bcentr",
    "current",
)


def _profile_digest(profile: Any) -> Optional[str]:
    if profile is None:
        return None
    digest = hashlib.sha256()
    for name in _PROFILE_FIELDS:
        value = getattr(profile, name, None)
        digest.update(name.encode())
        if value is None:
            digest.update(b"<none>")
        else:
            digest.update(np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes())
    return digest.hexdigest()


def _finite_profile(values: Any) -> bool:
    if values is None:
        return False
    array = np.atleast_1d(np.asarray(values, dtype=float))
    return bool(array.size > 1 and np.all(np.isfinite(array)))


def _usable(values: Any) -> bool:
    """A profile that can stand for a measured temperature: finite, not identically <= 0.

    Positivity is enforced where it matters -- inside ``rho_max`` -- by the GACODE
    converter; a fitted profile that reaches zero at the separatrix is still a profile.
    """
    if values is None:
        return False
    array = np.atleast_1d(np.asarray(values, dtype=float))
    return bool(array.size > 1 and np.all(np.isfinite(array)) and np.max(array) > 0.0)


def _digest(payload: Any) -> str:
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(ods: Any, path: str) -> Any:
    """Read a leaf without creating it (omas materializes a missing path on read)."""
    from vaft.ods_access import path_value

    return path_value(ods, path, None)


def _times(ods: Any, path: str) -> Optional[np.ndarray]:
    values = _get(ods, path)
    if values is None:
        return None
    array = np.atleast_1d(np.asarray(values, dtype=float))
    return array if array.size else None


def _slice_times(ods: Any, ids: str) -> Optional[np.ndarray]:
    """``<ids>.time``: the homogeneous time vector the GACODE converter also reads."""
    return _times(ods, f"{ids}.time")


def _nearest(times: Optional[np.ndarray], target: float) -> tuple[Optional[int], float]:
    if times is None or not times.size:
        return None, float("nan")
    index = int(np.argmin(np.abs(times - target)))
    return index, float(times[index] - target)


def _insufficient(base: dict[str, Any], *reasons: str) -> ResolvedTransportState:
    return ResolvedTransportState(status="insufficient", reasons=tuple(reasons), **base)


_SHAPE = ("elongation", "triangularity_upper", "triangularity_lower")


def _derive_shape_profiles(work: Any, eq_index: int) -> Optional[dict[str, Any]]:
    """Write elongation and triangularities into slice ``eq_index`` of ``work``.

    Traces each ``profiles_1d.psi`` level of the 2-D map and keeps only contours that
    lie inside the slice's own boundary outline -- the longest-contour choice of the
    g-file converter picks open vacuum contours on limited VEST slices, and the
    product then drops the whole shape set.  The edge level is the outline itself;
    a level with no closed interior contour is interpolated from its neighbours in
    psi_N, never extrapolated past the outermost traced one.  Returns a provenance
    record, or ``None`` when the slice cannot support the trace.
    """
    from matplotlib.path import Path as _Polygon

    from vaft.process.equilibrium import contour_shape_parameters, extract_flux_surface_contours

    ts = f"equilibrium.time_slice.{eq_index}"
    r_grid = _get(work, f"{ts}.profiles_2d.0.grid.dim1")
    z_grid = _get(work, f"{ts}.profiles_2d.0.grid.dim2")
    psi_2d = _get(work, f"{ts}.profiles_2d.0.psi")
    psi_1d = _get(work, f"{ts}.profiles_1d.psi")
    axis = _get(work, f"{ts}.global_quantities.psi_axis")
    edge = _get(work, f"{ts}.global_quantities.psi_boundary")
    br = _get(work, f"{ts}.boundary.outline.r")
    bz = _get(work, f"{ts}.boundary.outline.z")
    if any(v is None for v in (r_grid, z_grid, psi_2d, psi_1d, axis, edge, br, bz)):
        return None
    axis, edge = float(axis), float(edge)
    if axis == edge:
        return None
    br, bz = np.asarray(br, dtype=float), np.asarray(bz, dtype=float)
    psin = (np.asarray(psi_1d, dtype=float) - axis) / (edge - axis)
    polygon = _Polygon(np.column_stack([br, bz]))
    profiles = {name: np.full(psin.size, np.nan) for name in _SHAPE}
    try:
        outline = contour_shape_parameters(br, bz)
    except ValueError:
        return None
    for name in _SHAPE:
        profiles[name][-1] = outline[name]
    contours = extract_flux_surface_contours(
        np.asarray(psi_2d, dtype=float), np.asarray(r_grid, dtype=float),
        np.asarray(z_grid, dtype=float), axis, edge, psin[1:-1],
    )
    traced = 0
    for index, level in enumerate(psin[1:-1], start=1):
        inside = [
            (r, z) for r, z in (contours.get(float(level)) or [])
            if r.size >= 16 and np.all(polygon.contains_points(np.column_stack([r, z]), radius=1e-6))
        ]
        if not inside:
            continue
        r_seg, z_seg = max(inside, key=lambda seg: seg[0].size)
        try:
            shape = contour_shape_parameters(r_seg, z_seg)
        except ValueError:
            continue
        for name in _SHAPE:
            profiles[name][index] = shape[name]
        traced += 1
    if traced < 3:
        return None
    for name, values in profiles.items():
        good = np.isfinite(values)
        # The axis level has no contour; it takes the innermost traced value, as the
        # g-file converter's fill does.  Interior gaps are interpolated in psi_N.
        # anti-alias: not a time series and not a downsample. This fills the few
        # psi_N levels of one equilibrium slice whose contour could not be traced,
        # on the same radial grid; there is no sample rate to reduce.
        values[~good] = np.interp(psin[~good], psin[good], values[good])
        work[f"{ts}.profiles_1d.{name}"] = values
    return {"kind": "derived", "levels_traced": traced, "levels": int(psin.size),
            "source": "closed contours of profiles_2d psi inside boundary.outline",
            "routine": "vaft.process.equilibrium.contour_shape_parameters"}


def _stored_inferred_ti(ods: Any, prefix: str, ions: list, time: float) -> Optional[dict]:
    """An ion temperature the product itself labels as inferred, as an ``inferred_ti`` record.

    Every ion must carry the inferred label and one common temperature (lane K writes
    H+ and C6+ with the same partitioned T_i); otherwise ``None``. ``species`` is the
    number of ions the product supplied, so the caller keeps its composition.
    """
    if not ions:
        return None
    notes = [_get(ods, f"{prefix}.ion.{i}.temperature_fit.parameters") for i in ions]
    if any(note is None or "origin=inferred" not in str(note) for note in notes):
        return None
    arrays = [_get(ods, f"{prefix}.ion.{i}.temperature") for i in ions]
    if any(a is None for a in arrays):
        return None
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    if any(a.shape != arrays[0].shape or not np.allclose(a, arrays[0], equal_nan=True) for a in arrays[1:]):
        return None
    method = "equilibrium_pressure_partition"
    for part in str(notes[0]).split(";"):
        if part.strip().startswith("method="):
            method = part.strip().split("=", 1)[1]
    return {"temperature": arrays[0], "method": method, "time": float(time),
            "source": str(notes[0]), "species": len(ions)}


def _segments(rho: np.ndarray, valid: np.ndarray) -> list[list[float]]:
    """Contiguous runs of valid points as ``[[rho_first, rho_last], ...]``."""
    out, start = [], None
    for k, ok in enumerate(valid):
        if ok and start is None:
            start = k
        if (not ok or k == len(valid) - 1) and start is not None:
            end = k if ok else k - 1
            out.append([float(rho[start]), float(rho[end])])
            start = None
    return out


def inferred_ti_supported(state: "ResolvedTransportState", r_over_a: float, *,
                          solver: str = "tglf", local: Any = None) -> bool:
    """Whether a surface's solver input is independent of the inferred-Ti gap fill.

    Parameters
    ----------
    state : ResolvedTransportState
        From :func:`resolve_transport_state` [-].
    r_over_a : float
        The surface [-].
    solver : str
        ``"tglf"`` compares the full TGLF local input; ``"neo"`` compares the ion
        temperatures and their log-gradients the way NEO reads them [-].
    local : TGLFInput, optional
        The surface's input already built from ``state.profile``, to avoid rebuilding
        it [-].

    Returns
    -------
    bool
        True when no gap was filled, or when the solver's input agrees (rtol 1e-4,
        atol 1e-4) between the profile and its ``fill_probe`` [-].

    Convention
    ----------
    GACODE differentiates on three grid points and evaluates with a spline over the
    whole profile, so a filled node can reach a surface several grid steps away.
    Rather than guess a margin, the gate builds the solver's input twice -- once from
    the profile, once from ``fill_probe``, whose gaps are filled with a different
    scale *and* offset -- and runs the surface only when nothing moves. TGLF's local
    input uses a not-a-knot spline; NEO (PROFILE_MODEL=2, ``expro_locsim``'s
    ``cub_spline1``) a natural one, so each solver is checked with its own.

    Applicability
    -------------
    Machine-independent.
    """
    probe = getattr(state, "fill_probe", None)
    if probe is None:
        return True
    if solver == "neo":
        return _neo_ion_inputs_agree(state.profile, probe, float(r_over_a))
    from vaft.code.gacode.tglf.inputs import LocalConversionError, prepare_tglf_input

    try:
        a = local if local is not None else prepare_tglf_input(state.profile, float(r_over_a))
        b = prepare_tglf_input(probe, float(r_over_a))
    except LocalConversionError:
        return False
    numeric = ("taus", "rlts", "as_", "rlns", "zs", "mass", "betae", "xnue", "zeff", "debye",
               "p_prime_loc", "q_loc", "q_prime_loc", "rmin_loc", "rmaj_loc", "kappa_loc",
               "delta_loc")
    for name in numeric:
        x = np.atleast_1d(np.asarray(getattr(a, name), dtype=float))
        y = np.atleast_1d(np.asarray(getattr(b, name), dtype=float))
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-4, atol=1e-4, equal_nan=True):
            return False
    return True


def _neo_ion_inputs_agree(profile: Any, probe: Any, r_over_a: float) -> bool:
    """NEO's local ion T and a/L_T, natural spline as in expro_locsim, profile vs probe."""
    from scipy.interpolate import CubicSpline

    from vaft.code.gacode.tglf.inputs import bound_deriv

    def ion_inputs(prof):
        rmin = np.asarray(prof.rmin, dtype=float)
        grid = rmin / rmin[-1]
        ti = np.atleast_2d(np.asarray(prof.ti, dtype=float))
        out = []
        for row in ti:
            if np.any(row <= 0.0) or not np.all(np.isfinite(row)):
                return None
            # anti-alias: not a time series and not a downsample; one radial profile
            # evaluated at one surface the way NEO's expro_locsim does.
            value = CubicSpline(grid, row, bc_type="natural")(r_over_a)
            gradient = CubicSpline(grid, bound_deriv(-np.log(row), rmin), bc_type="natural")(r_over_a)
            out += [float(value), float(rmin[-1] * gradient)]
        return np.asarray(out)

    a, b = ion_inputs(profile), ion_inputs(probe)
    if a is None or b is None or a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=1e-4, atol=1e-4))


# --------------------------------------------------------------------------- resolve


def resolve_transport_state(
    ods: Any,
    key: TransportStateKey,
    *,
    efit_quality: str,
    quality_source: str = "criteria",
    tolerance: float = DEFAULT_TIME_TOLERANCE_S,
    ti_te_ratio: Any = "policy",
    ti_te_ratio_sigma: Optional[float] = None,
    inferred_ti: Optional[Mapping[str, Any]] = None,
    use_stored_inferred_ti: bool = False,
    z_eff: Optional[float] = 2.0,
    impurity: Optional[str] = "C",
    rho_max: Optional[float] = DEFAULT_RHO_MAX,
    inputs: Optional[Mapping[str, Any]] = None,
) -> ResolvedTransportState:
    """Resolve one plasma state into a shared GACODE profile with per-quantity provenance.

    Parameters
    ----------
    ods : omas.ODS
        Carries ``equilibrium`` and ``core_profiles`` for the shot; never modified [-].
    key : TransportStateKey
        ``(shot, time_efit_s, efit_lineage)``; ``time_efit_s`` must be an equilibrium
        slice time within ``tolerance`` [s].
    efit_quality : str
        The #1331 slice label, ``"good"`` or ``"admissible"``; anything else is
        refused with :class:`ValueError`, an unreconstructible slice not being a state [-].
    quality_source : str
        Where ``efit_quality`` came from, e.g. ``"criteria"`` or
        ``"magnetic_slice_at_same_time"`` [-].
    tolerance : float
        Largest allowed |t_core_profiles - t_equilibrium| and |t_equilibrium -
        time_efit_s| [s].
    ti_te_ratio : str or float
        Ion-temperature fallback when no ion temperature is measured: ``"policy"``
        reads ``vest.yaml`` through
        :func:`vaft.machine_mapping.core_profiles.policy_for_ods`; a number is used
        as given and recorded as caller-supplied; ``None`` disables the fallback [-].
    ti_te_ratio_sigma : float, optional
        Its 1-sigma, for provenance; the policy's when ``None`` [-].
    inferred_ti : mapping, optional
        A pressure-partition result (#1426), ``{"temperature", "method", "time"}`` on
        the core_profiles grid of the matched slice, preferred over the policy
        fallback; its ``time`` must match the paired slice within ``tolerance`` [eV].
    use_stored_inferred_ti : bool
        Whether an ion temperature the product itself labels as inferred
        (``temperature_fit.parameters`` containing ``origin=inferred``, lane K's #1426
        partition) is used as the ``inferred_ti`` step. Off by default: the routine atlas
        keeps Ti = Te (#1414), and such a state is refused as
        ``inferred_ti_not_enabled``. A labelled temperature is never taken as measured [-].
    z_eff : float, optional
        Effective charge the composition closure realizes [-].
    impurity : str, optional
        Impurity species of the closure (``"C"`` -> C6+); ``None`` keeps one main ion [-].
    rho_max : float, optional
        Outer edge of the converted profile, as rho_tor_norm [-].
    inputs : mapping, optional
        Identity of the products the ODS was assembled from (paths, sha256) [-].

    Returns
    -------
    ResolvedTransportState
        ``status`` ``"resolved"`` with ``profile`` set, or ``"insufficient"`` with
        machine-readable ``reasons`` [-].

    Processing steps
    ----------------
    1. Match the equilibrium slice to ``time_efit_s`` and the core_profiles slice to
       that equilibrium time, both by time within ``tolerance``.
    2. Resolve the ion temperature: measured, then ``inferred_ti``, then the policy
       ratio, else insufficient; write a main ion into a deep copy when needed.
    3. Derive ``r_inboard``/``r_outboard`` and, when absent, the elongation and
       triangularity profiles from the 2-D flux map (on the copy), recording that
       they were derived here.
    4. Convert with :func:`vaft.code.gacode.inputs.prepare_gacode_profile`, which
       applies the composition closure and refuses non-positive profiles.

    Applicability
    -------------
    VEST-specific. The policy default reads VEST's ``vest.yaml``
    (``diagnostics.core_profiles.ti_te_ratio``) and the composition default is the
    VEST modelling policy (H+, C6+, Z_eff = 2); campaign core_profiles and EFIT
    stage products are the data it was built against.

    Provenance
    ----------
    .. [1] Issue #1428 sections 2-7: the Ti hierarchy (measured, #1426 pressure
       partition, empirical ratio, insufficient), the VEST H+/C6+ Z_eff = 2 composition
       policy and explicit time alignment.
    .. [2] Issue #1414: VEST Thomson-only slices take Ti = Te with sigma 0.5
       (``vest.yaml`` ``diagnostics.core_profiles.ti_te_ratio``, status ``assumed``).
    .. [3] GACODE ``input.gacode``: ``rmin``/``rmaj`` are the midplane half-width and
       centre at the axis height (https://gacode.io/input_gacode.html), derived here
       with :func:`vaft.omas.update_equilibrium_profiles_1d_radial_coordinates`.
    """
    if efit_quality not in EFIT_QUALITIES:
        raise ValueError(
            f"efit_quality must be one of {EFIT_QUALITIES}; got {efit_quality!r}. "
            "Unreconstructible slices are not resolved into transport states."
        )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and non-negative; got {tolerance!r}")

    settings = {
        "tolerance_s": float(tolerance),
        "rho_max": None if rho_max is None else float(rho_max),
        "z_eff": None if z_eff is None else float(z_eff),
        "impurity": impurity,
    }
    base: dict[str, Any] = {
        "key": key,
        "efit_quality": efit_quality,
        "quality_source": quality_source,
        "inputs": dict(inputs or {}),
        "settings": settings,
    }

    # 1. times ---------------------------------------------------------------------
    eq_times = _slice_times(ods, "equilibrium")
    eq_index, eq_offset = _nearest(eq_times, key.time_efit_s)
    if eq_index is None:
        return _insufficient(base, "no_equilibrium")
    if abs(eq_offset) > tolerance:
        return _insufficient(base, "no_equilibrium_within_tolerance")
    eq_time = float(eq_times[eq_index])
    cp_times = _slice_times(ods, "core_profiles")
    cp_index, cp_offset = _nearest(cp_times, eq_time)
    times = {
        "time_equilibrium_s": eq_time,
        "equilibrium_index": eq_index,
        "time_profile_s": None if cp_index is None else float(cp_times[cp_index]),
        "core_profiles_index": cp_index,
        "dt_s": None if cp_index is None else float(cp_offset),
        "tolerance_s": float(tolerance),
        "method": "nearest in time within tolerance",
    }
    base["times"] = times
    if cp_index is None:
        return _insufficient(base, "no_core_profiles")
    if abs(cp_offset) > tolerance:
        return _insufficient(base, "no_core_profiles_within_tolerance")

    prefix = f"core_profiles.profiles_1d.{cp_index}"
    te = _get(ods, f"{prefix}.electrons.temperature")
    ne = _get(ods, f"{prefix}.electrons.density_thermal")
    if ne is None:
        ne = _get(ods, f"{prefix}.electrons.density")
    missing = [name for name, value in (("ne", ne), ("te", te)) if value is None]
    if missing:
        return _insufficient(base, *[f"missing_{name}" for name in missing])
    te = np.asarray(te, dtype=float)
    ne = np.asarray(ne, dtype=float)

    # 2. ion temperature ------------------------------------------------------------
    ions = []
    position = 0
    while _get(ods, f"{prefix}.ion.{position}.label") is not None or _get(
        ods, f"{prefix}.ion.{position}.z_ion"
    ) is not None:
        ions.append(position)
        position += 1
    # A product may carry an ion temperature that was inferred, not measured (lane K's
    # #1426 pressure partition writes ``temperature_fit.parameters = "origin=inferred;
    # method=..."``). It is never taken as a measurement; it is used as the inferred
    # step only when the caller opts in (``use_stored_inferred_ti``), else refused.
    fill = None
    stored = _stored_inferred_ti(ods, prefix, ions, cp_times[cp_index])
    labelled = [i for i in ions if "origin=inferred" in str(
        _get(ods, f"{prefix}.ion.{i}.temperature_fit.parameters") or "")]
    if labelled and not use_stored_inferred_ti and inferred_ti is None:
        # The product's T_i is an inference and the caller did not opt in: refusing is
        # the only honest choice, since the measured step would mislabel it.
        return _insufficient(base, "inferred_ti_not_enabled")
    if labelled and stored is None and inferred_ti is None:
        # Some ion says its T_i was inferred but the set is not one common inferred
        # profile: it is neither a measurement nor a usable inference.
        return _insufficient(base, "inferred_ti_species_disagree")
    if use_stored_inferred_ti and stored is not None and inferred_ti is None:
        inferred_ti = stored
    measured = [i for i in ions
                if _usable(_get(ods, f"{prefix}.ion.{i}.temperature")) and i not in labelled]
    hierarchy: list[dict[str, Any]] = []
    work = ods
    if ions and len(measured) == len(ions):
        ti = {"lineage": "measured", "method": "core_profiles ion temperature",
              "ratio": None, "sigma": None, "kind": "measured"}
        hierarchy.append({"step": "measured", "status": "used"})
    else:
        hierarchy.append({"step": "measured", "status": "unavailable"})
        if len(ions) > 1 and not (stored is not None and stored["species"] == len(ions)):
            return _insufficient(
                {**base, "ti": {"lineage": "unresolved", "hierarchy": hierarchy}},
                "ion_temperature_unresolved_multiple_species",
            )
        temperature = None
        if inferred_ti is not None:
            hierarchy.append({"step": "pressure_inferred", "status": "used"})
            temperature = np.asarray(inferred_ti["temperature"], dtype=float)
            if temperature.shape != te.shape:
                return _insufficient(base, "inferred_ti_grid_mismatch")
            # The array carries no grid identity of its own, so its time is required
            # and checked: a result for another slice of the same length is refused.
            inferred_time = inferred_ti.get("time")
            if inferred_time is None or abs(float(inferred_time) - float(cp_times[cp_index])) > tolerance:
                return _insufficient(base, "inferred_ti_time_mismatch")
            valid = np.isfinite(temperature) & (temperature > 0.0)
            if not valid.any():
                return _insufficient(base, "inferred_ti_not_positive")
            ti = {"lineage": "pressure_partition_inferred",
                  "method": str(inferred_ti.get("method", "equilibrium_pressure_partition")),
                  "ratio": None, "sigma": None, "kind": "inferred",
                  "temperature_sha256": hashlib.sha256(
                      np.ascontiguousarray(np.nan_to_num(temperature, nan=-1.0)).tobytes()).hexdigest()}
            if "source" in inferred_ti:
                ti["source"] = str(inferred_ti["source"])
            if not valid.all():
                # The inference is undefined where p_i <= 0 or not significant. GACODE
                # needs a complete profile, so those points take the policy ratio --
                # declared here, and no surface whose stencil reaches them is run
                # (readiness: ti_not_inferred_here), so every solved surface's Ti is
                # purely inferred.
                if ti_te_ratio is None:
                    return _insufficient(base, "inferred_ti_incomplete")
                if isinstance(ti_te_ratio, str):
                    if ti_te_ratio != "policy":
                        raise ValueError(
                            f"ti_te_ratio must be 'policy', a number or None; got {ti_te_ratio!r}")
                    from vaft.machine_mapping.core_profiles import policy_for_ods

                    fill_ratio, fill_source = float(policy_for_ods(ods, key.shot).ti_te_ratio), "policy"
                else:
                    fill_ratio, fill_source = float(ti_te_ratio), "caller"
                if not np.isfinite(fill_ratio) or fill_ratio <= 0.0:
                    return _insufficient(base, "non_positive_ti_te_ratio")
                fill = (valid, np.asarray(temperature, dtype=float), fill_ratio)
                temperature = np.where(valid, temperature, fill_ratio * te)
                ti["fill"] = {"points": int((~valid).sum()), "of": int(valid.size),
                              "value": f"ti_te_ratio {fill_ratio:g} ({fill_source}); a surface whose "
                                       "solver input depends on it is not run"}
            grid = _get(ods, f"{prefix}.grid.rho_tor_norm")
            if grid is not None and np.size(grid) == valid.size:
                ti["support_rho"] = _segments(np.asarray(grid, dtype=float), valid)
        else:
            hierarchy.append({"step": "pressure_inferred", "status": "unavailable",
                              "reason": "no #1426 result supplied"})
            if ti_te_ratio is None:
                return _insufficient(
                    {**base, "ti": {"lineage": "unresolved", "hierarchy": hierarchy}},
                    "no_defensible_ion_temperature",
                )
            if isinstance(ti_te_ratio, str):
                if ti_te_ratio != "policy":
                    raise ValueError(
                        f"ti_te_ratio must be 'policy', a number or None; got {ti_te_ratio!r}"
                    )
                from vaft.machine_mapping.core_profiles import policy_for_ods

                policy = policy_for_ods(ods, key.shot)
                ratio = float(policy.ti_te_ratio)
                sigma = float(policy.ti_te_ratio_sigma if ti_te_ratio_sigma is None
                              else ti_te_ratio_sigma)
                status = str(policy.ti_te_ratio_status)
                source = policy.ti_te_ratio_text()
            else:
                ratio = float(ti_te_ratio)
                sigma = None if ti_te_ratio_sigma is None else float(ti_te_ratio_sigma)
                status = "assumed"
                source = "caller argument"
            if not np.isfinite(ratio) or ratio <= 0.0:
                return _insufficient(base, "non_positive_ti_te_ratio")
            hierarchy.append({"step": "policy_ratio", "status": "used"})
            temperature = ratio * te
            # Contract v1 spells the #1414 Ti = Te policy ``ti_eq_te_assumed``; any
            # other ratio keeps its value in the name so two ratios never share one.
            lineage = "ti_eq_te_assumed" if ratio == 1.0 else f"ti_te_{ratio:g}_{status}"
            ti = {"lineage": lineage, "method": "ti_te_ratio",
                  "ratio": ratio, "sigma": sigma, "kind": status, "source": source}
        work = copy.deepcopy(ods) if work is ods else work
        ion = f"{prefix}.ion.0"
        if not ions:
            # An electron-only product stores `ion` as a null leaf (NaN once loaded),
            # which an array of structures cannot be written into.
            slice_node = work[prefix]
            if "ion" in slice_node.keys() and not hasattr(slice_node.getraw("ion"), "keys"):
                del slice_node["ion"]
            work[f"{ion}.label"] = _HYDROGEN["label"]
            work[f"{ion}.z_ion"] = _HYDROGEN["z"]
            work[f"{ion}.element.0.z_n"] = _HYDROGEN["z"]
            work[f"{ion}.element.0.a"] = _HYDROGEN["a"]
            # A placeholder density: the composition closure below rederives both ion
            # densities from n_e, and without a closure a lone H+ at n_e is
            # quasi-neutrality, which is what is recorded.
            work[f"{ion}.density_thermal"] = ne
        for position in (ions or [0]):
            work[f"{prefix}.ion.{position}.temperature"] = temperature
    ti["hierarchy"] = hierarchy
    base["ti"] = ti

    # 3. geometry --------------------------------------------------------------------
    # GACODE's rmin/rmaj are the midplane half-width and centre at the axis height.
    # EFIT products carry no r_inboard/r_outboard, so derive them from the 2-D flux
    # map on the working copy -- the same routine the NEO comparison path uses.
    eq_base = f"equilibrium.time_slice.{eq_index}.profiles_1d"
    geometry = {"kind": "reconstructed", "source": f"{eq_base}.r_inboard/r_outboard"}
    # Finiteness, not presence: a NaN pair would reach GACODE's rmin unchecked and
    # the TGLF mapper then refuses every surface without naming the cause.
    if not all(_finite_profile(_get(ods, f"{eq_base}.{name}")) for name in ("r_inboard", "r_outboard")):
        from vaft.omas import update_equilibrium_profiles_1d_radial_coordinates

        work = copy.deepcopy(ods) if work is ods else work
        update_equilibrium_profiles_1d_radial_coordinates(work, time_slice=eq_index)
        geometry = {"kind": "derived",
                    "source": "midplane crossings of profiles_2d psi at the axis height",
                    "routine": "vaft.omas.update_equilibrium_profiles_1d_radial_coordinates"}
    shape = {"kind": "reconstructed", "source": f"{eq_base}.elongation/triangularity_*"}
    if not all(_finite_profile(_get(ods, f"{eq_base}.{name}")) for name in _SHAPE):
        work = copy.deepcopy(ods) if work is ods else work
        shape = _derive_shape_profiles(work, eq_index)
        if shape is None:
            return _insufficient(base, "equilibrium_shape_underivable")

    # 4. composition and conversion -------------------------------------------------
    from vaft.code.gacode.inputs import ProfileConversionError, prepare_gacode_profile

    try:
        # A product that already carries its ion species (lane K's H+/C6+ closure) is
        # converted as given: re-applying the closure would assume a single main ion.
        # The product's ion list is its composition whoever supplies T_i: re-applying
        # the single-main-ion closure to two ions would be refused (or double-count C).
        product_species = stored is not None and stored["species"] > 1
        convert = dict(
            time=eq_time,
            tolerance=max(float(tolerance), 1e-9),
            rho_max=rho_max,
            z_eff=None if product_species else (z_eff if impurity is not None else None),
            impurity=None if product_species else impurity,
            shot=int(key.shot),
        )
        profile = prepare_gacode_profile(work, **convert)
        fill_probe = None
        if fill is not None:
            # The same profile with the gaps filled differently: the readiness gate
            # compares every surface's solver input across the two.
            valid_mask, raw, ratio_used = fill
            probe = copy.deepcopy(work)
            # Different scale *and* offset: a pure rescale leaves log-gradients inside
            # the gap unchanged, and Te = 0 points would stay 0 under any rescale.
            alternative = np.where(valid_mask, raw, 2.0 * ratio_used * te + 0.25 * float(np.nanmax(te)))
            for position in (ions or [0]):
                probe[f"{prefix}.ion.{position}.temperature"] = alternative
            fill_probe = prepare_gacode_profile(probe, **convert)
    except ProfileConversionError as error:
        return _insufficient(base, f"profile_conversion: {error}")

    provenance = dict(getattr(profile, "provenance", {}) or {})
    provenance["ti"] = {"kind": ti["kind"], "method": ti["method"],
                        **({"source": ti["source"]} if "source" in ti else {}),
                        **({"ratio": ti["ratio"], "sigma": ti["sigma"]}
                           if ti.get("ratio") is not None else {})}
    provenance["equilibrium"] = {"kind": "reconstructed", "lineage": key.efit_lineage,
                                 "label": efit_quality, "quality_source": quality_source}
    provenance["q"] = {"kind": "reconstructed", "source": "equilibrium"}
    provenance["midplane_geometry"] = geometry
    provenance["shape"] = shape
    provenance["magnetic_shear"] = {"kind": "derived", "source": "equilibrium q"}
    profile.provenance = provenance
    if product_species:
        settings["z_eff"], settings["impurity"] = None, None
        # From the species the solvers read, not from any z_eff column (#803).
        ni = np.atleast_2d(np.asarray(profile.ni, dtype=float))
        charge = np.asarray(profile.z, dtype=float)[:, None]
        ne_profile = np.asarray(profile.ne, dtype=float)
        realized = (ni * charge ** 2).sum(axis=0) / ne_profile
        neutrality = float(np.max(np.abs((ni * charge).sum(axis=0) / ne_profile - 1.0)))
        composition = {
            "species": list(getattr(profile, "name", []) or []),
            "z_eff": float(np.median(realized)),
            "quasineutrality_error": neutrality,
            "impurity": None,
            # The product's species list is itself a modelling closure, not a measurement.
            "origin": "policy_assumption",
            "source": "the core_profiles product's own ion list",
        }
    else:
        composition = {
            "species": list(getattr(profile, "name", []) or []),
            "z_eff": settings["z_eff"],
            "impurity": impurity,
            "origin": "policy_assumption" if impurity is not None else "measured_or_single_ion",
        }
    return ResolvedTransportState(
        status="resolved", profile=profile, composition=composition,
        provenance=provenance, fill_probe=fill_probe, **base,
    )


# --------------------------------------------------------------------------- readiness


def _state_conditions(state: ResolvedTransportState) -> tuple[str, ...]:
    conditions = []
    if state.ti.get("kind") not in ("measured",):
        conditions.append(f"ti_{state.ti.get('kind', 'unknown')}")
    if state.composition.get("origin") == "policy_assumption":
        conditions.append("composition_policy")
    vtor = state.provenance.get("vtor", {}) if state.provenance else {}
    if vtor.get("kind") == "unavailable":
        conditions.append("no_rotation")
    if state.quality_source != "criteria":
        conditions.append(f"efit_quality_from_{state.quality_source}")
    return tuple(conditions)


def _surface_code(error: Exception) -> str:
    """The readiness code of a ``LocalConversionError`` from ``prepare_tglf_input``.

    Only the domain refusal is told apart; a non-positive profile is refused one
    level up, by ``prepare_gacode_profile`` at state resolution, and never reaches
    a surface.
    """
    text = str(error)
    if "outside the converted profile" in text or "strictly inside" in text:
        return "outside_profile_domain"
    return "local_conversion_failure"


def assess_tglf_readiness(
    state: ResolvedTransportState,
    surfaces: Sequence[float] = DEFAULT_SURFACES,
    *,
    config: Any = None,
) -> ReadinessReport:
    """Decide whether a resolved state, and each requested surface, can go to TGLF.

    Parameters
    ----------
    state : ResolvedTransportState
        From :func:`resolve_transport_state` [-].
    surfaces : sequence of float
        Requested surfaces as r/a [-].
    config : TGLFConfig, optional
        Only ``n_species`` is consulted, through ``prepare_tglf_input`` [-].

    Returns
    -------
    ReadinessReport
        State verdict plus one :class:`SurfaceReadiness` per surface, each carrying
        the built ``TGLFInput`` when ready [-].

    Processing steps
    ----------------
    1. An unresolved state is ``insufficient`` and no surface is built.
    2. Each surface is projected with ``prepare_tglf_input``; its documented refusal
       becomes a reason code, then ``check_tglf_requirements`` and finite gradients
       are required.
    3. A runnable state with declared assumptions is ``conditional``.

    Applicability
    -------------
    Machine-independent.
    """
    if not state.resolved:
        return ReadinessReport("tglf", "insufficient", reasons=state.reasons)
    from vaft.code.gacode.tglf.inputs import LocalConversionError, prepare_tglf_input

    results = []
    for radius in surfaces:
        try:
            local = prepare_tglf_input(state.profile, float(radius), config=config)
        except LocalConversionError as error:
            results.append(SurfaceReadiness(float(radius), _surface_code(error), str(error)))
            continue
        if not inferred_ti_supported(state, float(radius), local=local):
            results.append(SurfaceReadiness(float(radius), "ti_not_inferred_here",
                                             "this surface's solver input depends on the inferred-Ti gap fill"))
            continue
        absent = local.check_tglf_requirements()
        if absent:
            code = "invalid_gradient" if set(absent) & {"rlns", "rlts"} else "local_conversion_failure"
            results.append(SurfaceReadiness(float(radius), code, ", ".join(absent), local))
            continue
        results.append(SurfaceReadiness(float(radius), "ready", "", local))
    if not any(s.ready for s in results):
        return ReadinessReport("tglf", "insufficient", reasons=("no_ready_surface",),
                               surfaces=tuple(results))
    conditions = _state_conditions(state)
    if any(s.ready and s.local_input.vexb_shear is None for s in results):
        # prepare_tglf_input does not derive ExB shear yet (#553), so TGLF runs with
        # its own zero whatever rotation core_profiles carries.
        conditions = (*conditions, "no_exb_shear")
    return ReadinessReport("tglf", "conditional" if conditions else "ready",
                           conditions=conditions, surfaces=tuple(results))


def assess_neo_readiness(state: ResolvedTransportState,
                         surfaces: Sequence[float] = DEFAULT_SURFACES) -> ReadinessReport:
    """Decide whether a resolved state can go to NEO.

    Parameters
    ----------
    state : ResolvedTransportState
        From :func:`resolve_transport_state` [-].
    surfaces : sequence of float
        The r/a NEO will solve; with an inferred-Ti gap fill, at least one must have
        an input independent of the fill [-].

    Returns
    -------
    ReadinessReport
        ``ready``/``conditional``/``insufficient``; NEO is a profile code, so no
        per-surface entries [-].

    Processing steps
    ----------------
    1. An unresolved state is ``insufficient``.
    2. ``GACODEProfile.check_neo_requirements`` (the executable-input contract) must
       report nothing missing.
    3. Declared assumptions make the state ``conditional``.

    Applicability
    -------------
    Machine-independent.
    """
    if not state.resolved:
        return ReadinessReport("neo", "insufficient", reasons=state.reasons)
    missing = tuple(state.profile.check_neo_requirements())
    if missing:
        return ReadinessReport("neo", "insufficient",
                               reasons=tuple(f"missing_{name}" for name in missing))
    if state.fill_probe is not None and not any(
            inferred_ti_supported(state, r, solver="neo") for r in surfaces):
        return ReadinessReport("neo", "insufficient", reasons=("no_surface_independent_of_ti_fill",))
    conditions = _state_conditions(state)
    return ReadinessReport("neo", "conditional" if conditions else "ready",
                           conditions=conditions)


# --------------------------------------------------------------------------- identity


def run_identity(
    state: ResolvedTransportState,
    *,
    solver: str,
    parameters: Mapping[str, Any],
    surface: Optional[float] = None,
    solver_revision: Optional[str] = None,
) -> str:
    """Deterministic identity of one solver calculation on one resolved state.

    Parameters
    ----------
    state : ResolvedTransportState
        The upstream state; its own identity is folded in [-].
    solver : str
        ``"tglf"``, ``"neo"`` or ``"classical"`` [-].
    parameters : mapping
        The solver's effective settings (e.g. ``dataclasses.asdict(config)`` minus
        runtime-only fields) [-].
    surface : float, optional
        r/a of a local calculation [-].
    solver_revision : str, optional
        The GACODE revision the run used or will use [-].

    Returns
    -------
    str
        sha256 hex digest; equal inputs give equal identities, which is the cache key [-].

    Applicability
    -------------
    Machine-independent.
    """
    return _digest({
        "state": state.identity,
        "solver": solver,
        "parameters": dict(parameters),
        "surface": None if surface is None else round(float(surface), 6),
        "solver_revision": solver_revision,
    })


def physics_parameters(config: Any, *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """The physics half of a GACODE config, with runtime-only fields dropped.

    ``backend``, ``timeout``, ``env``, ``home``, ``executable``, ``platform`` and the
    MPI/OMP counts decide *where* a run happens, not *what* it computes, so they stay
    out of identity.

    Parameters
    ----------
    config : GACODEConfig
        A ``TGLFConfig`` or ``NEOConfig`` dataclass [-].
    exclude : iterable of str
        Further field names to leave out [-].

    Returns
    -------
    dict
        Field name to JSON-ready value, for :func:`run_identity` [-].

    Applicability
    -------------
    Machine-independent.
    """
    from dataclasses import fields as dataclass_fields

    runtime = {"backend", "timeout", "env", "home", "executable", "workdir", "args",
               "platform", "n_mpi", "n_omp", "memory_mb", *exclude}
    out = {}
    for entry in dataclass_fields(config):
        if entry.name in runtime:
            continue
        out[entry.name] = _jsonable(getattr(config, entry.name))
    return out


# --------------------------------------------------------------------------- partition

#: Magnitudes below this many W/m^2 (or m^-2 s^-1) make a ratio undefined, not huge.
_RATIO_FLOOR = {"energy": 1e-6, "particle": 1e6}


def _channels(row: Mapping[str, Any], charges: Optional[Mapping[str, float]] = None) -> dict[str, float]:
    """Flatten one mapped surface row into ``{channel: SI flux}``.

    Ion species are keyed by charge (``z=1``), because NEO writes no labels and the
    charge is what the two models demonstrably share; ``charges`` maps a labelled
    row's species names onto it.
    """
    out: dict[str, float] = {}
    for kind, key in (("energy", "electron_energy_flux_W_m2"), ("particle", "electron_particle_flux_m2_s")):
        if row.get(key) is not None:
            out[f"electron_{kind}"] = float(row[key])
    for kind, key in (("energy", "ion_energy_flux_W_m2"), ("particle", "ion_particle_flux_m2_s")):
        for name, value in (row.get(key) or {}).items():
            if value is None:
                continue
            label = name if name.startswith("z=") else (
                f"z={float(charges[name]):g}" if charges and name in charges else None)
            if label is None:
                raise ValueError(f"no charge for species {name!r}; pass charges=")
            if f"ion_{kind}_{label}" in out:
                # Two ions with one charge (H+ and D+) cannot be told apart by charge,
                # and summing or overwriting them would both be a silent choice.
                raise ValueError(f"two ion species share {label}; charge does not identify them")
            out[f"ion_{kind}_{label}"] = float(value)
    return out


def _ion_total(channels: dict[str, float], kind: str) -> Optional[float]:
    ions = [v for k, v in channels.items() if k.startswith(f"ion_{kind}_z=")]
    return float(sum(ions)) if ions else None


def transport_partition(
    neoclassical: Mapping[str, Any],
    turbulent: Mapping[str, Any],
    *,
    turbulent_charges: Optional[Mapping[str, float]] = None,
    classical: Optional[Mapping[str, Any]] = None,
    rho_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Signed model fluxes and their magnitude partition at one surface of one state.

    Parameters
    ----------
    neoclassical : mapping
        One mapped NEO surface: ``r_over_a``, ``rho_tor_norm`` and electron/ion energy
        and particle fluxes in SI, ions keyed ``z=<charge>`` [W/m^2].
    turbulent : mapping
        One mapped TGLF surface in the same shape, ions keyed by label [W/m^2].
    turbulent_charges : mapping, optional
        Species label -> charge for the TGLF row (``{"H+": 1, "C6+": 6}``) [e].
    classical : mapping, optional
        A classical row in the same shape, added as a third component when given [-].
    rho_tolerance : float
        Largest allowed rho_tor_norm disagreement between the two rows [-].

    Returns
    -------
    dict
        ``status`` ``"available"`` with per-channel ``neo``, ``turb``, (``classical``),
        ``model`` (the modelled sum, not a measurement), ``f_neo`` =
        |neo| / (|neo| + |turb| [+ |classical|]) and ``neo_over_turb`` (``None`` below a
        floor), or ``"unavailable"`` with a reason [-].

    Applicability
    -------------
    Machine-independent. Callers join only rows with one ``state_identity``.
    """
    if neoclassical is None or turbulent is None:
        missing = "neoclassical" if neoclassical is None else "turbulent"
        return {"status": "unavailable", "reason": f"missing_{missing}_component"}
    if abs(float(neoclassical["r_over_a"]) - float(turbulent["r_over_a"])) > 1e-4:
        return {"status": "unavailable", "reason": "surfaces_differ_in_r_over_a"}
    if abs(float(neoclassical["rho_tor_norm"]) - float(turbulent["rho_tor_norm"])) > rho_tolerance:
        # Same r/a but a different flux surface means two different profiles.
        return {"status": "unavailable", "reason": "surfaces_differ_in_rho_tor_norm"}
    if classical is not None and abs(float(classical["r_over_a"]) - float(turbulent["r_over_a"])) > 1e-4:
        return {"status": "unavailable", "reason": "classical_surface_differs"}
    neo = _channels(neoclassical)
    turb = _channels(turbulent, turbulent_charges)
    cl = _channels(classical) if classical is not None else {}
    # The ion total is a channel only when both models carry the same ion species;
    # a classical row (main ion only) does not take part in it.
    for kind in ("energy", "particle"):
        ion_n = {k for k in neo if k.startswith(f"ion_{kind}_z=")}
        ion_t = {k for k in turb if k.startswith(f"ion_{kind}_z=")}
        if ion_n and ion_n == ion_t:
            neo[f"ion_{kind}_total"] = _ion_total(neo, kind)
            turb[f"ion_{kind}_total"] = _ion_total(turb, kind)
    channels = {}
    for name in sorted(set(neo) & set(turb)):
        kind = "energy" if "_energy" in name else "particle"
        n, t = neo[name], turb[name]
        parts = [abs(n), abs(t)] + ([abs(cl[name])] if name in cl else [])
        denominator = sum(parts)
        entry = {
            "neo": n, "turb": t,
            "model": n + t + (cl[name] if name in cl else 0.0),
            "f_neo": None if denominator == 0.0 else abs(n) / denominator,
            "f_turb": None if denominator == 0.0 else abs(t) / denominator,
            "neo_over_turb": None if abs(t) < _RATIO_FLOOR[kind] else n / t,
        }
        if name in cl:
            entry["classical"] = cl[name]
            entry["f_classical"] = None if denominator == 0.0 else abs(cl[name]) / denominator
        channels[name] = entry
    if not channels:
        return {"status": "unavailable", "reason": "no_common_channel"}
    return {"status": "available", "r_over_a": float(turbulent["r_over_a"]),
            "rho_tor_norm": float(turbulent["rho_tor_norm"]),
            "components": ["neo", "turb"] + (["classical"] if classical is not None else []),
            "channels": channels}


# --------------------------------------------------------------------------- classical (#1435)

#: GACODE's deuterium mass, the unit TGLF's MASS_* carry [kg].
_MASS_DEUTERIUM_KG = 3.34358e-27

CLASSICAL_MODEL = {
    "formulation": "Braginskii perpendicular conductive heat flux, strongly magnetised limit",
    "terms": "q_perp,s = -kappa_perp,s dT_s/dr for electrons and the main ion only; "
             "no particle flux, no thermal-force cross terms, no impurity heat flux",
    "coefficients": {"electron": "Braginskii gamma_1' at Z_eff (4.66 at Z=1, 4.0 at Z=2, "
                                 "3.6 at Z=4, 3.25 at Z=inf; held at 3.6 above Z=4)",
                     "ion": 2.0},
    "collision_times": "tau_e with n_i Z^2 -> n_e Z_eff; tau_i against every ion species, "
                       "Z_i^2 sum_j n_j Z_j^2 (like-particle form for unlike field ions); "
                       "lnLambda_e from vaft.formula.equilibrium.coulomb_logarithm_from_n_T, "
                       "lnLambda_ii from the NRL ion-ion form",
    "geometry": "|B| = toroidal field at the surface centre, |B_centr| R_centr / R (B_pol "
                "neglected); slab <|grad r|^2> = 1; dT/dr from the TGLF local a/L_T "
                "(GACODE's differentiate-then-interpolate order)",
    "reference": "S. I. Braginskii, Rev. Plasma Phys. 1 (1965) 205; NRL Plasma Formulary",
}

#: Braginskii's gamma_1' (electron perpendicular heat conductivity) against Z,
#: Braginskii 1965, Table 2: 4.66, 4.0, 3.7, 3.6, 3.25 for Z = 1, 2, 3, 4, inf.  The
#: Z = inf knot sits at 1e9, so a linear lookup in Z is held at 3.6 for Z > 4.
_GAMMA1_PERP = ((1.0, 4.66), (2.0, 4.0), (3.0, 3.7), (4.0, 3.6), (1e9, 3.25))


def surface_toroidal_field(profile: Any, local: Any) -> float:
    """The toroidal field at a surface's centre, ``|B_centr| R_centr / R``.

    GACODE's ``B_unit`` is a flux-coordinate field (about twice B0 on VEST), not the
    field a gyro-orbit sees, so the classical baseline does not use it.

    Parameters
    ----------
    profile : GACODEProfile
        Supplies ``bcentr`` [T] and ``rcentr`` [m].
    local : TGLFInput
        Supplies ``rmaj_loc`` and the minor radius of the surface [-].

    Returns
    -------
    float
        ``|B_T|`` at ``R = rmaj_loc * a``; the poloidal field is neglected [T].

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1] Vacuum toroidal field ``B_T = B_0 R_0 / R``; GACODE ``input.gacode``
       ``bcentr``/``rcentr`` (https://gacode.io/input_gacode.html).
    """
    r_surface = float(local.rmaj_loc) * float(local.normalisation.minor_radius)
    return abs(float(profile.bcentr)) * float(profile.rcentr) / r_surface


def classical_heat_fluxes(local: Any, b_tesla: float) -> dict[str, Any]:
    """Classical perpendicular heat fluxes at one surface, from a TGLF local input.

    Parameters
    ----------
    local : TGLFInput
        From :func:`assess_tglf_readiness`, supplying n_e, T_e, Ti/Te, every ion's
        n/n_e and charge, a/L_T, the main ion's mass and a at the surface [-].
    b_tesla : float
        Magnetic-field magnitude at the surface, e.g. :func:`surface_toroidal_field`;
        must be non-zero and finite, else ``ValueError`` [T].

    Returns
    -------
    dict
        ``r_over_a``; ``electron_energy_flux_W_m2``; ``ion_energy_flux_W_m2`` keyed
        ``z=<charge>`` for the main ion; ``chi_e_m2_s``, ``chi_i_m2_s``; the collision
        times, Coulomb logarithms, ``coulomb_log_valid`` and ``model``
        (:data:`CLASSICAL_MODEL`) [W/m^2].

    Convention
    ----------
    ``kappa_perp,e = gamma_1'(Z_eff) n_e T_e / (m_e Omega_e^2 tau_e)`` and
    ``kappa_perp,i = 2 n_i T_i / (m_i Omega_i^2 tau_i)``; ``q = kappa T (a/L_T) / a``,
    so a positive flux runs down the temperature gradient, the sign the TGLF and NEO
    mappers use.  This is a physical flux of a stated reduced model, not the
    order-of-magnitude ``nu rho^2`` reference scale (#780/#1112), and nothing here is
    projected into ``core_transport``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Electrons and one main ion only; no particle flux or thermal-force terms; unlike
    field ions enter tau_i in the like-particle form; slab geometry and B_pol neglected.
    ``coulomb_log_valid`` is False below 10 eV, where the electron form is not
    defined.  ``gamma_1'`` is interpolated linearly in Z between Braginskii's tabulated
    charges and is held at its Z = 4 value (3.6) for every Z_eff above 4, the Z = inf
    asymptote (3.25) being unreachable; below Z = 1 it is clamped to 4.66.  The rest is
    listed in #1453.

    Provenance
    ----------
    .. [1] S. I. Braginskii, "Transport processes in a plasma", Rev. Plasma Phys. 1
       (1965) 205, Table 2: kappa_perp,e coefficient gamma_1'(Z) (4.66, 4.0, 3.7, 3.6,
       3.25 for Z = 1, 2, 3, 4, inf) and kappa_perp,i coefficient 2.
    .. [2] NRL Plasma Formulary: tau_e = 3.44e5 T_e^1.5 / (n lnLambda),
       tau_i = 2.09e7 T_i^1.5 mu^0.5 / (n lnLambda Z^4) and the ion-ion lnLambda
       (cgs, eV), which the SI forms here reproduce (test_transport_state.py).
    """
    from scipy import constants as c

    from vaft.formula.equilibrium import coulomb_logarithm_from_n_T

    norm = local.normalisation
    ne = float(norm.electron_density)
    te = float(norm.electron_temperature)               # J
    te_ev = te / c.e
    a = float(norm.minor_radius)
    b = abs(float(b_tesla))
    if not (b > 0.0 and np.isfinite(b)):
        # Omega -> 0 makes chi infinite, which json.dumps would write as the
        # non-standard token ``Infinity``; a surface without a field has no baseline.
        raise ValueError(f"b_tesla must be positive and finite, got {b_tesla!r}")
    zeff = float(local.zeff)
    charges = np.asarray(local.zs, dtype=float)[1:]
    fractions = np.asarray(local.as_, dtype=float)[1:]
    z_i = float(charges[0])
    if z_i != float(charges.min()):
        raise ValueError(f"species 1 (z={z_i:g}) is not the lowest-charge ion; it is not the main ion")
    m_i = float(local.mass[1]) * _MASS_DEUTERIUM_KG
    n_i = float(fractions[0]) * ne
    ti = float(local.taus[1]) * te
    ti_ev = ti / c.e
    lnl_e = float(coulomb_logarithm_from_n_T(ne, te_ev))
    # NRL ion-ion, same species: 23 - ln[(Z^2 / T_i) (2 n_i Z^2 / T_i)^(1/2)], cgs/eV.
    lnl_i = 23.0 - np.log(z_i ** 2 / ti_ev * np.sqrt(2.0 * n_i * 1e-6 * z_i ** 2 / ti_ev))
    field_density = float(np.sum(fractions * ne * charges ** 2))   # sum_j n_j Z_j^2

    tau_e = (6.0 * np.sqrt(2.0) * np.pi ** 1.5 * c.epsilon_0 ** 2 * np.sqrt(c.m_e) * te ** 1.5
             / (lnl_e * c.e ** 4 * ne * zeff))
    # Braginskii: tau_i / tau_e = sqrt(2 m_i / m_e) (T_i/T_e)^1.5 at equal density and
    # Z = 1, i.e. 12 here against 6 sqrt(2) above (NRL: 2.09e7 vs 3.44e5).
    tau_i = (12.0 * np.pi ** 1.5 * c.epsilon_0 ** 2 * np.sqrt(m_i) * ti ** 1.5
             / (lnl_i * c.e ** 4 * z_i ** 2 * field_density))
    # anti-alias: not a time series and not a downsample; a lookup in Braginskii's
    # gamma_1'(Z) table at one effective charge.
    gamma1 = float(np.interp(zeff, *zip(*_GAMMA1_PERP)))
    omega_e = c.e * b / c.m_e
    omega_i = z_i * c.e * b / m_i
    chi_e = gamma1 * te / (c.m_e * omega_e ** 2 * tau_e)
    chi_i = 2.0 * ti / (m_i * omega_i ** 2 * tau_i)
    q_e = ne * chi_e * te * float(local.rlts[0]) / a
    q_i = n_i * chi_i * ti * float(local.rlts[1]) / a
    return {
        "r_over_a": float(local.rho),
        "electron_energy_flux_W_m2": float(q_e),
        "electron_particle_flux_m2_s": None,
        "ion_energy_flux_W_m2": {f"z={z_i:g}": float(q_i)},
        "ion_particle_flux_m2_s": {},
        "chi_e_m2_s": float(chi_e), "chi_i_m2_s": float(chi_i),
        "tau_e_s": float(tau_e), "tau_i_s": float(tau_i),
        "coulomb_logarithm": lnl_e, "coulomb_logarithm_ion": float(lnl_i),
        "coulomb_log_valid": bool(te_ev >= 10.0),
        "gamma1_perp": gamma1, "b_tesla": b,
        "model": CLASSICAL_MODEL,
    }
