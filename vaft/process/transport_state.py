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

Applicability
-------------
VEST-specific: the default ion-temperature policy and composition (H+ with C6+ at
Z_eff = 2) are VEST's, read through :mod:`vaft.machine_mapping.core_profiles`.  The
mechanics are machine-independent and every policy value is an argument.
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
    "EFIT_LABELS",
    "EFIT_LINEAGES",
    "RESOLVER_VERSION",
    "ReadinessReport",
    "ResolvedTransportState",
    "SurfaceReadiness",
    "TransportStateKey",
    "assess_neo_readiness",
    "assess_tglf_readiness",
    "physics_parameters",
    "resolve_transport_state",
    "run_identity",
]

#: Bumped whenever a change here can alter a resolved state; part of every identity.
RESOLVER_VERSION = 1

#: The two EFIT lineages a Tier A state can come from (lane N uses the same words).
EFIT_LINEAGES = ("magnetics-only", "electron-kinetic")

#: Only these #1331 slice labels are resolved; an unreconstructible slice is refused.
EFIT_LABELS = ("good", "admissible")

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

    The provisional state key of the conference lanes until lane K's contract
    replaces it.  ``time_efit_s`` is the equilibrium slice time in seconds.
    """

    shot: int
    time_efit_s: float
    efit_lineage: str

    def __post_init__(self) -> None:
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
    efit_label: str
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

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def ti_lineage(self) -> str:
        return str(self.ti.get("lineage", "unresolved"))

    def identity_payload(self) -> dict[str, Any]:
        """Everything that, changed, makes this a different state."""
        return {
            "resolver_version": RESOLVER_VERSION,
            "key": self.key.as_dict(),
            "times": dict(self.times),
            "ti": {k: self.ti.get(k) for k in ("lineage", "method", "ratio", "sigma")},
            "composition": dict(self.composition),
            "inputs": dict(self.inputs),
            "settings": dict(self.settings),
        }

    @property
    def identity(self) -> str:
        """sha256 of :meth:`identity_payload` (the upstream half of every run identity)."""
        return _digest(self.identity_payload())

    def summary(self) -> dict[str, Any]:
        """A JSON-ready record of the state without the profile arrays."""
        return {
            **self.key.as_dict(),
            "efit_label": self.efit_label,
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


def _digest(payload: Any) -> str:
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(ods: Any, path: str) -> Any:
    """Read a leaf without creating it (omas materializes a missing path on read)."""
    try:
        if path not in ods:
            return None
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return ods[path]


def _times(ods: Any, path: str) -> Optional[np.ndarray]:
    values = _get(ods, path)
    if values is None:
        return None
    array = np.atleast_1d(np.asarray(values, dtype=float))
    return array if array.size else None


def _slice_times(ods: Any, ids: str, aos: str) -> Optional[np.ndarray]:
    """Per-slice times, from ``<ids>.time`` or else each slice's own ``time`` leaf."""
    times = _times(ods, f"{ids}.time")
    if times is not None:
        return times
    collected = []
    index = 0
    while True:
        value = _get(ods, f"{ids}.{aos}.{index}.time")
        if value is None:
            break
        collected.append(float(value))
        index += 1
    return np.asarray(collected, dtype=float) if collected else None


def _nearest(times: Optional[np.ndarray], target: float) -> tuple[Optional[int], float]:
    if times is None or not times.size:
        return None, float("nan")
    index = int(np.argmin(np.abs(times - target)))
    return index, float(times[index] - target)


def _insufficient(base: dict[str, Any], *reasons: str) -> ResolvedTransportState:
    return ResolvedTransportState(status="insufficient", reasons=tuple(reasons), **base)


# --------------------------------------------------------------------------- resolve


def resolve_transport_state(
    ods: Any,
    key: TransportStateKey,
    *,
    efit_label: str,
    quality_source: str = "criteria",
    tolerance: float = DEFAULT_TIME_TOLERANCE_S,
    ti_te_ratio: Any = "policy",
    ti_te_ratio_sigma: Optional[float] = None,
    inferred_ti: Optional[Mapping[str, Any]] = None,
    z_eff: Optional[float] = 2.0,
    impurity: Optional[str] = "C",
    rho_max: Optional[float] = DEFAULT_RHO_MAX,
    inputs: Optional[Mapping[str, Any]] = None,
) -> ResolvedTransportState:
    """Resolve one plasma state into a shared GACODE profile with per-quantity provenance.

    Parameters
    ----------
    ods : omas.ODS
        Carries ``equilibrium`` and ``core_profiles`` for the shot [-].  Never modified.
    key : TransportStateKey
        ``(shot, time_efit_s, efit_lineage)``; ``time_efit_s`` must be an equilibrium
        slice time within ``tolerance`` [s].
    efit_label : str
        The #1331 slice label, ``"good"`` or ``"admissible"`` [-].  Anything else is
        refused with :class:`ValueError`: an unreconstructible slice is not a state.
    quality_source : str
        Where ``efit_label`` came from, e.g. ``"criteria"`` or
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
        A pressure-partition result (#1426): ``{"temperature": array [eV], "method":
        str, ...}`` on the core_profiles grid of the matched slice [eV].  Preferred
        over the policy fallback when given.
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

    Processing
    ----------
    1. Match the equilibrium slice to ``time_efit_s`` and the core_profiles slice to
       that equilibrium time, both by time within ``tolerance``.
    2. Resolve the ion temperature: measured, then ``inferred_ti``, then the policy
       ratio, else insufficient; write a main ion into a deep copy when needed.
    3. Convert with :func:`vaft.code.gacode.inputs.prepare_gacode_profile`, which
       applies the composition closure and refuses non-positive profiles.

    Applicability
    -------------
    VEST-specific: the policy default reads VEST's ``vest.yaml``; the composition
    default is the VEST modelling policy (H+, C6+, Z_eff = 2).
    """
    if efit_label not in EFIT_LABELS:
        raise ValueError(
            f"efit_label must be one of {EFIT_LABELS}; got {efit_label!r}. "
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
        "efit_label": efit_label,
        "quality_source": quality_source,
        "inputs": dict(inputs or {}),
        "settings": settings,
    }

    # 1. times ---------------------------------------------------------------------
    eq_times = _slice_times(ods, "equilibrium", "time_slice")
    eq_index, eq_offset = _nearest(eq_times, key.time_efit_s)
    if eq_index is None:
        return _insufficient(base, "no_equilibrium")
    if abs(eq_offset) > tolerance:
        return _insufficient(base, "no_equilibrium_within_tolerance")
    eq_time = float(eq_times[eq_index])
    cp_times = _slice_times(ods, "core_profiles", "profiles_1d")
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
    measured = [i for i in ions if _get(ods, f"{prefix}.ion.{i}.temperature") is not None]
    hierarchy: list[dict[str, Any]] = []
    work = ods
    if ions and len(measured) == len(ions):
        ti = {"lineage": "measured", "method": "core_profiles ion temperature",
              "ratio": None, "sigma": None, "kind": "measured"}
        hierarchy.append({"step": "measured", "status": "used"})
    else:
        hierarchy.append({"step": "measured", "status": "unavailable"})
        if len(ions) > 1:
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
            ti = {"lineage": f"inferred_{inferred_ti.get('method', 'pressure_partition')}",
                  "method": str(inferred_ti.get("method", "equilibrium_pressure_partition")),
                  "ratio": None, "sigma": None, "kind": "inferred"}
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
            ti = {"lineage": f"{status}_ti_te_{ratio:g}", "method": "ti_te_ratio",
                  "ratio": ratio, "sigma": sigma, "kind": status, "source": source}
        work = copy.deepcopy(ods)
        ion = f"{prefix}.ion.0"
        if not ions:
            work[f"{ion}.label"] = _HYDROGEN["label"]
            work[f"{ion}.z_ion"] = _HYDROGEN["z"]
            work[f"{ion}.element.0.z_n"] = _HYDROGEN["z"]
            work[f"{ion}.element.0.a"] = _HYDROGEN["a"]
            # A placeholder density: the composition closure below rederives both ion
            # densities from n_e, and without a closure a lone H+ at n_e is
            # quasi-neutrality, which is what is recorded.
            work[f"{ion}.density_thermal"] = ne
        work[f"{ion}.temperature"] = temperature
    ti["hierarchy"] = hierarchy
    base["ti"] = ti

    # 3. composition and conversion -------------------------------------------------
    from vaft.code.gacode.inputs import ProfileConversionError, prepare_gacode_profile

    try:
        profile = prepare_gacode_profile(
            work,
            time=eq_time,
            tolerance=max(float(tolerance), 1e-9),
            rho_max=rho_max,
            z_eff=z_eff if impurity is not None else None,
            impurity=impurity,
            shot=int(key.shot),
        )
    except ProfileConversionError as error:
        return _insufficient(base, f"profile_conversion: {error}")

    provenance = dict(getattr(profile, "provenance", {}) or {})
    provenance["ti"] = {"kind": ti["kind"], "method": ti["method"],
                        **({"source": ti["source"]} if "source" in ti else {}),
                        **({"ratio": ti["ratio"], "sigma": ti["sigma"]}
                           if ti.get("ratio") is not None else {})}
    provenance["equilibrium"] = {"kind": "reconstructed", "lineage": key.efit_lineage,
                                 "label": efit_label, "quality_source": quality_source}
    provenance["q"] = {"kind": "reconstructed", "source": "equilibrium"}
    provenance["magnetic_shear"] = {"kind": "derived", "source": "equilibrium q"}
    profile.provenance = provenance
    composition = {
        "species": list(getattr(profile, "name", []) or []),
        "z_eff": settings["z_eff"],
        "impurity": impurity,
        "origin": "policy_assumption" if impurity is not None else "measured_or_single_ion",
    }
    return ResolvedTransportState(
        status="resolved", profile=profile, composition=composition,
        provenance=provenance, **base,
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
        conditions.append(f"efit_label_from_{state.quality_source}")
    return tuple(conditions)


def _surface_code(error: Exception) -> str:
    text = str(error)
    if "outside the converted profile" in text or "strictly inside" in text:
        return "outside_profile_domain"
    if "not positive" in text:
        return "non_positive_profile"
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

    Processing
    ----------
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


def assess_neo_readiness(state: ResolvedTransportState) -> ReadinessReport:
    """Decide whether a resolved state can go to NEO.

    Parameters
    ----------
    state : ResolvedTransportState
        From :func:`resolve_transport_state` [-].

    Returns
    -------
    ReadinessReport
        ``ready``/``conditional``/``insufficient``; NEO is a profile code, so no
        per-surface entries [-].

    Processing
    ----------
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
    """The physics half of a GACODE config: runtime-only fields dropped.

    ``backend``, ``timeout``, ``env``, ``home``, ``executable`` and the MPI/OMP counts
    decide *where* a run happens, not *what* it computes, so they stay out of identity.
    """
    from dataclasses import fields as dataclass_fields

    runtime = {"backend", "timeout", "env", "home", "executable", "workdir", "args",
               "platform", "n_mpi", "n_omp", *exclude}
    out = {}
    for entry in dataclass_fields(config):
        if entry.name in runtime:
            continue
        out[entry.name] = _jsonable(getattr(config, entry.name))
    return out
