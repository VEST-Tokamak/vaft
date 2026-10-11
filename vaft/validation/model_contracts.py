"""Which ordering contracts a calculation consumes, from its plasma formalism (#1823).

The ordering contracts of :mod:`vaft.validation.orderings` (#1627) say what a
model assumes; the :class:`vaft.code.formalism.PlasmaFormalism` record of a run
(#1727) says which model the run used. This module joins the two:

* :func:`contracts_for` maps a formalism to the contracts it consumes, as
  ``(contract, role)`` pairs. The role says *how the assumption enters the
  calculation*, not how much of the plasma it covers: it ``governs`` the
  calculation; it holds in the ideal ``outer_solution`` or the resistive
  ``layer_solution`` of a matched calculation (RDCON); it underlies the
  ``fluid_part`` or the ``kinetic_response`` of a hybrid one (DCON kinetic) or
  a kinetic ``closure`` on a given displacement (PENTRC); a reduced model has
  ``inherited`` it from its parent theory without solving that theory (TGLF);
  or local fluxes drive a transport solver (``transport_coupling``). Nothing is
  added to the formalism record: the mapping reads its fields as they are.
* :data:`ADAPTER_MODES` lists the scientifically distinct modes of VAFT's code
  adapters with their formalism, following the audit of
  ``docs/_guide/Plasma_models.md`` (#1725). A mode whose backend builds its own
  record (CGYRO) is taken from that builder, so the two cannot disagree.
* :func:`advisory_ordering_check` evaluates those contracts on a run's input
  state -- the rows :mod:`vaft.process.ordering_state` builds -- with
  :func:`~vaft.validation.applicability.evaluate_contract`, and places each
  result on the credibility ``applicability`` axis with
  :func:`~vaft.validation.applicability.as_evidence`. It is advisory: it reports
  and never blocks a run, and a quantity the input cannot provide (a mode's
  ``k``, a layer width) is ``UNASSESSED``, never a pass. The drift-kinetic contract orders
  ``omega/Omega_ci`` and ``k_perp rho_i``; a steady axisymmetric solve (NEO) or
  a zero-frequency one (DCON kinetic) has no such input, so that contract stays
  ``UNASSESSED`` for them -- a known gap of the input, not of the run.

The module imports neither matplotlib nor pandas, so the guide and the contract
map can render their tables from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .applicability import ContractEvaluation, as_evidence, evaluate_contract
from .orderings import CONTRACTS, ORDERING_QUANTITIES

__all__ = [
    "ADAPTER_MODES",
    "AdapterMode",
    "AdvisoryResult",
    "ContractUse",
    "ROLES",
    "advisory_ordering_check",
    "adapter_mode",
    "contracts_for",
    "ordering_provenance",
    "safe_ordering_provenance",
]

#: How an ordering contract enters a calculation. A role describes the model's
#: structure, not the spatial extent of the plasma it is applied to.
ROLES = (
    "governing",           # the model the calculation solves rests on this contract
    "outer_solution",        # the ideal outer solution of a matched (outer/inner) calculation
    "layer_solution",         # the resistive inner-layer solution of a matched calculation
    "fluid_part",          # the fluid energy principle of a hybrid calculation
    "kinetic_response",    # the kinetic term a hybrid calculation adds to its fluid model
    "closure",             # a kinetic quantity evaluated on a given fluid displacement, not fed back
    "inherited",           # a reduced model derived from this theory, which it does not solve
    "transport_coupling",  # local turbulent fluxes drive a transport solver
)


@dataclass(frozen=True)
class ContractUse:
    """One contract a calculation consumes, and the role it plays there (:data:`ROLES`)."""

    contract: str
    role: str

    def __post_init__(self) -> None:
        if self.contract not in CONTRACTS:
            raise KeyError(f"no ordering contract {self.contract!r}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")


def _fluid(model: Optional[str], operation: str, role: str) -> Tuple[List[ContractUse], List[str]]:
    if model == "ideal_mhd":
        # an ideal-MHD Delta-prime calculation (STRIDE, RDCON without a solved layer) is the outer solution only
        where = "outer_solution" if (operation == "resistive_stability" and role == "governing") else role
        return [ContractUse("ideal_single_fluid_mhd", where)], []
    if model == "resistive_mhd":
        # matched asymptotics: ideal outer solution, single-fluid resistive inner layer (GGJ)
        return [ContractUse("ideal_single_fluid_mhd", "outer_solution"), ContractUse("resistive_mhd", "layer_solution")], []
    if model == "extended_mhd":
        return [ContractUse("hall_mhd", role)], []
    return [], [f"fluid_model={model!r} has no ordering contract"]


def _kinetic(formalism, role: str, transport_coupled: bool = False) -> Tuple[List[ContractUse], List[str]]:
    equation = formalism.kinetic_equation
    operation = formalism.scientific_operation
    if equation == "drift_kinetic":
        uses = [ContractUse("drift_kinetic", role)]
        if operation == "neoclassical_transport" and formalism.spatial_domain == "local":
            uses.append(ContractUse("local_neoclassical", role))
        return uses, []
    if equation == "gyrokinetic":
        if formalism.distribution_formulation != "delta_f":
            return [], [f"gyrokinetics with distribution_formulation={formalism.distribution_formulation!r} "
                        "has no registered contract (only delta-f)"]
        uses = [ContractUse("gyrokinetic_delta_f", role)]
        # A standalone flux-tube run imposes a fixed background; comparing its fluxes with experiment still
        # assumes tau_transp/tau_turb >> 1, but the registry books that contract only when the fluxes drive a
        # transport solver -- a bookkeeping choice, since nothing in the record says how the fluxes are used.
        if transport_coupled and operation == "turbulent_transport":
            uses.append(ContractUse("gyrokinetic_transport_separation", "transport_coupling"))
        return uses, []
    return [], [f"kinetic_equation={equation!r} has no ordering contract"]


def contracts_for(formalism, *, transport_coupled: bool = False
                  ) -> Tuple[Tuple[ContractUse, ...], Tuple[str, ...]]:
    """The contracts a formalism consumes, and why any part of it maps to none.

    Parameters
    ----------
    formalism : vaft.code.formalism.PlasmaFormalism
        The run's record, read as it is.
    transport_coupled : bool
        The run's turbulent fluxes drive a transport solver (a PORTALS-type
        flux-matching loop). Only then is the gyrokinetic-transport
        scale-separation contract consumed. The record carries no such field,
        so the workflow says so.

    Returns
    -------
    (uses, unmapped)
        ``uses`` is a tuple of :class:`ContractUse` in a fixed order, each
        contract and role once; ``unmapped`` names, in words, any part of the
        formalism no registered contract covers (an empty tuple when none). A
        formalism with nothing mapped returns no uses and its reasons -- never a
        guessed contract.
    """
    operation = formalism.scientific_operation
    bulk = formalism.bulk_description
    uses: List[ContractUse] = []
    unmapped: List[str] = []
    if operation == "equilibrium":
        # static force balance j x B = grad p, re-established faster than the state moves
        uses.append(ContractUse("quasi_static_equilibrium", "governing"))
        if bulk != "fluid":
            unmapped.append(f"an equilibrium with bulk_description={bulk!r} is not classified")
    elif bulk == "fluid":
        u, r = _fluid(formalism.fluid_model, operation, "governing")
        uses += u
        unmapped += r
    elif bulk == "hybrid":
        coupling = formalism.kinetic_coupling
        u, r = _fluid(formalism.fluid_model, operation, "fluid_part")
        uses += u
        unmapped += r
        u, r = _kinetic(formalism, "closure" if coupling == "closure" else "kinetic_response", transport_coupled)
        uses += u
        unmapped += r
    elif bulk == "kinetic":
        u, r = _kinetic(formalism, "governing", transport_coupled)
        uses += u
        unmapped += r
    elif bulk == "reduced":
        parent = formalism.derived_from
        if parent in ("drift_kinetic", "gyrokinetic"):
            proxy = _Parent(parent, operation, formalism.spatial_domain, "delta_f")
            u, r = _kinetic(proxy, "inherited", transport_coupled)
        elif parent in ("ideal_mhd", "resistive_mhd", "extended_mhd"):
            u, r = _fluid(parent, operation, "inherited")
        else:
            u, r = [], [f"derived_from={parent!r} names no parent theory with an ordering contract"]
        uses += u
        unmapped += r
        unmapped.append("a reduced model adds its own closure or fit, which no ordering contract covers")
    elif bulk == "particle":
        unmapped.append("test-particle orbit orderings (rho/L_B, slowing down on a fixed background) "
                        "are not a registered contract")
    seen, ordered = set(), []
    for use in uses:
        if use not in seen:
            seen.add(use)
            ordered.append(use)
    return tuple(ordered), tuple(unmapped)


@dataclass(frozen=True)
class _Parent:
    """The parent theory of a reduced model, read with the kinetic rules."""

    kinetic_equation: str
    scientific_operation: str
    spatial_domain: Optional[str]
    distribution_formulation: str


# ---------------------------------------------------------------------------
# the modes of VAFT's adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterMode:
    """A scientifically distinct mode of one code adapter, with its formalism.

    ``formalism`` is ``None`` for a mode the vocabulary does not classify yet;
    ``note`` then says why. ``audit`` names where the classification comes from.
    """

    adapter: str
    mode: str
    family: str
    formalism: Any
    audit: str
    note: str = ""
    #: the mode's fluxes drive a transport solver (see :func:`contracts_for`); no VAFT mode does yet
    transport_coupled: bool = False

    @property
    def key(self) -> str:
        return f"{self.adapter}:{self.mode}"

    def contracts(self) -> Tuple[Tuple[ContractUse, ...], Tuple[str, ...]]:
        if self.formalism is None:
            return (), (self.note,)
        return contracts_for(self.formalism, transport_coupled=self.transport_coupled)


def _record(**fields):
    from vaft.code.formalism import PlasmaFormalism

    return PlasmaFormalism(**fields)


def _equilibrium(solver: str, topology: str):
    # a fixed-boundary solve (CHEASE) stays inside the last closed surface; a free-boundary solve or a
    # reconstruction solves Grad-Shafranov across plasma, vacuum and coils
    return _record(scientific_operation="equilibrium", bulk_description="fluid", fluid_model="ideal_mhd",
                   spatial_domain="whole_volume", topology_domain=topology, regime="static", solver=solver)


_GPEC_COMMON = dict(field_model="mhd_displacement", spatial_domain="whole_volume",
                    topology_domain="closed_flux_surface")


def _dcon_kinetic(operation: str, solver: str):
    return _record(scientific_operation=operation, bulk_description="hybrid", fluid_model="ideal_mhd",
                   kinetic_equation="drift_kinetic", kinetic_coupling="energy", kinetic_population=("thermal_ions",),
                   distribution_formulation="delta_f", orbit_representation="bounce_averaged", regime="static",
                   solver=solver, **_GPEC_COMMON)


def _cgyro(nonlinear: bool):
    from vaft.code.gacode.cgyro import CGYROConfig, plasma_formalism

    return plasma_formalism(CGYROConfig(nonlinear=True, n_toroidal=8) if nonlinear else CGYROConfig())


def _build_modes() -> Tuple[AdapterMode, ...]:
    pm = "Plasma_models.md"
    return (
        AdapterMode("efit", "reconstruction", "equilibrium", _equilibrium("efit", "across_separatrix"), f"{pm} §12"),
        AdapterMode("chease", "fixed_boundary", "equilibrium", _equilibrium("chease", "closed_flux_surface"), f"{pm} §12"),
        AdapterMode("tokamaker", "free_boundary", "equilibrium", _equilibrium("tokamaker", "across_separatrix"), f"{pm} §12"),
        AdapterMode("tes", "free_boundary", "equilibrium", _equilibrium("tes", "across_separatrix"), f"{pm} §12"),
        AdapterMode("nice", "reconstruction", "equilibrium", _equilibrium("nice", "across_separatrix"), f"{pm} §12"),
        AdapterMode("gpec", "dcon_ideal", "mhd_stability",
                    _record(scientific_operation="ideal_stability", bulk_description="fluid", fluid_model="ideal_mhd",
                            regime="static", solver="dcon", **_GPEC_COMMON), f"{pm} §3.1"),
        AdapterMode("gpec", "dcon_kinetic", "mhd_stability", _dcon_kinetic("ideal_stability", "dcon"), f"{pm} §3.2"),
        AdapterMode("gpec", "gpec_ideal", "mhd_stability",
                    _record(scientific_operation="perturbed_equilibrium", bulk_description="fluid",
                            fluid_model="ideal_mhd", regime="static", solver="gpec", **_GPEC_COMMON), f"{pm} §3"),
        AdapterMode("gpec", "gpec_kinetic", "mhd_stability", _dcon_kinetic("perturbed_equilibrium", "gpec"),
                    f"{pm} §3.3"),
        AdapterMode("gpec", "rdcon_rmatch", "mhd_stability",
                    _record(scientific_operation="resistive_stability", bulk_description="fluid",
                            fluid_model="resistive_mhd", regime="linear", solver="rdcon", **_GPEC_COMMON),
                    f"{pm} §3.4"),
        AdapterMode("gpec", "stride", "mhd_stability",
                    _record(scientific_operation="resistive_stability", bulk_description="fluid",
                            fluid_model="ideal_mhd", regime="linear", solver="stride", **_GPEC_COMMON), f"{pm} §3.4"),
        AdapterMode("gpec", "thresholds", "mhd_stability", None, f"{pm} §3 (provisional)",
                    "the singthresh_* layer models (SLAYER, Callen) are auxiliary reduced layer models whose "
                    "classification is unresolved in the audit"),
        AdapterMode("pentrc", "ntv_closure", "mhd_stability",
                    _record(scientific_operation="toroidal_torque", bulk_description="hybrid", fluid_model="ideal_mhd",
                            kinetic_equation="drift_kinetic", kinetic_coupling="closure",
                            kinetic_population=("thermal_ions",), distribution_formulation="delta_f",
                            orbit_representation="bounce_averaged", regime="static", solver="pentrc",
                            **_GPEC_COMMON), f"{pm} §3.5"),
        AdapterMode("gacode", "neo", "neoclassical",
                    _record(scientific_operation="neoclassical_transport", bulk_description="kinetic",
                            kinetic_equation="drift_kinetic", kinetic_population=("all",),
                            distribution_formulation="delta_f", spatial_domain="local",
                            topology_domain="closed_flux_surface", regime="steady_state", solver="neo"), f"{pm} §4"),
        AdapterMode("gacode", "neo_analytic", "neoclassical",
                    _record(scientific_operation="neoclassical_transport", bulk_description="reduced",
                            kinetic_population=("all",), spatial_domain="local", topology_domain="closed_flux_surface",
                            regime="steady_state", derived_from="drift_kinetic", solver="neo"),
                    f"{pm} §4 (Hinton-Hazeltine, Sauter values in a NEO run's theory output)"),
        AdapterMode("gacode", "cgyro_linear", "gyrokinetic", _cgyro(False), f"{pm} §4 (cgyro.plasma_formalism)"),
        AdapterMode("gacode", "cgyro_nonlinear", "gyrokinetic", _cgyro(True), f"{pm} §4 (cgyro.plasma_formalism)"),
        # tglf_linear / tglf: TGLFConfig.use_transport_model False (one mode) / True (quasilinear fluxes)
        AdapterMode("gacode", "tglf_linear", "gyrokinetic",
                    _record(scientific_operation="microstability", bulk_description="reduced",
                            kinetic_population=("all",), spatial_domain="local", topology_domain="closed_flux_surface",
                            regime="linear", derived_from="gyrokinetic", solver="tglf"), f"{pm} §4"),
        AdapterMode("gacode", "tglf", "gyrokinetic",
                    _record(scientific_operation="turbulent_transport", bulk_description="reduced",
                            kinetic_population=("all",), spatial_domain="local", topology_domain="closed_flux_surface",
                            regime="quasilinear", derived_from="gyrokinetic", solver="tglf"), f"{pm} §4"),
        AdapterMode("gacode", "tglf_nn", "gyrokinetic",
                    _record(scientific_operation="turbulent_transport", bulk_description="reduced",
                            kinetic_population=("all",), spatial_domain="local", topology_domain="closed_flux_surface",
                            regime="quasilinear", derived_from="gyrokinetic", solver="tglf_nn"),
                    f"{pm} §4 (provisional; a learned surrogate, valid only in its training domain)"),
        AdapterMode("nubeam", "fast_ions", "fast_ions",
                    _record(scientific_operation="fast_ion_sources", bulk_description="particle",
                            kinetic_equation="fokker_planck", kinetic_coupling="sources",
                            kinetic_population=("beam_ions",), orbit_representation="guiding_center",
                            spatial_domain="whole_volume", regime="time_dependent", solver="nubeam"), f"{pm} §5"),
        AdapterMode("genray", "ray_tracing", "waves", None, "not audited",
                    "ray tracing (WKB) is neither a fluid nor a kinetic bulk description in the vocabulary"),
        AdapterMode("transp", "integrated", "integrated", None, "not audited",
                    "an integrated-modelling run combines several formalisms; its components are classified"),
    )


_MODES_CACHE: Optional[Tuple[AdapterMode, ...]] = None


def _modes() -> Tuple[AdapterMode, ...]:
    global _MODES_CACHE
    if _MODES_CACHE is None:
        _MODES_CACHE = _build_modes()
    return _MODES_CACHE


class _Modes(Sequence):
    """Lazy tuple of :class:`AdapterMode`: the records are built on first use."""

    def __getitem__(self, index):
        return _modes()[index]

    def __len__(self) -> int:
        return len(_modes())

    def __repr__(self) -> str:
        return f"ADAPTER_MODES({', '.join(m.key for m in _modes())})"


#: The audited modes of VAFT's code adapters, each with its formalism.
ADAPTER_MODES: Sequence[AdapterMode] = _Modes()


def adapter_mode(key: str) -> AdapterMode:
    """The mode ``"<adapter>:<mode>"``; ``KeyError`` lists the known ones."""
    for mode in _modes():
        if mode.key == key:
            return mode
    raise KeyError(f"no adapter mode {key!r}; known: {', '.join(m.key for m in _modes())}")


# ---------------------------------------------------------------------------
# advisory check of a run's input
# ---------------------------------------------------------------------------

_STATE_GROUPS = ("global", "time_history")


@dataclass(frozen=True)
class AdvisoryResult:
    """One contract evaluated on a run's input.

    ``evaluation`` is the representative result: the state's own for a
    state-level contract; for a local one, the most constraining point (an
    ``OUTSIDE`` point with the smallest margin if any, else an ``UNASSESSED``
    point if any, else the ``SUPPORTED`` point with the smallest margin).
    ``counts`` gives every point's status, and ``point`` the representative
    point's label (the DataFrame index, else its position). ``evidence`` is its
    :func:`~vaft.validation.applicability.as_evidence`, ``None`` when not
    applicable.
    """

    use: ContractUse
    evaluation: ContractEvaluation
    counts: Mapping[str, int]
    point: Optional[int]
    evidence: Any

    def as_dict(self) -> Dict[str, Any]:
        return {"contract": self.use.contract, "role": self.use.role, "status": self.evaluation.status,
                "limiting": self.evaluation.limiting, "point": self.point, "counts": dict(self.counts),
                "margins": {a.quantity: a.margin for a in self.evaluation.assumptions}}


def _rows(points) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Profile rows and their labels: a DataFrame's index, else positions. Scalars broadcast."""
    if points is None:
        return [], []
    if hasattr(points, "to_dict") and hasattr(points, "columns"):
        return [dict(row) for row in points.to_dict("records")], list(points.index)
    columns: Dict[str, Any] = {}
    for name, value in dict(points).items():
        if isinstance(value, (str, bytes)) or np.ndim(value) == 0:
            columns[name] = value                      # a scalar: the same on every point
        elif np.ndim(value) == 1:
            columns[name] = list(np.asarray(value, dtype=object))
        else:
            raise ValueError(f"profile column {name!r} must be one-dimensional, got shape {np.shape(value)}")
    lengths = {len(v) for v in columns.values() if isinstance(v, list)}
    if len(lengths) > 1:
        raise ValueError(f"profile columns differ in length: {sorted(lengths)}")
    # only scalars: one point, never silently none
    n = lengths.pop() if lengths else (1 if columns else 0)
    rows = [{k: (v[i] if isinstance(v, list) else v) for k, v in columns.items()} for i in range(n)]
    return rows, list(range(n))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _overlay(state: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    """The state with a point's values on top; a NaN at the point never hides the state's finite value."""
    merged = dict(state)
    for name, value in row.items():
        if _finite(value) or name not in merged:
            merged[name] = value
    return merged


def _limiting_margin(result: ContractEvaluation) -> float:
    margins = [a.margin for a in result.assumptions if a.margin == a.margin]
    return min(margins) if margins else math.inf


def _representative(results: List[ContractEvaluation]) -> Tuple[int, ContractEvaluation]:
    for status in ("OUTSIDE", "UNASSESSED", "SUPPORTED", "NOT_APPLICABLE"):
        chosen = [(i, r) for i, r in enumerate(results) if r.status == status]
        if chosen:
            return min(chosen, key=lambda item: _limiting_margin(item[1]))
    raise ValueError("no evaluation")


def advisory_ordering_check(formalism, state: Mapping[str, Any], points: Any = None, *,
                            key: Optional[str] = None, transport_coupled: bool = False
                            ) -> Tuple[Tuple[AdvisoryResult, ...], Tuple[str, ...]]:
    """Evaluate the contracts a run's formalism consumes on that run's input.

    Parameters
    ----------
    formalism : vaft.code.formalism.PlasmaFormalism
        The record of the run about to be launched (or that was launched).
    state : mapping
        The run's global and time-history ordering quantities, by
        :data:`~vaft.validation.orderings.ORDERING_QUANTITIES` name -- what
        :func:`vaft.process.ordering_state.global_ordering_quantities` and
        :func:`~vaft.process.ordering_state.time_history_ordering_quantities`
        return. Mode quantities (``k_perp_rho_i``, ``omega_tau_i``, ...) and
        layer quantities go here too when the run's input defines them.
    points : mapping of arrays or DataFrame, optional
        Profile quantities per flux surface, as
        :func:`~vaft.process.ordering_state.profile_ordering_quantities`
        returns them. A contract with a local ordering is evaluated at every
        point, with the state's quantities attached.
    key : str, optional
        Prefix of each evidence key; default ``"applicability.<solver>"``.
    transport_coupled : bool
        Passed to :func:`contracts_for`: the run's fluxes drive a transport solver.

    Returns
    -------
    (results, unmapped)
        One :class:`AdvisoryResult` per :func:`contracts_for` use, in its
        order, and the formalism's unmapped reasons. The check reports and
        never raises on an ordering that fails; a quantity the input does not
        carry leaves its contract ``UNASSESSED``.
    """
    uses, unmapped = contracts_for(formalism, transport_coupled=transport_coupled)
    prefix = key or f"applicability.{formalism.solver or 'run'}"
    rows, labels = _rows(points)
    results = []
    for use in uses:
        contract = CONTRACTS[use.contract]
        local = any(ORDERING_QUANTITIES[a.quantity].group not in _STATE_GROUPS for a in contract.assumptions)
        if local and rows:
            evaluations = [evaluate_contract(contract, _overlay(state, row)) for row in rows]
            index, evaluation = _representative(evaluations)
            point = labels[index]
        else:
            evaluations = [evaluate_contract(contract, state)]
            point, evaluation = None, evaluations[0]
        counts: Dict[str, int] = {}
        for item in evaluations:
            counts[item.status] = counts.get(item.status, 0) + 1
        evidence = as_evidence(evaluation, key=f"{prefix}.{use.contract}.{use.role}")
        results.append(AdvisoryResult(use, evaluation, counts, point, evidence))
    return tuple(results), unmapped


def _json_number(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_label(label: Any) -> Any:
    if label is None or isinstance(label, (str, bool)):
        return label
    if isinstance(label, (int, np.integer)):
        return int(label)
    number = _json_number(label)
    return number if number is not None and not isinstance(label, tuple) else str(label)


def ordering_provenance(formalism, state: Optional[Mapping[str, Any]] = None, points: Any = None, *,
                        transport_coupled: bool = False,
                        notes: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """The advisory ordering check of one run, as a JSON-ready provenance record (#1823).

    :func:`advisory_ordering_check` on ``state`` and ``points``, reduced to
    plain values an adapter can store in its run provenance: the contracts the
    formalism consumes with their roles, each one's status, limiting ordering,
    per-point status counts and per-ordering margins in decades, and the parts
    of the formalism no contract covers. A margin that has no value (an input
    the run does not carry) is ``None``, never NaN, so the record survives a
    strict JSON writer. State keys that are not ordering quantities are listed
    under ``ignored`` rather than dropped silently, and ``notes`` (an adapter's
    caveats) are carried as given. ``None`` for ``formalism`` gives ``None``:
    no record, no guess.
    """
    if formalism is None:
        return None
    results, unmapped = advisory_ordering_check(formalism, dict(state or {}), points,
                                                transport_coupled=transport_coupled)
    contracts = []
    for result in results:
        record = result.as_dict()
        record["margins"] = {name: _json_number(margin) for name, margin in record["margins"].items()}
        record["point"] = _json_label(record["point"])
        record["values"] = {a.quantity: _json_number(a.value) for a in result.evaluation.assumptions}
        contracts.append(record)
    ignored = sorted(str(k) for k in dict(state or {}) if k not in ORDERING_QUANTITIES)
    return {"schema_version": 1, "advisory": True, "solver": formalism.solver,
            "transport_coupled": bool(transport_coupled), "contracts": contracts, "unmapped": list(unmapped),
            "ignored": ignored, "notes": [str(n) for n in notes]}


def safe_ordering_provenance(formalism, state: Optional[Mapping[str, Any]] = None, points: Any = None,
                             **kwargs) -> Optional[Dict[str, Any]]:
    """:func:`ordering_provenance` that never raises: an advisory record must not cost a run its result.

    An exception is recorded as ``{"advisory": True, "error": ...}`` in place of
    the check.
    """
    try:
        return ordering_provenance(formalism, state, points, **kwargs)
    except Exception as error:  # noqa: BLE001 - provenance is advisory, the run's result is not
        return {"schema_version": 1, "advisory": True, "error": f"{type(error).__name__}: {error}"}
