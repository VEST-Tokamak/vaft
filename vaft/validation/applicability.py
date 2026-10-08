"""Approximation applicability: whether a model's orderings hold for a state (#1639 A, #1628).

A reduced model is derived under orderings -- ``S >> 1``, ``d_i/a << 1``,
``rho_i/L_T << 1``, ``tau_evol/tau_A >> 1`` -- and a result computed with it is
only as credible as those orderings are for the state it was applied to.  This
module evaluates that, and only that.

Ownership
---------
This module owns the *evaluation semantics*: how an ordering is turned into a
margin and a status, how missing inputs are reported, and how per-assumption
results compose into one contract result.  It owns no physics:

* the ordering quantities themselves (``S``, ``d_i/L``, ``Kn``, ...) are defined
  and computed by the formula/process layers (#1627);
* which assumptions a given model makes -- the contracts -- are declared beside
  the model they describe (#1627 for the ordering families, #1628 for formulas,
  codes and workflows) as :class:`ApproximationContract` objects;
* plotting the result is :mod:`vaft.plot` (#1664).

Margins first, labels second
----------------------------
The primary output is the continuous **ordering margin** (#1627):

$$m_x = -\\log_{10} x \\quad (x \\ll 1), \\qquad m_y = \\log_{10} y \\quad (y \\gg 1)$$

measured against the assumption's threshold, so ``m > 0`` means the ordering is
satisfied with ``m`` decades to spare and ``m < 0`` that it is violated.  The
default threshold is ``1``: the point where the expansion parameter is of order
unity and the ordering, by its own definition, no longer separates the scales.
That is the only cutoff this module will apply on its own.  Any other threshold
must carry its source in :attr:`OrderingAssumption.threshold_source`; none is
invented here (#1639 §12).

Status vocabulary
-----------------
:data:`APPLICABILITY_STATUSES` is the vocabulary Lane V's renderer uses for an
operational-space boundary (#1664), kept identical so a contract result and a
drawn boundary say the same thing in the same words:

``SUPPORTED``
    every assumption was evaluated and lies on its permitted side;
``OUTSIDE``
    at least one evaluated assumption lies on the wrong side;
``UNASSESSED``
    nothing is outside, but at least one assumption could not be evaluated --
    its quantity is missing, non-finite or non-positive -- or the state does
    not say whether the contract's scope covers it;
``NOT_APPLICABLE``
    the contract does not apply to this state (its scope excludes it).

``SUPPORTED`` means only "on the permitted side of the threshold": a state at
``x = 0.99`` against a threshold of 1 is supported with a margin of 0.004
decades.  The label carries no notion of *how well* an ordering holds -- read
the margin for that.

There is no ``MARGINAL``: a margin band would be a cutoff nobody derived.  The
margin is there for anyone who wants to look at how close a state is.
:func:`as_evidence` places a result on the credibility axes of
:mod:`vaft.validation.credibility` (``SUPPORTED`` -> pass, ``OUTSIDE`` -> fail,
``UNASSESSED`` -> not_available).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .model import ValidationStatus

__all__ = [
    "APPLICABILITY_STATUSES",
    "ORDERINGS",
    "SCOPES",
    "ApproximationContract",
    "AssumptionResult",
    "ContractEvaluation",
    "OrderingAssumption",
    "as_evidence",
    "evaluate_contract",
    "evaluate_population",
    "ordering_margin",
    "successive_discrepancy",
]

#: Identical to ``vaft.plot.operational_space.APPLICABILITY_STATUSES`` (#1664);
#: a test asserts it once that tuple exists on the branch under test.  Kept here rather than imported because the validation
#: core must not import the plotting layer.
APPLICABILITY_STATUSES = ("SUPPORTED", "OUTSIDE", "UNASSESSED", "NOT_APPLICABLE")

#: Which side of the threshold an ordering requires.
ORDERINGS = ("small", "large")

#: Where a quantity is defined (#1627 "Summary layer").  A global ``d_i/a`` and
#: a layer-local ``d_i/delta`` are different quantities, so the scope is part
#: of the assumption, not a comment on it.
SCOPES = ("global", "local", "profile", "edge", "time_history", "event", "mode")

_TO_VALIDATION = {
    "SUPPORTED": ValidationStatus.PASS,
    "OUTSIDE": ValidationStatus.FAIL,
    "UNASSESSED": ValidationStatus.NOT_AVAILABLE,
}


@dataclass(frozen=True)
class OrderingAssumption:
    """One asymptotic assumption a model makes: ``quantity`` is ``small`` or ``large``.

    Parameters
    ----------
    quantity
        The column or key the state carries the dimensionless value under,
        e.g. ``"ion_skin_depth_over_L"``.  Its definition belongs to whoever
        computes it; this names it.
    ordering
        ``"small"`` (``x << threshold``) or ``"large"`` (``x >> threshold``).
    scale
        The characteristic scale the quantity is normalized by (``"a"``,
        ``"L_Te"``, ``"delta_layer"``, ...).  Required, so an ordering on
        ``a`` can never stand in for one on ``L_T`` unnoticed.
    scope
        One of :data:`SCOPES`.
    threshold
        The value at which the ordering is lost.  ``1`` is the order-unity
        point of the expansion and needs no source; anything else does.
    threshold_source
        Where a non-unit threshold comes from (a reference, an equation).
    meaning
        What the assumption buys the model, in one phrase.
    """

    quantity: str
    ordering: str
    scale: str
    scope: str = "global"
    threshold: float = 1.0
    threshold_source: str = ""
    meaning: str = ""

    def __post_init__(self) -> None:
        if self.ordering not in ORDERINGS:
            raise ValueError(f"ordering must be one of {ORDERINGS}, got {self.ordering!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        if not self.scale:
            raise ValueError(f"{self.quantity}: an ordering needs its characteristic scale")
        threshold = float(self.threshold)
        if not (math.isfinite(threshold) and threshold > 0):
            raise ValueError(f"{self.quantity}: threshold must be finite and positive, got {self.threshold}")
        if threshold != 1.0 and not self.threshold_source:
            raise ValueError(
                f"{self.quantity}: a threshold other than order unity needs a threshold_source "
                "(#1639 s12: no invented cutoffs)"
            )
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class ApproximationContract:
    """The orderings one model or approximation relies on, with where they come from.

    The assumptions are kept separate on purpose: ideal single-fluid MHD needs
    ``S >> 1`` *and* ``d_i/L << 1`` *and* ``rho_i/L << 1`` *and*
    ``tau_evol/tau_A >> 1``, and these test different things (#1627).  A
    contract never reduces them to one condition.

    ``applies_to`` optionally restricts the contract to states whose named
    fields hold one of the listed values (e.g. ``{"phase": ("flat",)}``); a
    state outside it is ``NOT_APPLICABLE``, not ``OUTSIDE``, and a state that
    does not carry the field at all, or carries it as something other than
    one scalar (a list, an array), is ``UNASSESSED``.
    """

    name: str
    physical_model: str
    assumptions: tuple[OrderingAssumption, ...]
    references: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    applies_to: Mapping[str, tuple] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        assumptions = tuple(self.assumptions)
        if not assumptions:
            raise ValueError(f"{self.name}: a contract needs at least one assumption")
        names = [item.quantity for item in assumptions]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: each quantity may appear once, got {names}")
        scope = {}
        for key, allowed in dict(self.applies_to).items():
            if isinstance(allowed, str) or not isinstance(allowed, (tuple, list, set, frozenset)):
                raise TypeError(f"{self.name}: applies_to[{key!r}] must be a collection of values, "
                                f"got {allowed!r}")
            scope[key] = tuple(allowed)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        object.__setattr__(self, "applies_to", MappingProxyType(scope))


@dataclass(frozen=True)
class AssumptionResult:
    """One assumption evaluated on one state."""

    quantity: str
    value: float
    margin: float
    status: str
    reason: str = ""


@dataclass(frozen=True)
class ContractEvaluation:
    """A contract evaluated on one state: every assumption, then the composition.

    ``limiting`` is the evaluated assumption with the smallest margin -- the
    one that would break first -- or ``None`` when nothing could be evaluated.
    """

    contract: str
    status: str
    assumptions: tuple[AssumptionResult, ...]
    limiting: str | None
    reason: str = ""

    @property
    def unassessed(self) -> tuple[str, ...]:
        """Quantities that could not be evaluated on this state."""
        return tuple(item.quantity for item in self.assumptions if item.status == "UNASSESSED")

    def as_dict(self) -> dict[str, Any]:
        """Plain fields; the per-assumption results as ``{quantity: {...}}``."""
        return {
            "contract": self.contract,
            "status": self.status,
            "limiting": self.limiting,
            "reason": self.reason,
            "assumptions": {
                item.quantity: {"value": item.value, "margin": item.margin,
                                "status": item.status, "reason": item.reason}
                for item in self.assumptions
            },
        }


def _missing(value: Any) -> bool:
    """``None``, pandas' ``NA``/``NaT`` or any float-convertible NaN."""
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def ordering_margin(value: Any, ordering: str, threshold: float = 1.0) -> float:
    """Decades by which ``value`` satisfies an ordering against ``threshold``.

    ``log10(threshold / value)`` when the quantity must be small,
    ``log10(value / threshold)`` when it must be large.  Positive means
    satisfied, zero is the threshold itself, negative is violated.  A missing,
    non-finite or non-positive value -- for which no ordering statement can be
    made -- gives ``nan``.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    number = _float(value)
    if not (math.isfinite(number) and number > 0):
        return math.nan
    ratio = threshold / number if ordering == "small" else number / threshold
    return math.log10(ratio)


def _lookup(state: Any, key: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(key)
    getter = getattr(state, "get", None)
    if callable(getter):
        return getter(key)
    return getattr(state, key, None)


def _is_scalar(value: Any) -> bool:
    """One categorical value: a string, or anything without a length (a 0-d array counts)."""
    if isinstance(value, (str, bytes)):
        return True
    try:
        len(value)
    except TypeError:
        return True
    return False


def _applies(contract: ApproximationContract, state: Any) -> tuple[str, str]:
    """``("", "")`` in scope, else the status the scope test decides and why.

    A scope field is one categorical scalar.  A list or array in it (even of
    one element) cannot be placed in or out of scope, so it is ``UNASSESSED``
    with a reason -- never a silent ``NOT_APPLICABLE`` for a list, nor an
    exception out of a population evaluation for an array.
    """
    for key, allowed in contract.applies_to.items():
        value = _lookup(state, key)
        if _missing(value):
            return "UNASSESSED", f"{key} not provided, so the contract's scope {allowed} cannot be decided"
        if not _is_scalar(value):
            return "UNASSESSED", (f"{key}={value!r} is not one scalar value, so the contract's scope "
                                  f"{allowed} cannot be decided")
        if value not in allowed:
            return "NOT_APPLICABLE", f"{key}={value!r} is outside the contract's scope {allowed}"
    return "", ""


def evaluate_contract(contract: ApproximationContract, state: Any) -> ContractEvaluation:
    """Evaluate every assumption of ``contract`` on one state.

    ``state`` is anything keyed by quantity name -- a dict, a pandas row, an
    object with attributes.  A quantity the state does not carry, or carries as
    non-finite or non-positive, is ``UNASSESSED`` with its reason; it never
    removes the state and never counts as a violation.  The contract status is
    ``OUTSIDE`` if any evaluated assumption is violated (a violation is decided
    whatever else is missing), else ``UNASSESSED`` if any is unevaluated, else
    ``SUPPORTED``.  A margin of exactly zero is on neither side and counts as
    ``OUTSIDE``.

    A state that does not carry a field ``applies_to`` names is ``UNASSESSED``
    whatever its margins say -- they are still reported -- because whether
    they bear on the contract is unknown.
    """
    scope_status, out_of_scope = _applies(contract, state)
    if scope_status == "NOT_APPLICABLE":
        results = tuple(
            AssumptionResult(item.quantity, _float(_lookup(state, item.quantity)), math.nan,
                             "NOT_APPLICABLE", out_of_scope)
            for item in contract.assumptions
        )
        return ContractEvaluation(contract.name, "NOT_APPLICABLE", results, None, out_of_scope)

    results = []
    for item in contract.assumptions:
        raw = _lookup(state, item.quantity)
        value = _float(raw)
        margin = ordering_margin(value, item.ordering, item.threshold)
        if math.isnan(margin):
            reason = "not provided" if _missing(raw) else f"value {raw!r} admits no ordering statement"
            results.append(AssumptionResult(item.quantity, value, margin, "UNASSESSED", reason))
        elif margin > 0:
            results.append(AssumptionResult(item.quantity, value, margin, "SUPPORTED"))
        else:
            side = "<<" if item.ordering == "small" else ">>"
            results.append(AssumptionResult(
                item.quantity, value, margin, "OUTSIDE",
                f"{item.quantity} = {value:.3g} violates {item.quantity} {side} {item.threshold:g}"))

    evaluated = [item for item in results if item.status in ("SUPPORTED", "OUTSIDE")]
    limiting = min(evaluated, key=lambda item: item.margin).quantity if evaluated else None
    statuses = {item.status for item in results}
    if scope_status == "UNASSESSED":
        return ContractEvaluation(contract.name, "UNASSESSED", tuple(results), limiting, out_of_scope)
    if "OUTSIDE" in statuses:
        status = "OUTSIDE"
    elif "UNASSESSED" in statuses:
        status = "UNASSESSED"
    else:
        status = "SUPPORTED"
    return ContractEvaluation(contract.name, status, tuple(results), limiting)


def evaluate_population(contract: ApproximationContract, table: Any, *,
                        keep: Sequence[str] = ()) -> Any:
    """Evaluate ``contract`` on every row of a table; one output row per input row.

    The result keeps the input index and the ``keep`` columns, then
    ``status``, ``limiting``, and per assumption ``<quantity>``,
    ``<quantity>_margin`` and ``<quantity>_status`` -- the individual checks are
    never hidden behind the aggregate (#1629 §8).  No row is dropped for a
    missing quantity.
    """
    import pandas as pd

    rows = []
    for index, row in table.iterrows():
        result = evaluate_contract(contract, row)
        record = {name: row.get(name) for name in keep}
        record.update(status=result.status, limiting=result.limiting)
        for item in result.assumptions:
            record[item.quantity] = item.value
            record[f"{item.quantity}_margin"] = item.margin
            record[f"{item.quantity}_status"] = item.status
        rows.append((index, record))
    frame = pd.DataFrame([record for _, record in rows], index=[index for index, _ in rows])
    frame.attrs["contract"] = contract.name
    frame.attrs["physical_model"] = contract.physical_model
    return frame


def as_evidence(result: ContractEvaluation, *, key: str | None = None):
    """Place a contract result on the ``applicability`` credibility axis.

    ``SUPPORTED`` is a pass, ``OUTSIDE`` a fail, ``UNASSESSED`` not available;
    a ``NOT_APPLICABLE`` result is not evidence about this state and returns
    ``None``.  The margins travel in ``metrics``.
    """
    from .credibility import Evidence

    if result.status == "NOT_APPLICABLE":
        return None
    metrics = {item.quantity: {"value": item.value, "margin": item.margin, "status": item.status}
               for item in result.assumptions}
    reasons = tuple(item.reason for item in result.assumptions if item.reason)
    return Evidence("applicability", key or f"applicability.{result.contract}",
                    _TO_VALIDATION[result.status], metrics, reasons)


def successive_discrepancy(values: Iterable[Any], *, reference: Any = None) -> list[float]:
    """Relative change of an observable along a model or representation hierarchy.

    For ``Q_0, Q_1, ..., Q_n`` from successively fuller models (or moment
    orders, #1639 s11) this returns ``|Q_{k+1} - Q_k| / |Q_ref|`` for each
    step, with ``Q_ref`` the last (fullest) value unless ``reference`` is
    given.  Paired with the ordering margin it is the strongest applicability
    evidence #1639 s12 names: the ordering parameter *and* the convergence it
    predicts.  A step involving a non-finite value, or a zero reference, is
    ``nan``.
    """
    sequence = [_float(value) for value in values]
    if len(sequence) < 2:
        return []
    ref = _float(reference) if reference is not None else sequence[-1]
    out = []
    for before, after in zip(sequence[:-1], sequence[1:]):
        if not (math.isfinite(before) and math.isfinite(after) and math.isfinite(ref)) or ref == 0:
            out.append(math.nan)
        else:
            out.append(abs(after - before) / abs(ref))
    return out
