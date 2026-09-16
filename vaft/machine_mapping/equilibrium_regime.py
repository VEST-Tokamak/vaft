"""What operating regime one reconstructed slice belongs to (issue #76).

A reconstruction outcome is only interpretable against the state the plasma
was in.  "High-current shots do not converge" and "ramp-up equilibria have
unreliable pressure" are claims about cohorts, and until the cohorts are
defined by a stated rule they cannot be checked.

The rule lives here, in configuration, for the reason every other machine fact
does: it is a choice about VEST, and :mod:`vaft.validation` should be able to
apply it without knowing which machine it is applying it to.

Why the phase rule is a **level** and not a slope
-------------------------------------------------

Two rules existed in ``workflow/`` when this was written, and on the same
abscissa they agree on 295 of 300 reference slices -- the disagreements are
all at the ramp/flat boundary, where an arbitrary threshold has to choose
something.  So the choice was made on what sets the threshold's scale, which
was measured over 39915, 41524 and 41672:

============  ==================================  ====================
rule          scale set by                        measured spread
============  ==================================  ====================
slope         that shot's ``max|dIp/dt|``         2.73 / 9.38 / 2.34 MA/s
level         that shot's peak ``|Ip|``           --
absolute      a fixed MA/s                        cohort size 4 to 24
============  ==================================  ====================

The slope rule's flat-top criterion is therefore set by how violent the
*termination* was, which varies 3.4x across three discharges and has nothing
to do with how flat the flat top is.

An absolute threshold was tried on the expectation that it would transfer
across shots better, since a cohort study needs "flat" to mean the same thing
everywhere.  **It measured worse**, and so did normalising by peak current:
the median ``|Ip|/peak`` inside the flat cohort spread 0.12 and 0.10 across
the three shots, against 0.02 for either shot-normalised rule.  Flatness is
intrinsically relative -- the plasma is near *its own* peak -- and an absolute
slope conflates a fast small discharge with a slow large one.

The cost of the level rule is stated rather than hidden: it needs the peak, so
it is **not causal** and cannot label a discharge while it is running.  This
is a post-hoc classification and nothing here should be wired to real time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from .utils import VestConfigurationError, _policy_document, _resolve_info_file_path

__all__ = [
    "EquilibriumRegimePolicy",
    "vest_equilibrium_regime_policy",
]

#: The phase rules this resolver knows how to apply.  ``level`` is the one
#: VEST uses; ``slope`` is kept because two studies were written against it
#: and an A/B has to be able to ask for it by name.
PHASE_RULES = ("level", "slope")

_STATUSES = ("assumed", "measured", "inferred")


@dataclass(frozen=True)
class EquilibriumRegimePolicy:
    """The stated rule for putting a slice in a cohort."""

    phase_rule: str
    flat_fraction: float
    slope_fraction: float
    vacuum_current_amperes: float
    topology_labels: Mapping[str, str]
    status: Mapping[str, str]
    provenance: Mapping[str, str]
    source: str = "vest.yaml:equilibrium_regime"

    def topology(self, limloc: Any) -> str:
        """The topology name for an EFIT ``limloc``, or ``unknown``.

        EFIT writes a four-character field naming where the boundary is
        defined.  It is mapped rather than interpreted: an unrecognised value
        is ``unknown``, which is a distinct answer from any of the known ones
        and must not be read as "not diverted" (issue #76 requires exactly
        that separation).
        """
        if limloc is None:
            return "unknown"
        key = str(limloc).strip().upper()
        if not key:
            return "unknown"
        return self.topology_labels.get(key, "unknown")


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise VestConfigurationError(f"{where} has no '{key}'")
    return block[key]


def _status(block: Mapping[str, Any], where: str) -> str:
    value = str(_require(block, "status", where))
    if value not in _STATUSES:
        raise VestConfigurationError(
            f"{where}.status must be one of {', '.join(_STATUSES)}, got {value!r}"
        )
    return value


def _fraction(block: Mapping[str, Any], key: str, where: str) -> float:
    value = float(_require(block, key, where))
    if not 0.0 < value <= 1.0:
        raise VestConfigurationError(f"{where}.{key} must be in (0, 1], got {value}")
    return value


def _build(document: Mapping[str, Any]) -> EquilibriumRegimePolicy:
    where = "equilibrium_regime"
    block = document.get(where)
    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"vest.yaml has no '{where}' block")

    phase = block.get("phase") or {}
    if not isinstance(phase, Mapping):
        raise VestConfigurationError(f"{where}.phase must be a mapping")
    rule = str(_require(phase, "rule", f"{where}.phase"))
    if rule not in PHASE_RULES:
        raise VestConfigurationError(
            f"{where}.phase.rule must be one of {', '.join(PHASE_RULES)}, got {rule!r}"
        )
    flat_fraction = _fraction(phase, "flat_fraction_of_peak", f"{where}.phase")
    slope_fraction = _fraction(phase, "slope_fraction_of_peak_rate", f"{where}.phase")
    vacuum = float(_require(phase, "vacuum_current_amperes", f"{where}.phase"))
    if vacuum < 0.0:
        raise VestConfigurationError(
            f"{where}.phase.vacuum_current_amperes must be non-negative, got {vacuum}"
        )

    topology = block.get("topology") or {}
    if not isinstance(topology, Mapping):
        raise VestConfigurationError(f"{where}.topology must be a mapping")
    labels = {
        str(key).strip().upper(): str(value)
        for key, value in (topology.get("limloc") or {}).items()
    }
    if not labels:
        raise VestConfigurationError(f"{where}.topology.limloc lists no values")

    status = {
        "phase": _status(phase, f"{where}.phase"),
        "topology": _status(topology, f"{where}.topology"),
    }
    provenance = {
        "phase": str(_require(phase, "provenance", f"{where}.phase")),
        "topology": str(_require(topology, "provenance", f"{where}.topology")),
    }
    return EquilibriumRegimePolicy(
        phase_rule=rule,
        flat_fraction=flat_fraction,
        slope_fraction=slope_fraction,
        vacuum_current_amperes=vacuum,
        topology_labels=labels,
        status=status,
        provenance=provenance,
    )


@lru_cache(maxsize=8)
def _cached(info_file: str | None) -> EquilibriumRegimePolicy:
    return _build(_policy_document(_resolve_info_file_path(info_file)))


def vest_equilibrium_regime_policy(
    *, info_file: str | None = None
) -> EquilibriumRegimePolicy:
    """The regime rule VEST reconstructions are classified by.

    Not keyed by shot: a cohort definition that changed between discharges
    could not group them, which is the whole point of having one.
    """
    return _cached(info_file)
