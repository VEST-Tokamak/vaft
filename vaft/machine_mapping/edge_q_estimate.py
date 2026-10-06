"""How VEST estimates an edge safety factor without an equilibrium (issue #1583).

Most VEST discharges have no converged equilibrium, so their q95 is unknown.
#1580 checked the analytic edge-q proxies against the Tier A EFIT equilibria
and found the START scaling of Akers et al. (2000) reproduces the equilibrium
q95 (median ratio 0.99). Which scaling, which START configuration and which
shape stands in for a shot with no boundary are choices about VEST, so they
live in ``vest.yaml:edge_q_estimate`` and are read here; the formula
(:func:`vaft.formula.equilibrium.estimated_q95`) knows no machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from .utils import VestConfigurationError, _policy_document, _resolve_info_file_path

__all__ = [
    "EdgeQEstimatePolicy",
    "vest_edge_q_estimate_policy",
]

_SCALINGS = ("start", "iter")
_CONFIGURATIONS = ("limiter", "double_null")
_STATUSES = ("assumed", "measured", "inferred")
#: The boundary scalars a default shape must state, in IMAS boundary names.
SHAPE_KEYS = ("minor_radius", "major_radius", "elongation", "triangularity")


@dataclass(frozen=True)
class EdgeQEstimatePolicy:
    """The stated scaling and stand-in shape for an edge-q estimate."""

    scaling: str
    configuration: str
    default_shape: Mapping[str, float]
    status: Mapping[str, str]
    provenance: Mapping[str, str]
    source: str = "vest.yaml:edge_q_estimate"


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise VestConfigurationError(f"{where} has no '{key}'")
    return block[key]


def _choice(block: Mapping[str, Any], key: str, where: str, allowed: tuple[str, ...]) -> str:
    value = str(_require(block, key, where))
    if value not in allowed:
        raise VestConfigurationError(f"{where}.{key} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _build(document: Mapping[str, Any]) -> EdgeQEstimatePolicy:
    where = "edge_q_estimate"
    block = document.get(where)
    if not isinstance(block, Mapping):
        raise VestConfigurationError(f"vest.yaml has no '{where}' block")
    scaling = _choice(block, "scaling", where, _SCALINGS)
    configuration = _choice(block, "configuration", where, _CONFIGURATIONS)
    if scaling == "iter" and configuration != "limiter":
        raise VestConfigurationError(f"{where}.configuration applies to the START scaling only")

    shape_block = _require(block, "default_shape", where)
    if not isinstance(shape_block, Mapping):
        raise VestConfigurationError(f"{where}.default_shape must be a mapping")
    shape = {key: float(_require(shape_block, key, f"{where}.default_shape")) for key in SHAPE_KEYS}
    for key in ("minor_radius", "major_radius", "elongation"):
        if not shape[key] > 0.0:
            raise VestConfigurationError(f"{where}.default_shape.{key} must be positive, got {shape[key]}")
    if shape["minor_radius"] >= shape["major_radius"]:
        raise VestConfigurationError(f"{where}.default_shape.minor_radius must be smaller than major_radius")

    status = {
        "scaling": _choice(block, "status", where, _STATUSES),
        "default_shape": _choice(shape_block, "status", f"{where}.default_shape", _STATUSES),
    }
    provenance = {
        "scaling": str(_require(block, "provenance", where)),
        "default_shape": str(_require(shape_block, "provenance", f"{where}.default_shape")),
    }
    return EdgeQEstimatePolicy(
        scaling=scaling,
        configuration=configuration,
        default_shape=shape,
        status=status,
        provenance=provenance,
    )


@lru_cache(maxsize=8)
def _cached(info_file: str | None) -> EdgeQEstimatePolicy:
    return _build(_policy_document(_resolve_info_file_path(info_file)))


def vest_edge_q_estimate_policy(*, info_file: str | None = None) -> EdgeQEstimatePolicy:
    """The scaling and stand-in shape VEST's edge-q estimates use.

    Not keyed by shot: the estimate is a cohort-calibrated stand-in, and a
    policy that changed between discharges could not be compared across them.
    """
    return _cached(info_file)
