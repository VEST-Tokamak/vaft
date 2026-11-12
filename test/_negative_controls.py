"""Controlled magnetics perturbations for negative-control validation (issue #2166, D2).

A validation that cannot see a known error is absorbing it.  These helpers
take a magnetics ODS and return a *perturbed deep copy* carrying one
deliberate, physically meaningful configuration error:

``swap_measured_signals``
    Two sensors' measured records exchanged while their names and positions
    stay put -- what a DAQ field-code swap or a re-ordered channel table does.
``flip_measured_sign``
    One sensor's record negated -- a reversed calibration polarity, applied
    after calibration (the processed ODS stores calibrated, integrated data,
    and negating the calibration negates the product).
``move_sensor``
    One sensor's recorded position displaced -- a wrong R/Z in the geometry.

Every helper returns ``(perturbed, record)``; ``record`` names the affected
channels so a test can state its expected observable *before* it runs the
benchmark.  The input ODS is never modified and nothing is written to disk:
negative controls must never reach a stored product.

These live under ``test/`` rather than ``vaft.validation`` on purpose: the
validation layer evaluates data, it does not generate it (#253), and a
perturbation generator is test infrastructure.
"""

from __future__ import annotations

import copy
from typing import Any

#: The measured quantity of each magnetics array, as the vacuum benchmark reads it.
MEASURED = {"b_field_pol_probe": "field", "flux_loop": "flux"}


def _check_kind(kind: str) -> str:
    if kind not in MEASURED:
        raise ValueError(f"kind must be one of {sorted(MEASURED)}, got {kind!r}")
    return MEASURED[kind]


def channel_name(ods: Any, kind: str, index: int) -> str:
    return str(ods[f"magnetics.{kind}.{int(index)}.name"])


def swap_measured_signals(ods: Any, kind: str, first: int, second: int) -> tuple[Any, dict[str, Any]]:
    """Exchange the whole measured subtree (data, time, validity) of two channels."""
    quantity = _check_kind(kind)
    if int(first) == int(second):
        raise ValueError("a swap needs two different channels")
    perturbed = copy.deepcopy(ods)
    a = f"magnetics.{kind}.{int(first)}.{quantity}"
    b = f"magnetics.{kind}.{int(second)}.{quantity}"
    node_a = copy.deepcopy(perturbed[a])
    node_b = copy.deepcopy(perturbed[b])
    perturbed[a] = node_b
    perturbed[b] = node_a
    return perturbed, {
        "perturbation": "swap_measured_signals",
        "kind": kind,
        "affected": [channel_name(ods, kind, first), channel_name(ods, kind, second)],
    }


def flip_measured_sign(ods: Any, kind: str, index: int) -> tuple[Any, dict[str, Any]]:
    """Negate one channel's measured data (a reversed calibration polarity)."""
    quantity = _check_kind(kind)
    perturbed = copy.deepcopy(ods)
    path = f"magnetics.{kind}.{int(index)}.{quantity}.data"
    perturbed[path] = -1.0 * perturbed[path]
    return perturbed, {
        "perturbation": "flip_measured_sign",
        "kind": kind,
        "affected": [channel_name(ods, kind, index)],
    }


def move_sensor(ods: Any, kind: str, index: int, *, dr: float = 0.0, dz: float = 0.0) -> tuple[Any, dict[str, Any]]:
    """Displace one sensor's recorded position by ``(dr, dz)`` metres."""
    _check_kind(kind)
    perturbed = copy.deepcopy(ods)
    base = f"magnetics.{kind}.{int(index)}.position" + (".0" if kind == "flux_loop" else "")
    old_r, old_z = float(perturbed[f"{base}.r"]), float(perturbed[f"{base}.z"])
    perturbed[f"{base}.r"] = old_r + float(dr)
    perturbed[f"{base}.z"] = old_z + float(dz)
    return perturbed, {
        "perturbation": "move_sensor",
        "kind": kind,
        "affected": [channel_name(ods, kind, index)],
        "from": (old_r, old_z),
        "to": (old_r + float(dr), old_z + float(dz)),
    }
