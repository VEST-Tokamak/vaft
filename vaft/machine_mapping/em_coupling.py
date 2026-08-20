"""`em_coupling` IDS mapping helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from omas import ODS, load_omas_json

from vaft.data.resources import data_path
from vaft.machine_mapping.pf_active import (
    pf_geometry_version_for_shot,
    vfit_pf_active_static,
)


DEFAULT_REFERENCE_ODS = data_path("omas/39915.json")
DEFAULT_VERSIONED_COUPLING = data_path("geometry/VEST_em_coupling_pf_versions.npz")


def _coordinate_uris(ids_name: str, structure_name: str, count: int) -> list[str]:
    """Build one-based, same-entry IMAS URI fragments in matrix order."""
    return [f"#{ids_name}/{structure_name}({index})" for index in range(1, count + 1)]


def _ordered_labels(ods: Any, path: str, count: int) -> list[str]:
    labels = []
    for index in range(count):
        identifier = ods.get(f"{path}.{index}.identifier")
        name = ods.get(f"{path}.{index}.name")
        labels.append(str(identifier or name or ""))
    return labels


def _static_signature(node: Any) -> tuple:
    signature = []
    for key, value in node.flat().items():
        if key == "current" or key.startswith("current."):
            continue
        array = np.asarray(value)
        signature.append((key, array.dtype.str, array.shape, array.tobytes()))
    return tuple(signature)


def _validate_coordinate_order(
    ods: Any,
    reference: Any,
    *,
    shot: int | None,
    n_active: int,
    n_passive: int,
) -> None:
    if "pf_active.coil" not in ods or "pf_passive.loop" not in ods:
        raise ValueError(
            "em_coupling requires pf_active.coil and pf_passive.loop so matrix "
            "coordinate ordering can be validated"
        )

    actual_active = _ordered_labels(ods, "pf_active.coil", n_active)
    expected_active = [f"PF{index}" for index in range(1, n_active + 1)]
    if actual_active != expected_active:
        raise ValueError(
            "pf_active coil ordering does not match the versioned coupling "
            f"columns: expected {expected_active}, got {actual_active}"
        )

    expected_active_ods = ODS(consistency_check=False)
    vfit_pf_active_static(expected_active_ods, shot=shot)
    for index in range(n_active):
        if _static_signature(ods[f"pf_active.coil.{index}"]) != _static_signature(
            expected_active_ods[f"pf_active.coil.{index}"]
        ):
            raise ValueError(
                "pf_active geometry does not match the coupling version selected "
                f"for shot {shot}: mismatch at coil {index + 1}"
            )

    actual_passive = _ordered_labels(ods, "pf_passive.loop", n_passive)
    expected_passive = _ordered_labels(reference, "pf_passive.loop", n_passive)
    if actual_passive != expected_passive:
        raise ValueError(
            "pf_passive loop ordering does not match the versioned coupling rows"
        )
    for index in range(n_passive):
        if _static_signature(ods[f"pf_passive.loop.{index}"]) != _static_signature(
            reference[f"pf_passive.loop.{index}"]
        ):
            raise ValueError(
                "pf_passive geometry/order does not match the versioned coupling "
                f"rows: mismatch at loop {index + 1}"
            )


def _resolve_reference(source: str | Path | None, options: dict | None) -> Path:
    if options is None:
        options = {}
    candidate = source or options.get("reference_ods") or options.get("source") or DEFAULT_REFERENCE_ODS
    path = Path(candidate).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"em_coupling reference ODS not found: {path}")
    return path


def em_coupling(
    ods: Any,
    source: str | Path | None = None,
    options: dict | None = None,
    *,
    shot: int | None = None,
) -> None:
    """Populate canonical, PF-geometry-versioned VEST coupling data.

    The reference ODS remains the source for the unchanged passive-to-passive
    matrix. Active-coil matrices come from a compact packaged asset selected by
    the same shot boundary as :func:`pf_active`.
    """
    if shot is None and options is not None:
        shot = options.get("shot")
    reference_path = _resolve_reference(source, options)
    reference = load_omas_json(str(reference_path), consistency_check=False)
    if "em_coupling" not in reference:
        raise KeyError(f"Reference ODS has no em_coupling IDS: {reference_path}")

    geometry_version = pf_geometry_version_for_shot(shot)
    with np.load(DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        mutual_aa = np.asarray(
            versioned[f"mutual_active_active_{geometry_version}"], dtype=float
        )
        mutual_pa = np.asarray(
            versioned[f"mutual_passive_active_{geometry_version}"], dtype=float
        )

    n_passive, n_active = mutual_pa.shape
    if mutual_aa.shape != (n_active, n_active):
        raise ValueError(
            "Versioned mutual_active_active matrix has incompatible shape "
            f"{mutual_aa.shape} for {n_active} active coils"
        )
    mutual_pp = np.asarray(
        reference["em_coupling.mutual_passive_passive"], dtype=float
    )
    if mutual_pp.shape != (n_passive, n_passive):
        raise ValueError(
            "Reference mutual_passive_passive matrix has incompatible shape "
            f"{mutual_pp.shape}; expected ({n_passive}, {n_passive})"
        )
    _validate_coordinate_order(
        ods,
        reference,
        shot=shot,
        n_active=n_active,
        n_passive=n_passive,
    )

    ods["em_coupling"] = copy.deepcopy(reference["em_coupling"])
    ods["em_coupling.active_coils"] = _coordinate_uris(
        "pf_active", "coil", n_active
    )
    ods["em_coupling.passive_loops"] = _coordinate_uris(
        "pf_passive", "loop", n_passive
    )
    ods["em_coupling.mutual_active_active"] = mutual_aa
    ods["em_coupling.mutual_passive_active"] = mutual_pa
    ods["em_coupling.ids_properties.comment"] = (
        "VEST electromagnetic coupling for PF geometry "
        f"{geometry_version}; selected for shot {shot if shot is not None else 'unspecified'}"
    )


def calculate_em_coupling_from_raw_database(
    ods: Any,
    shot: int | dict | None = None,
    options: dict | None = None,
) -> None:
    """Populate coupling for a shot through the raw-mapping entry point.

    Passing an options dictionary as the second positional argument remains
    supported for callers using the historical ``(ods, options)`` signature.
    """
    if isinstance(shot, dict):
        if options is not None:
            raise TypeError("options were provided both positionally and by keyword")
        options = shot
        shot = options.get("shot")
    em_coupling(ods, options=options, shot=shot)


__all__ = [
    "calculate_em_coupling_from_raw_database",
    "em_coupling",
    "DEFAULT_REFERENCE_ODS",
    "DEFAULT_VERSIONED_COUPLING",
]
