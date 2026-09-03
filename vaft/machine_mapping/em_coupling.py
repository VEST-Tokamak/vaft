"""`em_coupling` IDS mapping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import warnings

import numpy as np
from omas import ODS

from vaft.data.resources import data_path
from vaft.machine_mapping.pf_active import (
    pf_geometry_version_for_shot,
    vfit_pf_active_static,
)
from vaft.machine_mapping.static_geometry import load_static_ods


DEFAULT_STATIC_GEOMETRY = data_path("geometry/VEST_static_geometry.json.gz")
# Kept for callers importing the historical name.
DEFAULT_REFERENCE_ODS = DEFAULT_STATIC_GEOMETRY
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


# Fields excluded from the geometry fingerprint below. `current` is dynamic.
# `resistance` is a derived material scalar, not geometry: it is a pure
# function of the coil width/radius constants, the shot-era height profile and
# `turns_with_sign` -- all of which the signature already covers or which are
# identical by construction on both sides of the comparison. Including it made
# the guard reject every packaged ODS whenever the resistance *formula*
# changed, even though the coupling matrices were still valid (issue #117).
_SIGNATURE_EXCLUDED_FIELDS = (
    "current",
    "resistance",
    # Electrical, not geometric: the coupling rows depend on outlines and
    # areas alone. resistivity is the nominal material value since #388 and
    # older products still carry the inherited vector.
    "resistivity",
)


def _static_signature(node: Any) -> tuple:
    signature = []
    for key, value in node.flat().items():
        if any(
            key == field or key.startswith(f"{field}.")
            for field in _SIGNATURE_EXCLUDED_FIELDS
        ):
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


#: Relative asymmetry max|M - M^T| / max|M| above which a passive-passive
#: coupling matrix is reported. Reciprocity requires M_ij == M_ji exactly;
#: float64 round-off in a stored matrix sits near 1e-15, so anything above
#: this is a defect in the asset, not noise. The packaged asset reads 1.27e-3
#: (issue #347), which is why loading it warns until it is regenerated.
PASSIVE_COUPLING_ASYMMETRY_WARN = 1.0e-6
#: Above this the matrix is not plausibly a mutual-inductance matrix at all,
#: and averaging it into symmetry would manufacture physics; refuse instead.
PASSIVE_COUPLING_ASYMMETRY_REJECT = 1.0e-1


def _symmetrize_passive_coupling(mutual_pp: np.ndarray, *, source: str) -> tuple[np.ndarray, float]:
    """Enforce reciprocity on the passive-passive coupling and report how far off it was.

    Returns the symmetrized matrix and the measured relative asymmetry of the
    input, so the caller can record the correction as provenance rather than
    apply it silently.
    """
    if not np.all(np.isfinite(mutual_pp)):
        # Measured before symmetrizing, so a NaN or inf cannot masquerade as
        # a perfectly symmetric input in the provenance record.
        raise ValueError(
            f"{source}: mutual_passive_passive contains non-finite entries and "
            "cannot be assessed for reciprocity"
        )
    scale = float(np.max(np.abs(mutual_pp))) if mutual_pp.size else 0.0
    asymmetry = (
        float(np.max(np.abs(mutual_pp - mutual_pp.T))) / scale if scale > 0.0 else 0.0
    )
    if asymmetry > PASSIVE_COUPLING_ASYMMETRY_REJECT:
        raise ValueError(
            f"{source}: mutual_passive_passive has relative asymmetry "
            f"{asymmetry:.3g} (> {PASSIVE_COUPLING_ASYMMETRY_REJECT:g}); that is not a "
            "mutual-inductance matrix and will not be symmetrized into one"
        )
    if asymmetry > PASSIVE_COUPLING_ASYMMETRY_WARN:
        warnings.warn(
            f"{source}: mutual_passive_passive violates reciprocity by "
            f"{asymmetry:.3g} (max |M - M^T| / max |M|); symmetrized to (M + M^T)/2 "
            "on load. See issue #347.",
            RuntimeWarning,
            stacklevel=3,
        )
    return (mutual_pp + mutual_pp.T) / 2.0, asymmetry


def em_coupling(
    ods: Any,
    source: str | Path | None = None,
    options: dict | None = None,
    *,
    shot: int | None = None,
) -> None:
    """Populate canonical, PF-geometry-versioned VEST coupling data.

    All matrices come from a compact packaged asset selected by the same shot
    boundary as :func:`pf_active`. An explicitly supplied legacy ODS remains
    accepted as an override for passive geometry and its passive-passive matrix.
    """
    if shot is None and options is not None:
        shot = options.get("shot")
    reference_path = _resolve_reference(source, options)
    reference = load_static_ods(reference_path)
    geometry_version = pf_geometry_version_for_shot(shot)
    with np.load(DEFAULT_VERSIONED_COUPLING, allow_pickle=False) as versioned:
        mutual_aa = np.asarray(
            versioned[f"mutual_active_active_{geometry_version}"], dtype=float
        )
        mutual_pa = np.asarray(
            versioned[f"mutual_passive_active_{geometry_version}"], dtype=float
        )
        packaged_mutual_pp = np.asarray(
            versioned["mutual_passive_passive"], dtype=float
        )

    n_passive, n_active = mutual_pa.shape
    if mutual_aa.shape != (n_active, n_active):
        raise ValueError(
            "Versioned mutual_active_active matrix has incompatible shape "
            f"{mutual_aa.shape} for {n_active} active coils"
        )
    mutual_pp = (
        np.asarray(reference["em_coupling.mutual_passive_passive"], dtype=float)
        if "em_coupling.mutual_passive_passive" in reference
        else packaged_mutual_pp
    )
    if mutual_pp.shape != (n_passive, n_passive):
        raise ValueError(
            "Reference mutual_passive_passive matrix has incompatible shape "
            f"{mutual_pp.shape}; expected ({n_passive}, {n_passive})"
        )
    # Reciprocity is a property of the physics, not of the file the matrix came
    # from, so it is enforced on whichever source won above (issue #347).
    mutual_pp, passive_asymmetry = _symmetrize_passive_coupling(
        mutual_pp,
        source=(
            "reference ODS" if "em_coupling.mutual_passive_passive" in reference
            else str(DEFAULT_VERSIONED_COUPLING.name)
        ),
    )
    _validate_coordinate_order(
        ods,
        reference,
        shot=shot,
        n_active=n_active,
        n_passive=n_passive,
    )

    ods["em_coupling.active_coils"] = _coordinate_uris(
        "pf_active", "coil", n_active
    )
    ods["em_coupling.passive_loops"] = _coordinate_uris(
        "pf_passive", "loop", n_passive
    )
    ods["em_coupling.mutual_active_active"] = mutual_aa
    ods["em_coupling.mutual_passive_active"] = mutual_pa
    ods["em_coupling.mutual_passive_passive"] = mutual_pp
    ods["em_coupling.ids_properties.comment"] = (
        "VEST electromagnetic coupling for PF geometry "
        f"{geometry_version}; selected for shot {shot if shot is not None else 'unspecified'}"
        "; mutual_passive_passive symmetrized to (M + M^T)/2 on load"
        f" (input asymmetry {passive_asymmetry:.3g})"
    )
    # DD-sanctioned home for the numeric record, so a consumer can see how far
    # the stored asset was from reciprocity without re-reading the asset.
    ods["em_coupling.code.parameters"] = (
        f"passive_passive_symmetrized=true\n"
        f"passive_passive_input_asymmetry={passive_asymmetry:.6e}\n"
    )
    # em_coupling has no dynamic counterpart in VAFT, so per the DD's
    # `homogeneous_time` rule ("if only constant or static nodes are filled,
    # homogeneous_time must be set to 2") this IDS is always independent.
    ods["em_coupling.ids_properties.homogeneous_time"] = 2


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
    "DEFAULT_STATIC_GEOMETRY",
    "DEFAULT_VERSIONED_COUPLING",
]
