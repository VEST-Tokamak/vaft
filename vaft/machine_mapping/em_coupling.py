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
from vaft.machine_mapping.pf_passive import (
    append_wall_loops,
    load_wall_2409_additions,
    wall_geometry_version_for_shot,
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


#: Relative asymmetry max|M - M^T| / max|M| at or below which a stored matrix
#: is reciprocal to float64 round-off: the fold below is then a bitwise no-op
#: and the record says so rather than claiming a correction was applied.
PASSIVE_COUPLING_ASYMMETRY_ROUNDOFF = 1.0e-12
#: Above this a passive-passive coupling matrix is reported. Reciprocity
#: requires M_ij == M_ji exactly, so anything above round-off is a defect in
#: the source, not noise. The packaged asset read 1.27e-3 until issue #373
#: repaired the material factor the donor code applied one-sidedly; it is now
#: exactly reciprocal and carries that repair as provenance (see
#: ``workflow/em_coupling``). The threshold stays for a caller-supplied
#: reference matrix, and for any product materialized before the repair.
PASSIVE_COUPLING_ASYMMETRY_WARN = 1.0e-6
#: Above this the matrix is not plausibly a mutual-inductance matrix at all,
#: and averaging it into symmetry would manufacture physics; refuse instead.
PASSIVE_COUPLING_ASYMMETRY_REJECT = 1.0e-1

#: Optional npz key holding the asset's own provenance as a JSON document
#: (``workflow/em_coupling/regenerate_passive_coupling.py`` writes it).
COUPLING_PROVENANCE_KEY = "provenance"
#: Provenance fields the mapper surfaces into ``em_coupling.code.parameters``.
#: Deterministic per asset, so two loads of one asset write identical records.
_SURFACED_PROVENANCE = (
    "generator",
    "generated",
    "git_commit",
    "source_sha256",
    "passive_material_factor",
    "convention",
)


def _load_versioned_coupling(
    path: str | Path = DEFAULT_VERSIONED_COUPLING,
) -> tuple[dict[str, np.ndarray], dict | None]:
    """The packaged coupling matrices and, when the asset carries one, its provenance.

    An asset written before #373 has no provenance key and loads unchanged;
    ``None`` then says "the asset does not say where it came from", which the
    record distinguishes from a repaired asset.
    """
    import json

    matrices: dict[str, np.ndarray] = {}
    provenance: dict | None = None
    with np.load(path, allow_pickle=False) as versioned:
        for key in versioned.files:
            if key == COUPLING_PROVENANCE_KEY:
                try:
                    provenance = json.loads(str(versioned[key][()]))
                except (TypeError, ValueError):
                    provenance = None
                continue
            matrices[key] = np.asarray(versioned[key], dtype=float)
    return matrices, provenance


def load_versioned_coupling_provenance(
    path: str | Path = DEFAULT_VERSIONED_COUPLING,
) -> dict | None:
    """The provenance record stored in the packaged coupling asset, or ``None``."""
    return _load_versioned_coupling(path)[1]


def _coupling_provenance_lines(provenance: dict | None) -> list[str]:
    """``key=value`` lines for ``code.parameters`` naming the asset's origin."""
    if not provenance:
        return ["coupling_asset_provenance=absent"]
    lines = []
    for key in _SURFACED_PROVENANCE:
        if key in provenance and provenance[key] is not None:
            lines.append(f"coupling_asset_{key}={provenance[key]}")
    return lines


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


def _extend_to_wall(
    mutual_pa: np.ndarray, mutual_pp: np.ndarray, *, pf_version: str, wall_version: str
) -> tuple[np.ndarray, np.ndarray]:
    """Append the wall's additional rows to the 950-loop base matrices.

    The additions asset carries the new rows against every loop, base and new
    (`workflow/em_coupling/import_wall_2409.py`); the base-by-new block is the
    transpose of their base columns, so the result is reciprocal exactly when
    the base block is.
    """
    if wall_version == "1512":
        return mutual_pa, mutual_pp
    additions = load_wall_2409_additions()
    rows = np.asarray(additions["mutual_passive_passive_rows"], dtype=float)
    n_base = mutual_pp.shape[0]
    if rows.shape[1] != n_base + rows.shape[0]:
        raise ValueError(
            f"wall {wall_version} rows span {rows.shape[1]} loops; the base coupling has {n_base}"
        )
    extended_pp = np.block([[mutual_pp, rows[:, :n_base].T], [rows[:, :n_base], rows[:, n_base:]]])
    extended_pa = np.vstack([mutual_pa, additions["mutual_passive_active"][pf_version]])
    return extended_pa, extended_pp


def em_coupling(
    ods: Any,
    source: str | Path | None = None,
    options: dict | None = None,
    *,
    shot: int | None = None,
) -> None:
    """Populate canonical, PF- and wall-versioned VEST coupling data.

    All matrices come from compact packaged assets selected by the same shot
    boundaries as :func:`pf_active` and :func:`pf_passive`: the 950-loop base,
    plus the wall 2409 rows from shot 43017. An explicitly supplied legacy ODS
    remains accepted as an override for the base passive geometry and its
    passive-passive matrix.
    """
    if shot is None and options is not None:
        shot = options.get("shot")
    reference_path = _resolve_reference(source, options)
    reference = load_static_ods(reference_path)
    wall_version = wall_geometry_version_for_shot(shot)
    append_wall_loops(reference, wall_version)
    geometry_version = pf_geometry_version_for_shot(shot)
    versioned, asset_provenance = _load_versioned_coupling(DEFAULT_VERSIONED_COUPLING)
    mutual_aa = versioned[f"mutual_active_active_{geometry_version}"]
    mutual_pa = versioned[f"mutual_passive_active_{geometry_version}"]
    packaged_mutual_pp = versioned["mutual_passive_passive"]

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
    n_added = (
        len(load_wall_2409_additions()["loops"]) if wall_version != "1512" else 0
    )
    # A wall 2409 product passed back in already carries the added rows; its
    # matrix is used as it stands and only the base is ever extended.
    reference_extended = n_added > 0 and mutual_pp.shape == (n_passive + n_added,) * 2
    if mutual_pp.shape != (n_passive, n_passive) and not reference_extended:
        raise ValueError(
            "Reference mutual_passive_passive matrix has incompatible shape "
            f"{mutual_pp.shape}; expected ({n_passive}, {n_passive})"
        )
    if n_added and not reference_extended and Path(reference_path) != Path(DEFAULT_REFERENCE_ODS):
        # The added rows were computed against the packaged base loops; a
        # reference with other base geometry would get rows for a wall it
        # does not have.
        packaged = load_static_ods(DEFAULT_REFERENCE_ODS)
        for index in range(n_passive):
            if _static_signature(reference[f"pf_passive.loop.{index}"]) != _static_signature(
                packaged[f"pf_passive.loop.{index}"]
            ):
                raise ValueError(
                    f"reference {reference_path} has base loop {index + 1} unlike the packaged "
                    f"wall, but shot {shot} is on wall {wall_version}, whose added coupling "
                    "rows were computed against the packaged base loops"
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
    if reference_extended:
        mutual_pa, _ = _extend_to_wall(
            mutual_pa, packaged_mutual_pp, pf_version=geometry_version, wall_version=wall_version
        )
    else:
        mutual_pa, mutual_pp = _extend_to_wall(
            mutual_pa, mutual_pp, pf_version=geometry_version, wall_version=wall_version
        )
    n_passive = mutual_pp.shape[0]
    present = len(ods["pf_passive.loop"]) if "pf_passive.loop" in ods else 0
    if present and present != n_passive:
        raise ValueError(
            f"pf_passive carries {present} loops, but shot {shot} is on wall "
            f"{wall_version}, whose coupling has {n_passive} rows; populate it with "
            f"pf_passive(ods, shot={shot}) so both select the same wall (issue #956)"
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
    from_reference = "em_coupling.mutual_passive_passive" in reference
    symmetrized = passive_asymmetry > PASSIVE_COUPLING_ASYMMETRY_ROUNDOFF
    if symmetrized:
        reciprocity = (
            "mutual_passive_passive symmetrized to (M + M^T)/2 on load"
            f" (input asymmetry {passive_asymmetry:.3g})"
        )
    elif asset_provenance and not from_reference:
        reciprocity = (
            "mutual_passive_passive reciprocity exact in the packaged asset"
            f" (input asymmetry {passive_asymmetry:.3g}; material factor "
            f"{asset_provenance.get('passive_material_factor', 'unspecified')} repaired by "
            f"{asset_provenance.get('generator', 'unspecified')}, issue #373)"
        )
    else:
        reciprocity = (
            "mutual_passive_passive reciprocity exact"
            f" (input asymmetry {passive_asymmetry:.3g})"
        )
    ods["em_coupling.ids_properties.comment"] = (
        "VEST electromagnetic coupling for PF geometry "
        f"{geometry_version} and wall {wall_version}; selected for shot "
        f"{shot if shot is not None else 'unspecified'}"
        f"; {reciprocity}"
    )
    # DD-sanctioned home for the numeric record, so a consumer can see how far
    # the stored matrix was from reciprocity, and where the packaged asset came
    # from, without re-reading the asset.  ``passive_passive_input_asymmetry``
    # is always present: ``vaft.omas.process_wrapper`` harvests it.
    ods["em_coupling.code.parameters"] = "\n".join(
        [
            f"passive_passive_symmetrized={'true' if symmetrized else 'false'}",
            f"passive_passive_input_asymmetry={passive_asymmetry:.6e}",
            *_coupling_provenance_lines(None if from_reference else asset_provenance),
            f"wall_geometry={wall_version}",
            *(
                [f"wall_additions_generator={load_wall_2409_additions()['provenance'].get('generator')}"]
                if wall_version != "1512"
                else []
            ),
        ]
    ) + "\n"
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
    "load_versioned_coupling_provenance",
    "COUPLING_PROVENANCE_KEY",
    "DEFAULT_REFERENCE_ODS",
    "DEFAULT_STATIC_GEOMETRY",
    "DEFAULT_VERSIONED_COUPLING",
    "PASSIVE_COUPLING_ASYMMETRY_REJECT",
    "PASSIVE_COUPLING_ASYMMETRY_ROUNDOFF",
    "PASSIVE_COUPLING_ASYMMETRY_WARN",
]
