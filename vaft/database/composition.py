"""Explicit composition of one shot across named HSDS sources (issue #305).

Reading a source returns that source and nothing else: `main` never silently
carries `impa`, and no union of the two is presented as one shot.  That is what
keeps a missing entry meaningful in each -- absence from `impa` says only that no
IMPA product was published, never that the baseline shot is incomplete.

Analysis that wants both asks for both, here, and the result says where each
channel came from.  The composition is deliberately one-directional: the base
source's probe indices never move, because k-files and the EFIT constraint
builder address probes by index, so the optional channels are appended after
them exactly as an in-product mapping would have.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..machine_mapping.utils import path_exists
from . import sources as _sources
from .filedb import OMAS_MANIFEST_NAME


#: Both nodes the array can land on: Hall channels are mounted for the toroidal
#: field, the vertical-field sensors are poloidal-plane probes.
_PROBE_NODES = ("magnetics.b_field_tor_probe", "magnetics.b_field_pol_probe")

__all__ = ["compose", "compose_stage_products", "impa_channels"]


def _probe_count(ods: Any, node: str) -> int:
    try:
        return len(ods[node])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def impa_channels(ods: Any, node: str) -> list[int]:
    """Return the indices of ``node`` holding IMPA channels, by identifier."""
    from ..machine_mapping.impa import impa_probe_indices

    return impa_probe_indices(ods, node)


def compose(
    shot: int,
    sources: Sequence[str] = ("main", "impa"),
    *,
    paths: Iterable[str] | None = None,
    occurrence: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load ``shot`` from several sources and append the optional channels.

    ``sources[0]`` is the base whose indices are preserved; every further source
    contributes its IMPA-identified probes, appended after whatever the base
    already holds.  Returns the composed ODS and a provenance record naming, for
    each appended index, the source and the index it had there.

    Nothing else in VAFT calls this: composition is the caller's decision, and
    source identity stays visible in the result rather than being flattened
    away.
    """
    from . import load as load_source

    names = [_sources.resolve(name) for name in sources]
    if len(names) < 2:
        raise ValueError(
            "compose needs a base source and at least one source to compose onto "
            f"it; got {names}."
        )

    base_name = names[0]
    base = load_source(
        shot, source=base_name, paths=list(paths) if paths else None, occurrence=occurrence
    )
    provenance: dict[str, Any] = {
        "shot": int(shot),
        "base": base_name,
        "sources": names,
        "appended": [],
        "contributed": {},
    }

    for name in names[1:]:
        extra = load_source(
            shot, source=name, paths=list(paths) if paths else None, occurrence=occurrence
        )
        contributed = 0
        for node in _PROBE_NODES:
            if not path_exists(extra, node):
                continue
            for source_index in impa_channels(extra, node):
                index = _probe_count(base, node)
                base[f"{node}.{index}"] = copy.deepcopy(extra[f"{node}.{source_index}"])
                provenance["appended"].append(
                    {
                        "source": name,
                        "node": node,
                        "source_index": int(source_index),
                        "index": int(index),
                    }
                )
                contributed += 1
        provenance["contributed"][name] = contributed
    return base, provenance


class StageCompositionError(Exception):
    """Two stage products that cannot be composed into one shot."""


def _sha256_file(path: Path) -> str:
    # Imported lazily: `vaft.omas.vest_upstream` reads this package's stage
    # registry, so a module-level import here would close the cycle.
    from ..omas.vest_upstream import sha256_file

    return sha256_file(path)


def _load_product(path: Path) -> Any:
    from ..omas import load as load_product

    return load_product(path)


def _superseded_sha256(product: Path) -> str | None:
    """The hash of the file this product was migrated from, if it was.

    `migrate-products` rewrites a manifest's `output` to the file that now
    exists and records the one it replaced under `migration.previous_output`.
    Read defensively: a tree that has never been migrated has no such block, a
    manifest may be absent on a hand-assembled product, and neither is an error
    here -- both simply mean there is no superseded hash to accept.
    """
    manifest_path = product.parent.parent / "metadata" / OMAS_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    previous = (manifest.get("migration") or {}).get("previous_output") or {}
    recorded = previous.get("sha256")
    return recorded if isinstance(recorded, str) else None


def compose_stage_products(
    *,
    diagnostics: str | Path,
    eddy: str | Path,
    eddy_manifest: str | Path | None = None,
    strict: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Union a diagnostics product with the eddy product computed from it.

    This is the inverse of :func:`vaft.database.replication._project`, which
    splits a shot into the per-stage subtrees each stage owns on the way out to
    HSDS.  The eddy stage owns ``pf_passive`` and publishes only that, so
    anything that needs the passive currents *and* the magnetics they were
    solved against -- the EFIT constraint builder, the eddy stage's own
    validation figures -- asks for both products here rather than relying on one
    of them to carry the other.

    Two checks run, and the first cannot be switched off:

    The passive-current time grid **is** ``pf_active.time``, by construction:
    :func:`~vaft.omas.vest_upstream.build_eddy_ods` interpolates the plasma
    current onto that grid and
    :func:`~vaft.omas.process_wrapper.compute_eddy_currents` writes the loop
    currents on it.  The EFIT constraint builder then interpolates *on*
    ``pf_passive.time`` for each equilibrium slice, so a pairing whose grids
    disagree does not fail -- it silently produces wall currents for the wrong
    instants.  Requiring exact equality is what makes that unrepresentable.  A
    length check is not enough: the same shot reprocessed with a different
    magnetics configuration yields an identically shaped, differently valued
    grid.

    ``strict`` governs only the second check, which compares the diagnostics
    file against the hash the eddy manifest recorded for the product it read.
    Harnesses pointed at hand-assembled products pass ``strict=False`` to
    downgrade it to a warning; nothing downgrades the grid check.

    That second check has one accepted way to disagree.  Migrating a product
    onto a new container (``vaft.cli filedb migrate-products``, #813) re-encodes
    bytes that are already correct, so the diagnostics file hashes differently
    while holding the same data -- and the migration deliberately does **not**
    rewrite the eddy manifest's ``input``, because those hashes record the files
    the run actually read and editing them would assert the physics was computed
    from a file that did not exist then.  The join is closed from the other side
    instead: the diagnostics manifest records what it was migrated *from*, and a
    recorded hash matching that is the same file, so it passes.  Without this the
    migration would break every composition on the deployment at once.

    Returns the composed ODS and a provenance record naming both inputs.
    """
    diagnostics_path = Path(diagnostics)
    eddy_path = Path(eddy)

    composed = _load_product(diagnostics_path)
    eddy_ods = _load_product(eddy_path)

    if not path_exists(eddy_ods, "pf_passive"):
        raise StageCompositionError(
            f"{eddy_path} carries no pf_passive; it is not an eddy product."
        )

    # Both grids are required rather than checked when present: an eddy product
    # always has `pf_passive.time` (compute_eddy_currents writes it) and its
    # diagnostics always has `pf_active.time` (build_eddy_ods refuses to run
    # without it), so an absent one means these are not the pair they claim to
    # be. Skipping the check for a missing grid would turn the guard off in
    # exactly the case that needs it.
    for ods, path, node in (
        (eddy_ods, eddy_path, "pf_passive.time"),
        (composed, diagnostics_path, "pf_active.time"),
    ):
        if not path_exists(ods, node):
            raise StageCompositionError(
                f"{path} carries no {node}, so the two products cannot be "
                "checked for having come from the same run."
            )
    passive_time = np.asarray(eddy_ods["pf_passive.time"], dtype=float)
    active_time = np.asarray(composed["pf_active.time"], dtype=float)
    if passive_time.shape != active_time.shape or not np.array_equal(
        passive_time, active_time
    ):
        raise StageCompositionError(
            "The eddy product's pf_passive.time is not this diagnostics "
            "product's pf_active.time, so the two were not produced from the "
            "same run and the wall currents would be interpolated onto the "
            f"wrong instants. eddy={eddy_path} (n={passive_time.size}) "
            f"diagnostics={diagnostics_path} (n={active_time.size})."
        )

    recorded: str | None = None
    if eddy_manifest is not None:
        manifest = json.loads(Path(eddy_manifest).read_text())
        recorded = (manifest.get("input") or {}).get("diagnostics_sha256")

    actual = _sha256_file(diagnostics_path)
    superseded = _superseded_sha256(diagnostics_path)
    if recorded is not None and recorded != actual and recorded != superseded:
        message = (
            "The eddy product was computed from a different diagnostics file: "
            f"its manifest records {recorded}, but {diagnostics_path} hashes to "
            f"{actual}."
        )
        if superseded is not None:
            message += (
                f" Its manifest records a migration from {superseded}, which "
                "does not match either."
            )
        if strict:
            raise StageCompositionError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    contributed: list[str] = []
    for name in _sources.STAGE_REPLICATION["eddy"].ids:
        if path_exists(eddy_ods, name):
            composed[name] = copy.deepcopy(eddy_ods[name])
            contributed.append(name)

    provenance = {
        "diagnostics": {"path": str(diagnostics_path), "sha256": actual},
        "eddy": {
            "path": str(eddy_path),
            "sha256": _sha256_file(eddy_path),
            "contributed": tuple(contributed),
            "manifest_diagnostics_sha256": recorded,
        },
    }
    return composed, provenance
