"""Move the OMAS stage products already on disk onto their declared shape (#813).

Two things changed about what a stage product is, and a deployment written
before them holds products that no longer match either:

* the container moved to gzipped JSON
  (:data:`vaft.database.filedb.OMAS_PRODUCT_SUFFIX`), so a product is now
  resolved under a name that does not exist in such a tree; and
* the eddy product became exactly the IDS its stage owns, so the ones already
  written carry the whole diagnostics product as well.

Neither needs the physics re-run. A container change re-encodes bytes that are
already correct, and the eddy change is a *projection* of a product that already
holds the right ``pf_passive`` -- the same projection
:func:`vaft.database.replication._project` has always applied on the way to
HSDS, which is why no replica changes and nothing has to be re-uploaded.

This is modelled on ``filedb relocate`` (#758) and keeps its contract: a dry run
by default, a JSON report with disjoint buckets, a non-zero exit when the plan
is not safe to apply, and a refusal that names what it refuses rather than
resolving it. One guarantee does not carry over. ``relocate`` renames
directories and never opens a file, so an interrupted run could not corrupt
anything; this reads, decodes and rewrites, so per-product atomicity has to be
bought explicitly -- temp file in the destination directory, fsync, verify,
:func:`os.replace`, then fsync the directory. Every interruption point leaves a
state the next run classifies correctly, and no reachable state puts a
half-written file under the canonical name.

The superseded originals are **not** deleted here. That is
:func:`sweep_superseded_products`, a separate operator step gated on a clean
report, following the retirement precedent: deletion is irreversible, so it is
never something a migration does on its way past.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from .filedb import (
    OMAS_MANIFEST_NAME,
    OMAS_PRODUCT_SUFFIX,
    OMAS_PRODUCT_SUFFIXES,
    OMASStage,
)
from .sources import STAGE_REPLICATION


class ProductMigrationError(Exception):
    """A migration that refuses to run rather than guess."""


#: Stages whose stored product held more than the stage owns, mapped to what it
#: should hold. Only eddy: it was built by returning the diagnostics ODS it had
#: solved on. Every other stage already stored its own subtree, so every other
#: entry in :data:`MIGRATIONS` is a container change and nothing more.
#:
#: Read from the replication registry rather than restated, so a product
#: migrated here and a product published by `_project` cannot disagree about
#: what the stage owns.
PROJECTIONS: dict[str, tuple[str, ...]] = {
    "eddy": STAGE_REPLICATION["eddy"].ids,
}

#: What a product written before #813 was called. Only ever `.json`: the three
#: HDF5 stages were introduced already declaring `.h5` (#599), so they have
#: nothing to move.
LEGACY_SUFFIX = ".json"


@dataclass(frozen=True)
class Recontainment:
    """One stage's move from what is on disk to what it now declares.

    ``keep_ids`` is ``None`` for a pure container change, which lets the
    migration take the cheap and stronger path: the stored ``.json`` *is* the
    plain bytes `vaft.omas.save` gzips, so re-encoding is a gzip of the file and
    verification is a byte comparison rather than an ODS one. A projection has
    to go through an ODS, and is verified semantically.
    """

    stage: str
    from_suffix: str
    to_suffix: str
    keep_ids: tuple[str, ...] | None

    @property
    def verify(self) -> str:
        return "bytes" if self.keep_ids is None else "semantic"


def _migrations() -> tuple[Recontainment, ...]:
    """Every stage whose stored product no longer matches what it declares.

    Derived from the declarations rather than listed, so a stage that changes
    container later is migrated by this tool without an edit -- and a stage
    that never moved (the three HDF5 ones) is skipped without having to be
    named.
    """
    rows = []
    for stage in OMASStage:
        declared = OMAS_PRODUCT_SUFFIXES.get(stage.value, OMAS_PRODUCT_SUFFIX)
        keep = PROJECTIONS.get(stage.value)
        if declared == LEGACY_SUFFIX and keep is None:
            continue  # nothing about this stage's product has moved
        if declared in {".h5", ".hdf5"}:
            continue  # declared HDF5 from the start; never had a .json to move
        rows.append(
            Recontainment(
                stage=stage.value,
                from_suffix=LEGACY_SUFFIX,
                to_suffix=declared,
                keep_ids=keep,
            )
        )
    return tuple(rows)


MIGRATIONS: tuple[Recontainment, ...] = _migrations()


@dataclass(frozen=True)
class ProductMigration:
    """One product's move, as the plan describes it."""

    stage: str
    directory: str
    source: str
    target: str
    verify: str


@dataclass(frozen=True)
class ProductMigrationReport:
    """What a canonical root needs before its products match their declarations.

    The buckets are disjoint and cover every `output/` directory found. Unlike
    ``relocate``'s, ``migrated`` is a *normal* end state rather than an error:
    the superseded original stays on disk until the sweep, which is what makes
    everything up to that point reversible.
    """

    root: str
    applied: bool
    pending: tuple[ProductMigration, ...] = ()
    migrated: tuple[str, ...] = ()
    settled: tuple[str, ...] = ()
    conflicting: tuple[str, ...] = ()
    orphan_temporaries: tuple[str, ...] = ()
    unrecognized: tuple[str, ...] = ()
    deletion_blocked: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def safe_to_apply(self) -> bool:
        """Whether every product found can be moved without a judgement call.

        A conflict here is narrower than ``relocate``'s "the destination is
        occupied", because with a separate sweep both files present is the
        ordinary post-migration state. What cannot be resolved is a superseded
        original *newer* than the product that superseded it: something wrote
        the old name after the migration passed, which means a writer on the old
        code is still live, and which of the two is authoritative is not a
        question this can answer.
        """
        return not self.conflicting and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dry_run": not self.applied,
            "root": self.root,
            "summary": {
                "pending": len(self.pending),
                "migrated": len(self.migrated),
                "settled": len(self.settled),
                "conflicting": len(self.conflicting),
                "orphan_temporaries": len(self.orphan_temporaries),
                "unrecognized": len(self.unrecognized),
                "deletion_blocked": len(self.deletion_blocked),
                "failures": len(self.failures),
                "safe_to_apply": self.safe_to_apply,
            },
            "pending": [asdict(item) for item in self.pending],
            "migrated": list(self.migrated),
            "settled": list(self.settled),
            "conflicting": list(self.conflicting),
            "orphan_temporaries": list(self.orphan_temporaries),
            "unrecognized": list(self.unrecognized),
            "deletion_blocked": list(self.deletion_blocked),
            "failures": list(self.failures),
        }


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def _temporary_prefix(stage: str) -> str:
    return f".{stage}.migrating."


def _output_directories(root: Path, stage: str) -> list[Path]:
    """Every `output/` directory under this stage, at any lineage depth.

    Walked rather than resolved from a shot list: a shot nobody remembers is
    still a product that has to move, and the lineage segments between the stage
    and the shot differ per stage (`efit/magnetic/{shot}` against
    `eddy/{shot}`).
    """
    stage_root = root / "omas" / stage
    if not stage_root.is_dir():
        return []
    return sorted(
        path for path in stage_root.rglob("output") if path.is_dir()
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_gzip_member(path: Path) -> str:
    """Hash what a gzip file decompresses to, without holding it in memory."""
    import hashlib

    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_product_containers(
    root: str | Path,
    *,
    stages: Sequence[str] | None = None,
    verify_shape: bool = False,
) -> ProductMigrationReport:
    """Classify every stage product under ``root``. Reads only; writes nothing.

    ``verify_shape`` opens each already-migrated product and checks its
    top-level IDS against what the stage owns. That makes the idempotency
    discriminator exact by inspection rather than by an argument about which
    writers exist, which is what the sweep needs before it deletes anything. It
    costs a full read pass, so it is off by default.
    """
    root = Path(root)
    if not root.is_dir():
        raise ProductMigrationError(f"{root} is not a directory.")

    selected = [
        row
        for row in MIGRATIONS
        if stages is None or row.stage in set(stages)
    ]
    unknown = set(stages or ()) - {row.stage for row in MIGRATIONS}
    if unknown:
        raise ProductMigrationError(
            "No product migration is declared for: " + ", ".join(sorted(unknown))
        )

    pending: list[ProductMigration] = []
    migrated: list[str] = []
    settled: list[str] = []
    conflicting: list[str] = []
    orphans: list[str] = []
    unrecognized: list[str] = []
    deletion_blocked: list[str] = []
    failures: list[str] = []

    for row in selected:
        for directory in _output_directories(root, row.stage):
            source = directory / f"{row.stage}{row.from_suffix}"
            target = directory / f"{row.stage}{row.to_suffix}"

            for stray in directory.glob(_temporary_prefix(row.stage) + "*"):
                orphans.append(str(stray))

            if source.is_file() and target.is_file():
                if source.stat().st_mtime > target.stat().st_mtime:
                    conflicting.append(str(source))
                else:
                    migrated.append(str(target))
                    # Answered here as well as in the sweep: an operator gating
                    # a script on this report has to be able to see that a
                    # deletion will be refused before running the step that
                    # refuses it.
                    blocking = _recoverability_problem(root, directory, row)
                    if blocking is not None:
                        deletion_blocked.append(f"{source}: {blocking}")
                    if verify_shape:
                        problem = _shape_problem(target, row)
                        if problem is not None:
                            failures.append(f"{target}: {problem}")
            elif target.is_file():
                settled.append(str(target))
                if verify_shape:
                    problem = _shape_problem(target, row)
                    if problem is not None:
                        failures.append(f"{target}: {problem}")
            elif source.is_file():
                pending.append(
                    ProductMigration(
                        stage=row.stage,
                        directory=str(directory),
                        source=str(source),
                        target=str(target),
                        verify=row.verify,
                    )
                )
            else:
                unrecognized.append(str(directory))

    return ProductMigrationReport(
        root=str(root),
        applied=False,
        pending=tuple(pending),
        migrated=tuple(migrated),
        settled=tuple(settled),
        conflicting=tuple(conflicting),
        orphan_temporaries=tuple(orphans),
        unrecognized=tuple(unrecognized),
        deletion_blocked=tuple(deletion_blocked),
        failures=tuple(failures),
    )


def _shape_problem(product: Path, row: Recontainment) -> str | None:
    """Whether an already-migrated product is what the stage should have written.

    Both branches open the file, because this is the sweep's only evidence that
    the product it is about to delete the original of can still be read. The
    write-time check proved that when it ran; the sweep is a separate step, days
    later, and the point of re-auditing is not to trust a flag from a process
    that may have been killed.

    A container-only product has no shape to check -- its content is the content
    it always had -- so what is checked is that the container still decodes.
    That catches a truncation, a bad restore, a half-copied file: the states in
    which deleting the original loses the product.
    """
    if row.keep_ids is None:
        try:
            _sha256_gzip_member(product)
        except Exception as error:
            return f"could not be decompressed ({type(error).__name__}: {error})"
        return None
    from ..omas import load as load_product

    try:
        ods = load_product(product)
    except Exception as error:  # a product that will not open is a finding
        return f"could not be read ({type(error).__name__}: {error})"
    present = {key.split(".")[0] for key in ods.keys()} - {"dataset_description"}
    extra = sorted(present - set(row.keep_ids))
    if extra:
        return "carries IDS the stage does not own: " + ", ".join(extra)
    return None


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #
def _fsync_directory(directory: Path) -> None:
    """Make the rename durable, not just visible.

    Without this the `os.replace` survives a killed process but not a power
    loss, and this runs unattended for hours.
    """
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write_container_only(source: Path, temporary: Path) -> tuple[str, str]:
    """Gzip the stored bytes, and return (source sha256, decompressed sha256).

    The stored `.json` *is* what `vaft.omas.save` writes before gzipping it, so
    this reproduces that writer's output without building an ODS -- and lets
    verification be a byte comparison, which is stronger than comparing two
    ODS objects and immune to the encoding differences (gzip mtime, indent)
    that make a byte comparison of the *containers* meaningless.

    The gzip parameters are `vaft.omas.save`'s, `mtime=0` included, so a
    migrated product is byte-identical to a freshly written one.
    """
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as compressed:
            with source.open("rb") as plain:
                shutil.copyfileobj(plain, compressed)
        raw.flush()
        os.fsync(raw.fileno())
    return _sha256_file(source), _sha256_gzip_member(temporary)


def _write_projection(
    source: Path, temporary: Path, keep_ids: tuple[str, ...]
) -> tuple[Any, Any, tuple[str, ...]]:
    """Write the stage's owned subtree.

    Returns the ODS as it was written back out, the projection that was meant,
    and the top-level IDS the projection dropped. The dropped set comes from
    here rather than from a second read in the caller: loading the source is
    what this migration's cost is made of, and one product would otherwise be
    decoded twice to answer a question the first decode already held.
    """
    from omas import ODS

    from ..omas import load as load_product
    from ..omas import save as save_product

    stored = load_product(source)
    expected = ODS(consistency_check=False)
    if "dataset_description" in stored:
        expected["dataset_description"] = stored["dataset_description"]
    for name in keep_ids:
        if name in stored:
            expected[name] = stored[name]
    dropped = tuple(
        sorted(
            {key.split(".")[0] for key in stored.keys()}
            - set(keep_ids)
            - {"dataset_description"}
        )
    )

    save_product(expected, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    return load_product(temporary), expected, dropped


def _same_ods(written: Any, expected: Any) -> str | None:
    """Compare path by path, never as bytes.

    Two gzip streams of equivalent content differ (mtime, member name), and two
    JSON encodings of the same ODS differ in whitespace, so a byte comparison
    here would report a difference that is not one. Re-loading the file that was
    actually written is what makes this a check rather than a tautology: it
    catches truncation and anything the encode/decode round trip loses.
    """
    import numpy as np

    left = dict(written.flat())
    right = dict(expected.flat())
    if left.keys() != right.keys():
        missing = sorted(set(right) - set(left))[:5]
        added = sorted(set(left) - set(right))[:5]
        return f"paths differ (missing {missing}, unexpected {added})"
    for key, value in right.items():
        try:
            if not np.array_equal(
                np.asarray(left[key]), np.asarray(value), equal_nan=True
            ):
                return f"value differs at {key}"
        except TypeError:
            # equal_nan is rejected for non-numeric dtypes; compare directly.
            if not np.array_equal(np.asarray(left[key]), np.asarray(value)):
                return f"value differs at {key}"
    return None


def _rewrite_manifest(
    directory: Path,
    row: Recontainment,
    *,
    source: Path,
    target: Path,
    previous_sha256: str,
    dropped_ids: tuple[str, ...],
) -> None:
    """Point the manifest's `output` at the file that now exists, and say why.

    `output` describes the file and is updated. **`input` is never touched**: it
    records the hashes of the files the run actually read, and rewriting it to
    match files that did not exist then would assert the physics was computed
    from something it was not. The join those `input` hashes belong to is closed
    from the other side instead -- a reader holding a dangling
    `input.diagnostics_sha256` matches it against the diagnostics manifest's
    `migration.previous_output.sha256`, which is the whole reason to rewrite the
    manifest here rather than record the migration somewhere else.

    The presence of the `migration` block is also how a reader tells a projected
    product from a freshly built one: their contents are the same projection of
    the same physics, but a migrated one's `input` names a file that has been
    superseded.
    """
    from datetime import datetime, timezone

    manifest_path = directory.parent / "metadata" / OMAS_MANIFEST_NAME
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    if not isinstance(manifest, dict):
        return

    import vaft

    manifest["output"] = {"name": target.name, "sha256": _sha256_file(target)}
    manifest["migration"] = {
        "schema_version": 1,
        "tool": "vaft.cli filedb migrate-products",
        "vaft_version": getattr(vaft, "__version__", "unknown"),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "container": {"from": source.name, "to": target.name},
        "projection": {
            "kept_ids": list(row.keep_ids or ()),
            "dropped_ids": list(dropped_ids),
        },
        "previous_output": {"name": source.name, "sha256": previous_sha256},
        "verified": row.verify,
    }
    _atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    _fsync_directory(path.parent)


def migrate_product_containers(
    root: str | Path,
    *,
    stages: Sequence[str] | None = None,
    apply: bool = False,
) -> ProductMigrationReport:
    """Bring every stage product under ``root`` onto its declared shape.

    A dry run by default: the returned report is the plan. With ``apply``, each
    pending product is written to a temporary file in its own `output/`
    directory, fsynced, verified by re-reading, and only then renamed into
    place. The superseded original is left for :func:`sweep_superseded_products`
    -- which is what makes every step up to that one reversible.
    """
    report = audit_product_containers(root, stages=stages)
    if not apply:
        return report
    if not report.safe_to_apply:
        raise ProductMigrationError(
            f"Refusing to migrate: {len(report.conflicting)} product(s) have a "
            "superseded original newer than the product that superseded it "
            f"({', '.join(report.conflicting[:3])}...). Something is still "
            "writing the old name; stop it and re-run the dry run."
        )

    by_stage = {row.stage: row for row in MIGRATIONS}
    done: list[str] = []
    failures: list[str] = []

    for stray in report.orphan_temporaries:
        Path(stray).unlink(missing_ok=True)

    for item in report.pending:
        row = by_stage[item.stage]
        directory = Path(item.directory)
        source = Path(item.source)
        target = Path(item.target)
        temporary = directory / (
            _temporary_prefix(row.stage) + f"{os.getpid()}{row.to_suffix}"
        )
        try:
            if row.keep_ids is None:
                previous_sha256, decompressed = _write_container_only(
                    source, temporary
                )
                if decompressed != previous_sha256:
                    raise ProductMigrationError(
                        "the written product does not decompress to the "
                        "original bytes"
                    )
                dropped: tuple[str, ...] = ()
            else:
                previous_sha256 = _sha256_file(source)
                written, expected, dropped = _write_projection(
                    source, temporary, row.keep_ids
                )
                problem = _same_ods(written, expected)
                if problem is not None:
                    raise ProductMigrationError(problem)

            os.replace(temporary, target)
            _fsync_directory(directory)
            _rewrite_manifest(
                directory,
                row,
                source=source,
                target=target,
                previous_sha256=previous_sha256,
                dropped_ids=dropped,
            )
            done.append(str(target))
        except Exception as error:
            temporary.unlink(missing_ok=True)
            failures.append(f"{source}: {type(error).__name__}: {error}")

    return ProductMigrationReport(
        root=report.root,
        applied=True,
        pending=(),
        migrated=tuple(sorted(set(report.migrated) | set(done))),
        settled=report.settled,
        conflicting=report.conflicting,
        orphan_temporaries=(),
        unrecognized=report.unrecognized,
        deletion_blocked=report.deletion_blocked,
        failures=tuple(failures),
    )


# --------------------------------------------------------------------------- #
# The gated deletion
# --------------------------------------------------------------------------- #
def _recoverability_problem(root: Path, item_directory: Path, row: Recontainment) -> str | None:
    """Why this stage's superseded original must not be deleted yet.

    Only projections can lose anything. A container change is reversible in
    substance -- `gunzip -c` reproduces the deleted file byte for byte, and the
    migration proved it does -- so there is nothing to gate.

    For the eddy projection the dropped IDS are recoverable from the same shot's
    diagnostics product and the era's static product, and that recoverability is
    a *precondition* of the deletion, not a given. The one genuinely
    unrecoverable act is deleting the original of an eddy product whose shot has
    no diagnostics product: that file is then the only local copy of that shot's
    magnetics.
    """
    if row.keep_ids is None:
        return None
    shot_directory = item_directory.parent
    if not shot_directory.name.isdigit():
        return f"cannot identify the shot from {shot_directory}"
    shot = shot_directory.name
    companion = root / "omas" / "diagnostics" / shot / "output"
    for suffix in (OMAS_PRODUCT_SUFFIX, LEGACY_SUFFIX):
        if (companion / f"diagnostics{suffix}").is_file():
            return None
    return (
        f"shot {shot} has no diagnostics product, so this file is the only "
        "local copy of its magnetics"
    )


def sweep_superseded_products(
    report: ProductMigrationReport,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Delete the originals the migration superseded. Irreversible.

    Takes an **audit** report, not a migration one: a deletion tool must not
    write. The precondition it checks is the one that matters and is visible
    without writing -- nothing still pending, nothing conflicting, nothing that
    failed verification.

    Gated the way `vaft.database.retirement` gates its deletion: a report this
    has not been shown clean is refused, and an empty report is refused rather
    than treated as "nothing to do" -- an empty plan and an unexamined tree look
    identical, and only one of them is safe.

    Run it per stage rather than over the whole tree at once. On the deployment
    this was written for, writing every new product before deleting any original
    peaks at ~1107 G against ~1126 G free; migrating and sweeping one stage at a
    time never exceeds ~1058 G.
    """
    if report.pending:
        raise ProductMigrationError(
            f"Refusing to sweep: {len(report.pending)} product(s) have not been "
            "migrated yet, so their originals are still the only copy "
            f"({', '.join(item.source for item in report.pending[:3])}...). "
            "Run `migrate-products --apply` first."
        )
    if not report.safe_to_apply:
        raise ProductMigrationError(
            "Refusing to sweep: the migration reported "
            f"{len(report.conflicting)} conflict(s) and {len(report.failures)} "
            "failure(s). Every superseded original is still the only copy of "
            "what its product failed to become."
        )
    if not report.migrated:
        raise ProductMigrationError(
            "Refusing to sweep an empty report. Nothing was migrated, which is "
            "either a tree that needs no sweep or one that was never examined "
            "-- and those look identical from here."
        )

    root = Path(report.root)
    by_stage = {row.stage: row for row in MIGRATIONS}
    removed: list[str] = []
    blocked: list[str] = []
    missing: list[str] = []

    for target_text in report.migrated:
        target = Path(target_text)
        directory = target.parent
        stage = target.name.split(".")[0]
        row = by_stage.get(stage)
        if row is None:
            continue
        source = directory / f"{stage}{row.from_suffix}"
        if not source.is_file():
            missing.append(str(source))
            continue
        problem = _recoverability_problem(root, directory, row)
        if problem is not None:
            blocked.append(f"{source}: {problem}")
            continue
        if apply:
            source.unlink()
            _fsync_directory(directory)
        removed.append(str(source))

    return {
        "schema_version": 1,
        "dry_run": not apply,
        "root": report.root,
        "summary": {
            "removed": len(removed),
            "deletion_blocked": len(blocked),
            "already_absent": len(missing),
        },
        "removed": removed,
        "deletion_blocked": blocked,
        "already_absent": missing,
    }
