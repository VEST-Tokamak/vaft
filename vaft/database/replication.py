"""Replicate finalized canonical FileDB products into a named HSDS source.

The canonical FileDB is the authoritative store. HSDS is a synchronized remote
representation of it, so this module copies an already-finalized product; it
does not construct one, and it disseminates nothing.

Three states are kept apart on purpose:

1. the local stage product is completed
2. it has been replicated to HSDS
3. the replica has been read back and compared against what was sent

A product sitting on disk says nothing about (2), so replication records its own
state in a ``replication.json`` beside the stage manifest rather than letting the
presence of either imply the other.

Each stage replicates only the IDS subtree it owns
(:data:`vaft.database.sources.STAGE_REPLICATION`), so stages that share a shot
folder -- or an IDS name across two sources -- do not overwrite one another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from collections.abc import Iterable
from typing import Any

from . import sources as _sources
from .filedb import FileDB, OMASStage
from .sources import MissingSourceError


logger = logging.getLogger(__name__)

REPLICATION_SCHEMA_VERSION = 1

#: Manifest statuses whose product is finished enough to replicate. A stage that
#: was skipped or blocked (#205) has no product to send, and a failed one must
#: not have its partial output mistaken for a result.
REPLICABLE_STATUSES = frozenset({"success", "partial", "completed"})


class ReplicationError(RuntimeError):
    """Raised when a stage product cannot be replicated."""


class StageNotReplicableError(ReplicationError):
    """Raised when a stage has no destination or no wired rule."""


class ProductNotEligibleError(ReplicationError):
    """Raised when the local stage product is not finished enough to send."""


class RoundTripValidationError(ReplicationError):
    """Raised when the replica does not match what was sent."""


@dataclass(frozen=True)
class ReplicationRecord:
    """What happened when one stage product was sent to a named HSDS source."""

    stage: str
    shot: int
    source: str
    remote_uri: str
    ids: tuple[str, ...]
    occurrence: int
    product_sha256: str
    state: str
    attempts: int
    started_at: str
    completed_at: str | None = None
    round_trip: dict[str, Any] | None = None
    error: str | None = None
    schema_version: int = REPLICATION_SCHEMA_VERSION
    vaft_version: str = field(default_factory=lambda: _vaft_version())

    @property
    def replicated(self) -> bool:
        return self.state in {"replicated", "validated"}

    @property
    def validated(self) -> bool:
        return self.state == "validated"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ids"] = list(self.ids)
        return payload


def _vaft_version() -> str:
    from ..version import __version__

    return __version__


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path) -> ReplicationRecord | None:
    """Return a previously written record, or ``None`` when there is none."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != REPLICATION_SCHEMA_VERSION:
        return None
    known = {f for f in ReplicationRecord.__dataclass_fields__}
    payload = {key: value for key, value in payload.items() if key in known}
    payload["ids"] = tuple(payload.get("ids", ()))
    try:
        return ReplicationRecord(**payload)
    except TypeError:
        return None


def write_record(record: ReplicationRecord, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def is_reusable(
    record: ReplicationRecord | None,
    *,
    product_sha256: str,
    require_validation: bool,
) -> bool:
    """Whether a recorded replication still stands for the current product.

    Existence is not enough: the record has to describe *this* product, and to
    have reached the state the caller's contract asks for. A rebuilt product
    with the same path is a different product.
    """
    if record is None:
        return False
    if record.product_sha256 != product_sha256:
        return False
    return record.validated if require_validation else record.replicated


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProductNotEligibleError(
            f"No stage manifest at {path}; a product is only replicated once the "
            "stage that produced it has recorded how."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductNotEligibleError(f"Unreadable stage manifest {path}: {exc}") from exc


def _check_eligible(manifest: dict[str, Any], stage: str, product: Path) -> None:
    if not product.exists():
        raise ProductNotEligibleError(f"No {stage} product at {product}")
    status = str(manifest.get("status", "")).strip().lower()
    if not status:
        raise ProductNotEligibleError(
            f"The {stage} manifest records no status; refusing to replicate a "
            "product whose completion state is unknown."
        )
    if status not in REPLICABLE_STATUSES:
        raise ProductNotEligibleError(
            f"The {stage} stage is {status!r}, not one of "
            f"{', '.join(sorted(REPLICABLE_STATUSES))}; there is nothing to replicate."
        )


def _project(ods, ids_names: tuple[str, ...]) -> tuple[Any, tuple[str, ...]]:
    """Return a copy holding only the IDS this stage owns, plus provenance."""
    from omas import ODS

    projected = ODS(consistency_check=False)
    if "dataset_description" in ods:
        projected["dataset_description"] = ods["dataset_description"]
    present: list[str] = []
    for name in ids_names:
        if name in ods:
            projected[name] = ods[name]
            present.append(name)
    return projected, tuple(present)


def _remote_entries(source: str, shot: int) -> tuple[str, ...]:
    """List a shot folder, treating an absent folder as empty."""
    from .utils import h5pyd

    try:
        return tuple(sorted(h5pyd.Folder(f"/{source}/{shot}/", mode="r")))
    except Exception as exc:  # noqa: BLE001 - a shot's first write has no folder
        logger.debug("No shot folder at %s/%s: %s", source, shot, exc)
        return ()


def _remote_canonical_files(entries: Iterable[str]) -> tuple[str, ...]:
    from .h5image import is_derived_filename

    return tuple(
        sorted(
            name
            for name in entries
            if name.endswith(".h5")
            and name != "master.h5"
            and not is_derived_filename(name)
        )
    )


def _fetch_remote_master(source: str, shot: int, target: Path) -> Path | None:
    """Download the shot's master, or ``None`` when the shot has none yet.

    A failed download of a master that *does* exist is an error, not an absence.
    Reading it as an absence would skip the merge and quietly drop every other
    stage's IDS from the shot, so the two cases are distinguished by listing the
    folder first.
    """
    from .transport import run_hsget

    if "master.h5" not in _remote_entries(source, shot):
        return None
    run_hsget(f"hdf5://{source}/{shot}/master.h5", target)
    if not target.exists():
        raise ReplicationError(
            f"hdf5://{source}/{shot}/master.h5 is listed but could not be "
            "materialized; refusing to write a master that would hide the "
            "IDS already stored for this shot."
        )
    return target


def merge_remote_master(source: str, shot: int, previous_master: Path | None) -> tuple[str, ...]:
    """Fold a pre-write master's links back into the one the write just left.

    Writing a stage leaves a master describing only that stage's IDS. The eager
    reader resolves a shot's contents from those links, so without this every
    other stage's IDS would go missing from the shot the moment one stage was
    replicated.
    """
    if previous_master is None:
        return ()
    from .staging import merge_master_links
    from .transport import run_hsload, verify_uploaded_image

    with tempfile.TemporaryDirectory(prefix="vaft-master-merge-") as workdir:
        current = Path(workdir) / "master.h5"
        if _fetch_remote_master(source, shot, current) is None:
            # The write we just made must have left one.
            raise ReplicationError(
                f"hdf5://{source}/{shot}/master.h5 is missing after a write; "
                "cannot merge the IDS already stored for this shot."
            )
        added = merge_master_links(
            current,
            previous_master,
            present_files=_remote_canonical_files(_remote_entries(source, shot)),
        )
        if not added:
            return ()
        remote_uri = f"hdf5://{source}/{shot}/master.h5"
        run_hsload(current, remote_uri)
        verify_uploaded_image(current, remote_uri)
    return added


#: Leaves the IMAS Access Layer stamps into an IDS as it writes it. They record
#: which library performed the write -- ``access_layer`` is the imas_core
#: version, ``access_layer_language`` the binding -- so they exist in a replica
#: and cannot exist in the OMAS JSON product it was built from. Comparing them
#: measures the writer, not the data.
WRITE_PROVENANCE_LEAVES = (
    "access_layer",
    "access_layer_language",
)


def _is_write_provenance(path: str) -> bool:
    """Whether an ODS path is an Access Layer write stamp rather than data."""
    return any(
        path.endswith(f"ids_properties.version_put.{leaf}")
        for leaf in WRITE_PROVENANCE_LEAVES
    )


def _is_empty_left_unwritten(entry) -> bool:
    """Whether a difference is just an empty local array the AL did not store.

    A diagnostic channel that was unavailable for a shot -- a disabled PF coil,
    a field code the DAQ never recorded -- reaches the product as a zero-length
    array. IMAS stores no such node, so the replica simply lacks the path. Both
    sides say "no data here"; only the representation differs, and reporting it
    as a regression would leave almost every shot unvalidated, since some
    channel is unavailable on most of them.

    Deliberately narrow: it fires only when the local side is genuinely empty.
    A path missing from the replica while the local product has values is real
    data loss and still fails.
    """
    from ..omas.comparison import DifferenceKind

    if entry.kind is not DifferenceKind.MISSING_CANDIDATE:
        return False
    shape = entry.reference_shape
    if shape is None:
        return False
    try:
        return any(int(dim) == 0 for dim in shape)
    except (TypeError, ValueError):
        return False


def _round_trip(sent, *, shot: int, source: str, ids: tuple[str, ...], occurrence: int) -> dict[str, Any]:
    """Read the replica back and compare it against what was sent."""
    from ..omas.comparison import ParityClassification, compare_ods
    from . import load as load_remote

    replica = load_remote(
        shot,
        source=source,
        paths=list(ids),
        occurrence=occurrence or None,
    )
    comparison = compare_ods(
        sent,
        replica,
        scope="reference",
        paths=[f"{name}.*" for name in ids],
        reference_label="local product",
        candidate_label=f"hdf5://{source}/{shot}/",
    )
    summary = comparison.summary()

    # The comparator has no exclusion filter, so the write stamps are dropped
    # from the verdict here rather than from the inputs -- nothing is mutated,
    # nothing is copied, and the count stays visible in the record.
    regressions = [
        entry
        for entry in comparison.entries
        if entry.classification is ParityClassification.REGRESSION
        and not _is_write_provenance(entry.path)
        and not _is_empty_left_unwritten(entry)
    ]
    excluded = sum(
        1
        for entry in comparison.entries
        if _is_write_provenance(entry.path)
    )
    empty_unwritten = sum(
        1 for entry in comparison.entries if _is_empty_left_unwritten(entry)
    )
    summary = {
        **summary,
        "passed": not regressions,
        "write_provenance_excluded": excluded,
        "empty_arrays_unwritten": empty_unwritten,
    }
    if regressions:
        paths = ", ".join(entry.path for entry in regressions[:5])
        raise RoundTripValidationError(
            f"The {source} replica of shot {shot} does not match what was sent "
            f"({len(regressions)} path(s), e.g. {paths}): {summary}"
        )
    return summary


def replicate_stage(
    stage: str | OMASStage,
    shot: int,
    *,
    filedb: FileDB,
    attempts: int = 3,
    retry_delay: float = 5.0,
    validate: bool = True,
    force: bool = False,
) -> ReplicationRecord:
    """Replicate one completed canonical FileDB stage product into HSDS.

    Returns the record that was written. Re-running is cheap: a record that
    already describes this exact product in the required state is reused rather
    than re-sent, so a rerun never repeats the solver stage that produced it.
    """
    entry = _sources.replication_for_stage(stage)
    name = OMASStage(stage).value
    if entry.source is None:
        raise StageNotReplicableError(
            f"The {name} stage is not replicated to HSDS"
            + (f" ({entry.note})" if entry.note else "")
        )
    if entry.deferred_to is not None:
        raise StageNotReplicableError(
            f"Replication of the {name} stage is not wired yet; it belongs to "
            f"issue {entry.deferred_to}."
        )
    # writable=True is what makes it impossible to replicate into `public`.
    source = _sources.resolve(entry.source, writable=True)

    shot = int(shot)
    product = filedb.omas_product(name, shot=shot)
    manifest_path = filedb.omas_manifest(name, shot=shot)
    record_path = filedb.omas_replication_record(name, shot=shot)

    _check_eligible(_read_manifest(manifest_path), name, product)
    product_sha256 = sha256_file(product)

    previous = read_record(record_path)
    if not force and is_reusable(
        previous, product_sha256=product_sha256, require_validation=validate
    ):
        logger.info(
            "shot %s %s already replicated to %s; product unchanged", shot, name, source
        )
        return previous

    from ..omas import load as load_local
    from . import save as save_remote

    ods = load_local(product)
    projected, present = _project(ods, entry.ids)
    if not present:
        raise ProductNotEligibleError(
            f"The {name} product at {product} carries none of the IDS this stage "
            f"owns ({', '.join(entry.ids)}); there is nothing to replicate."
        )

    occurrence = {ids_name: entry.occurrence for ids_name in present} if entry.occurrence else {}
    started_at = _now()
    remote_uri = f"hdf5://{source}/{shot}/"
    last_error: Exception | None = None
    made = 0

    with tempfile.TemporaryDirectory(prefix="vaft-replicate-") as workdir:
        from .utils import require_source_exists

        previous_master: Path | None = None
        captured = False

        for attempt in range(1, max(1, attempts) + 1):
            made = attempt
            try:
                # Inside the loop: a genuinely missing namespace still fails
                # fast and unretried (MissingSourceError is not transient), but
                # a server too busy to answer the probe is retried like any
                # other transient remote failure.
                require_source_exists(source)
                if not captured:
                    # Captured once, before any write: a write leaves behind a
                    # master describing only this stage, so a retry that
                    # re-read it would merge against its own first attempt and
                    # lose the other stages for good. Behind the probe so an
                    # unprovisioned namespace costs no remote fetch; a fetch
                    # that fails is retried, having written nothing.
                    previous_master = _fetch_remote_master(
                        source, shot, Path(workdir) / "master.previous.h5"
                    )
                    captured = True
                save_remote(
                    projected,
                    shot,
                    source=source,
                    occurrence=occurrence or None,
                )
                merge_remote_master(source, shot, previous_master)
                last_error = None
                break
            except MissingSourceError:
                # Provisioning is an operator action; retrying cannot fix it.
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then recorded
                last_error = exc
                logger.warning(
                    "shot %s %s replication attempt %s/%s failed: %s",
                    shot, name, attempt, attempts, exc,
                )
                if attempt < attempts:
                    time.sleep(retry_delay)

    if last_error is not None:
        record = ReplicationRecord(
            stage=name, shot=shot, source=source, remote_uri=remote_uri,
            ids=present, occurrence=entry.occurrence,
            product_sha256=product_sha256, state="failed", attempts=made,
            started_at=started_at, error=str(last_error),
        )
        write_record(record, record_path)
        raise ReplicationError(
            f"Could not replicate shot {shot} {name} to {source} after "
            f"{attempts} attempt(s): {last_error}"
        ) from last_error

    replicated = ReplicationRecord(
        stage=name, shot=shot, source=source, remote_uri=remote_uri,
        ids=present, occurrence=entry.occurrence,
        product_sha256=product_sha256, state="replicated",
        attempts=made, started_at=started_at, completed_at=_now(),
    )
    if not validate:
        write_record(replicated, record_path)
        logger.info(
            "shot %s %s replicated to %s (%s)", shot, name, source, ", ".join(present)
        )
        return replicated

    try:
        round_trip = _round_trip(
            projected, shot=shot, source=source, ids=present, occurrence=entry.occurrence
        )
    except Exception as exc:  # noqa: BLE001 - the write stands, the check did not
        # The bytes are on the server. Saying only "failed" here would send a
        # rerun back through the upload instead of straight to the comparison.
        write_record(replace(replicated, error=str(exc)), record_path)
        raise

    record = replace(replicated, state="validated", round_trip=round_trip)
    write_record(record, record_path)
    logger.info("shot %s %s replicated to %s (%s)", shot, name, source, ", ".join(present))
    return record


__all__ = [
    "REPLICABLE_STATUSES",
    "REPLICATION_SCHEMA_VERSION",
    "ProductNotEligibleError",
    "ReplicationError",
    "ReplicationRecord",
    "RoundTripValidationError",
    "StageNotReplicableError",
    "is_reusable",
    "merge_remote_master",
    "read_record",
    "replicate_stage",
    "write_record",
]
