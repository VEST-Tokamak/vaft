"""The corrective-pipeline worker: run pipeline 2 when a Thomson or CES upload arrives (issue #1842).

The routine worker (:mod:`vaft.database.worker.service`) is driven by the VEST
SQL ``shot`` table: a new shot number appears, its upload finishes, the routine
pipeline runs once.  Pipeline 2 is driven by something else.  A Thomson or CES
``.mat`` file is uploaded by hand for a shot that was processed days ago, in any
order, and sometimes replaced by a later version.  So this worker watches the
upload directory (``external.data_root`` of the pipeline configuration), keeps
one signature per shot, and runs pipeline 2 for the shots whose signature is new
or has changed and has been quiet for ``quiet_seconds``.

It shares the routine worker's :class:`~vaft.database.worker.runner.SnakemakeRunner`
(the directory lock, the process group, the timeout) and its reading of stage
manifests, and nothing else: its own state file, its own lock, its own run
directory.  The routine worker is not touched, so the two can be deployed and
stopped independently.

What a shot's outcome is
------------------------
Every pipeline 2 stage writes a manifest and exits 0; the worker reads the
manifests, never the exit code.  ``unavailable`` and ``no_output`` are normal
results, not failures -- most shots carry no Thomson upload at all, and a shot
is the kinetic *or* the electron-only lineage, never both.  Only ``failed`` and
a missing or unreadable manifest (or, with ``hsds.replicate``, a failed
replication) are retried.

Why a finished shot is *forced* to rerun
----------------------------------------
Pipeline 2's roots have no input files, so Snakemake sees an existing output as
up to date even after the upload it was built from has been replaced.  A shot
that changed, or that failed, is therefore run with ``--forcerun`` on the two
root rules, which reruns everything downstream of them.

One writer per shot
-------------------
Only this worker's stages write the shot's Thomson, CES, ``core_profiles`` and
kinetic-EFIT products, but a regeneration of the routine products, or an EFIT
rerun, reads and writes the same shots.  ``hold_file`` is the stop: while the
file exists the worker keeps scanning and recording uploads but starts no run.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

from vaft.database.corrective_lineage import scoped, stage_scope
from vaft.database.sources import STAGE_REPLICATION

from .config import WorkerConfigError, _command, _deep_merge, _path, _read_yaml, _stages
from .runner import RunPlan, RunResult, SnakemakeRunner
from .status import (
    MANIFEST,
    MISSING,
    OK_STATUSES,
    UNREADABLE,
    REPLICATION,
    PipelineHarvester,
    StageStatus,
    StageTarget,
    load_pipeline_paths_module,
    _expand,
)

logger = logging.getLogger(__name__)

#: Environment variable naming the corrective worker's configuration file.
CONFIG_ENVIRONMENT_VARIABLE = "VAFT_CORRECTIVE_WORKER_CONFIG"

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
UNAVAILABLE = "unavailable"
FAILED = "failed"
GAVE_UP = "gave_up"
EXCLUDED = "excluded"
SHOT_STATES = (PENDING, RUNNING, COMPLETED, UNAVAILABLE, FAILED, GAVE_UP, EXCLUDED)

#: Manifest statuses that mean "this stage has nothing to give this shot, and said so".
ABSENT_BY_DESIGN = frozenset({"unavailable", "no_output"})
#: Manifest statuses that are failures: retried, then given up on.
STAGE_FAILURES = frozenset({"failed", MISSING, UNREADABLE})
#: Replication record states that are not failures (``skipped``: nothing eligible to send).
REPLICATION_SETTLED = frozenset({"validated", "replicated", "skipped"})
#: The two rules with no input file; forcing them reruns the whole lineage downstream.
ROOT_RULES = ("generate_thomson_ods", "generate_ces_ods")

_REQUIRED = (
    "state_db",
    "log_dir",
    "workflow_dir",
    "pipeline_config",
    "first_shot",
    "poll_interval",
    "quiet_seconds",
    "cores",
    "snakemake_cmd",
)

_CES_UPLOAD = re.compile(r"^(?:CES|IDS)_(?P<shot>\d+)\.mat$", re.IGNORECASE)
#: Where the Thomson mapper looks, in its own order: the upload root and two subdirectories.
_THOMSON_DIRECTORIES = ("thomson_scattering", "legacy", "")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectiveWorkerConfig:
    """What one deployment of the corrective worker runs, and where it keeps its state.

    Carries the attributes :class:`~vaft.database.worker.runner.SnakemakeRunner`
    reads, so the routine worker's runner serves it unchanged.
    """

    state_db: Path
    log_dir: Path
    workflow_dir: Path
    pipeline_config: Path
    #: The routine pipeline's directory: its ``paths.py`` resolves every product path.
    paths_workflow_dir: Path
    first_shot: int
    poll_interval: float
    #: A shot's upload counts as finished once no file of it has changed for this long.
    quiet_seconds: float
    cores: int
    snakemake_cmd: tuple[str, ...]
    last_shot: int | None = None
    extra_args: tuple[str, ...] = ()
    max_shots_per_run: int = 20
    max_attempts: int = 3
    run_timeout: float | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    #: The stages to build and judge; ``None`` is the whole lineage (``corrective_lineage.stage_scope``).
    stages: tuple[str, ...] | None = None
    #: While this file exists no run starts; uploads are still recorded.
    hold_file: Path | None = None

    @property
    def snakefile(self) -> Path:
        return self.workflow_dir / "Snakefile"

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(os.environ if base is None else base)
        for key, value in self.env.items():
            merged[str(key)] = os.path.expandvars(str(value))
        return merged


def corrective_config_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> CorrectiveWorkerConfig:
    """Build a :class:`CorrectiveWorkerConfig` from parsed YAML; relative paths resolve against ``base_dir``."""
    if not isinstance(data, Mapping):
        raise WorkerConfigError("the corrective worker configuration must be a mapping")
    missing = [key for key in _REQUIRED if data.get(key) is None]
    if missing:
        raise WorkerConfigError("corrective worker configuration is missing required key(s): " + ", ".join(missing))
    known = set(CorrectiveWorkerConfig.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise WorkerConfigError("unknown corrective worker configuration key(s): " + ", ".join(unknown))
    snakemake_cmd = _command(data["snakemake_cmd"], "snakemake_cmd")
    if not snakemake_cmd:
        raise WorkerConfigError("snakemake_cmd must not be empty")
    workflow_dir = _path(data["workflow_dir"], relative_to=base_dir, key="workflow_dir")
    paths_dir = (
        _path(data["paths_workflow_dir"], relative_to=base_dir, key="paths_workflow_dir")
        if data.get("paths_workflow_dir") is not None
        else workflow_dir.parent / "automatic_pipeline_1_routine_data_processing"
    )
    timeout = data.get("run_timeout")
    last_shot = data.get("last_shot")
    config = CorrectiveWorkerConfig(
        state_db=_path(data["state_db"], relative_to=base_dir, key="state_db"),
        log_dir=_path(data["log_dir"], relative_to=base_dir, key="log_dir"),
        workflow_dir=workflow_dir,
        pipeline_config=_path(data["pipeline_config"], relative_to=base_dir, key="pipeline_config"),
        paths_workflow_dir=paths_dir,
        first_shot=int(data["first_shot"]),
        poll_interval=float(data["poll_interval"]),
        quiet_seconds=float(data["quiet_seconds"]),
        cores=int(data["cores"]),
        snakemake_cmd=snakemake_cmd,
        last_shot=None if last_shot is None else int(last_shot),
        extra_args=_command(data.get("extra_args") or [], "extra_args"),
        max_shots_per_run=int(data.get("max_shots_per_run", 20)),
        max_attempts=int(data.get("max_attempts", 3)),
        run_timeout=None if timeout is None else float(timeout),
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        stages=_stages(data.get("stages")),
        hold_file=(
            None if data.get("hold_file") is None
            else _path(data["hold_file"], relative_to=base_dir, key="hold_file")
        ),
    )
    for name in ("poll_interval", "cores", "max_shots_per_run", "max_attempts"):
        if getattr(config, name) <= 0:
            raise WorkerConfigError(f"{name} must be positive")
    if config.quiet_seconds < 0:
        raise WorkerConfigError("quiet_seconds must not be negative")
    if config.run_timeout is not None and config.run_timeout <= 0:
        raise WorkerConfigError("run_timeout must be positive or null")
    if config.first_shot <= 0 or (config.last_shot is not None and config.last_shot < config.first_shot):
        raise WorkerConfigError("first_shot must be positive and not above last_shot")
    if not config.snakefile.is_file():
        raise WorkerConfigError(f"workflow_dir has no Snakefile: {config.workflow_dir}")
    if not (config.paths_workflow_dir / "paths.py").is_file():
        raise WorkerConfigError(f"paths_workflow_dir has no paths.py: {config.paths_workflow_dir}")
    try:
        stage_scope(config.stages)
    except ValueError as error:
        raise WorkerConfigError(f"stages: {error}") from None
    return config


def load_corrective_config(path: str | os.PathLike[str] | None = None) -> CorrectiveWorkerConfig:
    """Read the configuration from ``path`` or ``$VAFT_CORRECTIVE_WORKER_CONFIG``."""
    if path is None:
        path = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not path:
        raise WorkerConfigError(
            f"no corrective worker configuration given; pass --config or set {CONFIG_ENVIRONMENT_VARIABLE}"
        )
    config_path = Path(path).expanduser().resolve()
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise WorkerConfigError(f"corrective worker configuration not found: {config_path}") from None
    return corrective_config_from_mapping(data or {}, base_dir=config_path.parent)


def load_corrective_pipeline_config(config: CorrectiveWorkerConfig) -> dict[str, Any]:
    """The configuration Snakemake will run with: the workflow's own, overlaid by the deployment's.

    The same merge Snakemake performs, so the runner and the harvester see what
    it sees.  Refused for the two settings the worker cannot run under: no upload
    directory to watch, and a layout that is not the canonical FileDB.
    """
    default = config.workflow_dir / "config.yaml"
    data = _read_yaml(default, "workflow configuration") if default.is_file() else {}
    if config.pipeline_config.resolve() != default.resolve():
        data = _deep_merge(data, _read_yaml(config.pipeline_config, "pipeline configuration"))
    data["conda"] = None
    if (data.get("layout") or "filedb") != "filedb":
        raise WorkerConfigError("pipeline 2 requires layout: filedb")
    if not (data.get("external") or {}).get("data_root"):
        raise WorkerConfigError(
            "the pipeline configuration needs external.data_root: it is the directory the worker watches for uploads"
        )
    if config.stages is not None:
        data["stages"] = list(config.stages)
    return data


# ---------------------------------------------------------------------------
# watching the uploads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadSignature:
    """The files uploaded for one shot, as ``(relative name, size, mtime_ns)``."""

    files: tuple[tuple[str, int, int], ...]

    @property
    def newest_mtime(self) -> float:
        return max(mtime for _, _, mtime in self.files) / 1e9

    @property
    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.files).encode("utf-8")).hexdigest()[:16]


class UploadScanner:
    """List the Thomson and CES uploads under ``data_root`` and group them by shot.

    Thomson files are recognised by :func:`vaft.machine_mapping.thomson_scattering.thomson_file_shot`
    -- the very patterns the mapper resolves with, in the directories it searches --
    and CES files are ``CES_{shot}.mat`` or ``IDS_{shot}.mat`` in the upload root.
    """

    def __init__(self, data_root: str | os.PathLike[str]):
        self.data_root = Path(data_root)

    def scan(self) -> dict[int, UploadSignature]:
        """Every shot with an upload.  Raises :class:`OSError` if the root cannot be read."""
        from vaft.machine_mapping.thomson_scattering import thomson_file_shot

        if not self.data_root.is_dir():
            raise OSError(f"the upload directory is not available: {self.data_root}")
        found: dict[int, list[tuple[str, int, int]]] = {}

        def add(shot: int | None, entry: os.DirEntry) -> None:
            if shot is None:
                return
            info = entry.stat()
            relative = os.path.relpath(entry.path, self.data_root)
            found.setdefault(shot, []).append((relative, info.st_size, info.st_mtime_ns))

        for subdirectory in _THOMSON_DIRECTORIES:
            directory = self.data_root / subdirectory if subdirectory else self.data_root
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name)
            except OSError:
                continue
            for entry in entries:
                if not entry.is_file():
                    continue
                add(thomson_file_shot(entry.name), entry)
                if not subdirectory:
                    match = _CES_UPLOAD.match(entry.name)
                    if match is not None:
                        add(int(match.group("shot")), entry)
        return {shot: UploadSignature(tuple(sorted(files))) for shot, files in found.items()}


# ---------------------------------------------------------------------------
# durable state
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS uploads (
    shot INTEGER PRIMARY KEY,
    digest TEXT NOT NULL,
    files TEXT NOT NULL,
    newest_mtime REAL NOT NULL,
    state TEXT NOT NULL,
    force INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    operator_excluded INTEGER NOT NULL DEFAULT 0,
    run_digest TEXT,
    last_run_id INTEGER,
    first_seen TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS uploads_state ON uploads(state);
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    shots TEXT NOT NULL,
    command TEXT,
    log_path TEXT,
    pid INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    exit_code INTEGER,
    outcome TEXT
);
CREATE TABLE IF NOT EXISTS stage_status (
    shot INTEGER NOT NULL,
    stage TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    path TEXT,
    run_id INTEGER,
    PRIMARY KEY (shot, stage, kind)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    shot INTEGER,
    kind TEXT NOT NULL,
    detail TEXT
);
"""


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


class CorrectiveState:
    """The corrective worker's SQLite file: one row per shot with an upload, plus runs and events.

    Lifecycle::

        (new upload) ──> pending ──run──> running ──harvest──> completed       (>= 1 stage produced a product)
             ▲  ▲                                     │         unavailable    (every stage said so)
             │  └── the upload changed ───────────────┤         failed ──(attempts left)──> pending, forced
             │                                        └──>      gave_up        (attempts exhausted)
             └── operator retry                       excluded: never scheduled (operator)
    """

    def __init__(self, path: str | os.PathLike[str], *, read_only: bool = False):
        self.path = Path(path)
        if read_only:
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30.0, isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CorrectiveState":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- recording ---------------------------------------------------------
    def event(self, kind: str, *, shot: int | None = None, detail: str | None = None, at: datetime | None = None) -> None:
        self._conn.execute(
            "INSERT INTO events(at, shot, kind, detail) VALUES (?, ?, ?, ?)",
            (_stamp(at or datetime.now()), shot, kind, detail),
        )

    def observe(self, shot: int, signature: UploadSignature, *, at: datetime) -> str | None:
        """Record what the scan saw; returns ``"new"``, ``"changed"`` or ``None`` (nothing new)."""
        row = self.shot(shot)
        files = json.dumps(signature.files)
        if row is None:
            self._conn.execute(
                "INSERT INTO uploads(shot, digest, files, newest_mtime, state, force, first_seen, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (shot, signature.digest, files, signature.newest_mtime, PENDING, _stamp(at), _stamp(at)),
            )
            self.event("detected", shot=shot, detail=f"{len(signature.files)} file(s)", at=at)
            return "new"
        if row["digest"] == signature.digest:
            return None
        # A change during a run is held in the row and acted on when the run ends.
        state = row["state"] if row["state"] == RUNNING or row["operator_excluded"] else PENDING
        self._conn.execute(
            "UPDATE uploads SET digest=?, files=?, newest_mtime=?, state=?, force=1, attempts=?, updated_at=? WHERE shot=?",
            (signature.digest, files, signature.newest_mtime, state,
             row["attempts"] if state == RUNNING else 0, _stamp(at), shot),
        )
        self.event("changed", shot=shot, detail=f"was {row['state']}", at=at)
        return "changed"

    def ready(self, *, now: float, quiet_seconds: float, limit: int) -> list[sqlite3.Row]:
        """Pending shots whose newest file is at least ``quiet_seconds`` old, lowest shot first."""
        return list(self._conn.execute(
            "SELECT * FROM uploads WHERE state=? AND operator_excluded=0 AND (? - newest_mtime) >= ? "
            "ORDER BY shot LIMIT ?",
            (PENDING, now, quiet_seconds, limit),
        ))

    def new_run(self, shots: Sequence[int], *, at: datetime) -> int:
        cursor = self._conn.execute(
            "INSERT INTO runs(shots, started_at) VALUES (?, ?)", (json.dumps(list(shots)), _stamp(at))
        )
        run_id = int(cursor.lastrowid)
        for shot in shots:
            self._conn.execute(
                "UPDATE uploads SET state=?, run_digest=digest, last_run_id=?, updated_at=? WHERE shot=?",
                (RUNNING, run_id, _stamp(at), shot),
            )
        return run_id

    def note_process(self, run_id: int, *, command: Sequence[str], log_path: str, pid: int) -> None:
        self._conn.execute(
            "UPDATE runs SET command=?, log_path=?, pid=? WHERE run_id=?",
            (json.dumps(list(command)), log_path, pid, run_id),
        )

    def finish_run(self, run_id: int, *, exit_code: int | None, outcome: str, at: datetime) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at=?, exit_code=?, outcome=? WHERE run_id=?",
            (_stamp(at), exit_code, outcome, run_id),
        )

    def record_stages(self, shot: int, stages: Sequence[StageStatus], *, run_id: int | None) -> None:
        for stage in stages:
            self._conn.execute(
                "INSERT OR REPLACE INTO stage_status(shot, stage, kind, status, path, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                (shot, stage.stage, stage.kind, stage.status, stage.path, run_id),
            )

    def settle(self, shot: int, *, state: str, reason: str, attempts: int, force: bool, at: datetime) -> None:
        """Give a shot its post-run state; a change made while it ran sends it back to pending."""
        row = self.shot(shot)
        changed_while_running = row is not None and row["run_digest"] is not None and row["run_digest"] != row["digest"]
        if changed_while_running:
            state, attempts, force, reason = PENDING, 0, True, "the upload changed while the run was in progress"
        self._conn.execute(
            "UPDATE uploads SET state=?, reason=?, attempts=?, force=?, run_digest=NULL, updated_at=? WHERE shot=?",
            (state, reason, attempts, 1 if force else 0, _stamp(at), shot),
        )
        self.event("settled", shot=shot, detail=f"{state}: {reason}", at=at)

    def release(self, shots: Sequence[int], *, reason: str, at: datetime) -> None:
        """Put shots back to pending without counting an attempt (busy workflow, stop request)."""
        for shot in shots:
            self._conn.execute(
                "UPDATE uploads SET state=?, run_digest=NULL, reason=?, updated_at=? WHERE shot=? AND state=?",
                (PENDING, reason, _stamp(at), shot, RUNNING),
            )

    def recover(self, *, at: datetime) -> list[int]:
        """Shots a killed worker left running go back to pending (forced: their outputs may be incomplete)."""
        shots = [row["shot"] for row in self._conn.execute("SELECT shot FROM uploads WHERE state=?", (RUNNING,))]
        for shot in shots:
            self._conn.execute(
                "UPDATE uploads SET state=?, force=1, run_digest=NULL, reason=?, updated_at=? WHERE shot=?",
                (PENDING, "the worker stopped during the run", _stamp(at), shot),
            )
            self.event("recovered", shot=shot, at=at)
        self._conn.execute(
            "UPDATE runs SET finished_at=?, outcome='interrupted' WHERE finished_at IS NULL", (_stamp(at),)
        )
        return shots

    def unfinished_runs(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM runs WHERE finished_at IS NULL").fetchone()[0])

    # -- operator ----------------------------------------------------------
    def requeue(self, shot: int, *, reason: str, at: datetime | None = None) -> bool:
        at = at or datetime.now()
        cursor = self._conn.execute(
            "UPDATE uploads SET state=?, force=1, attempts=0, operator_excluded=0, reason=?, updated_at=? "
            "WHERE shot=? AND state<>?",
            (PENDING, reason, _stamp(at), shot, RUNNING),
        )
        if cursor.rowcount:
            self.event("requeued", shot=shot, detail=reason, at=at)
        return bool(cursor.rowcount)

    def exclude(self, shot: int, *, reason: str, at: datetime | None = None) -> bool:
        at = at or datetime.now()
        cursor = self._conn.execute(
            "UPDATE uploads SET state=?, operator_excluded=1, reason=?, updated_at=? WHERE shot=? AND state<>?",
            (EXCLUDED, reason, _stamp(at), shot, RUNNING),
        )
        if cursor.rowcount:
            self.event("excluded", shot=shot, detail=reason, at=at)
        return bool(cursor.rowcount)

    # -- reading -----------------------------------------------------------
    def shot(self, shot: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM uploads WHERE shot=?", (shot,)).fetchone()

    def shots(self, *, state: str | None = None) -> list[sqlite3.Row]:
        if state is None:
            return list(self._conn.execute("SELECT * FROM uploads ORDER BY shot"))
        return list(self._conn.execute("SELECT * FROM uploads WHERE state=? ORDER BY shot", (state,)))

    def counts(self) -> dict[str, int]:
        return {row["state"]: row["n"] for row in
                self._conn.execute("SELECT state, COUNT(*) AS n FROM uploads GROUP BY state")}

    def stage_status(self, shot: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute(
            "SELECT stage, kind, status, path FROM stage_status WHERE shot=? ORDER BY rowid", (shot,))]

    def events(self, *, shot: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if shot is None:
            rows = self._conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self._conn.execute("SELECT * FROM events WHERE shot=? ORDER BY id DESC LIMIT ?", (shot, limit))
        return [dict(row) for row in rows]


@contextmanager
def read_corrective_state(path: str | os.PathLike[str]) -> Iterator[CorrectiveState]:
    """Open the state file read-only, for monitoring."""
    state = CorrectiveState(path, read_only=True)
    try:
        yield state
    finally:
        state.close()


# ---------------------------------------------------------------------------
# reading back what a run left
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotOutcome:
    state: str
    reason: str
    stages: tuple[StageStatus, ...]


class CorrectiveHarvester:
    """Judge each shot by the manifests of the stages in scope, never by an exit code."""

    def __init__(
        self,
        pipeline_config: Mapping[str, Any],
        *,
        paths_workflow_dir: Path,
        environment: Mapping[str, str] | None = None,
    ):
        config = _expand(dict(pipeline_config), dict(os.environ if environment is None else environment))
        module = load_pipeline_paths_module(paths_workflow_dir)
        self.paths = module.PipelinePaths.from_config(config)
        if self.paths.layout != module.FILEDB:
            raise WorkerConfigError("pipeline 2 requires layout: filedb")
        self.scope = stage_scope(config.get("stages"))
        self.replicate = _flag((config.get("hsds") or {}).get("replicate"), False)

    def targets(self, shot: int) -> list[StageTarget]:
        targets = []
        for stage in scoped(self.scope):
            targets.append(StageTarget(stage, None, MANIFEST, getattr(self.paths, f"{stage}_manifest")(shot)))
            if self.replicate and STAGE_REPLICATION[stage].replicable:
                targets.append(StageTarget(stage, None, REPLICATION, self.paths.replication_record(shot, stage)))
        return targets

    def harvest(self, shot: int) -> ShotOutcome:
        stages = tuple(PipelineHarvester.read(target) for target in self.targets(shot))
        failures = [
            s for s in stages
            if (s.kind == MANIFEST and s.status in STAGE_FAILURES)
            or (s.kind == REPLICATION and s.status not in REPLICATION_SETTLED)
        ]
        if failures:
            return ShotOutcome(FAILED, "; ".join(f"{s.label}={s.status}" for s in failures), stages)
        produced = [s for s in stages if s.kind == MANIFEST and s.status in OK_STATUSES]
        if produced:
            return ShotOutcome(COMPLETED, "produced: " + ", ".join(s.stage for s in produced), stages)
        return ShotOutcome(
            UNAVAILABLE,
            "no stage produced a product: "
            + ", ".join(f"{s.stage}={s.status}" for s in stages if s.kind == MANIFEST),
            stages,
        )


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectivePlan(RunPlan):
    """A :class:`RunPlan` that may force the root rules (a changed or failed shot is not up to date)."""

    force: bool = False


class CorrectiveRunner(SnakemakeRunner):
    """The routine worker's runner, plus ``--forcerun`` on the two root rules when the plan asks."""

    def command(self, plan: RunPlan, configfile: Path) -> list[str]:
        command = super().command(plan, configfile)
        if getattr(plan, "force", False):
            command += ["--forcerun", *ROOT_RULES]
        return command


@dataclass
class CorrectiveCycleReport:
    recovered: list[int] = field(default_factory=list)
    scanned: int = 0
    new: list[int] = field(default_factory=list)
    changed: list[int] = field(default_factory=list)
    run_id: int | None = None
    run_shots: list[int] = field(default_factory=list)
    busy: bool = False
    held: bool = False
    scan_error: str | None = None
    outcomes: dict[int, str] = field(default_factory=dict)
    gave_up: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class CorrectiveWorker:
    """Watch the uploads, run pipeline 2, harvest.  Collaborators are injectable for tests."""

    def __init__(
        self,
        config: CorrectiveWorkerConfig,
        *,
        state: CorrectiveState | None = None,
        scanner: Any = None,
        runner: Any = None,
        harvester: CorrectiveHarvester | None = None,
        pipeline_config: Mapping[str, Any] | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.config = config
        self.pipeline_config = (
            dict(pipeline_config) if pipeline_config is not None else load_corrective_pipeline_config(config)
        )
        self.state = state if state is not None else CorrectiveState(config.state_db)
        self.scanner = scanner if scanner is not None else UploadScanner(self.pipeline_config["external"]["data_root"])
        self.runner = runner if runner is not None else CorrectiveRunner(config, self.pipeline_config)
        self.harvester = harvester if harvester is not None else CorrectiveHarvester(
            self.pipeline_config, paths_workflow_dir=config.paths_workflow_dir, environment=config.environment()
        )
        self.clock = clock
        self._stop = threading.Event()

    # -- the cycle ---------------------------------------------------------
    def recover(self) -> list[int]:
        stale = self.state.unfinished_runs()
        shots = self.state.recover(at=self.clock())
        if stale and self.runner.unlock_if_stale():
            self.state.event("unlocked", detail="no live Snakemake held the workflow lock", at=self.clock())
        if shots:
            logger.warning("recovered shots left running by a previous worker: %s", shots)
        return shots

    def scan(self, report: CorrectiveCycleReport) -> None:
        try:
            signatures = self.scanner.scan()
        except OSError as error:
            report.scan_error = str(error)
            logger.warning("upload scan failed: %s", error)
            return
        low, high = self.config.first_shot, self.config.last_shot
        for shot, signature in sorted(signatures.items()):
            if shot < low or (high is not None and shot > high):
                continue
            report.scanned += 1
            seen = self.state.observe(shot, signature, at=self.clock())
            if seen == "new":
                report.new.append(shot)
            elif seen == "changed":
                report.changed.append(shot)

    def held(self) -> bool:
        hold = self.config.hold_file
        return hold is not None and hold.exists()

    def run_batch(self, report: CorrectiveCycleReport) -> None:
        rows = self.state.ready(
            now=self.clock().timestamp(), quiet_seconds=self.config.quiet_seconds,
            limit=self.config.max_shots_per_run,
        )
        if not rows:
            return
        shots = [int(row["shot"]) for row in rows]
        force = any(bool(row["force"]) for row in rows)
        run_id = self.state.new_run(shots, at=self.clock())
        report.run_id, report.run_shots = run_id, shots
        logger.info("run %s: shots %s%s", run_id, shots, " (forced)" if force else "")
        result: RunResult = self.runner.run(
            CorrectivePlan(run_id=run_id, full_shots=tuple(shots), force=force),
            on_start=lambda command, log, pid: self.state.note_process(run_id, command=command, log_path=log, pid=pid),
        )
        if result.busy:
            self.state.finish_run(run_id, exit_code=result.exit_code, outcome="busy", at=self.clock())
            self.state.release(shots, reason="the workflow directory is locked by another run", at=self.clock())
            report.busy = True
            return
        if result.interrupted:
            self.state.finish_run(run_id, exit_code=result.exit_code, outcome="interrupted", at=self.clock())
            self.state.release(shots, reason="the worker was asked to stop", at=self.clock())
            return
        outcome = "timed_out" if result.timed_out else ("ok" if result.exit_code == 0 else "errors")
        self.state.finish_run(run_id, exit_code=result.exit_code, outcome=outcome, at=self.clock())
        incomplete = None if outcome == "ok" else (
            "the run timed out" if result.timed_out else f"snakemake exited with {result.exit_code}"
        )
        for row in rows:
            self.settle(row, run_id, report, incomplete=incomplete, forced=force)

    def settle(self, row: Any, run_id: int, report: CorrectiveCycleReport, *,
               incomplete: str | None = None, forced: bool = False) -> None:
        """Judge one shot from its manifests -- unless the run did not finish.

        Stage scripts always exit 0 and record failure in their manifest, so a
        nonzero exit or a timeout means Snakemake stopped part-way: manifests from
        an *earlier* run can still be on disk and would read as complete.  Such a
        shot is retried (counted as an attempt, forced again if this run was).
        """
        shot = int(row["shot"])
        verdict = self.harvester.harvest(shot)
        self.state.record_stages(shot, verdict.stages, run_id=run_id)
        attempts = int(row["attempts"])
        if incomplete is not None and verdict.state != FAILED:
            attempts += 1
            if attempts >= self.config.max_attempts:
                state, force = GAVE_UP, True
                report.gave_up.append(shot)
            else:
                state, force = PENDING, forced
            self.state.settle(shot, state=state, reason=f"{incomplete}; the manifests may be stale",
                              attempts=attempts, force=force, at=self.clock())
            report.outcomes[shot] = state
            return
        if verdict.state == FAILED:
            attempts += 1
            if attempts >= self.config.max_attempts:
                state, force = GAVE_UP, True
                report.gave_up.append(shot)
            else:
                # A failed stage left its manifest behind, so Snakemake would call it done.
                state, force = PENDING, True
        else:
            state, force, attempts = verdict.state, False, 0
        self.state.settle(shot, state=state, reason=verdict.reason, attempts=attempts, force=force, at=self.clock())
        report.outcomes[shot] = state

    def run_cycle(self, *, recover: bool = False) -> CorrectiveCycleReport:
        report = CorrectiveCycleReport()
        if recover:
            report.recovered = self.recover()
        self.scan(report)
        if self.held():
            report.held = True
            return report
        if not self._stop.is_set():
            self.run_batch(report)
        return report

    def run_forever(self) -> None:
        logger.info("corrective worker started; watching %s", self.pipeline_config["external"]["data_root"])
        self.run_cycle(recover=True)
        while not self._stop.wait(self.config.poll_interval):
            try:
                self.run_cycle()
            except Exception:  # noqa: BLE001 - a bad cycle must not end the service
                logger.exception("corrective worker cycle failed; continuing")
        logger.info("corrective worker stopped")

    def stop(self) -> None:
        self._stop.set()
        self.runner.stop()


def _flag(value: Any, default: bool) -> bool:
    """A configuration switch read as the Pipeline 2 Snakefile reads it.

    ``--config hsds={"replicate": false}`` reaches both as the *string* "false";
    ``bool()`` would call that true and expect replication records the workflow
    never writes, so every shot would be judged failed.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off", "none", "null")
    return bool(value)


__all__ = [
    "CONFIG_ENVIRONMENT_VARIABLE",
    "CorrectiveCycleReport",
    "CorrectiveHarvester",
    "CorrectivePlan",
    "CorrectiveRunner",
    "CorrectiveState",
    "CorrectiveWorker",
    "CorrectiveWorkerConfig",
    "SHOT_STATES",
    "ShotOutcome",
    "UploadScanner",
    "UploadSignature",
    "corrective_config_from_mapping",
    "load_corrective_config",
    "load_corrective_pipeline_config",
    "read_corrective_state",
]
