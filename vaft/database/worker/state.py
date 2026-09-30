"""Durable state of the new-shot pipeline worker (issue #58).

One SQLite file holds everything a restart needs: the watermark, every shot the
worker has seen and where it is, each pipeline run, the per-stage status each
run left behind, and an event log.  It is also the read side for monitoring
(#1347): :func:`read_worker_state` opens it read-only.

Shot lifecycle::

    detected ──settle + classify──> queued ──run──> running ──harvest──> completed
        │                                              │                  partial
        └── classifier: nothing to run ──> excluded    │                  excluded
                                                       └──> failed ──(attempts exhausted)──> gave_up
                                                              └── retried as queued is

``failed`` is the only retryable terminal-looking state; ``gave_up``,
``excluded``, ``completed`` and ``partial`` are left alone until an operator
says otherwise.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = 1

DETECTED = "detected"
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
PARTIAL = "partial"
EXCLUDED = "excluded"
FAILED = "failed"
GAVE_UP = "gave_up"

SHOT_STATES = (DETECTED, QUEUED, RUNNING, COMPLETED, PARTIAL, EXCLUDED, FAILED, GAVE_UP)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shots (
    shot INTEGER PRIMARY KEY,
    record_datetime TEXT,
    detected_at TEXT NOT NULL,
    field_count INTEGER,
    field_count_at TEXT,
    settled_at TEXT,
    state TEXT NOT NULL,
    classification TEXT,
    applicable_stages TEXT,
    reason TEXT,
    reference_class TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_run_id INTEGER,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS shots_state ON shots(state);
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
    product TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    path TEXT NOT NULL,
    run_id INTEGER,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (shot, stage, product, kind)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    shot INTEGER,
    run_id INTEGER,
    kind TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS events_shot ON events(shot);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _iso(value: datetime | str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.isoformat(timespec="seconds")


class WorkerState:
    """The worker's SQLite store.  Every mutation is one transaction."""

    def __init__(self, path: str | os.PathLike[str], *, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            if not self.path.exists():
                raise FileNotFoundError(f"worker state not found: {self.path}")
            self._conn = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            with self._transaction():
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
        self._conn.row_factory = sqlite3.Row
        self.readonly = readonly

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "WorkerState":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # -- watermark -------------------------------------------------------
    def watermark(self, default: int) -> int:
        """The largest shot ever detected, or ``default`` before the first.

        SQL is asked only for shots above it, which assumes VEST numbers shots
        in acquisition order: a row inserted later *below* the watermark is
        not seen.  Such a shot can be added with ``add_detected`` by hand.
        """
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'watermark'").fetchone()
        return int(row[0]) if row is not None else int(default)

    # -- detection -------------------------------------------------------
    def add_detected(
        self, observations: Iterable[tuple[int, datetime | str | None]]
    ) -> list[int]:
        """Insert newly seen shots; return those that were not already known.

        A shot observed twice -- the same SQL row on two polls, or a poll
        overlapping the watermark -- is ignored the second time, which is what
        makes detection idempotent.
        """
        added: list[int] = []
        now = _now()
        with self._transaction() as conn:
            for shot, recorded in observations:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO shots(shot, record_datetime, detected_at, state, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (int(shot), _iso(recorded), now, DETECTED, now),
                )
                if cursor.rowcount:
                    added.append(int(shot))
                    self._event(conn, "detected", shot=int(shot), detail=_iso(recorded))
            if added:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('watermark', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(excluded.value AS INTEGER))",
                    (max(added),),
                )
        return added

    def observe_field_count(self, shot: int, count: int) -> int | None:
        """Record this poll's field count; return the previous poll's."""
        now = _now()
        with self._transaction() as conn:
            row = conn.execute("SELECT field_count FROM shots WHERE shot = ?", (shot,)).fetchone()
            conn.execute(
                "UPDATE shots SET field_count = ?, field_count_at = ?, updated_at = ? WHERE shot = ?",
                (int(count), now, now, int(shot)),
            )
        return None if row is None or row[0] is None else int(row[0])

    def settle(
        self,
        shot: int,
        *,
        state: str,
        classification: str,
        reason: str,
        applicable_stages: Sequence[str] | None,
    ) -> None:
        """Move a detected shot to ``queued`` or ``excluded`` with its verdict."""
        now = _now()
        stages = None if applicable_stages is None else json.dumps(list(applicable_stages))
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE shots SET state = ?, classification = ?, reason = ?, applicable_stages = ?, "
                "settled_at = ?, updated_at = ? WHERE shot = ? AND state = ?",
                (state, classification, reason, stages, now, now, int(shot), DETECTED),
            )
            if cursor.rowcount:  # an operator may have excluded it meanwhile
                kind = "enqueued" if state == QUEUED else "skipped"
                self._event(conn, kind, shot=int(shot), detail=f"{classification}: {reason}")

    # -- scheduling ------------------------------------------------------
    def runnable(self, *, limit: int, max_attempts: int) -> list[dict[str, Any]]:
        """Queued shots, then failed shots with attempts left, oldest shot first."""
        rows = self._conn.execute(
            "SELECT * FROM shots WHERE state = ? OR (state = ? AND attempts < ?) "
            "ORDER BY shot ASC LIMIT ?",
            (QUEUED, FAILED, int(max_attempts), int(limit)),
        ).fetchall()
        return [_shot_dict(row) for row in rows]

    def start_run(self, shots: Sequence[int]) -> tuple[int, list[int]]:
        """Open a run and mark its shots ``running``.

        Returns the run id and the shots actually claimed: one an operator
        excluded since it was selected is left out rather than overwritten.
        """
        now = _now()
        with self._transaction() as conn:
            cursor = conn.execute("INSERT INTO runs(shots, started_at) VALUES ('[]', ?)", (now,))
            run_id = int(cursor.lastrowid)
            claimed = [
                int(shot)
                for shot in shots
                if conn.execute(
                    "UPDATE shots SET state = ?, last_run_id = ?, updated_at = ? "
                    "WHERE shot = ? AND state IN (?, ?)",
                    (RUNNING, run_id, now, int(shot), QUEUED, FAILED),
                ).rowcount
            ]
            conn.execute("UPDATE runs SET shots = ? WHERE run_id = ?", (json.dumps(claimed), run_id))
            self._event(conn, "run_started", run_id=run_id, detail=json.dumps(claimed))
        return run_id, claimed

    def record_run_process(self, run_id: int, *, command: Sequence[str], log_path: str, pid: int) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE runs SET command = ?, log_path = ?, pid = ? WHERE run_id = ?",
                (json.dumps(list(command)), str(log_path), int(pid), int(run_id)),
            )

    def finish_run(self, run_id: int, *, exit_code: int | None, outcome: str) -> None:
        now = _now()
        with self._transaction() as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, exit_code = ?, outcome = ? WHERE run_id = ?",
                (now, exit_code, outcome, int(run_id)),
            )
            self._event(conn, "run_finished", run_id=run_id, detail=f"{outcome} (exit {exit_code})")

    def unfinished_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs WHERE finished_at IS NULL").fetchall()
        return [dict(row) for row in rows]

    def conclude(
        self,
        shot: int,
        *,
        state: str,
        reason: str,
        run_id: int | None,
        count_attempt: bool,
    ) -> None:
        """Record a run's verdict for one shot."""
        now = _now()
        with self._transaction() as conn:
            conn.execute(
                "UPDATE shots SET state = ?, reason = ?, attempts = attempts + ?, updated_at = ? "
                "WHERE shot = ?",
                (state, reason, 1 if count_attempt else 0, now, int(shot)),
            )
            self._event(conn, state, shot=int(shot), run_id=run_id, detail=reason)

    def give_up_exhausted(self, max_attempts: int) -> list[int]:
        """Move failed shots that have used every attempt to ``gave_up``."""
        now = _now()
        with self._transaction() as conn:
            shots = [
                int(row[0])
                for row in conn.execute(
                    "SELECT shot FROM shots WHERE state = ? AND attempts >= ?",
                    (FAILED, int(max_attempts)),
                )
            ]
            for shot in shots:
                conn.execute(
                    "UPDATE shots SET state = ?, updated_at = ? WHERE shot = ?", (GAVE_UP, now, shot)
                )
                self._event(conn, GAVE_UP, shot=shot, detail=f"{max_attempts} attempts used")
        return shots

    def set_reference_class(self, shot: int, label: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE shots SET reference_class = ?, updated_at = ? WHERE shot = ?",
                (label, _now(), int(shot)),
            )

    def record_stage_status(
        self, shot: int, statuses: Iterable[Any], *, run_id: int | None
    ) -> None:
        """Replace the per-stage status rows the harvest found for ``shot``."""
        now = _now()
        with self._transaction() as conn:
            conn.execute("DELETE FROM stage_status WHERE shot = ?", (int(shot),))
            for item in statuses:
                conn.execute(
                    "INSERT INTO stage_status(shot, stage, product, kind, status, path, run_id, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(shot),
                        item.stage,
                        item.product or "",
                        item.kind,
                        item.status,
                        str(item.path),
                        run_id,
                        now,
                    ),
                )

    # -- recovery and operator actions ------------------------------------
    def recover(self) -> list[int]:
        """Return shots a crashed worker left ``running`` to ``queued``.

        Their attempt is not counted: whether the run failed is unknown, and
        the next run re-derives it from the products on disk.
        """
        now = _now()
        with self._transaction() as conn:
            shots = [
                int(row[0])
                for row in conn.execute("SELECT shot FROM shots WHERE state = ?", (RUNNING,))
            ]
            for shot in shots:
                conn.execute(
                    "UPDATE shots SET state = ?, updated_at = ? WHERE shot = ?", (QUEUED, now, shot)
                )
                self._event(conn, "recovered", shot=shot, detail="left running by a previous worker")
            conn.execute(
                "UPDATE runs SET finished_at = ?, outcome = 'abandoned' WHERE finished_at IS NULL",
                (now,),
            )
        return shots

    def requeue(self, shot: int, *, reason: str) -> bool:
        """Operator retry: queue ``shot`` again with a fresh attempt budget.

        A shot the classifier gave nothing to run, limited to the raw dump, or
        that never settled goes back to ``detected`` so it is observed and
        classified afresh -- its SQL record may have been completed since.  For
        a raw-only shot the existing raw dump must be deleted first: it is an
        immutable Snakemake output and would otherwise be reused as is.
        """
        now = _now()
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE shots SET "
                "state = CASE WHEN settled_at IS NULL OR applicable_stages IN ('[]', '[\"raw\"]') "
                "             THEN ? ELSE ? END, "
                "field_count = CASE WHEN settled_at IS NULL OR applicable_stages IN ('[]', '[\"raw\"]') "
                "              THEN NULL ELSE field_count END, "
                "attempts = 0, reason = ?, updated_at = ? "
                "WHERE shot = ? AND state != ?",
                (DETECTED, QUEUED, reason, now, int(shot), RUNNING),
            )
            if cursor.rowcount:
                self._event(conn, "requeued", shot=int(shot), detail=reason)
        return bool(cursor.rowcount)

    def exclude(self, shot: int, *, reason: str) -> bool:
        """Operator exclusion: never schedule ``shot`` again."""
        now = _now()
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE shots SET state = ?, classification = COALESCE(classification, 'operator'), "
                "reason = ?, updated_at = ? WHERE shot = ? AND state != ?",
                (EXCLUDED, reason, now, int(shot), RUNNING),
            )
            if cursor.rowcount:
                self._event(conn, "excluded", shot=int(shot), detail=reason)
        return bool(cursor.rowcount)

    # -- reads -------------------------------------------------------------
    def shot(self, shot: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM shots WHERE shot = ?", (int(shot),)).fetchone()
        return None if row is None else _shot_dict(row)

    def shots(self, *, state: str | None = None) -> list[dict[str, Any]]:
        if state is None:
            rows = self._conn.execute("SELECT * FROM shots ORDER BY shot").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM shots WHERE state = ? ORDER BY shot", (state,)
            ).fetchall()
        return [_shot_dict(row) for row in rows]

    def stage_status(self, shot: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM stage_status WHERE shot = ? ORDER BY stage, product, kind", (int(shot),)
        ).fetchall()
        return [dict(row) for row in rows]

    def runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        return [{**dict(row), "shots": json.loads(row["shots"])} for row in rows]

    def events(self, *, shot: int | None = None) -> list[dict[str, Any]]:
        if shot is None:
            rows = self._conn.execute("SELECT * FROM events ORDER BY event_id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE shot = ? ORDER BY event_id", (int(shot),)
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT state, COUNT(*) FROM shots GROUP BY state").fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def event(self, kind: str, *, shot: int | None = None, run_id: int | None = None, detail: str | None = None) -> None:
        with self._transaction() as conn:
            self._event(conn, kind, shot=shot, run_id=run_id, detail=detail)

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        kind: str,
        *,
        shot: int | None = None,
        run_id: int | None = None,
        detail: str | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO events(at, shot, run_id, kind, detail) VALUES (?, ?, ?, ?, ?)",
            (_now(), shot, run_id, kind, detail),
        )


def _shot_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    stages = data.get("applicable_stages")
    data["applicable_stages"] = None if stages is None else tuple(json.loads(stages))
    return data


def read_worker_state(path: str | os.PathLike[str]) -> WorkerState:
    """Open a worker's state read-only -- the monitoring entry point (#1347)."""
    return WorkerState(path, readonly=True)


__all__ = [
    "COMPLETED",
    "DETECTED",
    "EXCLUDED",
    "FAILED",
    "GAVE_UP",
    "PARTIAL",
    "QUEUED",
    "RUNNING",
    "SHOT_STATES",
    "WorkerState",
    "read_worker_state",
]
