"""Where the worker learns about new shots: the VEST SQL ``shot`` table (issue #58).

A shot row appears in SQL before its waveforms, which the DAQ then writes one
complete field per row.  A shot's upload is taken as finished -- and the
shot processed -- as soon as either holds:

- **its inventory matches the previous shot's**: every field the previous
  shot had is present, and none has arrived for :data:`MATCH_QUIET_SECONDS`.
  On VEST the 177 core fields arrive 2-3 minutes after the shot record, at
  most ~10 s apart, so this fires before the next shot is fired -- and the
  short quiet guard keeps a small previous inventory (a partial-DAQ shot)
  from calling a shot finished in the middle of its upload; or
- **its upload has been quiet for** ``quiet_seconds``: no field row arrived
  for that long, so whatever is missing (the slow Pressure / Plasma Current /
  ECH group, a campaign-specific diagnostic) is not coming soon.

Fields that arrive after a shot was processed are caught by the worker's
re-check and the shot is reprocessed; see :mod:`vaft.database.worker.service`.
"""

from __future__ import annotations

from datetime import datetime
import os
from typing import Protocol

from vaft.database import raw as raw_db

from .config import WorkerConfigError


class ShotSource(Protocol):
    def new_shots(self, after: int) -> list[tuple[int, datetime]]:
        """``(shot, recordDateTime)`` for every shot numbered above ``after``."""

    def upload_status(self, shot: int) -> tuple[frozenset[int], float | None]:
        """Field codes recorded for ``shot`` and seconds since its last upload."""

    def field_codes_by_shot(self, shots: list[int]) -> dict[int, frozenset[int]]:
        """Field codes currently recorded for each of ``shots``."""


def require_sql_credentials() -> None:
    """Refuse to start without provisioned SQL credentials.

    :class:`vaft.database.raw.SecureConfigManager` falls back to ``input()``
    when its file is missing, which in a service blocks forever on a closed
    stdin; a missing key file silently generates a new key that cannot
    decrypt the stored password.
    """
    if not raw_db.sql_loading_available():
        raise WorkerConfigError("mysql-connector-python is not installed")
    for path, what in ((raw_db.CONFIG_FILE, "credentials"), (raw_db.KEY_FILE, "encryption key")):
        if not os.path.exists(path):
            raise WorkerConfigError(
                f"VEST SQL {what} not found at {path}; run "
                "`python -c 'import vaft.database.raw as r; r.setup_raw_db()'` "
                "once, interactively, as the account the worker runs as"
            )


class SqlShotSource:
    """The live VEST SQL database."""

    def __init__(self) -> None:
        require_sql_credentials()

    def new_shots(self, after: int) -> list[tuple[int, datetime]]:
        return raw_db.list_shots(shot_min=int(after) + 1)

    def upload_status(self, shot: int) -> tuple[frozenset[int], float | None]:
        return raw_db.shot_upload_status(int(shot))

    def field_codes_by_shot(self, shots: list[int]) -> dict[int, frozenset[int]]:
        return raw_db.field_codes_by_shot(shots)


#: Quiet time the inventory-match rule still requires.  Fields inside one VEST
#: core upload arrive at most ~10 s apart (measured, shots 48909-48916).
MATCH_QUIET_SECONDS = 30.0


def upload_finished(
    *,
    field_codes: frozenset[int],
    reference: frozenset[int] | None,
    quiet_seconds: float | None,
    quiet_limit: float,
) -> str | None:
    """Why a shot's upload counts as finished, or ``None`` while it may continue."""
    if (
        field_codes
        and reference
        and field_codes >= reference
        and quiet_seconds is not None
        and quiet_seconds >= MATCH_QUIET_SECONDS
    ):
        return f"inventory matches the previous shot's ({len(reference)} fields)"
    if quiet_seconds is not None and quiet_seconds >= quiet_limit:
        return f"no field uploaded for {quiet_seconds:.0f} s ({len(field_codes)} fields)"
    return None


__all__ = ["MATCH_QUIET_SECONDS", "ShotSource", "SqlShotSource", "require_sql_credentials", "upload_finished"]
