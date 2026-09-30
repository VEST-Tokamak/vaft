"""Where the worker learns about new shots: the VEST SQL ``shot`` table (issue #58).

A shot row appears in SQL before its waveforms are complete, and the raw dump
is an immutable Snakemake output -- a dump taken mid-write would be kept for
good.  So a shot is *settled* only when both hold:

- SQL's ``recordDateTime`` is at least ``settle_seconds`` old, and
- its waveform field inventory is the same size on two consecutive polls.
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

    def field_codes(self, shot: int) -> frozenset[int]:
        """The waveform field codes currently recorded for ``shot``."""


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

    def field_codes(self, shot: int) -> frozenset[int]:
        return raw_db.shot_field_codes(int(shot))


def is_settled(
    *,
    record_datetime: datetime | str | None,
    field_count: int,
    previous_field_count: int | None,
    settle_seconds: float,
    now: datetime,
) -> bool:
    """Whether a shot's SQL record has stopped changing."""
    if previous_field_count is None or previous_field_count != field_count:
        return False
    if record_datetime is None:
        return False
    if isinstance(record_datetime, str):
        record_datetime = datetime.fromisoformat(record_datetime)
    return (now - record_datetime).total_seconds() >= settle_seconds


__all__ = ["ShotSource", "SqlShotSource", "is_settled", "require_sql_credentials"]
