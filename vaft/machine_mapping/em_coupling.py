"""`em_coupling` IDS mapping helpers (stub)."""

from __future__ import annotations

from typing import Any


def em_coupling(ods: Any, *args, **kwargs) -> None:
    del ods, args, kwargs
    raise NotImplementedError("em_coupling mapping is not implemented yet.")


def calculate_em_coupling_from_raw_database(ods: Any, options: dict | None = None) -> None:
    del options
    em_coupling(ods)


__all__ = ["calculate_em_coupling_from_raw_database", "em_coupling"]
