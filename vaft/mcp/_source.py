"""Where the MCP adapter gets data from: the private source seam (#1423).

The MCP protocol names a dataset semantically -- today, a packaged reference
shot number.  How that shot is opened (currently the OMAS reference sample) is
decided here and nowhere else, so a later DD/DDView-backed loader (#1127,
#1132, #1135) replaces this module without changing any tool schema.

Only packaged samples are resolved: nothing here reaches HSDS, a database, the
network or an arbitrary file path.
"""

from __future__ import annotations

from typing import Any

__all__ = ["known_shots", "load_reference_shot"]


def known_shots() -> tuple[int, ...]:
    """The packaged reference shots this installation declares."""
    from vaft.data import available_samples

    return tuple(int(shot) for shot in available_samples())


def load_reference_shot(shot: int) -> Any:
    """Open packaged reference shot ``shot`` for :func:`vaft.plot.extract`.

    A fresh object every call: extraction may materialise paths on the loaded
    data, and a cached object would carry that into the next answer.
    """
    shot = int(shot)
    shots = known_shots()
    if shot not in shots:
        raise ValueError(f"no packaged reference shot {shot}; available: {list(shots)}")
    from vaft.omas.sample import sample_ods

    try:
        return sample_ods(shot)
    except FileNotFoundError as error:
        raise ValueError(
            f"reference shot {shot} is declared but its data is not installed here "
            f"(repository-only sample): {error}"
        ) from None
